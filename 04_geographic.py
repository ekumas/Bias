#!/usr/bin/env python3
"""
Script 04 — G dimension: geographic concentration.

    G(C) = 1 - H(country distribution) / log2(N_REF_COUNTRIES)

H is Shannon entropy in bits over country-level counts. N_REF_COUNTRIES is
FIXED A PRIORI in config.py (193 UN member states) rather than derived from the
observed data. This matters: normalising by the number of countries a corpus
happens to contain makes G non-comparable across corpora and unreproducible.
State the constant in the paper.

Because H <= log2(observed countries) <= log2(N_REF) is not guaranteed when a
corpus mentions more than N_REF distinct entities, G is clipped to [0, 1] and
any clipping is reported.

Per-document country lists are written out so script 06 can bootstrap G.

Usage:
    python scripts/04_geographic.py
    python -m spacy download en_core_web_sm     # one-time prerequisite
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def build_geo_index() -> dict[str, str]:
    """Delegates to geo_index.build_index(). Kept as a thin wrapper so the
    script's call sites are unchanged."""
    try:
        from geo_index import build_index
    except ImportError:
        print("geo_index.py not found next to config.py.")
        return {}
    try:
        return build_index()
    except ImportError:
        print("geonamescache not installed.  pip install geonamescache")
        return {}


def load_nlp():
    try:
        import spacy
    except ImportError:
        print("spacy not installed.  pip install spacy")
        return None
    try:
        nlp = spacy.load(C.SPACY_MODEL, disable=["lemmatizer", "textcat"])
    except OSError:
        print(f"spaCy model '{C.SPACY_MODEL}' missing. "
              f"Run: python -m spacy download {C.SPACY_MODEL}")
        return None
    return nlp


def shannon_entropy_bits(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for v in counts.values():
        p = v / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def g_from_counts(counts: Counter) -> tuple[float, float, bool]:
    h = shannon_entropy_bits(counts)
    denom = math.log2(C.N_REF_COUNTRIES)
    raw = 1.0 - h / denom
    clipped = raw < 0.0 or raw > 1.0
    return min(max(raw, 0.0), 1.0), h, clipped


def process(key: str, nlp, geo: dict[str, str]) -> bool:
    src = C.DATA_DIR / f"sample_{key}.parquet"
    if not src.exists():
        print(f"  [{key}] SKIP — {src.name} not found. Run 01_sample.py first.")
        return False

    df = pd.read_parquet(src, columns=["doc_id", "text"])
    texts = [t[: C.GEO_MAX_CHARS] for t in df["text"]]

    from geo_index import classify

    per_doc: list[str] = []
    corpus_counts: Counter = Counter()
    buckets: Counter = Counter()

    docs = nlp.pipe(texts, batch_size=C.SPACY_BATCH_SIZE, n_process=C.SPACY_N_PROCESS)
    for i, doc in enumerate(docs):
        codes: list[str] = []
        for ent in doc.ents:
            if ent.label_ not in C.GEO_ENTITY_TYPES:
                continue
            bucket, cc = classify(ent.text, geo)
            buckets[bucket] += 1
            if bucket == "country":
                codes.append(cc)
        corpus_counts.update(codes)
        per_doc.append(";".join(codes))
        if (i + 1) % 10_000 == 0:
            print(f"  [{key}] {i + 1:,}/{len(texts):,} docs", flush=True)

    n_country = buckets["country"]
    n_unmapped = buckets["unmapped"]
    n_ents = sum(buckets.values())
    # Denominator excludes regions and NER false positives: neither could ever
    # resolve to a country, so counting them as failures understates the index.
    resolvable = n_country + n_unmapped

    out_df = pd.DataFrame({
        "doc_id": df.doc_id, "corpus": key, "countries": per_doc,
    })
    out_df.to_parquet(C.RESULTS_DIR / f"perdoc_geographic_{key}.parquet", index=False)

    pd.DataFrame(
        sorted(corpus_counts.items(), key=lambda kv: -kv[1]),
        columns=["iso2", "count"],
    ).to_csv(C.RESULTS_DIR / f"country_counts_{key}.csv", index=False)

    g, h, clipped = g_from_counts(corpus_counts)
    rate = n_country / resolvable if resolvable else 0.0
    print(f"  [{key}] G={g:.4f}  H={h:.3f} bits  "
          f"{len(corpus_counts)} countries"
          + ("  [CLIPPED]" if clipped else ""))
    print(f"           entities={n_ents:,}  country={n_country:,}  "
          f"region={buckets['region']:,}  nongeo={buckets['nongeo']:,}  "
          f"unmapped={n_unmapped:,}")
    print(f"           map_rate={rate:.1%} "
          f"(country / (country + unmapped))")
    if rate < 0.60:
        print(f"  [{key}] WARNING: {1 - rate:.0%} of resolvable place names "
              f"are still unknown to the index. Run 04b_geo_audit.py and "
              f"check whether the residual is geographically skewed before "
              f"quoting G.")
    if n_country < 5_000:
        print(f"  [{key}] WARNING: only {n_country:,} country mentions in "
              f"{len(texts):,} documents. G is estimated from very little "
              f"evidence here and should be reported as undefined rather "
              f"than as a low-diversity finding.")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    a = ap.parse_args()

    nlp = load_nlp()
    geo = build_geo_index()
    if nlp is None or not geo:
        return 1

    keys = [a.corpus] if a.corpus else list(C.CORPORA)
    print("\n=== G dimension ===")
    ok = [process(k, nlp, geo) for k in keys]
    return 0 if any(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
