#!/usr/bin/env python3
"""
04b_geo_audit.py — what is the geo index failing to map, and is that failure
systematically biased?

Script 04 maps roughly half of all GPE/LOC entities to a country and silently
discards the rest. That is only acceptable if the discarded half is random with
respect to geography. If non-Western place names map at a lower rate than
Western ones, then G understates Western concentration -- a bias pointing in
the opposite direction to the paper's claim, which would be fatal if left
undeclared.

This script re-runs NER on a subsample and writes, per corpus:
  unmapped_entities_<corpus>.csv   the most frequent unmapped surface forms
  mapped_entities_<corpus>.csv     the most frequent mapped ones, for contrast

It cannot decide the bias question by itself. It produces the evidence; you
read the unmapped list and judge. Look for: non-Latin scripts, transliterated
place names, cities outside Europe and North America, and sub-national units.
If the unmapped list is dominated by US states and English county names, the
failure is benign for this purpose. If it is dominated by African, South Asian
and Middle Eastern toponyms, G must be reported as biased and the index
extended before the number goes in the paper.

    python scripts/04b_geo_audit.py                # 10,000 docs per corpus
    python scripts/04b_geo_audit.py --limit 50000  # all of them, slower
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
geo_mod = __import__("04_geographic")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--corpus", default=None)
    a = ap.parse_args()

    nlp = geo_mod.load_nlp()
    geo = geo_mod.build_geo_index()
    if nlp is None or not geo:
        return 1

    keys = [a.corpus] if a.corpus else list(C.CORPORA)
    summary = []

    for key in keys:
        src = C.DATA_DIR / f"sample_{key}.parquet"
        if not src.exists():
            print(f"  [{key}] SKIP — {src.name} not found.")
            continue

        df = pd.read_parquet(src, columns=["text"]).head(a.limit)
        texts = [t[: C.GEO_MAX_CHARS] for t in df["text"]]
        print(f"  [{key}] auditing {len(texts):,} docs ...", flush=True)

        from geo_index import classify

        unmapped: Counter = Counter()
        mapped: Counter = Counter()
        buckets: Counter = Counter()
        for i, doc in enumerate(nlp.pipe(texts, batch_size=C.SPACY_BATCH_SIZE,
                                         n_process=C.SPACY_N_PROCESS)):
            for ent in doc.ents:
                if ent.label_ not in C.GEO_ENTITY_TYPES:
                    continue
                surface = ent.text.strip()
                bucket, _cc = classify(surface, geo)
                buckets[bucket] += 1
                if bucket == "country":
                    mapped[surface] += 1
                elif bucket == "unmapped":
                    unmapped[surface] += 1
            if (i + 1) % 5_000 == 0:
                print(f"  [{key}] {i + 1:,} docs", flush=True)

        n_m, n_u = buckets["country"], buckets["unmapped"]
        tot = n_m + n_u
        pd.DataFrame(unmapped.most_common(a.top), columns=["surface", "count"]) \
          .to_csv(C.RESULTS_DIR / f"unmapped_entities_{key}.csv", index=False)
        pd.DataFrame(mapped.most_common(a.top), columns=["surface", "count"]) \
          .to_csv(C.RESULTS_DIR / f"mapped_entities_{key}.csv", index=False)

        # Concentration of the unmapped tail: if a handful of surface forms
        # account for most failures, extending the index is cheap and worth it.
        top20 = sum(c for _, c in unmapped.most_common(20))
        share = top20 / n_u if n_u else 0.0
        summary.append({
            "corpus": key, "docs_audited": len(texts),
            "entities": sum(buckets.values()), "mapped": n_m,
            "unmapped": n_u, "region": buckets["region"],
            "nongeo": buckets["nongeo"],
            "map_rate": round(n_m / tot, 4) if tot else 0.0,
            "distinct_unmapped": len(unmapped),
            "top20_share_of_unmapped": round(share, 4),
        })
        print(f"  [{key}] map_rate={n_m / tot:.1%}  "
              f"{len(unmapped):,} distinct unmapped forms; "
              f"top 20 cover {share:.1%} of failures")
        print(f"  [{key}] most frequent unmapped: "
              + ", ".join(s for s, _ in unmapped.most_common(15)))

    if not summary:
        return 1
    out = pd.DataFrame(summary)
    out.to_csv(C.RESULTS_DIR / "geo_audit_summary.csv", index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nWritten -> {C.RESULTS_DIR}/unmapped_entities_*.csv")
    print("Read those lists before quoting G. If non-Western toponyms dominate "
          "the failures, extend the index in 04_geographic.py and re-run 04.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
