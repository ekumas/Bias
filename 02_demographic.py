#!/usr/bin/env python3
"""
Script 02 — D dimension: binary-gendered reference imbalance.

    D_pooled(C) = |f_female(C) - 0.5| * 2 ,  f_female = female / (female + male)

Three companion statistics are emitted alongside D, each answering a question
a reviewer will otherwise ask:

  f_female   SIGNED direction. D is symmetric and throws the sign away, so a
             male-skewed and a female-skewed corpus at equal distance from
             parity are indistinguishable in D alone. Report f_female so the
             direction survives.
  coverage   Fraction of documents containing at least one gendered term. D is
             estimated only over that subpopulation, and the subpopulation
             differs sharply between corpora. Without coverage, D values are
             not comparable.
  D_docavg   Document-averaged variant: the mean over documents of each
             document's own f_female, so every document weighs equally. The
             pooled estimator is dominated by long documents, which matters for
             mixed corpora containing arXiv papers and books alongside web
             text. If pooled and doc-averaged disagree, say so and report both.

Emits PER-DOCUMENT counts, not a single corpus-level number. Script 06 needs
per-document values to bootstrap a confidence interval; reporting D without a
CI is one of the defects this pipeline exists to remove.

Pure regex over a closed-class lexicon — no model, no GPU, no network.

Usage:
    python scripts/02_demographic.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from lexicons import FEMALE_TERMS, MALE_TERMS  # noqa: E402


def build_pattern(terms: list[str]) -> re.Pattern:
    # Longest-first so that e.g. "grandmother" is not consumed by "mother".
    alts = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.compile(rf"\b(?:{alts})\b", re.IGNORECASE)


FEM_RE = build_pattern(FEMALE_TERMS)
MAL_RE = build_pattern(MALE_TERMS)


def dump_lexicon_audit() -> None:
    rows = ([{"term": t, "class": "female"} for t in FEMALE_TERMS]
            + [{"term": t, "class": "male"} for t in MALE_TERMS])
    out = C.RESULTS_DIR / "lexicon_audit.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Lexicon audit ({len(rows)} terms) -> {out}")


def process(key: str) -> pd.DataFrame | None:
    src = C.DATA_DIR / f"sample_{key}.parquet"
    if not src.exists():
        print(f"  [{key}] SKIP — {src.name} not found. Run 01_sample.py first.")
        return None

    df = pd.read_parquet(src, columns=["doc_id", "text"])
    fem, mal = [], []
    for t in df["text"]:
        fem.append(len(FEM_RE.findall(t)))
        mal.append(len(MAL_RE.findall(t)))

    out_df = pd.DataFrame({
        "doc_id": df["doc_id"],
        "corpus": key,
        "n_female": fem,
        "n_male": mal,
    })
    tot_doc = out_df.n_female + out_df.n_male
    out_df["has_terms"] = (tot_doc > 0).astype(int)
    out_df["f_female_doc"] = (out_df.n_female / tot_doc.where(tot_doc > 0))
    out = C.RESULTS_DIR / f"perdoc_demographic_{key}.parquet"
    out_df.to_parquet(out, index=False)

    tf, tm = int(out_df.n_female.sum()), int(out_df.n_male.sum())
    total = tf + tm
    if total == 0:
        print(f"  [{key}] WARNING: zero gendered terms found — D undefined.")
        return out_df

    f_female = tf / total
    d_pooled = abs(f_female - 0.5) * 2
    covered = out_df[out_df.has_terms == 1]
    coverage = len(covered) / len(out_df)
    f_doc = float(covered.f_female_doc.mean())
    d_docavg = abs(f_doc - 0.5) * 2
    skew = "female-skewed" if f_female > 0.5 else "male-skewed"

    print(f"  [{key}] D_pooled={d_pooled:.4f}  D_docavg={d_docavg:.4f}")
    print(f"           f_female={f_female:.4f} ({skew})  "
          f"f_female_docavg={f_doc:.4f}")
    print(f"           coverage={coverage:.1%}  "
          f"terms/doc={(total/len(out_df)):.1f}  (F={tf:,} M={tm:,})")
    if abs(d_pooled - d_docavg) > 0.05:
        print(f"           NOTE: pooled and doc-averaged D differ by "
              f"{abs(d_pooled - d_docavg):.3f} — long documents are "
              f"influential here. Report both.")
    if coverage < 0.50:
        print(f"           NOTE: D estimated from only {coverage:.0%} of "
              f"documents. Quote coverage next to D.")
    return out_df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    a = ap.parse_args()

    dump_lexicon_audit()
    keys = [a.corpus] if a.corpus else list(C.CORPORA)
    print("\n=== D dimension ===")
    any_ok = False
    for k in keys:
        if process(k) is not None:
            any_ok = True
    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
