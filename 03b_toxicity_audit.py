#!/usr/bin/env python3
"""
Script 03b — toxicity audit: is T measuring what we think it is?

TinyStories returns T = 0.0084, three times C4's rate. Children's fiction being
three times more toxic than unfiltered web text is implausible on its face, and
the most likely explanation is that Detoxify flags narrative conflict — a
villain's dialogue, a character being frightened or hurt — as toxic.

This matters beyond one corpus. If T is systematically inflated on narrative
text, then T is not comparable across corpora of different genres, and the
paper cannot place a narrative corpus and a web corpus on the same T axis.
Identifying that is a result, not a defeat: it bounds where a
classifier-mediated measure can be used.

The script cannot decide the question. It extracts the evidence:

    results/toxicity_top_<corpus>.csv     highest-scoring documents, with text
    results/toxicity_distribution.csv     score distribution per corpus
    results/toxicity_audit_sheet.md       a form to record your judgements

READ THE EXCERPTS AND LABEL THEM. The paper needs a number — "k of 20 top-scoring
TinyStories documents contain no toxic content" — not an impression.

    python scripts/03b_toxicity_audit.py
    python scripts/03b_toxicity_audit.py --top 40 --chars 600
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--chars", type=int, default=500,
                    help="characters of each document to write out")
    ap.add_argument("--corpus", default=None)
    a = ap.parse_args()

    keys = [a.corpus] if a.corpus else list(C.CORPORA)
    dist_rows = []
    audited = []

    for k in keys:
        tox_p = C.RESULTS_DIR / f"perdoc_toxicity_{k}.parquet"
        samp_p = C.DATA_DIR / f"sample_{k}.parquet"
        if not tox_p.exists() or not samp_p.exists():
            print(f"  [{k}] SKIP — need both {tox_p.name} and {samp_p.name}")
            continue

        tox = pd.read_parquet(tox_p)
        samp = pd.read_parquet(samp_p, columns=["doc_id", "text"])
        df = tox.merge(samp, on="doc_id", how="left")
        v = df.loc[df.toxicity.notna(), "toxicity"]

        dist_rows.append({
            "corpus": k, "n_scored": len(v),
            "mean": round(float(v.mean()), 5),
            "median": round(float(v.median()), 5),
            "p90": round(float(v.quantile(0.90)), 5),
            "p99": round(float(v.quantile(0.99)), 5),
            "p999": round(float(v.quantile(0.999)), 5),
            "max": round(float(v.max()), 5),
            "n_above_0.5": int((v >= 0.5).sum()),
            "n_above_0.9": int((v >= 0.9).sum()),
        })

        top = df.nlargest(a.top, "toxicity")[["doc_id", "toxicity", "text"]].copy()
        top["excerpt"] = top.text.str.replace(r"\s+", " ", regex=True).str[:a.chars]
        top["your_label"] = ""       # toxic / not_toxic / borderline
        top["note"] = ""
        out = C.RESULTS_DIR / f"toxicity_top_{k}.csv"
        top[["doc_id", "toxicity", "your_label", "note", "excerpt"]].to_csv(
            out, index=False)
        audited.append(k)
        print(f"  [{k}] top {a.top} written -> {out.name}   "
              f"(scores {top.toxicity.min():.3f}–{top.toxicity.max():.3f})")

    if not dist_rows:
        print("Nothing audited. Ensure perdoc_toxicity_*.parquet are in results/.")
        return 1

    dist = pd.DataFrame(dist_rows)
    dist.to_csv(C.RESULTS_DIR / "toxicity_distribution.csv", index=False)
    print("\n=== Score distribution ===")
    print(dist.to_string(index=False))

    # A shape check that is diagnostic on its own: if a corpus's flagged
    # documents sit just above the threshold while another's are decisively
    # toxic, the two T values are not measuring the same thing.
    print("\n=== Threshold proximity of flagged documents ===")
    for _, r in dist.iterrows():
        n5, n9 = r["n_above_0.5"], r["n_above_0.9"]
        share = n9 / n5 if n5 else float("nan")
        print(f"  {r['corpus']:16} {n5:5} docs >=0.50, {n9:4} >=0.90  "
              f"({share:.1%} of flagged are high-confidence)")
    print("  A low share means the corpus's flagged documents cluster near the")
    print("  threshold — consistent with borderline or false-positive flagging.")

    sheet = ["# Toxicity audit sheet\n",
             "Open each `results/toxicity_top_<corpus>.csv`, read the excerpts, "
             "and fill `your_label` with `toxic`, `not_toxic`, or `borderline`. "
             "Then record the counts here and paste the resulting sentence into "
             "the paper.\n",
             "| Corpus | Reviewed | toxic | not_toxic | borderline | False-positive rate |",
             "|---|---|---|---|---|---|"]
    for k in audited:
        sheet.append(f"| {k} | {a.top} |  |  |  |  |")
    sheet += [
        "\n## What each outcome means for the paper\n",
        "- **Mostly `toxic`** — T is behaving. Report the audit as a validity "
        "check and keep T as it stands.",
        "- **Mostly `not_toxic` for TinyStories only** — Detoxify false-positives "
        "on narrative conflict. T is not comparable across genres. Report T for "
        "TinyStories with an explicit caveat, or mark it undefined as we did for "
        "G, and say why.",
        "- **Mostly `not_toxic` everywhere** — T is unreliable at this threshold "
        "across the board. Re-examine the threshold sweep and consider dropping "
        "T from the composite entirely.\n",
        "Whichever holds, state the audited count in the paper. An unexamined "
        "measure that the authors themselves flagged as suspicious is a stronger "
        "objection than any result it produces.\n",
    ]
    (C.RESULTS_DIR / "toxicity_audit_sheet.md").write_text(
        "\n".join(sheet), encoding="utf-8")
    print(f"\nAudit sheet -> {C.RESULTS_DIR / 'toxicity_audit_sheet.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
