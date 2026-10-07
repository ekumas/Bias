#!/usr/bin/env python3
"""
Script 03c — build the stratified annotation set for the toxicity audit.

THREE STRATA, AND WHAT EACH IS FOR
----------------------------------
  <0.50        a spot check for false negatives. NOT an estimate: see below.
  0.50-0.90    where T is actually decided -- 384 of TinyStories' 421 flagged
               documents and 129 of C4's 144 lie here. This stratum carries
               the false-positive result.
  >=0.90       previously unaudited and generously assumed to be true
               positives, which is why the corrected T values in the paper are
               upper bounds. Small enough to enumerate exhaustively (15 in C4,
               37 in TinyStories), so it is a census rather than a sample.

The >=0.90 stratum also repairs a measurement problem in the reliability
statistic. With 37 of 40 documents in one class, inter-annotator kappa
collapses even at high raw agreement, because a skewed marginal distribution
inflates expected chance agreement. At 90% raw agreement kappa is about 0.28 on
the skewed set and about 0.80 on a balanced one -- the same annotators, the
same disagreement rate, a different verdict.

WHAT THE LOW STRATUM CAN AND CANNOT SHOW
----------------------------------------
Documents below 0.50 are 99.7% of the corpus (49,856 of 50,000 in C4), so a
sample of 20 supports no quantitative claim. Finding zero toxic documents in 20
puts the upper bound on the rate at 16%, which scaled to the stratum is 8,000
documents -- fifty-five times the 144 that T flags in the whole corpus. Roughly
1,330 documents would be needed for a bound comparable to T's own count.

The stratum is included anyway, on an asymmetric argument. It cannot establish
the absence of missed toxic content. It can establish its presence: if even two
or three of the twenty are plainly toxic, that is reportable at n = 20 and
would materially change what can be said about T. Report it as a spot check and
say plainly what it does not license.

WHAT IT WRITES
--------------
    results/annotation/annotator_A_<corpus>.csv    blind, shuffled, no scores
    results/annotation/annotator_B_<corpus>.csv    identical content
    results/annotation/KEY_<corpus>.csv            doc_id -> score, stratum

The annotator files carry no toxicity score and no stratum label, and the rows
are shuffled. An annotator who can see the classifier's score is no longer
independent of it, and the whole point of the exercise is to check the
classifier against a judgement formed without it.

    python scripts/03c_annotation_set.py
    python scripts/03c_annotation_set.py --n-high 20 --seed 7
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

AUDIT_CORPORA = ["tinystories", "c4_cc_proxy"]


def read_csv_any(path: Path) -> pd.DataFrame:
    """Read a CSV whatever the spreadsheet application did to it.

    Files round-tripped through Numbers or Excel arrive with any of several
    encodings and, less obviously, with a delimiter that may not be a comma.
    Encodings are tried in order of likelihood and the delimiter is sniffed;
    a lossy read is the last resort so that a stray byte cannot cost the file.
    """
    last = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                if df.shape[1] > 1:
                    if enc != "utf-8" or sep != ",":
                        print(f"       (read {path.name} as {enc}, "
                              f"delimiter {sep!r})")
                    return df
            except Exception as exc:
                last = exc
    raise RuntimeError(f"could not parse {path.name}: {last}")


def extract_doc_ids(path: Path, prefix: str) -> list[str]:
    """Pull document identifiers out of a file whose structure cannot be
    trusted.

    The only thing needed from a previous annotation file is which documents
    were in it. Identifiers have a fixed, unambiguous shape, so scanning for
    that shape succeeds on files whose column structure a CSV parser rejects --
    ragged rows, embedded newlines in excerpts, mismatched quoting. Structured
    parsing is tried first; this is the fallback, and it reports which path it
    took so the difference is never silent.
    """
    try:
        df = read_csv_any(path)
        if "doc_id" in df.columns:
            ids = [str(x) for x in df.doc_id.dropna().unique()
                   if str(x).startswith(prefix)]
            if ids:
                return ids
        print(f"       ({path.name} parsed but has no usable doc_id column; "
              f"scanning for identifiers instead)")
    except Exception as exc:
        print(f"       ({path.name} could not be parsed as a table: "
              f"{str(exc)[:70]}; scanning for identifiers instead)")

    raw = path.read_bytes().decode("utf-8", errors="replace")
    ids = re.findall(rf"\b{re.escape(prefix)}\d+\b", raw)
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    print(f"       (recovered {len(out)} identifiers from {path.name})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-low", type=int, default=20,
                    help="documents from the <0.50 stratum (spot check)")
    ap.add_argument("--n-mid", type=int, default=40,
                    help="documents from the 0.50-0.90 stratum")
    ap.add_argument("--n-high", type=int, default=40,
                    help="documents from the >=0.90 stratum; the whole "
                         "stratum is taken when it is smaller")
    ap.add_argument("--chars", type=int, default=600)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()

    out_dir = C.RESULTS_DIR / "annotation"
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in AUDIT_CORPORA:
        tox_p = C.RESULTS_DIR / f"perdoc_toxicity_{key}.parquet"
        samp_p = C.DATA_DIR / f"sample_{key}.parquet"
        if not tox_p.exists() or not samp_p.exists():
            print(f"  [{key}] SKIP — need {tox_p.name} and {samp_p.name}")
            continue

        df = (pd.read_parquet(tox_p)
              .merge(pd.read_parquet(samp_p, columns=["doc_id", "text"]),
                     on="doc_id", how="left"))

        # Stratum 1 (mid): reuse every document already audited, so that the
        # earlier labels remain usable, then top up to n_mid with fresh draws.
        band_p = C.RESULTS_DIR / f"toxicity_band_{key}.csv"
        pool_mid = df[(df.toxicity >= 0.50) & (df.toxicity < 0.90)]
        if band_p.exists():
            prev_ids = extract_doc_ids(band_p, f"{key}_")
            reused = df[df.doc_id.isin(prev_ids)]
            extra_n = max(0, min(a.n_mid - len(reused), len(pool_mid) - len(reused)))
            extra = (pool_mid[~pool_mid.doc_id.isin(prev_ids)]
                     .sample(n=extra_n, random_state=a.seed))
            band = pd.concat([reused, extra], ignore_index=True).copy()
            print(f"  [{key}] mid stratum: {len(reused)} reused + {len(extra)} new "
                  f"= {len(band)}")
        else:
            band = pool_mid.sample(n=min(a.n_mid, len(pool_mid)),
                                   random_state=a.seed).copy()
            print(f"  [{key}] mid stratum: {band_p.name} not found, drew {len(band)}")
        band["stratum"] = "0.50-0.90"

        # Stratum 2 (high): census where the stratum is smaller than requested.
        pool_high = df[df.toxicity >= 0.90]
        n_high = min(a.n_high, len(pool_high))
        high = pool_high.sample(n=n_high, random_state=a.seed).copy()
        high["stratum"] = ">=0.90"
        if len(pool_high) <= a.n_high:
            print(f"  [{key}] high stratum: {len(pool_high)} documents exist; "
                  f"taking all of them (census, not a sample)")
        else:
            print(f"  [{key}] high stratum: {n_high} of {len(pool_high)}")

        # Stratum 3 (low): spot check only. This stratum is ~99.7% of the
        # corpus, so n=20 cannot estimate the false-negative rate; it can only
        # detect it if it is gross.
        pool_low = df[df.toxicity < 0.50]
        low = pool_low.sample(n=min(a.n_low, len(pool_low)),
                              random_state=a.seed).copy()
        low["stratum"] = "<0.50"
        print(f"  [{key}] low stratum: {len(low)} of {len(pool_low):,} "
              f"({100 * len(pool_low) / len(df):.1f}% of the corpus) — spot check")

        both = pd.concat([low, band, high], ignore_index=True)
        both["excerpt"] = (both.text.str.replace(r"\s+", " ", regex=True)
                           .str[: a.chars])

        # Shuffle so that stratum is not recoverable from row order.
        both = both.sample(frac=1.0, random_state=a.seed + 1).reset_index(drop=True)
        both["item"] = [f"{key[:4]}_{i + 1:03d}" for i in range(len(both))]

        blind = both[["item", "excerpt"]].copy()
        blind["label"] = ""       # toxic / not_toxic / borderline
        blind["note"] = ""
        # Both formats are written. An annotator who opens the .xlsx can save
        # in place without an export step, which is where labelled work most
        # often gets lost; script 03d accepts either.
        for who in ("A", "B"):
            blind.to_csv(out_dir / f"annotator_{who}_{key}.csv", index=False)
            try:
                blind.to_excel(out_dir / f"annotator_{who}_{key}.xlsx",
                               index=False)
            except Exception as exc:
                print(f"       (no .xlsx written: {str(exc)[:60]}; "
                      f"pip install openpyxl if you want one)")
        both[["item", "doc_id", "toxicity", "stratum"]].to_csv(
            out_dir / f"KEY_{key}.csv", index=False)

    # A pre-formatted sheet for recording adjudications. It deliberately
    # carries no excerpt column: excerpts are what make disagreements.csv
    # unparseable after a spreadsheet round-trip, and the decisions recorded
    # here are too consequential to lose to a delimiter.
    adj_p = out_dir / "adjudications.xlsx"
    if not adj_p.exists():
        try:
            pd.DataFrame({"item": pd.Series(dtype=str),
                          "label": pd.Series(dtype=str),
                          "rationale": pd.Series(dtype=str)}).to_excel(
                adj_p, index=False)
            print(f"\nBlank adjudication sheet -> {adj_p.name} "
                  f"(columns: item, label, rationale)")
        except Exception as exc:
            print(f"\n(no adjudication sheet written: {str(exc)[:60]})")

        counts = both.stratum.value_counts().to_dict()
        print(f"  [{key}] {len(both)} items total "
              f"({counts.get('<0.50', 0)} low / "
              f"{counts.get('0.50-0.90', 0)} mid / "
              f"{counts.get('>=0.90', 0)} high) "
              f"-> annotator_A/B_{key}.csv, KEY_{key}.csv")

    print(f"\nWritten to {out_dir}")
    print("Give annotator_B_*.csv to the second annotator together with the")
    print("guidelines. Neither file contains the classifier score. Do not share")
    print("existing labels: an annotator who has seen them is not independent,")
    print("and the agreement statistic then measures influence, not reliability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
