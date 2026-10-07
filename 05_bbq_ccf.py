#!/usr/bin/env python3
"""
Script 05 — BBQ Category Coverage Failure (CCF).

    CCF(C) = |{g in G : df(g, C) < theta}| / |G|

where G is the set of BBQ demographic subgroup terms, df(g, C) is the DOCUMENT
FREQUENCY of g per 10,000 documents, and theta is a coverage threshold.

The question CCF answers: if a model saw the term "Yemeni" in almost none of
its pretraining data, what is BBQ's Nationality score for that subgroup
actually measuring? Coverage is a precondition for a benchmark score to be
interpretable, and it is measurable directly from the corpus.

SOURCE
------
Data comes from the authors' GitHub repository (nyu-mll/BBQ) as raw JSONL, not
from a HuggingFace re-upload. Two reasons: HF mirrors of BBQ are backed by
loading scripts, which datasets 4.0 removed; and the original artefact is the
right thing to extract terms from anyway.

TERM EXTRACTION -- THE ANALYST DECISION IN THIS SCRIPT
-----------------------------------------------------
BBQ's group labels mix two incompatible kinds of string:

  natural language   Jewish, Kenyan, lesbian, obese, disabled
  annotation codes   lowSES-Hispanic, F-Black, nonOld, negDress, ArabStates

Only the first kind can be searched for in a corpus. Searching for
"lowSES-Hispanic" returns zero in every corpus, which would make CCF report
near-total failure everywhere and measure nothing. This script therefore
decomposes compound codes into their natural-language parts (F-Black -> Black),
drops pure schema codes, and supplies an explicit surface vocabulary for Age,
whose labels are old/nonOld and would otherwise leave the category empty.

That filtering is a judgement call and it moves CCF. It is dumped in full to
results/bbq_terms.csv. Audit that file and reproduce its contents in an
appendix; a reviewer cannot check CCF without it.

    python scripts/05_bbq_ccf.py
    python scripts/05_bbq_ccf.py --terms-only    # extract and stop
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

GITHUB_BASE = "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/"
CATEGORIES = [
    "Age", "Disability_status", "Gender_identity", "Nationality",
    "Physical_appearance", "Race_ethnicity", "Race_x_SES", "Race_x_gender",
    "Religion", "SES", "Sexual_orientation",
]

COMPOUND = re.compile(r"^(lowSES|highSES|[FM])[-_]|[-_](lowSES|highSES|[FM])$", re.I)
DROP = {
    "unknown", "f", "m", "nonold", "nondisabled", "nontrans", "nonobese",
    "notpregnant", "novisibledifference", "visibledifference", "negdress",
    "posdress", "arabstates", "low ses", "high ses",
}
RENAME = {
    "lowses": "low-income", "highses": "wealthy",
    "asiapacific": "Asian", "latinsouthamerica": "Latin American",
    "middleeast": "Middle Eastern", "northamerica": "North American",
    "africa": "African", "europe": "European", "d/deaf": "deaf",
    "mentally-ill": "mentally ill",
    "people with blindness or low-vision": "blind",
}
# BBQ encodes Age as old/nonOld only. Without an explicit surface vocabulary
# the whole category drops out of CCF. These terms are the analyst's, not
# BBQ's -- say so in the paper.
AGE_TERMS = ["old", "elderly", "old man", "old woman", "senior citizen",
             "retiree", "teenager", "young man", "young woman", "middle-aged"]


def clean_term(term: str) -> list[str]:
    t = str(term).strip()
    if not t:
        return []
    low = t.lower()
    if low in RENAME:
        return [RENAME[low]]
    if low in DROP:
        return []
    parts = re.split(r"[-_]", t) if COMPOUND.search(t) else [t]
    out = []
    for p in parts:
        p = p.strip()
        low = p.lower()
        if not p or low in DROP or len(p) <= 2:
            continue
        if low in RENAME:
            out.append(RENAME[low])
            continue
        if re.search(r"[a-z][A-Z]", p):
            continue
        out.append(p)
    return out


def extract_bbq_terms() -> pd.DataFrame:
    print(f"Fetching BBQ from {GITHUB_BASE}")
    terms: dict[str, dict[str, str]] = defaultdict(dict)
    n_records = 0
    failed = []

    for cat in CATEGORIES:
        url = f"{GITHUB_BASE}{cat}.jsonl"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                lines = r.read().decode("utf-8").strip().split("\n")
        except Exception as exc:
            print(f"  {cat}: FAILED ({str(exc)[:70]})")
            failed.append(cat)
            continue
        n_records += len(lines)
        for ln in lines:
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            for v in (rec.get("answer_info") or {}).values():
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    for t in clean_term(v[1]):
                        terms[cat].setdefault(t.lower(), t)
            md = rec.get("additional_metadata") or {}
            for g in (md.get("stereotyped_groups") or []):
                for t in clean_term(g):
                    terms[cat].setdefault(t.lower(), t)
        if cat == "Age":
            for t in AGE_TERMS:
                terms["Age"].setdefault(t.lower(), t)
        print(f"  {cat:22} {len(lines):6,} records -> {len(terms[cat]):3} terms"
              + ("  (incl. analyst-supplied age vocabulary)" if cat == "Age" else ""))

    if failed:
        raise RuntimeError(
            f"Could not fetch: {failed}. Check connectivity, or download the "
            f"JSONL files manually from github.com/nyu-mll/BBQ into data/bbq/."
        )

    rows = [{"category": c, "term": v} for c, d in terms.items() for v in d.values()]
    df = pd.DataFrame(rows).sort_values(["category", "term"]).reset_index(drop=True)
    out = C.RESULTS_DIR / "bbq_terms.csv"
    df.to_csv(out, index=False)
    n_distinct = df.term.str.lower().nunique()
    print(f"\n  {n_records:,} BBQ records -> {len(df)} (category, term) rows, "
          f"{n_distinct} distinct terms -> {out}")
    print(f"  AUDIT {out.name} BEFORE REPORTING CCF.")
    return df


def doc_frequency(texts: list[str], terms: list[str]) -> dict[str, int]:
    pats = {t: re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in terms}
    counts = dict.fromkeys(terms, 0)
    for txt in texts:
        for t, p in pats.items():
            if p.search(txt):
                counts[t] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--terms-only", action="store_true")
    a = ap.parse_args()

    terms_df = extract_bbq_terms()
    if a.terms_only:
        print("\n--terms-only: stopping. Inspect results/bbq_terms.csv.")
        return 0

    terms = sorted({t for t in terms_df.term}, key=str.lower)
    cat_of = terms_df.groupby("term").category.apply(list).to_dict()
    keys = [a.corpus] if a.corpus else list(C.CORPORA)

    detail, summary = [], []
    print("\n=== BBQ CCF ===")
    for k in keys:
        src = C.DATA_DIR / f"sample_{k}.parquet"
        if not src.exists():
            print(f"  [{k}] SKIP — {src.name} not found.")
            continue
        texts = pd.read_parquet(src, columns=["text"]).text.tolist()
        n_docs = len(texts)
        print(f"  [{k}] scanning {len(terms)} terms over {n_docs:,} docs ...",
              flush=True)
        counts = doc_frequency(texts, terms)

        for t in terms:
            detail.append({
                "corpus": k, "term": t,
                "categories": "|".join(cat_of.get(t, [])),
                "doc_count": counts[t],
                "per_10k": round(10_000 * counts[t] / n_docs, 4) if n_docs else 0.0,
            })

        row = {"corpus": k, "n_docs": n_docs, "n_terms": len(terms)}
        for th in C.BBQ_THRESHOLD_SWEEP:
            fails = sum(1 for t in terms
                        if (10_000 * counts[t] / n_docs if n_docs else 0) < th)
            row[f"CCF@{th:g}"] = round(fails / len(terms), 4)
        row["CCF"] = row[f"CCF@{C.BBQ_THRESHOLD_PER_10K:g}"]
        row["n_zero"] = sum(1 for t in terms if counts[t] == 0)
        summary.append(row)
        print(f"  [{k}] CCF(theta={C.BBQ_THRESHOLD_PER_10K:g}) = {row['CCF']:.4f}   "
              f"({row['n_zero']} terms absent entirely)")

    if not summary:
        print("No corpora processed. Run 01_sample.py first.")
        return 1

    det = pd.DataFrame(detail)
    det.to_csv(C.RESULTS_DIR / "bbq_ccf_detail.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(C.RESULTS_DIR / "bbq_ccf.csv", index=False)

    print("\nThreshold sweep — check whether the corpus RANKING is stable:")
    cols = ["corpus"] + [f"CCF@{th:g}" for th in C.BBQ_THRESHOLD_SWEEP]
    print(summ[cols].to_string(index=False))

    orders = {th: tuple(summ.sort_values(f"CCF@{th:g}", ascending=False).corpus)
              for th in C.BBQ_THRESHOLD_SWEEP}
    n_orders = len(set(orders.values()))
    print(f"\n{n_orders} distinct ordering(s) across "
          f"{len(C.BBQ_THRESHOLD_SWEEP)} thresholds.")
    print("One ordering means the threshold choice does not affect conclusions "
          "— report the sweep and the objection is answered. More than one "
          "means CCF must be reported as a curve, not a number.")

    if not det.empty:
        det2 = det.assign(cat=det.categories.str.split("|")).explode("cat")
        piv = det2.pivot_table(index="cat", columns="corpus",
                               values="per_10k", aggfunc="median").round(2)
        piv.to_csv(C.RESULTS_DIR / "bbq_category_coverage.csv")
        print("\nMedian mentions per 10k documents, by BBQ category:")
        print(piv.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
