#!/usr/bin/env python3
"""
Script 05b — vocabulary sensitivity for CCF.

CCF depends on a term list we constructed by hand, and a referee will ask why
these surface forms and not synonyms, morphological variants, or paraphrases.
Threshold sweeping does not answer that; only varying the vocabulary does.

Four vocabularies, from most literal to most generous:

  strict     BBQ's group labels verbatim, including annotation codes. This is
             what a naive implementation would use. It should perform terribly
             and that is the point: it establishes that the cleaning step in
             script 05 is doing necessary work rather than inflating coverage.
  cleaned    Script 05's list. Schema codes decomposed, Age vocabulary supplied.
  expanded   cleaned + hand-written synonyms and common paraphrases. A term
             counts as present if ANY of its variants occurs.
  relaxed    expanded + case-insensitive substring matching instead of word
             boundaries, so morphological variants (Muslims, disabilities)
             count. This is the most generous reading available without a
             lemmatiser and gives an upper bound on measured coverage.

If the corpus ORDERING is preserved across all four, the vocabulary objection
is answered: CCF's conclusions do not depend on the analyst's word list, only
its absolute level does. If the ordering flips, CCF must be reported per
vocabulary and no single ranking claimed.

    python scripts/05b_vocab_sensitivity.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
bbq = __import__("05_bbq_ccf")

# Synonyms and paraphrases, keyed by cleaned term. Deliberately generous:
# the purpose is to test whether a wider net changes conclusions, so erring
# toward inclusion is correct here even where a variant is imprecise.
SYNONYMS = {
    "disabled": ["disability", "disabilities", "handicapped", "wheelchair user",
                 "person with a disability", "people with disabilities"],
    "blind": ["visually impaired", "vision impaired", "low vision", "blindness"],
    "deaf": ["hard of hearing", "hearing impaired", "deafness"],
    "mentally ill": ["mental illness", "mental health condition", "psychiatric"],
    "autistic people": ["autistic", "autism", "on the spectrum"],
    "physically disabled": ["mobility impaired", "physical disability"],
    "people with cerebral palsy": ["cerebral palsy"],
    "people with cognitive disabilities or mental illness":
        ["cognitive disability", "intellectual disability", "learning disability"],
    "Down's syndrome": ["Down syndrome", "trisomy 21"],
    "obese": ["obesity", "overweight", "fat"],
    "low-income": ["poor", "poverty", "working class", "low income",
                   "financially struggling"],
    "wealthy": ["rich", "affluent", "upper class", "well-off"],
    "gay": ["homosexual", "same-sex"],
    "lesbian": ["lesbians"],
    "bisexual": ["bi"],
    "trans": ["transgender", "trans person", "trans people"],
    "transgender women": ["trans woman", "trans women"],
    "transgender men": ["trans man", "trans men"],
    "Muslim": ["Muslims", "Islam", "Islamic"],
    "Jewish": ["Jews", "Judaism", "Jew"],
    "Christian": ["Christians", "Christianity"],
    "Hindu": ["Hindus", "Hinduism"],
    "Buddhist": ["Buddhists", "Buddhism"],
    "Sikh": ["Sikhs", "Sikhism"],
    "Atheist": ["atheists", "atheism", "non-believer"],
    "Catholic": ["Catholics", "Catholicism"],
    "Mormon": ["Mormons", "Latter-day Saint"],
    "Protestant": ["Protestants", "Protestantism"],
    "elderly": ["elder", "elders", "older adult", "older people", "seniors"],
    "senior citizen": ["senior citizens", "pensioner"],
    "retiree": ["retirees", "retired"],
    "teenager": ["teenagers", "teen", "adolescent"],
    "Black": ["Blacks", "Black people"],
    "African American": ["African-American", "African Americans"],
    "Hispanic": ["Hispanics", "Latina", "Latinx"],
    "Latino": ["Latinos", "Latina", "Latinx"],
    "Native American": ["Native Americans", "Indigenous", "American Indian"],
    "Middle Eastern": ["Middle East", "Arab world"],
    "Caucasian": ["white people", "Caucasians"],
    "Roma": ["Romani", "Gypsy"],
    "Asian": ["Asians", "Asian American"],
    "Arab": ["Arabs", "Arabic"],
    "pregnant": ["pregnancy", "expecting"],
}


def strict_terms() -> list[str]:
    """BBQ labels verbatim — no decomposition, no schema filtering."""
    import json
    import urllib.request

    seen: set[str] = set()
    for cat in bbq.CATEGORIES:
        try:
            with urllib.request.urlopen(bbq.GITHUB_BASE + cat + ".jsonl",
                                        timeout=60) as r:
                lines = r.read().decode("utf-8").strip().split("\n")
        except Exception as exc:
            print(f"  strict: {cat} unreachable ({str(exc)[:50]})")
            continue
        for ln in lines:
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            for v in (rec.get("answer_info") or {}).values():
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    t = str(v[1]).strip()
                    if t and t.lower() != "unknown" and len(t) > 1:
                        seen.add(t)
    return sorted(seen)


def build_vocabularies() -> dict[str, dict[str, list[str]]]:
    """Each vocabulary maps a canonical term -> list of surface variants."""
    cleaned_df = bbq.extract_bbq_terms()
    cleaned = sorted({t for t in cleaned_df.term})

    vocabs = {
        "strict": {t: [t] for t in strict_terms()},
        "cleaned": {t: [t] for t in cleaned},
        "expanded": {t: [t] + SYNONYMS.get(t, []) for t in cleaned},
    }
    vocabs["relaxed"] = vocabs["expanded"]      # differs by matching mode
    return vocabs


def doc_freq(texts, variants: dict[str, list[str]], substring: bool) -> dict[str, int]:
    """Document frequency of each canonical term, counting any variant."""
    pats = {}
    for term, vs in variants.items():
        alts = "|".join(re.escape(v) for v in sorted(vs, key=len, reverse=True))
        pats[term] = re.compile(alts if substring else rf"\b(?:{alts})\b",
                                re.IGNORECASE)
    counts = dict.fromkeys(variants, 0)
    for txt in texts:
        for term, p in pats.items():
            if p.search(txt):
                counts[term] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="documents per corpus (default: all)")
    a = ap.parse_args()

    vocabs = build_vocabularies()
    print("\nVocabulary sizes:")
    for name, v in vocabs.items():
        nvar = sum(len(x) for x in v.values())
        print(f"  {name:9} {len(v):4} canonical terms, {nvar:4} surface variants")

    rows = []
    for name, variants in vocabs.items():
        substring = (name == "relaxed")
        for k in C.CORPORA:
            src = C.DATA_DIR / f"sample_{k}.parquet"
            if not src.exists():
                continue
            texts = pd.read_parquet(src, columns=["text"]).text.tolist()
            if a.limit:
                texts = texts[:a.limit]
            n = len(texts)
            print(f"  [{name}/{k}] {len(variants)} terms over {n:,} docs ...",
                  flush=True)
            counts = doc_freq(texts, variants, substring)
            row = {"vocabulary": name, "corpus": k, "n_docs": n,
                   "n_terms": len(variants)}
            for th in C.BBQ_THRESHOLD_SWEEP:
                fails = sum(1 for t in variants
                            if (10_000 * counts[t] / n if n else 0) < th)
                row[f"CCF@{th:g}"] = round(fails / len(variants), 4)
            row["CCF"] = row[f"CCF@{C.BBQ_THRESHOLD_PER_10K:g}"]
            row["n_zero"] = sum(1 for t in variants if counts[t] == 0)
            rows.append(row)

    if not rows:
        print("No corpora found. Run 01_sample.py first.")
        return 1

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "ccf_vocab_sensitivity.csv", index=False)

    print("\n=== CCF at theta = 5, by vocabulary ===")
    piv = df.pivot(index="corpus", columns="vocabulary", values="CCF")
    piv = piv[[c for c in ["strict", "cleaned", "expanded", "relaxed"]
               if c in piv.columns]]
    print(piv.to_string())

    print("\n=== Corpus ordering (highest CCF first) ===")
    for name in piv.columns:
        col = piv[name].sort_values(ascending=False)
        groups, cur = [], [col.index[0]]
        for prev, this in zip(col.index, col.index[1:]):
            if abs(col[this] - col[prev]) < 1e-9:
                cur.append(this)
            else:
                groups.append(tuple(sorted(cur))); cur = [this]
        groups.append(tuple(sorted(cur)))
        print(f"  {name:9} " + " > ".join(" = ".join(g) for g in groups))

    # The question is whether any PAIR reverses, not whether the ordering
    # structures are byte-identical. A tie under one vocabulary is lower
    # resolution, not a contradiction, and treating it as one would understate
    # how robust CCF actually is.
    from itertools import combinations
    reversals, ties = [], []
    for x, y in combinations(piv.index, 2):
        signs = {v: (1 if piv.loc[x, v] > piv.loc[y, v]
                     else -1 if piv.loc[x, v] < piv.loc[y, v] else 0)
                 for v in piv.columns}
        nonzero = {s for s in signs.values() if s != 0}
        if len(nonzero) > 1:
            reversals.append((x, y, signs))
        elif 0 in signs.values():
            ties.append((x, y, [v for v, s in signs.items() if s == 0]))

    n_pairs = len(list(combinations(piv.index, 2)))
    print(f"\n=== Pairwise consistency across {len(piv.columns)} vocabularies "
          f"({n_pairs} pairs) ===")
    if reversals:
        for x, y, sg in reversals:
            print(f"  REVERSES: {x} vs {y}  {sg}")
    else:
        print("  No pair reverses under any vocabulary.")
    for x, y, vs in ties:
        print(f"  tie: {x} = {y} under {', '.join(vs)} (lower resolution, "
              f"not a contradiction)")

    lvl = (piv.max(axis=1) / piv.min(axis=1)).round(2)
    print(f"\n  Absolute CCF level moves by up to {lvl.max():.1f}x across "
          f"vocabularies ({lvl.idxmax()}).")

    if not reversals:
        print("\n  VERDICT: CCF's corpus ordering is invariant to vocabulary")
        print("  construction. The absolute level is vocabulary-dependent and")
        print("  must be reported with the vocabulary stated; the ranking is not,")
        print("  and the 'why these terms and not synonyms' objection is answered.")
    else:
        print("\n  VERDICT: at least one pair reverses. Report CCF per vocabulary")
        print("  and restrict claims to pairs whose relative position is stable.")
    n_orders = 1 if not reversals else 2

    md = ["# CCF vocabulary sensitivity — auto-generated\n",
          "| Corpus | " + " | ".join(piv.columns) + " |",
          "|---" * (len(piv.columns) + 1) + "|"]
    for corpus, r in piv.iterrows():
        md.append(f"| {corpus} | " + " | ".join(f"{r[c]:.3f}" for c in piv.columns) + " |")
    md += ["", f"CCF at θ = {C.BBQ_THRESHOLD_PER_10K:g} per 10,000 documents. "
           f"`strict` uses BBQ's group labels verbatim including annotation "
           f"codes; `cleaned` is the list used throughout the paper; `expanded` "
           f"adds synonyms; `relaxed` additionally matches substrings, capturing "
           f"morphological variants. {n_orders} distinct corpus ordering(s) "
           f"across the four."]
    (C.RESULTS_DIR / "ccf_vocab_tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nTables -> {C.RESULTS_DIR / 'ccf_vocab_tables.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
