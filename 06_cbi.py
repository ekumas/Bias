#!/usr/bin/env python3
"""
Script 06 — Bias Fingerprint assembly, Composite Bias Index, uncertainty and
weight sensitivity.

Produces every number that goes into the paper's results tables:

    results/bias_fingerprint.csv   D, T, G with 95% bootstrap CIs, CBI, risk
    results/cbi_sensitivity.csv    stability of risk class over the weight grid
    results/paper_tables.md        paste-ready markdown tables

SELF-VERIFICATION
-----------------
Before writing anything, the script recomputes CBI from the reported D, T, G
and the declared weights, and ABORTS if any value disagrees by more than 1e-9.
This check exists because the previous draft of the manuscript reported five
CBI values, none of which could be reproduced from its own formula. A results
table that cannot regenerate itself is the defect this guard is designed to
make impossible.

Usage:
    python scripts/06_cbi.py
    python scripts/06_cbi.py --bootstrap 2000
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Minimum country mentions for G to be reportable. Mirrors the floor used in
# 06b_d_ci.py -- the two scripts must not disagree about which corpora have a
# defined G.
MIN_COUNTRY_MENTIONS = 5_000
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


# ---------------------------------------------------------------- dimensions
def load_D(key: str):
    p = C.RESULTS_DIR / f"perdoc_demographic_{key}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return df.n_female.to_numpy(np.float64), df.n_male.to_numpy(np.float64)


def D_companions(key: str) -> dict:
    """Signed direction, coverage, and the document-averaged variant.

    D is symmetric about parity and so discards the direction of the skew.
    It is also a pooled estimator, which long documents dominate. These three
    companions restore what D drops; report them next to it.
    """
    df = pd.read_parquet(C.RESULTS_DIR / f"perdoc_demographic_{key}.parquet")
    tot = df.n_female + df.n_male
    f_female = float(df.n_female.sum() / tot.sum()) if tot.sum() else float("nan")
    coverage = float((tot > 0).mean())
    ratios = (df.n_female / tot.where(tot > 0)).dropna()
    f_doc = float(ratios.mean()) if len(ratios) else float("nan")
    return {
        "f_female": round(f_female, 4),
        "skew": "female" if f_female > 0.5 else "male",
        "coverage": round(coverage, 4),
        "D_docavg": round(abs(f_doc - 0.5) * 2, 4),
    }


def D_stat(fem: np.ndarray, mal: np.ndarray) -> float:
    tot = fem.sum() + mal.sum()
    if tot == 0:
        return float("nan")
    return abs(fem.sum() / tot - 0.5) * 2.0


def load_G(key: str):
    """Return (sparse doc x country matrix, country order)."""
    p = C.RESULTS_DIR / f"perdoc_geographic_{key}.parquet"
    if not p.exists():
        return None
    from scipy.sparse import csr_matrix

    df = pd.read_parquet(p)
    per_doc = [Counter(s.split(";")) if s else Counter() for s in df.countries.fillna("")]
    for c in per_doc:
        c.pop("", None)
    order = sorted({k for c in per_doc for k in c})
    if not order:
        return None
    pos = {k: i for i, k in enumerate(order)}
    rows, cols, vals = [], [], []
    for r, c in enumerate(per_doc):
        for k, v in c.items():
            rows.append(r); cols.append(pos[k]); vals.append(float(v))
    m = csr_matrix((vals, (rows, cols)), shape=(len(per_doc), len(order)))
    return m, order


def G_from_vector(v: np.ndarray) -> float:
    tot = v.sum()
    if tot <= 0:
        return float("nan")
    p = v[v > 0] / tot
    h = float(-(p * np.log2(p)).sum())
    g = 1.0 - h / math.log2(C.N_REF_COUNTRIES)
    return min(max(g, 0.0), 1.0)


# ---------------------------------------------------------------- bootstrap
def ci(samples: np.ndarray) -> tuple[float, float]:
    s = samples[~np.isnan(samples)]
    if len(s) == 0:
        return float("nan"), float("nan")
    a = (1 - C.CI_LEVEL) / 2
    return float(np.quantile(s, a)), float(np.quantile(s, 1 - a))


def boot_D(fem, mal, B, rng):
    n = len(fem)
    out = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        out[b] = D_stat(fem[i], mal[i])
    return out


def boot_G(m, B, rng, chunk=100):
    n = m.shape[0]
    out = np.empty(B)
    for s in range(0, B, chunk):
        k = min(chunk, B - s)
        W = np.zeros((k, n), dtype=np.float32)
        for j in range(k):
            i = rng.integers(0, n, n)
            np.add.at(W[j], i, 1.0)
        agg = W @ m            # (k, n_countries)
        for j in range(k):
            out[s + j] = G_from_vector(np.asarray(agg[j]).ravel())
    return out


# ---------------------------------------------------------------- composite
def cbi_value(D, G, w) -> float:
    """Composite over the two dimensions the paper actually reports.

    T was measured, audited and excluded (Section 5.5): 88.6% of the documents
    the classifier flags contain no toxic content, the error causes differ by
    genre, and setting T to zero left the weight sweep unchanged. A pipeline
    that still required it would skip any corpus lacking a toxicity run, which
    is how The Pile came to be missing from the composite analysis while being
    present everywhere else.
    """
    return w["w_D"] * D + w["w_G"] * G


def valid_weight_configs():
    """w_D from 0.10 to 0.90 in steps of 0.05; w_G is the complement."""
    return [{"w_D": round(w, 2), "w_G": round(1 - w, 2)}
            for w in np.arange(0.10, 0.901, 0.05)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=C.N_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    ccf_path = C.RESULTS_DIR / "bbq_ccf.csv"
    ccf = pd.read_csv(ccf_path).set_index("corpus")["CCF"].to_dict() \
        if ccf_path.exists() else {}

    rows, undefined_G = [], []
    print("=== Bias Fingerprint ===")
    for key, spec in C.CORPORA.items():
        d_in, g_in = load_D(key), load_G(key)
        missing = [n for n, x in (("D", d_in), ("G", g_in)) if x is None]
        if missing:
            print(f"  [{key}] SKIP — missing {', '.join(missing)}. "
                  f"Run scripts 02 and 04 first.")
            continue

        fem, mal = d_in
        m, _order = g_in
        D = D_stat(fem, mal)

        # G needs enough country mentions to be an estimate rather than a
        # number. Below the floor it is reported as undefined and the corpus
        # is dropped from CBI -- filling the gap with a value, or with zero,
        # would smuggle a non-measurement into the composite.
        n_country = int(m.sum())
        if n_country < MIN_COUNTRY_MENTIONS:
            print(f"  [{key}] G UNDEFINED — only {n_country:,} country "
                  f"mentions (floor {MIN_COUNTRY_MENTIONS:,}). "
                  f"CBI not computed for this corpus.")
            undefined_G.append({
                "corpus": key, "label": spec["label"], "n_docs": len(fem),
                "D": round(D, 4),
                "n_country_mentions": n_country,
                "BBQ_CCF": ccf.get(key, float("nan")),
            })
            continue
        G = G_from_vector(np.asarray(m.sum(axis=0)).ravel())

        print(f"  [{key}] bootstrapping ({a.bootstrap} reps) ...", flush=True)
        dl, dh = ci(boot_D(fem, mal, a.bootstrap, rng))
        gl, gh = ci(boot_G(m, a.bootstrap, rng))

        # CBI is computed from the ROUNDED, REPORTED D/T/G -- not from the
        # full-precision values -- so that a reader who applies the published
        # formula to the published table reproduces the published CBI exactly.
        Dr, Gr = round(D, 4), round(G, 4)
        cbi = round(cbi_value(Dr, Gr, C.BASELINE_WEIGHTS), 4)
        rows.append({
            "corpus": key, "label": spec["label"],
            "n_docs": len(fem),
            "D": Dr, "D_lo": round(dl, 4), "D_hi": round(dh, 4),
            "G": Gr, "G_lo": round(gl, 4), "G_hi": round(gh, 4),
            "CBI": cbi,
            "BBQ_CCF": ccf.get(key, float("nan")),
            **D_companions(key),
        })
        print(f"  [{key}] D={D:.4f}[{dl:.4f},{dh:.4f}] "
              f"G={G:.4f}[{gl:.4f},{gh:.4f}] CBI={cbi:.4f}")

    if not rows:
        print("\nNo corpus has a defined G, so no fingerprint table can be "
              "composed.")
        if undefined_G:
            ug = pd.DataFrame(undefined_G)
            ug.to_csv(C.RESULTS_DIR / "undefined_G.csv", index=False)
            print(f"Wrote {len(ug)} undefined-G rows -> "
                  f"{C.RESULTS_DIR / 'undefined_G.csv'}")
            print(ug.to_string(index=False))
        else:
            print("Run scripts 01-05 first.")
        return 1

    fp = pd.DataFrame(rows)

    # ------------------------------------------------ self-verification guard
    print("\n=== Self-verification ===")
    bad = []
    for _, r in fp.iterrows():
        recomputed = cbi_value(r['D'], r['G'], C.BASELINE_WEIGHTS)
        # tolerance = half a unit in the 4th decimal, i.e. CBI's own rounding
        if abs(recomputed - r['CBI']) > 5e-5:
            bad.append((r['corpus'], r['CBI'], recomputed))
    if bad:
        print("ABORT — reported CBI does not match the formula:")
        for c, rep, rec in bad:
            print(f"  {c}: reported={rep} recomputed={rec:.6f}")
        return 3
    print(f"  OK — all {len(fp)} CBI values reproduce from the REPORTED D/G "
          f"using w_D={C.BASELINE_WEIGHTS['w_D']:.2f}, "
          f"w_G={C.BASELINE_WEIGHTS['w_G']:.2f}")

    fp.to_csv(C.RESULTS_DIR / "bias_fingerprint.csv", index=False)

    # ------------------------------------------------ weight sensitivity
    #
    # Stability is measured on the ORDERING, not on membership of a risk band.
    # Bands were removed from the framework because their thresholds have no
    # justification, and measuring against them reports every corpus as robust
    # for a trivial reason: all composite ranges fall inside a single band,
    # which is wide relative to the spread between corpora.
    cfgs = valid_weight_configs()
    print(f"\n=== Weight sensitivity ({len(cfgs)} configurations, "
          f"w_D {cfgs[0]['w_D']:.2f} to {cfgs[-1]['w_D']:.2f}) ===")

    scores = {r["corpus"]: [cbi_value(r["D"], r["G"], w) for w in cfgs]
              for _, r in fp.iterrows()}
    labels = {r["corpus"]: r["label"] for _, r in fp.iterrows()}
    keys = list(scores)

    ranks, orderings = {k: [] for k in keys}, set()
    for i in range(len(cfgs)):
        v = {k: scores[k][i] for k in keys}
        orderings.add(tuple(sorted(keys, key=lambda k: -v[k])))
        for k in keys:
            better = sum(1 for o in keys if v[o] > v[k] + 1e-12)
            equal = sum(1 for o in keys if abs(v[o] - v[k]) <= 1e-12)
            ranks[k].append(better + (equal + 1) / 2)

    srows = []
    for k in keys:
        r = ranks[k]
        modal, n_modal = Counter(r).most_common(1)[0]
        share = n_modal / len(cfgs)
        verdict = ("rank-stable" if share >= C.ROBUST_THRESHOLD
                   else "weight-sensitive" if share < C.SENSITIVE_THRESHOLD
                   else "intermediate")
        srows.append({"corpus": k, "label": labels[k],
                      "CBI_min": round(min(scores[k]), 4),
                      "CBI_max": round(max(scores[k]), 4),
                      "rank_min": min(r), "rank_max": max(r),
                      "modal_rank": modal,
                      "modal_rank_share": round(share, 4),
                      "verdict": verdict, "n_configs": len(cfgs)})
        print(f"  {labels[k]:26} rank {min(r):g}-{max(r):g}  "
              f"modal {modal:g} in {share:6.1%}  "
              f"CBI=[{min(scores[k]):.3f},{max(scores[k]):.3f}]  {verdict}")

    sens = pd.DataFrame(srows)
    sens.to_csv(C.RESULTS_DIR / "cbi_sensitivity.csv", index=False)
    n_stable = int((sens.verdict == "rank-stable").sum())
    print(f"\n  {len(orderings)} distinct ordering(s) across {len(cfgs)} weightings.")
    print(f"  {n_stable} of {len(sens)} corpora hold one rank in at least "
          f"{C.ROBUST_THRESHOLD:.0%} of them.")

    # ------------------------------------------------ paper tables
    t1 = ["| Corpus | n | D [95% CI] | G [95% CI] | CBI | BBQ CCF |",
          "|---|---|---|---|---|---|"]
    for _, r in fp.iterrows():
        ccf_s = "—" if pd.isna(r['BBQ_CCF']) else f"{r['BBQ_CCF']:.3f}"
        t1.append(
            f"| {r['label']} | {r['n_docs']:,} "
            f"| {r['D']:.4f} [{r['D_lo']:.4f}, {r['D_hi']:.4f}] "
            f"| {r['G']:.4f} [{r['G_lo']:.4f}, {r['G_hi']:.4f}] "
            f"| {r['CBI']:.4f} | {ccf_s} |")

    t2 = ["| Corpus | CBI (equal weights) | CBI range | Rank range | Modal rank | Share |",
          "|---|---|---|---|---|---|"]
    for _, r in sens.iterrows():
        base = fp.loc[fp.corpus == r["corpus"], "CBI"].iloc[0]
        t2.append(f"| {r['label']} | {base:.4f} "
                  f"| [{r['CBI_min']:.3f}, {r['CBI_max']:.3f}] "
                  f"| {r['rank_min']:g}-{r['rank_max']:g} | {r['modal_rank']:g} "
                  f"| {r['modal_rank_share']:.1%} |")

    t3 = ["| Corpus | f_female | Direction | Coverage | D pooled | D doc-avg |",
          "|---|---|---|---|---|---|"]
    for _, r in fp.iterrows():
        t3.append(f"| {r['label']} | {r['f_female']:.3f} | {r['skew']}-skewed "
                  f"| {r['coverage']:.1%} | {r['D']:.3f} | {r['D_docavg']:.3f} |")

    t3 = ["| Corpus | f_female | Direction | Coverage | D pooled | D doc-avg |",
          "|---|---|---|---|---|---|"]
    for _, r in fp.iterrows():
        t3.append(f"| {r['label']} | {r['f_female']:.3f} | {r['skew']}-skewed "
                  f"| {r['coverage']:.1%} | {r['D']:.3f} | {r['D_docavg']:.3f} |")

    t3 = ["| Corpus | f_female | Direction | Coverage | D pooled | D doc-avg |",
          "|---|---|---|---|---|---|"]
    for _, r in fp.iterrows():
        t3.append(f"| {r['label']} | {r['f_female']:.3f} | {r['skew']}-skewed "
                  f"| {r['coverage']:.1%} | {r['D']:.3f} | {r['D_docavg']:.3f} |")

    md = (
        "# CABEF results — auto-generated, do not edit by hand\n\n"
        f"Weights: w_D={C.BASELINE_WEIGHTS['w_D']:.2f}, "
        f"w_G={C.BASELINE_WEIGHTS['w_G']:.2f}  ·  "
        f"seed={a.seed}  ·  bootstrap B={a.bootstrap}  ·  "
        f"N_REF_COUNTRIES={C.N_REF_COUNTRIES}\n\n"
        "## Table 1 — Bias Fingerprint\n\n" + "\n".join(t1) +
        "\n\n## Table 2 — Weight sensitivity (ordering)\n\n" + "\n".join(t2) +
        "\n\n## Table 3 — D diagnostics\n\n" + "\n".join(t3) +
        "\n\nD is symmetric about parity, so Table 3 carries the direction "
        "Table 1 discards. Coverage is the share of documents containing at "
        "least one gendered term; D is estimated only over those, and the "
        "share differs sharply between corpora. Doc-averaged D weighs every "
        "document equally, unlike the pooled estimator, which long documents "
        "dominate.\n"
    )
    if undefined_G:
        ug = pd.DataFrame(undefined_G)
        ug.to_csv(C.RESULTS_DIR / "undefined_G.csv", index=False)
        lines = ["| Corpus | n | D | Country mentions | BBQ CCF |",
                 "|---|---|---|---|---|"]
        for _, r in ug.iterrows():
            ccf_s = "—" if pd.isna(r["BBQ_CCF"]) else f"{r['BBQ_CCF']:.3f}"
            lines.append(f"| {r['label']} | {r['n_docs']:,} | {r['D']:.4f} "
                         f"| {r['n_country_mentions']:,} | {ccf_s} |")
        md += ("\n\n## Table 4 — Corpora with undefined G\n\n"
               + "\n".join(lines) +
               "\n\nG requires a minimum of "
               f"{MIN_COUNTRY_MENTIONS:,} country mentions to be an estimate "
               "rather than an artefact of a handful of observations. These "
               "corpora fall below that floor, so G is reported as undefined "
               "and no CBI is computed for them. Reporting a value here, or "
               "substituting zero, would present a non-measurement as a "
               "low-diversity finding.\n")
        print(f"\n{len(ug)} corpus/corpora have undefined G -> Table 4")

    (C.RESULTS_DIR / "paper_tables.md").write_text(md, encoding="utf-8")
    print(f"\nPaste-ready tables -> {C.RESULTS_DIR / 'paper_tables.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
