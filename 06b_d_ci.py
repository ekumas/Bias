#!/usr/bin/env python3
"""
06b_d_ci.py — bootstrap confidence intervals for D and G, before T exists.

Script 06 needs all three dimensions before it will compose anything. This one
runs on D only, so the D ranking can be tested as soon as script 02 finishes.

It answers the question a reviewer asks first: is corpus A really more
imbalanced than corpus B, or is the gap inside the noise?

For each corpus it reports 95% bootstrap CIs for both estimators, and for each
pair it reports P(D_A > D_B) across paired bootstrap replicates. Read that
probability as evidence strength, not as a p-value: values near 0.5 mean the
two corpora are indistinguishable on this measure; values near 1.0 mean the
ordering is stable under resampling.

Resampling is over DOCUMENTS, which is the unit that was sampled. It captures
sampling variability within the drawn sample. It does NOT capture the
head-of-stream limitation described in the README -- no bootstrap can, because
that is a bias, not a variance.

    python scripts/06b_d_ci.py
    python scripts/06b_d_ci.py --bootstrap 5000
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def d_pooled(f: np.ndarray, m: np.ndarray) -> float:
    tot = f.sum() + m.sum()
    return abs(f.sum() / tot - 0.5) * 2 if tot else np.nan


def d_docavg(f: np.ndarray, m: np.ndarray) -> float:
    tot = f + m
    mask = tot > 0
    return abs((f[mask] / tot[mask]).mean() - 0.5) * 2 if mask.any() else np.nan


def load_G(key: str):
    """Sparse doc x country matrix, or None if G was never computed."""
    p = C.RESULTS_DIR / f"perdoc_geographic_{key}.parquet"
    if not p.exists():
        return None
    from scipy.sparse import csr_matrix

    df = pd.read_parquet(p)
    per_doc = [Counter(x.split(";")) if x else Counter()
               for x in df.countries.fillna("")]
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
    return csr_matrix((vals, (rows, cols)), shape=(len(per_doc), len(order)))


def g_from_vector(v: np.ndarray) -> float:
    import math
    tot = v.sum()
    if tot <= 0:
        return np.nan
    p = v[v > 0] / tot
    h = float(-(p * np.log2(p)).sum())
    return min(max(1.0 - h / math.log2(C.N_REF_COUNTRIES), 0.0), 1.0)


def boot_G(m, B, rng, chunk=100):
    n = m.shape[0]
    out = np.empty(B)
    for s0 in range(0, B, chunk):
        k = min(chunk, B - s0)
        W = np.zeros((k, n), dtype=np.float32)
        for j in range(k):
            np.add.at(W[j], rng.integers(0, n, n), 1.0)
        agg = W @ m
        for j in range(k):
            out[s0 + j] = g_from_vector(np.asarray(agg[j]).ravel())
    return out


def ci(s: np.ndarray) -> tuple[float, float]:
    s = s[~np.isnan(s)]
    a = (1 - C.CI_LEVEL) / 2
    return float(np.quantile(s, a)), float(np.quantile(s, 1 - a))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=C.N_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    data, reps_p, reps_d, reps_g = {}, {}, {}, {}

    print(f"=== D bootstrap ({a.bootstrap} replicates, seed {a.seed}) ===\n")
    for key, spec in C.CORPORA.items():
        p = C.RESULTS_DIR / f"perdoc_demographic_{key}.parquet"
        if not p.exists():
            print(f"  [{key}] SKIP — run 02_demographic.py first.")
            continue
        df = pd.read_parquet(p)
        f = df.n_female.to_numpy(np.float64)
        m = df.n_male.to_numpy(np.float64)
        n = len(f)

        bp = np.empty(a.bootstrap)
        bd = np.empty(a.bootstrap)
        for b in range(a.bootstrap):
            i = rng.integers(0, n, n)
            bp[b] = d_pooled(f[i], m[i])
            bd[b] = d_docavg(f[i], m[i])

        data[key] = {
            "label": spec["label"], "n": n,
            "pooled": d_pooled(f, m), "docavg": d_docavg(f, m),
            "p_ci": ci(bp), "d_ci": ci(bd),
            "G": np.nan, "g_ci": (np.nan, np.nan), "n_country": 0,
        }
        reps_p[key], reps_d[key] = bp, bd

        gm = load_G(key)
        if gm is not None:
            n_country = int(gm.sum())
            data[key]["n_country"] = n_country
            if n_country >= 5_000:
                bg = boot_G(gm, a.bootstrap, rng)
                data[key]["G"] = g_from_vector(np.asarray(gm.sum(axis=0)).ravel())
                data[key]["g_ci"] = ci(bg)
                reps_g[key] = bg
        r = data[key]
        print(f"  [{key}] n={n:,}")
        print(f"    D_pooled = {r['pooled']:.4f}  "
              f"95% CI [{r['p_ci'][0]:.4f}, {r['p_ci'][1]:.4f}]  "
              f"width {r['p_ci'][1] - r['p_ci'][0]:.4f}")
        print(f"    D_docavg = {r['docavg']:.4f}  "
              f"95% CI [{r['d_ci'][0]:.4f}, {r['d_ci'][1]:.4f}]  "
              f"width {r['d_ci'][1] - r['d_ci'][0]:.4f}")
        if not np.isnan(r["G"]):
            print(f"    G        = {r['G']:.4f}  "
                  f"95% CI [{r['g_ci'][0]:.4f}, {r['g_ci'][1]:.4f}]  "
                  f"width {r['g_ci'][1] - r['g_ci'][0]:.4f}  "
                  f"({r['n_country']:,} country mentions)")
        elif r["n_country"]:
            print(f"    G        = NOT REPORTED — only {r['n_country']:,} "
                  f"country mentions. Report G as undefined for this corpus "
                  f"rather than as a low-diversity finding.")

    if len(data) < 2:
        print("\nNeed at least two corpora.")
        return 1

    rows = []
    print("\n=== Pairwise: P(row > column), paired bootstrap ===\n")
    for est, reps in (("pooled", reps_p), ("docavg", reps_d), ("G", reps_g)):
        if len(reps) < 2:
            continue
        print(f"-- {est} --")
        for x, y in combinations([k for k in data if k in reps], 2):
            pr = float((reps[x] > reps[y]).mean())
            gap = data[x][est] - data[y][est]
            if 0.10 <= pr <= 0.90:
                verdict = "INDISTINGUISHABLE"
            elif 0.025 < pr < 0.975:
                verdict = "weak"
            else:
                verdict = "separated"
            rows.append({"estimator": est, "a": x, "b": y,
                         "gap": round(gap, 4), "P_a_gt_b": round(pr, 4),
                         "verdict": verdict})
            print(f"  {x:<16} vs {y:<16} gap={gap:+.4f}  "
                  f"P={pr:.3f}  {verdict}")
        print()

    pd.DataFrame([{
        "corpus": k, "label": v["label"], "n_docs": v["n"],
        "D_pooled": round(v["pooled"], 4),
        "D_pooled_lo": round(v["p_ci"][0], 4), "D_pooled_hi": round(v["p_ci"][1], 4),
        "D_docavg": round(v["docavg"], 4),
        "D_docavg_lo": round(v["d_ci"][0], 4), "D_docavg_hi": round(v["d_ci"][1], 4),
        "G": None if np.isnan(v["G"]) else round(v["G"], 4),
        "G_lo": None if np.isnan(v["g_ci"][0]) else round(v["g_ci"][0], 4),
        "G_hi": None if np.isnan(v["g_ci"][1]) else round(v["g_ci"][1], 4),
        "n_country_mentions": v["n_country"],
    } for k, v in data.items()]).to_csv(C.RESULTS_DIR / "dim_ci.csv", index=False)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "d_pairwise.csv", index=False)

    for est, reps in (("pooled", reps_p), ("docavg", reps_d), ("G", reps_g)):
        ks = [k for k in data if k in reps]
        if len(ks) < 2:
            continue
        order = sorted(ks, key=lambda k: -data[k][est])
        print(f"Ranking ({est}): " + " > ".join(data[k]["label"] for k in order))
    print("\nAny pair marked INDISTINGUISHABLE must not be ordered in the paper.")
    print(f"Written -> {C.RESULTS_DIR / 'dim_ci.csv'}, "
          f"{C.RESULTS_DIR / 'd_pairwise.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
