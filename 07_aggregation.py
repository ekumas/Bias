#!/usr/bin/env python3
"""
Script 07 — aggregation robustness under alternative normalisation schemes.

WHY THIS EXISTS
---------------
The claim "the composite index separates corpora far less than its best
constituent does" is vulnerable to an obvious objection: the dimensions are on
different scales. T spans 0.018 across our corpora while D spans 0.119, so a
weighted sum is dominated by D and G whatever the weights say. A referee will
argue that the composite's narrow range is an artefact of not normalising, and
that standard practice — min-max, z-score, or rank transformation before
aggregation — would fix it.

This script answers that objection with measurement rather than assertion. It
recomputes the composite under four schemes and asks, for each, two separate
questions that the original analysis conflated:

  (Q1) SEPARATION -- how wide is the range of composite scores across corpora?
  (Q2) RANK STABILITY -- does the ordering of corpora survive a change of
       weights?

These can come apart, and in our data they do. Normalisation improves Q1
substantially; it does not rescue Q2. That distinction is the honest result and
the paper should be built on Q2, which is robust, rather than Q1, which is not.

Outputs
    results/aggregation_robustness.csv   per scheme x corpus
    results/aggregation_summary.csv      per scheme
    results/aggregation_tables.md        paste-ready

    python scripts/07_aggregation.py
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

DIMS = ["D", "G"]


# ------------------------------------------------------------ normalisation
def norm_raw(df):
    """No transformation. Dimensions keep their native scales."""
    return df[DIMS].copy()


def norm_minmax(df):
    """Each dimension rescaled to [0, 1] across the corpora in this study.

    Note the dependency this introduces: the normalised value of a corpus
    depends on which other corpora are in the sample. Adding a corpus changes
    every other corpus's score. This is a real cost of min-max and should be
    stated, not hidden.
    """
    out = df[DIMS].copy()
    for d in DIMS:
        lo, hi = out[d].min(), out[d].max()
        out[d] = 0.5 if hi == lo else (out[d] - lo) / (hi - lo)
    return out


def norm_zscore(df):
    """Standardised to mean 0, sd 1, then shifted to be non-negative.

    Same sample-dependency as min-max, plus sensitivity to outliers, which
    matters at n = 4.
    """
    out = df[DIMS].copy()
    for d in DIMS:
        mu, sd = out[d].mean(), out[d].std(ddof=1)
        out[d] = 0.0 if sd == 0 else (out[d] - mu) / sd
    return out - out.min().min()


def norm_rank(df):
    """Rank transform, scaled to [0, 1]. Discards magnitudes entirely.

    Included because it is the most aggressive answer to the scale objection:
    if the composite still fails here, the failure cannot be about scale.
    """
    out = df[DIMS].copy()
    n = len(out)
    for d in DIMS:
        out[d] = (out[d].rank(method="average") - 1) / (n - 1) if n > 1 else 0.5
    return out


SCHEMES = {
    "raw": norm_raw,
    "minmax": norm_minmax,
    "zscore": norm_zscore,
    "rank": norm_rank,
}


# Aggregation rules. The weighted sum is COMPENSATORY: a high score on one
# dimension offsets a low score on another, which is exactly why inversely
# ordered dimensions cancel. Non-compensatory rules do not have that property
# and a referee will ask whether they were tried, so they are.
#
#   max   the worst dimension governs; no offsetting at all
#   min   the best dimension governs
#   geom  geometric mean; partially compensatory, penalises imbalance
#
# The distinction matters for what can be claimed. "Composite indices fail" is
# too broad. "Compensatory aggregation cancels under inverse ordering, while
# non-compensatory rules are stable but yield a partial order rather than a
# ranking" is the defensible version.
RULES = {
    "weighted sum": lambda d, g, w: w[0] * d + w[1] * g,
    "geometric mean": lambda d, g, w: max(d, 1e-9) ** w[0] * max(g, 1e-9) ** w[1],
    "max (worst dimension)": lambda d, g, w: max(d, g),
    "min (best dimension)": lambda d, g, w: min(d, g),
}


def pareto_relations(X, labels):
    """Count strict dominance: does any corpus beat another on every dimension?

    Zero dominance relations means every corpus is Pareto-optimal, and any
    total ordering must therefore impose a trade-off rate that the data do not
    supply. That is the precise sense in which the ranking is normative rather
    than empirical.
    """
    import itertools

    dom = []
    for i, j in itertools.permutations(range(len(X)), 2):
        a, b = X[i], X[j]
        if all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b)):
            dom.append((labels[i], labels[j]))
    return dom


def weight_configs(grid=None):
    """w_D from 0.10 to 0.90 in steps of 0.05; w_G is the complement."""
    return [(round(w, 2), round(1 - w, 2)) for w in np.arange(0.10, 0.901, 0.05)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="corpus keys to exclude (e.g. --exclude tinystories)")
    a = ap.parse_args()

    src = C.RESULTS_DIR / "bias_fingerprint.csv"
    if not src.exists():
        print(f"{src} not found. Run 06_cbi.py first.")
        return 1
    fp = pd.read_csv(src)
    if a.exclude:
        fp = fp[~fp.corpus.isin(a.exclude)].reset_index(drop=True)
    fp = fp.dropna(subset=DIMS).reset_index(drop=True)
    if len(fp) < 3:
        print(f"Only {len(fp)} corpora with all three dimensions defined. "
              f"Need at least 3.")
        return 1

    labels = fp.label.tolist()
    cfgs = weight_configs()
    print(f"{len(fp)} corpora, {len(cfgs)} weight configurations\n")

    per_corpus, per_scheme = [], []

    for sname, fn in SCHEMES.items():
        X = fn(fp)
        vals = X.to_numpy(float)                       # (n_corpora, 2)
        W = np.array(cfgs, float)                      # (n_configs, 2)
        S = vals @ W.T                                 # (n_corpora, n_configs)

        equal = vals.mean(axis=1)
        spread_eq = float(equal.max() - equal.min())

        # Q1: separation, averaged over weightings
        spreads = S.max(axis=0) - S.min(axis=0)
        # Q2: rank stability
        order_idx = np.argsort(-S, axis=0)             # rank 1 first
        orderings = {tuple(col) for col in order_idx.T}
        ranks = np.empty_like(order_idx)
        for j in range(order_idx.shape[1]):
            ranks[order_idx[:, j], j] = np.arange(1, len(fp) + 1)

        print(f"--- {sname} ---")
        print(f"  equal-weight spread = {spread_eq:.4f}   "
              f"mean spread over weightings = {spreads.mean():.4f}")
        print(f"  distinct full orderings = {len(orderings)} of "
              f"{len(cfgs)} weightings")

        modal_shares = []
        for i, lab in enumerate(labels):
            r = ranks[i]
            cnt = Counter(r.tolist())
            modal, n_modal = cnt.most_common(1)[0]
            share = n_modal / len(cfgs)
            modal_shares.append(share)
            # how much does the weighting move THIS corpus, vs how far apart
            # are the corpora at fixed weights?
            per_corpus.append({
                "scheme": sname, "corpus": fp.corpus[i], "label": lab,
                "score_equal_weights": round(float(equal[i]), 4),
                "score_min": round(float(S[i].min()), 4),
                "score_max": round(float(S[i].max()), 4),
                "weight_induced_range": round(float(S[i].max() - S[i].min()), 4),
                "rank_min": int(r.min()), "rank_max": int(r.max()),
                "modal_rank": int(modal), "modal_rank_share": round(share, 4),
            })
            print(f"    {lab:22} rank {r.min()}-{r.max()}  "
                  f"modal {modal} in {share:6.1%}")

        mean_wr = float(np.mean([S[i].max() - S[i].min() for i in range(len(fp))]))
        per_scheme.append({
            "scheme": sname,
            "n_corpora": len(fp),
            "spread_equal_weights": round(spread_eq, 4),
            "mean_spread_over_weightings": round(float(spreads.mean()), 4),
            "mean_weight_induced_range": round(mean_wr, 4),
            "weight_vs_corpus_ratio": round(mean_wr / spread_eq, 2) if spread_eq else None,
            "distinct_orderings": len(orderings),
            "min_modal_rank_share": round(float(min(modal_shares)), 4),
            "mean_modal_rank_share": round(float(np.mean(modal_shares)), 4),
            "all_ranks_stable": bool(len(orderings) == 1),
        })
        print(f"  weight-induced range / corpus spread = "
              f"{mean_wr / spread_eq:.2f}x\n" if spread_eq else "\n")

    # ---- non-compensatory rules, on the min-max normalised dimensions ----
    print("\n" + "=" * 68)
    print("Aggregation rule comparison (min-max normalised, equal weights)")
    print("=" * 68)
    Xn = norm_minmax(fp)
    vals = Xn.to_numpy(float)
    W = np.array(cfgs, float)
    eqw = np.full(2, 0.5)
    rule_rows = []
    for rname, rfn in RULES.items():
        eq = np.array([rfn(*row, eqw) for row in vals])
        orders = {tuple(np.argsort(-np.array([rfn(*row, w) for row in vals])))
                  for w in W}
        order_eq = " > ".join(labels[i] for i in np.argsort(-eq))
        rule_rows.append({"rule": rname,
                          "distinct_orderings": len(orders),
                          "spread_equal_weights": round(float(eq.max() - eq.min()), 4),
                          "ordering_at_equal_weights": order_eq})
        print(f"  {rname:24} {len(orders):2} distinct ordering(s) over "
              f"{len(cfgs)} weightings")
    pd.DataFrame(rule_rows).to_csv(C.RESULTS_DIR / "aggregation_rules.csv",
                                   index=False)

    dom = pareto_relations(vals.tolist(), labels)
    print(f"\n  Pareto dominance relations: {len(dom)}")
    if not dom:
        print("  Every corpus is Pareto-optimal: none is at least as high as")
        print("  another on all dimensions. Any total ordering therefore rests")
        print("  on a trade-off rate the data do not supply.")
    else:
        for a, b in dom:
            print(f"    {a} dominates {b}")
    print("\n  Non-compensatory rules are stable BECAUSE they discard the")
    print("  conflict rather than resolve it: max reports only the worst")
    print("  dimension. Stability here is not agreement between dimensions.")

    pc = pd.DataFrame(per_corpus)
    ps = pd.DataFrame(per_scheme)
    pc.to_csv(C.RESULTS_DIR / "aggregation_robustness.csv", index=False)
    ps.to_csv(C.RESULTS_DIR / "aggregation_summary.csv", index=False)

    print("=" * 68)
    print(ps.to_string(index=False))
    print("=" * 68)

    # ---- the interpretation the paper depends on ----
    raw = ps[ps.scheme == "raw"].iloc[0]
    best = ps.loc[ps.spread_equal_weights.idxmax()]
    any_stable = ps.all_ranks_stable.any()
    print("\nREADING THIS TABLE")
    print(f"  Q1 separation: raw spread {raw.spread_equal_weights:.4f}; best "
          f"scheme ({best.scheme}) reaches {best.spread_equal_weights:.4f}.")
    if best.spread_equal_weights > 2 * raw.spread_equal_weights:
        print("     -> Normalisation MATERIALLY improves separation. A claim that")
        print("        'the composite cannot separate the corpora' does not")
        print("        survive normalisation and must not be the paper's headline.")
    else:
        print("     -> Normalisation does not rescue separation; the narrow range")
        print("        is not a scale artefact.")
    print(f"  Q2 rank stability: {'no' if not any_stable else 'at least one'} "
          f"scheme produces a single ordering across all weightings.")
    if not any_stable:
        worst = ps.loc[ps.min_modal_rank_share.idxmin()]
        print(f"     -> Under every scheme the ordering depends on the weights.")
        print(f"        Worst case: {worst.scheme}, where some corpus holds its")
        print(f"        modal rank in only {worst.min_modal_rank_share:.1%} of")
        print(f"        weightings. THIS is the robust finding — build on it.")

    # ---- markdown ----
    t1 = ["| Scheme | Spread (equal weights) | Mean weight-induced range | Ratio | Distinct orderings | Min modal-rank share |",
          "|---|---|---|---|---|---|"]
    for _, r in ps.iterrows():
        t1.append(f"| {r['scheme']} | {r['spread_equal_weights']:.4f} "
                  f"| {r['mean_weight_induced_range']:.4f} "
                  f"| {r['weight_vs_corpus_ratio']}× | {r['distinct_orderings']} "
                  f"| {r['min_modal_rank_share']:.1%} |")

    t2 = ["| Scheme | Corpus | Score (equal w) | Rank range | Modal rank | Share |",
          "|---|---|---|---|---|---|"]
    for _, r in pc.iterrows():
        t2.append(f"| {r['scheme']} | {r['label']} | {r['score_equal_weights']:.3f} "
                  f"| {r['rank_min']}–{r['rank_max']} | {r['modal_rank']} "
                  f"| {r['modal_rank_share']:.1%} |")

    md = ("# Aggregation robustness — auto-generated\n\n"
          f"{len(fp)} corpora, {len(cfgs)} weight configurations, "
          f"grid {C.WEIGHT_GRID}\n\n"
          "## Separation and rank stability by normalisation scheme\n\n"
          + "\n".join(t1) +
          "\n\n\"Ratio\" is the mean range induced in a single corpus's score by "
          "varying the weights, divided by the range of scores across corpora at "
          "equal weights. Above 1 means the analyst's weighting choice moves the "
          "number more than the identity of the corpus does.\n\n"
          "## Per-corpus detail\n\n" + "\n".join(t2) + "\n")
    (C.RESULTS_DIR / "aggregation_tables.md").write_text(md, encoding="utf-8")
    print(f"\nTables -> {C.RESULTS_DIR / 'aggregation_tables.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
