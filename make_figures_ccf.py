#!/usr/bin/env python3
"""
make_figures_ccf.py — Figures 2 and 3, read from the pipeline.

Figure 2, both panels, comes from results/ccf_vocab_sensitivity.csv. That file
holds the full threshold sweep for each of four vocabulary constructions, and
its `cleaned` rows are the main CCF measurement: same 95 terms, same values as
results/bbq_ccf.csv where the two overlap.

Figure 3 comes from results/bbq_category_coverage.csv.

A CAVEAT ABOUT bbq_category_coverage.csv
----------------------------------------
Running script 05 for a single corpus overwrites that file rather than updating
it, so after the run for The Pile it holds The Pile alone. The other five
corpora's category medians are not recoverable from any other output: no file
carries the per-category breakdown.

Until script 05 has been run across all corpora, this script falls back to
coverage_manuscript.csv, a transcription of Table 8 from the manuscript, for
the columns the pipeline file is missing. Every value it takes from that
fallback is listed on stdout, and The Pile — present in both — is cross-checked
between them. The fallback is a stopgap for drafting, not a substitute for the
measurement: re-run script 05 across all corpora before release, after which
this script will read everything from the pipeline and the fallback can be
deleted.

    python make_figures_ccf.py [--results DIR]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "lines.linewidth": 1.2, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

SHORT = {"c4_cc_proxy": "C4", "olmo_mix": "OLMo-mix", "fineweb_edu": "FineWeb-Edu",
         "falcon_refined": "Falcon", "pile_uncopyrighted": "The Pile",
         "tinystories": "TinyStories"}
ORDER = ["C4", "OLMo-mix", "FineWeb-Edu", "Falcon", "The Pile", "TinyStories"]
# Distinguished by marker and line style as well as colour, so the figures
# survive greyscale printing.
STYLE = {
    "C4":          ("o", "-",               "#1b1b1b"),
    "OLMo-mix":    ("s", "--",              "#4d4d4d"),
    "FineWeb-Edu": ("^", "-.",              "#7a7a7a"),
    "Falcon":      ("D", ":",               "#a6a6a6"),
    "The Pile":    ("P", (0, (5, 1, 1, 1)), "#2f5d8a"),
    "TinyStories": ("v", (0, (1, 1)),       "#b03030"),
}
THRESH = [1, 2, 5, 10, 20, 50]
VOCABS = ["strict", "cleaned", "expanded", "relaxed"]
OUT = "figures_ccf"


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{stem}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  wrote {OUT}/{stem}.pdf and .png")


def load_ccf(results: Path):
    p = results / "ccf_vocab_sensitivity.csv"
    if not p.exists():
        sys.exit(f"missing {p}\nRun scripts/05b_vocab_sensitivity.py first.")
    v = pd.read_csv(p)
    v["name"] = v.corpus.map(SHORT)
    names = [c for c in ORDER if c in set(v.name)]
    sweep = {c: [float(v[(v.name == c) & (v.vocabulary == "cleaned")][f"CCF@{t}"].iloc[0])
                 for t in THRESH] for c in names}
    vocab = {c: [float(v[(v.name == c) & (v.vocabulary == k)]["CCF@5"].iloc[0])
                 for k in VOCABS] for c in names}
    print(f"  Figure 2: {len(names)} corpora x {len(VOCABS)} vocabularies "
          f"from {p.name}")
    # The cleaned vocabulary is the main CCF; confirm against bbq_ccf.csv
    # wherever that file still has a matching row.
    q = results / "bbq_ccf.csv"
    if q.exists():
        m = pd.read_csv(q)
        m["name"] = m.corpus.map(SHORT)
        for _, r in m.iterrows():
            if r["name"] in sweep:
                a, b = float(r["CCF@5"]), sweep[r["name"]][2]
                mark = "OK" if abs(a - b) < 1e-9 else "MISMATCH"
                print(f"    cross-check {r['name']}: bbq_ccf {a} vs cleaned "
                      f"{b}  {mark}")
                if mark == "MISMATCH":
                    sys.exit("cleaned vocabulary does not match bbq_ccf.csv")
    return names, sweep, vocab


def load_coverage(results: Path, fallback: Path):
    p = results / "bbq_category_coverage.csv"
    pipe = pd.read_csv(p).set_index("cat") if p.exists() else pd.DataFrame()
    pipe.columns = [SHORT.get(c, c) for c in pipe.columns]

    missing = [c for c in ORDER if c not in pipe.columns]
    if missing:
        if not fallback.exists():
            sys.exit(f"{p.name} holds only {list(pipe.columns)}.\n"
                     f"Run scripts/05_bbq_ccf.py across all corpora, or supply "
                     f"{fallback}.")
        fb = pd.read_csv(fallback).set_index(pd.read_csv(fallback).columns[0])
        print(f"  Figure 3: {p.name} holds only {list(pipe.columns)}")
        print(f"            taking {missing} from {fallback.name} "
              f"(manuscript Table 8)")
        for c in pipe.columns:
            if c in fb.columns:
                d = (pipe[c].dropna() - fb[c].dropna()).abs().max()
                print(f"            cross-check {c}: max difference {d:.4f}  "
                      f"{'OK' if d < 1e-9 else 'MISMATCH'}")
        cov = fb.copy()
        for c in pipe.columns:
            cov[c] = pipe[c]
    else:
        cov = pipe
        print(f"  Figure 3: all corpora from {p.name}")

    web = [c for c in ORDER if c != "TinyStories" and c in cov.columns]
    cov = cov.loc[[i for i in cov.index if not i.startswith("Race_x")], web]
    return cov


def figure2(names, sweep, vocab):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.1))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.30,
                        wspace=0.30)
    for c in names:
        m, ls, col = STYLE[c]
        a1.plot(THRESH, sweep[c], ls=ls, color=col, marker=m, markersize=4,
                markerfacecolor="white", markeredgewidth=1.0)
        a2.plot(range(len(VOCABS)), vocab[c], ls=ls, color=col, marker=m,
                markersize=4, markerfacecolor="white", markeredgewidth=1.0)
    a1.set_xscale("log")
    a1.set_xticks(THRESH)
    a1.set_xticklabels([str(t) for t in THRESH])
    a1.set_xlabel("Coverage threshold θ (mentions per 10,000 documents)")
    a1.set_ylabel("Category Coverage Failure")
    a1.set_ylim(0, 1)
    a1.set_title("(a)  Threshold sweep", loc="left")
    a2.set_xticks(range(len(VOCABS)))
    a2.set_xticklabels(VOCABS)
    a2.set_xlabel("Vocabulary construction")
    a2.set_ylabel("CCF at θ = 5")
    a2.set_ylim(0, 1)
    a2.set_title("(b)  Vocabulary sensitivity", loc="left")
    for ax in (a1, a2):
        ax.grid(axis="y", lw=0.25, color="#ececec", zorder=0)
    handles = [Line2D([], [], marker=STYLE[c][0], ls=STYLE[c][1],
                      color=STYLE[c][2], markerfacecolor="white",
                      markersize=3.6, lw=1.1, label=c) for c in names]
    fig.legend(handles=handles, frameon=False, ncol=len(names), fontsize=7.5,
               loc="lower center", bbox_to_anchor=(0.5, 0.005),
               handlelength=2.2, columnspacing=1.4)
    save(fig, "figure2_ccf")


def figure3(cov):
    order = cov.median(axis=1).sort_values().index.tolist()
    web = list(cov.columns)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    fig.subplots_adjust(left=0.235, right=0.975, top=0.88, bottom=0.16)
    y = np.arange(len(order))
    for i, c in enumerate(web):
        m, _, col = STYLE[c]
        ax.plot(cov.loc[order, c], y, ls="none", marker=m, color=col,
                markerfacecolor="white", markersize=5, markeredgewidth=1.1,
                label=c, zorder=3)
    for j, cat in enumerate(order):
        lo, hi = cov.loc[cat].min(), cov.loc[cat].max()
        ax.plot([lo, hi], [j, j], color="#d2d2d2", lw=1.6, zorder=1,
                solid_capstyle="round")
    ax.set_xscale("log")
    ax.set_xlim(0.3, 600)
    ax.set_xticks([0.3, 1, 3, 10, 30, 100, 300])
    ax.set_xticklabels(["0.3", "1", "3", "10", "30", "100", "300"])
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in order], fontsize=7.5)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Median mentions per 10,000 documents (log scale)",
                  fontsize=7.5)
    ax.grid(axis="x", lw=0.25, color="#ececec", zorder=0)
    ax.legend(frameon=False, ncol=len(web), fontsize=7.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), handletextpad=0.4,
              columnspacing=1.4)
    save(fig, "figure3_bbq_coverage")

    d = cov.loc["Disability_status"]
    g = cov.loc["Gender_identity"]
    print(f"\n  Disability lowest in every corpus: "
          f"{all(cov[c].idxmin() == 'Disability_status' for c in web)}")
    print(f"  Gender/Disability ratio: {(g / d).min():.0f}x to {(g / d).max():.0f}x")
    nxt = cov.drop(index="Disability_status").min()
    print(f"  gap to the next sparsest category: "
          f"{(nxt / d).min():.1f}x to {(nxt / d).max():.1f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", type=Path)
    ap.add_argument("--fallback", default=Path("coverage_manuscript.csv"),
                    type=Path)
    a = ap.parse_args()
    print("Generating Figures 2 and 3 ...")
    names, sweep, vocab = load_ccf(a.results)
    figure2(names, sweep, vocab)
    cov = load_coverage(a.results, a.fallback)
    figure3(cov)
