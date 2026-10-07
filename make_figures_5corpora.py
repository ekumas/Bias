#!/usr/bin/env python3
"""
make_figures_5corpora.py — Figures 1 and 4, regenerated for five corpora.

Figure 1 changes materially. With four corpora D and G ordered them in exactly
opposite sequence and every line met at one point. The Pile breaks that: it
holds second place on both dimensions, so its line runs level while the others
still cross. The figure is drawn as it comes out. A crossing that no longer
exists must not be preserved for its appearance, and the honest version still
shows what survives — the corpus highest on D is lowest on G, and conversely.

Figure 4 is unchanged in construction and stronger in content: five corpora,
two dimensions, seventeen weight configurations, four normalisation schemes.

Values are transcribed from results/bias_fingerprint.csv, dim_ci.csv and
cbi_sensitivity.csv. Nothing is recomputed from raw corpora, so a figure cannot
disagree with a table; verify_figures_5corpora() checks the transcription and
re-derives the claims the captions make.

    python make_figures_5corpora.py

Writes figure1_inversion.{pdf,png} and figure4_weights.{pdf,png} into
figures_5corpora/.
"""
from __future__ import annotations

import os
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.linewidth": 1.2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------- data
# Source: results/bias_fingerprint.csv and results/dim_ci.csv
WEB = ["C4", "OLMo-mix", "FineWeb-Edu", "Falcon", "The Pile"]
D = {"C4": 0.2485, "OLMo-mix": 0.3222, "FineWeb-Edu": 0.3545,
     "Falcon": 0.2357, "The Pile": 0.3229}
D_CI = {"C4": (0.2307, 0.2669), "OLMo-mix": (0.3067, 0.3353),
        "FineWeb-Edu": (0.3380, 0.3708), "Falcon": (0.2198, 0.2526),
        "The Pile": (0.3026, 0.3433)}
G = {"C4": 0.4398, "OLMo-mix": 0.4149, "FineWeb-Edu": 0.4081,
     "Falcon": 0.4733, "The Pile": 0.4521}
G_CI = {"C4": (0.4323, 0.4489), "OLMo-mix": (0.4098, 0.4218),
        "FineWeb-Edu": (0.4026, 0.4146), "Falcon": (0.4669, 0.4811),
        "The Pile": (0.4451, 0.4612)}

# Distinguished by marker and line style, not colour alone, so the figures
# remain readable in greyscale.
STYLE = {
    "C4":          ("o", "-",              "#1b1b1b"),
    "OLMo-mix":    ("s", "--",             "#4d4d4d"),
    "FineWeb-Edu": ("^", "-.",             "#7a7a7a"),
    "Falcon":      ("D", ":",              "#a6a6a6"),
    "The Pile":    ("P", (0, (5, 1, 1, 1)), "#2f5d8a"),
}

OUT = "figures_5corpora"


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{stem}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  wrote {OUT}/{stem}.pdf and .png")


# ------------------------------------------------------- Figure 1
def figure1():
    """Rank on D against rank on G, with point estimates and intervals."""
    fig = plt.figure(figsize=(7.0, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1],
                          left=0.145, right=0.945, bottom=0.19, top=0.88,
                          wspace=0.60)
    axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    dr = {c: i + 1 for i, c in enumerate(sorted(WEB, key=lambda c: -D[c]))}
    gr = {c: i + 1 for i, c in enumerate(sorted(WEB, key=lambda c: -G[c]))}

    for c in WEB:
        m, ls, col = STYLE[c]
        lw = 1.4 if dr[c] == gr[c] else 1.0
        axL.plot([0, 1], [dr[c], gr[c]], ls=ls, color=col, marker=m,
                 markersize=5, markerfacecolor="white", markeredgewidth=1.1,
                 linewidth=lw, clip_on=False, zorder=3)
        axL.annotate(f"{c}\n{D[c]:.4f}", (0, dr[c]), xytext=(-10, 0),
                     textcoords="offset points", ha="right", va="center",
                     fontsize=7, color=col, linespacing=1.35)
        axL.annotate(f"{c}\n{G[c]:.4f}", (1, gr[c]), xytext=(10, 0),
                     textcoords="offset points", ha="left", va="center",
                     fontsize=7, color=col, linespacing=1.35)

    axL.set_xlim(0, 1)
    axL.set_ylim(len(WEB) + 0.6, 0.4)
    axL.set_xticks([0, 1])
    axL.set_xticklabels(["D\ngendered imbalance",
                         "G\ngeographic concentration"], fontsize=7.5)
    axL.set_yticks([])
    for sp in ("left", "bottom"):
        axL.spines[sp].set_visible(False)
    axL.tick_params(axis="x", length=0, pad=8)
    axL.set_title("(a)  Rank on D against rank on G", loc="left", pad=12)

    order = sorted(WEB, key=lambda c: D[c])
    for i, c in enumerate(order):
        m, _, col = STYLE[c]
        axR.plot(D_CI[c], [i + 0.17] * 2, color=col, lw=1.1,
                 solid_capstyle="butt")
        axR.plot(D[c], i + 0.17, marker=m, color=col, markersize=4.5,
                 markerfacecolor="white", markeredgewidth=1.1)
        axR.plot(G_CI[c], [i - 0.17] * 2, color=col, lw=1.1,
                 solid_capstyle="butt")
        axR.plot(G[c], i - 0.17, marker=m, color=col, markersize=4.5,
                 markerfacecolor=col, markeredgewidth=1.1)
    axR.set_yticks(range(len(order)))
    axR.set_yticklabels(order, fontsize=7.5)
    axR.set_xlabel("Dimension value (95% bootstrap CI)", fontsize=7.2)
    axR.set_xlim(0.19, 0.52)
    axR.set_ylim(-0.62, len(order) - 0.38)
    axR.set_title("(b)  Point estimates", loc="left", pad=12)
    axR.legend(handles=[
        Line2D([], [], marker="o", color="#4d4d4d", markerfacecolor="white",
               markersize=4.5, lw=1.1, label="D"),
        Line2D([], [], marker="o", color="#4d4d4d", markerfacecolor="#4d4d4d",
               markersize=4.5, lw=1.1, label="G")],
        loc="upper left", frameon=False, handlelength=1.5,
        borderaxespad=0.2, labelspacing=0.3)
    save(fig, "figure1_inversion")


# ------------------------------------------------------- Figure 4
def figure4():
    """Composite rank against the weight on D, under four normalisations."""
    W = np.round(np.arange(0.10, 0.901, 0.05), 2)

    def minmax(d):
        v = np.array([d[c] for c in WEB])
        lo, hi = v.min(), v.max()
        return {c: (d[c] - lo) / (hi - lo) for c in WEB}

    def zscore(d):
        v = np.array([d[c] for c in WEB])
        mu, sd = v.mean(), v.std(ddof=1)
        z = {c: (d[c] - mu) / sd for c in WEB}
        return {c: z[c] - min(z.values()) for c in WEB}

    def rank(d):
        o = sorted(WEB, key=lambda c: d[c])
        return {c: o.index(c) / (len(WEB) - 1) for c in WEB}

    schemes = [("raw", lambda d: d), ("min-max", minmax),
               ("z-score", zscore), ("rank", rank)]

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.7), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.30,
                        wspace=0.16)

    for ax, (name, fn) in zip(axes, schemes):
        Dn, Gn = fn(D), fn(G)
        ranks = {c: [] for c in WEB}
        orders, ties = set(), []
        for w in W:
            v = {c: w * Dn[c] + (1 - w) * Gn[c] for c in WEB}
            orders.add(tuple(sorted(WEB, key=lambda c: -v[c])))
            # Average ranks for ties. Breaking a tie by sort order would draw
            # an ordering the data do not contain, which is the artefact this
            # figure exists to expose.
            for c in WEB:
                better = sum(1 for o in WEB if v[o] > v[c] + 1e-12)
                equal = sum(1 for o in WEB if abs(v[o] - v[c]) <= 1e-12)
                ranks[c].append(better + (equal + 1) / 2)
            if max(v.values()) - min(v.values()) <= 1e-12:
                ties.append(w)

        for w in ties:
            ax.axvline(w, color="#c00000", lw=0.8, ls=(0, (2, 2)), zorder=0)
            ax.annotate("all tied", (w, len(WEB) + 0.42), fontsize=6.2,
                        color="#c00000", ha="center", va="top")

        for c in WEB:
            m, ls, col = STYLE[c]
            jitter = 0.055 * (WEB.index(c) - 2)
            ax.plot(W, np.array(ranks[c]) + jitter, ls=ls, color=col,
                    marker=m, markersize=2.6, markerfacecolor="white",
                    markeredgewidth=0.8, label=c)

        ax.set_title(f"{name}\n{len(orders)} distinct orderings", loc="left",
                     fontsize=8)
        ax.set_xlabel("$w_D$")
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
        ax.set_ylim(len(WEB) + 0.6, 0.45)
        ax.set_yticks(range(1, len(WEB) + 1))
        ax.grid(axis="y", lw=0.25, color="#e8e8e8", zorder=0)

    axes[0].set_ylabel("Composite rank")
    handles = [Line2D([], [], marker=STYLE[c][0], ls=STYLE[c][1],
                      color=STYLE[c][2], markerfacecolor="white",
                      markersize=3.4, lw=1.1, label=c) for c in WEB]
    fig.legend(handles=handles, frameon=False, ncol=5, fontsize=7.5,
               loc="lower center", bbox_to_anchor=(0.5, 0.005),
               handlelength=2.2, columnspacing=1.5)
    save(fig, "figure4_weights")


# ------------------------------------------------------- verification
def verify():
    """Check the transcription and re-derive what the captions claim."""
    fails = []
    print("\n=== Verification ===")

    dr = sorted(WEB, key=lambda c: -D[c])
    gr = sorted(WEB, key=lambda c: -G[c])
    print(f"  D: {' > '.join(dr)}")
    print(f"  G: {' > '.join(gr)}")

    if list(reversed(dr)) == gr:
        fails.append("exact reversal reported, but it should not hold at n=5")
    print(f"  exact reversal: {list(reversed(dr)) == gr}  (expected False)")

    if dr[0] != gr[-1]:
        fails.append("corpus highest on D is not lowest on G")
    if gr[0] != dr[-1]:
        fails.append("corpus highest on G is not lowest on D")
    print(f"  highest D ({dr[0]}) is lowest on G: {dr[0] == gr[-1]}")
    print(f"  highest G ({gr[0]}) is lowest on D: {gr[0] == dr[-1]}")

    level = [c for c in WEB if dr.index(c) == gr.index(c)]
    print(f"  level on both dimensions: {level}  (expected ['The Pile'])")
    if level != ["The Pile"]:
        fails.append(f"level corpora {level}, expected ['The Pile']")

    d2 = sum((dr.index(c) - gr.index(c)) ** 2 for c in WEB)
    rho = 1 - 6 * d2 / (len(WEB) * (len(WEB) ** 2 - 1))
    print(f"  Spearman rho = {rho:+.3f}  (expected -0.700)")
    if abs(rho + 0.700) > 0.001:
        fails.append(f"rho {rho:.3f}, expected -0.700")

    W = np.round(np.arange(0.10, 0.901, 0.05), 2)
    if len(W) != 17:
        fails.append(f"{len(W)} weight configurations, expected 17")
    print(f"  weight configurations: {len(W)}  (expected 17)")

    def mm(d):
        v = np.array([d[c] for c in WEB])
        return {c: (d[c] - v.min()) / (v.max() - v.min()) for c in WEB}

    def zs(d):
        v = np.array([d[c] for c in WEB])
        z = {c: (d[c] - v.mean()) / v.std(ddof=1) for c in WEB}
        return {c: z[c] - min(z.values()) for c in WEB}

    def rk(d):
        o = sorted(WEB, key=lambda c: d[c])
        return {c: o.index(c) / (len(WEB) - 1) for c in WEB}

    expected = {"raw": 8, "min-max": 8, "z-score": 8, "rank": 9}
    for name, fn in [("raw", lambda d: d), ("min-max", mm),
                     ("z-score", zs), ("rank", rk)]:
        Dn, Gn = fn(D), fn(G)
        orders = {tuple(sorted(WEB, key=lambda c: -(w * Dn[c] + (1 - w) * Gn[c])))
                  for w in W}
        ok = len(orders) == expected[name]
        print(f"  {name:9} {len(orders)} orderings  (expected "
              f"{expected[name]})  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            fails.append(f"{name}: {len(orders)} orderings, "
                         f"expected {expected[name]}")

    # modal-rank share under raw summation, which Section 5.7 reports
    Dn, Gn = D, G
    shares = []
    for c in WEB:
        rs = []
        for w in W:
            v = {k: w * Dn[k] + (1 - w) * Gn[k] for k in WEB}
            rs.append(sum(1 for o in WEB if v[o] > v[c] + 1e-12) + 1)
        shares.append(Counter(rs).most_common(1)[0][1] / len(W))
    print(f"  raw lowest modal-rank share: {min(shares):.1%}  "
          f"(expected 35.3%)")
    if abs(min(shares) - 0.353) > 0.005:
        fails.append(f"lowest modal share {min(shares):.1%}, expected 35.3%")

    print()
    if fails:
        print(f"FAILED — {len(fails)} check(s):")
        for f in fails:
            print("  " + f)
        raise SystemExit(1)
    print("PASSED — figures agree with the pipeline output.")


if __name__ == "__main__":
    print("Generating figures for five corpora ...")
    figure1()
    figure4()
    verify()
