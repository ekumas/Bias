#!/usr/bin/env python3
"""
Script 03d — inter-annotator agreement and the corrected false-positive rate.

Reads the two completed annotation files, reports raw agreement and Cohen's
kappa overall and by stratum, lists every disagreement for adjudication, and
recomputes the false-positive rate and corrected T from the adjudicated labels.

READING KAPPA HONESTLY
----------------------
Kappa corrects raw agreement for agreement expected by chance, and that
correction depends on the marginal distribution. When one class dominates,
expected chance agreement is high and kappa is low even when the annotators
agree on almost everything -- the well-known first kappa paradox. This is why
the annotation set was stratified: the >=0.90 stratum supplies positive cases
and keeps the margins from collapsing.

The script therefore reports, alongside kappa, the quantities needed to
interpret it: raw agreement, the observed prevalence, expected chance
agreement, and prevalence-adjusted bias-adjusted kappa (PABAK). Reporting kappa
alone from a skewed set invites a reviewer to read a low value as unreliable
annotation when it reflects the class balance instead.

    python scripts/03d_agreement.py
    python scripts/03d_agreement.py --strict     # borderline counts as toxic
"""
from __future__ import annotations

import argparse
import math
import sys
from itertools import product
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

AUDIT_CORPORA = ["tinystories", "c4_cc_proxy"]
VALID = {"toxic", "not_toxic", "borderline"}

# Stratum sizes in the 50,000-document sample, from
# results/toxicity_distribution.csv. Needed to scale a stratum rate to the
# corpus: the <0.50 stratum is ~99.7% of it, so a rate estimated there
# dominates any corpus-level total no matter how small the sample.
STRATUM_POP = {
    "c4_cc_proxy":  {"<0.50": 49_856, "0.50-0.90": 129, ">=0.90": 15},
    "tinystories":  {"<0.50": 49_579, "0.50-0.90": 384, ">=0.90": 37},
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - s), min(1.0, c + s)


def cohen_kappa(a: list[str], b: list[str]) -> dict:
    labels = sorted(set(a) | set(b))
    n = len(a)
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = pd.Series(a).value_counts(), pd.Series(b).value_counts()
    exp = sum((ca.get(l, 0) / n) * (cb.get(l, 0) / n) for l in labels)
    k = (obs - exp) / (1 - exp) if exp < 1 else float("nan")
    # PABAK removes the dependence on prevalence and bias. It is defined for
    # two categories as 2*p_o - 1, so it is computed on the binarised labels
    # (borderline folded into not-toxic) and reported alongside the
    # three-category kappa rather than in place of it.
    ab = [x == "toxic" for x in a]
    bb = [y == "toxic" for y in b]
    obs_bin = sum(1 for x, y in zip(ab, bb) if x == y) / n
    pabak = 2 * obs_bin - 1
    return {"n": n, "raw_agreement": obs, "expected_chance": exp,
            "kappa": k, "pabak": pabak, "labels": labels}


def interpret(k: float) -> str:
    if math.isnan(k):
        return "undefined"
    for lo, txt in ((0.81, "almost perfect"), (0.61, "substantial"),
                    (0.41, "moderate"), (0.21, "fair"), (0.0, "slight")):
        if k >= lo:
            return txt
    return "poor"


def read_table_any(path: Path) -> pd.DataFrame:
    """Read an annotation table saved in whatever format the annotator used.

    Excel is accepted alongside CSV because asking an annotator to remember an
    export step is a reliable way to lose an hour of their work. For CSV both
    the encoding and the delimiter are sniffed rather than assumed: files
    round-tripped through Numbers or Excel commonly use Latin-1 or
    Windows-1252, and in some locales a semicolon.
    """
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path)
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


def find_annotation_file(key: str, who: str) -> Path | None:
    """Locate the annotation file for one annotator, in any accepted format."""
    d = C.RESULTS_DIR / "annotation"
    stem = f"annotator_{who}_{key}"
    for ext in (".csv", ".xlsx", ".xlsm", ".xls", ".tsv", ".txt"):
        p = d / (stem + ext)
        if p.exists():
            return p
    # Spreadsheet exports frequently append a suffix of their own.
    hits = sorted(q for q in d.glob(f"{stem}*")
                  if q.suffix.lower() in (".csv", ".xlsx", ".xlsm", ".xls",
                                          ".tsv", ".txt"))
    return hits[0] if hits else None


def load(key: str, who: str) -> pd.DataFrame | None:
    p = find_annotation_file(key, who)
    if p is None:
        print(f"  [{key}] no annotator_{who}_{key} file found "
              f"(.csv, .xlsx, .tsv all accepted)")
        return None
    if p.name != f"annotator_{who}_{key}.csv":
        print(f"  [{key}/{who}] using {p.name}")
    try:
        df = read_table_any(p)
    except RuntimeError as exc:
        print(f"  [{key}/{who}] {exc}")
        print(f"  [{key}/{who}] re-export it from the spreadsheet as CSV, "
              f"keeping the 'item' and 'label' columns")
        return None
    if "item" not in df.columns or "label" not in df.columns:
        print(f"  [{key}/{who}] {p.name} has columns {list(df.columns)}; "
              f"'item' and 'label' are both required")
        return None
    df["label"] = df.label.astype(str).str.strip().str.lower()
    blank = int((~df.label.isin(VALID)).sum())
    if blank:
        print(f"  [{key}/{who}] {blank} of {len(df)} rows unlabelled or "
              f"invalid — complete them before running this")
        return None
    return df[["item", "label"]].rename(columns={"label": who})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="count borderline as toxic (default: as not toxic)")
    a = ap.parse_args()

    # Adjudicated labels, if the disagreements file has been completed. Until
    # then every rate is computed on agreed items only, so that no number in
    # the paper depends on a decision that has not been made.
    # Adjudicated labels. Two sources are accepted, in this order:
    #
    #   adjudications.txt   one "item label" pair per line
    #   disagreements.csv   the 'adjudicated' column
    #
    # The plain-text file exists because disagreements.csv embeds document
    # excerpts, and excerpts contain commas, quotation marks and newlines that
    # spreadsheet round-trips routinely mangle beyond what any CSV parser can
    # recover. A two-column text file has no such failure mode, and the
    # decisions are too consequential to lose to a delimiter.
    adj_dir = C.RESULTS_DIR / "annotation"
    adj: dict[str, str] = {}

    # Adjudicated labels. Any of three sources is accepted, in this order:
    #
    #   adjudications.xlsx   columns 'item' and 'label' (or 'adjudicated')
    #   adjudications.txt    one "item label" pair per line
    #   disagreements.csv    the 'adjudicated' column
    #
    # The first two exist because disagreements.csv embeds document excerpts,
    # and excerpts contain commas, quotation marks and newlines that
    # spreadsheet round-trips routinely mangle beyond what any CSV parser can
    # recover. A two-column sheet carries no excerpt and so has no such
    # failure mode; these decisions are too consequential to lose to a
    # delimiter.
    def _harvest(df, source: str) -> int:
        cols = {str(c).strip().lower(): c for c in df.columns}
        item_col = cols.get("item") or cols.get("doc_id")
        lab_col = (cols.get("label") or cols.get("adjudicated")
                   or cols.get("final") or cols.get("decision"))
        if not item_col or not lab_col:
            print(f"{source}: needs an 'item' column and a 'label' "
                  f"(or 'adjudicated') column; found {list(df.columns)}")
            return 0
        n = 0
        for _, r in df.iterrows():
            v = str(r[lab_col]).strip().lower()
            if v in VALID:
                adj[str(r[item_col]).strip()] = v
                n += 1
        return n

    xlsx_path = next((q for q in (adj_dir / f"adjudications{e}"
                                  for e in (".xlsx", ".xlsm", ".xls"))
                      if q.exists()), None)
    if xlsx_path is not None:
        try:
            n = _harvest(pd.read_excel(xlsx_path), xlsx_path.name)
            if n:
                print(f"Adjudications read from {xlsx_path.name}: {n} item(s).")
        except Exception as exc:
            print(f"{xlsx_path.name} could not be read ({str(exc)[:60]}).")

    if not adj:
        txt_path = adj_dir / "adjudications.txt"
        if txt_path.exists():
            for raw in txt_path.read_text(encoding="utf-8",
                                          errors="replace").splitlines():
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.replace(",", " ").replace("\t", " ").split()
                if len(parts) >= 2 and parts[-1].lower() in VALID:
                    adj[parts[0].strip()] = parts[-1].lower()
            if adj:
                print(f"Adjudications read from {txt_path.name}: "
                      f"{len(adj)} item(s).")

    if not adj:
        csv_path = adj_dir / "disagreements.csv"
        if csv_path.exists():
            try:
                n = _harvest(read_table_any(csv_path), csv_path.name)
                if n:
                    print(f"Adjudications read from {csv_path.name}: "
                          f"{n} item(s).")
            except Exception as exc:
                print(f"{csv_path.name} could not be parsed "
                      f"({str(exc)[:55]}).")
                print(f"Put the decisions in "
                      f"{adj_dir / 'adjudications.xlsx'} instead — two "
                      f"columns, 'item' and 'label'.")
    if adj:
        print()

    summary, all_dis, corrected = [], [], {}
    for key in AUDIT_CORPORA:
        A, B = load(key, "A"), load(key, "B")
        keyf = C.RESULTS_DIR / "annotation" / f"KEY_{key}.csv"
        if A is None or B is None or not keyf.exists():
            continue
        m = A.merge(B, on="item").merge(read_table_any(keyf), on="item")
        # Resolve each item: the agreed label, or the adjudicated one where the
        # annotators differed. Items still unresolved stay out of every rate.
        m["final"] = [r.A if r.A == r.B else adj.get(str(r.item).strip())
                      for r in m.itertuples()]
        unresolved = int(m.final.isna().sum())
        n_adj = int((m.A != m.B).sum()) - unresolved
        POP = STRATUM_POP.get(key)

        print(f"\n=== {key} ===")
        if n_adj:
            print(f"  {n_adj} adjudicated item(s) folded in"
                  + (f"; {unresolved} still unresolved" if unresolved else ""))
        for stratum in ["ALL"] + sorted(m.stratum.unique()):
            sub = m if stratum == "ALL" else m[m.stratum == stratum]
            if len(sub) < 2:
                continue
            r = cohen_kappa(sub.A.tolist(), sub.B.tolist())
            prev = max(pd.Series(sub.A.tolist() + sub.B.tolist())
                       .value_counts(normalize=True))
            print(f"  {stratum:10} n={r['n']:3}  raw={r['raw_agreement']:.1%}  "
                  f"chance={r['expected_chance']:.1%}  "
                  f"kappa={r['kappa']:+.2f} ({interpret(r['kappa'])})  "
                  f"PABAK={r['pabak']:+.2f}  majority class={prev:.0%}")
            if math.isnan(r["kappa"]) or r["expected_chance"] >= 0.999:
                print(f"  {'':10} kappa is undefined here: both annotators used "
                      f"a single label, so expected chance agreement is 1 and "
                      f"the correction divides by zero.")
                print(f"  {'':10} Report raw agreement "
                      f"({r['raw_agreement']:.0%}) and PABAK "
                      f"({r['pabak']:+.2f}) for this stratum instead.")
            elif r["kappa"] < 0.6 and r["raw_agreement"] >= 0.85:
                print(f"  {'':10} note: kappa is depressed by a "
                      f"{prev:.0%} majority class, not by disagreement "
                      f"(raw agreement {r['raw_agreement']:.0%}). Report both.")
            summary.append({"corpus": key, "stratum": stratum, **{
                k: v for k, v in r.items() if k != "labels"},
                "majority_class_share": round(prev, 4)})

        dis = m[m.A != m.B]
        if len(dis):
            print(f"  {len(dis)} disagreement(s) to adjudicate:")
            for _, r in dis.iterrows():
                print(f"    {r['item']}  score={r['toxicity']:.3f}  "
                      f"[{r['stratum']}]  A={r['A']}  B={r['B']}")
        else:
            print("  no disagreements")
        dis = dis.assign(corpus=key)
        all_dis.append(dis)

        # Error rates by stratum, on agreed items only, so that the figures do
        # not depend on an adjudication that has not happened yet.
        agreed = m[m.final.notna()]
        tox_set = {"toxic", "borderline"} if a.strict else {"toxic"}
        print("  error rates (agreed and adjudicated items):")
        for stratum in ["<0.50", "0.50-0.90", ">=0.90"]:
            sub = agreed[agreed.stratum == stratum]
            if not len(sub):
                continue
            n_tox = int(sub.final.isin(tox_set).sum())
            if stratum == "<0.50":
                # Below threshold the classifier says "not toxic", so a toxic
                # label here is a FALSE NEGATIVE, not a false positive.
                lo, hi = wilson(n_tox, len(sub))
                line = (f"    {stratum:10} false negatives {n_tox}/{len(sub)} "
                        f"= {n_tox / len(sub):.0%} [95% CI {lo:.0%}, {hi:.0%}]")
                if POP:
                    line += f"  -> up to {hi * POP[stratum]:,.0f} documents"
                print(line)
                if n_tox == 0:
                    print(f"    {'':10} SPOT CHECK ONLY. n = {len(sub)} against a "
                          f"stratum of {POP[stratum]:,} supports no estimate of")
                    print(f"    {'':10} the false-negative rate; the upper bound "
                          f"exceeds T's own count by an order of magnitude.")
                else:
                    print(f"    {'':10} NOTE: toxic content found below the "
                          f"threshold. This is reportable at n = {len(sub)} and "
                          f"means T under-counts as well as over-counts.")
            else:
                fp = len(sub) - n_tox
                lo, hi = wilson(fp, len(sub))
                kind = "census" if stratum == ">=0.90" and POP and \
                    len(sub) >= POP[stratum] * 0.9 else "sample"
                print(f"    {stratum:10} false positives {fp}/{len(sub)} "
                      f"= {fp / len(sub):.0%} [95% CI {lo:.0%}, {hi:.0%}]  ({kind})")

        # Corrected T, once every disagreement has been resolved. The >=0.90
        # stratum is enumerated, so its contribution is a count; only the
        # 0.50-0.90 band is scaled, by its audited true-positive rate.
        if unresolved == 0 and POP:
            band = agreed[agreed.stratum == "0.50-0.90"]
            high = agreed[agreed.stratum == ">=0.90"]
            rate = float(band.final.isin(tox_set).mean()) if len(band) else 0.0
            tp_high = int(high.final.isin(tox_set).sum())
            n_flag = POP["0.50-0.90"] + POP[">=0.90"]
            corr = (POP["0.50-0.90"] * rate + tp_high) / 50_000
            corrected[key] = {"raw": round(n_flag / 50_000, 5),
                              "corrected": round(corr, 6),
                              "band_tp_rate": round(rate, 4),
                              "high_tp": tp_high, "high_n": POP[">=0.90"]}

    if not summary:
        print("\nNothing to compute. Run 03c_annotation_set.py, have both "
              "annotators fill in the label column, then re-run this.")
        return 1

    out = C.RESULTS_DIR / "annotation"
    pd.DataFrame(summary).to_csv(out / "agreement_summary.csv", index=False)

    if corrected:
        print("\n=== Corrected T from the adjudicated labels ===")
        print(f"  {'corpus':14}{'T raw':>9}{'T corrected':>13}{'inflation':>11}")
        for c, v in corrected.items():
            infl = v["raw"] / v["corrected"] if v["corrected"] else float("nan")
            print(f"  {c:14}{v['raw']:>9.4f}{v['corrected']:>13.5f}{infl:>10.1f}x")
        pd.DataFrame([{"corpus": c, **v} for c, v in corrected.items()]).to_csv(
            out / "corrected_T.csv", index=False)
        print(f"  -> {out / 'corrected_T.csv'}")
    if all_dis:
        d = pd.concat(all_dis, ignore_index=True)
        d["adjudicated"] = ""
        d.to_csv(out / "disagreements.csv", index=False)
        print(f"\nDisagreements -> {out / 'disagreements.csv'}  "
              f"(fill 'adjudicated' and re-run)")
    print(f"Summary -> {out / 'agreement_summary.csv'}")

    print("\nFOR THE PAPER: report raw agreement, kappa, and the majority-class")
    print("share together. A kappa that looks low next to a raw agreement above")
    print("90% is a property of the class balance, not of the annotators, and")
    print("saying so pre-empts the objection rather than inviting it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
