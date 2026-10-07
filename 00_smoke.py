#!/usr/bin/env python3
"""
Script 00 — smoke test.

Generates SYNTHETIC stand-ins for the outputs of scripts 01/03/04 so that the
whole chain can be exercised without HuggingFace downloads, a GPU, spaCy or
Detoxify. Use it to confirm the pipeline runs end-to-end before committing
hours of compute.

    python scripts/00_smoke.py
    python scripts/02_demographic.py     # runs for real on synthetic text
    python scripts/06_cbi.py --bootstrap 200

EVERY FILE THIS SCRIPT WRITES IS FAKE. It stamps `synthetic=True` into
data/SMOKE_TEST_ARTEFACTS_PRESENT.txt. Delete data/ and results/ before the
real run — do not let a smoke number reach the manuscript.

    rm -rf data results && mkdir -p data results
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

N = 600
rng = np.random.default_rng(C.SEED)

# Distinct synthetic profiles so the composite step has something to separate.
PROFILES = {
    "c4_cc_proxy":    dict(p_fem=0.30, tox=0.16, geo_alpha=0.30, n_geo=40),
    "olmo_mix":       dict(p_fem=0.42, tox=0.08, geo_alpha=0.80, n_geo=70),
    "fineweb_edu":    dict(p_fem=0.45, tox=0.03, geo_alpha=0.60, n_geo=55),
    "falcon_refined": dict(p_fem=0.35, tox=0.11, geo_alpha=0.40, n_geo=45),
    "tinystories":    dict(p_fem=0.47, tox=0.01, geo_alpha=0.15, n_geo=12),
}

FEM_W = ["she", "her", "woman", "mother", "daughter", "sister"]
MAL_W = ["he", "him", "man", "father", "son", "brother"]
FILLER = ("the quick document mentions several ordinary things and continues "
          "for a while so that it passes the minimum length filter applied "
          "during sampling in the first stage of the pipeline ")


def make_corpus(key: str, prof: dict) -> None:
    ids = [f"{key}_{i:07d}" for i in range(N)]

    # --- synthetic text (script 02 will genuinely parse this) ---
    texts = []
    for _ in range(N):
        k = int(rng.integers(3, 12))
        words = [(FEM_W if rng.random() < prof["p_fem"] else MAL_W)[
            int(rng.integers(0, len(FEM_W)))] for _ in range(k)]
        texts.append(FILLER * 3 + " ".join(words) + ".")
    pd.DataFrame({"doc_id": ids, "corpus": key, "text": texts}) \
      .to_parquet(C.DATA_DIR / f"sample_{key}.parquet", index=False)

    # --- fake toxicity (stand-in for script 03) ---
    toxic = rng.random(N) < prof["tox"]
    tox = np.where(toxic, rng.uniform(0.50, 0.99, N), rng.uniform(0.00, 0.49, N))
    tdf = pd.DataFrame({"doc_id": ids, "corpus": key, "toxicity": tox})
    tdf["is_toxic"] = (tdf.toxicity >= C.TOXICITY_THRESHOLD).astype(int)
    tdf.to_parquet(C.RESULTS_DIR / f"perdoc_toxicity_{key}.parquet", index=False)

    # --- fake geography (stand-in for script 04) ---
    codes = [f"C{i:02d}" for i in range(prof["n_geo"])]
    probs = rng.dirichlet(np.full(prof["n_geo"], prof["geo_alpha"]))
    per_doc = []
    for _ in range(N):
        k = int(rng.integers(0, 6))
        per_doc.append(";".join(rng.choice(codes, size=k, p=probs)) if k else "")
    pd.DataFrame({"doc_id": ids, "corpus": key, "countries": per_doc}) \
      .to_parquet(C.RESULTS_DIR / f"perdoc_geographic_{key}.parquet", index=False)


def main() -> int:
    for key, prof in PROFILES.items():
        make_corpus(key, prof)
        print(f"  [{key}] synthetic sample + toxicity + geography written")

    # Plausible-shaped fake CCF so script 06 has the column.
    pd.DataFrame([{"corpus": k, "n_docs": N, "n_terms": 100,
                   "CCF": round(float(rng.uniform(0.15, 0.55)), 4)}
                  for k in PROFILES]).to_csv(C.RESULTS_DIR / "bbq_ccf.csv", index=False)

    (C.DATA_DIR / "SMOKE_TEST_ARTEFACTS_PRESENT.txt").write_text(
        "synthetic=True\n"
        "These files are FAKE. Delete data/ and results/ before the real run.\n",
        encoding="utf-8")
    print("\nSmoke fixtures ready. Now run:")
    print("  python scripts/02_demographic.py")
    print("  python scripts/06_cbi.py --bootstrap 200")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
