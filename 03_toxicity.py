#!/usr/bin/env python3
"""
Script 03 — T dimension: toxicity rate.

    T(C) = proportion of documents with Detoxify toxicity score >= threshold

The only script in the pipeline that needs a GPU to be comfortable. On CPU it
still runs; expect roughly an order of magnitude more wall-clock time.

Writes per-document scores AND checkpoints every CHECKPOINT_EVERY batches, so
an interrupted Colab session resumes instead of restarting.

Usage:
    python scripts/03_toxicity.py
    python scripts/03_toxicity.py --corpus dolma --batch-size 32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

CHECKPOINT_EVERY = 50


def load_detoxify():
    try:
        from detoxify import Detoxify
    except ImportError:
        print("detoxify not installed.  pip install detoxify")
        return None, "cpu"
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass
    print(f"Loading Detoxify('{C.DETOXIFY_MODEL}') on {device} ...")
    return Detoxify(C.DETOXIFY_MODEL, device=device), device


def process(key: str, model, batch_size: int) -> bool:
    src = C.DATA_DIR / f"sample_{key}.parquet"
    if not src.exists():
        print(f"  [{key}] SKIP — {src.name} not found. Run 01_sample.py first.")
        return False

    out = C.RESULTS_DIR / f"perdoc_toxicity_{key}.parquet"
    ckpt = C.RESULTS_DIR / f".ckpt_toxicity_{key}.parquet"

    df = pd.read_parquet(src, columns=["doc_id", "text"])
    done: dict[str, float] = {}
    if ckpt.exists():
        prev = pd.read_parquet(ckpt)
        done = dict(zip(prev.doc_id, prev.toxicity))
        print(f"  [{key}] resuming from checkpoint: {len(done):,} docs already scored")

    todo = df[~df.doc_id.isin(done)].reset_index(drop=True)
    n_batches = (len(todo) + batch_size - 1) // batch_size

    for bi in range(n_batches):
        chunk = todo.iloc[bi * batch_size:(bi + 1) * batch_size]
        texts = [t[: C.TOXICITY_MAX_CHARS] for t in chunk.text]
        try:
            res = model.predict(texts)
            scores = list(res["toxicity"])
        except Exception as exc:
            print(f"  [{key}] batch {bi} failed ({exc}); scoring individually")
            scores = []
            for t in texts:
                try:
                    scores.append(float(model.predict(t)["toxicity"]))
                except Exception:
                    scores.append(float("nan"))
        for did, s in zip(chunk.doc_id, scores):
            done[did] = float(s)

        if (bi + 1) % CHECKPOINT_EVERY == 0 or bi == n_batches - 1:
            pd.DataFrame({"doc_id": list(done), "toxicity": list(done.values())}) \
              .to_parquet(ckpt, index=False)
            print(f"  [{key}] {len(done):,}/{len(df):,} scored", flush=True)

    res_df = pd.DataFrame({
        "doc_id": df.doc_id,
        "corpus": key,
        "toxicity": [done.get(d, float("nan")) for d in df.doc_id],
    })
    res_df["is_toxic"] = (res_df.toxicity >= C.TOXICITY_THRESHOLD).astype(int)
    res_df.to_parquet(out, index=False)
    ckpt.unlink(missing_ok=True)

    valid = res_df.toxicity.notna()
    t = float(res_df.loc[valid, "is_toxic"].mean())
    print(f"  [{key}] T={t:.4f}  ({int(res_df.loc[valid,'is_toxic'].sum()):,} toxic "
          f"of {int(valid.sum()):,} scored; threshold={C.TOXICITY_THRESHOLD})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--batch-size", type=int, default=C.TOXICITY_BATCH_SIZE)
    a = ap.parse_args()

    model, _ = load_detoxify()
    if model is None:
        return 1

    keys = [a.corpus] if a.corpus else list(C.CORPORA)
    print("\n=== T dimension ===")
    ok = [process(k, model, a.batch_size) for k in keys]
    return 0 if any(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
