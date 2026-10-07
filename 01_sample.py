#!/usr/bin/env python3
"""
Script 01 — Corpus sampling.

Draws a uniform random sample of N_SAMPLE documents from the first
(OVERSAMPLE_FACTOR * N_SAMPLE) eligible documents of each corpus stream, using
reservoir sampling (Vitter's Algorithm R) with a fixed seed.

WHAT THIS IS, PRECISELY -- state it this way in the paper:
    "a uniform random sample of the first K documents of the HuggingFace
     streaming iterator, K = 1,000,000"
It is NOT a corpus-wide random sample. Streaming order is shard order, not
random order, so head-of-stream bias remains and must be declared as a
limitation. Taking the *first* N documents (as an earlier draft's code did)
is strictly worse: it is not random even within the head.

Deduplication: SHA-256 over whitespace-normalised, case-folded text, applied
BEFORE the reservoir so that duplicates cannot displace unique documents.

Usage:
    python scripts/01_sample.py                 # all corpora
    python scripts/01_sample.py --corpus dolma  # one corpus
    python scripts/01_sample.py --n 2000 --oversample 5   # quick trial run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

_WS = re.compile(r"\s+")


def norm_for_hash(text: str) -> str:
    return _WS.sub(" ", text).strip().casefold()


def doc_hash(text: str) -> str:
    return hashlib.sha256(norm_for_hash(text).encode("utf-8")).hexdigest()


def open_stream(spec: dict):
    """Open a streaming dataset, with a readable error if the config is wrong."""
    from datasets import load_dataset

    # data_files mode: point directly at parquet/jsonl shard URLs. Bypasses
    # both loading scripts (removed in datasets 4.0) and repo-level gating.
    if spec.get("data_files"):
        return load_dataset(
            spec.get("builder", "json"),
            data_files=spec["data_files"],
            split=spec["split"],
            streaming=True,
        )

    args = [spec["hf_path"]]
    if spec["config_name"] is not None:
        args.append(spec["config_name"])
    try:
        return load_dataset(*args, split=spec["split"], streaming=True)
    except Exception as exc:
        try:
            from datasets import get_dataset_config_names

            available = get_dataset_config_names(spec["hf_path"])
        except Exception:
            available = ["<could not enumerate>"]
        hint = ""
        m = str(exc)
        if "scripts are no longer supported" in m or "trust_remote_code" in m:
            hint = ("\nCAUSE: this repo uses a loading script. Support was "
                    "removed in datasets 4.0. Either set data_files= in "
                    "config.py to point at the shard URLs directly, or choose "
                    "a parquet-native mirror. Do not silently substitute a "
                    "different corpus without renaming it in the paper.")
        elif "gated" in m.lower() or "401" in m or "403" in m:
            hint = ("\nCAUSE: gated dataset. Accept the licence on the HF "
                    "dataset page, then run: hf auth login")
        raise RuntimeError(
            f"Could not open {spec['hf_path']} "
            f"(config={spec['config_name']!r}, split={spec['split']!r}).\n"
            f"Available configs: {available}{hint}\n"
            f"Original error: {exc}"
        ) from exc


def sample_corpus(key: str, spec: dict, n: int, oversample: int, seed: int) -> dict:
    ds = open_stream(spec)
    shuffled = False
    if C.STREAM_SHUFFLE:
        try:
            ds = ds.shuffle(seed=seed, buffer_size=C.SHUFFLE_BUFFER)
            shuffled = True
        except Exception as exc:
            print(f"  [{key}] WARNING: shard shuffle unavailable ({exc}). "
                  f"Sample is head-of-stream and may reflect a single source. "
                  f"Report this for this corpus.")
    print(f"  [{key}] shard_shuffle={shuffled}")
    rng = random.Random(seed)
    pool_target = n * oversample

    reservoir: list[str] = []
    seen_hashes: set[str] = set()
    eligible = 0          # documents that passed all filters (reservoir index)
    scanned = 0
    n_dupes = 0
    n_short = 0
    n_missing = 0
    t0 = time.time()

    for row in ds:
        scanned += 1
        if eligible >= pool_target:
            break

        text = row.get(spec["text_col"])
        if not isinstance(text, str):
            n_missing += 1
            continue
        text = text.strip()
        if len(text) < C.MIN_DOC_CHARS:
            n_short += 1
            continue
        if len(text) > C.MAX_DOC_CHARS:
            text = text[: C.MAX_DOC_CHARS]

        h = doc_hash(text)
        if h in seen_hashes:
            n_dupes += 1
            continue
        seen_hashes.add(h)

        eligible += 1
        if len(reservoir) < n:
            reservoir.append(text)
        else:
            j = rng.randrange(eligible)   # eligible is 1-indexed here
            if j < n:
                reservoir[j] = text

        if scanned % 100_000 == 0:
            print(f"  [{key}] scanned={scanned:,} eligible={eligible:,} "
                  f"kept={len(reservoir):,}", flush=True)

    elapsed = time.time() - t0
    df = pd.DataFrame({
        "doc_id": [f"{key}_{i:07d}" for i in range(len(reservoir))],
        "corpus": key,
        "text": reservoir,
    })
    out = C.DATA_DIR / f"sample_{key}.parquet"
    df.to_parquet(out, index=False)

    manifest = {
        "corpus": key,
        "label": spec["label"],
        "hf_path": spec["hf_path"],
        "config_name": spec["config_name"],
        "split": spec["split"],
        "seed": seed,
        "shard_shuffled": shuffled,
        "shuffle_buffer": C.SHUFFLE_BUFFER if shuffled else None,
        "n_requested": n,
        "n_retained": len(df),
        "candidate_pool_target": pool_target,
        "n_eligible_seen": eligible,
        "n_scanned": scanned,
        "n_dropped_duplicate": n_dupes,
        "n_dropped_too_short": n_short,
        "n_dropped_missing_text": n_missing,
        "min_doc_chars": C.MIN_DOC_CHARS,
        "max_doc_chars": C.MAX_DOC_CHARS,
        "elapsed_sec": round(elapsed, 1),
        "output": str(out),
    }
    (C.LOGS_DIR / f"manifest_{key}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"  [{key}] DONE  kept={len(df):,}  from {eligible:,} eligible "
          f"({scanned:,} scanned, {n_dupes:,} dupes) in {elapsed:.0f}s")
    if len(df) < n:
        print(f"  [{key}] WARNING: retained {len(df):,} < requested {n:,}. "
              f"Report the actual n in the paper, not the target.")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None, help="single corpus key")
    ap.add_argument("--n", type=int, default=C.N_SAMPLE)
    ap.add_argument("--oversample", type=int, default=C.OVERSAMPLE_FACTOR)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()

    keys = [a.corpus] if a.corpus else list(C.CORPORA)
    unknown = [k for k in keys if k not in C.CORPORA]
    if unknown:
        print(f"Unknown corpus key(s): {unknown}. Known: {list(C.CORPORA)}")
        return 2

    manifests, failed = [], []
    for k in keys:
        print(f"\n=== Sampling {k} ===", flush=True)
        try:
            manifests.append(sample_corpus(k, C.CORPORA[k], a.n, a.oversample, a.seed))
        except Exception as exc:
            print(f"  [{k}] FAILED: {exc}")
            failed.append(k)

    if manifests:
        pd.DataFrame(manifests).to_csv(C.LOGS_DIR / "sampling_manifest.csv", index=False)
        print(f"\nManifest -> {C.LOGS_DIR / 'sampling_manifest.csv'}")
    if failed:
        print(f"FAILED corpora: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
