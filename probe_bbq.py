#!/usr/bin/env python3
"""
probe_bbq.py — find a working source for the BBQ dataset.

`heegyu/bbq` is backed by a loading script (bbq.py), and loading scripts were
removed in datasets 4.0. Rather than guessing at mirrors, this probes the
options empirically.

The first candidate is the authors' own GitHub repository, which serves raw
JSONL over HTTPS. That is the better source regardless of the datasets issue:
it is the original artefact rather than a third-party re-upload, so the terms
extracted from it are the terms BBQ actually defines.

    python probe_bbq.py

Send the output back.
"""
from __future__ import annotations

import json
import urllib.request

GITHUB_BASE = ("https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/")
CATEGORIES = [
    "Age", "Disability_status", "Gender_identity", "Nationality",
    "Physical_appearance", "Race_ethnicity", "Race_x_SES", "Race_x_gender",
    "Religion", "SES", "Sexual_orientation",
]

HF_CANDIDATES = ["heegyu/bbq", "walledai/BBQ", "oskarvanderwal/bbq", "lighteval/bbq_helm"]


def probe_github() -> None:
    print("=== GitHub: nyu-mll/BBQ (raw JSONL) ===")
    ok = 0
    for cat in CATEGORIES:
        url = f"{GITHUB_BASE}{cat}.jsonl"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                first = r.readline().decode("utf-8")
            rec = json.loads(first)
            ok += 1
            if ok == 1:
                print(f"  {cat}: OK")
                print(f"    keys: {list(rec)}")
                ai = rec.get("answer_info")
                print(f"    answer_info: {json.dumps(ai)[:200] if ai else 'ABSENT'}")
                print(f"    ans0/1/2: {rec.get('ans0')!r} | "
                      f"{rec.get('ans1')!r} | {rec.get('ans2')!r}")
            else:
                print(f"  {cat}: OK")
        except Exception as exc:
            print(f"  {cat}: FAIL {type(exc).__name__}: {str(exc)[:90]}")
    print(f"  -> {ok}/{len(CATEGORIES)} categories reachable")


def probe_hf() -> None:
    print("\n=== HuggingFace mirrors ===")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets not installed")
        return
    for path in HF_CANDIDATES:
        try:
            ds = load_dataset(path, split="test", streaming=True)
            rec = next(iter(ds))
            print(f"  {path}: OK   keys={list(rec)[:8]}")
        except Exception as exc:
            print(f"  {path}: FAIL {str(exc)[:110]}")


if __name__ == "__main__":
    probe_github()
    probe_hf()
    print("\nSend this whole output back.")
