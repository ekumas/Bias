#!/usr/bin/env python3
"""
probe_dolma.py — find a Dolma-family corpus that actually loads.

The `allenai/dolma` repository is backed by a loading script (dolma.py) and is
license-gated. Loading scripts were removed in datasets 4.0, so that path is
closed unless you downgrade AND authenticate.

Rather than guessing which mirror works, this probes candidates empirically:
for each, it opens a streaming iterator, pulls one record, and reports the
record's text column and a text preview.

    python probe_dolma.py

Send the output back. We pick the corpus from what actually responds, and
whatever we pick gets named honestly in the manuscript -- if it is not Dolma
v1.7, the paper says so, because OLMo-1B's training corpus is a claim about
provenance and cannot be approximated silently.
"""
from __future__ import annotations

import sys

CANDIDATES = [
    # (hf_path, config_name)
    ("allenai/dolma",              "v1_7"),
    ("allenai/dolma",              None),
    ("allenai/dolma3_pool",        None),
    ("allenai/olmo-mix-1124",      None),
    ("allenai/dolmino-mix-1124",   None),
    ("emozilla/dolma-v1_7-305B",   None),
    ("HuggingFaceFW/fineweb",      "sample-10BT"),   # control: should work
]

TEXT_COLS = ("text", "content", "raw_content", "document")


def probe(path: str, cfg: str | None) -> None:
    label = f"{path}" + (f" [{cfg}]" if cfg else "")
    try:
        from datasets import load_dataset

        args = [path] + ([cfg] if cfg else [])
        ds = load_dataset(*args, split="train", streaming=True)
        rec = next(iter(ds))
        col = next((c for c in TEXT_COLS if c in rec), None)
        keys = list(rec)[:8]
        if col is None:
            print(f"  LOADS, but no known text column. keys={keys}")
            return
        preview = " ".join(str(rec[col]).split())[:110]
        print(f"  OK   text_col='{col}'  keys={keys}")
        print(f"       preview: {preview}...")
    except Exception as exc:
        msg = " ".join(str(exc).split())[:220]
        print(f"  FAIL {type(exc).__name__}: {msg}")


def main() -> int:
    try:
        import datasets

        print(f"datasets version: {datasets.__version__}")
    except ImportError:
        print("datasets not installed.")
        return 1

    try:
        from huggingface_hub import whoami

        print(f"HF login: {whoami()['name']}")
    except Exception:
        print("HF login: NOT logged in  (run: hf auth login)")

    for path, cfg in CANDIDATES:
        print(f"\n=== {path}" + (f" [{cfg}]" if cfg else "") + " ===")
        probe(path, cfg)

    print("\nSend this whole output back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
