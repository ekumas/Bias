#!/usr/bin/env python3
"""
probe_new_corpora.py — find loadable paths for The Pile and RedPajama.

Written because Dolma cost an afternoon: its repository is backed by a loading
script, unsupported since datasets 4.0, and is licence-gated, and none of that
was visible until the download failed. Both additions here have the same
exposure — The Pile's canonical repository was withdrawn and survives only in
mirrors, and RedPajama exists in several incompatible versions.

Each candidate is opened as a streaming iterator, one record is pulled, and the
text column and a preview are reported. Nothing is downloaded beyond the first
shard header.

    python probe_new_corpora.py

Send the output back. config.py entries are written from what actually
responds, not from what ought to.
"""
from __future__ import annotations

import sys

CANDIDATES = [
    # (label, hf_path, config_name, split)
    # --- The Pile: EleutherAI, 22 sources, academically weighted. The most
    #     valuable addition because its construction shares no pipeline with
    #     any corpus already measured.
    ("Pile / monology copy",      "monology/pile-uncopyrighted", None, "train"),
    ("Pile / NeelNanda 10k",      "NeelNanda/pile-10k",          None, "train"),
    ("Pile / EleutherAI orig",    "EleutherAI/pile",             None, "train"),
    ("Pile / mit-han-lab",        "mit-han-lab/pile-val-backup", None, "validation"),

    # --- RedPajama: an independent LLaMA reproduction pipeline. V2 is the
    #     current release; V1 is the 1T-token original.
    ("RedPajama V2 sample",       "togethercomputer/RedPajama-Data-V2",
                                  "sample", "train"),
    ("RedPajama V2 sample-10B",   "togethercomputer/RedPajama-Data-V2",
                                  "sample-10B", "train"),
    ("RedPajama V1 1T sample",    "togethercomputer/RedPajama-Data-1T-Sample",
                                  None, "train"),
    ("RedPajama V1 1T",           "togethercomputer/RedPajama-Data-1T",
                                  None, "train"),

    # --- Control: already known to work. If this fails the problem is the
    #     network, not the corpus.
    ("CONTROL FineWeb",           "HuggingFaceFW/fineweb", "sample-10BT", "train"),
]

TEXT_COLS = ("text", "content", "raw_content", "document")


def probe(label, path, cfg, split):
    header = f"{path}" + (f" [{cfg}]" if cfg else "")
    print(f"\n=== {label} ===\n    {header}")
    try:
        from datasets import load_dataset

        args = [path] + ([cfg] if cfg else [])
        ds = load_dataset(*args, split=split, streaming=True)
        rec = next(iter(ds))
        col = next((c for c in TEXT_COLS if c in rec), None)
        keys = list(rec)[:9]
        if col is None:
            print(f"    LOADS, but no known text column. keys={keys}")
            return
        preview = " ".join(str(rec[col]).split())[:110]
        print(f"    OK   text_col='{col}'")
        print(f"         keys={keys}")
        print(f"         preview: {preview}...")
    except Exception as exc:
        msg = " ".join(str(exc).split())
        hint = ""
        low = msg.lower()
        if "scripts are no longer supported" in low or "trust_remote_code" in low:
            hint = "  [loading script — unusable, as with Dolma]"
        elif "gated" in low or "401" in low or "403" in low:
            hint = "  [gated — accept the licence, then: hf auth login]"
        elif "doesn't exist" in low or "not found" in low or "404" in low:
            hint = "  [repository withdrawn or renamed]"
        elif "config" in low and "name" in low:
            hint = "  [wrong config name — see the list in the message]"
        print(f"    FAIL {type(exc).__name__}: {msg[:150]}{hint}")


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
        print("HF login: NOT logged in  (hf auth login — raises rate limits)")

    for row in CANDIDATES:
        probe(*row)

    print("\n" + "=" * 60)
    print("Send this whole output back. Only one Pile candidate and one")
    print("RedPajama candidate are needed; the rest are alternatives in case")
    print("the first choice is unavailable.")
    print("\nIndependence matters more than size here. Dolma and DCLM were")
    print("excluded because both are components of OLMo-mix-1124, so measuring")
    print("them would count one corpus twice and invalidate the 1/n! argument")
    print("for the ranking inversion. SlimPajama was excluded as a deduplicated")
    print("derivative of RedPajama; only one of the pair can be counted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
