#!/usr/bin/env python3
"""
probe_second_corpus.py — find a usable sixth corpus.

RedPajama is out. All four of its repositories are backed by loading scripts,
unsupported since datasets 4.0, exactly as Dolma was.

That changes the case for SlimPajama. It was excluded earlier as a
deduplicated derivative of RedPajama, on the reasoning that only one of a
dependent pair can be counted. With RedPajama unmeasurable, SlimPajama is the
only representative of that pipeline, and it is independent of everything else
in the set: it is not a component of OLMo-mix-1124, nor derived from C4,
FineWeb or Falcon. The exclusion no longer applies.

Candidates are ordered by how independent their construction is from the five
corpora already measured.

    python probe_second_corpus.py
"""
from __future__ import annotations

CANDIDATES = [
    ("SlimPajama-627B",      "cerebras/SlimPajama-627B",        None, "train"),
    ("SlimPajama-6B",        "DKYoon/SlimPajama-6B",            None, "train"),
    ("Zyda-2",               "Zyphra/Zyda-2",                   None, "train"),
    ("Zyda",                 "Zyphra/Zyda",                     None, "train"),
    ("Nemotron-CC (sample)", "nvidia/Nemotron-CC",              None, "train"),
    ("Pile deduplicated",    "EleutherAI/the_pile_deduplicated", None, "train"),
    ("OSCAR 2301 en",        "oscar-corpus/OSCAR-2301",         "en", "train"),
    ("mC4 en",               "legacy-datasets/mc4",             "en", "train"),
    ("CONTROL Pile",         "monology/pile-uncopyrighted",     None, "train"),
]
TEXT_COLS = ("text", "content", "raw_content", "document")


def probe(label, path, cfg, split):
    print(f"\n=== {label} ===\n    {path}" + (f" [{cfg}]" if cfg else ""))
    try:
        from datasets import load_dataset

        args = [path] + ([cfg] if cfg else [])
        ds = load_dataset(*args, split=split, streaming=True)
        rec = next(iter(ds))
        col = next((c for c in TEXT_COLS if c in rec), None)
        if col is None:
            print(f"    LOADS, but no known text column. keys={list(rec)[:9]}")
            return
        print(f"    OK   text_col='{col}'  keys={list(rec)[:9]}")
        print(f"         preview: {' '.join(str(rec[col]).split())[:110]}...")
    except Exception as exc:
        msg = " ".join(str(exc).split())
        low = msg.lower()
        hint = ("  [loading script]" if "scripts are no longer" in low else
                "  [gated]" if ("gated" in low or "401" in low or "403" in low) else
                "  [withdrawn]" if ("doesn't exist" in low or "404" in low) else "")
        print(f"    FAIL {type(exc).__name__}: {msg[:130]}{hint}")


if __name__ == "__main__":
    try:
        import datasets
        print(f"datasets version: {datasets.__version__}")
    except ImportError:
        raise SystemExit("datasets not installed")
    for row in CANDIDATES:
        probe(*row)
    print("\nOne working candidate is enough. Send the output back.")
