# CABEF pipeline — corpus-intrinsic bias fingerprinting

Computes the three-dimensional Bias Fingerprint **BF(C) = (D, T, G)**, the
Composite Bias Index, and the BBQ Category Coverage Failure rate for a set of
pre-training corpora — with bootstrap confidence intervals and a full weight
sensitivity analysis.

Every quantity here is **corpus-intrinsic**: none of it requires a language
model. That is the whole point. Nothing in this pipeline can confound a corpus
effect with a model-size or model-architecture effect, because no model is
involved.

---

## What was deliberately removed, and why

| Removed | Reason |
|---|---|
| **E dimension** (`\|WinoBias delta\|`) | Circular: a benchmark score was used as an input to the factor that adjusts that same benchmark score. Also model-dependent, so it reintroduced the confound. |
| **Cross-corpus WinoBias / StereoSet / CrowS-Pairs comparison** | Not identifiable. The available "native" models span 1M–7B parameters and two architectures (MLM and causal LM). Score differences cannot be attributed to the corpus. Keep these as stated hypotheses, not results. |
| **CABS** (Corpus-Adjusted Benchmark Score) | Not scale-invariant: the same α and β were applied to a 0–1 metric and a 0–100 metric. Also did not reduce to the raw score at CBI = 0 (α(0) = 0.924, not 1). Reformulate before reintroducing; list as future work meanwhile. |

State each removal explicitly in the manuscript. A reviewer who sees a
framework shrink from four dimensions to three without explanation will assume
the worst.

---

## Install

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Verify before you spend compute

```bash
python scripts/00_smoke.py            # writes SYNTHETIC fixtures
python scripts/02_demographic.py
python scripts/06_cbi.py --bootstrap 200
rm -rf data results && mkdir -p data results   # MANDATORY cleanup
```

If script 06 prints `Self-verification OK`, the chain is wired correctly.
**Delete `data/` and `results/` afterwards.** Smoke numbers are fake and must
never reach the manuscript.

## Real run

```bash
python scripts/01_sample.py        # ~2-6 h, network-bound
python scripts/02_demographic.py   # ~5 min, CPU
python scripts/03_toxicity.py      # ~2-4 h GPU / ~20-40 h CPU
python scripts/04_geographic.py    # ~3-8 h CPU
python scripts/05_bbq_ccf.py       # ~1-3 h CPU
python scripts/06_cbi.py           # ~10 min
```

Trial run first, with a small sample, to confirm every dataset config resolves:

```bash
python scripts/01_sample.py --n 2000 --oversample 5
```

Script 03 checkpoints every 50 batches and resumes automatically, so a dropped
Colab session costs minutes, not hours.

## Outputs

| File | Contents |
|---|---|
| `results/bias_fingerprint.csv` | D, T, G with 95% bootstrap CIs, CBI, risk class |
| `results/cbi_sensitivity.csv` | Risk-class stability across all valid weight configurations |
| `results/bbq_ccf.csv` | CCF at every threshold in the sweep |
| `results/bbq_ccf_detail.csv` | Per-term document frequencies |
| `results/bbq_terms.csv` | Extracted BBQ subgroup terms — **audit this by hand** |
| `results/country_counts_*.csv` | Country frequency per corpus |
| `results/lexicon_audit.csv` | Every gendered term counted |
| `results/paper_tables.md` | Paste-ready markdown tables |
| `logs/sampling_manifest.csv` | Documents scanned, retained, deduplicated, per corpus |

---

## The self-verification guard

Script 06 recomputes CBI from the **reported, rounded** D/T/G and aborts if any
value disagrees with the published figure by more than half a unit in the
fourth decimal.

This exists for a specific reason. The previous draft of this manuscript
reported five CBI values, and not one of them could be regenerated from the
paper's own formula and its own D/T/G values — C4 was off by 0.24, and three of
five risk classifications were wrong as a result. The guard makes that class of
error impossible to ship. Do not disable it.

---

## What must be written into the manuscript

These are limitations of the design, not bugs. Declaring them is what separates
this from the draft it replaces.

1. **Sampling is shard-shuffled, not corpus-wide.** Script 01 permutes shard
   order (`.shuffle`, buffer 10,000, seed 42) and then draws a uniform random
   sample of the first `K = OVERSAMPLE_FACTOR × N_SAMPLE` eligible documents.
   Shuffling matters for multi-source corpora, which are laid out source by
   source: without it the head of the stream can be entirely arXiv or entirely
   Common Crawl, and the measurement describes one source rather than the
   corpus. Report `shard_shuffled` from the manifest per corpus.
1b. **The Dolma substitution.** `allenai/dolma` is backed by a loading script
   (unsupported since datasets 4.0) and is licence-gated, so it is not used.
   The corpus measured is **OLMo-mix-1124** (`allenai/olmo-mix-1124`), the
   official AI2 parquet-native successor. Name it that way throughout. Do not
   write "Dolma".
2. **Sample size.** 50,000 documents against corpora of 10^11–10^12 tokens.
   Report the actual retained `n` from `logs/sampling_manifest.csv`, never the
   target.
3. **D is a binary-gender measure** built from a closed-class pronoun and
   kinship lexicon. Name the dimension accordingly and address the tension with
   the paper's own critique of narrow operationalisations. See `lexicons.py`.
4. **G depends on NER quality.** Script 04 reports the entity mapping rate;
   quote it. Below 30% the measure is unreliable and the script warns.
5. **G's normaliser is fixed at 193** UN member states (`N_REF_COUNTRIES`).
   State the constant — leaving it undefined makes G unreproducible.
6. **CCF's threshold is arbitrary.** Report the full sweep from
   `bbq_ccf.csv`. If corpus ranking is stable across thresholds, say so and the
   objection dissolves. If it is not, report CCF as a curve, not a number.
7. **n = 5 corpora.** No correlational claim between CBI and anything else is
   supportable at this n. Report differences descriptively, with CIs.
8. **Weights are normative.** The sensitivity analysis shows whether
   classifications are stable; it does not identify correct weights.

## Reproducibility statement

Record and publish: `pip freeze`, the spaCy model version, the Detoxify model
version, `logs/sampling_manifest.csv`, and the seed (42). Deposit
`results/` and this directory in a public repository and cite the DOI in the
Data & Code Availability section — the previous draft promised "reproducible
Python code" with no repository behind it.
