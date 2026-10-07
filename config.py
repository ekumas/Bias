"""
CABEF pipeline — central configuration.

All tunable constants live here so that every number reported in the paper
can be traced back to a single, auditable source.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"
for _d in (DATA_DIR, RESULTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------
SEED = 42
N_SAMPLE = 50_000          # documents retained per corpus
OVERSAMPLE_FACTOR = 4
# Shard-order shuffling. Multi-source corpora (the Dolma family) are laid out
# source by source, so the head of the stream can be entirely arXiv or entirely
# Common Crawl. Reading the first K documents would then measure one source,
# not the corpus. .shuffle() permutes shard order and buffers within it.
# Leave this ON. Turning it off silently reintroduces single-source bias.
STREAM_SHUFFLE = True
SHUFFLE_BUFFER = 10_000

MIN_DOC_CHARS = 200        # documents shorter than this are discarded
MAX_DOC_CHARS = 20_000     # truncate longer documents (keeps Detoxify tractable)

# ----------------------------------------------------------------------------
# Corpora
#   config_name = HuggingFace dataset *configuration* (2nd positional arg of
#   load_dataset), NOT the split. Getting this wrong is the single most common
#   failure mode.
# ----------------------------------------------------------------------------
CORPORA = {
    "c4_cc_proxy": {
        "label": "C4 (CC proxy)",
        "hf_path": "allenai/c4",
        "config_name": "en",
        "split": "train",
        "text_col": "text",
    },
    # NOT Dolma v1.7. allenai/dolma is backed by a loading script (removed in
    # datasets 4.0) and is licence-gated. OLMo-mix-1124 is the official AI2
    # parquet-native successor in the same family: documented composition,
    # ungated, citable. Name it OLMo-mix-1124 in the manuscript -- never
    # "Dolma". Alternatives that also load: allenai/dolmino-mix-1124,
    # emozilla/dolma-v1_7-305B (community re-upload, subsetting procedure
    # undocumented -- avoid), allenai/dolma3_pool (pre-filtering pool, would
    # inflate T -- avoid).
    "olmo_mix": {
        "label": "OLMo-mix-1124",
        "hf_path": "allenai/olmo-mix-1124",
        "config_name": None,
        "split": "train",
        "text_col": "text",
    },
    "fineweb_edu": {
        "label": "FineWeb-Edu",
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "config_name": "sample-10BT",
        "split": "train",
        "text_col": "text",
    },
    "falcon_refined": {
        "label": "Falcon-RefinedWeb",
        "hf_path": "tiiuae/falcon-refinedweb",
        "config_name": None,
        "split": "train",
        "text_col": "content",
    },
    "tinystories": {
        "label": "TinyStories",
        "hf_path": "roneneldan/TinyStories",
        "config_name": None,
        "split": "train",
        "text_col": "text",
    },

    # A sixth corpus, added to test whether the inverse ordering of D and G
    # survives beyond four observations. With two dimensions ranking n corpora,
    # an exact reversal arises by chance with probability 1/n!: 4.2% at n = 4,
    # 0.14% at n = 6. Four observations cannot distinguish a structural
    # inversion from a coincidence; six can.
    #
    # NAMING. This is the uncopyrighted redistribution of The Pile, not The
    # Pile as published. Books3, BookCorpus2, OpenSubtitles, YouTube Subtitles
    # and OpenWebText2 are absent from it, which removes most of the corpus's
    # book-length prose and alters its composition materially. It is named
    # accordingly throughout, for the same reason OLMo-mix-1124 is not called
    # Dolma: a corpus that is not the one named is a different corpus.
    "pile_uncopyrighted": {
        "label": "The Pile (uncopyrighted)",
        "hf_path": "monology/pile-uncopyrighted",
        "config_name": None,
        "split": "train",
        "text_col": "text",
    },

    # RedPajama was the intended second addition and is unusable: all four of
    # its repositories are backed by loading scripts, unsupported since
    # datasets 4.0, exactly as Dolma was. That reopens the case for SlimPajama,
    # excluded earlier as a deduplicated derivative of RedPajama on the
    # reasoning that only one of a dependent pair can be counted. With
    # RedPajama unmeasurable, SlimPajama is the sole representative of that
    # pipeline and is independent of everything else in the set.
    #
    # Fill in from probe_second_corpus.py output. Do not guess.
    #
    # "slimpajama": {
    #     "label": "SlimPajama",
    #     "hf_path": "",
    #     "config_name": None,
    #     "split": "train",
    #     "text_col": "text",
    # },
}

# ----------------------------------------------------------------------------
# Bias Fingerprint — three dimensions only.
#
# The E (evaluator instability) dimension of earlier drafts is DELIBERATELY
# ABSENT. It was defined as |WinoBias delta|, which made the composite index
# circular (a benchmark score was used to adjust that same benchmark score)
# and confounded corpus effects with model size/architecture. D, T and G are
# corpus-intrinsic: they require no language model and are therefore free of
# that confound.
# ----------------------------------------------------------------------------

# Baseline weights, equal over the two reported dimensions. The equality is
# illustrative, not substantive: no normative argument supports it, and the
# weight sensitivity analysis exists precisely because the choice determines
# the outcome. T is absent because it was measured, audited and excluded.
BASELINE_WEIGHTS = {
    "w_D": 0.5,
    "w_G": 0.5,
}

RISK_BANDS = [
    ("LOW", 0.00, 0.25),
    ("MODERATE", 0.25, 0.50),
    ("HIGH", 0.50, 0.75),
    ("CRITICAL", 0.75, 1.01),
]

# Weight-sensitivity grid (weights must sum to 1.0)
WEIGHT_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
ROBUST_THRESHOLD = 0.90    # >= 90% agreement -> "robust"
SENSITIVE_THRESHOLD = 0.75  # <  75% agreement -> "weight-sensitive"

# ----------------------------------------------------------------------------
# Toxicity (T)
# ----------------------------------------------------------------------------
DETOXIFY_MODEL = "multilingual"
TOXICITY_THRESHOLD = 0.50
TOXICITY_BATCH_SIZE = 64
TOXICITY_MAX_CHARS = 1_500   # Detoxify truncates anyway; keeps memory bounded

# ----------------------------------------------------------------------------
# Geographic concentration (G)
#
# G = 1 - H(countries) / log2(N_REF)
#
# N_REF is fixed a priori at the number of UN member states so that G is
# comparable across corpora. Reporting this constant explicitly matters: an
# earlier draft left the normaliser undefined, which makes G unreproducible.
# ----------------------------------------------------------------------------
SPACY_MODEL = "en_core_web_sm"
GEO_ENTITY_TYPES = {"GPE", "LOC"}
N_REF_COUNTRIES = 193
SPACY_BATCH_SIZE = 64
SPACY_N_PROCESS = 1
GEO_MAX_CHARS = 5_000

# ----------------------------------------------------------------------------
# BBQ Category Coverage Failure (CCF)
# ----------------------------------------------------------------------------
BBQ_HF_PATH = "heegyu/bbq"
BBQ_THRESHOLD_PER_10K = 5.0
# Threshold sweep addresses the "arbitrary threshold" objection directly.
BBQ_THRESHOLD_SWEEP = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

# ----------------------------------------------------------------------------
# Uncertainty
# ----------------------------------------------------------------------------
N_BOOTSTRAP = 1_000
CI_LEVEL = 0.95


def risk_class(cbi: float) -> str:
    """Map a Composite Bias Index value onto its risk band."""
    for name, lo, hi in RISK_BANDS:
        if lo <= cbi < hi:
            return name
    return "CRITICAL"
