# CABEF results — auto-generated, do not edit by hand

Weights: w_D=0.50, w_G=0.50  ·  seed=42  ·  bootstrap B=1000  ·  N_REF_COUNTRIES=193

## Table 1 — Bias Fingerprint

| Corpus | n | D [95% CI] | G [95% CI] | CBI | BBQ CCF |
|---|---|---|---|---|---|
| C4 (CC proxy) | 50,000 | 0.2485 [0.2307, 0.2669] | 0.4398 [0.4323, 0.4489] | 0.3442 | — |
| OLMo-mix-1124 | 50,000 | 0.3222 [0.3067, 0.3353] | 0.4149 [0.4098, 0.4218] | 0.3686 | — |
| FineWeb-Edu | 50,000 | 0.3545 [0.3380, 0.3708] | 0.4081 [0.4026, 0.4146] | 0.3813 | — |
| Falcon-RefinedWeb | 50,000 | 0.2357 [0.2198, 0.2526] | 0.4733 [0.4669, 0.4811] | 0.3545 | — |
| The Pile (uncopyrighted) | 50,000 | 0.3229 [0.3026, 0.3433] | 0.4521 [0.4451, 0.4612] | 0.3875 | 0.210 |

## Table 2 — Weight sensitivity (ordering)

| Corpus | CBI (equal weights) | CBI range | Rank range | Modal rank | Share |
|---|---|---|---|---|---|
| C4 (CC proxy) | 0.3442 | [0.268, 0.421] | 3-5 | 5 | 52.9% |
| OLMo-mix-1124 | 0.3686 | [0.332, 0.406] | 3-5 | 3 | 58.8% |
| FineWeb-Edu | 0.3813 | [0.360, 0.403] | 1-5 | 1 | 41.2% |
| Falcon-RefinedWeb | 0.3545 | [0.260, 0.450] | 1-5 | 4 | 35.3% |
| The Pile (uncopyrighted) | 0.3875 | [0.336, 0.439] | 1-2 | 2 | 52.9% |

## Table 3 — D diagnostics

| Corpus | f_female | Direction | Coverage | D pooled | D doc-avg |
|---|---|---|---|---|---|
| C4 (CC proxy) | 0.376 | male-skewed | 37.8% | 0.248 | 0.241 |
| OLMo-mix-1124 | 0.339 | male-skewed | 59.0% | 0.322 | 0.368 |
| FineWeb-Edu | 0.323 | male-skewed | 50.1% | 0.354 | 0.298 |
| Falcon-RefinedWeb | 0.382 | male-skewed | 46.7% | 0.236 | 0.234 |
| The Pile (uncopyrighted) | 0.339 | male-skewed | 33.3% | 0.323 | 0.319 |

D is symmetric about parity, so Table 3 carries the direction Table 1 discards. Coverage is the share of documents containing at least one gendered term; D is estimated only over those, and the share differs sharply between corpora. Doc-averaged D weighs every document equally, unlike the pooled estimator, which long documents dominate.


## Table 4 — Corpora with undefined G

| Corpus | n | D | Country mentions | BBQ CCF |
|---|---|---|---|---|
| TinyStories | 50,000 | 0.0841 | 293 | — |

G requires a minimum of 5,000 country mentions to be an estimate rather than an artefact of a handful of observations. These corpora fall below that floor, so G is reported as undefined and no CBI is computed for them. Reporting a value here, or substituting zero, would present a non-measurement as a low-diversity finding.
