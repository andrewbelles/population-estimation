# Population Estimation in Post-Census Years

## Overview

Population estimation is a unique challenge. The standard benchmark in this setting is the U.S. Population Estimates Program (PEP). The PEP model itself is a simple arithmetic approach: it takes the previous year's population, adds net population change from births and deaths, and then approximates migration and residual terms. This repo aims to improve PEP by learning an additional correction term.

I use remote-sensing and administrative metadata jointly for that correction. Contemporary work shows that, in terms of raw predictive power, remote-sensing often loses to administrative metadata; however, the two are fundamentally different in what they inform on.

## Method

Through self-supervised learning, I approximate latent manifolds from VIIRS nighttime lights and Sentinel-5P NO2. Likewise, using USPS vacancy data and migration-related administrative signals, I learn a denoised administrative manifold. The SSL embeddings are then used to approximate a graph topology over the county set, learning how counties interact in a manner beyond physical geography.

I derive an adjacency matrix at the county level using the learned graph topology. The Moran Eigenvector Maps (MEM) represent the dominant spatially autocorrelated directions of that adjacency matrix. The top-K eigenvectors are kept as features, with the working assumption that these latent spatial directions align with population movement and correction structure.

The MEM features and the administrative metadata are fed through a lasso regression model. I avoid non-linear models here because the correction term is fit on 2020 data and then extrapolated into postcensal years; overfitting to regime-specific noise would make that drift unstable. The combination of a linear model, L1 regularization, residual correction of PEP, and graph-derived spatial features is intended to keep the correction term well bounded.

I refer to the model as VNA (VIIRS, Nitrous, administrative).

## Main Pipeline Modules

The repository is split into parent modules that represent the core portions of dataset generation, SSL training, graph training, nowcasting, and analysis. Each module has a single parent script that consumes a `.yaml` config file. Config files can be found in `configs/`, with subfolders for each parent module.

The main modules are:

- `ingestion/`: builds annual tabular datasets and spatial bag roots.
- `manifold/`: trains embeddings and exports learned representations.
- `graph/`: learns the county graph topology and writes graph-derived features.
- `nowcast/`: runs strict censal evaluation and postcensal rollout.
- `analysis/`: runs hypothesis testing and supporting analysis.

Module-owned outputs live under the relevant module's `data/` folder. Most downstream outputs are stored in parquet; the spatial SSL bag roots use a `dataset.bin` / `index.csv` / `stats.bin` contract.

### Running the Pipeline

Each stage can be called as follows:

```bash
python -m ingestion.ingest --config configs/ingestion/config.ingest.yaml
python -m manifold.embeddings --config configs/manifold/config.embeddings.yaml
python -m graph.topology --config configs/graph/config.topology.yaml
python -m nowcast.censal --config configs/nowcast/config.nowcast.yaml
python -m nowcast.postcensal --config configs/nowcast/config.nowcast.yaml
python -m analysis.hypothesis --config configs/analysis/config.hypothesis.yaml
```

### Input Assumptions

The ingestion script assumes that yearly VIIRS and S5P rasters already exist as processed `.tif` inputs, and that USPS vacancy data for the first quarter of the referenced year is available. VIIRS and S5P acquisition can be further automated; USPS data is less straightforward because access may require authentication and virtual signature for intent of use.

## Testing and Validation

I emphasize strict validation and test practices. A large hurdle with this work is that the goal is to improve PEP while only having one modern census anchor year in the relevant regime. Because the downstream correction is evaluated against the 2020 census, the main supervised comparison has to be built around held-out structure within 2020 rather than naive random county splits or loose temporal comparisons.

My solution is to use strict cross-validation. Since I use spatial features in the model, I cannot naively perform train/test splits on counties. Instead I split by state groups balanced across major U.S. divisions, which greatly limits transductive leakage from adjacency in learning the correction term. Leakage is not assumed away; part of the analysis measures how much apparent improvement can be attributed to it and reports a more conservative adjusted result.

## Analysis

The unadjusted result is stronger than the adjusted confirmatory result, so I treat the adjusted result as the more conservative statement. The model's aggressive regularization gives the advantage that it stays relatively bounded to the PEP estimate, but an aggregate MAPE gain alone is not sufficient evidence if it is driven by only one population regime or by the easiest cases.

With that in mind, I analyze the corrected model against PEP by state, by population strata, and by hard cases. In both heavily rural and heavily urban settings, administrative data may fail to capture the full extent of migration, so the learned graph topology is intended to recover additional spatial structure that PEP and raw administrative features miss.

### Census Results

Correlation below means simple Pearson correlation, since the downstream estimator is linear. Residual correlation refers to alignment with the true PEP residual signal; signal correlation refers to alignment with the underlying census signal.

| Model | MAPE | Adjusted MAPE | Residual Corr. | Signal Corr. | Adjusted Residual Corr. |
| --- | ---: | ---: | ---: | ---: | ---: |
| `embeddings_mem` | 2.7087 | 2.7638 | 0.2468 | 0.9996 | 0.2472 |
| `mem` | 2.7492 | 2.7937 | 0.2317 | 0.9996 | 0.2198 |
| `pep` | 2.8808 | 2.8808 | 0.0000 | 0.9995 | 0.0000 |
| `embeddings` | 2.8812 | 2.8812 | 0.1494 | 0.9995 | 0.1494 |

The main takeaway is that `embeddings_mem` is the best strict censal model by both raw and adjusted MAPE, with `mem` close behind and plain `embeddings` offering little gain over `pep`. 

Reference ablations:

| Model | MAPE | Adjusted MAPE | Residual Corr. | Signal Corr. |
| --- | ---: | ---: | ---: | ---: |
| `embeddings_only` | 54.3384 | 54.3384 | -0.1349 | 0.9475 |
| `embeddings_mem_only` | 88.0346 | 88.0346 | -0.0157 | 0.9188 |

These ablations show that raw embedding-only regression is not a viable downstream path by itself; the gains come from using the spatial graph signal in a bounded correction model rather than from replacing the baseline structure outright.

## Hypotheses

I am carrying out hypothesis testing on the following propositions:

- VNA non-negligibly improves PEP in adjusted county-level MAPE under strict cross-validation. This is the primary confirmatory test, using a spatial block HAC estimator, with a mass-weighted block permutation fallback in low-block regimes.
- VNA non-negligibly improves population-weighted state aggregate error. This is the key robustness test, allowing large population centers to matter proportionally rather than treating all states as equal mass.
- VNA non-negligibly improves the counties that were already in the top 10% hardest cases under the PEP model.
- VNA non-negligibly improves PEP in population strata: < 5k, 5k - 50k, 250k - 1M, > 1M.
- VNA non-negligibly improves a meaningful share of state estimations.
- VNA's worst disimprovements are majority concentrated in strata < 25k.

Likewise, for postcensal nowcasting, I am testing safety propositions around bounded correction magnitude, bounded year-over-year growth distortion, and whether the correction term remains well behaved when extrapolated beyond the census anchor year.

### Hypotheses Results

To handle low effective sample size and spatial dependence, I do not treat counties as independent draws. Instead I use spatial block HAC when there are enough blocks, and switch to a mass-weighted block permutation test when the block count is too small for the asymptotics to be trustworthy.

| Priority | Proposition | Estimate | p-value | Result |
| --- | --- | ---: | ---: | --- |
| Primary | Adjusted county-level MAPE improves under strict cross-validation | 3.07% | 0.0050 | Passed |
| Primary | Top 10% hardest counties under PEP improve | 5.80% | < 1e-4 | Passed |
| Supporting | A meaningful share of states improve | 63.8% | 0.0395 | Passed |
| Supporting | `5k - 50k` stratum improves | 3.29% | 0.0031 | Passed |

| Proposition | Estimate | p-value | Result | Notes |
| --- | ---: | ---: | --- | --- |
| `< 5k` stratum improves | 3.59% | 0.0647 | Not significant | Positive, near threshold |
| `250k - 1M` stratum improves | 6.37% | 0.2255 | Not significant | Positive |
| `> 1M` stratum improves | 6.88% | 0.4199 | Not significant | Uses low-block mass-weighted block permutation fallback |
| Population-weighted state aggregate error improves | 0.0977 | 0.2803 | Not significant | Positive robustness result |
| Worst disimprovements are concentrated in `< 25k` | 0.5476 | 0.3220 | Not significant | Majority share not distinguishable from threshold |

| Safety Proposition | Estimate | p-value | Result |
| --- | ---: | ---: | --- |
| Post-anchor correction magnitude remains bounded | 100.0% within bound | < 1e-4 | Passed |
| Post-anchor year-over-year growth distortion remains bounded | 99.98% within bound | < 1e-4 | Passed |
