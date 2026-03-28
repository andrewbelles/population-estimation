# Population Estimation in Post-Census Years

## Overview

Population estimation is a unique challenge. The best label for a standard forecasting model is the U.S. population estimation program (PEP). The PEP model itself is a simple arithmetic approach, taking the previous year's population, adding the net population change from births/deaths, then approximating constant migration and residual terms. This repo aims to improve the PEP, by learning an additional correction term.

I propose a novel method for correction through the use of remote-sensing and administrative metadata. Contemporary work shows that in terms of raw predictive power, remote-sensing loses to administrative metadata; however, they do not consider that the two are fundamentally different in what they inform on.

## Method

Through Self-Supervised Learning, I approximate latent manifolds from VIIRS nighttime lights and Sentinel-5P's $N0_2$. Likewise, using USPS Vacancy data, and migration vectors derived from county-aggregate housing market data, I learn a denoised manifold. The SSL embeddings are used to approximate a graph topology over the CONUS, learning how counties interact in a manner beyond physical geography.

I derive an adjacency matrix at a county-level over CONUS using the learned graph topology. The Moran Eigenvectors represent the spectrum of spatially autocorrelated directions of the adjacency matrix. The top-K eigenvectors are kept as features, representing dominant modes of how "energy" travels between counties. The key assumption is that this concept of "energy" aligns with population.

The top-K eigenmaps and the administrative metadata are fed through a lasso regression model. I avoid non-linear models. For this type of problem, non-linear models tend to overfit to regime specific noise, forcing the correction term to poorly generalize. Since the model is bootstrapped, being fit on 2020 data and extrapolated to the N-th post-census year, I cannot afford errors to grow exponentially. This is a joint effort by using a linear model, L1 regularization, correcting residual error of the PEP model, and emphasizing the use of eigenvectors from abstract remote-sensing embeddings. All of which contribute to the correction term being well bounded, limiting drift in counties that the correction term harms.

I will refer to the model as VNA (monicker for VIIRS, Nitrous, administrative).

## Main Pipeline Modules

The repository is split up into parent modules that represent the core portions of dataset generation, ssl training, graph training, and testing. Each module has a single parent script that consumes a `.yaml` config file that orchestrates how that module should operate. Config files can be found in `configs/`, with subfolders for each parent module that contain the relevant configuration file.

Data that is generated/owned by a module is stored within that folder in a `/data/` subfolder. All data is stored in parquet files.

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

Key note: The ingestion script assumes that there exists `.tif` files for VIIRS and S5P, median masked, aggregated through the year it references. Likewise it assumes the USPS Vacancy data for the first quarter of the referenced year exists. I have intentions to automate this process for VIIRS and S5P, but USPS data requires authentication because of the potentially sensitive nature of the data.

## Testing and Validation

I emphasize strict validation/test practices. A large hurdle with this work is that the aim is to improve the PEP model. Since the PEP model is the best proxy for the census in years between two censuses (2010-2020), I cannot use those years for training. Likewise, due to COVID-19, there is large concept drift, requiring the model to see the census anchor year when comparing PEP performance to my own model's.

My solution is to use strict cross-validation. Since I use spatial features in my model (the top-K spatial eigenvectors), I cannot naively perform train/test splits on counties. Instead I safely split by state-groups, regions of the U.S., which greatly limits transductive leakage from adjacency in the learning of the correction term. It is unavoidable, which is why a portion of analysis has been devoted to measuring the total number of counties that can "cheat" as well as the percent improvement that can be contributed to transductive leakage.

## Key Analysis

It is important to note that although the model outperforms the PEP model by roughly 5.9% in relative MAPE, this number is slightly optimistic due to aforementioned transductive leakage, and because it does not necessarily carry over in future years. The model's aggressive regularization gives the advantage that it is relatively bounded to the PEP estimate. Likewise, the model can hack raw MAPE by performing better on either only rural or only urban counties, performing worse in general for the rest.

With that in mind, I have extensive analysis on the performance of the corrected model against PEP both by state and population strata. In both heavily rural and urban population centers we expect administrative data to fail to capture the full extent of population migration, allowing the learned graph topology to excel, modeling undocumented migration well.

### Hypotheses

I am carrying out hypothesis testing on the following propositions:

- VNA non-negligibly improves PEP in adjusted county-level MAPE under strict cross-validation. This is the primary confirmatory test, using state-clustered sign-flip permutation testing so that the comparison remains apples-to-apples with the reported MAPE improvement while still respecting spatial dependence.
- VNA non-negligibly improves population-weighted state aggregate error. This is the key robustness test, allowing large population centers to matter proportionally rather than treating all states as equal mass.
- VNA non-negligibly improves PEP in population stratas: < 5k, 5k - 50k, 250k - 1M, > 1M.
- VNA non-negligibly improves a meaningful share of state estimations.
- VNA's worst disimprovements are majority concentrated in strata < 25k.

Likewise, for postcensal nowcasting, I am testing safety propositions around bounded correction magnitude, bounded year-over-year growth distortion, and whether the correction term remains well behaved when extrapolated beyond the census anchor year.
