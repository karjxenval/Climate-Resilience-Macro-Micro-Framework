# Open Resilience Uganda: real-data macro--micro climate resilience workflow

This repository is a replication-grade Python pipeline for turning open climate, vegetation, agriculture, infrastructure, population, and macroeconomic data into district-level climate resilience planning evidence.

It is designed for the real-data follow-up paper to the macro--micro climate resilience framework. The aim is not to produce a vague map. The aim is to produce auditable evidence that a ministry, insurer, food-security analyst, infrastructure planner, or climate-risk team can inspect and reuse.

## What the pipeline delivers

Main-text outputs:

- data coverage and reproducibility table
- missingness heatmap
- Uganda district climate stress map
- vegetation/agricultural response map
- fixed-effects district response model table
- macro climate--economy model table where country-level remote-sensing data are available
- baseline comparison table against rainfall-only, NDVI-only, weak-capacity-only, exposure-only, equal-weight, PCA, random forest, and gradient boosting benchmarks
- improved RAS/priority surface
- district priority map
- act-now / measure-first / monitor / lower-priority action typology map
- top priority district action table
- robustness and uncertainty summary

Appendix/supplementary outputs:

- index construction audit JSON
- missingness by year and variable
- model summary text files
- all baseline scores
- machine-learning leave-one-year-out metrics
- East Africa transferability table
- external validation audit

## Data sources used

The pipeline uses real public data at runtime:

- CHIRPS rainfall through Google Earth Engine
- MODIS MOD13Q1 NDVI/EVI through Google Earth Engine
- VIIRS/NASA Black Marble night lights through Google Earth Engine
- WorldPop population through Google Earth Engine
- World Bank WDI through the public World Bank API
- geoBoundaries administrative boundaries by default, with optional user-supplied boundaries
- optional FAOSTAT, LSMS/UNPS, or DHS validation files supplied by the user under access rules

Synthetic data are not used for empirical claims. If you later add a smoke test, clearly mark it as software-only.

## Install

Using Anaconda Prompt:

```bash
conda env create -f environment.yml
conda activate open-resilience
```

Or using pip:

```bash
python -m pip install -r requirements.txt
```

## Earth Engine setup

Authenticate once:

```bash
earthengine authenticate
```

Then run:

```bash
python scripts/09_reproduce_all.py --config configs/uganda.yaml
```

## Running without Earth Engine

If Earth Engine is not available, provide a real extracted CSV at:

```text
data/interim/district_year_remote_sensing.csv
```

with columns:

```text
district_id,district_name,year,rain_season,rain_annual,ndvi,evi,nightlights,population
```

Then run:

```bash
python scripts/09_reproduce_all.py --config configs/uganda.yaml --skip-gee
```

## Output folders

```text
outputs/main_text/tables/
outputs/main_text/figures/
outputs/main_text/maps/
outputs/appendix/tables/
outputs/logs/
```

## Interpretation boundary

The action categories are decision-support labels, not automatic funding or intervention decisions. A district marked `Act now` means the evidence is strong enough for immediate planning review. A district marked `Measure first` means the signal is concerning but uncertainty is too high for major action without verification.
