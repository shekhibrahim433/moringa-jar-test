# Moringa jar test ML

Machine learning analysis of bench-scale jar tests that use *Moringa oleifera* seed extract as a natural coagulant for river water clarification.

## Data

Coutinho, L. F. N., Oliveira, D. S. de, and Donadel, C. B. (2026). Experimental jar test dataset for river water clarification using *Moringa oleifera*. *Data in Brief*, 69, 113341. https://doi.org/10.1016/j.dib.2026.113341

The dataset (CC BY 4.0) is on Zenodo: https://doi.org/10.5281/zenodo.21652410. It is not redistributed here. Download `DataSet.xlsx` and place it in `data/raw/`.

## Layout

```
data/raw/          DataSet.xlsx (download from Zenodo)
data/processed/    written by notebook 01
src/               paths, loading, features, folds, metrics, kinetic models, plot style
notebooks/         run in order
results/           metrics, predictions, stats
tables/            CSVs behind the manuscript tables
figures/           eda, models, paper
```

## Notebooks

| Notebook | Content |
|---|---|
| `01_data_eda.ipynb` | Data audit, design structure, replicate noise, outliers, correlations, variance by day |
| `02_kinetics.ipynb` | Settling-curve model selection: fit quality, identifiability, extrapolation |
| `03_ml_models.ipynb` | Nested leave-one-day-out validation of baseline, linear, GP, SVR, tree and MLP models; random-split comparison |
| `04_tabpfn.ipynb` | TabPFN on the same folds (run on Colab with a GPU) |
| `05_hybrid.ipynb` | Hybrid models: ML predicts San parameters, the San equation gives the curve |
| `06_comparison_stats.ipynb` | Friedman, Nemenyi and Wilcoxon-Holm tests over held-out days; random vs day split; full vs core features; prediction intervals |
| `07_xai.ipynb` | SHAP and ALE for final ExtraTrees and hybrid models; hybrid kinetic parameters over dose and raw-water turbidity |
| `08_dose_recommendation.ipynb` | Dose-response, recommended dose by raw-water turbidity, leave-one-day-out check of dose choice, one-jar calibration |
| `09_paper_figures.ipynb` | Numbered main and supplementary figures and tables for the manuscript |

## Setup

```
conda create -n ml_env python=3.11
conda activate ml_env
pip install -r requirements.txt
```

Run the notebooks from the `notebooks/` folder.

## Licence

Code: MIT. Contact: info.shekhibrahim.25@gmail.com
