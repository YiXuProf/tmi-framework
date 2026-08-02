# IMERG precipitation correction (Hunan)

This repository contains the code accompanying the paper:
_"Mechanism purity, not algorithmic complexity, governs machine learning correction of satellite precipitation: The TMI framework and a case study in Hunan Province, China"_ (under review).

Train **random forest (RF)** and **linear regression (LR)** models to bias-correct GPM IMERG daily precipitation against a gridded gauge analysis (**CN05.1**), with **ERA5** (u10, v10, tcwv) and **DEM** as extra predictors. The workflow interpolates all fields to a common 0.25° grid over Hunan Province (China), fits on summer months (JJA) for selected years, evaluates on held-out years, and exports **figures**, **CSV metrics**, and an optional **Excel** summary.

## Pipeline

1. **`train`** — Load NetCDF inputs, align grids/times, build train/test samples, fit RF & LR, save `joblib` models + `npz` predictions + NetCDF slices for downstream use.
2. **`analysis`** — Run ablation, SHAP-based diagnostics, seasonal transfer analysis, and bootstrap statistics; writes CSV/MD analysis artifacts.
3. **`figures`** — Generate manuscript figures (PNG/SVG) from model outputs and analysis-ready artifacts.

Entry point:

```bash
python main.py
```

Optional: run specific stages only:

```bash
python main.py --steps train
python main.py --steps analysis
python main.py --steps figures
python main.py --steps train analysis figures
```

Run stages separately if needed (from the repo root, with `src` on `PYTHONPATH`):

```bash
python -c "import sys; sys.path.insert(0,'src'); from pipelines.stage_train import main; main()"
python -c "import sys; sys.path.insert(0,'src'); from analysis.stage_analysis import main; main()"
python -c "import sys; sys.path.insert(0,'src'); from figures.stage_figures import main; main()"
```

## Project structure

```text
.
├─ assets/                  # Boundary files and optional fonts
├─ models/                  # Generated locally; not tracked in git
├─ results/                 # Generated locally; not tracked in git
├─ src/
│  ├─ core/
│  │  ├─ project_config.py          # Runtime configuration
│  │  ├─ plot_palette.py            # Shared plot color aliases
│  │  └─ shap_cache.py              # Shared SHAP cache utility
│  ├─ pipelines/
│  │  ├─ pipeline_train_model.py    # Core training pipeline
│  │  └─ stage_train.py             # Train stage entrypoint
│  ├─ analysis/
│  │  ├─ stage_analysis.py          # Analysis stage entrypoint
│  │  └─ *.py                       # Analysis scripts (CSV/MD outputs)
│  └─ figures/
│     ├─ stage_figures.py           # Figures stage entrypoint
│     └─ figure_*.py                # Figure scripts (PNG/SVG outputs)
└─ main.py                  # Repository-level entrypoint
```

## Requirements

- **Python** 3.10+ (developed and tested with **Python 3.10.19** on Linux)
- Install dependencies:

```bash
pip install -r requirements.txt
```

See `requirements.txt` for pinned versions (numpy, pandas, xarray, scikit-learn, matplotlib, netCDF4, joblib, scipy, shap, geopandas, openpyxl, SciencePlots).

> **Note:** Random Forest results are sensitive to the scikit-learn
> version (pinned: 1.7.2). Other versions may produce slightly
> different numbers.

## Input data

Set the data directory with the environment variable **`DATA_ROOT`** (default in `src/core/project_config.py` is `/mnt/data/imerg_correction_hunan`). Under `DATA_ROOT`, the pipeline expects:

| File               | Role                                                               |
| ------------------ | ------------------------------------------------------------------ |
| `imerg.nc`         | IMERG precipitation (time × lat × lon)                             |
| `cn051.nc`         | CN05.1 (or other) gridded daily precipitation for training targets |
| `era5.nc`          | ERA5: u10, v10, tcwv (and consistent `time`)                       |
| `dem_hunan_025.nc` | Static DEM on or near the target grid                              |

Variable names are resolved via candidate lists in `src/core/project_config.py` (`*_VAR_CANDIDATES`). Datasets are renamed to `lat` / `lon` / `time` where needed and interpolated to the target extent:

- Longitude **108.65–114.35°E**, latitude **24.50–30.30°N**, resolution **0.25°** (WGS84).

Training and test years, and seasonal months, are configured in `project_config.py` (`TRAIN_YEARS`, `TEST_YEARS`, `SUMMER_MONTHS`, `SPRING_MONTHS`).

## Configuration

Edit **`src/core/project_config.py`** for:

- `DATA_ROOT` default or use `export DATA_ROOT=/path/to/data`
- Train/test years and `SUMMER_MONTHS`
- `MAX_SAMPLES_PER_TIME`, `N_ESTIMATORS`, `RANDOM_STATE`
- Optional overrides (not required): `SHAP_PLOT_SAMPLES` and model/data output paths.

## Assets

- **`assets/hunan.geojson`** — Province boundary for map clipping and masks.
- **`assets/fonts/`** (optional) — Drop `.ttf`/`.otf` here if you want custom sans-serif fonts for figures.

## Outputs

| Location       | Contents                                                                                                                                                                                                                                                 |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`models/`**  | `summer/rf_model_summer.joblib`, `spring/rf_model_spring.joblib`, `summer/train_test_data.npz`, seasonal NetCDF slices, and SHAP cache (`summer/shap_values_test_cache.npz`)                                                                             |
| **`results/`** | Analysis CSV/MD (`ablation_results.csv`, `shap_by_intensity.csv`, `summer_subregions_performance_shap.csv`, `summer_central_south_r2_gap_bootstrap_ci.csv`, `shap_disorder_bootstrap.csv`, `spring_summer_shap_direction.csv`, ...), plus figure PNG/SVG |

## SHAP cache

To avoid recomputing the same SHAP values in both analysis and figure scripts, the pipeline now uses a shared cache:

- Cache file: `models/summer/shap_values_test_cache.npz`
- Source utility: `src/core/shap_cache.py`
- Stage behavior: `analysis` warms the cache once; downstream scripts reuse it

The cache is automatically invalidated and recomputed when:

- The summer model file (`RF_SUMMER_MODEL_PATH`) changes
- The test NPZ file (`DATA_NPZ_PATH`) changes
- Sample count or feature order no longer matches

## Figures (non-exhaustive)

- Scatter: raw IMERG vs CN05.1 and RF vs CN05.1
- Spatial: observed vs RF-corrected precipitation by year
- RMSE by rain-intensity class (IMERG / LR / RF)
- Domain-mean daily time series
- Permutation importance and SHAP (summary + tcwv dependence)
- IMERG vs RF bias maps
- Supplementary raw IMERG spatial means

## Troubleshooting

- **Missing Python packages (`shap`, `matplotlib`, etc.)** — Install dependencies with `pip install -r requirements.txt` in the same environment used to run `python main.py`.
- **`geopandas` / boundary errors** — Install geopandas and ensure `assets/hunan.geojson` exists; plotting uses the same mask philosophy as training.
- **Empty time intersection** — Check that IMERG, CN05.1, and ERA5 share overlapping `time` after preprocessing.
- **Excel skipped** — Install `openpyxl`; CSV tables are still written.
- **Pipeline dependency errors** — Run full stages in order: `python main.py --steps train analysis figures`.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in published work, please cite the accompanying paper:

> Yi Xu. _Mechanism purity, not algorithmic complexity, governs machine
> learning correction of satellite precipitation: The TMI framework._
> (under review)

Additionally, remember to cite the IMERG, CN05.1, and ERA5 products you used.
