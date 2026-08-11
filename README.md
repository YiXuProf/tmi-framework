# IMERG precipitation correction with the TMI framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21757101.svg)](https://doi.org/10.5281/zenodo.21757101)

Code accompanying the paper:

> Yi Xu. _Machine learning correction of satellite precipitation is governed
> by mechanism purity, not algorithmic complexity: a proof-of-concept study
> in Hunan, China, with pre-registered cross-regional validation._
> (under review)

Random forest (RF) and linear regression (LR) bias correction of GPM IMERG daily precipitation against the CN05.1 gridded gauge analysis, with ERA5 (u10, v10, tcwv) and SRTM DEM as additional predictors. All fields are remapped to a common 0.25° grid; models are trained on JJA 2016–2020 and evaluated on JJA 2021–2022. The repository also contains the **pre-registered cross-regional validation** over Guangxi and Guangdong (manuscript Sections 2.3.5 and 3.6).

## Pipeline

Four dependency-aware stages (each maps to the code location in parentheses):

1. **`train`** (`src/pipelines/stage_train.py` → `pipeline_train_model.py`) — load NetCDF inputs, align grids/times, fit summer & spring RF/LR models, export `joblib` models, `npz` samples, and NetCDF slices under `models/`.
2. **`analysis`** (`src/analysis/`) — ablation (Table 1), intensity-stratified SHAP (Table 2), subregional performance + bootstrap CIs (Table 3), seasonal transfer (Table 4), seasonal SHAP direction (Table 5), K-means clustering (Tables S1–S2), SHAP disorder bootstrap (Table S4), and Moran's I (Table S5).
3. **`figures`** (`src/figures/`) — manuscript Figs. 1–6 and Fig. S1 (SVG).
4. **`xreg`** (`src/pipelines/xreg_validation.py`; shared Moran's I in `src/core/morans_proxies.py`) — cross-regional two-arm validation: a priori proxies → **pre-registered predictions** → LORO transfer/retraining evaluation → Tables S6–S8, Figs. 7–8, Fig. S2.

```bash
python main.py                                   # all four stages, in order
python main.py --steps train analysis            # selected stages
python main.py --dry-run                         # show execution plan only
python main.py --stop-on-error                   # halt at first failure
```

Stages can also be run individually (repo root, `src` on `PYTHONPATH`):

```bash
python -c "import sys; sys.path.insert(0,'src'); from pipelines.stage_train import main; main()"
python -c "import sys; sys.path.insert(0,'src'); from analysis.stage_analysis import main; main()"
python -c "import sys; sys.path.insert(0,'src'); from figures.stage_figures import main; main()"
python src/pipelines/xreg_validation.py
```

## Project structure

```text
.
├─ main.py                      # Dependency-aware entrypoint (train → analysis → figures → xreg)
├─ requirements.txt             # Pinned environment (scikit-learn 1.7.2)
├─ assets/                      # Province boundaries: hunan / guangxi / guangdong .geojson
├─ src/
│  ├─ core/
│  │  ├─ project_config.py      # Paths, years, RF and grid constants
│  │  ├─ morans_proxies.py      # Single Moran's I definition (Table S5 / xreg)
│  │  ├─ shap_cache.py          # Shared SHAP cache utility
│  │  └─ plot_palette.py        # Shared figure palette
│  ├─ pipelines/
│  │  ├─ pipeline_train_model.py# Core training pipeline
│  │  ├─ stage_train.py         # train stage entrypoint
│  │  └─ xreg_validation.py     # xreg: proxies → pre-registration → LORO eval → Figs. 7/8/S2
│  ├─ analysis/                 # 9 analysis scripts + stage_analysis.py (see Pipeline)
│  └─ figures/                  # figure_*.py (Figs. 1–6, S1) + stage_figures.py
├─ models/                      # NOT tracked: trained joblib models + intermediates (see Outputs)
└─ results/                     # TRACKED: curated artifacts reported in the paper
   ├─ tables/                   # Tables 1–5 and S1–S4 (CSV)
   ├─ figures/                  # Figs. 1–6 and S1 (SVG)
   ├─ data/                     # Underlying CSVs (daily bias, seasonal transfer, K4 assignments)
   └─ xreg/                     # Cross-regional artifacts incl. the pre-registration record
```

## Where each manuscript item lives

| Manuscript item | Repository file(s) |
| --------------- | ------------------ |
| Table 1 (ablation) | `results/tables/Table1_ablation_results.csv` |
| Table 2 (SHAP by intensity) | `results/tables/Table2_shap_direction_by_intensity.csv` |
| Table 3 (subregions + CI) | `results/tables/Table3_subregional_performance.csv`, `Table3_bootstrap_ci.csv` |
| Table 4 (interannual/transfer) | `results/tables/Table4_interannual_performance_2021_2022.csv` |
| Table 5 (seasonal SHAP) | `results/tables/Table5_seasonal_shap_direction.csv` |
| Tables S1–S2 (K-means) | `results/tables/TableS1_*.csv`, `TableS2_*.csv`; raw labels in `results/data/TableS2_raw_kmeans_K4_assignments.csv` |
| Table S3 (t-SNE) | `results/tables/TableS3_tsne_sensitivity.csv` |
| Table S4 (disorder) | `results/tables/TableS4_shap_disorder_bootstrap.csv` |
| Table S5 (Hunan coherence) | Hunan rows of `results/xreg/proxies.csv` and `table_subregion_summary.csv`; generator `src/analysis/table_s5_morans_i.py` |
| Table S6 (transfer matrix) | `results/xreg/table_loro_matrix.csv`, `table_loro_deltaR2.csv`, `eval_matrix.csv` |
| Table S7 (prediction check) | `results/xreg/prediction_check.csv` |
| Table S8 (pre-registration) | `results/xreg/predictions_registered.csv` (+ `run_log.txt`) |
| Figs. 1–6, S1 | `results/figures/*.svg` |
| Figs. 7–8, S2 | `results/xreg/*.svg` |

## Requirements

- **Python** 3.10+ (developed and tested with **Python 3.10.19** on Linux)

```bash
pip install -r requirements.txt
```

> **Note:** Random Forest results are sensitive to the scikit-learn
> version (pinned: **1.7.2**). Other versions may produce slightly
> different numbers. Reproducibility constants: `RANDOM_STATE = 42`,
> `N_ESTIMATORS = 500`, `MAX_SAMPLES_PER_TIME = 2000`
> (`src/core/project_config.py`).

## Input data

Raw inputs are **not redistributed** with this repository (product licenses). Download from the original sources and place them under **`DATA_ROOT`** (environment variable; default `/mnt/data/imerg_correction_hunan` in `src/core/project_config.py`):

| Dataset | Source |
| ------- | ------ |
| GPM IMERG Final Run daily precipitation (V07) | NASA GES DISC — https://doi.org/10.5067/GPM/IMERGDF/DAY/07 |
| ERA5 hourly reanalysis, single levels (u10, v10, tcwv) | Copernicus Climate Data Store — https://doi.org/10.24381/cds.adbb2d47 |
| CN05.1 gridded daily precipitation | Nansen-Zhu International Research Centre, IAP, Chinese Academy of Sciences (on request) |
| SRTM 30 m DEM | Big Earth Data Science Data Center (CASEarth) — https://doi.org/10.12237/casearth.67ad563083917d6a7fa53543 |

Expected layout:

| Path under `DATA_ROOT` | Role |
| ---------------------- | ---- |
| `hunan/imerg_hunan.nc` | IMERG precipitation (Hunan) |
| `cn051.nc` | CN05.1 daily precipitation (target, shared by all provinces) |
| `hunan/era5_hunan.nc` | ERA5 u10 / v10 / tcwv (Hunan) |
| `hunan/dem_hunan_025.nc` | DEM pre-aggregated to ~0.25° (Hunan) |
| `guangxi/imerg_guangxi.nc4` (or `.nc`), `guangxi/era5_guangxi.nc`, `guangxi/dem_guangxi_025.nc` | Guangxi inputs (`xreg` only) |
| `guangdong/imerg_guangdong.nc4` (or `.nc`), `guangdong/era5_guangdong.nc`, `guangdong/dem_guangdong_025.nc` | Guangdong inputs (`xreg` only) |

Variable names are resolved via candidate lists in `project_config.py` (`*_VAR_CANDIDATES`); fields are renamed to `lat` / `lon` / `time` and interpolated to the Hunan target grid (108.65–114.35°E, 24.50–30.30°N, 0.25°, WGS84) or the corresponding provincial grid. Train/test years and seasons: `TRAIN_YEARS = 2016–2020`, `TEST_YEARS = 2021–2022`, `SUMMER_MONTHS = [6, 7, 8]`, `SPRING_MONTHS = [3, 4, 5]`.

## Pre-registration record (xreg)

`results/xreg/predictions_registered.csv` is the archived pre-registration record for the cross-regional test:

- The coherence–efficiency rule (**efficiency = 84.30 × Moran's I − 28.98**) was fitted on the four Hunan subregions only and locked at **2026-08-07T03:49:16 UTC**, before any Guangxi or Guangdong model was evaluated (enforced by code order in `xreg_validation.py`; see `results/xreg/run_log.txt`).
- Efficiency classes: low < 20%, medium 20–30%, high > 30%. Result: **4/5 class hits; MAE 2.6 percentage points** (Supplementary Tables S7–S8).
- Re-running `xreg_validation.py` overwrites this file with a fresh timestamp; the archived record shipped here is the one reported in the paper.

## Outputs

| Location | Contents |
| -------- | -------- |
| **`models/`** (not tracked) | `summer/rf_model_summer.joblib`, `summer/lr_model.joblib`, `spring/rf_model_spring.joblib`, `train_test_data.npz`, seasonal NetCDF slices, SHAP cache. Trained RF binaries reach several hundred MB and are regenerated locally via the `train` and `xreg` stages with pinned scikit-learn 1.7.2 |
| **`results/`** (tracked) | Curated paper artifacts — see "Where each manuscript item lives" above |

## SHAP cache

- Cache file: `models/summer/shap_values_test_cache.npz` (utility: `src/core/shap_cache.py`)
- The `analysis` stage warms the cache once; downstream figure scripts reuse it
- Automatically invalidated when the summer model, test NPZ, sample count, or feature order changes

## Troubleshooting

- **Missing packages** — `pip install -r requirements.txt` in the same environment used for `python main.py`.
- **`geopandas` / boundary errors** — install geopandas; ensure `assets/*.geojson` exist.
- **Empty time intersection** — check IMERG / CN05.1 / ERA5 overlapping `time` after preprocessing.
- **`xreg` missing GX/GD files** — confirm the `guangxi/` and `guangdong/` folders under `DATA_ROOT` (`.nc4` / `.nc` both accepted).
- **Stage dependency errors** — run in order (`train → analysis → figures → xreg`); `main.py` checks required artifacts before each stage.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this code in published work, please cite the accompanying paper:

> Yi Xu. _Machine learning correction of satellite precipitation is governed
> by mechanism purity, not algorithmic complexity: a proof-of-concept study
> in Hunan, China, with pre-registered cross-regional validation._
> (under review)

Additionally, please cite the IMERG, CN05.1, ERA5, and SRTM products listed under "Input data".
