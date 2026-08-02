"""Build summer subregion performance + SHAP mechanism shift table.

Output columns:
- Subregion
- DEM_mean
- N
- R²_RF
- RMSE_RF
- Bias_RF
- imerg_ρ
- tcwv_ρ
- dem_ρ
- v10_ρ
- v10_direction

Reads:
- models/rf_model_summer.joblib (or config.RF_SUMMER_MODEL_PATH)
- models/train_test_data.npz (X_test, y_test, rf_pred, lat_test, lon_test)

Writes:
- results/summer_subregions_performance_shap.csv
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

from core import project_config as config
from core.shap_cache import DEFAULT_FEATURE_NAMES, get_test_shap_values
from pipelines.pipeline_train_model import recompute_test_lat_lon_match_npz


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SUMMER_MODELS_DIR = getattr(config, "SUMMER_MODELS_DIR", MODELS_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RF_MODEL_PATH = getattr(
    config,
    "RF_SUMMER_MODEL_PATH",
    getattr(config, "RF_MODEL_PATH", os.path.join(SUMMER_MODELS_DIR, "rf_model_summer.joblib")),
)
DATA_NPZ_PATH = getattr(config, "DATA_NPZ_PATH", os.path.join(SUMMER_MODELS_DIR, "train_test_data.npz"))
OUT_CSV_PATH = os.path.join(RESULTS_DIR, "summer_subregions_performance_shap.csv")


def _assign_subregion(lat, lon):
    """Rule-based split into 4 Hunan subregions."""
    if lon < 110.5:
        return "West Hunan"
    if lat >= 28.0:
        return "North Hunan"
    if lat < 27.0:
        return "South Hunan"
    return "Central Hunan"


def _safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(valid)) < 2:
        return np.nan
    rho, _ = spearmanr(x[valid], y[valid], nan_policy="omit")
    return float(rho) if np.isfinite(rho) else np.nan


def _direction_label(v):
    if not np.isfinite(v) or np.isclose(v, 0.0):
        return "neutral"
    return "positive" if v > 0 else "negative"


def build_summer_subregions_table():
    data = np.load(DATA_NPZ_PATH)
    required = {"X_test", "y_test", "rf_pred"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(
            f"NPZ missing required arrays: {sorted(missing)}. "
            "Please re-run train_model.py first."
        )

    x_test = data["X_test"]
    y_test = data["y_test"]
    rf_pred = data["rf_pred"]

    if {"lat_test", "lon_test"}.issubset(set(data.files)):
        lat_test = data["lat_test"]
        lon_test = data["lon_test"]
    else:
        rebuilt = recompute_test_lat_lon_match_npz(DATA_NPZ_PATH)
        if rebuilt is None:
            raise ValueError(
                "NPZ missing lat_test/lon_test and auto-rebuild failed. "
                "Please re-run train_model.py."
            )
        lat_test, lon_test = rebuilt
        print(
            "[subregions_table] lat_test/lon_test missing in NPZ; "
            "auto-rebuilt row alignment from raw inputs."
        )

    n = len(y_test)
    if not (len(x_test) == len(rf_pred) == len(lat_test) == len(lon_test) == n):
        raise ValueError("X_test/y_test/rf_pred/lat_test/lon_test length mismatch.")

    feature_names = list(DEFAULT_FEATURE_NAMES)
    _, _, shap_values = get_test_shap_values(feature_names=feature_names)

    subregions = np.array([_assign_subregion(la, lo) for la, lo in zip(lat_test, lon_test)])
    ordered_regions = ["West Hunan", "Central Hunan", "North Hunan", "South Hunan"]

    rows = []
    for region in ordered_regions:
        m = subregions == region
        n_region = int(np.sum(m))
        if n_region == 0:
            rows.append(
                {
                    "Subregion": region,
                    "DEM_mean": np.nan,
                    "N": 0,
                    "R²_RF": np.nan,
                    "RMSE_RF": np.nan,
                    "Bias_RF": np.nan,
                    "imerg_ρ": np.nan,
                    "tcwv_ρ": np.nan,
                    "dem_ρ": np.nan,
                    "v10_ρ": np.nan,
                    "v10_direction": "neutral",
                }
            )
            continue

        yt = y_test[m]
        yp = rf_pred[m]
        dem_vals = x_test[m, 4]

        if n_region >= 2:
            r2 = float(r2_score(yt, yp))
        else:
            r2 = np.nan
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        bias = float(np.mean(yp - yt))

        imerg_rho = _safe_spearman(x_test[m, 0], shap_values[m, 0])
        tcwv_rho = _safe_spearman(x_test[m, 3], shap_values[m, 3])
        dem_rho = _safe_spearman(x_test[m, 4], shap_values[m, 4])
        v10_rho = _safe_spearman(x_test[m, 2], shap_values[m, 2])

        rows.append(
            {
                "Subregion": region,
                "DEM_mean": float(np.nanmean(dem_vals)),
                "N": n_region,
                "R²_RF": r2,
                "RMSE_RF": rmse,
                "Bias_RF": bias,
                "imerg_ρ": imerg_rho,
                "tcwv_ρ": tcwv_rho,
                "dem_ρ": dem_rho,
                "v10_ρ": v10_rho,
                "v10_direction": _direction_label(v10_rho),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV_PATH, index=False)
    print(f"[subregions_shap] model source: {RF_MODEL_PATH}")
    print(OUT_CSV_PATH)
    return OUT_CSV_PATH


if __name__ == "__main__":
    build_summer_subregions_table()
