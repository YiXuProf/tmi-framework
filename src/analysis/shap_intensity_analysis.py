"""Standalone SHAP ranking comparison by rainfall intensity.

Reads:
- models/rf_model_summer.joblib (or config.RF_SUMMER_MODEL_PATH)
- models/train_test_data.npz

Writes:
- results/shap_by_intensity.csv
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from core import project_config as config
from core.shap_cache import DEFAULT_FEATURE_NAMES, get_test_shap_values

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


def _safe_spearman(x, y):
    if x.size < 2 or y.size < 2:
        return np.nan
    rho, _ = spearmanr(x, y, nan_policy="omit")
    return float(rho) if np.isfinite(rho) else np.nan


def _direction_symbol(rho):
    if np.isnan(rho) or np.isclose(rho, 0.0):
        return "0"
    return "+" if rho > 0 else "-"


def build_shap_by_intensity():
    feature_names = list(DEFAULT_FEATURE_NAMES)
    X_test, y_test, shap_values = get_test_shap_values(feature_names=feature_names)
    X_df = pd.DataFrame(X_test, columns=feature_names)

    classes = {
        "Light SHAP ρ": (y_test >= 0.1) & (y_test < 10.0),
        "Moderate SHAP ρ": (y_test >= 10.0) & (y_test < 25.0),
        "Heavy SHAP ρ": (y_test >= 25.0) & (y_test < 50.0),
        "Torrential SHAP ρ": y_test >= 50.0,
    }

    rows = []
    for feat_idx, feat in enumerate(feature_names):
        row = {"Feature": feat}
        class_rhos = []
        for col_name, mask in classes.items():
            rho = _safe_spearman(
                X_df.loc[mask, feat].values,
                shap_values[mask, feat_idx],
            )
            row[col_name] = rho
            if np.isfinite(rho):
                class_rhos.append(rho)

        direction_rho = np.nan if len(class_rhos) == 0 else float(np.nanmean(class_rhos))
        row["Direction"] = _direction_symbol(direction_rho)
        row["_rank_score"] = (
            np.nanmean(np.abs(class_rhos)) if len(class_rhos) > 0 else -np.inf
        )
        rows.append(row)

    out_df = (
        pd.DataFrame(rows)
        .sort_values("_rank_score", ascending=False)
        .drop(columns=["_rank_score"])
        .reset_index(drop=True)
    )

    out_csv = os.path.join(RESULTS_DIR, "shap_by_intensity.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"[shap_by_intensity] model source: {RF_MODEL_PATH}")
    print(out_csv)
    return out_csv


if __name__ == "__main__":
    build_shap_by_intensity()
