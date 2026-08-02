"""Standalone SHAP core explainability figure.

Builds a two-panel figure:
1) SHAP Summary (global importance + direction)
2) SHAP Dependence for tcwv colored by DEM (local behavior + interaction)

Inputs:
- models/rf_model_summer.joblib (or config.RF_SUMMER_MODEL_PATH)
- models/train_test_data.npz

Outputs:
- results/SHAP_Summary_Dependence.png
- results/SHAP_Summary_Dependence.svg
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

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

RANDOM_STATE = 42
SHAP_PLOT_SAMPLES = 300


def make_shap_core_figure():
    feature_names_lower = list(DEFAULT_FEATURE_NAMES)
    feature_names = np.array(["IMERG", "u10", "v10", "tcwv", "DEM"])
    X_test, _, shap_values_all = get_test_shap_values(feature_names=feature_names_lower)

    n_plot = min(SHAP_PLOT_SAMPLES, X_test.shape[0])
    rng = np.random.default_rng(RANDOM_STATE)
    plot_idx = rng.choice(X_test.shape[0], size=n_plot, replace=False)
    X_plot = X_test[plot_idx]
    shap_values = shap_values_all[plot_idx]
    X_plot_df = pd.DataFrame(X_plot, columns=feature_names)

    # Global importance ranking by mean absolute SHAP.
    order = np.argsort(np.abs(shap_values).mean(axis=0))[::-1]
    shap_ordered = shap_values[:, order]
    ordered_names = feature_names[order]
    X_ordered = X_plot_df[ordered_names]

    # Do not enable constrained_layout here: shap.summary_plot internally calls
    # plt.tight_layout(), which conflicts with constrained_layout when colorbars
    # are present (matplotlib RuntimeError about mixed layout engines).
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5.6))

    # Left panel: SHAP Summary Plot.
    plt.sca(ax_left)
    shap.summary_plot(
        shap_ordered,
        X_ordered,
        feature_names=[name.lower() for name in ordered_names],
        show=False,
        sort=False,
        plot_size=None,
    )
    ax_left.set_title("SHAP Summary (Global Importance + Direction)")
    ax_left.set_xlabel("SHAP value (impact on model output)")

    # Right panel: tcwv dependence colored by DEM.
    tcwv_idx = int(np.where(feature_names == "tcwv")[0][0])
    dem_idx = int(np.where(feature_names == "DEM")[0][0])
    tcwv_vals = X_plot[:, tcwv_idx]
    dem_vals = X_plot[:, dem_idx]
    shap_tcwv = shap_values[:, tcwv_idx]

    sc = ax_right.scatter(
        tcwv_vals,
        shap_tcwv,
        c=dem_vals,
        cmap="viridis",
        s=10,
        alpha=0.75,
        edgecolors="none",
    )
    ax_right.axhline(y=0.0, color="#9E9E9E", lw=0.9, linestyle="--")
    ax_right.set_xlabel("tcwv (kg m$^{-2}$)")
    ax_right.set_ylabel("SHAP value for tcwv")
    ax_right.set_title("SHAP Dependence: tcwv colored by DEM")
    ax_right.grid(True, alpha=0.3)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    cbar = fig.colorbar(sc, ax=ax_right, pad=0.02)
    cbar.set_label("DEM elevation (m)")

    # Panel marks.
    ax_left.text(0.01, 0.98, "(a)", transform=ax_left.transAxes, va="top", ha="left")
    ax_right.text(0.01, 0.98, "(b)", transform=ax_right.transAxes, va="top", ha="left")

    png_path = os.path.join(RESULTS_DIR, "SHAP_Summary_Dependence.png")
    svg_path = os.path.join(RESULTS_DIR, "SHAP_Summary_Dependence.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(png_path)
    print(svg_path)
    print(f"[shap_summary_dependence] model source: {RF_MODEL_PATH}")


if __name__ == "__main__":
    make_shap_core_figure()
