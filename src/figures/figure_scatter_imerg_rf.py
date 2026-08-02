"""Standalone double-scatter plot: Observed vs IMERG / Observed vs RF-Full.

Reads predictions from ``models/train_test_data.npz`` and writes figure files to
``results/``.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core import project_config as config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SUMMER_MODELS_DIR = getattr(config, "SUMMER_MODELS_DIR", MODELS_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_NPZ_PATH = getattr(config, "DATA_NPZ_PATH", os.path.join(SUMMER_MODELS_DIR, "train_test_data.npz"))


def _metrics(y_true, y_pred):
    return (
        r2_score(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        float(np.mean(y_pred - y_true)),
    )


def make_double_scatter():
    data = np.load(DATA_NPZ_PATH)
    y_test = data["y_test"]
    raw_pred = data["raw_pred"]  # IMERG
    rf_pred = data["rf_pred"]  # RF-Full

    imerg_metrics = _metrics(y_test, raw_pred)
    rf_metrics = _metrics(y_test, rf_pred)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    lim_min = float(min(np.nanmin(y_test), np.nanmin(raw_pred), np.nanmin(rf_pred)))
    lim_max = float(max(np.nanmax(y_test), np.nanmax(raw_pred), np.nanmax(rf_pred)))
    lim = [lim_min, lim_max]

    axes[0].scatter(y_test, raw_pred, s=5, alpha=0.3, color="#0C5DA5", edgecolors="none")
    axes[0].plot(lim, lim, ls="--", lw=1.1, color="#9E9E9E")
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_xlabel("Observed")
    axes[0].set_ylabel("IMERG")
    axes[0].set_title("Observed vs IMERG")
    axes[0].text(
        0.05,
        0.95,
        (
            f"R2={imerg_metrics[0]:.3f}\n"
            f"RMSE={imerg_metrics[1]:.3f}\n"
            f"MAE={imerg_metrics[2]:.3f}\n"
            f"Bias={imerg_metrics[3]:.3f}\n"
            f"n={len(y_test)}"
        ),
        transform=axes[0].transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    axes[1].scatter(y_test, rf_pred, s=5, alpha=0.3, color="#00B945", edgecolors="none")
    axes[1].plot(lim, lim, ls="--", lw=1.1, color="#9E9E9E")
    axes[1].set_xlim(lim)
    axes[1].set_ylim(lim)
    axes[1].set_xlabel("Observed")
    axes[1].set_ylabel("RF-Full")
    axes[1].set_title("Observed vs RF-Full")
    axes[1].text(
        0.05,
        0.95,
        (
            f"R2={rf_metrics[0]:.3f}\n"
            f"RMSE={rf_metrics[1]:.3f}\n"
            f"MAE={rf_metrics[2]:.3f}\n"
            f"Bias={rf_metrics[3]:.3f}\n"
            f"n={len(y_test)}"
        ),
        transform=axes[1].transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    png_path = os.path.join(RESULTS_DIR, "Scatter_Observed_IMERG_RF.png")
    svg_path = os.path.join(RESULTS_DIR, "Scatter_Observed_IMERG_RF.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    make_double_scatter()
