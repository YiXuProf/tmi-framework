"""Plot a 3-panel SHAP summary figure for seasonal transfer comparison.

Panels:
- (a) RF-Summer -> Spring
- (b) RF-Summer -> Summer
- (c) RF-Spring -> Spring

Each panel uses fixed feature order and horizontal bars:
- bar length: abs(mean_shap)
- bar color: signed mean_shap direction (red positive, blue negative)

Outputs:
- results/shap_summary_triptych.png
- results/shap_summary_triptych.svg
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_ORDER = ["imerg", "tcwv", "dem", "v10", "u10"]


def _build_input_tables():
    """Build the three scenario tables from provided summary values."""
    shap_summer_to_spring = pd.DataFrame(
        {
            "feature": ["imerg", "tcwv", "dem", "v10", "u10"],
            "mean_shap": [0.937, 0.812, 0.523, -0.412, -0.156],
        }
    )

    shap_summer_to_summer = pd.DataFrame(
        {
            "feature": ["imerg", "tcwv", "dem", "v10", "u10"],
            "mean_shap": [0.951, 0.897, 0.700, -0.744, 0.096],
        }
    )

    shap_spring_to_spring = pd.DataFrame(
        {
            "feature": ["imerg", "tcwv", "dem", "v10", "u10"],
            "mean_shap": [0.942, 0.823, 0.534, -0.423, -0.148],
        }
    )

    return {
        "(a) RF-Summer applied to Spring": shap_summer_to_spring,
        "(b) RF-Summer applied to Summer": shap_summer_to_summer,
        "(c) RF-Spring applied to Spring": shap_spring_to_spring,
    }


def _prepare_panel_df(df):
    """Validate schema and enforce shared feature order."""
    required = {"feature", "mean_shap"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["feature"] = out["feature"].astype(str).str.lower()
    out = out.set_index("feature").reindex(FEATURE_ORDER).reset_index()
    if out["mean_shap"].isna().any():
        missing_features = out.loc[out["mean_shap"].isna(), "feature"].tolist()
        raise ValueError(f"Missing values for features: {missing_features}")
    out["mean_shap"] = pd.to_numeric(out["mean_shap"], errors="coerce")
    if out["mean_shap"].isna().any():
        raise ValueError("mean_shap contains non-numeric values.")
    out["impact_abs"] = np.abs(out["mean_shap"].values)
    return out


def plot_shap_summary_triptych():
    panel_tables = _build_input_tables()
    prepared = {title: _prepare_panel_df(df) for title, df in panel_tables.items()}

    signed_max = max(np.abs(df["mean_shap"]).max() for df in prepared.values())
    abs_max = max(df["impact_abs"].max() for df in prepared.values())

    cmap = plt.get_cmap("coolwarm")
    norm = TwoSlopeNorm(vmin=-signed_max, vcenter=0.0, vmax=signed_max)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.9), sharey=True, constrained_layout=True)

    y = np.arange(len(FEATURE_ORDER))
    for ax, (title, df) in zip(axes, prepared.items()):
        colors = cmap(norm(df["mean_shap"].values))
        ax.barh(y, df["impact_abs"].values, color=colors, edgecolor="white", linewidth=0.7)

        for yi, width, signed in zip(y, df["impact_abs"].values, df["mean_shap"].values):
            ax.text(width + 0.012 * abs_max, yi, f"{signed:+.3f}", va="center", ha="left", fontsize=8.4)

        ax.set_title(title)
        ax.set_xlabel("|SHAP mean impact|")
        ax.set_yticks(y)
        ax.set_yticklabels(FEATURE_ORDER)
        ax.invert_yaxis()
        ax.set_xlim(0.0, abs_max * 1.14)
        ax.grid(axis="x", alpha=0.28, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Feature")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.01, shrink=0.92)
    cbar.set_label("SHAP direction")

    png_path = os.path.join(RESULTS_DIR, "shap_summary_triptych.png")
    svg_path = os.path.join(RESULTS_DIR, "shap_summary_triptych.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(png_path)
    print(svg_path)
    return png_path, svg_path


if __name__ == "__main__":
    plot_shap_summary_triptych()
