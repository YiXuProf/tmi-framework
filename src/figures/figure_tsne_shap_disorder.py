"""Two-panel figure: t-SNE heavy-rain highlight + SHAP disorder vs intensity.

Panel (a): t-SNE feature space
- Non-heavy (<50 mm/d): pure blue
- Heavy (>=50 mm/d): pure red
- Sampling: keep 300 non-heavy + all heavy

Panel (b): SHAP disorder vs rainfall intensity
- Disorder metric: SHAP standard deviation (sigma)
- Bins: Light / Moderate / Heavy / Torrential
- Curves: imerg / tcwv / dem / v10

Outputs (no tables):
- results/tsne_shap_disorder_panel.png
- results/tsne_shap_disorder_panel.svg
"""

import inspect
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from core import project_config as config
from core.shap_cache import DEFAULT_FEATURE_NAMES, get_test_shap_values


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SUMMER_MODELS_DIR = getattr(config, "SUMMER_MODELS_DIR", MODELS_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_NPZ_PATH = getattr(
    config, "DATA_NPZ_PATH", os.path.join(SUMMER_MODELS_DIR, "train_test_data.npz")
)
RF_MODEL_PATH = getattr(
    config,
    "RF_SUMMER_MODEL_PATH",
    getattr(config, "RF_MODEL_PATH", os.path.join(SUMMER_MODELS_DIR, "rf_model_summer.joblib")),
)

HEAVY_THRESHOLD = 50.0
MAX_NON_HEAVY_POINTS = 300
RANDOM_STATE = int(getattr(config, "RANDOM_STATE", 42))
NON_HEAVY_COLOR = "#0000FF"
HEAVY_COLOR = "#FF0000"

INTENSITY_BINS = [
    ("Light", 0.1, 10.0),
    ("Moderate", 10.0, 25.0),
    ("Heavy", 25.0, 50.0),
    ("Torrential", 50.0, np.inf),
]
FEATURE_ORDER = ["imerg", "tcwv", "dem", "v10"]
FEATURE_COLORS = {
    "imerg": "#1f77b4",
    "tcwv": "#2ca02c",
    "dem": "#ff7f0e",
    "v10": "#9467bd",
}


def _load_test_samples():
    x_test, y_test, _ = get_test_shap_values(feature_names=list(DEFAULT_FEATURE_NAMES))
    return x_test, y_test


def _subsample_tsne_points(x, y):
    rng = np.random.default_rng(RANDOM_STATE)
    heavy_mask = y >= HEAVY_THRESHOLD
    heavy_idx = np.where(heavy_mask)[0]
    non_heavy_idx = np.where(~heavy_mask)[0]
    if non_heavy_idx.size > MAX_NON_HEAVY_POINTS:
        non_heavy_keep = rng.choice(non_heavy_idx, size=MAX_NON_HEAVY_POINTS, replace=False)
    else:
        non_heavy_keep = non_heavy_idx
    keep_idx = np.concatenate([non_heavy_keep, heavy_idx])
    rng.shuffle(keep_idx)
    return x[keep_idx], y[keep_idx]


def _compute_tsne_embedding(x):
    x_scaled = StandardScaler().fit_transform(x)
    n = x_scaled.shape[0]
    if n < 50:
        perplexity = 10
    else:
        perplexity = int(min(40, max(20, (n - 1) // 3)))
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "learning_rate": "auto",
        "init": "pca",
        "random_state": RANDOM_STATE,
        "verbose": 0,
    }
    sig = inspect.signature(TSNE.__init__)
    if "max_iter" in sig.parameters:
        tsne_kwargs["max_iter"] = 1500
    else:
        tsne_kwargs["n_iter"] = 1500
    emb = TSNE(**tsne_kwargs).fit_transform(x_scaled)
    return emb


def _compute_shap_sigma_by_bin(shap_values, y_test):
    shap_values = np.asarray(shap_values, dtype=float)
    idx_map = {name: i for i, name in enumerate(DEFAULT_FEATURE_NAMES)}

    out = {}
    for feat in FEATURE_ORDER:
        feat_vals = []
        for _, low, high in INTENSITY_BINS:
            if np.isinf(high):
                mask = y_test >= low
            else:
                mask = (y_test >= low) & (y_test < high)
            sv = shap_values[mask, idx_map[feat]]
            sv = sv[np.isfinite(sv)]
            if sv.size >= 2:
                feat_vals.append(float(np.nanstd(sv, ddof=1)))
            else:
                feat_vals.append(np.nan)
        out[feat] = feat_vals
    return out


def make_tsne_shap_disorder_panel():
    x_test, y_test, shap_values = get_test_shap_values(feature_names=list(DEFAULT_FEATURE_NAMES))

    x_tsne, y_tsne = _subsample_tsne_points(x_test, y_test)
    emb = _compute_tsne_embedding(x_tsne)
    heavy_mask = y_tsne >= HEAVY_THRESHOLD
    non_heavy_mask = ~heavy_mask

    sigma_curves = _compute_shap_sigma_by_bin(shap_values, y_test)
    class_order = [name for name, _, _ in INTENSITY_BINS]
    x_pos = np.arange(len(class_order))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.2), constrained_layout=True)

    if np.any(heavy_mask):
        ax1.scatter(
            emb[heavy_mask, 0],
            emb[heavy_mask, 1],
            s=14,
            c=HEAVY_COLOR,
            alpha=0.75,
            edgecolors="none",
            label=f"Heavy (>= {HEAVY_THRESHOLD:.0f} mm/d)",
        )
    ax1.scatter(
        emb[non_heavy_mask, 0],
        emb[non_heavy_mask, 1],
        s=18,
        c=NON_HEAVY_COLOR,
        alpha=1.0,
        edgecolors="white",
        linewidths=0.25,
        label=f"Non-heavy (< {HEAVY_THRESHOLD:.0f} mm/d)",
        zorder=4,
    )
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    ax1.grid(True, alpha=0.2, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False, loc="best")
    ax1.text(0.01, 0.99, "(a)", transform=ax1.transAxes, ha="left", va="top", fontweight="bold")

    for feat in FEATURE_ORDER:
        ax2.plot(
            x_pos,
            sigma_curves[feat],
            marker="o",
            ms=6,
            lw=2.0,
            color=FEATURE_COLORS[feat],
            label=feat,
        )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(class_order)
    ax2.set_xlabel("Rainfall Intensity Class")
    ax2.set_ylabel("SHAP Standard Deviation, σ")
    ax2.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(title="Feature", frameon=False, loc="best")
    ax2.text(0.01, 0.99, "(b)", transform=ax2.transAxes, ha="left", va="top", fontweight="bold")

    png_path = os.path.join(RESULTS_DIR, "tsne_shap_disorder_panel.png")
    svg_path = os.path.join(RESULTS_DIR, "tsne_shap_disorder_panel.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[panel] NPZ source: {DATA_NPZ_PATH}")
    print(f"[panel] model source: {RF_MODEL_PATH}")
    print(png_path)
    print(svg_path)
    return png_path, svg_path


if __name__ == "__main__":
    make_tsne_shap_disorder_panel()
