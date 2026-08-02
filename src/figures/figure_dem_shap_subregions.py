"""Standalone DEM-SHAP curve by four Hunan subregions.

Figure elements:
- X axis: DEM value (m)
- Y axis: SHAP value of DEM
- Scatter color: subregion (Xiangxi / Xiangzhong / Xiangbei / Xiangnan)
- Smoothed trend: LOWESS-like curve per subregion

Inputs:
- models/rf_model_summer.joblib (or config.RF_SUMMER_MODEL_PATH)
- models/train_test_data.npz (lat_test/lon_test can be auto-rebuilt if missing)

Outputs:
- results/DEM_SHAP_Subregions.png
- results/DEM_SHAP_Subregions.svg
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from core import project_config as config
from core.shap_cache import DEFAULT_FEATURE_NAMES, get_test_shap_values
from pipelines.pipeline_train_model import recompute_test_lat_lon_match_npz


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SUMMER_MODELS_DIR = getattr(config, "SUMMER_MODELS_DIR", MODELS_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_NPZ_PATH = getattr(config, "DATA_NPZ_PATH", os.path.join(SUMMER_MODELS_DIR, "train_test_data.npz"))
RF_MODEL_PATH = getattr(
    config,
    "RF_SUMMER_MODEL_PATH",
    getattr(config, "RF_MODEL_PATH", os.path.join(SUMMER_MODELS_DIR, "rf_model_summer.joblib")),
)

RANDOM_STATE = 42
MAX_SHAP_SAMPLES = 15000
COLOR_MAP = {
    "West Hunan": "#0000FF",
    "Central Hunan": "#FF0000",
    "North Hunan": "#00FF00",
    "South Hunan": "#FF8C00",
}


def _assign_subregion(lat, lon):
    """Simple geographic split for Hunan into four subregions."""
    # West mountainous area.
    if lon < 110.5:
        return "West Hunan"
    # Northern plains/hills.
    if lat >= 28.0:
        return "North Hunan"
    # Southern mountainous area.
    if lat < 27.0:
        return "South Hunan"
    # Central basin/hills.
    return "Central Hunan"


def _lowess_like(x, y, frac=0.25, n_grid=180):
    """Lightweight LOWESS-like local linear smoother without extra deps."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n = x.size
    if n < 8:
        return np.array([]), np.array([])

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_grid = np.linspace(float(x.min()), float(x.max()), n_grid)

    k = max(int(np.ceil(frac * n)), 6)
    y_hat = np.full_like(x_grid, np.nan, dtype=float)

    for i, x0 in enumerate(x_grid):
        dist = np.abs(x - x0)
        h = np.partition(dist, k - 1)[k - 1]
        if h <= 0:
            h = np.max(dist)
            if h <= 0:
                y_hat[i] = float(np.mean(y))
                continue
        u = dist / h
        w = np.where(u < 1, (1 - u**3) ** 3, 0.0)  # tricube kernel
        if np.sum(w) <= 0:
            continue

        # Weighted local linear regression around x0.
        xc = x - x0
        s0 = np.sum(w)
        s1 = np.sum(w * xc)
        s2 = np.sum(w * xc * xc)
        t0 = np.sum(w * y)
        t1 = np.sum(w * y * xc)
        det = s0 * s2 - s1 * s1
        if np.abs(det) < 1e-12:
            y_hat[i] = t0 / s0
        else:
            beta0 = (t0 * s2 - t1 * s1) / det
            y_hat[i] = beta0

    keep = np.isfinite(y_hat)
    return x_grid[keep], y_hat[keep]


def plot_dem_shap_subregions():
    data = np.load(DATA_NPZ_PATH)
    required = {"X_test"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"NPZ missing required arrays: {sorted(missing)}.")

    X_test = data["X_test"]
    if {"lat_test", "lon_test"}.issubset(set(data.files)):
        lat_test = data["lat_test"]
        lon_test = data["lon_test"]
    else:
        rebuilt = recompute_test_lat_lon_match_npz(DATA_NPZ_PATH)
        if rebuilt is None:
            raise ValueError(
                "NPZ missing lat_test/lon_test and auto-rebuild failed. "
                "Please re-run train_model.py to regenerate train_test_data.npz."
            )
        lat_test, lon_test = rebuilt
        print(
            "[dem_shap_subregions] lat_test/lon_test missing in NPZ; "
            "auto-rebuilt row alignment from raw inputs."
        )
    if not (len(X_test) == len(lat_test) == len(lon_test)):
        raise ValueError("X_test, lat_test, lon_test length mismatch.")

    rng = np.random.default_rng(RANDOM_STATE)
    n_total = len(X_test)
    if n_total > MAX_SHAP_SAMPLES:
        idx = rng.choice(n_total, size=MAX_SHAP_SAMPLES, replace=False)
    else:
        idx = np.arange(n_total)

    X_plot = X_test[idx]
    lat_plot = lat_test[idx]
    lon_plot = lon_test[idx]

    _, _, shap_values_all = get_test_shap_values(feature_names=list(DEFAULT_FEATURE_NAMES))
    shap_values = shap_values_all[idx]

    dem = X_plot[:, 4]
    shap_dem = shap_values[:, 4]
    subregions = np.array([_assign_subregion(la, lo) for la, lo in zip(lat_plot, lon_plot)])

    ordered_regions = ["North Hunan", "South Hunan", "West Hunan", "Central Hunan"]

    fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
    for i, region in enumerate(ordered_regions):
        m = subregions == region
        if np.sum(m) == 0:
            continue
        alpha = 0.6 if region == "Central Hunan" else 0.35
        size = 15 if region == "Central Hunan" else 8
        zorder = 10 if region == "Central Hunan" else (2 + i)
        ax.scatter(
            dem[m],
            shap_dem[m],
            s=size,
            alpha=alpha,
            color=COLOR_MAP[region],
            edgecolors="none",
            zorder=zorder,
            label=f"{region} (n={int(np.sum(m))})",
        )
        xs, ys = _lowess_like(dem[m], shap_dem[m], frac=0.25, n_grid=180)
        if xs.size > 0:
            ax.plot(xs, ys, color=COLOR_MAP[region], lw=2.2, zorder=zorder + 0.1)

    ax.axhline(0.0, color="#9E9E9E", lw=1.0, ls="--")
    ax.set_xlabel("DEM (m)")
    ax.set_ylabel("SHAP value for DEM")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False)

    png_path = os.path.join(RESULTS_DIR, "DEM_SHAP_Subregions.png")
    svg_path = os.path.join(RESULTS_DIR, "DEM_SHAP_Subregions.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(png_path)
    print(svg_path)
    print(f"[dem_shap_subregions] model source: {RF_MODEL_PATH}")


if __name__ == "__main__":
    plot_dem_shap_subregions()
