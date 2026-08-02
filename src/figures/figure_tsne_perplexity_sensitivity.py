"""Perplexity sensitivity test for Fig.6 t-SNE heavy-rain separation.

Goal:
- Reproduce the Fig.6 panel-(a) t-SNE setup with a fixed sample.
- Test perplexity in {5, 10, 30, 50, 100}.
- Quantify whether heavy-rain cluster (>=50 mm/d) separation remains stable.

Outputs:
- results/tsne_perplexity_sensitivity_panel.png
- results/tsne_perplexity_sensitivity_panel.svg
- results/tsne_perplexity_sensitivity_metrics.csv
- results/tsne_perplexity_sensitivity_metrics.md
"""

from __future__ import annotations

import inspect
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from core import project_config as config


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SUMMER_MODELS_DIR = getattr(config, "SUMMER_MODELS_DIR", MODELS_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_NPZ_PATH = getattr(
    config, "DATA_NPZ_PATH", os.path.join(SUMMER_MODELS_DIR, "train_test_data.npz")
)

HEAVY_THRESHOLD = 50.0
MAX_NON_HEAVY_POINTS = 300
RANDOM_STATE = int(getattr(config, "RANDOM_STATE", 42))
PERPLEXITIES = [5, 10, 30, 50, 100]
NON_HEAVY_COLOR = "#0000FF"
HEAVY_COLOR = "#FF0000"
NN_K = 10


def _load_test_samples() -> tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(DATA_NPZ_PATH):
        raise FileNotFoundError(f"NPZ not found: {DATA_NPZ_PATH}")
    data = np.load(DATA_NPZ_PATH)
    required = {"X_test", "y_test"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"NPZ missing arrays: {sorted(missing)}")
    x_test = np.asarray(data["X_test"], dtype=float)
    y_test = np.asarray(data["y_test"], dtype=float).ravel()
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError("X_test and y_test length mismatch.")
    return x_test, y_test


def _subsample_tsne_points(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def _run_tsne(x_scaled: np.ndarray, perplexity: int) -> np.ndarray:
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
    return TSNE(**tsne_kwargs).fit_transform(x_scaled)


def _centroid_separation_ratio(emb: np.ndarray, heavy_mask: np.ndarray) -> float:
    non_heavy_mask = ~heavy_mask
    if np.sum(heavy_mask) < 2 or np.sum(non_heavy_mask) < 2:
        return np.nan
    emb_h = emb[heavy_mask]
    emb_n = emb[non_heavy_mask]
    mu_h = np.mean(emb_h, axis=0)
    mu_n = np.mean(emb_n, axis=0)
    dist = float(np.linalg.norm(mu_h - mu_n))
    disp_h = float(np.sqrt(np.mean(np.sum((emb_h - mu_h) ** 2, axis=1))))
    disp_n = float(np.sqrt(np.mean(np.sum((emb_n - mu_n) ** 2, axis=1))))
    return dist / (disp_h + disp_n + 1e-12)


def _nearest_neighbor_purity(emb: np.ndarray, heavy_mask: np.ndarray, k: int = NN_K) -> float:
    n = emb.shape[0]
    heavy_idx = np.where(heavy_mask)[0]
    if n <= 2 or heavy_idx.size == 0:
        return np.nan
    k_eff = int(min(k + 1, n))
    if k_eff <= 1:
        return np.nan
    nn = NearestNeighbors(n_neighbors=k_eff)
    nn.fit(emb)
    neigh = nn.kneighbors(emb[heavy_idx], return_distance=False)
    neigh = neigh[:, 1:]
    if neigh.size == 0:
        return np.nan
    return float(np.mean(heavy_mask[neigh]))


def _safe_cv(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    mean = float(np.mean(values))
    if np.isclose(mean, 0.0):
        return np.nan
    return float(np.std(values, ddof=1) / abs(mean)) if values.size >= 2 else 0.0


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def run_tsne_perplexity_sensitivity() -> tuple[str, str, str, str]:
    x_test, y_test = _load_test_samples()
    x_tsne, y_tsne = _subsample_tsne_points(x_test, y_test)
    heavy_mask = y_tsne >= HEAVY_THRESHOLD
    non_heavy_mask = ~heavy_mask
    labels = heavy_mask.astype(int)

    x_scaled = StandardScaler().fit_transform(x_tsne)
    n_points = x_scaled.shape[0]
    if n_points < 5:
        raise ValueError("Too few points for t-SNE sensitivity test.")

    valid_perplexities = [p for p in PERPLEXITIES if p < n_points]
    if len(valid_perplexities) == 0:
        raise ValueError(
            f"No valid perplexity in {PERPLEXITIES} for n={n_points}. "
            "Need perplexity < number of samples."
        )

    rows: list[dict[str, float | int | str | bool]] = []
    embeddings: dict[int, np.ndarray] = {}
    for p in valid_perplexities:
        emb = _run_tsne(x_scaled, perplexity=p)
        embeddings[p] = emb

        sil = np.nan
        if np.sum(heavy_mask) >= 2 and np.sum(non_heavy_mask) >= 2:
            sil = float(silhouette_score(emb, labels))
        sep_ratio = _centroid_separation_ratio(emb, heavy_mask)
        nn_purity = _nearest_neighbor_purity(emb, heavy_mask, k=NN_K)

        rows.append(
            {
                "perplexity": int(p),
                "N_total": int(n_points),
                "N_heavy": int(np.sum(heavy_mask)),
                "N_non_heavy": int(np.sum(non_heavy_mask)),
                "silhouette_heavy_vs_nonheavy": sil,
                "centroid_separation_ratio": sep_ratio,
                "nearest_neighbor_purity_k10": nn_purity,
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("perplexity").reset_index(drop=True)
    sil_vals = metrics_df["silhouette_heavy_vs_nonheavy"].to_numpy(dtype=float)
    sep_vals = metrics_df["centroid_separation_ratio"].to_numpy(dtype=float)
    pur_vals = metrics_df["nearest_neighbor_purity_k10"].to_numpy(dtype=float)

    silhouette_positive_all = (
        bool(np.all(sil_vals[np.isfinite(sil_vals)] > 0.0)) if np.any(np.isfinite(sil_vals)) else False
    )
    stability_score = np.nanmean(
        np.array(
            [
                _safe_cv(sil_vals),
                _safe_cv(sep_vals),
                _safe_cv(pur_vals),
            ],
            dtype=float,
        )
    )
    stable_pattern = bool(
        silhouette_positive_all and np.isfinite(stability_score) and stability_score <= 0.25
    )

    summary = pd.DataFrame(
        [
            {
                "tested_perplexities": ",".join(str(v) for v in metrics_df["perplexity"].tolist()),
                "silhouette_min": float(np.nanmin(sil_vals)) if np.any(np.isfinite(sil_vals)) else np.nan,
                "silhouette_max": float(np.nanmax(sil_vals)) if np.any(np.isfinite(sil_vals)) else np.nan,
                "centroid_separation_ratio_min": float(np.nanmin(sep_vals)) if np.any(np.isfinite(sep_vals)) else np.nan,
                "centroid_separation_ratio_max": float(np.nanmax(sep_vals)) if np.any(np.isfinite(sep_vals)) else np.nan,
                "nearest_neighbor_purity_min": float(np.nanmin(pur_vals)) if np.any(np.isfinite(pur_vals)) else np.nan,
                "nearest_neighbor_purity_max": float(np.nanmax(pur_vals)) if np.any(np.isfinite(pur_vals)) else np.nan,
                "stability_cv_mean": float(stability_score) if np.isfinite(stability_score) else np.nan,
                "stable_pattern_flag": stable_pattern,
            }
        ]
    )

    # 2x3 layout: first 5 panels used, last one for textual summary.
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.2), constrained_layout=True)
    flat_axes = axes.ravel()
    for i, p in enumerate(valid_perplexities):
        ax = flat_axes[i]
        emb = embeddings[p]
        if np.any(heavy_mask):
            ax.scatter(
                emb[heavy_mask, 0],
                emb[heavy_mask, 1],
                s=14,
                c=HEAVY_COLOR,
                alpha=0.75,
                edgecolors="none",
                label=f"Heavy (>= {HEAVY_THRESHOLD:.0f} mm/d)",
            )
        ax.scatter(
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
        row = metrics_df.loc[metrics_df["perplexity"] == p].iloc[0]
        ax.set_title(
            f"perplexity={p} | sil={row['silhouette_heavy_vs_nonheavy']:.3f} | "
            f"sep={row['centroid_separation_ratio']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused subplot slots except summary panel.
    if len(valid_perplexities) < 5:
        for j in range(len(valid_perplexities), 5):
            flat_axes[j].axis("off")

    ax_info = flat_axes[5]
    ax_info.axis("off")
    s = summary.iloc[0]
    info_lines = [
        "t-SNE perplexity sensitivity",
        f"Tested: {s['tested_perplexities']}",
        f"Silhouette range: {s['silhouette_min']:.3f} to {s['silhouette_max']:.3f}",
        (
            "Centroid-separation range: "
            f"{s['centroid_separation_ratio_min']:.3f} to {s['centroid_separation_ratio_max']:.3f}"
        ),
        (
            "Nearest-neighbor purity range: "
            f"{s['nearest_neighbor_purity_min']:.3f} to {s['nearest_neighbor_purity_max']:.3f}"
        ),
        f"Mean CV (3 metrics): {s['stability_cv_mean']:.3f}",
        f"Stable pattern flag: {bool(s['stable_pattern_flag'])}",
    ]
    ax_info.text(0.0, 0.98, "\n".join(info_lines), ha="left", va="top", fontsize=11)

    handles, labels_legend = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_legend, loc="lower center", ncol=2, frameon=False)

    panel_png = os.path.join(RESULTS_DIR, "tsne_perplexity_sensitivity_panel.png")
    panel_svg = os.path.join(RESULTS_DIR, "tsne_perplexity_sensitivity_panel.svg")
    metrics_csv = os.path.join(RESULTS_DIR, "tsne_perplexity_sensitivity_metrics.csv")
    metrics_md = os.path.join(RESULTS_DIR, "tsne_perplexity_sensitivity_metrics.md")

    fig.savefig(panel_png, dpi=300, bbox_inches="tight")
    fig.savefig(panel_svg, bbox_inches="tight")
    plt.close(fig)

    out_df = metrics_df.copy()
    out_df.to_csv(metrics_csv, index=False)
    with open(metrics_md, "w", encoding="utf-8") as f:
        f.write("# t-SNE perplexity sensitivity (Fig.6 panel-a)\n\n")
        f.write("## Perplexity-wise metrics\n\n")
        f.write(_to_markdown_table(out_df.round(6)))
        f.write("\n\n## Stability summary\n\n")
        f.write(_to_markdown_table(summary.round(6)))
        f.write("\n\n")
        if stable_pattern:
            f.write(
                "- Conclusion: Heavy-rain cluster separation pattern is stable across tested perplexities.\n"
            )
        else:
            f.write(
                "- Conclusion: Heavy-rain cluster separation pattern shows notable sensitivity to perplexity.\n"
            )
        f.write(
            "- Note: Stability flag uses silhouette positivity and mean CV <= 0.25 across three separation metrics.\n"
        )

    print(f"[tsne-sensitivity] NPZ source: {DATA_NPZ_PATH}")
    print(panel_png)
    print(panel_svg)
    print(metrics_csv)
    print(metrics_md)
    return panel_png, panel_svg, metrics_csv, metrics_md


if __name__ == "__main__":
    run_tsne_perplexity_sensitivity()
