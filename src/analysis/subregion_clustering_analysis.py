"""Objective subregion validation by data-driven clustering.

Use K-means (default) or a natural-breaks style split to test whether
four subregions (NW/NE/SW/Central) are data-supported rather than subjective.

Outputs:
- Subregion_Objective_Clustering_<method>_KScan.csv
- Subregion_Objective_Clustering_<method>_K4_Summary.csv
- Subregion_Objective_Clustering_<method>_Assignments.csv
- Subregion_Objective_Clustering_<method>_Report.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

def _ensure_src_on_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_src_on_path()

from core import project_config as config
from pipelines.pipeline_train_model import recompute_test_lat_lon_match_npz


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SUMMER_MODELS_DIR = getattr(config, "SUMMER_MODELS_DIR", os.path.join(MODELS_DIR, "summer"))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_NPZ_PATH = getattr(config, "DATA_NPZ_PATH", os.path.join(SUMMER_MODELS_DIR, "train_test_data.npz"))
RANDOM_STATE = int(getattr(config, "RANDOM_STATE", 42))

FEATURE_NAMES = ["imerg", "u10", "v10", "tcwv", "dem"]
SCAN_K_VALUES = list(range(2, 9))
ZONE_ORDER = ["NW", "NE", "SW", "Central"]
ZONE_COLOR = {
    "NW": "#2E8B57",
    "NE": "#4169E1",
    "SW": "#FF8C00",
    "Central": "#8B5A96",
}


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _load_samples() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print(f"[subregion-cluster] loading NPZ: {DATA_NPZ_PATH}", flush=True)
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
        raise ValueError("X_test/y_test length mismatch.")
    if x_test.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Unexpected feature count: got {x_test.shape[1]}, expected {len(FEATURE_NAMES)}."
        )

    if {"lat_test", "lon_test"}.issubset(set(data.files)):
        lat_test = np.asarray(data["lat_test"], dtype=float).ravel()
        lon_test = np.asarray(data["lon_test"], dtype=float).ravel()
    else:
        rebuilt = recompute_test_lat_lon_match_npz(DATA_NPZ_PATH)
        if rebuilt is None:
            raise ValueError(
                "NPZ missing lat_test/lon_test and auto-rebuild failed. Please re-run train_model.py."
            )
        lat_test, lon_test = rebuilt
        print(
            "[subregion-cluster] lat_test/lon_test missing in NPZ; "
            "auto-rebuilt row alignment from raw inputs."
        )

    if not (len(lat_test) == len(lon_test) == x_test.shape[0]):
        raise ValueError("lat/lon length mismatch with X_test.")
    return x_test, y_test, lat_test, lon_test


def _build_feature_df(
    x_test: np.ndarray,
    y_test: np.ndarray,
    lat_test: np.ndarray,
    lon_test: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat": lat_test,
            "lon": lon_test,
            "dem": x_test[:, 4],
            "imerg": x_test[:, 0],
            "obs": y_test,
        }
    )


def _scaled_matrix(feature_df: pd.DataFrame) -> np.ndarray:
    cols = ["lat", "lon", "dem", "imerg", "obs"]
    return StandardScaler().fit_transform(feature_df[cols].values)


def _kmeans_scan(x_matrix: np.ndarray) -> tuple[pd.DataFrame, int]:
    rows = []
    for k in SCAN_K_VALUES:
        print(f"[subregion-cluster] K-scan running K={k}", flush=True)
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(x_matrix)
        rows.append(
            {
                "K": int(k),
                "silhouette": float(silhouette_score(x_matrix, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(x_matrix, labels)),
                "davies_bouldin": float(davies_bouldin_score(x_matrix, labels)),
                "inertia": float(model.inertia_),
            }
        )
    k_df = pd.DataFrame(rows)
    k_df["rank_sil"] = k_df["silhouette"].rank(ascending=False, method="min")
    k_df["rank_ch"] = k_df["calinski_harabasz"].rank(ascending=False, method="min")
    k_df["rank_db"] = k_df["davies_bouldin"].rank(ascending=True, method="min")
    k_df["rank_sum"] = k_df["rank_sil"] + k_df["rank_ch"] + k_df["rank_db"]
    k_df["is_K4"] = k_df["K"] == 4
    best_k = int(k_df.sort_values(["rank_sum", "rank_sil", "rank_ch"]).iloc[0]["K"])
    return k_df.sort_values("K").reset_index(drop=True), best_k


def _fit_kmeans_labels(x_matrix: np.ndarray, n_clusters: int = 4) -> np.ndarray:
    print(f"[subregion-cluster] fitting final KMeans (K={n_clusters})", flush=True)
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=30)
    return model.fit_predict(x_matrix).astype(int)


def _fit_natural_breaks_labels(feature_df: pd.DataFrame) -> np.ndarray:
    dem = feature_df["dem"].to_numpy(dtype=float)
    obs = feature_df["obs"].to_numpy(dtype=float)

    try:
        import jenkspy  # type: ignore

        dem_breaks = jenkspy.jenks_breaks(dem, n_classes=2)
        obs_breaks = jenkspy.jenks_breaks(obs, n_classes=2)
        dem_cut = float(dem_breaks[1])
        obs_cut = float(obs_breaks[1])
        method_note = "jenks"
    except Exception:
        dem_cut = float(np.nanmedian(dem))
        obs_cut = float(np.nanmedian(obs))
        method_note = "median-fallback"

    dem_bin = (dem > dem_cut).astype(int)
    obs_bin = (obs > obs_cut).astype(int)
    code = dem_bin * 2 + obs_bin  # 0,1,2,3
    unique = np.unique(code)
    if unique.size < 4:
        # Keep script robust when natural breaks collapse bins.
        code = _fit_kmeans_labels(_scaled_matrix(feature_df), n_clusters=4)
        method_note += "+kmeans-fallback"

    print(f"[subregion-cluster] natural-breaks method: {method_note}")
    _, relabeled = np.unique(code, return_inverse=True)
    return relabeled.astype(int)


def _map_to_geo_zone(labels: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> Dict[int, str]:
    cluster_ids = sorted(np.unique(labels).tolist())
    if len(cluster_ids) != 4:
        return {cid: f"Cluster_{cid}" for cid in cluster_ids}

    centroids = []
    for cid in cluster_ids:
        m = labels == cid
        centroids.append([float(np.mean(lat[m])), float(np.mean(lon[m]))])
    centroids = np.asarray(centroids, dtype=float)

    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
    lat_mid, lon_mid = float(np.median(lat)), float(np.median(lon))
    targets = np.asarray(
        [
            [lat_max, lon_min],  # NW
            [lat_max, lon_max],  # NE
            [lat_min, lon_min],  # SW
            [lat_mid, lon_mid],  # Central
        ],
        dtype=float,
    )

    cost = np.sqrt(((centroids[:, None, :] - targets[None, :, :]) ** 2).sum(axis=2))
    row_ind, col_ind = linear_sum_assignment(cost)
    names = ["NW", "NE", "SW", "Central"]
    return {int(cluster_ids[r]): names[int(c)] for r, c in zip(row_ind, col_ind)}


def _cluster_quality_metrics(x_matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if np.unique(labels).size < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    return {
        "silhouette": float(silhouette_score(x_matrix, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(x_matrix, labels)),
        "davies_bouldin": float(davies_bouldin_score(x_matrix, labels)),
    }


def run_subregion_objective_clustering(method: str = "kmeans") -> tuple[str, str, str, str]:
    method = method.lower().strip()
    if method not in {"kmeans", "natural"}:
        raise ValueError("method must be one of: kmeans, natural")

    out_kscan_csv = os.path.join(RESULTS_DIR, f"Subregion_Objective_Clustering_{method}_KScan.csv")
    out_k4_csv = os.path.join(RESULTS_DIR, f"Subregion_Objective_Clustering_{method}_K4_Summary.csv")
    out_assign_csv = os.path.join(RESULTS_DIR, f"Subregion_Objective_Clustering_{method}_Assignments.csv")
    out_md = os.path.join(RESULTS_DIR, f"Subregion_Objective_Clustering_{method}_Report.md")

    print(f"[subregion-cluster] start, method={method}", flush=True)
    x_test, y_test, lat_test, lon_test = _load_samples()
    feature_df = _build_feature_df(x_test, y_test, lat_test, lon_test)
    x_matrix = _scaled_matrix(feature_df)
    print(f"[subregion-cluster] samples={len(feature_df)}, features_for_cluster=5", flush=True)

    kscan_df, best_k = _kmeans_scan(x_matrix)
    kscan_df.to_csv(out_kscan_csv, index=False)
    k4_row = kscan_df.loc[kscan_df["K"] == 4].iloc[0]

    if method == "kmeans":
        labels = _fit_kmeans_labels(x_matrix, n_clusters=4)
    else:
        labels = _fit_natural_breaks_labels(feature_df)

    name_map = _map_to_geo_zone(labels, lat_test, lon_test)
    cluster_name = np.array([name_map[int(v)] for v in labels], dtype=object)

    assign_df = feature_df.copy()
    assign_df["cluster_id"] = labels
    assign_df["cluster_name"] = cluster_name
    assign_df.to_csv(out_assign_csv, index=False)
    print("[subregion-cluster] assignment CSV written", flush=True)

    rows = []
    for cid in sorted(np.unique(labels).tolist()):
        m = labels == cid
        zname = name_map.get(int(cid), f"Cluster_{cid}")
        rows.append(
            {
                "cluster_id": int(cid),
                "zone": zname,
                "N": int(np.sum(m)),
                "lat_mean": float(np.mean(lat_test[m])) if np.any(m) else np.nan,
                "lon_mean": float(np.mean(lon_test[m])) if np.any(m) else np.nan,
                "dem_mean": float(np.mean(feature_df.loc[m, "dem"])) if np.any(m) else np.nan,
                "obs_mean": float(np.mean(feature_df.loc[m, "obs"])) if np.any(m) else np.nan,
                "imerg_mean": float(np.mean(feature_df.loc[m, "imerg"])) if np.any(m) else np.nan,
            }
        )
    k4_df = pd.DataFrame(rows)
    k4_df["zone"] = pd.Categorical(k4_df["zone"], categories=ZONE_ORDER, ordered=True)
    k4_df = k4_df.sort_values(["zone", "cluster_id"]).reset_index(drop=True)
    k4_df["zone"] = k4_df["zone"].astype(str)
    k4_df.to_csv(out_k4_csv, index=False)
    print("[subregion-cluster] K4 summary CSV written", flush=True)

    method_metrics = _cluster_quality_metrics(x_matrix, labels)
    k4_optimal = bool(best_k == 4)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Objective basis for subregion partition\n\n")
        f.write(f"- Clustering method used for assignments: `{method}`\n")
        f.write("- K scan uses K-means over K=2..8 for objective model-order check.\n\n")
        f.write("## K scan metrics (K-means)\n\n")
        f.write(_to_markdown_table(kscan_df.round(6)))
        f.write("\n\n## Assignment summary (4 clusters)\n\n")
        f.write(_to_markdown_table(k4_df.round(6)))
        f.write("\n\n## Conclusion\n\n")
        f.write(f"- Best K from K-means rank aggregation: K={best_k}.\n")
        f.write(
            "- K=4 (K-means scan) metrics: "
            f"silhouette={k4_row['silhouette']:.6f}, "
            f"Calinski-Harabasz={k4_row['calinski_harabasz']:.6f}, "
            f"Davies-Bouldin={k4_row['davies_bouldin']:.6f}.\n"
        )
        f.write(
            f"- Current assignment ({method}) metrics: "
            f"silhouette={method_metrics['silhouette']:.6f}, "
            f"Calinski-Harabasz={method_metrics['calinski_harabasz']:.6f}, "
            f"Davies-Bouldin={method_metrics['davies_bouldin']:.6f}.\n"
        )
        if k4_optimal:
            f.write("- Interpretation: Four-zone partition is data-supported as the optimal choice.\n")
        else:
            f.write(
                "- Interpretation: Four-zone partition is plausible but not globally optimal by K-means scan.\n"
            )
        f.write("- Zone names are centroid-matched to west/north/south/central geographic templates and labeled as NW/NE/SW/Central.\n")

    print(f"[subregion-cluster] NPZ source: {DATA_NPZ_PATH}")
    print(out_kscan_csv)
    print(out_k4_csv)
    print(out_assign_csv)
    print(out_md)
    return out_kscan_csv, out_k4_csv, out_assign_csv, out_md


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Objective subregion validation with K-means or natural-breaks style clustering."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="kmeans",
        choices=["kmeans", "natural"],
        help="Assignment method for the 4-zone output.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_subregion_objective_clustering(method=args.method)
