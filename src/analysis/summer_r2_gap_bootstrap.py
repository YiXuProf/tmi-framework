"""Bootstrap CI for Central-South summer R2 gap.

Purpose:
- Quantify uncertainty of the Central Hunan vs South Hunan RF R2 gap.
- Validate whether the reported reference gap is statistically significant.

Inputs:
- models/summer/train_test_data.npz (or config.DATA_NPZ_PATH)
  required arrays: y_test, rf_pred
  preferred arrays: lat_test, lon_test

Outputs:
- results/summer_central_south_r2_gap_bootstrap_ci.csv
- results/summer_central_south_r2_gap_bootstrap_ci.md
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from core import project_config as config
from pipelines.pipeline_train_model import recompute_test_lat_lon_match_npz


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_NPZ_PATH = getattr(config, "DATA_NPZ_PATH", os.path.join(BASE_DIR, "models", "summer", "train_test_data.npz"))
TABLE4_CSV_PATH = os.path.join(RESULTS_DIR, "summer_subregions_performance_shap.csv")
OUT_CSV_PATH = os.path.join(RESULTS_DIR, "summer_central_south_r2_gap_bootstrap_ci.csv")
OUT_MD_PATH = os.path.join(RESULTS_DIR, "summer_central_south_r2_gap_bootstrap_ci.md")


def _assign_subregion(lat: float, lon: float) -> str:
    """Rule-based split into 4 Hunan subregions (same as analysis pipeline)."""
    if lon < 110.5:
        return "West Hunan"
    if lat >= 28.0:
        return "North Hunan"
    if lat < 27.0:
        return "South Hunan"
    return "Central Hunan"


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _bootstrap_r2_gap(
    y_central: np.ndarray,
    p_central: np.ndarray,
    y_south: np.ndarray,
    p_south: np.ndarray,
    n_boot: int,
    random_seed: int,
) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(random_seed)
    n_c = y_central.size
    n_s = y_south.size
    if n_c < 2 or n_s < 2:
        raise ValueError("Central or South sample size < 2, cannot compute bootstrap R2 CI.")

    gaps = np.empty(n_boot, dtype=float)
    valid_count = 0
    for _ in range(n_boot):
        idx_c = rng.integers(0, n_c, size=n_c)
        idx_s = rng.integers(0, n_s, size=n_s)
        r2_c = r2_score(y_central[idx_c], p_central[idx_c])
        r2_s = r2_score(y_south[idx_s], p_south[idx_s])
        gap = float(r2_c - r2_s)
        if np.isfinite(gap):
            gaps[valid_count] = gap
            valid_count += 1

    if valid_count == 0:
        raise ValueError("All bootstrap replicates are invalid; cannot estimate CI.")

    gaps = gaps[:valid_count]
    ci_low, ci_high = np.quantile(gaps, [0.025, 0.975])
    return gaps, float(ci_low), float(ci_high)


def _infer_subregion_col(df: pd.DataFrame) -> str:
    for col in ["Subregion", "subregion", "Region", "region"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find subregion column in summer_subregions_performance_shap.csv.")


def _infer_r2_col(df: pd.DataFrame) -> str:
    for col in ["R²_RF", "R2_RF", "R2", "R²"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find R2 column in summer_subregions_performance_shap.csv.")


def _read_table4_gap_from_csv(csv_path: str) -> tuple[float, float, float]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Table4 CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    sub_col = _infer_subregion_col(df)
    r2_col = _infer_r2_col(df)

    work = df[[sub_col, r2_col]].copy()
    work[sub_col] = work[sub_col].astype(str).str.strip()
    work[r2_col] = pd.to_numeric(work[r2_col], errors="coerce")

    c_row = work.loc[work[sub_col] == "Central Hunan", r2_col]
    s_row = work.loc[work[sub_col] == "South Hunan", r2_col]
    if c_row.empty or s_row.empty:
        raise ValueError(
            "summer_subregions_performance_shap.csv must include 'Central Hunan' and 'South Hunan' rows."
        )
    r2_central = float(c_row.iloc[0])
    r2_south = float(s_row.iloc[0])
    if not (np.isfinite(r2_central) and np.isfinite(r2_south)):
        raise ValueError("Central/South R2 values in summer_subregions_performance_shap.csv are invalid.")
    return r2_central, r2_south, float(r2_central - r2_south)


def compute_table4_summer_bootstrap_ci(
    n_boot: int = 1000,
    random_seed: int = 42,
    table4_csv: str = TABLE4_CSV_PATH,
) -> tuple[str, str]:
    table4_r2_central, table4_r2_south, table4_gap = _read_table4_gap_from_csv(table4_csv)

    data = np.load(DATA_NPZ_PATH)
    required = {"y_test", "rf_pred"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(
            f"NPZ missing required arrays: {sorted(missing)}. "
            "Please re-run train_model.py first."
        )

    y_test = np.asarray(data["y_test"], dtype=float).ravel()
    rf_pred = np.asarray(data["rf_pred"], dtype=float).ravel()

    if {"lat_test", "lon_test"}.issubset(set(data.files)):
        lat_test = np.asarray(data["lat_test"], dtype=float).ravel()
        lon_test = np.asarray(data["lon_test"], dtype=float).ravel()
    else:
        rebuilt = recompute_test_lat_lon_match_npz(DATA_NPZ_PATH)
        if rebuilt is None:
            raise ValueError(
                "NPZ missing lat_test/lon_test and auto-rebuild failed. "
                "Please re-run train_model.py."
            )
        lat_test, lon_test = rebuilt
        print(
            "[bootstrap_r2_ci] lat_test/lon_test missing in NPZ; "
            "auto-rebuilt row alignment from raw inputs."
        )

    if not (len(y_test) == len(rf_pred) == len(lat_test) == len(lon_test)):
        raise ValueError("y_test/rf_pred/lat_test/lon_test length mismatch.")

    subregions = np.array([_assign_subregion(la, lo) for la, lo in zip(lat_test, lon_test)])

    m_central = subregions == "Central Hunan"
    m_south = subregions == "South Hunan"
    n_central = int(np.sum(m_central))
    n_south = int(np.sum(m_south))
    if n_central < 2 or n_south < 2:
        raise ValueError(
            f"Insufficient samples for R2 gap: Central={n_central}, South={n_south}. "
            "Need at least 2 samples in each subregion."
        )

    y_central, p_central = y_test[m_central], rf_pred[m_central]
    y_south, p_south = y_test[m_south], rf_pred[m_south]

    r2_central = float(r2_score(y_central, p_central))
    r2_south = float(r2_score(y_south, p_south))
    point_gap = float(r2_central - r2_south)

    boot_gaps, ci_low, ci_high = _bootstrap_r2_gap(
        y_central=y_central,
        p_central=p_central,
        y_south=y_south,
        p_south=p_south,
        n_boot=n_boot,
        random_seed=random_seed,
    )

    # Two-sided bootstrap p-value for H0: gap = 0.
    p_two_sided = float(2.0 * min(np.mean(boot_gaps <= 0.0), np.mean(boot_gaps >= 0.0)))
    p_two_sided = min(max(p_two_sided, 0.0), 1.0)
    significant = (ci_low > 0.0) or (ci_high < 0.0)
    table4_gap_in_ci = ci_low <= table4_gap <= ci_high

    summary = pd.DataFrame(
        [
            {
                "Subregion_A": "Central Hunan",
                "Subregion_B": "South Hunan",
                "N_A": n_central,
                "N_B": n_south,
                "R2_A": r2_central,
                "R2_B": r2_south,
                "R2_gap_A_minus_B": point_gap,
                "Bootstrap_N": int(boot_gaps.size),
                "CI95_low": ci_low,
                "CI95_high": ci_high,
                "Bootstrap_p_two_sided(H0:gap=0)": p_two_sided,
                "Significant_at_0.05": bool(significant),
                "reported_gap_reference": float(table4_gap),
                "reference_r2_central": float(table4_r2_central),
                "reference_r2_south": float(table4_r2_south),
                "reference_gap_within_ci95": bool(table4_gap_in_ci),
                "abs_diff_point_vs_reference": float(abs(point_gap - table4_gap)),
            }
        ]
    )
    summary.to_csv(OUT_CSV_PATH, index=False)

    with open(OUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# Summer Central-South R2 Gap Bootstrap CI\n\n")
        f.write(_to_markdown_table(summary.round(6)))
        f.write("\n\n")
        if significant:
            f.write("- Conclusion: The Central-South R2 gap is statistically significant at alpha=0.05.\n")
        else:
            f.write("- Conclusion: The Central-South R2 gap is not statistically significant at alpha=0.05.\n")
        if table4_gap_in_ci:
            f.write("- The reported gap (from summer_subregions_performance_shap.csv) lies within the bootstrap 95% CI.\n")
        else:
            f.write("- The reported gap (from summer_subregions_performance_shap.csv) is outside the bootstrap 95% CI.\n")

    print(OUT_CSV_PATH)
    print(OUT_MD_PATH)
    return OUT_CSV_PATH, OUT_MD_PATH


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap 95% CI for the summer Central-South R2 gap."
    )
    parser.add_argument("--n_boot", type=int, default=1000, help="Bootstrap resample count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--table4_csv",
        type=str,
        default=TABLE4_CSV_PATH,
        help="Path to summer_subregions_performance_shap.csv for reading the reference R2 gap.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    compute_table4_summer_bootstrap_ci(
        n_boot=args.n_boot,
        random_seed=args.seed,
        table4_csv=args.table4_csv,
    )
