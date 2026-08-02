"""Bootstrap test for SHAP disorder (sigma) gap.

Purpose:
- For Fig.6b variables, test whether SHAP sigma in high-intensity rainfall
  (80-100 percentile) is significantly higher than low-intensity rainfall
  (0-20 percentile).
- Provide statistical support for the "progressive failure" narrative.

Outputs:
- results/shap_disorder_bootstrap.csv
- results/shap_disorder_bootstrap.md
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

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
OUT_CSV_PATH = os.path.join(RESULTS_DIR, "shap_disorder_bootstrap.csv")
OUT_MD_PATH = os.path.join(RESULTS_DIR, "shap_disorder_bootstrap.md")

FEATURE_NAMES = list(DEFAULT_FEATURE_NAMES)
FIG6B_FEATURES = ["imerg", "tcwv", "dem", "v10"]


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _bootstrap_sigma_gap(
    shap_low: np.ndarray,
    shap_high: np.ndarray,
    n_boot: int,
    random_seed: int,
) -> tuple[float, float, float, float, float]:
    low = shap_low[np.isfinite(shap_low)]
    high = shap_high[np.isfinite(shap_high)]
    n_low = low.size
    n_high = high.size
    if n_low < 2 or n_high < 2:
        raise ValueError("Not enough finite SHAP values in low/high group for bootstrap.")

    sigma_low = float(np.std(low, ddof=1))
    sigma_high = float(np.std(high, ddof=1))
    point_gap = sigma_high - sigma_low

    rng = np.random.default_rng(random_seed)
    gaps = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx_low = rng.integers(0, n_low, size=n_low)
        idx_high = rng.integers(0, n_high, size=n_high)
        sig_l = float(np.std(low[idx_low], ddof=1))
        sig_h = float(np.std(high[idx_high], ddof=1))
        gaps[i] = sig_h - sig_l

    ci_low, ci_high = np.quantile(gaps, [0.025, 0.975])
    p_two_sided = float(2.0 * min(np.mean(gaps <= 0.0), np.mean(gaps >= 0.0)))
    p_two_sided = min(max(p_two_sided, 0.0), 1.0)
    return sigma_low, sigma_high, float(point_gap), float(ci_low), float(ci_high), p_two_sided


def build_s1_shap_disorder_bootstrap(
    n_boot: int = 1000,
    random_seed: int = 42,
) -> tuple[str, str]:
    _, y_test, shap_values = get_test_shap_values(feature_names=FEATURE_NAMES)

    q20 = float(np.nanquantile(y_test, 0.20))
    q80 = float(np.nanquantile(y_test, 0.80))
    low_mask = y_test <= q20
    high_mask = y_test >= q80
    n_low = int(np.sum(low_mask))
    n_high = int(np.sum(high_mask))
    if n_low < 2 or n_high < 2:
        raise ValueError(
            f"Too few samples in percentile groups: low={n_low}, high={n_high}."
        )

    feature_index = {f: i for i, f in enumerate(FEATURE_NAMES)}
    rows: list[dict[str, float | int | str | bool]] = []
    for feat in FIG6B_FEATURES:
        idx = feature_index[feat]
        sigma_low, sigma_high, gap, ci95_low, ci95_high, p_two = _bootstrap_sigma_gap(
            shap_low=shap_values[low_mask, idx],
            shap_high=shap_values[high_mask, idx],
            n_boot=n_boot,
            random_seed=random_seed,
        )
        rows.append(
            {
                "feature": feat,
                "n_low_0_20pct": n_low,
                "n_high_80_100pct": n_high,
                "y_q20_mm_d": q20,
                "y_q80_mm_d": q80,
                "sigma_low": sigma_low,
                "sigma_high": sigma_high,
                "sigma_gap_high_minus_low": gap,
                "ci95_low": ci95_low,
                "ci95_high": ci95_high,
                "bootstrap_p_two_sided_h0_gap0": p_two,
                "significant_at_0_05": bool((ci95_low > 0.0) or (ci95_high < 0.0)),
            }
        )

    out_df = pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)
    out_df.to_csv(OUT_CSV_PATH, index=False)

    n_sig = int(np.sum(out_df["significant_at_0_05"].astype(bool)))
    with open(OUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# SHAP disorder bootstrap test (0-20% vs 80-100%)\n\n")
        f.write(_to_markdown_table(out_df.round(6)))
        f.write("\n\n")
        f.write(
            "- Test design: for each Fig.6b feature, bootstrap SHAP sigma gap "
            "(sigma_high - sigma_low) using low (0-20%) and high (80-100%) rainfall quantile groups.\n"
        )
        f.write(
            f"- Result overview: {n_sig}/{len(out_df)} features show significant sigma gaps at alpha=0.05.\n"
        )

    print(f"[s1-bootstrap] NPZ source: {DATA_NPZ_PATH}")
    print(f"[s1-bootstrap] model source: {RF_MODEL_PATH}")
    print(OUT_CSV_PATH)
    print(OUT_MD_PATH)
    return OUT_CSV_PATH, OUT_MD_PATH


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="S1 bootstrap test for SHAP sigma difference (high vs low rainfall quantiles)."
    )
    parser.add_argument("--n_boot", type=int, default=1000, help="Bootstrap resample count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    build_s1_shap_disorder_bootstrap(n_boot=args.n_boot, random_seed=args.seed)
