"""Spring-Summer reverse transfer test with SHAP direction comparison.

Purpose:
- Evaluate reverse transfer RF-Spring -> Summer and compare with forward transfer
  RF-Summer -> Spring.
- Test directionality of mechanism mismatch:
  * both directions fail -> supports "mechanism purity"
  * only one direction fails -> suggests seasonal asymmetry

Outputs:
- results/Spring_Summer_Transfer_Performance.csv
- results/Spring_Summer_Transfer_SHAP_Direction.csv
- results/Spring_Summer_Transfer_Evaluation.md
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core import project_config as config


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SUMMER_MODELS_DIR = getattr(config, "SUMMER_MODELS_DIR", os.path.join(MODELS_DIR, "summer"))
SPRING_MODELS_DIR = getattr(config, "SPRING_MODELS_DIR", os.path.join(MODELS_DIR, "spring"))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SUMMER_NPZ_PATH = getattr(config, "DATA_NPZ_PATH", os.path.join(SUMMER_MODELS_DIR, "train_test_data.npz"))
SPRING_NPZ_PATH = getattr(config, "SPRING_DATA_NPZ_PATH", os.path.join(SPRING_MODELS_DIR, "train_test_data.npz"))
SUMMER_MODEL_PATH = getattr(
    config,
    "RF_SUMMER_MODEL_PATH",
    getattr(config, "RF_MODEL_PATH", os.path.join(SUMMER_MODELS_DIR, "rf_model_summer.joblib")),
)
SPRING_MODEL_PATH = getattr(
    config,
    "RF_SPRING_MODEL_PATH",
    os.path.join(SPRING_MODELS_DIR, "rf_model_spring.joblib"),
)

OUT_PERF_CSV = os.path.join(RESULTS_DIR, "Spring_Summer_Transfer_Performance.csv")
OUT_SHAP_CSV = os.path.join(RESULTS_DIR, "Spring_Summer_Transfer_SHAP_Direction.csv")
OUT_MD = os.path.join(RESULTS_DIR, "Spring_Summer_Transfer_Evaluation.md")

FEATURE_NAMES = ["imerg", "u10", "v10", "tcwv", "dem"]
FEATURE_ORDER = ["imerg", "tcwv", "dem", "v10", "u10"]
EPS = 1e-12


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _load_npz_xy(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    data = np.load(npz_path)
    required = {"X_test", "y_test"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"NPZ missing arrays {sorted(missing)}: {npz_path}")
    x_test = np.asarray(data["X_test"], dtype=float)
    y_test = np.asarray(data["y_test"], dtype=float).ravel()
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"X_test/y_test length mismatch: {npz_path}")
    if x_test.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Unexpected feature count in {npz_path}: {x_test.shape[1]} != {len(FEATURE_NAMES)}"
        )
    return x_test, y_test


def _load_model(model_path: str):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }


def _direction_symbol(v: float) -> str:
    if (not np.isfinite(v)) or abs(float(v)) <= EPS:
        return "0"
    return "+" if float(v) > 0 else "-"


def _mean_shap_by_feature(model, x: np.ndarray) -> dict[str, float]:
    x_df = pd.DataFrame(x, columns=FEATURE_NAMES)
    shap_values = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_values(
        x_df, check_additivity=False, approximate=True
    )
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values, dtype=float)
    means = np.nanmean(shap_values, axis=0)
    return {f: float(means[i]) for i, f in enumerate(FEATURE_NAMES)}


def evaluate_spring_summer_transfer(r2_drop_threshold: float = 0.05) -> tuple[str, str, str]:
    # Load data
    x_su, y_su = _load_npz_xy(SUMMER_NPZ_PATH)
    x_sp, y_sp = _load_npz_xy(SPRING_NPZ_PATH)

    # Load models
    model_su = _load_model(SUMMER_MODEL_PATH)
    model_sp = _load_model(SPRING_MODEL_PATH)

    # Four scenario predictions
    pred_su_on_su = np.asarray(model_su.predict(x_su), dtype=float).ravel()
    pred_sp_on_sp = np.asarray(model_sp.predict(x_sp), dtype=float).ravel()
    pred_su_on_sp = np.asarray(model_su.predict(x_sp), dtype=float).ravel()  # Summer -> Spring
    pred_sp_on_su = np.asarray(model_sp.predict(x_su), dtype=float).ravel()  # Spring -> Summer

    perf_rows = []
    perf_rows.append(
        {
            "Scenario": "RF-Summer->Summer (in-season)",
            "Train_Season": "Summer",
            "Test_Season": "Summer",
            "N": int(y_su.size),
            **_calc_metrics(y_su, pred_su_on_su),
        }
    )
    perf_rows.append(
        {
            "Scenario": "RF-Spring->Spring (in-season)",
            "Train_Season": "Spring",
            "Test_Season": "Spring",
            "N": int(y_sp.size),
            **_calc_metrics(y_sp, pred_sp_on_sp),
        }
    )
    perf_rows.append(
        {
            "Scenario": "RF-Summer->Spring (forward-transfer)",
            "Train_Season": "Summer",
            "Test_Season": "Spring",
            "N": int(y_sp.size),
            **_calc_metrics(y_sp, pred_su_on_sp),
        }
    )
    perf_rows.append(
        {
            "Scenario": "RF-Spring->Summer (reverse-transfer)",
            "Train_Season": "Spring",
            "Test_Season": "Summer",
            "N": int(y_su.size),
            **_calc_metrics(y_su, pred_sp_on_su),
        }
    )
    perf_df = pd.DataFrame(perf_rows)

    # Failure logic: compare transfer model to native in-season model on the same target season.
    r2_sp_native = float(perf_df.loc[perf_df["Scenario"] == "RF-Spring->Spring (in-season)", "R2"].iloc[0])
    r2_su_native = float(perf_df.loc[perf_df["Scenario"] == "RF-Summer->Summer (in-season)", "R2"].iloc[0])
    r2_forward = float(
        perf_df.loc[perf_df["Scenario"] == "RF-Summer->Spring (forward-transfer)", "R2"].iloc[0]
    )
    r2_reverse = float(
        perf_df.loc[perf_df["Scenario"] == "RF-Spring->Summer (reverse-transfer)", "R2"].iloc[0]
    )
    forward_drop = r2_sp_native - r2_forward
    reverse_drop = r2_su_native - r2_reverse
    forward_fail = bool(forward_drop > r2_drop_threshold)
    reverse_fail = bool(reverse_drop > r2_drop_threshold)

    if forward_fail and reverse_fail:
        directionality_conclusion = "Both-direction failure -> supports mechanism purity."
    elif forward_fail ^ reverse_fail:
        directionality_conclusion = "One-direction failure -> indicates seasonal asymmetry."
    else:
        directionality_conclusion = "No clear directional failure -> transfer mismatch is weak."

    # SHAP mean direction in four scenarios
    mean_su_on_su = _mean_shap_by_feature(model_su, x_su)
    mean_sp_on_sp = _mean_shap_by_feature(model_sp, x_sp)
    mean_su_on_sp = _mean_shap_by_feature(model_su, x_sp)
    mean_sp_on_su = _mean_shap_by_feature(model_sp, x_su)

    shap_rows = []
    for feat in FEATURE_ORDER:
        row = {
            "Feature": feat,
            "RF-Summer->Spring": mean_su_on_sp[feat],
            "RF-Summer->Spring direction": _direction_symbol(mean_su_on_sp[feat]),
            "RF-Summer->Summer": mean_su_on_su[feat],
            "RF-Summer->Summer direction": _direction_symbol(mean_su_on_su[feat]),
            "RF-Spring->Spring": mean_sp_on_sp[feat],
            "RF-Spring->Spring direction": _direction_symbol(mean_sp_on_sp[feat]),
            "RF-Spring->Summer": mean_sp_on_su[feat],
            "RF-Spring->Summer direction": _direction_symbol(mean_sp_on_su[feat]),
        }
        row["SummerModel Cross-Season Consistency"] = (
            "Yes"
            if row["RF-Summer->Spring direction"] == row["RF-Summer->Summer direction"]
            else "No"
        )
        row["SpringModel Cross-Season Consistency"] = (
            "Yes"
            if row["RF-Spring->Summer direction"] == row["RF-Spring->Spring direction"]
            else "No"
        )
        shap_rows.append(row)
    shap_df = pd.DataFrame(shap_rows)

    perf_out = perf_df.copy()
    perf_out["R2_drop_vs_target_native"] = np.nan
    perf_out.loc[
        perf_out["Scenario"] == "RF-Summer->Spring (forward-transfer)", "R2_drop_vs_target_native"
    ] = forward_drop
    perf_out.loc[
        perf_out["Scenario"] == "RF-Spring->Summer (reverse-transfer)", "R2_drop_vs_target_native"
    ] = reverse_drop
    perf_out["Fail_by_threshold"] = False
    perf_out.loc[
        perf_out["Scenario"] == "RF-Summer->Spring (forward-transfer)", "Fail_by_threshold"
    ] = forward_fail
    perf_out.loc[
        perf_out["Scenario"] == "RF-Spring->Summer (reverse-transfer)", "Fail_by_threshold"
    ] = reverse_fail

    perf_out.to_csv(OUT_PERF_CSV, index=False)
    shap_df.to_csv(OUT_SHAP_CSV, index=False)

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# Spring-Summer transfer directionality test\n\n")
        f.write("## Performance (in-season vs cross-season)\n\n")
        f.write(_to_markdown_table(perf_out.round(6)))
        f.write("\n\n## SHAP direction comparison\n\n")
        f.write(_to_markdown_table(shap_df.round(6)))
        f.write("\n\n## Directionality conclusion\n\n")
        f.write(
            f"- R2 drop threshold for failure = {r2_drop_threshold:.3f} (transfer vs target-native).\n"
        )
        f.write(f"- Forward (Summer->Spring) R2 drop = {forward_drop:.6f}; fail={forward_fail}.\n")
        f.write(f"- Reverse (Spring->Summer) R2 drop = {reverse_drop:.6f}; fail={reverse_fail}.\n")
        f.write(f"- Conclusion: {directionality_conclusion}\n")

    print(f"[transfer-test] summer npz: {SUMMER_NPZ_PATH}")
    print(f"[transfer-test] spring npz: {SPRING_NPZ_PATH}")
    print(f"[transfer-test] summer model: {SUMMER_MODEL_PATH}")
    print(f"[transfer-test] spring model: {SPRING_MODEL_PATH}")
    print(OUT_PERF_CSV)
    print(OUT_SHAP_CSV)
    print(OUT_MD)
    return OUT_PERF_CSV, OUT_SHAP_CSV, OUT_MD


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reverse transfer RF-Spring->Summer vs forward transfer RF-Summer->Spring."
    )
    parser.add_argument(
        "--r2_drop_threshold",
        type=float,
        default=0.05,
        help="Failure threshold on R2 drop vs target native in-season model.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    evaluate_spring_summer_transfer(r2_drop_threshold=args.r2_drop_threshold)
