"""Build a standalone table: spring-summer SHAP direction comparison.

Table purpose:
- Compare SHAP direction/sign across three scenarios:
  (a) RF-Summer applied to Spring
  (b) RF-Summer applied to Summer
  (c) RF-Spring applied to Spring

Outputs:
- results/spring_summer_shap_direction.csv
- results/spring_summer_shap_direction.md
"""

import os

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_ORDER = ["imerg", "tcwv", "dem", "v10", "u10"]
EPS = 1e-12


def _direction_symbol(v):
    if not np.isfinite(v) or abs(float(v)) <= EPS:
        return "0"
    return "+" if float(v) > 0 else "-"


def _build_demo_inputs():
    """Input SHAP summaries (replace with your recomputed values if needed)."""
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
    return shap_summer_to_spring, shap_summer_to_summer, shap_spring_to_spring


def _to_markdown_table(df):
    """Convert DataFrame to markdown table without external dependencies."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _prep(df, value_col_name):
    required = {"feature", "mean_shap"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {value_col_name}")

    out = df.copy()
    out["feature"] = out["feature"].astype(str).str.lower()
    out = out.set_index("feature").reindex(FEATURE_ORDER).reset_index()
    out["mean_shap"] = pd.to_numeric(out["mean_shap"], errors="coerce")
    if out["mean_shap"].isna().any():
        missing_feats = out.loc[out["mean_shap"].isna(), "feature"].tolist()
        raise ValueError(f"Invalid/missing mean_shap for features: {missing_feats}")
    out = out.rename(columns={"mean_shap": value_col_name})
    return out


def build_spring_summer_shap_direction_table():
    s2sp, s2su, sp2sp = _build_demo_inputs()

    a = _prep(s2sp, "RF-Summer->Spring")
    b = _prep(s2su, "RF-Summer->Summer")
    c = _prep(sp2sp, "RF-Spring->Spring")

    table = a.merge(b, on="feature").merge(c, on="feature")
    table = table.rename(columns={"feature": "Feature"})

    for col in ["RF-Summer->Spring", "RF-Summer->Summer", "RF-Spring->Spring"]:
        table[f"{col} direction"] = table[col].map(_direction_symbol)

    dir_a = table["RF-Summer->Spring direction"]
    dir_b = table["RF-Summer->Summer direction"]
    dir_c = table["RF-Spring->Spring direction"]

    table["SummerModel Cross-Season Consistency"] = np.where(dir_a == dir_b, "Yes", "No")
    table["SpringDomain Consistency"] = np.where(dir_a == dir_c, "Yes", "No")
    table["All Three Same Direction"] = np.where((dir_a == dir_b) & (dir_b == dir_c), "Yes", "No")

    ordered_cols = [
        "Feature",
        "RF-Summer->Spring",
        "RF-Summer->Spring direction",
        "RF-Summer->Summer",
        "RF-Summer->Summer direction",
        "RF-Spring->Spring",
        "RF-Spring->Spring direction",
        "SummerModel Cross-Season Consistency",
        "SpringDomain Consistency",
        "All Three Same Direction",
    ]
    table = table[ordered_cols]

    csv_path = os.path.join(RESULTS_DIR, "spring_summer_shap_direction.csv")
    md_path = os.path.join(RESULTS_DIR, "spring_summer_shap_direction.md")

    table.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Table: Spring-Summer SHAP Direction Comparison\n\n")
        f.write(_to_markdown_table(table))
        f.write("\n")

    print(csv_path)
    print(md_path)
    return csv_path, md_path


if __name__ == "__main__":
    build_spring_summer_shap_direction_table()
