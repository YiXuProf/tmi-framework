"""Plot summer daily Bias panels (2021 vs 2022) with 5-day smoothing.

Uses `models/train_test_data.npz` directly so daily Bias matches the same
test-sample definition used by training/evaluation scripts.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import project_config as config


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SUMMER_MODELS_DIR = getattr(config, "SUMMER_MODELS_DIR", MODELS_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_NPZ_PATH = getattr(
    config, "DATA_NPZ_PATH", os.path.join(SUMMER_MODELS_DIR, "train_test_data.npz")
)

TARGET_YEARS = [2021, 2022]
TARGET_MONTHS = [6, 7, 8]
ROLLING_WINDOW = 5
LOW_VALLEY_PER_YEAR = 3
MIN_SAMPLES_PER_DAY = 30

YEAR_COLOR = {2021: "#1f77b4", 2022: "#ff7f0e"}


def _safe_daily_bias(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(np.sum(valid))
    if n < MIN_SAMPLES_PER_DAY:
        return np.nan, n
    return float(np.nanmean(y_pred[valid] - y_true[valid])), n


def _calc_daily_bias_table():
    if not os.path.isfile(DATA_NPZ_PATH):
        raise FileNotFoundError(f"NPZ not found: {DATA_NPZ_PATH}")

    data = np.load(DATA_NPZ_PATH)
    required = {"y_test", "raw_pred", "rf_pred", "time_test"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(
            f"NPZ missing arrays {sorted(missing)} in {DATA_NPZ_PATH}. "
            "Please re-run train_model.py."
        )

    raw = np.asarray(data["raw_pred"]).ravel()
    rf = np.asarray(data["rf_pred"]).ravel()
    y = np.asarray(data["y_test"]).ravel()
    t = pd.to_datetime(np.asarray(data["time_test"]).astype("datetime64[ns]"))
    if not (len(raw) == len(rf) == len(y) == len(t)):
        raise ValueError("NPZ arrays raw_pred/rf_pred/y_test/time_test length mismatch.")

    sample_df = pd.DataFrame(
        {
            "time": t,
            "year": t.year.astype(int),
            "month": t.month.astype(int),
            "y": y,
            "raw_pred": raw,
            "rf_pred": rf,
        }
    )
    sample_df = sample_df[
        sample_df["year"].isin(TARGET_YEARS) & sample_df["month"].isin(TARGET_MONTHS)
    ].copy()
    if sample_df.empty:
        raise ValueError("No test samples for 2021/2022 summer in NPZ.")

    sample_df["date"] = sample_df["time"].dt.normalize()
    rows = []
    for date, g in sample_df.groupby("date", sort=True):
        rf_bias, n_rf = _safe_daily_bias(g["y"].values, g["rf_pred"].values)
        imerg_bias, n_imerg = _safe_daily_bias(g["y"].values, g["raw_pred"].values)
        rows.append(
            {
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "year": int(pd.Timestamp(date).year),
                "month": int(pd.Timestamp(date).month),
                "day": int(pd.Timestamp(date).day),
                "rf_bias_daily": rf_bias,
                "imerg_bias_daily": imerg_bias,
                "n_samples_rf": int(n_rf),
                "n_samples_imerg": int(n_imerg),
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def plot_summer_daily_bias_panels():
    df = _calc_daily_bias_table()
    if df.empty:
        raise ValueError("Daily Bias table is empty.")

    out_csv = os.path.join(RESULTS_DIR, "Summer_Daily_Bias_Panels.csv")
    df.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.3), sharey=True, constrained_layout=True)
    panel_labels = {2021: "(a)", 2022: "(b)"}

    for ax, yr in zip(axes, TARGET_YEARS):
        d = df[df["year"] == yr].copy()
        d["x_date"] = pd.to_datetime(d["date"])
        d = d.sort_values("x_date")
        if d.empty:
            ax.set_title(f"{yr} Summer (no data)")
            continue

        d["rf_bias_ma5"] = d["rf_bias_daily"].rolling(window=ROLLING_WINDOW, center=True, min_periods=1).mean()
        d["imerg_bias_ma5"] = d["imerg_bias_daily"].rolling(window=ROLLING_WINDOW, center=True, min_periods=1).mean()
        summer_mean = float(np.nanmean(d["rf_bias_daily"].values))

        color = YEAR_COLOR.get(yr, "#333333")
        ax.plot(
            d["x_date"].values,
            d["rf_bias_ma5"].values,
            color=color,
            lw=2.0,
            label=f"RF {yr} (5-day mean)",
        )
        ax.plot(
            d["x_date"].values,
            d["imerg_bias_ma5"].values,
            color=color,
            lw=1.8,
            ls="--",
            alpha=0.9,
            label=f"IMERG {yr} (5-day mean)",
        )
        ax.axhline(
            summer_mean,
            color=color,
            lw=1.1,
            ls=":",
            alpha=0.95,
            label=f"{yr} mean Bias={summer_mean:.3f}",
        )

        # Annotate significant low valleys by smoothed RF Bias curve.
        low = d[np.isfinite(d["rf_bias_ma5"])].nsmallest(LOW_VALLEY_PER_YEAR, "rf_bias_ma5")
        for _, row in low.iterrows():
            x = row["x_date"]
            y = row["rf_bias_ma5"]
            ax.scatter([x], [y], s=28, color=color, edgecolors="black", linewidths=0.35, zorder=4)
            ax.annotate(
                pd.Timestamp(row["date"]).strftime("%m-%d"),
                (x, y),
                xytext=(4, -13),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )

        tick_dates = pd.to_datetime(
            [f"{yr}-06-01", f"{yr}-06-15", f"{yr}-07-01", f"{yr}-07-15", f"{yr}-08-01", f"{yr}-08-15", f"{yr}-08-31"]
        )
        ax.set_xticks(tick_dates)
        ax.set_xticklabels([t.strftime("%m-%d") for t in tick_dates])
        ax.set_xlim(pd.Timestamp(f"{yr}-06-01"), pd.Timestamp(f"{yr}-08-31"))
        ax.set_xlabel("Date (06-01 to 08-31)")
        ax.set_title(f"{panel_labels[yr]} {yr} Summer")
        ax.grid(True, alpha=0.28, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", frameon=False, fontsize=8.8)

    axes[0].set_ylabel("Daily Bias (mm/d)")
    png_path = os.path.join(RESULTS_DIR, "Summer_Daily_Bias_Panels.png")
    svg_path = os.path.join(RESULTS_DIR, "Summer_Daily_Bias_Panels.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot_summer_daily_bias_panels] NPZ source: {DATA_NPZ_PATH}")
    print(out_csv)
    print(png_path)
    print(svg_path)
    return out_csv, png_path, svg_path


def plot_summer_daily_r2_panels():
    """Backward-compatible entrypoint name."""
    return plot_summer_daily_bias_panels()


if __name__ == "__main__":
    plot_summer_daily_bias_panels()
