"""Build manuscript Table S5: Hunan subregion Moran's I proxies.

Matches the definition used by pipelines/xreg_validation.py via
core.morans_proxies (JJA TEST_YEARS, queen W, persistent mask, no anomaly).

Outputs:
- results/table_s5_morans_i.csv
- results/table_s5_morans_i.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from core import project_config as config
from core.morans_proxies import PROXY_YEARS, compute_hunan_subregion_morans


BASE_DIR = Path(config.BASE_DIR)
RESULTS_DIR = Path(getattr(config, "RESULTS_DIR", BASE_DIR / "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV_PATH = RESULTS_DIR / "table_s5_morans_i.csv"
OUT_MD_PATH = RESULTS_DIR / "table_s5_morans_i.md"


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def build_table_s5_morans_i(years=None) -> str:
    years = list(years) if years is not None else list(PROXY_YEARS)
    rows, mask_src = compute_hunan_subregion_morans(years=years)
    df = pd.DataFrame(rows)

    # Manuscript-facing columns first.
    out = df[
        [
            "subregion_label",
            "subregion",
            "morans_i",
            "morans_i_se",
            "obs_variance",
            "n_cells",
            "n_days",
            "years",
            "season",
            "mask_source",
        ]
    ].copy()
    out["morans_i"] = out["morans_i"].round(3)
    out["morans_i_se"] = out["morans_i_se"].round(3)
    out["obs_variance"] = out["obs_variance"].round(3)
    out.to_csv(OUT_CSV_PATH, index=False)

    with open(OUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# Table S5. Hunan subregion Moran's I of observed precipitation\n\n")
        f.write(
            f"- Definition: mean daily Moran's I (queen contiguity, row-standardized W) "
            f"over JJA {min(years)}–{max(years)}; no deseasonalization.\n"
        )
        f.write(
            "- Spatial support: cells finite on every proxy day "
            f"(persistent mask); Hunan mask source = `{mask_src}`.\n"
        )
        f.write(
            "- Shared implementation: `core/morans_proxies.py` "
            "(also used by `pipelines/xreg_validation.py`).\n\n"
        )
        f.write(_to_markdown_table(out))
        f.write("\n")

    print(f"[table_s5] mask={mask_src}")
    print(OUT_CSV_PATH)
    print(OUT_MD_PATH)
    for _, r in out.iterrows():
        print(
            f"  {r['subregion']}: I={r['morans_i']} "
            f"(SE={r['morans_i_se']}, n_cells={r['n_cells']}, n_days={r['n_days']})"
        )
    return str(OUT_CSV_PATH)


if __name__ == "__main__":
    build_table_s5_morans_i()
