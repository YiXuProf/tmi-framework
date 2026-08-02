"""Analysis stage entrypoint.

Runs analysis scripts in dependency-safe order.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> int:
    _ensure_src_on_path()

    print("[analysis-stage] start")
    from core.shap_cache import get_test_shap_values

    from analysis.ablation_study import run_ablation
    from analysis.shap_disorder_bootstrap import build_s1_shap_disorder_bootstrap
    from analysis.shap_intensity_analysis import build_shap_by_intensity
    from analysis.seasonal_reverse_transfer_analysis import evaluate_spring_summer_transfer
    from analysis.seasonal_shap_direction_table import build_spring_summer_shap_direction_table
    from analysis.subregion_clustering_analysis import run_subregion_objective_clustering
    from analysis.subregion_performance_shap import build_summer_subregions_table
    from analysis.summer_r2_gap_bootstrap import compute_table4_summer_bootstrap_ci

    run_ablation()
    # Warm SHAP cache once; downstream scripts reuse it.
    get_test_shap_values()
    build_summer_subregions_table()
    compute_table4_summer_bootstrap_ci()
    build_shap_by_intensity()
    build_s1_shap_disorder_bootstrap()
    evaluate_spring_summer_transfer()
    build_spring_summer_shap_direction_table()
    run_subregion_objective_clustering(method="kmeans")

    print("[analysis-stage] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
