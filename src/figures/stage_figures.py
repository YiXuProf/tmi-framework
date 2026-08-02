"""Figures stage entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> int:
    _ensure_src_on_path()

    print("[figures-stage] start")

    from figures.figure_dem_shap_subregions import plot_dem_shap_subregions
    from figures.figure_scatter_imerg_rf import make_double_scatter
    from figures.figure_summer_daily_bias import plot_summer_daily_bias_panels
    from figures.figure_shap_triptych import plot_shap_summary_triptych
    from figures.figure_tsne_shap_disorder import make_tsne_shap_disorder_panel
    from figures.figure_tsne_perplexity_sensitivity import run_tsne_perplexity_sensitivity
    from figures.figure_shap_summary_dependence import make_shap_core_figure

    make_double_scatter()
    plot_summer_daily_bias_panels()
    make_shap_core_figure()
    plot_dem_shap_subregions()
    make_tsne_shap_disorder_panel()
    run_tsne_perplexity_sensitivity()
    plot_shap_summary_triptych()

    print("[figures-stage] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
