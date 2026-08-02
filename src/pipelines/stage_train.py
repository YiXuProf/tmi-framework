"""Train stage entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> int:
    _ensure_src_on_path()
    from pipelines.pipeline_train_model import train
    from core import project_config as config

    print("[train-stage] start")
    required_inputs = [
        ("IMERG", config.IMERG_DIR),
        ("OBS", config.OBS_FILE),
        ("ERA5", config.ERA5_FILE),
        ("DEM", config.DEM_FILE),
    ]
    missing = [f"{name}: {path}" for name, path in required_inputs if not os.path.isfile(path)]
    if missing:
        detail = "\n".join(f"  - {m}" for m in missing)
        raise FileNotFoundError(
            "[train-stage] Missing required input files:\n"
            f"{detail}\n"
            "Set DATA_ROOT correctly or update paths in core/project_config.py."
        )
    train()
    print("[train-stage] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
