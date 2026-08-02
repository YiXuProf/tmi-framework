"""
Backward-compatible aliases for ``config.SP_*`` (SciencePlots ``science`` prop_cycle).

Prefer ``from core import project_config as config`` and ``config.SP_IMERG`` in new code. Uses ``getattr`` so old ``config`` without ``SP_*`` still loads.
"""
from __future__ import annotations

from core import project_config as _cfg

_SP = lambda n, fb: getattr(_cfg, n, fb)

SCIENCE_CYCLE = (
    _SP("SP_IMERG", "#0C5DA5"),
    _SP("SP_RF_FULL", "#00B945"),
    _SP("SP_LR_FULL", "#FF9500"),
    _SP("SP_ACCENT_WARN", "#FF2C00"),
    _SP("SP_VIOLET", "#845B97"),
    _SP("SP_RAW_BAR", "#474747"),
    _SP("SP_IDENTITY_LINE", "#9E9E9E"),
)

BLUE = SCIENCE_CYCLE[0]
GREEN = SCIENCE_CYCLE[1]
ORANGE = SCIENCE_CYCLE[2]
RED = SCIENCE_CYCLE[3]
VIOLET = SCIENCE_CYCLE[4]
GRAY_DARK = SCIENCE_CYCLE[5]
GRAY_MID = SCIENCE_CYCLE[6]

IMERG = _SP("SP_IMERG", "#0C5DA5")
RF_FULL = _SP("SP_RF_FULL", "#00B945")
LR_FULL = _SP("SP_LR_FULL", "#FF9500")
RAW_IMERG_BAR = _SP("SP_RAW_BAR", "#474747")
ACCENT_WARN = _SP("SP_ACCENT_WARN", "#FF2C00")
IDENTITY_LINE = _SP("SP_IDENTITY_LINE", "#9E9E9E")
GRID = _SP("SP_GRID", "#CFCFCF")
OBS = _SP("SP_OBS", "#1A1A1A")

PROVINCE_HUNAN = _SP("SP_PROVINCE_HUNAN", IMERG)
PROVINCE_GUANGXI = _SP("SP_PROVINCE_GUANGXI", RF_FULL)
PROVINCE_GUANGDONG = _SP("SP_PROVINCE_GUANGDONG", LR_FULL)
