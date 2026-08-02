"""Shared SHAP cache utilities for summer test samples."""

from __future__ import annotations

import os
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import shap

from core import project_config as config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SUMMER_MODELS_DIR = getattr(config, "SUMMER_MODELS_DIR", MODELS_DIR)

DATA_NPZ_PATH = getattr(config, "DATA_NPZ_PATH", os.path.join(SUMMER_MODELS_DIR, "train_test_data.npz"))
RF_MODEL_PATH = getattr(
    config,
    "RF_SUMMER_MODEL_PATH",
    getattr(config, "RF_MODEL_PATH", os.path.join(SUMMER_MODELS_DIR, "rf_model_summer.joblib")),
)
SHAP_CACHE_PATH = getattr(
    config,
    "SHAP_TEST_CACHE_PATH",
    os.path.join(SUMMER_MODELS_DIR, "shap_values_test_cache.npz"),
)

DEFAULT_FEATURE_NAMES = ("imerg", "u10", "v10", "tcwv", "dem")


def _load_test_xy() -> tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(DATA_NPZ_PATH):
        raise FileNotFoundError(f"NPZ not found: {DATA_NPZ_PATH}")
    data = np.load(DATA_NPZ_PATH)
    required = {"X_test", "y_test"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"NPZ missing arrays: {sorted(missing)}")
    x_test = np.asarray(data["X_test"], dtype=float)
    y_test = np.asarray(data["y_test"], dtype=float).ravel()
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError("X_test and y_test length mismatch.")
    return x_test, y_test


def _is_cache_valid(cache: np.lib.npyio.NpzFile, x_test: np.ndarray, feature_names: Sequence[str]) -> bool:
    try:
        cached_features = tuple(str(v) for v in cache["feature_names"].tolist())
        if cached_features != tuple(feature_names):
            return False
        if int(cache["n_samples"]) != int(x_test.shape[0]):
            return False
        shap_values = np.asarray(cache["shap_values"], dtype=float)
        if shap_values.shape != (x_test.shape[0], len(feature_names)):
            return False
        model_mtime = float(cache["model_mtime"])
        data_mtime = float(cache["data_mtime"])
        if not np.isclose(model_mtime, os.path.getmtime(RF_MODEL_PATH)):
            return False
        if not np.isclose(data_mtime, os.path.getmtime(DATA_NPZ_PATH)):
            return False
        return True
    except Exception:
        return False


def get_test_shap_values(
    feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
    force_recompute: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_test, y_test, shap_values) with persistent on-disk cache."""
    x_test, y_test = _load_test_xy()
    feature_names = tuple(feature_names)

    if (not force_recompute) and os.path.isfile(SHAP_CACHE_PATH):
        with np.load(SHAP_CACHE_PATH, allow_pickle=False) as cache:
            if _is_cache_valid(cache, x_test, feature_names):
                return x_test, y_test, np.asarray(cache["shap_values"], dtype=float)

    if not os.path.isfile(RF_MODEL_PATH):
        raise FileNotFoundError(f"RF summer model not found: {RF_MODEL_PATH}")
    model = joblib.load(RF_MODEL_PATH)
    x_df = pd.DataFrame(x_test, columns=list(feature_names))
    shap_values = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_values(
        x_df, check_additivity=False, approximate=True
    )
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.shape != (x_test.shape[0], len(feature_names)):
        raise ValueError(
            f"Unexpected SHAP shape: {shap_values.shape}, expected {(x_test.shape[0], len(feature_names))}"
        )

    os.makedirs(os.path.dirname(SHAP_CACHE_PATH), exist_ok=True)
    np.savez(
        SHAP_CACHE_PATH,
        shap_values=shap_values,
        feature_names=np.array(feature_names, dtype="<U16"),
        n_samples=np.int64(x_test.shape[0]),
        model_mtime=np.float64(os.path.getmtime(RF_MODEL_PATH)),
        data_mtime=np.float64(os.path.getmtime(DATA_NPZ_PATH)),
    )
    return x_test, y_test, shap_values
