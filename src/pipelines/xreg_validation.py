#!/usr/bin/env python3
"""
xreg_validation.py — Cross-regional two-arm validation (Hunan / Guangxi / Guangdong).

From repo root:
  python src/pipelines/xreg_validation.py [--force]

From src/:
  python pipelines/xreg_validation.py [--force]

Reads ONLY existing artifacts (Hunan summer RF/LR joblibs + train_test_data.npz,
CN05.1 obs, Guangxi/Guangdong raw NetCDF under DATA_ROOT) and writes everything
to <repo>/results/xreg/. Never modifies any existing manuscript pipeline file.

Numerical conventions mirror core/pipeline_train_model.py:
  - interp: IMERG/OBS/DEM -> nearest, ERA5 (u10/v10/tcwv) -> linear
  - feature stack: [imerg, u10, v10, tcwv, dem]
  - train 2016-2020, test 2021-2022, JJA only
  - RF: RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
"""
from __future__ import annotations

import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------- path setup
# Same convention as other stage scripts: put src/ on sys.path.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from core import project_config as config  # noqa: E402
from core.morans_proxies import (  # noqa: E402
    PROXY_YEARS as _PROXY_YEARS,
    boundary_mask_from_gdf,
    label_hunan_grid,
    load_boundary_gdf,
    load_hunan_mask,
    morans_i_mean,
    persistent_mask,
)
from pipelines.pipeline_train_model import (  # noqa: E402
    deduplicate_time,
    ensure_latlon_order,
    find_var,
    standardize_latlon,
    standardize_time,
    to_dataarray,
)

_ROOT = Path(config.BASE_DIR)  # true project root, anchored to config itself

# ------------------------------------------------------------------ constants
DATA_ROOT = Path(config.DATA_ROOT)
OUT = Path(getattr(config, "RESULTS_DIR", _ROOT / "results")) / "xreg"
CACHE = OUT / "cache"
MODELS_OUT = OUT / "models"
LOG_FILE = OUT / "run_log.txt"
TABLE_S5_CSV = Path(getattr(config, "RESULTS_DIR", _ROOT / "results")) / "table_s5_morans_i.csv"

TRAIN_YEARS = list(config.TRAIN_YEARS)          # 2016-2020
TEST_YEARS = list(config.TEST_YEARS)            # 2021-2022
SUMMER_MONTHS = list(config.SUMMER_MONTHS)      # 6,7,8
RANDOM_STATE = int(getattr(config, "RANDOM_STATE", 42))
PROXY_YEARS = list(_PROXY_YEARS)   # = TEST_YEARS; same as Table S5
N_BOOT = 1000


def _load_hunan_i_ref() -> dict:
    """Prefer regenerated Table S5 CSV; empty dict if not built yet."""
    if not TABLE_S5_CSV.is_file():
        return {}
    try:
        df = pd.read_csv(TABLE_S5_CSV)
        if {"subregion", "morans_i"}.issubset(df.columns):
            return {
                str(r["subregion"]): float(r["morans_i"])
                for _, r in df.iterrows()
                if pd.notna(r["morans_i"])
            }
    except Exception:
        return {}
    return {}


HUNAN_I_REF = _load_hunan_i_ref()

# Filename candidates keep original names first; also accept .nc/.nc4 variants.
REGIONS = {
    "guangxi": {
        "name": "Guangxi",
        "dir": DATA_ROOT / "guangxi",
        "imerg": ["imerg_guangxi.nc4", "imerg_guangxi.nc"],
        "era5": ["era5_guangxi.nc", "era5_guangxi.nc4"],
        "dem": ["dem_guangxi_025.nc", "dem_guangxi_025.nc4"],
        "boundary": "guangxi.geojson",
    },
    "guangdong": {
        "name": "Guangdong",
        "dir": DATA_ROOT / "guangdong",
        "imerg": ["imerg_guangdong.nc4", "imerg_guangdong.nc"],
        "era5": ["era5_guangdong.nc", "era5_guangdong.nc4"],
        "dem": ["dem_guangdong_025.nc", "dem_guangdong_025.nc4"],
        "boundary": "guangdong.geojson",
    },
}
def _resolve_data_file(dir_path: Path, candidates, kind: str, rkey: str) -> Path:
    """Pick the first existing file among candidates under dir_path."""
    names = [candidates] if isinstance(candidates, (str, Path)) else list(candidates)
    for name in names:
        path = dir_path / name
        if path.is_file():
            return path
    tried = ", ".join(str(dir_path / name) for name in names)
    raise FileNotFoundError(f"[NO-GO] missing {rkey} {kind}; tried: {tried}")

EFF_CLASS = lambda e: "low" if e < 20 else ("medium" if e <= 30 else "high")  # noqa: E731


# ---------------------------------------------------------------------- utils
def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ------------------------------------------------------------- subregion rules
def label_grid(region: str, lat2d, lon2d, dem2d=None):
    """Vectorized subregion label grid (object array)."""
    if region == "hunan":
        return label_hunan_grid(lat2d, lon2d)
    if region == "guangxi":
        lab = np.full(lat2d.shape, "gx_se_hills", dtype=object)
        lab[(lon2d < 108.0) | (dem2d > 500.0)] = "gx_karst_nw"
        return lab
    if region == "guangdong":
        lab = np.full(lat2d.shape, "gd_inland_hills", dtype=object)
        lab[lat2d < 22.7] = "gd_coastal"
        lab[lat2d >= 24.0] = "gd_north_mtn"
        return lab
    raise KeyError(region)


def label_samples(region: str, lat, lon, dem=None):
    lab = label_grid(region, np.asarray(lat), np.asarray(lon),
                     None if dem is None else np.asarray(dem))
    return lab


def subregions_of(region: str):
    return {
        "hunan": ["west", "north", "south", "central"],
        "guangxi": ["gx_karst_nw", "gx_se_hills"],
        "guangdong": ["gd_coastal", "gd_inland_hills", "gd_north_mtn"],
    }[region]


# --------------------------------------------------------------- data loading
def _clean_era5(da):
    da = standardize_time(standardize_latlon(to_dataarray(da)))
    for dim in ("number", "expver"):
        if dim in da.dims:
            da = da.isel({dim: 0}, drop=True)
    return ensure_latlon_order(da)


def load_region_cube(rkey: str) -> dict:
    """Load & align IMERG/CN05.1/ERA5/DEM for Guangxi/Guangdong to the DEM grid."""
    rc = REGIONS[rkey]
    d = rc["dir"]
    bnd_assets = _ROOT / "assets" / rc["boundary"]   # preferred: assets/
    bnd_local = d / rc["boundary"]                    # fallback: province dir
    files = {
        "imerg": _resolve_data_file(d, rc["imerg"], "imerg", rkey),
        "era5": _resolve_data_file(d, rc["era5"], "era5", rkey),
        "dem": _resolve_data_file(d, rc["dem"], "dem", rkey),
        "obs": Path(config.OBS_FILE),
        "boundary": bnd_assets if bnd_assets.is_file() else bnd_local,
    }
    for k, p in files.items():
        if not Path(p).is_file():
            raise FileNotFoundError(f"[NO-GO] missing {rkey} {k}: {p}")
    log(f"{rkey}: using imerg={files['imerg'].name}, era5={files['era5'].name}, "
        f"dem={files['dem'].name}, boundary={files['boundary']}")

    imerg = ensure_latlon_order(standardize_time(standardize_latlon(to_dataarray(
        find_var(xr.open_dataset(files["imerg"]), config.IMERG_VAR_CANDIDATES, "IMERG")))))
    obs = ensure_latlon_order(standardize_time(standardize_latlon(to_dataarray(
        find_var(xr.open_dataset(files["obs"]), config.OBS_VAR_CANDIDATES, "OBS")))))
    era5_ds = xr.open_dataset(files["era5"])
    u10 = _clean_era5(find_var(era5_ds, config.U10_VAR_CANDIDATES, "U10"))
    v10 = _clean_era5(find_var(era5_ds, config.V10_VAR_CANDIDATES, "V10"))
    tcwv = _clean_era5(find_var(era5_ds, config.TCWV_VAR_CANDIDATES, "TCWV"))
    dem = ensure_latlon_order(standardize_latlon(to_dataarray(
        find_var(xr.open_dataset(files["dem"]), config.DEM_VAR_CANDIDATES, "DEM"))))
    if "time" in dem.dims:
        dem = dem.isel(time=0, drop=True)
    dem = dem.transpose("lat", "lon")

    tlat, tlon = dem.lat.values.astype(float), dem.lon.values.astype(float)
    imerg = imerg.interp(lat=tlat, lon=tlon, method="nearest")
    obs = obs.interp(lat=tlat, lon=tlon, method="nearest")
    u10 = u10.interp(lat=tlat, lon=tlon, method="linear")
    v10 = v10.interp(lat=tlat, lon=tlon, method="linear")
    tcwv = tcwv.interp(lat=tlat, lon=tlon, method="linear")

    imerg = deduplicate_time(imerg, "IMERG")
    obs = deduplicate_time(obs, "OBS")
    u10 = deduplicate_time(u10, "U10")
    v10 = deduplicate_time(v10, "V10")
    tcwv = deduplicate_time(tcwv, "TCWV")

    common = imerg.time.values
    for da in (obs, u10, v10, tcwv):
        common = np.intersect1d(common, da.time.values)
    if common.size == 0:
        raise ValueError(f"[NO-GO] {rkey}: empty time intersection")
    imerg, obs, u10, v10, tcwv = (da.sel(time=common) for da in (imerg, obs, u10, v10, tcwv))

    gdf = load_boundary_gdf(files["boundary"])
    mask2d = boundary_mask_from_gdf(tlat, tlon, gdf)
    if not mask2d.any():
        raise ValueError(f"[NO-GO] {rkey}: empty boundary mask")

    log(f"{rkey}: grid lat={tlat.size} lon={tlon.size}, cells_in_boundary={int(mask2d.sum())}, "
        f"days={common.size} ({str(common[0])[:10]}..{str(common[-1])[:10]})")
    return {"imerg": imerg, "obs": obs, "u10": u10, "v10": v10, "tcwv": tcwv,
            "dem": dem, "mask2d": mask2d, "lat": tlat, "lon": tlon}


# ------------------------------------------------------------ sample building
def build_xy(rkey: str, cube: dict, years, split: str):
    """Day-stacked samples within boundary mask; caches to npz."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{rkey}_{split}.npz"
    if cache.is_file() and not FORCE:
        # Older caches may store string labels as object arrays.
        try:
            z = np.load(cache, allow_pickle=False)
            out = {k: z[k] for k in z.files}
        except ValueError:
            z = np.load(cache, allow_pickle=True)
            out = {k: z[k] for k in z.files}
        log(f"{rkey}/{split}: loaded cache ({cache.name}), n={out['y'].size}")
        return out

    tm = (cube["imerg"].time.dt.year.isin(list(years))
          & cube["imerg"].time.dt.month.isin(SUMMER_MONTHS)).values
    dem2d = cube["dem"].values
    lon2d, lat2d = np.meshgrid(cube["lon"], cube["lat"])
    lab2d = label_grid(rkey, lat2d, lon2d, dem2d)
    X_l, y_l, t_l, lat_l, lon_l, lab_l = [], [], [], [], [], []

    for t_idx in np.where(tm)[0]:
        im = cube["imerg"].isel(time=t_idx).transpose("lat", "lon").values
        u = cube["u10"].isel(time=t_idx).transpose("lat", "lon").values
        v = cube["v10"].isel(time=t_idx).transpose("lat", "lon").values
        q = cube["tcwv"].isel(time=t_idx).transpose("lat", "lon").values
        ob = cube["obs"].isel(time=t_idx).transpose("lat", "lon").values
        m = (np.isfinite(im) & np.isfinite(u) & np.isfinite(v) & np.isfinite(q)
             & np.isfinite(dem2d) & np.isfinite(ob) & cube["mask2d"])
        if m.sum() == 0:
            continue
        X_l.append(np.stack([im[m], u[m], v[m], q[m], dem2d[m]], axis=1))
        y_l.append(ob[m])
        t_l.append(np.full(m.sum(), cube["imerg"].time.values[t_idx], dtype="datetime64[ns]"))
        lat_l.append(lat2d[m]); lon_l.append(lon2d[m]); lab_l.append(lab2d[m])

    if not X_l:
        raise ValueError(f"[NO-GO] {rkey}/{split}: no samples (check time coverage)")
    # Store labels as unicode strings so caches load with allow_pickle=False.
    out = {"X": np.concatenate(X_l), "y": np.concatenate(y_l),
           "time": np.concatenate(t_l), "lat": np.concatenate(lat_l),
           "lon": np.concatenate(lon_l),
           "label": np.asarray(np.concatenate(lab_l), dtype="U64")}
    np.savez(cache, **out)
    log(f"{rkey}/{split}: built n={out['y'].size}, days={len(X_l)} -> cached {cache.name}")
    return out


# ------------------------------------------------------------------ Moran's I
def compute_proxies(hunan_obs: xr.DataArray, hunan_mask2d, cubes: dict) -> pd.DataFrame:
    rows = []
    lon2d, lat2d = np.meshgrid(hunan_obs.lon.values, hunan_obs.lat.values)
    lab2d = label_grid("hunan", lat2d, lon2d)
    for sub in subregions_of("hunan"):
        m = persistent_mask(hunan_obs, PROXY_YEARS, hunan_mask2d & (lab2d == sub))
        if m.sum() < 9:
            log(f"WARNING hunan/{sub}: only {int(m.sum())} cells")
            continue
        i_mean, i_se, nd = morans_i_mean(hunan_obs, PROXY_YEARS, m)
        tm = (hunan_obs.time.dt.year.isin(PROXY_YEARS)
              & hunan_obs.time.dt.month.isin(SUMMER_MONTHS)).values
        var = float(np.nanvar(hunan_obs.isel(time=tm).values[:, m]))
        rows.append({"region": "hunan", "subregion": sub, "morans_i": i_mean,
                     "morans_i_se": i_se, "obs_variance": var, "n_days": nd,
                     "n_cells": int(m.sum())})
    for rkey, cube in cubes.items():
        lon2d, lat2d = np.meshgrid(cube["lon"], cube["lat"])
        lab2d = label_grid(rkey, lat2d, lon2d, cube["dem"].values)
        for sub in subregions_of(rkey):
            m = persistent_mask(cube["obs"], PROXY_YEARS, cube["mask2d"] & (lab2d == sub))
            if m.sum() < 9:
                log(f"WARNING {rkey}/{sub}: only {int(m.sum())} cells")
                continue
            i_mean, i_se, nd = morans_i_mean(cube["obs"], PROXY_YEARS, m)
            tm = (cube["obs"].time.dt.year.isin(PROXY_YEARS)
                  & cube["obs"].time.dt.month.isin(SUMMER_MONTHS)).values
            var = float(np.nanvar(cube["obs"].isel(time=tm).values[:, m]))
            rows.append({"region": rkey, "subregion": sub, "morans_i": i_mean,
                         "morans_i_se": i_se, "obs_variance": var, "n_days": nd,
                         "n_cells": int(m.sum())})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "proxies.csv", index=False)
    log(f"proxies.csv written ({len(df)} subregions)")
    # Self-check against regenerated Table S5 (same core.morans_proxies definition).
    refs = _load_hunan_i_ref() or HUNAN_I_REF
    if not refs:
        log("NOTE: Table S5 CSV not found; skip Hunan Moran's I self-check "
            f"(run analysis/table_s5_morans_i.py to create {TABLE_S5_CSV.name})")
    for _, r in df[df.region == "hunan"].iterrows():
        ref = refs.get(r["subregion"])
        if ref is not None and abs(r["morans_i"] - ref) > 0.01:
            log(f"WARNING self-check: hunan/{r['subregion']} Moran's I={r['morans_i']:.3f} "
                f"vs Table S5 {ref:.3f} (|diff|>0.01)")
    return df


# ------------------------------------------------------------------ modeling
def metrics_pack(y, yhat):
    return {"R2": float(r2_score(y, yhat)),
            "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
            "MAE": float(mean_absolute_error(y, yhat)),
            "Bias": float(np.mean(yhat - y))}


def efficiency(y, yhat, yraw):
    return (1.0 - np.sqrt(mean_squared_error(y, yhat))
            / np.sqrt(mean_squared_error(y, yraw))) * 100.0


def train_region_models(rkey: str, train: dict):
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    rf_p, lr_p = MODELS_OUT / f"{rkey}_rf.joblib", MODELS_OUT / f"{rkey}_lr.joblib"
    if rf_p.is_file() and not FORCE:
        rf = joblib.load(rf_p)
    else:
        t0 = time.perf_counter()
        rf = RandomForestRegressor(n_estimators=config.N_ESTIMATORS,
                                   random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(train["X"], train["y"])
        joblib.dump(rf, rf_p)
        log(f"{rkey}: RF trained n={train['y'].size} in {time.perf_counter()-t0:.0f}s")
    if lr_p.is_file() and not FORCE:
        lr = joblib.load(lr_p)
    else:
        lr = LinearRegression().fit(train["X"], train["y"])
        joblib.dump(lr, lr_p)
    return rf, lr


# ------------------------------------------------------------------ bootstrap
def block_bootstrap(y, yhat, yraw, days):
    """Day-block bootstrap 95% CI for efficiency and R2."""
    rng = np.random.RandomState(RANDOM_STATE)
    days = days.astype("datetime64[D]")
    uniq, inv = np.unique(days, return_inverse=True)
    by_day = [np.where(inv == k)[0] for k in range(uniq.size)]
    effs, r2s = [], []
    for _ in range(N_BOOT):
        sel = rng.choice(uniq.size, size=uniq.size, replace=True)
        idx = np.concatenate([by_day[k] for k in sel])
        effs.append(efficiency(y[idx], yhat[idx], yraw[idx]))
        r2s.append(r2_score(y[idx], yhat[idx]))
    return (float(np.percentile(effs, 2.5)), float(np.percentile(effs, 97.5)),
            float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5)))


# ------------------------------------------------------------- auto-registration
def register_predictions(proxies: pd.DataFrame, hunan_eff: dict) -> pd.DataFrame:
    """Fit Hunan I->efficiency map, predict GX/GD BEFORE any GX/GD model is evaluated."""
    h = proxies[proxies.region == "hunan"].set_index("subregion")
    subs = [s for s in h.index if s in hunan_eff]
    xs = h.loc[subs, "morans_i"].values.astype(float)
    ys = np.array([hunan_eff[s] for s in subs], dtype=float)
    b, a = np.polyfit(xs, ys, 1)
    ts = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, r in proxies[proxies.region != "hunan"].iterrows():
        pred = float(b * r["morans_i"] + a)
        rows.append({"region": r["region"], "subregion": r["subregion"],
                     "morans_i": r["morans_i"], "morans_i_se": r["morans_i_se"],
                     "predicted_efficiency": pred, "predicted_class": EFF_CLASS(pred),
                     "rule": f"eff = {b:.2f} * I + {a:.2f} (fit on Hunan subregions)",
                     "registered_at_utc": ts})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "predictions_registered.csv", index=False)
    log(f"PRE-REGISTRATION locked ({ts}): rule eff={b:.2f}*I+{a:.2f}; "
        f"predictions written for {len(df)} GX/GD subregions BEFORE evaluation")
    return df


# -------------------------------------------------------------------- evaluation
def evaluate_all(models, targets) -> pd.DataFrame:
    rows = []
    for src, (rf, lr) in models.items():
        for tgt, data in targets.items():
            X, y = data["X"], data["y"]
            yraw = X[:, 0]
            for kind, model in (("rf", rf), ("lr", lr)):
                yhat = model.predict(X)
                m = metrics_pack(y, yhat)
                eff = efficiency(y, yhat, yraw)
                row = {"source": src, "target": tgt, "subregion": "ALL", "model": kind,
                       **m, "efficiency": eff, "n_samples": int(y.size),
                       "n_days": int(np.unique(data["time"].astype("datetime64[D]")).size)}
                if kind == "rf":
                    lo, hi, r2lo, r2hi = block_bootstrap(y, yhat, yraw, data["time"])
                    row.update(ci_lo=lo, ci_hi=hi, r2_ci_lo=r2lo, r2_ci_hi=r2hi)
                rows.append(row)
                log(f"eval {src}->{tgt} [{kind}] R2={m['R2']:.3f} eff={eff:.1f}%")
            # subregion level: RF only, two arms (self / hunan-source)
            if src == tgt or src == "hunan":
                yhat = models[src][0].predict(X)
                for sub in np.unique(data["label"]):
                    sel = data["label"] == sub
                    if sel.sum() < 50:
                        continue
                    ys, yh, yr = y[sel], yhat[sel], yraw[sel]
                    m = metrics_pack(ys, yh)
                    eff = efficiency(ys, yh, yr)
                    lo, hi, r2lo, r2hi = block_bootstrap(ys, yh, yr, data["time"][sel])
                    rows.append({"source": src, "target": tgt, "subregion": sub,
                                 "model": "rf", **m, "efficiency": eff,
                                 "ci_lo": lo, "ci_hi": hi, "r2_ci_lo": r2lo, "r2_ci_hi": r2hi,
                                 "n_samples": int(sel.sum()),
                                 "n_days": int(np.unique(data["time"][sel].astype("datetime64[D]")).size)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "eval_matrix.csv", index=False)
    log(f"eval_matrix.csv written ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------- tables
def write_tables(proxies, evals):
    rf = evals[(evals.model == "rf") & (evals.subregion == "ALL")]
    eff_piv = rf.pivot(index="source", columns="target", values="efficiency")
    r2_piv = rf.pivot(index="source", columns="target", values="R2")
    dr2 = r2_piv.subtract(pd.Series(np.diag(r2_piv), index=r2_piv.index), axis=1)
    eff_piv.to_csv(OUT / "table_loro_matrix.csv")
    dr2.round(3).to_csv(OUT / "table_loro_deltaR2.csv")

    # Avoid variable name ``re`` (conflicts with ``import re``).
    re_ = evals[
        (evals.model == "rf")
        & (evals.subregion != "ALL")
        & (evals.source == evals.target)
    ][["target", "subregion", "efficiency", "ci_lo", "ci_hi"]].rename(
        columns={
            "target": "region",
            "efficiency": "retrained_eff",
            "ci_lo": "retrained_ci_lo",
            "ci_hi": "retrained_ci_hi",
        }
    )
    tr_ = evals[
        (evals.model == "rf")
        & (evals.subregion != "ALL")
        & (evals.source == "hunan")
        & (evals.target != "hunan")
    ][["target", "subregion", "efficiency", "ci_lo", "ci_hi"]].rename(
        columns={
            "target": "region",
            "efficiency": "hunan_transfer_eff",
            "ci_lo": "transfer_ci_lo",
            "ci_hi": "transfer_ci_hi",
        }
    )
    summ = (
        proxies.merge(re_, on=["region", "subregion"], how="left")
        .merge(tr_, on=["region", "subregion"], how="left")
    )
    summ.to_csv(OUT / "table_subregion_summary.csv", index=False)
    log("tables written: table_loro_matrix.csv, table_loro_deltaR2.csv, table_subregion_summary.csv")
    return summ


def write_prediction_check(preds, evals):
    re_ = evals[
        (evals.model == "rf")
        & (evals.subregion != "ALL")
        & (evals.source == evals.target)
    ][["target", "subregion", "efficiency"]].rename(
        columns={"target": "region", "efficiency": "actual_retrained_eff"}
    )
    chk = preds.merge(re_, on=["region", "subregion"], how="left")
    chk["actual_class"] = chk["actual_retrained_eff"].map(
        lambda e: EFF_CLASS(e) if pd.notna(e) else "n/a")
    chk["class_hit"] = chk["predicted_class"] == chk["actual_class"]
    chk["abs_error"] = (chk["actual_retrained_eff"] - chk["predicted_efficiency"]).abs()
    chk.to_csv(OUT / "prediction_check.csv", index=False)
    log(f"prediction_check.csv written; class hits: "
        f"{int(chk['class_hit'].sum())}/{len(chk)}")
    return chk


# ---------------------------------------------------------------------- figures
# Low-saturation province colors (paper-consistent muted triad).
REGION_COLOR = {
    "hunan": "#4A6FA5",
    "guangxi": "#C08B5C",
    "guangdong": "#6A8F7A",
}
# fig2: color by arm (not by province), so legend bars match the series.
ARM_COLOR = {
    "local": "#4A6FA5",
    "transfer": "#C08B5C",
}


def _parse_locked_rule(path: Path):
    """Parse ``eff = b * I + a`` from predictions_registered.csv (locked rule)."""
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if "rule" not in df.columns or df.empty:
        return None
    rule = str(df["rule"].iloc[0])
    m = re.search(
        r"eff\s*=\s*([+-]?\d+(?:\.\d+)?)\s*\*\s*I\s*\+\s*([+-]?\d+(?:\.\d+)?)",
        rule,
    )
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), rule


def _yerr_from_ci(y, lo, hi):
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return np.vstack([y - lo, hi - y])


def _savefig(fig, stem: str) -> None:
    """Write both PNG (300 dpi) and SVG next to each other."""
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")


def fig1(summ):
    """Coherence vs efficiency; dashed line = locked pre-registration rule."""
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    region_order = ["hunan", "guangxi", "guangdong"]
    # Journal-friendly labels (code keys -> display text).
    sub_display = {
        "west": "West",
        "north": "North",
        "south": "South",
        "central": "Central",
        "gx_karst_nw": "GX karst NW",
        "gx_se_hills": "GX SE hills",
        "gd_coastal": "GD coastal",
        "gd_inland_hills": "GD inland hills",
        "gd_north_mtn": "GD north mtn",
    }
    # Absolute data-coordinate label placement: (x, y, ha, va).
    # Tuned to current Hunan/GX/GD point positions to avoid collisions.
    lab_abs = {
        "west": (0.697, 31.2, "left", "center"),           # blue point upper-right, above error bars
        "gd_north_mtn": (0.660, 29.1, "right", "center"),  # left of green GD north mtn point
        "gx_se_hills": (0.773, 34.2, "center", "top"),     # below GX SE hills (0.773, 35.6)
        "gx_karst_nw": (0.780, 38.7, "right", "center"),   # upper-left of GX karst NW
        "north": None,  # keep relative lower-right placement (no collision)
    }
    lab_rel = {
        "north": (0.004, -1.0, "left", "top"),
    }

    # Locked rule from predictions_registered.csv (not a re-fit).
    rule_handle = None
    parsed = _parse_locked_rule(OUT / "predictions_registered.csv")
    if parsed is not None:
        b, a, rule = parsed
        x_line = np.linspace(
            float(np.nanmin(summ["morans_i"])) - 0.02,
            float(np.nanmax(summ["morans_i"])) + 0.02,
            100,
        )
        ax.plot(
            x_line,
            b * x_line + a,
            ls="--",
            lw=1.2,
            color="#7A8B99",
            zorder=1,
        )
        rule_handle = Line2D(
            [0], [0], ls="--", lw=1.2, color="#7A8B99", label="Pre-registered rule"
        )
        log(f"fig1: locked rule drawn from predictions_registered.csv ({rule})")
    else:
        log("WARNING fig1: could not parse locked rule; dashed line omitted")

    for region in region_order:
        g = summ[summ["region"] == region]
        if g.empty:
            continue
        c = REGION_COLOR[region]

        gg = g[g["retrained_eff"].notna()]
        if len(gg):
            yerr = None
            if {"retrained_ci_lo", "retrained_ci_hi"}.issubset(gg.columns):
                yerr = _yerr_from_ci(
                    gg["retrained_eff"], gg["retrained_ci_lo"], gg["retrained_ci_hi"]
                )
            ax.errorbar(
                gg["morans_i"],
                gg["retrained_eff"],
                xerr=gg["morans_i_se"],
                yerr=yerr,
                fmt="o",
                color=c,
                ecolor=c,
                mfc=c,
                mec=c,
                ms=8,
                capsize=3,
                linestyle="none",
                zorder=3,
            )

        gt = g[g["hunan_transfer_eff"].notna()]
        if len(gt):
            yerr = None
            if {"transfer_ci_lo", "transfer_ci_hi"}.issubset(gt.columns):
                yerr = _yerr_from_ci(
                    gt["hunan_transfer_eff"], gt["transfer_ci_lo"], gt["transfer_ci_hi"]
                )
            # Province color for marker edge AND error bars (no default blue).
            ax.errorbar(
                gt["morans_i"],
                gt["hunan_transfer_eff"],
                xerr=gt["morans_i_se"],
                yerr=yerr,
                fmt="o",
                color=c,
                ecolor=c,
                mfc="none",
                mec=c,
                mew=1.5,
                ms=9,
                capsize=3,
                linestyle="none",
                zorder=3,
            )

    for _, r in summ.iterrows():
        if not pd.notna(r.get("retrained_eff")):
            continue
        key = str(r["subregion"])
        label = sub_display.get(key, key)
        x0 = float(r["morans_i"])
        y0 = float(r["retrained_eff"])
        if key in lab_abs and lab_abs[key] is not None:
            xt, yt, ha, va = lab_abs[key]
            ax.text(xt, yt, label, ha=ha, va=va, fontsize=7, alpha=0.75)
        elif key in lab_rel:
            dx, dy, ha, va = lab_rel[key]
            ax.text(x0 + dx, y0 + dy, label, ha=ha, va=va, fontsize=7, alpha=0.75)
        else:
            ax.annotate(
                label,
                (x0, y0),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=7,
                alpha=0.75,
            )

    # Custom legend handles (no groupby auto-labels).
    handles = []
    for region in region_order:
        if (summ["region"] == region).any() and summ.loc[
            summ["region"] == region, "retrained_eff"
        ].notna().any():
            c = REGION_COLOR[region]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=c,
                    markeredgecolor=c,
                    markersize=8,
                    linestyle="None",
                    label=f"{region.capitalize()} (retrained)",
                )
            )
    if summ["hunan_transfer_eff"].notna().any():
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="none",
                markeredgecolor="#555555",
                markeredgewidth=1.5,
                markersize=9,
                linestyle="None",
                label="Hunan model (transfer)",
            )
        )
    if rule_handle is not None:
        handles.append(rule_handle)

    ax.set_xlabel("Mean daily Moran's I of observed precipitation (JJA 2021–2022)")
    ax.set_ylabel("Correction efficiency (%)")
    ax.legend(handles=handles, frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    _savefig(fig, "fig7_coherence_vs_efficiency")
    plt.close(fig)


def fig2(evals):
    """Two-arm bars colored by arm (local vs transfer), with matching legend."""
    rf = evals[(evals.model == "rf") & (evals.subregion == "ALL")]
    regions = ["hunan", "guangxi", "guangdong"]
    x = np.arange(len(regions))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for i, tgt in enumerate(regions):
        self_r = rf[(rf.source == tgt) & (rf.target == tgt)].iloc[0]
        ax.bar(
            i - w / 2,
            self_r["efficiency"],
            width=w,
            color=ARM_COLOR["local"],
            yerr=[[self_r["efficiency"] - self_r["ci_lo"]],
                  [self_r["ci_hi"] - self_r["efficiency"]]],
            capsize=4,
            label="Locally retrained" if i == 0 else None,
        )
        if tgt != "hunan":
            tr_r = rf[(rf.source == "hunan") & (rf.target == tgt)].iloc[0]
            ax.bar(
                i + w / 2,
                tr_r["efficiency"],
                width=w,
                color=ARM_COLOR["transfer"],
                yerr=[[tr_r["efficiency"] - tr_r["ci_lo"]],
                      [tr_r["ci_hi"] - tr_r["efficiency"]]],
                capsize=4,
                label="Hunan model (transfer)" if i == 1 else None,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in regions])
    ax.set_ylabel("Correction efficiency (%)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, "fig8_two_arm_bars")
    plt.close(fig)


def fig3(proxies, grids):
    """Moran's I maps with shared color scale (avoids false yellow/purple contrast)."""
    vmin, vmax = 0.54, 0.80
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    im = None
    for ax, (region, (lat, lon, lab2d, mask2d)) in zip(axes, grids.items()):
        field = np.full(lab2d.shape, np.nan)
        for _, r in proxies[proxies.region == region].iterrows():
            field[(lab2d == r["subregion"]) & mask2d] = r["morans_i"]
        # Log near-equal subregion I for Guangxi (fig3 should look almost same color).
        if region == "guangxi":
            vals = proxies.loc[proxies.region == "guangxi", ["subregion", "morans_i"]]
            log("fig3 guangxi Moran's I: "
                + ", ".join(f"{r.subregion}={r.morans_i:.3f}" for r in vals.itertuples()))
        ext = [lon.min() - 0.125, lon.max() + 0.125, lat.min() - 0.125, lat.max() + 0.125]
        im = ax.imshow(
            field,
            origin="lower",
            extent=ext,
            cmap="viridis",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        ax.contour(lon, lat, mask2d.astype(float), levels=[0.5], colors="k", linewidths=0.6)
        ax.set_title(region.capitalize())
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, location="right")
    cbar.set_label("Moran's I")
    _savefig(fig, "figS2_morans_map")
    plt.close(fig)


def audit_label_islands(region: str, lat, lon, lab2d, mask2d) -> None:
    """Flag singleton / tiny label islands (e.g. Guangdong coastal enclave)."""
    lon2d, lat2d = np.meshgrid(lon, lat)
    for sub in subregions_of(region):
        m = mask2d & (lab2d == sub)
        n = int(m.sum())
        if n == 0:
            continue
        if n <= 3:
            lats = lat2d[m]
            lons = lon2d[m]
            log(
                f"LABEL-AUDIT {region}/{sub}: n_cells={n} "
                f"(lat={lats.min():.3f}..{lats.max():.3f}, "
                f"lon={lons.min():.3f}..{lons.max():.3f}) "
                f"— likely threshold enclave, not a plotting bug"
            )


# ------------------------------------------------------------------------- main
FORCE = "--force" in sys.argv


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    log("=== xreg_validation start ===")

    # 1) Hunan artifacts (read-only)
    npz_p = Path(config.DATA_NPZ_PATH)
    rf_p = Path(config.RF_SUMMER_MODEL_PATH)
    lr_p = Path(config.LR_MODEL_PATH)
    for p, name in ((npz_p, "Hunan NPZ"), (rf_p, "Hunan RF"), (lr_p, "Hunan LR")):
        if not p.is_file():
            raise FileNotFoundError(f"[NO-GO] {name} not found: {p}")
    z = np.load(npz_p, allow_pickle=False)
    hunan_target = {"X": z["X_test"], "y": z["y_test"], "time": z["time_test"],
                    "lat": z["lat_test"], "lon": z["lon_test"]}
    hunan_target["label"] = label_samples("hunan", hunan_target["lat"], hunan_target["lon"])
    log(f"hunan: NPZ test n={hunan_target['y'].size}; RF/LR loaded (read-only)")
    models = {"hunan": (joblib.load(rf_p), joblib.load(lr_p))}
    targets = {"hunan": hunan_target}

    # 2) Guangxi / Guangdong datasets
    cubes, grids = {}, {}
    for rkey in REGIONS:
        cube = load_region_cube(rkey)
        cubes[rkey] = cube
        lon2d, lat2d = np.meshgrid(cube["lon"], cube["lat"])
        grids[rkey] = (cube["lat"], cube["lon"],
                       label_grid(rkey, lat2d, lon2d, cube["dem"].values), cube["mask2d"])
        train = build_xy(rkey, cube, TRAIN_YEARS, "train")
        test = build_xy(rkey, cube, TEST_YEARS, "test")
        targets[rkey] = test
        models[rkey] = train_region_models(rkey, train)

    # 3) A priori proxies (Moran's I, obs only)
    obs_nc = Path(getattr(config, "OBS_NC", ""))
    if not obs_nc.is_file():
        raise FileNotFoundError(f"[NO-GO] Hunan aligned obs not found: {obs_nc}")
    hunan_obs = xr.open_dataset(obs_nc)["obs"]
    # Same Hunan mask rule as Table S5 (core.morans_proxies.load_hunan_mask).
    hunan_mask2d, mask_src = load_hunan_mask(hunan_obs)
    log(f"hunan mask: {mask_src}; cells={int(hunan_mask2d.sum())}")
    lon2d, lat2d = np.meshgrid(hunan_obs.lon.values, hunan_obs.lat.values)
    grids = {"hunan": (hunan_obs.lat.values, hunan_obs.lon.values,
                       label_grid("hunan", lat2d, lon2d), hunan_mask2d), **grids}
    for region, (lat, lon, lab2d, mask2d) in grids.items():
        audit_label_islands(region, lat, lon, lab2d, mask2d)
    proxies = compute_proxies(hunan_obs, hunan_mask2d, cubes)

    # 4) Auto pre-registration (BEFORE any GX/GD evaluation)
    hunan_eff = {}
    raw_npz, rf_npz = z["raw_pred"], z["rf_pred"]
    for sub in np.unique(hunan_target["label"]):
        sel = hunan_target["label"] == sub
        hunan_eff[sub] = efficiency(hunan_target["y"][sel], rf_npz[sel], raw_npz[sel])
    log(f"hunan subregion efficiencies (NPZ): "
        + ", ".join(f"{k}={v:.1f}%" for k, v in sorted(hunan_eff.items())))
    preds = register_predictions(proxies, hunan_eff)

    # 5) Evaluation (LORO matrix + two arms at subregion level)
    evals = evaluate_all(models, targets)

    # 6) Tables, prediction check, figures
    summ = write_tables(proxies, evals)
    write_prediction_check(preds, evals)
    fig1(summ); fig2(evals); fig3(proxies, grids)
    log("figures written: Fig.7 / Fig.8 / Fig.S2 "
        "(fig7_coherence_vs_efficiency, fig8_two_arm_bars, figS2_morans_map)")
    log(f"=== done in {(time.perf_counter()-t0)/60:.1f} min; outputs in {OUT} ===")


if __name__ == "__main__":
    main()
