"""Train RF/LR correction models and write aligned NetCDF slices for plotting."""
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from core import project_config as config
import os
from pathlib import Path

def ensure_file_exists(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")

def find_var(ds, candidates, ds_name="dataset"):
    for v in candidates:
        if v in ds.data_vars:
            return ds[v]
    raise KeyError(
        f"{ds_name}: none of {candidates} in data_vars; available: {list(ds.data_vars)}"
    )

def to_dataarray(obj):
    if isinstance(obj, xr.Dataset):
        return obj[list(obj.data_vars)[0]]
    return obj

def standardize_latlon(da):
    for old, new in [("latitude","lat"),("longitude","lon"),("Latitude","lat"),("Longitude","lon")]:
        if old in da.coords:
            da = da.rename({old:new})
    return da.sortby("lat").sortby("lon")

def standardize_time(da):
    if "valid_time" in da.dims and "time" in da.dims:
        # Some ERA5 files carry both forecast reference time ("time")
        # and actual timestamp ("valid_time"). Keep one unique time axis.
        if da.sizes.get("time", 0) == 1:
            da = da.isel(time=0, drop=True).rename({"valid_time": "time"})
        elif da.sizes.get("valid_time", 0) == 1:
            da = da.isel(valid_time=0, drop=True)
        else:
            raise ValueError(
                "Invalid ERA5 time layout: both 'time' and 'valid_time' are non-singleton "
                f"(time={da.sizes.get('time')}, valid_time={da.sizes.get('valid_time')}). "
                "Please preprocess ERA5 so only one true time axis remains "
                "(e.g., collapse singleton forecast/reference dims, then rename valid_time->time)."
            )
    elif "valid_time" in da.dims:
        da = da.rename({"valid_time": "time"})
    return da

def ensure_latlon_order(da):
    if "lat" not in da.dims or "lon" not in da.dims:
        return da
    # Avoid ellipsis: xarray forbids "..." when dimension names repeat.
    lead_dims = tuple(d for d in da.dims if d not in ("lat", "lon"))
    return da.transpose(*(lead_dims + ("lat", "lon")))

def deduplicate_time(da, name="dataset"):
    if "time" not in da.dims:
        return da
    time_index = da.get_index("time")
    duplicated = time_index.duplicated(keep="first")
    if duplicated.any():
        keep_idx = np.where(~duplicated)[0]
        removed = int(duplicated.sum())
        print(f"{name}: removed {removed} duplicate time indices (kept first).")
        da = da.isel(time=keep_idx)
    return da

def write_with_clean_time_encoding(obj, path):
    encoding = {}
    if "time" in obj.coords:
        encoding["time"] = {
            "units": "days since 0001-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        }
    obj.to_netcdf(path, encoding=encoding if encoding else None)

def _grid_summary(da, name):
    lat = da.lat.values
    lon = da.lon.values
    dlat = float(np.median(np.diff(lat))) if lat.size > 1 else np.nan
    dlon = float(np.median(np.diff(lon))) if lon.size > 1 else np.nan
    lat_min = float(lat.min())
    lat_max = float(lat.max())
    lon_min = float(lon.min())
    lon_max = float(lon.max())
    return {
        "name": name,
        "shape": f"lat={lat.size}, lon={lon.size}",
        "lat_count": int(lat.size),
        "lon_count": int(lon.size),
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_range": f"[{lat_min:.4f}, {lat_max:.4f}]",
        "lon_range": f"[{lon_min:.4f}, {lon_max:.4f}]",
        "dlat": dlat,
        "dlon": dlon,
    }

def _coord_metadata(da, name):
    lat_attrs = da.lat.attrs if "lat" in da.coords else {}
    lon_attrs = da.lon.attrs if "lon" in da.coords else {}
    return {
        "dataset": name,
        "lat_dim": ",".join(da.lat.dims) if "lat" in da.coords else "",
        "lon_dim": ",".join(da.lon.dims) if "lon" in da.coords else "",
        "lat_standard_name": str(lat_attrs.get("standard_name", "")),
        "lon_standard_name": str(lon_attrs.get("standard_name", "")),
        "lat_axis": str(lat_attrs.get("axis", "")),
        "lon_axis": str(lon_attrs.get("axis", "")),
        "lat_units": str(lat_attrs.get("units", "")),
        "lon_units": str(lon_attrs.get("units", "")),
    }

def print_grid_alignment_report(dem, imerg, obs, u10, v10, tcwv, save_to_csv=True, report_dir=None):
    print("\n=== Grid alignment ===")
    summaries = [
        _grid_summary(dem, "DEM"),
        _grid_summary(imerg, "IMERG"),
        _grid_summary(obs, "OBS"),
        _grid_summary(u10, "U10"),
        _grid_summary(v10, "V10"),
        _grid_summary(tcwv, "TCWV"),
    ]
    for s in summaries:
        print(
            f"{s['name']:>6} | {s['shape']} | lat {s['lat_range']} | lon {s['lon_range']} | "
            f"dlat={s['dlat']:.4f}, dlon={s['dlon']:.4f}"
        )
    metadata_rows = [
        _coord_metadata(dem, "DEM"),
        _coord_metadata(imerg, "IMERG"),
        _coord_metadata(obs, "OBS"),
        _coord_metadata(u10, "U10"),
        _coord_metadata(v10, "V10"),
        _coord_metadata(tcwv, "TCWV"),
    ]
    print("--- Coordinate metadata ---")
    for m in metadata_rows:
        print(
            f"{m['dataset']:>6} | lat_dim={m['lat_dim']} ({m['lat_standard_name']},{m['lat_axis']},{m['lat_units']}) "
            f"| lon_dim={m['lon_dim']} ({m['lon_standard_name']},{m['lon_axis']},{m['lon_units']})"
        )

    target_lat = dem.lat.values
    target_lon = dem.lon.values
    offset_rows = []
    for da, name in [(imerg, "IMERG"), (obs, "OBS"), (u10, "U10"), (v10, "V10"), (tcwv, "TCWV")]:
        lat_offset = float(np.max(np.abs(da.lat.values - target_lat)))
        lon_offset = float(np.max(np.abs(da.lon.values - target_lon)))
        print(f"{name:>6} vs DEM max |Δlat|={lat_offset:.6f}, |Δlon|={lon_offset:.6f}")
        offset_rows.append({
            "source": name,
            "target": "DEM",
            "max_abs_dlat": lat_offset,
            "max_abs_dlon": lon_offset,
        })
    print("===\n")

    summary_rows = []
    for s in summaries:
        summary_rows.append({
            "section": "grid_summary",
            "dataset": s["name"],
            "lat_count": s["lat_count"],
            "lon_count": s["lon_count"],
            "lat_min": s["lat_min"],
            "lat_max": s["lat_max"],
            "lon_min": s["lon_min"],
            "lon_max": s["lon_max"],
            "dlat": float(s["dlat"]),
            "dlon": float(s["dlon"]),
            "target": "",
            "max_abs_dlat": np.nan,
            "max_abs_dlon": np.nan,
        })
    for row in offset_rows:
        summary_rows.append({
            "section": "grid_offset_vs_dem",
            "dataset": row["source"],
            "lat_count": np.nan,
            "lon_count": np.nan,
            "lat_min": np.nan,
            "lat_max": np.nan,
            "lon_min": np.nan,
            "lon_max": np.nan,
            "dlat": np.nan,
            "dlon": np.nan,
            "target": row["target"],
            "max_abs_dlat": row["max_abs_dlat"],
            "max_abs_dlon": row["max_abs_dlon"],
        })
    for m in metadata_rows:
        summary_rows.append({
            "section": "coord_metadata",
            "dataset": m["dataset"],
            "lat_count": np.nan,
            "lon_count": np.nan,
            "lat_min": np.nan,
            "lat_max": np.nan,
            "lon_min": np.nan,
            "lon_max": np.nan,
            "dlat": np.nan,
            "dlon": np.nan,
            "target": "",
            "max_abs_dlat": np.nan,
            "max_abs_dlon": np.nan,
            "lat_dim": m["lat_dim"],
            "lon_dim": m["lon_dim"],
            "lat_standard_name": m["lat_standard_name"],
            "lon_standard_name": m["lon_standard_name"],
            "lat_axis": m["lat_axis"],
            "lon_axis": m["lon_axis"],
            "lat_units": m["lat_units"],
            "lon_units": m["lon_units"],
        })
    if save_to_csv:
        report_df = pd.DataFrame(summary_rows)
        if report_dir is None:
            report_dir = getattr(
                config, "SUMMER_RESULTS_DIR", getattr(config, "SUMMER_MODELS_DIR", config.MODELS_DIR)
            )
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "grid_alignment_report.csv")
        report_df.to_csv(report_path, index=False)
        print(report_path)

def build_samples(imerg_da, u_da, v_da, q_da, dem_da, obs_da, time_mask, rng):
    X_list, y_list = [], []
    dem2d = dem_da.transpose("lat","lon").values
    for t_idx in np.where(time_mask)[0]:
        im = imerg_da.isel(time=t_idx).transpose("lat","lon").values
        u = u_da.isel(time=t_idx).transpose("lat","lon").values
        v = v_da.isel(time=t_idx).transpose("lat","lon").values
        q = q_da.isel(time=t_idx).transpose("lat","lon").values
        ob = obs_da.isel(time=t_idx).transpose("lat","lon").values
        mask = (np.isfinite(im) & np.isfinite(u) & np.isfinite(v) &
                np.isfinite(q) & np.isfinite(dem2d) & np.isfinite(ob))
        if mask.sum() == 0:
            continue
        X = np.stack([im[mask], u[mask], v[mask], q[mask], dem2d[mask]], axis=1)
        y = ob[mask]
        if len(y) > config.MAX_SAMPLES_PER_TIME:
            idx = rng.choice(len(y), config.MAX_SAMPLES_PER_TIME, replace=False)
            X, y = X[idx], y[idx]
        X_list.append(X)
        y_list.append(y)
    if not X_list:
        raise ValueError(
            "Empty training sample: check time range, variable names, and valid-data mask."
        )
    return np.concatenate(X_list), np.concatenate(y_list)

def build_samples_with_years(imerg_da, u_da, v_da, q_da, dem_da, obs_da, time_mask, time_coord, rng):
    X_list, y_list, year_list, time_list = [], [], [], []
    lat_list, lon_list = [], []
    dem2d = dem_da.transpose("lat","lon").values
    lat_axis = np.asarray(dem_da.lat.values, dtype=float)
    lon_axis = np.asarray(dem_da.lon.values, dtype=float)
    for t_idx in np.where(time_mask)[0]:
        im = imerg_da.isel(time=t_idx).transpose("lat","lon").values
        u = u_da.isel(time=t_idx).transpose("lat","lon").values
        v = v_da.isel(time=t_idx).transpose("lat","lon").values
        q = q_da.isel(time=t_idx).transpose("lat","lon").values
        ob = obs_da.isel(time=t_idx).transpose("lat","lon").values
        mask = (np.isfinite(im) & np.isfinite(u) & np.isfinite(v) &
                np.isfinite(q) & np.isfinite(dem2d) & np.isfinite(ob))
        if mask.sum() == 0:
            continue
        nlat, nlon = ob.shape
        lat_grid = np.broadcast_to(lat_axis.reshape(-1, 1), (nlat, nlon))
        lon_grid = np.broadcast_to(lon_axis.reshape(1, -1), (nlat, nlon))
        lat_m = lat_grid[mask]
        lon_m = lon_grid[mask]
        X = np.stack([im[mask], u[mask], v[mask], q[mask], dem2d[mask]], axis=1)
        y = ob[mask]
        if len(y) > config.MAX_SAMPLES_PER_TIME:
            idx = rng.choice(len(y), config.MAX_SAMPLES_PER_TIME, replace=False)
            X, y = X[idx], y[idx]
            lat_m = lat_m[idx]
            lon_m = lon_m[idx]
        year_val = time_coord.isel(time=t_idx).dt.year.values
        time_val = np.datetime64(time_coord.isel(time=t_idx).values)
        year_arr = np.full(len(y), year_val)
        time_arr = np.full(len(y), time_val, dtype="datetime64[ns]")
        X_list.append(X)
        y_list.append(y)
        year_list.append(year_arr)
        time_list.append(time_arr)
        lat_list.append(lat_m)
        lon_list.append(lon_m)
    if not X_list:
        raise ValueError(
            "Empty test sample: check TEST_YEARS, SUMMER_MONTHS, and time alignment."
        )
    return (
        np.concatenate(X_list),
        np.concatenate(y_list),
        np.concatenate(year_list),
        np.concatenate(time_list),
        np.concatenate(lat_list),
        np.concatenate(lon_list),
    )


def _load_valid_rf_model(model_path, feature_count, season_label):
    """Load RF model only when it is structurally usable for current features."""
    if not os.path.isfile(model_path):
        return None, "model file not found"
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        return None, f"failed to load model: {exc}"
    if not hasattr(model, "predict"):
        return None, "loaded object has no predict()"

    n_features_in = getattr(model, "n_features_in_", None)
    if n_features_in is not None and int(n_features_in) != int(feature_count):
        return None, (
            f"feature mismatch: model expects {int(n_features_in)}, "
            f"current pipeline uses {int(feature_count)}"
        )

    try:
        dummy_x = np.zeros((1, int(feature_count)), dtype=float)
        pred = np.asarray(model.predict(dummy_x)).ravel()
        if pred.size != 1 or not np.isfinite(pred[0]):
            return None, "predict() sanity check failed"
    except Exception as exc:
        return None, f"predict() failed: {exc}"

    print(f"[train_model] RF-{season_label}: using existing model -> {model_path}")
    return model, None


def _load_or_fit_rf_model(model_path, x_train, y_train, season_label, random_state):
    """Use existing valid RF model, otherwise retrain and overwrite."""
    feature_count = x_train.shape[1]
    model, reason = _load_valid_rf_model(model_path, feature_count, season_label)
    if model is not None:
        return model

    print(f"[train_model] RF-{season_label}: rebuild model ({reason})")
    model = RandomForestRegressor(
        n_estimators=config.N_ESTIMATORS,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[train_model] RF-{season_label}: saved -> {model_path}")
    return model


def recompute_test_lat_lon_match_npz(npz_path: str | os.PathLike[str] | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Rebuild ``lat_test``/``lon_test`` row order via the same sampling pipeline and ``np.random.seed`` as ``train()``,
    so older NPZ files (saved before lat/lon were stored) still align with ``y_test`` / predictions CSV.
    """
    npz_p = Path(os.path.expanduser(npz_path or config.DATA_NPZ_PATH))
    if not npz_p.is_file():
        print(f"[train_model] recompute_test_lat_lon: NPZ not found: {npz_p}")
        return None
    with np.load(npz_p) as z:
        if "y_test" not in z.files:
            print("[train_model] recompute_test_lat_lon: NPZ missing y_test")
            return None
        n_expect = int(np.asarray(z["y_test"]).ravel().size)

    test_rng = np.random.RandomState(config.RANDOM_STATE + 1)

    try:
        ensure_file_exists(config.IMERG_DIR, "IMERG_FILE")
        ensure_file_exists(config.OBS_FILE, "OBS_FILE")
        ensure_file_exists(config.ERA5_FILE, "ERA5_FILE")
        ensure_file_exists(config.DEM_FILE, "DEM_FILE")
    except FileNotFoundError as exc:
        print(f"[train_model] recompute_test_lat_lon: {exc}")
        return None

    imerg_ds = xr.open_dataset(config.IMERG_DIR)
    obs_ds = xr.open_dataset(config.OBS_FILE)
    era5_ds = xr.open_dataset(config.ERA5_FILE)
    dem_ds = xr.open_dataset(config.DEM_FILE)
    try:
        imerg = ensure_latlon_order(
            standardize_time(standardize_latlon(to_dataarray(find_var(imerg_ds, config.IMERG_VAR_CANDIDATES, "IMERG"))))
        )
        obs = ensure_latlon_order(
            standardize_time(standardize_latlon(to_dataarray(find_var(obs_ds, config.OBS_VAR_CANDIDATES, "OBS"))))
        )

        def clean_era5(da):
            da = standardize_time(standardize_latlon(to_dataarray(da)))
            for dim in ["number", "expver"]:
                if dim in da.dims:
                    da = da.isel({dim: 0}, drop=True)
            return ensure_latlon_order(da)

        u10 = clean_era5(find_var(era5_ds, config.U10_VAR_CANDIDATES, "U10"))
        v10 = clean_era5(find_var(era5_ds, config.V10_VAR_CANDIDATES, "V10"))
        tcwv = clean_era5(find_var(era5_ds, config.TCWV_VAR_CANDIDATES, "TCWV"))

        dem = ensure_latlon_order(
            standardize_latlon(to_dataarray(find_var(dem_ds, config.DEM_VAR_CANDIDATES, "DEM")))
        )
        if "time" in dem.dims:
            dem = dem.isel(time=0, drop=True)
        dem = dem.transpose("lat", "lon")

        target_lon = np.arange(
            config.TARGET_LON_MIN,
            config.TARGET_LON_MAX + config.TARGET_RES / 2.0,
            config.TARGET_RES,
        )
        target_lat = np.arange(
            config.TARGET_LAT_MIN,
            config.TARGET_LAT_MAX + config.TARGET_RES / 2.0,
            config.TARGET_RES,
        )

        imerg = imerg.interp(lat=target_lat, lon=target_lon, method="nearest")
        obs = obs.interp(lat=target_lat, lon=target_lon, method="nearest")
        u10 = u10.interp(lat=target_lat, lon=target_lon, method="linear")
        v10 = v10.interp(lat=target_lat, lon=target_lon, method="linear")
        tcwv = tcwv.interp(lat=target_lat, lon=target_lon, method="linear")
        dem = dem.interp(lat=target_lat, lon=target_lon, method="nearest")

        imerg = deduplicate_time(imerg, "IMERG")
        obs = deduplicate_time(obs, "OBS")
        u10 = deduplicate_time(u10, "U10")
        v10 = deduplicate_time(v10, "V10")
        tcwv = deduplicate_time(tcwv, "TCWV")

        common = imerg.time.values
        for da in [obs, u10, v10, tcwv]:
            common = np.intersect1d(common, da.time.values)
        if len(common) == 0:
            print("[train_model] recompute_test_lat_lon: empty time intersection")
            return None
        imerg = imerg.sel(time=common)
        obs = obs.sel(time=common)
        u10 = u10.sel(time=common)
        v10 = v10.sel(time=common)
        tcwv = tcwv.sel(time=common)

        test_mask = (imerg.time.dt.year.isin(config.TEST_YEARS)) & (
            imerg.time.dt.month.isin(config.SUMMER_MONTHS)
        )

        _x_unused, _y_unused, _years_unused, _time_unused, lat_test, lon_test = build_samples_with_years(
            imerg, u10, v10, tcwv, dem, obs, test_mask.values, imerg.time, test_rng
        )
    finally:
        imerg_ds.close()
        obs_ds.close()
        era5_ds.close()
        dem_ds.close()

    if lat_test.size != n_expect:
        print(
            f"[train_model] recompute_test_lat_lon: got {lat_test.size} samples vs NPZ y_test {n_expect} "
            "(training config or input files may differ from when the NPZ was written)."
        )
        return None
    return lat_test, lon_test


def train():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    train_rng = np.random.RandomState(config.RANDOM_STATE)
    test_rng = np.random.RandomState(config.RANDOM_STATE + 1)
    spring_rng = np.random.RandomState(config.RANDOM_STATE + 2)

    ensure_file_exists(config.IMERG_DIR, "IMERG_FILE")
    ensure_file_exists(config.OBS_FILE, "OBS_FILE")
    ensure_file_exists(config.ERA5_FILE, "ERA5_FILE")
    ensure_file_exists(config.DEM_FILE, "DEM_FILE")

    imerg_ds = xr.open_dataset(config.IMERG_DIR)
    imerg = ensure_latlon_order(standardize_time(standardize_latlon(
        to_dataarray(find_var(imerg_ds, config.IMERG_VAR_CANDIDATES, "IMERG")))))

    obs_ds = xr.open_dataset(config.OBS_FILE)
    obs = ensure_latlon_order(standardize_time(standardize_latlon(
        to_dataarray(find_var(obs_ds, config.OBS_VAR_CANDIDATES, "OBS")))))

    era5_ds = xr.open_dataset(config.ERA5_FILE)
    def clean_era5(da):
        da = standardize_time(standardize_latlon(to_dataarray(da)))
        for dim in ["number","expver"]:
            if dim in da.dims:
                da = da.isel({dim:0}, drop=True)
        return ensure_latlon_order(da)
    u10 = clean_era5(find_var(era5_ds, config.U10_VAR_CANDIDATES, "U10"))
    v10 = clean_era5(find_var(era5_ds, config.V10_VAR_CANDIDATES, "V10"))
    tcwv = clean_era5(find_var(era5_ds, config.TCWV_VAR_CANDIDATES, "TCWV"))

    dem_ds = xr.open_dataset(config.DEM_FILE)
    dem = ensure_latlon_order(standardize_latlon(to_dataarray(
        find_var(dem_ds, config.DEM_VAR_CANDIDATES, "DEM"))))
    if "time" in dem.dims:
        dem = dem.isel(time=0, drop=True)
    dem = dem.transpose("lat","lon")

    target_lon = np.arange(
        config.TARGET_LON_MIN,
        config.TARGET_LON_MAX + config.TARGET_RES / 2.0,
        config.TARGET_RES,
    )
    target_lat = np.arange(
        config.TARGET_LAT_MIN,
        config.TARGET_LAT_MAX + config.TARGET_RES / 2.0,
        config.TARGET_RES,
    )

    imerg = imerg.interp(lat=target_lat, lon=target_lon, method="nearest")
    obs   = obs.interp(lat=target_lat, lon=target_lon, method="nearest")
    u10   = u10.interp(lat=target_lat, lon=target_lon, method="linear")
    v10   = v10.interp(lat=target_lat, lon=target_lon, method="linear")
    tcwv  = tcwv.interp(lat=target_lat, lon=target_lon, method="linear")
    dem   = dem.interp(lat=target_lat, lon=target_lon, method="nearest")

    imerg = deduplicate_time(imerg, "IMERG")
    obs = deduplicate_time(obs, "OBS")
    u10 = deduplicate_time(u10, "U10")
    v10 = deduplicate_time(v10, "V10")
    tcwv = deduplicate_time(tcwv, "TCWV")

    common = imerg.time.values
    for da in [obs, u10, v10, tcwv]:
        common = np.intersect1d(common, da.time.values)
    if len(common) == 0:
        raise ValueError(
            "Empty time intersection for IMERG/OBS/ERA5: check `time` coverage in each file."
        )
    imerg = imerg.sel(time=common)
    obs   = obs.sel(time=common)
    u10   = u10.sel(time=common)
    v10   = v10.sel(time=common)
    tcwv  = tcwv.sel(time=common)

    train_summer_model = bool(getattr(config, "TRAIN_SUMMER_MODEL", True))
    train_spring_model = bool(getattr(config, "TRAIN_SPRING_MODEL", True))
    if not train_summer_model and not train_spring_model:
        print(
            "[train_model] No model training requested: "
            "TRAIN_SUMMER_MODEL=False and TRAIN_SPRING_MODEL=False."
        )
        return

    report_dir = (
        getattr(config, "SUMMER_RESULTS_DIR", getattr(config, "SUMMER_MODELS_DIR", config.MODELS_DIR))
        if train_summer_model
        else getattr(config, "SPRING_RESULTS_DIR", getattr(config, "SPRING_MODELS_DIR", config.MODELS_DIR))
    )
    print_grid_alignment_report(
        dem, imerg, obs, u10, v10, tcwv, save_to_csv=True, report_dir=report_dir
    )

    feature_names = np.array(["imerg", "u10", "v10", "tcwv", "dem"], dtype="<U10")

    # Summer split
    summer_train_mask = (imerg.time.dt.year.isin(config.TRAIN_YEARS) &
                         imerg.time.dt.month.isin(config.SUMMER_MONTHS))
    summer_test_mask = (imerg.time.dt.year.isin(config.TEST_YEARS) &
                        imerg.time.dt.month.isin(config.SUMMER_MONTHS))
    X_summer_train = y_summer_train = None
    X_summer_test = y_summer_test = None
    years_summer_test = time_summer_test = lat_summer_test = lon_summer_test = None
    if train_summer_model:
        print(
            f"[train_model] Summer split: train_days={int(summer_train_mask.sum())} "
            f"test_days={int(summer_test_mask.sum())}"
        )
        X_summer_train, y_summer_train = build_samples(
            imerg, u10, v10, tcwv, dem, obs, summer_train_mask.values, train_rng
        )
        (
            X_summer_test,
            y_summer_test,
            years_summer_test,
            time_summer_test,
            lat_summer_test,
            lon_summer_test,
        ) = build_samples_with_years(
            imerg, u10, v10, tcwv, dem, obs, summer_test_mask.values, imerg.time, test_rng
        )

    # Spring split
    spring_months = getattr(config, "SPRING_MONTHS", [3, 4, 5])
    spring_train_mask = (imerg.time.dt.year.isin(config.TRAIN_YEARS) &
                         imerg.time.dt.month.isin(spring_months))
    spring_test_mask = (imerg.time.dt.year.isin(config.TEST_YEARS) &
                        imerg.time.dt.month.isin(spring_months))
    X_spring_train = y_spring_train = None
    X_spring_test = y_spring_test = None
    years_spring_test = time_spring_test = lat_spring_test = lon_spring_test = None
    if train_spring_model:
        print(
            f"[train_model] Spring split: train_days={int(spring_train_mask.sum())} "
            f"test_days={int(spring_test_mask.sum())}"
        )
        if int(spring_train_mask.sum()) > 0:
            X_spring_train, y_spring_train = build_samples(
                imerg, u10, v10, tcwv, dem, obs, spring_train_mask.values, spring_rng
            )
        if int(spring_test_mask.sum()) > 0:
            (
                X_spring_test,
                y_spring_test,
                years_spring_test,
                time_spring_test,
                lat_spring_test,
                lon_spring_test,
            ) = build_samples_with_years(
                imerg, u10, v10, tcwv, dem, obs, spring_test_mask.values, imerg.time, test_rng
            )

    # Train / load summer model
    rf_summer_path = getattr(config, "RF_SUMMER_MODEL_PATH", config.RF_MODEL_PATH)
    rf = lr = None
    if train_summer_model:
        rf = _load_or_fit_rf_model(
            rf_summer_path,
            X_summer_train,
            y_summer_train,
            season_label="Summer",
            random_state=config.RANDOM_STATE,
        )
        lr = LinearRegression().fit(X_summer_train, y_summer_train)
    else:
        print("[train_model] RF-Summer training skipped by config (TRAIN_SUMMER_MODEL=False).")

    # Train / load spring model
    rf_spring_path = getattr(
        config, "RF_SPRING_MODEL_PATH", os.path.join(config.MODELS_DIR, "rf_model_spring.joblib")
    )
    rf_spring = lr_spring = None
    if not train_spring_model:
        print("[train_model] RF-Spring training skipped by config (TRAIN_SPRING_MODEL=False).")
    elif int(spring_train_mask.sum()) == 0 or X_spring_train is None:
        months_avail = np.unique(imerg.time.dt.month.values).astype(int).tolist()
        rf_spring, reason = _load_valid_rf_model(rf_spring_path, 5, "Spring")
        if rf_spring is None:
            print(
                "[train_model] RF-Spring skipped: no train samples and no valid existing model. "
                f"SPRING_MONTHS={spring_months}, TRAIN_YEARS={config.TRAIN_YEARS}, "
                f"available months={months_avail}. reason={reason}"
            )
        else:
            print("[train_model] RF-Spring: no new spring samples, kept existing spring model.")
    else:
        rf_spring = _load_or_fit_rf_model(
            rf_spring_path,
            X_spring_train,
            y_spring_train,
            season_label="Spring",
            random_state=config.RANDOM_STATE + 10,
        )
        lr_spring = LinearRegression().fit(X_spring_train, y_spring_train)

    if train_summer_model:
        rf_pred = rf.predict(X_summer_test)
        lr_pred = lr.predict(X_summer_test)
        raw_pred = X_summer_test[:, 0]

        # Summer model is the default model consumed by downstream scripts.
        joblib.dump(rf, rf_summer_path)
        legacy_rf_path = getattr(config, "RF_MODEL_PATH", rf_summer_path)
        if legacy_rf_path != rf_summer_path:
            os.makedirs(os.path.dirname(legacy_rf_path), exist_ok=True)
            joblib.dump(rf, legacy_rf_path)
            print(f"[train_model] RF-Summer: wrote compatibility alias -> {legacy_rf_path}")
        joblib.dump(lr, config.LR_MODEL_PATH)
        np.savez(
            config.DATA_NPZ_PATH,
            X_train=X_summer_train,
            y_train=y_summer_train,
            X_test=X_summer_test,
            y_test=y_summer_test,
            raw_pred=raw_pred,
            rf_pred=rf_pred,
            lr_pred=lr_pred,
            feature_names=feature_names,
            years_test=years_summer_test,
            time_test=time_summer_test,
            lat_test=lat_summer_test,
            lon_test=lon_summer_test,
        )

        unique_years = np.unique(years_summer_test)
        year_rows = []
        for yr in unique_years:
            mask_yr = years_summer_test == yr
            yt = y_summer_test[mask_yr]
            rawp = raw_pred[mask_yr]
            rfp = rf_pred[mask_yr]
            lrp = lr_pred[mask_yr]
            year_rows.append([
                int(yr), int(mask_yr.sum()),
                r2_score(yt, rawp), np.sqrt(mean_squared_error(yt, rawp)),
                mean_absolute_error(yt, rawp), np.mean(rawp - yt),
                r2_score(yt, lrp), np.sqrt(mean_squared_error(yt, lrp)),
                mean_absolute_error(yt, lrp), np.mean(lrp - yt),
                r2_score(yt, rfp), np.sqrt(mean_squared_error(yt, rfp)),
                mean_absolute_error(yt, rfp), np.mean(rfp - yt)
            ])
        year_df = pd.DataFrame(year_rows, columns=[
            "Year", "N",
            "IMERG_R2", "IMERG_RMSE", "IMERG_MAE", "IMERG_Bias",
            "LR_R2", "LR_RMSE", "LR_MAE", "LR_Bias",
            "RF_R2", "RF_RMSE", "RF_MAE", "RF_Bias"
        ])
        print(year_df.round(3))
        summer_results_dir = getattr(
            config, "SUMMER_RESULTS_DIR", getattr(config, "SUMMER_MODELS_DIR", config.MODELS_DIR)
        )
        os.makedirs(summer_results_dir, exist_ok=True)
        year_df.to_csv(os.path.join(summer_results_dir, "per_year_metrics.csv"), index=False)

        dem.to_netcdf(config.DEM_REF_NC)
        write_with_clean_time_encoding(imerg.to_dataset(name="imerg"), config.IMERG_NC)
        write_with_clean_time_encoding(obs.to_dataset(name="obs"), config.OBS_NC)
        write_with_clean_time_encoding(u10.to_dataset(name="u10"), config.U10_NC)
        write_with_clean_time_encoding(v10.to_dataset(name="v10"), config.V10_NC)
        write_with_clean_time_encoding(tcwv.to_dataset(name="tcwv"), config.TCWV_NC)
        time_index_ds = xr.Dataset(coords={"time": imerg.time})
        write_with_clean_time_encoding(time_index_ds, config.TIME_NC)
    else:
        print("[train_model] Summer artifacts skipped because TRAIN_SUMMER_MODEL=False.")

    if train_spring_model and rf_spring is not None and lr_spring is not None and X_spring_test is not None:
        rf_pred_spring = rf_spring.predict(X_spring_test)
        lr_pred_spring = lr_spring.predict(X_spring_test)
        raw_pred_spring = X_spring_test[:, 0]

        spring_lr_path = getattr(
            config, "SPRING_LR_MODEL_PATH",
            os.path.join(getattr(config, "SPRING_MODELS_DIR", config.MODELS_DIR), "lr_model.joblib"),
        )
        spring_npz_path = getattr(
            config, "SPRING_DATA_NPZ_PATH",
            os.path.join(getattr(config, "SPRING_MODELS_DIR", config.MODELS_DIR), "train_test_data.npz"),
        )
        spring_dem_ref_nc = getattr(config, "SPRING_DEM_REF_NC", getattr(config, "DEM_REF_NC", "dem_reference.nc"))
        spring_time_nc = getattr(config, "SPRING_TIME_NC", getattr(config, "TIME_NC", "time_index.nc"))
        spring_obs_nc = getattr(config, "SPRING_OBS_NC", getattr(config, "OBS_NC", "obs_for_plot.nc"))
        spring_imerg_nc = getattr(config, "SPRING_IMERG_NC", getattr(config, "IMERG_NC", "imerg_for_plot.nc"))
        spring_u10_nc = getattr(config, "SPRING_U10_NC", getattr(config, "U10_NC", "u10_for_plot.nc"))
        spring_v10_nc = getattr(config, "SPRING_V10_NC", getattr(config, "V10_NC", "v10_for_plot.nc"))
        spring_tcwv_nc = getattr(config, "SPRING_TCWV_NC", getattr(config, "TCWV_NC", "tcwv_for_plot.nc"))

        joblib.dump(lr_spring, spring_lr_path)
        np.savez(
            spring_npz_path,
            X_train=X_spring_train,
            y_train=y_spring_train,
            X_test=X_spring_test,
            y_test=y_spring_test,
            raw_pred=raw_pred_spring,
            rf_pred=rf_pred_spring,
            lr_pred=lr_pred_spring,
            feature_names=feature_names,
            years_test=years_spring_test,
            time_test=time_spring_test,
            lat_test=lat_spring_test,
            lon_test=lon_spring_test,
        )

        unique_years = np.unique(years_spring_test)
        year_rows = []
        for yr in unique_years:
            mask_yr = years_spring_test == yr
            yt = y_spring_test[mask_yr]
            rawp = raw_pred_spring[mask_yr]
            rfp = rf_pred_spring[mask_yr]
            lrp = lr_pred_spring[mask_yr]
            year_rows.append([
                int(yr), int(mask_yr.sum()),
                r2_score(yt, rawp), np.sqrt(mean_squared_error(yt, rawp)),
                mean_absolute_error(yt, rawp), np.mean(rawp - yt),
                r2_score(yt, lrp), np.sqrt(mean_squared_error(yt, lrp)),
                mean_absolute_error(yt, lrp), np.mean(lrp - yt),
                r2_score(yt, rfp), np.sqrt(mean_squared_error(yt, rfp)),
                mean_absolute_error(yt, rfp), np.mean(rfp - yt)
            ])
        year_df_spring = pd.DataFrame(year_rows, columns=[
            "Year", "N",
            "IMERG_R2", "IMERG_RMSE", "IMERG_MAE", "IMERG_Bias",
            "LR_R2", "LR_RMSE", "LR_MAE", "LR_Bias",
            "RF_R2", "RF_RMSE", "RF_MAE", "RF_Bias"
        ])
        print(year_df_spring.round(3))
        spring_results_dir = getattr(
            config, "SPRING_RESULTS_DIR", getattr(config, "SPRING_MODELS_DIR", config.MODELS_DIR)
        )
        os.makedirs(spring_results_dir, exist_ok=True)
        year_df_spring.to_csv(os.path.join(spring_results_dir, "per_year_metrics.csv"), index=False)
        print_grid_alignment_report(
            dem, imerg, obs, u10, v10, tcwv, save_to_csv=True, report_dir=spring_results_dir
        )

        dem.to_netcdf(spring_dem_ref_nc)
        write_with_clean_time_encoding(imerg.to_dataset(name="imerg"), spring_imerg_nc)
        write_with_clean_time_encoding(obs.to_dataset(name="obs"), spring_obs_nc)
        write_with_clean_time_encoding(u10.to_dataset(name="u10"), spring_u10_nc)
        write_with_clean_time_encoding(v10.to_dataset(name="v10"), spring_v10_nc)
        write_with_clean_time_encoding(tcwv.to_dataset(name="tcwv"), spring_tcwv_nc)
        time_index_ds = xr.Dataset(coords={"time": imerg.time})
        write_with_clean_time_encoding(time_index_ds, spring_time_nc)
    elif train_spring_model:
        print("[train_model] Spring artifacts skipped: missing spring train/test samples.")

    print(config.MODELS_DIR)
    print(rf_summer_path)
    if os.path.isfile(rf_spring_path):
        print(rf_spring_path)

if __name__ == "__main__":
    train()