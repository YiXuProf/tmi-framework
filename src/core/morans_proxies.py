"""Shared Moran's I proxy utilities (Table S5 / xreg_validation).

Definition (manuscript-reproducible):
- Daily Moran's I of observed precipitation on a queen-contiguity, row-standardized W
- Averaged over JJA days in PROXY_YEARS (= TEST_YEARS, default 2021–2022)
- No deseasonalization / anomaly transform
- Cells must be finite on every proxy day (persistent_mask)
- Hunan mask prefers dem_reference.nc finite cells; else province geojson
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib.path import Path as MplPath

from core import project_config as config


PROXY_YEARS = list(config.TEST_YEARS)
SUMMER_MONTHS = list(config.SUMMER_MONTHS)

HUNAN_SUBREGIONS = ["west", "north", "south", "central"]
HUNAN_SUBREGION_LABELS = {
    "west": "West Hunan",
    "north": "North Hunan",
    "south": "South Hunan",
    "central": "Central Hunan",
}

HUNAN_BOUNDARY_CANDIDATES = [
    Path(config.BASE_DIR) / "assets" / "hunan.geojson",
    Path(config.DATA_ROOT) / "hunan" / "hunan.geojson",
]


def load_boundary_gdf(path: Path):
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf


def boundary_mask_from_gdf(lat_vals, lon_vals, gdf) -> np.ndarray:
    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
    pts = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    mask = np.zeros(pts.shape[0], dtype=bool)
    for geom in gdf.geometry:
        if geom is None:
            continue
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            ext = np.asarray(poly.exterior.coords)
            inside = MplPath(ext).contains_points(pts)
            for hole in poly.interiors:
                inside &= ~MplPath(np.asarray(hole.coords)).contains_points(pts)
            mask |= inside
    return mask.reshape(lat2d.shape)


def label_hunan_grid(lat2d, lon2d) -> np.ndarray:
    """Rule-based Hunan subregions (west wins over N/S)."""
    lab = np.full(lat2d.shape, "central", dtype=object)
    lab[lat2d < 27.0] = "south"
    lab[lat2d >= 28.0] = "north"
    lab[lon2d < 110.5] = "west"
    return lab


def queen_W(mask2d: np.ndarray):
    cells = [tuple(c) for c in np.argwhere(mask2d)]
    loc = {c: k for k, c in enumerate(cells)}
    n = len(cells)
    W = np.zeros((n, n))
    for k, (i, j) in enumerate(cells):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                m = loc.get((i + di, j + dj))
                if m is not None:
                    W[k, m] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    W /= rs
    return W, float(W.sum())


def persistent_mask(obs_da: xr.DataArray, years, mask2d: np.ndarray) -> np.ndarray:
    """Cells of mask2d that are finite on EVERY proxy day."""
    tm = (
        obs_da.time.dt.year.isin(list(years))
        & obs_da.time.dt.month.isin(SUMMER_MONTHS)
    ).values
    data = obs_da.isel(time=np.where(tm)[0]).transpose("time", "lat", "lon").values
    return mask2d & np.isfinite(data).all(axis=0)


def morans_i_mean(obs_da: xr.DataArray, years, mask2d: np.ndarray):
    """Mean daily Moran's I (+ SE over days) of obs on masked grid, JJA of `years`."""
    tm = (
        obs_da.time.dt.year.isin(list(years))
        & obs_da.time.dt.month.isin(SUMMER_MONTHS)
    ).values
    mask2d = persistent_mask(obs_da, years, mask2d)
    if int(mask2d.sum()) < 9:
        return np.nan, np.nan, 0
    W, S0 = queen_W(mask2d)
    data = obs_da.isel(time=np.where(tm)[0]).transpose("time", "lat", "lon").values
    vals = []
    for k in range(data.shape[0]):
        f = data[k][mask2d]
        if not np.isfinite(f).all():
            continue
        z = f - f.mean()
        denom = float(z @ z)
        if denom == 0:
            continue
        vals.append(float((f.size / S0) * (z @ W @ z) / denom))
    if not vals:
        return np.nan, np.nan, 0
    vals = np.asarray(vals)
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(vals.size)), int(vals.size)


def load_hunan_obs() -> xr.DataArray:
    obs_nc = Path(getattr(config, "OBS_NC", ""))
    if not obs_nc.is_file():
        raise FileNotFoundError(f"Hunan aligned obs not found: {obs_nc}")
    return xr.open_dataset(obs_nc)["obs"]


def load_hunan_mask(obs_da: xr.DataArray) -> tuple[np.ndarray, str]:
    """Prefer dem_reference finite cells; fallback to province geojson / full rectangle."""
    dem_ref = Path(getattr(config, "DEM_REF_NC", ""))
    if dem_ref.is_file():
        dset = xr.open_dataset(dem_ref)
        darr = dset[list(dset.data_vars)[0]]
        if darr.ndim > 2:
            darr = darr.isel({d: 0 for d in darr.dims if d not in ("lat", "lon")})
        darr = darr.transpose("lat", "lon")
        if darr.shape == (obs_da.lat.size, obs_da.lon.size):
            return np.isfinite(darr.values), f"dem_reference:{dem_ref}"
    hb = next((p for p in HUNAN_BOUNDARY_CANDIDATES if p.is_file()), None)
    if hb is not None:
        mask = boundary_mask_from_gdf(
            obs_da.lat.values, obs_da.lon.values, load_boundary_gdf(hb)
        )
        return mask, f"geojson:{hb}"
    return (
        np.ones((obs_da.lat.size, obs_da.lon.size), dtype=bool),
        "full_rectangle",
    )


def compute_hunan_subregion_morans(
    years=None,
) -> tuple[list[dict], str]:
    """Compute Table-S5 rows for the four Hunan subregions."""
    years = list(years) if years is not None else list(PROXY_YEARS)
    obs = load_hunan_obs()
    mask2d, mask_src = load_hunan_mask(obs)
    lon2d, lat2d = np.meshgrid(obs.lon.values, obs.lat.values)
    lab2d = label_hunan_grid(lat2d, lon2d)

    rows = []
    for sub in HUNAN_SUBREGIONS:
        m = persistent_mask(obs, years, mask2d & (lab2d == sub))
        if int(m.sum()) < 9:
            rows.append(
                {
                    "subregion": sub,
                    "subregion_label": HUNAN_SUBREGION_LABELS[sub],
                    "morans_i": np.nan,
                    "morans_i_se": np.nan,
                    "obs_variance": np.nan,
                    "n_days": 0,
                    "n_cells": int(m.sum()),
                    "years": f"{min(years)}-{max(years)}",
                    "season": "JJA",
                    "mask_source": mask_src,
                }
            )
            continue
        i_mean, i_se, nd = morans_i_mean(obs, years, m)
        tm = (
            obs.time.dt.year.isin(years) & obs.time.dt.month.isin(SUMMER_MONTHS)
        ).values
        var = float(np.nanvar(obs.isel(time=tm).values[:, m]))
        rows.append(
            {
                "subregion": sub,
                "subregion_label": HUNAN_SUBREGION_LABELS[sub],
                "morans_i": i_mean,
                "morans_i_se": i_se,
                "obs_variance": var,
                "n_days": nd,
                "n_cells": int(m.sum()),
                "years": f"{min(years)}-{max(years)}",
                "season": "JJA",
                "mask_source": mask_src,
            }
        )
    return rows, mask_src
