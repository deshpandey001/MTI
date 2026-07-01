"""
preprocess.py
=============
Preprocess NOAA OISST and individual CMIP6 model SST data.

Each CMIP6 model is processed independently (never merged).  Output is one
NetCDF file per model in ``data/processed/``, plus per-model quick-look
figures in ``outputs/preprocessing/<model_name>/``.

Usage
-----
    python scripts/preprocess.py          # NOAA + all CMIP6 models
"""

from __future__ import annotations

import logging
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
NOAA_DIR = DATA_DIR / "noaa"
CMIP6_DIR = DATA_DIR / "cmip6_historical"
CMIP6_FUTURE_DIR = DATA_DIR / "cmip6_future"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "preprocessing"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SST_VARIABLE_NAMES = {"sst", "tos", "thetao"}
COORD_RENAME_MAP: dict[str, str] = {
    "latitude": "lat",
    "nav_lat": "lat",
    "longitude": "lon",
    "nav_lon": "lon",
}
INDIAN_OCEAN = dict(lat=slice(-40.0, 30.0), lon=slice(20.0, 120.0))
SST_RANGE = (-2.0, 40.0)
CMIP6_TARGET_RESOLUTION = 0.5

logger = logging.getLogger("preprocess")

# CMIP6 filename pattern:  <variable>_<mip>_<model_id>_<experiment>_...
# Model name is the 3rd underscore-delimited field.
CMIP6_FILENAME_RE = re.compile(
    r"^[^_]+_[^_]+_(?P<model>[^_]+(?:-[^_]+)*)_"
)


# ===================================================================
#  Logging helpers
# ===================================================================

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-12s | %(levelname)-6s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _log_done(name: str, sec: float) -> None:
    logger.info("%s  …  done  (%.2f s)", name, sec)


# ===================================================================
#  File discovery
# ===================================================================

def find_netcdf_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {directory}")
    logger.info("Found %d NetCDF file(s) in %s", len(files), directory)
    return files


def extract_model_name(filename: str) -> str:
    """Extract CMIP6 model ID from a standard CMIP6 filename.

    Example
    -------
    >>> extract_model_name("tos_Omon_EC-Earth3-CC_historical_r1i1p1f1_gn_19820116-20141216.nc")
    'EC-Earth3-CC'
    """
    m = CMIP6_FILENAME_RE.match(filename)
    if m is None:
        raise ValueError(f"Cannot parse model name from filename: {filename}")
    return m.group("model")


def discover_cmip6_models(directory: Path) -> list[tuple[str, Path]]:
    """Return list of (model_name, filepath) for each CMIP6 file found."""
    files = find_netcdf_files(directory)
    models: list[tuple[str, Path]] = []
    for fpath in files:
        name = extract_model_name(fpath.name)
        models.append((name, fpath))
    return models


# ===================================================================
#  Cleanup — remove old merged-pipeline outputs
# ===================================================================

def _clean_old_merged_outputs() -> None:
    """Delete artefacts from the old merged-CMIP6 pipeline.

    Only removes files that are known to come from the merged approach;
    NOAA outputs and per-model files are left untouched.
    """
    old_patterns = [
        PROCESSED_DIR / "cmip6_historical_processed.nc",
        PROCESSED_DIR / "cmip6_historical_regridded.nc",
        PROCESSED_DIR / "cmip6_future_processed.nc",
        PROCESSED_DIR / "cmip6_future_regridded.nc",
    ]
    for p in old_patterns:
        if p.exists():
            p.unlink()
            logger.info("Removed old merged file: %s", p.name)

    old_fig_fig_dires = ["cmip6_historical_", "cmip6_future_"]
    if FIGURES_DIR.exists():
        for f in FIGURES_DIR.iterdir():
            if f.is_file() and any(f.name.startswith(pre) for pre in old_fig_fig_dires):
                f.unlink()
                logger.info("Removed old merged figure: %s", f.name)


# ===================================================================
#  Variable & coordinate standardisation
# ===================================================================

def standardise_sst_variable(ds: xr.Dataset) -> xr.Dataset:
    found = [v for v in ds.data_vars if v in SST_VARIABLE_NAMES]
    if not found:
        raise ValueError(
            f"None of {SST_VARIABLE_NAMES} found among {list(ds.data_vars)}"
        )
    sst_var = found[0]
    return ds.rename({sst_var: "sst"}) if sst_var != "sst" else ds


def standardise_coordinates(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for existing in list(ds.coords):
        canonical = COORD_RENAME_MAP.get(existing)
        if canonical is not None and existing != canonical:
            rename[existing] = canonical
    if rename:
        ds = ds.rename(rename)

    non_essential = [
        v for v in ds.coords if v not in ("lat", "lon", "time") and v not in ds.dims
    ]
    if non_essential:
        ds = ds.drop_vars(non_essential, errors="ignore")

    return ds


def _ensure_lat_ascending(ds: xr.Dataset) -> xr.Dataset:
    if "lat" not in ds.coords or ds.lat.ndim != 1:
        return ds
    if float(ds.lat.values[0]) > float(ds.lat.values[-1]):
        logger.info("  Latitude is descending — reversing")
        ds = ds.sortby("lat")
    return ds


# ===================================================================
#  Unit conversion  (Kelvin → Celsius)
# ===================================================================

def convert_kelvin_to_celsius(ds: xr.Dataset) -> xr.Dataset:
    if "sst" not in ds.data_vars:
        return ds
    units = str(ds.sst.attrs.get("units", "")).lower().replace(" ", "").replace(".", "")
    if "kelvin" in units or units in ("k", "degk"):
        logger.info("  Converting SST Kelvin → Celsius")
        ds = ds.assign(sst=ds.sst - 273.15)
        ds.sst.attrs["units"] = "degC"
    return ds


# ===================================================================
#  Longitude conversion  (0–360 → –180…180)
# ===================================================================

def convert_longitude_1d(ds: xr.Dataset) -> xr.Dataset:
    lon = ds.lon.values
    new_lon = ((lon + 180) % 360) - 180
    ds = ds.assign_coords(lon=new_lon)
    ds = ds.sortby("lon")
    return ds


def convert_longitude_2d(ds: xr.Dataset) -> xr.Dataset:
    lon = ds.lon.values
    new_lon = ((lon + 180) % 360) - 180
    ds = ds.assign_coords(lon=xr.DataArray(new_lon, dims=ds.lon.dims))
    return ds


def convert_longitude(ds: xr.Dataset) -> xr.Dataset:
    if "lon" not in ds.coords:
        return ds
    if float(ds.lon.max()) <= 180.0:
        return ds
    logger.info("  Converting longitude 0–360 → –180…180")
    if ds.lon.ndim == 1:
        return convert_longitude_1d(ds)
    return convert_longitude_2d(ds)


# ===================================================================
#  Spatial subset  (Indian Ocean)
# ===================================================================

def subset_indian_ocean(ds: xr.Dataset) -> xr.Dataset:
    if "lat" not in ds.coords or "lon" not in ds.coords:
        return ds

    lat_1d = ds.lat.ndim == 1
    lon_1d = ds.lon.ndim == 1

    if lat_1d and lon_1d:
        ds = _ensure_lat_ascending(ds)
        return ds.sel(**INDIAN_OCEAN)

    mask = (
        (ds.lat >= INDIAN_OCEAN["lat"].start)
        & (ds.lat <= INDIAN_OCEAN["lat"].stop)
        & (ds.lon >= INDIAN_OCEAN["lon"].start)
        & (ds.lon <= INDIAN_OCEAN["lon"].stop)
    )
    if hasattr(mask, "compute"):
        mask = mask.compute()
    return ds.where(mask, drop=True)


# ===================================================================
#  Quality control
# ===================================================================

def remove_impossible_sst(ds: xr.Dataset) -> xr.Dataset:
    if "sst" in ds.data_vars:
        ds["sst"] = ds.sst.where(
            (ds.sst >= SST_RANGE[0]) & (ds.sst <= SST_RANGE[1]), other=np.nan
        )
    return ds


# ===================================================================
#  Time standardisation
# ===================================================================

def _try_import_cftime():
    try:
        import cftime
        return cftime
    except ImportError:
        return None


def _cftime_to_timestamp(t) -> pd.Timestamp:
    cftime = _try_import_cftime()
    if cftime is not None and isinstance(t, cftime.datetime):
        return pd.Timestamp(t.year, t.month, t.day, t.hour, t.minute, t.second)
    return pd.Timestamp(t)


def _time_needs_standardisation(time_values) -> bool:
    if len(time_values) == 0:
        return False
    try:
        pd.Timestamp(time_values[0])
        return False
    except (ValueError, TypeError):
        return True


def standardise_time_calendar(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords:
        return ds
    if not _time_needs_standardisation(ds.time.values):
        return ds
    logger.info("  Converting non-standard calendar → pandas Timestamp")
    times = np.array([_cftime_to_timestamp(t) for t in ds.time.values])
    ds = ds.assign_coords(time=times)
    return ds


def sortby_time_safe(ds: xr.Dataset) -> xr.Dataset:
    try:
        return ds.sortby("time")
    except TypeError:
        ds = standardise_time_calendar(ds)
        return ds.sortby("time")


# ===================================================================
#  Grid regridding  (CMIP6 → common 0.5° grid)
# ===================================================================

def _make_common_grid(resolution: float) -> tuple[np.ndarray, np.ndarray]:
    lat = np.arange(
        INDIAN_OCEAN["lat"].start,
        INDIAN_OCEAN["lat"].stop + resolution,
        resolution,
    )
    lon = np.arange(
        INDIAN_OCEAN["lon"].start,
        INDIAN_OCEAN["lon"].stop + resolution,
        resolution,
    )
    return lat, lon


def _regrid_1d(ds: xr.Dataset, target_lat: np.ndarray, target_lon: np.ndarray) -> xr.Dataset:
    return ds.interp(lat=target_lat, lon=target_lon)


def _regrid_2d_to_1d(
    ds: xr.Dataset, target_lat: np.ndarray, target_lon: np.ndarray
) -> xr.Dataset:
    from scipy.interpolate import griddata

    lat2d = ds.lat.values
    lon2d = ds.lon.values
    src_pts = np.column_stack([lat2d.ravel(), lon2d.ravel()])

    target_lon2d, target_lat2d = np.meshgrid(target_lon, target_lat)
    target_pts = np.column_stack([target_lat2d.ravel(), target_lon2d.ravel()])

    sst = ds.sst
    spatial_dims = [d for d in sst.dims if d != "time"]
    stacked = sst.stack(grid=spatial_dims)

    n_time = stacked.sizes["time"]
    n_target = len(target_pts)
    chunk_size = min(50, n_time)

    results = []
    for start in range(0, n_time, chunk_size):
        end = min(start + chunk_size, n_time)
        chunk = []
        for t in range(start, end):
            vals = griddata(src_pts, stacked.isel(time=t).values, target_pts, method="linear")
            chunk.append(vals)
        results.extend(chunk)
        logger.info("    Regridded time steps %d–%d / %d", start + 1, end, n_time)

    interp_data = np.array(results).reshape(n_time, len(target_lat), len(target_lon))
    return xr.Dataset({
        "sst": xr.DataArray(
            interp_data,
            dims=["time", "lat", "lon"],
            coords={"time": ds.time.values, "lat": target_lat, "lon": target_lon},
        )
    })


def regrid_to_common(
    ds: xr.Dataset, target_lat: np.ndarray, target_lon: np.ndarray
) -> xr.Dataset:
    if "lat" not in ds.coords or "lon" not in ds.coords:
        return ds
    if ds.lat.ndim == 1 and ds.lon.ndim == 1:
        return _regrid_1d(ds, target_lat, target_lon)
    return _regrid_2d_to_1d(ds, target_lat, target_lon)


# ===================================================================
#  Per-file processing
# ===================================================================

def process_one_file(filepath: Path, standardise: bool = True) -> xr.Dataset:
    ds = xr.open_dataset(filepath, chunks={})
    if standardise:
        ds = standardise_sst_variable(ds)
        ds = standardise_coordinates(ds)
    ds = convert_kelvin_to_celsius(ds)
    ds = convert_longitude(ds)
    ds = subset_indian_ocean(ds)
    ds = remove_impossible_sst(ds)
    return ds


# ===================================================================
#  NOAA pipeline
# ===================================================================

def process_noaa() -> xr.Dataset:
    t0 = time.perf_counter()
    files = find_netcdf_files(NOAA_DIR)
    logger.info("Processing %d NOAA years …", len(files))

    parts: list[xr.Dataset] = []
    for fpath in files:
        ds = process_one_file(fpath, standardise=False)
        parts.append(ds)

    logger.info("  Concatenating …")
    ds = xr.concat(parts, dim="time", coords="minimal", compat="override")
    ds = sortby_time_safe(ds)
    ds = ds.chunk({"time": 100, "lat": -1, "lon": -1})
    _log_done("Load NOAA dataset", time.perf_counter() - t0)
    _print_info(ds, "NOAA OISST")
    return ds


# ===================================================================
#  CMIP6 single-model pipeline
# ===================================================================

def process_one_cmip6_model(model_name: str, filepath: Path) -> xr.Dataset:
    """Process one CMIP6 model file end‑to‑end (no merging)."""
    t0 = time.perf_counter()
    logger.info("")
    logger.info("--------------------------------------------------")
    logger.info("  Processing %s", model_name)
    logger.info("--------------------------------------------------")

    target_lat, target_lon = _make_common_grid(CMIP6_TARGET_RESOLUTION)

    fname = filepath.name[:40]
    t1 = time.perf_counter()
    ds = process_one_file(filepath, standardise=True)
    logger.info("  %s — subset done (%.1f s)", fname, time.perf_counter() - t1)

    t2 = time.perf_counter()
    ds = regrid_to_common(ds, target_lat, target_lon)
    ds = standardise_time_calendar(ds)
    logger.info("  %s — regrid done (%.1f s)", fname, time.perf_counter() - t2)

    extras = [v for v in ds.data_vars if v not in ("sst", "lat", "lon")]
    if extras:
        ds = ds.drop_vars(extras, errors="ignore")
    ds = ds.chunk({"time": 100, "lat": -1, "lon": -1})

    _log_done(f"Process CMIP6 model  {model_name}", time.perf_counter() - t0)
    return ds


# ===================================================================
#  Print info
# ===================================================================

def _print_info(ds: xr.Dataset, label: str) -> None:
    logger.info("--- %s metadata ---", label)
    logger.info("  Dimensions   : %s", dict(ds.sizes))
    logger.info("  Coordinates  : %s", list(ds.coords))
    logger.info("  Data vars    : %s", list(ds.data_vars))
    if "time" in ds.dims and ds.sizes["time"] > 0:
        try:
            tv = ds.time.values
            logger.info("  Time range   : %s  ->  %s", str(tv[0])[:19], str(tv[-1])[:19])
            logger.info("  Time steps   : %d", ds.sizes["time"])
        except Exception:
            logger.info("  Time steps   : %d", ds.sizes["time"])


# ===================================================================
#  Time-axis validation
# ===================================================================

def validate_time_axis(ds: xr.Dataset) -> xr.Dataset:
    """Remove duplicate timestamps; report gaps."""
    t0 = time.perf_counter()
    if "time" not in ds.dims:
        logger.warning("No time dimension — skipping time validation")
        return ds

    ds = sortby_time_safe(ds)

    tv = ds.time.values
    _, index = np.unique(tv, return_index=True)
    if len(index) < ds.sizes["time"]:
        n_dup = ds.sizes["time"] - len(index)
        logger.warning("Removed %d duplicate time steps", n_dup)
        ds = ds.isel(time=np.sort(index))
        tv = ds.time.values

    logger.info("  Start date        : %s", str(tv[0])[:19])
    logger.info("  End date          : %s", str(tv[-1])[:19])
    logger.info("  Number of steps   : %d", len(tv))

    if len(tv) > 1:
        try:
            deltas = np.diff(tv.astype("datetime64[D]")).astype(int)
            missing = int(deltas[deltas > 1].sum())
            if missing:
                logger.warning("  Approx. missing days : %d", missing)
        except Exception:
            logger.warning("  Could not compute missing days (mixed calendars)")
    _log_done("Validate time axis", time.perf_counter() - t0)
    return ds


# ===================================================================
#  Dataset summary
# ===================================================================

def dataset_summary(ds: xr.Dataset, label: str) -> dict:
    logger.info("")
    logger.info("-" * 50)
    logger.info("  DATASET SUMMARY  ---  %s", label)
    logger.info("-" * 50)

    sst = ds.sst
    shape = dict(ds.sizes)
    total = int(sst.size)
    approx_gb = total * 4.0 / (1024 ** 3)

    res_lat = None
    res_lon = None
    if "lat" in ds.coords and ds.lat.ndim == 1 and len(ds.lat) > 1:
        res_lat = float(abs(np.diff(ds.lat.values[:2])[0]))
    if "lon" in ds.coords and ds.lon.ndim == 1 and len(ds.lon) > 1:
        res_lon = float(abs(np.diff(ds.lon.values[:2])[0]))

    logger.info("  Shape       : %s", shape)
    logger.info("  Approx      : %.1f GB (float32)", approx_gb)
    logger.info("  (computing global statistics …)")

    min_sst, max_sst, mean_sst, std_sst, missing = map(
        float,
        dask.compute(
            sst.min(), sst.max(), sst.mean(), sst.std(), sst.isnull().sum()
        ),
    )
    missing = int(missing)
    missing_pct = missing / total * 100.0 if total else 0.0

    logger.info("  Resolution  : lat=%.3f%s, lon=%.3f%s",
                 res_lat or 0, chr(176), res_lon or 0, chr(176))
    logger.info("  Min SST     : %.3f %sC", min_sst, chr(176))
    logger.info("  Max SST     : %.3f %sC", max_sst, chr(176))
    logger.info("  Mean SST    : %.3f %sC", mean_sst, chr(176))
    logger.info("  Std SST     : %.3f %sC", std_sst, chr(176))
    logger.info("  Missing     : %.2f %%", missing_pct)
    logger.info("-" * 50)

    return {
        "label": label,
        "shape": shape,
        "resolution_lat": res_lat,
        "resolution_lon": res_lon,
        "min_sst": min_sst,
        "max_sst": max_sst,
        "mean_sst": mean_sst,
        "std_sst": std_sst,
        "missing_pct": missing_pct,
    }


# ===================================================================
#  Save
# ===================================================================

def save_dataset(ds: xr.Dataset, filename: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / filename

    encoding = {v: {"zlib": True, "complevel": 1} for v in ds.data_vars}
    if "time" in ds.dims:
        ds = ds.chunk({"time": 100})

    fd, tmp_path = tempfile.mkstemp(suffix=".nc", dir=str(PROCESSED_DIR))
    os.close(fd)
    try:
        ds.to_netcdf(tmp_path, encoding=encoding)
        os.replace(tmp_path, str(path))
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("Saved  %s  (%.0f MB)", path, size_mb)
    return path


# ===================================================================
#  Figures  (per model, placed in a subdirectory)
# ===================================================================

def _maybe_import_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _get_figure_dir(label: str) -> Path:
    """Return the per‑model figure directory, creating it if needed."""
    fig_dir = FIGURES_DIR / label
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def generate_figures(ds: xr.Dataset, label: str) -> None:
    plt = _maybe_import_plt()
    if plt is None:
        logger.warning("matplotlib not available — skipping figures")
        return

    fig_dir = _get_figure_dir(label)
    sst = ds.sst

    spatial_dims = [d for d in sst.dims if d != "time"]

    logger.info("  Computing figure data …")
    sst_mean, ts, sample = dask.compute(
        sst.mean("time"),
        sst.mean(dim=spatial_dims),
        sst.isel(time=slice(None, None, 100)),
    )

    # (1) Mean SST map
    logger.info("  Plotting mean SST map …")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.pcolormesh(ds.lon, ds.lat, sst_mean, cmap="RdBu_r", shading="auto")
    plt.colorbar(ax.collections[0], ax=ax, label=f"SST ({chr(176)}C)")
    ax.set_title(f"{label} — Mean SST")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.savefig(fig_dir / "mean_sst.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (2) Histogram
    logger.info("  Plotting SST histogram …")
    vals = sample.values.ravel()
    vals = vals[~np.isnan(vals)]
    if len(vals) > 1_000_000:
        vals = np.random.default_rng(42).choice(vals, 1_000_000, replace=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_xlabel(f"SST ({chr(176)}C)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{label} — SST Distribution")
    fig.savefig(fig_dir / "histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (3) Basin-averaged time series
    logger.info("  Plotting basin-averaged time series …")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(ts.time, ts, linewidth=0.5, color="k")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"SST ({chr(176)}C)")
    ax.set_title(f"{label} — Basin-averaged SST")
    fig.autofmt_xdate()
    fig.savefig(fig_dir / "timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (4) Monthly climatology
    logger.info("  Computing monthly climatology …")
    clim = sst.groupby("time.month").mean("time")
    clim_spatial_dims = [d for d in clim.dims if d != "month"]
    clim_mean = clim.mean(dim=clim_spatial_dims).compute()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(clim_mean.month, clim_mean, marker="o", linestyle="-", color="crimson")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"SST ({chr(176)}C)")
    ax.set_title(f"{label} — Monthly Climatology")
    ax.set_xticks(range(1, 13))
    fig.savefig(fig_dir / "monthly_cycle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("  Figures saved to %s", fig_dir)


# ===================================================================
#  Main
# ===================================================================

def main() -> None:
    overall_start = time.perf_counter()
    setup_logging()

    _clean_old_merged_outputs()

    logger.info("=" * 60)
    logger.info("  START preprocess.py")
    logger.info("=" * 60)

    # ---- NOAA ----
    noaa_processed = PROCESSED_DIR / "noaa_processed.nc"
    if noaa_processed.exists():
        logger.info("'noaa_processed.nc' already exists — skipping NOAA")
    else:
        logger.info("")
        logger.info("=" * 58)
        logger.info("  Preprocessing NOAA OISST")
        logger.info("=" * 58)
        t_src = time.perf_counter()
        try:
            ds_noaa = process_noaa()
            ds_noaa = validate_time_axis(ds_noaa)
            dataset_summary(ds_noaa, "NOAA OISST")
            save_dataset(ds_noaa, "noaa_processed.nc")
            generate_figures(ds_noaa, "noaa")
            logger.info("NOAA finished in %.1f s", time.perf_counter() - t_src)
        except Exception as e:
            logger.error("NOAA pipeline FAILED: %s", e)
            traceback.print_exc()

    # ---- CMIP6 models (each independently) ----
    if not CMIP6_DIR.exists():
        logger.warning("CMIP6 directory does not exist: %s", CMIP6_DIR)
    else:
        models = discover_cmip6_models(CMIP6_DIR)
        logger.info("")
        logger.info("=" * 58)
        logger.info("  Found %d CMIP6 model(s)", len(models))
        logger.info("=" * 58)

        for model_name, filepath in models:
            out_name = f"{model_name}_processed.nc"
            out_path = PROCESSED_DIR / out_name

            if out_path.exists():
                logger.info("'%s' already exists — skipping %s", out_name, model_name)
                continue

            t_model = time.perf_counter()
            try:
                ds = process_one_cmip6_model(model_name, filepath)
                ds = validate_time_axis(ds)
                dataset_summary(ds, model_name)
                save_dataset(ds, out_name)
                generate_figures(ds, model_name)
                logger.info("%s finished in %.1f s", model_name, time.perf_counter() - t_model)
                logger.info("  Finished %s", model_name)
            except Exception as e:
                logger.error("%s pipeline FAILED: %s", model_name, e)
                traceback.print_exc()

    logger.info("")
    logger.info("  All done  ---  total time: %.1f s", time.perf_counter() - overall_start)


if __name__ == "__main__":
    main()
