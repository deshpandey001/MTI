"""
preprocess.py
=============
Research-grade preprocessing pipeline for NOAA OISST v2 and CMIP6
historical SST datasets.  This script:

  • Loads all NetCDF files from ``data/noaa/`` and ``data/cmip6_historical/``
  • Standardises variable names, coordinates, and units
  • Converts longitude from 0–360 to –180…180
  • Subsets the Indian Ocean (20°E–120°E, 40°S–30°N)
  • Filters physically impossible SST values (–2…40 °C)
  • Validates time axis (missing / duplicate dates)
  • Prints a detailed summary report
  • Saves harmonised files to ``data/processed/``
  • Generates quick-look figures in ``outputs/preprocessing/``

Usage
-----
    python scripts/preprocess.py

All configuration is inferred from the project directory structure
so no command-line arguments are required.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import xarray as xr
from dask import compute as dask_compute

# ---------------------------------------------------------------------------
# Paths  (relative to this script: scripts/ → project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
NOAA_DIR = DATA_DIR / "noaa"
CMIP6_DIR = DATA_DIR / "cmip6_historical"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "preprocessing"

# Coordinate-name mappings for CMIP6 models
_COORD_MAP: dict[str, str] = {
    "latitude": "lat",
    "lat": "lat",
    "nav_lat": "lat",
    "longitude": "lon",
    "lon": "lon",
    "nav_lon": "lon",
    "time": "time",
}

logger = logging.getLogger("preprocess")


# ===================================================================
#  Logging helpers
# ===================================================================

def setup_logging() -> None:
    """Configure a single root-logger for the whole module."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-12s | %(levelname)-6s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_step(step_name: str) -> None:
    """Print a banner marking the start of a processing step."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("START  %s", step_name)
    logger.info("=" * 60)


def log_done(step_name: str, elapsed: float) -> None:
    """Mark the end of a step together with wall-clock time."""
    logger.info("%s  …  done  (%.2f s)", step_name, elapsed)


# ===================================================================
#  File discovery
# ===================================================================

def find_netcdf_files(directory: Path) -> list[Path]:
    """Return a sorted list of all ``.nc`` files under *directory*.

    Raises
    ------
    FileNotFoundError
        If no NetCDF files are found.
    """
    files = sorted(directory.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {directory}")
    logger.info("Found %d NetCDF file(s) in %s", len(files), directory)
    return files


# ===================================================================
#  Dataset loading
# ===================================================================

def load_noaa_dataset() -> xr.Dataset:
    """Load all NOAA OISST NetCDF files with dask chunks.

    Each file is opened with time chunks, longitude is converted,
    and the Indian Ocean subset is applied lazily so that the global
    grid is never materialised in memory.
    """
    t0 = time.perf_counter()
    files = find_netcdf_files(NOAA_DIR)
    logger.info("Opening NOAA files with dask …")

    # Open as a single dask-backed dataset
    ds = xr.open_mfdataset(
        [str(f) for f in files],
        combine="by_coords",
        chunks={"time": 50},
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )

    # Apply longitude conversion (coordinate-only, cheap)
    if "lon" in ds.coords and _is_0_360(ds):
        lon = ds.lon.values  # 1-D coordinate, small
        ds = ds.assign_coords(lon=((lon + 180) % 360) - 180)
        ds = ds.sortby("lon")

    # Subset Indian Ocean
    ds = ds.sel(lat=slice(-40.0, 30.0), lon=slice(20.0, 120.0))

    log_done("Load NOAA dataset", time.perf_counter() - t0)
    _print_dataset_info(ds, "NOAA OISST")
    return ds


def load_cmip6_dataset() -> xr.Dataset:
    """Load all CMIP6 NetCDF files, handling mismatched grids gracefully.

    Files that share the same spatial grid are combined via
    ``open_mfdataset``; otherwise each file is opened individually and
    re-chunked before being concatenated along ``time``.
    """
    t0 = time.perf_counter()
    files = find_netcdf_files(CMIP6_DIR)

    # Try a single open_mfdataset first (fast path for homogeneous grids)
    try:
        ds = xr.open_mfdataset(
            [str(f) for f in files],
            combine="by_coords",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            chunks={"time": 50},
        )
        ds = ds.sortby("time")
        logger.info("All CMIP6 files share the same grid — combined directly.")
        log_done("Load CMIP6 dataset", time.perf_counter() - t0)
        _print_dataset_info(ds, "CMIP6 historical")
        return ds
    except (ValueError, KeyError):
        logger.info(
            "CMIP6 grids differ — loading files individually …"
        )

    # Fallback: open each file and concatenate along time
    datasets: list[xr.Dataset] = []
    for f in files:
        ds_i = xr.open_dataset(str(f), chunks={"time": 50})
        # Standardise internal variable name ASAP
        ds_i = _standardise_variable(ds_i)
        datasets.append(ds_i)

    # Combine along time — xr.concat will broadcast mismatched
    # spatial coordinates.
    ds = xr.concat(datasets, dim="time", coords="minimal", compat="override")
    ds = ds.sortby("time")
    log_done("Load CMIP6 dataset (fallback)", time.perf_counter() - t0)
    _print_dataset_info(ds, "CMIP6 historical")
    return ds


def _print_dataset_info(ds: xr.Dataset, label: str) -> None:
    """Log the key metadata of a dataset for inspection."""
    logger.info("─── %s metadata ───", label)
    logger.info("  Dimensions   : %s", dict(ds.sizes))
    logger.info("  Coordinates  : %s", list(ds.coords))
    logger.info("  Data vars    : %s", list(ds.data_vars))
    if "time" in ds.dims:
        logger.info(
            "  Time range   : %s  →  %s",
            str(ds.time.values.min())[:19],
            str(ds.time.values.max())[:19],
        )
        logger.info("  Time steps   : %d", ds.sizes["time"])
    logger.info("  Global attrs : %s", _summarise_attrs(ds.attrs))


def _summarise_attrs(attrs: dict) -> str:
    """Short human-readable summary of global attributes."""
    parts: list[str] = []
    for key in ("title", "source", "institution", "Conventions"):
        val = attrs.get(key)
        if val:
            parts.append(f"{key}={val}")
    return "; ".join(parts) if parts else "(none)"


# ===================================================================
#  Variable & coordinate standardisation
# ===================================================================

def _standardise_variable(ds: xr.Dataset) -> xr.Dataset:
    """Detect and rename the SST variable to ``sst``.

    Recognised input names: ``sst``, ``tos``, ``thetao``.
    """
    sst_names = {"sst", "tos", "thetao"}
    found = [v for v in ds.data_vars if v in sst_names]
    if not found:
        raise ValueError(
            f"None of {sst_names} found among data variables {list(ds.data_vars)}"
        )
    if len(found) > 1:
        logger.warning("Multiple SST-like variables found: %s — using '%s'", found, found[0])
    sst_var = found[0]
    if sst_var != "sst":
        ds = ds.rename({sst_var: "sst"})
    return ds


def standardise_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """Rename all recognised coordinate names to ``lat`` / ``lon`` / ``time``.

    Also drops auxiliary coords that are not needed downstream.
    """
    rename = {}
    for existing_name in list(ds.coords):
        canonical = _COORD_MAP.get(existing_name)
        if canonical is not None and existing_name != canonical:
            rename[existing_name] = canonical
    if rename:
        logger.info("Renaming coordinates: %s", rename)
        ds = ds.rename(rename)

    # Remove spurious 1D bounds / vertex arrays that can confuse later ops
    drop = [v for v in ds.coords if v not in ("lat", "lon", "time") and v not in ds.dims]
    if drop:
        logger.info("Dropping auxiliary coords: %s", drop)
        ds = ds.drop_vars(drop, errors="ignore")
    return ds


def _guess_lat_lon_dims(ds: xr.Dataset) -> tuple[list[str], list[str]]:
    """Identify the spatial dimension names for lat / lon.

    Returns
    -------
    lat_dims, lon_dims : list of str
    """
    lat_candidates = {"lat", "latitude", "nav_lat", "j", "y"}
    lon_candidates = {"lon", "longitude", "nav_lon", "i", "x"}
    lat_dims = [d for d in ds.dims if d in lat_candidates]
    lon_dims = [d for d in ds.dims if d in lon_candidates]
    return lat_dims, lon_dims


# ===================================================================
#  Unit conversion
# ===================================================================

def convert_kelvin_to_celsius(ds: xr.Dataset) -> xr.Dataset:
    """Convert ``sst`` from Kelvin to Celsius if ``units`` attribute suggests so.

    The check is case-insensitive and recognises: ``K``, ``Kelvin``,
    ``degK``, ``degrees_K``, *etc.*
    """
    if "sst" not in ds.data_vars:
        return ds
    attrs = ds.sst.attrs
    units = str(attrs.get("units", "")).lower().replace(" ", "").replace(".", "")

    kelvin_indicators = {"k", "kelvin", "degk", "degrees_k"}
    if units in kelvin_indicators or any(ind in units for ind in ("kelvin", "degk")):
        logger.info("Converting SST from Kelvin to Celsius (K – 273.15)")
        ds = ds.assign(sst=ds.sst - 273.15)
        ds.sst.attrs["units"] = "degC"
    else:
        logger.info("SST units are '%s' — no conversion needed", attrs.get("units", "unknown"))
    return ds


# ===================================================================
#  Longitude conversion  (0…360 → –180…180)
# ===================================================================

def _is_0_360(ds: xr.Dataset) -> bool:
    """Return ``True`` if the longitude coordinate spans 0…360."""
    if "lon" not in ds.coords:
        return False
    lon = ds.lon.values
    return bool(np.any(lon > 180))


def convert_longitude(ds: xr.Dataset) -> xr.Dataset:
    """Convert longitude from 0…360 → –180…180 and sort monotonically.

    Works for both 1D ``lon`` coordinates and 2D ``lon`` fields (e.g.
    ORCA curvilinear grids).
    """
    if not _is_0_360(ds):
        logger.info("Longitude already in –180…180 range — skipping conversion")
        return ds

    logger.info("Converting longitude from 0…360  →  –180…180")

    if "lon" not in ds.coords:
        logger.warning("'lon' coordinate not found — cannot convert")
        return ds

    lon = ds.lon.values

    if lon.ndim == 1:
        # 1-D regular grid — simple transformation
        ds = ds.assign_coords(lon=((lon + 180) % 360) - 180)
        ds = ds.sortby("lon")
    elif lon.ndim == 2:
        # 2-D curvilinear grid — shift each point individually
        ds = ds.assign_coords(lon=((lon + 180) % 360) - 180)
        # Sorting 2D lon is not trivial; skip sorting for curvilinear grids.
        logger.info("2-D longitude field detected — spatial sorting skipped")
    else:
        logger.warning("Unexpected lon dimensionality (%d) — skipping", lon.ndim)

    return ds


# ===================================================================
#  Spatial subset  — Indian Ocean
# ===================================================================

def subset_indian_ocean(ds: xr.Dataset) -> xr.Dataset:
    """Subset the dataset to the Indian Ocean domain.

    Domain: 20°E – 120°E,  40°S – 30°N.

    Works with both 1-D (label-based) and 2-D (boolean mask) spatial
    coordinates.
    """
    lon_min, lon_max = 20.0, 120.0
    lat_min, lat_max = -40.0, 30.0

    logger.info(
        "Subsetting Indian Ocean: lon=[%.0f, %.0f], lat=[%.0f, %.0f]",
        lon_min, lon_max, lat_min, lat_max,
    )

    lat_dims, lon_dims = _guess_lat_lon_dims(ds)

    # Determine if we have 1-D or 2-D coordinates
    lat_1d = "lat" in ds.coords and ds.lat.ndim == 1
    lon_1d = "lon" in ds.coords and ds.lon.ndim == 1

    if lat_1d and lon_1d:
        # Simple label-based selection (regular grid)
        ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    else:
        # 2-D curvilinear grid — build a boolean mask
        if "lat" in ds.coords and "lon" in ds.coords:
            mask = (
                (ds.lat >= lat_min) & (ds.lat <= lat_max) &
                (ds.lon >= lon_min) & (ds.lon <= lon_max)
            )
            ds = ds.where(mask, drop=True)
        else:
            logger.warning("Cannot subset — lat/lon coordinates not standardised")
    return ds


# ===================================================================
#  Quality control — SST range filtering
# ===================================================================

def remove_impossible_sst(ds: xr.Dataset) -> xr.Dataset:
    """Mask (set to NaN) SST values outside the physically plausible range.

    Thresholds:  –2 °C  ≤  SST  ≤  40 °C

    The operation is lazy (dask-backed).  The full count of removed
    values is *not* computed here to avoid loading the whole dataset
    into memory; only a log message is emitted.
    """
    t0 = time.perf_counter()
    if "sst" not in ds.data_vars:
        return ds

    ds = ds.where((ds.sst >= -2.0) & (ds.sst <= 40.0), other=np.nan)

    logger.info("SST range filter [–2, 40] °C applied (lazy)")
    log_done("Remove impossible SST", time.perf_counter() - t0)
    return ds


# ===================================================================
#  Time-axis validation
# ===================================================================

def validate_time_axis(ds: xr.Dataset) -> xr.Dataset:
    """Validate and standardise the time coordinate.

    Operations
    ----------
    1. Convert to ``datetime64`` if needed.
    2. Sort chronologically.
    3. Drop duplicate time steps.
    4. Report missing dates (daily data only).
    """
    t0 = time.perf_counter()
    if "time" not in ds.dims:
        logger.warning("No time dimension found — skipping time validation")
        return ds

    # Convert to datetime
    ds["time"] = xr.decode_cf(ds).time

    # Sort
    ds = ds.sortby("time")

    # Drop duplicates
    _, index = np.unique(ds.time.values, return_index=True)
    if len(index) < ds.sizes["time"]:
        n_dup = ds.sizes["time"] - len(index)
        logger.warning("Found %d duplicate time steps — removing", n_dup)
        ds = ds.isel(time=np.sort(index))

    # Report
    time_vals = ds.time.values
    logger.info("Start date        : %s", str(time_vals[0])[:19])
    logger.info("End date          : %s", str(time_vals[-1])[:19])
    logger.info("Number of steps   : %d", len(time_vals))

    # Check for missing dates (daily frequency)
    if len(time_vals) > 1:
        deltas = np.diff(time_vals).astype("timedelta64[D]").astype(int)
        expected_daily = np.timedelta64(1, "D").astype("timedelta64[D]").astype(int)
        missing = np.sum(deltas[deltas > expected_daily])
        if missing > 0:
            logger.warning("Approximate number of missing days: %d", int(missing))

    log_done("Validate time axis", time.perf_counter() - t0)
    return ds


# ===================================================================
#  Dataset summary report
# ===================================================================

def dataset_summary(ds: xr.Dataset, label: str) -> dict:
    """Print a detailed statistical summary of *ds* and return as a dict.

    All statistics are computed via dask reductions (chunk-by-chunk) so
    that the full dataset is never loaded into memory.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing ``sst`` variable.
    label : str
        Human-readable name for the dataset (printed in the report).

    Returns
    -------
    dict
        Summary statistics.
    """
    logger.info("")
    logger.info("─" * 50)
    logger.info("  DATASET SUMMARY  —  %s", label)
    logger.info("─" * 50)

    sst = ds.sst
    summary: dict = {
        "label": label,
        "shape": dict(ds.sizes),
        "resolution_lat": None,
        "resolution_lon": None,
        "min_sst": None,
        "max_sst": None,
        "mean_sst": None,
        "std_sst": None,
        "missing_pct": None,
    }

    if "lat" in ds.coords and ds.lat.ndim == 1 and len(ds.lat) > 1:
        summary["resolution_lat"] = float(
            abs(np.diff(ds.lat.values[:2])[0])
        )
        if "lon" in ds.coords and ds.lon.ndim == 1 and len(ds.lon) > 1:
            summary["resolution_lon"] = float(
                abs(np.diff(ds.lon.values[:2])[0])
            )

    total = int(sst.size)

    min_val, max_val, mean_val, std_val, n_missing = dask_compute(
        sst.min(), sst.max(), sst.mean(), sst.std(), sst.isnull().sum(),
    )
    missing = int(n_missing)
    summary["missing_pct"] = (missing / total * 100.0) if total else 0.0

    if total - missing > 0:
        summary["min_sst"] = float(min_val)
        summary["max_sst"] = float(max_val)
        summary["mean_sst"] = float(mean_val)
        summary["std_sst"] = float(std_val)

    logger.info("  Shape            : %s", summary["shape"])
    logger.info(
        "  Resolution       : lat=%.3f\xb0, lon=%.3f\xb0",
        summary["resolution_lat"] or 0,
        summary["resolution_lon"] or 0,
    )
    logger.info("  Min SST          : %.3f \xb0C", summary["min_sst"] or 0)
    logger.info("  Max SST          : %.3f \xb0C", summary["max_sst"] or 0)
    logger.info("  Mean SST         : %.3f \xb0C", summary["mean_sst"] or 0)
    logger.info("  Std SST          : %.3f \xb0C", summary["std_sst"] or 0)
    logger.info("  Missing values   : %.2f %%", summary["missing_pct"])
    logger.info("─" * 50)
    return summary


# ===================================================================
#  Save processed output
# ===================================================================

def save_dataset(ds: xr.Dataset, filename: str) -> Path:
    """Write a dataset to ``data/processed/`` as NetCDF.

    Compression is applied (zlib, complevel=5) to save disk space.

    Parameters
    ----------
    ds : xr.Dataset
    filename : str
        Output filename (e.g. ``noaa_processed.nc``).

    Returns
    -------
    Path
        Full path to the saved file.
    """
    t0 = time.perf_counter()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / filename

    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {"zlib": True, "complevel": 5}

    ds.to_netcdf(path, encoding=encoding)
    file_mb = path.stat().st_size / (1024 * 1024)
    logger.info("Saved  %s  →  %.1f MB", path, file_mb)
    log_done("Save dataset", time.perf_counter() - t0)
    return path


# ===================================================================
#  Quick-look figures
# ===================================================================

def generate_figures(ds: xr.Dataset, label: str) -> None:
    """Create four quick-look figures for the harmonised dataset.

    1. Mean SST map
    2. SST histogram (random sample of 1 × 10⁶ points)
    3. Basin-averaged SST time series
    4. Monthly climatology

    All figures are saved to ``outputs/preprocessing/``.
    Uses dask-friendly reductions — never loads the full cube.
    """
    t0 = time.perf_counter()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping figures")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sst = ds.sst
    prefix = FIGURES_DIR / label

    # ----- (1) Mean SST map -----
    logger.info("  Computing mean SST map …")
    sst_mean = sst.mean("time").compute()
    fig, ax = plt.subplots(figsize=(8, 4))
    lon2d = ds.lon.values if ds.lon.ndim == 2 else ds.lon.values
    lat2d = ds.lat.values if ds.lat.ndim == 2 else ds.lat.values
    p = ax.pcolormesh(lon2d, lat2d, sst_mean.values, cmap="RdBu_r", shading="auto")
    plt.colorbar(p, ax=ax, label="SST (\xb0C)")
    ax.set_title(f"{label} — Mean SST")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.savefig(prefix + "_mean_sst.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s_mean_sst.png", prefix)

    # ----- (2) SST histogram (subsample) -----
    logger.info("  Computing SST histogram (subsample) …")
    sample = sst.chunk({"time": -1}).isel(
        time=slice(None, None, max(1, sst.sizes["time"] // 100))
    ).load()
    sst_vals = sample.values.ravel()
    sst_vals = sst_vals[~np.isnan(sst_vals)]
    if len(sst_vals) > 1_000_000:
        rng = np.random.default_rng(42)
        sst_vals = rng.choice(sst_vals, 1_000_000, replace=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sst_vals, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_xlabel("SST (\xb0C)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{label} — SST Distribution (sample)")
    fig.savefig(prefix + "_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s_histogram.png", prefix)

    # ----- (3) Basin-averaged SST time series -----
    logger.info("  Computing basin-averaged time series …")
    sst_ts = sst.mean(dim=[d for d in sst.dims if d != "time"]).compute()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(sst_ts.time.values, sst_ts.values, linewidth=0.5, color="k")
    ax.set_xlabel("Time")
    ax.set_ylabel("SST (\xb0C)")
    ax.set_title(f"{label} — Basin-averaged SST")
    fig.autofmt_xdate()
    fig.savefig(prefix + "_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s_timeseries.png", prefix)

    # ----- (4) Monthly climatology -----
    logger.info("  Computing monthly climatology …")
    clim = sst.groupby("time.month").mean("time")
    clim_ts = clim.mean(dim=[d for d in sst.dims if d != "month"]).compute()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(clim_ts.month.values, clim_ts.values, marker="o", linestyle="-",
            color="crimson")
    ax.set_xlabel("Month")
    ax.set_ylabel("SST (\xb0C)")
    ax.set_title(f"{label} — Monthly Climatology")
    ax.set_xticks(range(1, 13))
    fig.savefig(prefix + "_climatology.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s_climatology.png", prefix)

    log_done("Generate figures", time.perf_counter() - t0)


# ===================================================================
#  Main pipeline
# ===================================================================

def run_preprocessing_pipeline(source: str) -> xr.Dataset:
    """Run the full preprocessing pipeline for one data source.

    Parameters
    ----------
    source : str
        ``"noaa"`` or ``"cmip6"``.

    Returns
    -------
    xr.Dataset
        Fully preprocessed dataset ready for downstream analysis.
    """
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  Preprocessing pipeline  —  %-17s  ║", source.upper())
    logger.info("╚══════════════════════════════════════════════════════╝")

    # ---- STEP 1 & 2 : Load ----
    if source == "noaa":
        step_name = "Load NOAA dataset"
        log_step(step_name)
        ds = load_noaa_dataset()
        log_done(step_name, 0.0)
    else:
        step_name = "Load CMIP6 dataset"
        log_step(step_name)
        ds = load_cmip6_dataset()
        log_done(step_name, 0.0)

    # ---- STEP 2b : Standardise variable name ----
    step_name = "Standardise variable names"
    log_step(step_name)
    ds = _standardise_variable(ds)
    log_done(step_name, 0.0)

    # ---- STEP 3 : Convert Kelvin → Celsius ----
    step_name = "Convert Kelvin → Celsius"
    log_step(step_name)
    ds = convert_kelvin_to_celsius(ds)
    log_done(step_name, 0.0)

    # ---- STEP 4 : Rename coordinates ----
    step_name = "Rename coordinates"
    log_step(step_name)
    ds = standardise_coordinates(ds)
    log_done(step_name, 0.0)

    # ---- STEP 5 : Convert longitude ----
    step_name = "Convert longitude 0–360 → –180…180"
    log_step(step_name)
    ds = convert_longitude(ds)
    log_done(step_name, 0.0)

    # ---- STEP 6 : Subset Indian Ocean ----
    step_name = "Subset Indian Ocean domain"
    log_step(step_name)
    ds = subset_indian_ocean(ds)
    log_done(step_name, 0.0)

    # ---- STEP 7 : Remove impossible SST ----
    step_name = "Remove impossible SST values"
    log_step(step_name)
    ds = remove_impossible_sst(ds)
    log_done(step_name, 0.0)

    # ---- STEP 8 : Validate time axis ----
    step_name = "Validate time axis"
    log_step(step_name)
    ds = validate_time_axis(ds)
    log_done(step_name, 0.0)

    # ---- STEP 9 : Dataset summary ----
    step_name = "Dataset summary"
    log_step(step_name)
    dataset_summary(ds, source.upper())
    log_done(step_name, 0.0)

    return ds


def main() -> None:
    """Run the full preprocessing pipeline for both NOAA and CMIP6."""
    overall_start = time.perf_counter()
    setup_logging()
    logger.info("")
    logger.info("████████████████████████████████████████████████████")
    logger.info("██  Preprocessing Pipeline  —  NOAA + CMIP6     ██")
    logger.info("████████████████████████████████████████████████████")

    # ---- Process NOAA ----
    t_noaa = time.perf_counter()
    ds_noaa = run_preprocessing_pipeline("noaa")
    # ---- STEP 10 : Save ----
    logger.info("Saving NOAA processed dataset …")
    save_dataset(ds_noaa, "noaa_processed.nc")
    # ---- STEP 11 : Figures ----
    logger.info("Generating NOAA figures …")
    generate_figures(ds_noaa, "noaa")
    logger.info("NOAA pipeline finished in %.2f s", time.perf_counter() - t_noaa)

    # ---- Process CMIP6 ----
    t_cmip = time.perf_counter()
    ds_cmip6 = run_preprocessing_pipeline("cmip6")
    save_dataset(ds_cmip6, "cmip6_processed.nc")
    generate_figures(ds_cmip6, "cmip6")
    logger.info("CMIP6 pipeline finished in %.2f s", time.perf_counter() - t_cmip)

    total = time.perf_counter() - overall_start
    logger.info("")
    logger.info("████████████████████████████████████████████████████")
    logger.info("██  All done  —  total time: %6.2f s          ██", total)
    logger.info("████████████████████████████████████████████████████")


if __name__ == "__main__":
    main()
