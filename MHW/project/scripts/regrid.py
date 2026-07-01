"""
regrid.py
=========
Regrid each CMIP6 model onto the NOAA OISST 0.25° grid using bilinear
interpolation (xarray.interp).  xESMF is not available in this environment.

Each model is regridded independently: one regridded file per model.

Outputs
-------
- data/processed/<model_name>_regridded.nc    (per model)
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

NOAA_PROCESSED = PROCESSED_DIR / "noaa_processed.nc"

logger = logging.getLogger("regrid")


# ===================================================================
#  Logging helpers
# ===================================================================

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-12s | %(levelname)-6s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ===================================================================
#  Target grid  (NOAA OISST)
# ===================================================================

def build_target_grid() -> xr.Dataset:
    """Read the NOAA processed file and return the target grid metadata."""
    if not NOAA_PROCESSED.exists():
        raise FileNotFoundError(
            f"NOAA processed file not found — run preprocess.py first\n  {NOAA_PROCESSED}"
        )
    logger.info("Reading target grid from %s", NOAA_PROCESSED.name)
    with xr.open_dataset(NOAA_PROCESSED, chunks={}) as ds:
        target_lat = ds.lat.values
        target_lon = ds.lon.values
    logger.info(
        "Target grid  lat: %.2f..%.2f (%d)  lon: %.2f..%.2f (%d)",
        target_lat[0], target_lat[-1], len(target_lat),
        target_lon[0], target_lon[-1], len(target_lon),
    )
    return xr.Dataset(coords={"lat": target_lat, "lon": target_lon})


# ===================================================================
#  Discover processed CMIP6 models
# ===================================================================

def discover_processed_models() -> list[tuple[str, Path]]:
    """Return list of (model_name, filepath) for every ``*_processed.nc``
    file in the processed directory, excluding NOAA."""
    models: list[tuple[str, Path]] = []
    for fpath in sorted(PROCESSED_DIR.glob("*_processed.nc")):
        if fpath.name == "noaa_processed.nc":
            continue
        name = fpath.stem.replace("_processed", "")
        models.append((name, fpath))
    return models


# ===================================================================
#  Regrid one model
# ===================================================================

def _regrid_model_to_target(
    model_name: str, src_path: Path, target: xr.Dataset
) -> None:
    """Regrid a single CMIP6 model to the NOAA target grid and save."""
    logger.info("")
    logger.info("--------------------------------------------------")
    logger.info("  Regridding %s", model_name)
    logger.info("--------------------------------------------------")

    t0 = time.perf_counter()
    logger.info("  Loading %s …", src_path.name)
    ds = xr.open_dataset(src_path, chunks={})
    logger.info("  Input  %s", dict(ds.sizes))

    logger.info("  Interpolating to target grid …")
    ds_regridded = ds.interp(
        lat=target.lat,
        lon=target.lon,
        method="linear",
        kwargs={"fill_value": np.nan},
    )
    ds.close()

    elapsed = time.perf_counter() - t0
    logger.info("  Output %s", dict(ds_regridded.sizes))
    logger.info("  Regridding  …  done  (%.1f s)", elapsed)

    out_name = f"{model_name}_regridded.nc"
    out_path = PROCESSED_DIR / out_name

    encoding = {v: {"zlib": True, "complevel": 1} for v in ds_regridded.data_vars}
    if "time" in ds_regridded.dims:
        ds_regridded = ds_regridded.chunk({"time": 100})

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".nc", dir=str(PROCESSED_DIR))
    os.close(fd)
    try:
        ds_regridded.to_netcdf(tmp_path, encoding=encoding)
        os.replace(tmp_path, out_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("Saved  %s  (%.0f MB)", out_path, size_mb)
    logger.info("  Finished %s", model_name)


# ===================================================================
#  Main
# ===================================================================

def main() -> None:
    overall_start = time.perf_counter()
    setup_logging()

    logger.info("=" * 60)
    logger.info("  START regrid.py")
    logger.info("=" * 60)

    target = build_target_grid()

    models = discover_processed_models()
    if not models:
        logger.warning("No processed CMIP6 models found in %s", PROCESSED_DIR)
        logger.warning("Run preprocess.py first")
    else:
        logger.info("Found %d CMIP6 model(s) to regrid", len(models))
        for model_name, src_path in models:
            out_path = PROCESSED_DIR / f"{model_name}_regridded.nc"
            if out_path.exists():
                logger.info("'%s_regridded.nc' already exists — skipping", model_name)
                continue
            try:
                _regrid_model_to_target(model_name, src_path, target)
            except Exception as e:
                logger.error("Regridding %s FAILED: %s", model_name, e)
                traceback.print_exc()

    logger.info("")
    logger.info("  All done  —  total time: %.1f s", time.perf_counter() - overall_start)


if __name__ == "__main__":
    main()
