"""
validation.py
=============
Validate each CMIP6 model independently against NOAA OISST.

Each model is loaded from its regridded file (already on the NOAA 0.25° grid),
matched to the common time period, and scored on bias, RMSE, correlation,
and standard-deviation ratio.  Output is per-model figures/metrics plus a
cross-model ranking table.

Outputs (in outputs/validation/<model_name>/)
----------------------------------------------
- mean_sst_map.png
- bias_map.png
- rmse_map.png
- monthly_climatology_comparison.png
- timeseries_comparison.png
- validation_metrics.csv
- validation_summary.txt

Plus a cross-model ranking:
- outputs/validation/model_ranking.csv
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import dask
import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "validation"

NOAA_PROCESSED = PROCESSED_DIR / "noaa_processed.nc"

# ---------------------------------------------------------------------------
# Region of interest — Indian Ocean
# ---------------------------------------------------------------------------
LAT_RANGE = (-40.0, 30.0)
LON_RANGE = (20.0, 120.0)

logger = logging.getLogger("validation")


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
#  File discovery
# ===================================================================

def discover_regridded_models() -> list[tuple[str, Path]]:
    """Return list of (model_name, filepath) for every ``*_regridded.nc``
    file in the processed directory."""
    models: list[tuple[str, Path]] = []
    for fpath in sorted(PROCESSED_DIR.glob("*_regridded.nc")):
        name = fpath.stem.replace("_regridded", "")
        models.append((name, fpath))
    return models


# ===================================================================
#  I/O helpers
# ===================================================================

def _standardise_ds(ds: xr.Dataset) -> xr.Dataset:
    """Rename coordinates and variables to a common convention."""
    rename_coords = {}
    for candidate in ("latitude", "nav_lat", "lat"):
        if candidate in ds.coords and "lat" not in ds.coords:
            rename_coords[candidate] = "lat"
    for candidate in ("longitude", "nav_lon", "lon"):
        if candidate in ds.coords and "lon" not in ds.coords:
            rename_coords[candidate] = "lon"
    if rename_coords:
        ds = ds.rename(rename_coords)

    rename_vars = {}
    if "tos" in ds.data_vars and "sst" not in ds.data_vars:
        rename_vars["tos"] = "sst"
    elif "sst" in ds.data_vars:
        pass
    elif "thetao" in ds.data_vars:
        rename_vars["thetao"] = "sst"
    if rename_vars:
        ds = ds.rename(rename_vars)

    return ds


def _ensure_lon_range(ds: xr.Dataset) -> xr.Dataset:
    if ds.lon.max() > 180.0:
        ds = ds.assign_coords(lon=(ds.lon % 360))
        ds = ds.sortby("lon")
    return ds


def load_noaa() -> xr.Dataset:
    """Load and standardise the NOAA OISST dataset."""
    logger.info("Loading NOAA OISST: %s", NOAA_PROCESSED.name)
    ds = xr.open_dataset(NOAA_PROCESSED, chunks={})
    if "sst" not in ds.data_vars:
        ds = _standardise_ds(ds)
    ds = _ensure_lon_range(ds)
    return subset_domain(ds)


def load_model(model_path: Path) -> xr.Dataset:
    """Load a regridded CMIP6 model file."""
    logger.info("Loading model: %s", model_path.name)
    ds = xr.open_dataset(model_path, chunks={})
    return subset_domain(ds)


# ===================================================================
#  Domain subsetting
# ===================================================================

def subset_domain(ds: xr.Dataset) -> xr.Dataset:
    """Clip dataset to the Indian Ocean domain (lat -40:30, lon 20:120)."""
    ds = ds.sel(lat=slice(*LAT_RANGE))
    lon_sel = ds.sel(lon=slice(*LON_RANGE))
    if lon_sel.sizes.get("lon", 0) == 0:
        ds = ds.sortby("lon")
        ds = ds.sel(lon=slice(*LON_RANGE))
    else:
        ds = lon_sel
    return ds


# ===================================================================
#  Common period matching
# ===================================================================

def match_common_period(obs: xr.Dataset, model: xr.Dataset):
    t0 = max(obs.time.values.min(), model.time.values.min())
    t1 = min(obs.time.values.max(), model.time.values.max())
    logger.info("  Common period: %s to %s", str(t0)[:7], str(t1)[:7])
    obs = obs.sel(time=slice(t0, t1))
    model = model.sel(time=slice(t0, t1))
    return obs, model


# ===================================================================
#  Metrics computation
# ===================================================================

def _sst(da: xr.Dataset) -> xr.DataArray:
    if "sst" in da.data_vars:
        arr = da["sst"]
    else:
        arr = da[list(da.data_vars)[0]]
    for d in ("lev", "depth", "lev_1", "depth_1"):
        if d in arr.dims:
            arr = arr.isel({d: 0})
    return arr


def compute_metrics(obs: xr.Dataset, model: xr.Dataset) -> dict:
    """Compute all validation metrics between obs and model on the same grid."""
    obs_sst = _sst(obs)
    model_sst = _sst(model)

    metrics = {}

    metrics["obs_mean"] = obs_sst.mean(dim="time")
    metrics["model_mean"] = model_sst.mean(dim="time")

    metrics["bias"] = metrics["model_mean"] - metrics["obs_mean"]

    se = (model_sst - obs_sst) ** 2
    metrics["rmse"] = np.sqrt(se.mean(dim="time"))

    metrics["obs_std"] = obs_sst.std(dim="time", ddof=1)
    metrics["model_std"] = model_sst.std(dim="time", ddof=1)
    metrics["std_ratio"] = metrics["model_std"] / metrics["obs_std"]

    metrics["spatial_corr"] = _spatial_correlation(obs_sst, model_sst)

    metrics["obs_monthly_clim"] = _monthly_climatology(obs_sst)
    metrics["model_monthly_clim"] = _monthly_climatology(model_sst)

    metrics["obs_domain_mean"] = float(obs_sst.mean(dim=("lat", "lon", "time")).values)
    metrics["model_domain_mean"] = float(model_sst.mean(dim=("lat", "lon", "time")).values)
    metrics["domain_bias"] = metrics["model_domain_mean"] - metrics["obs_domain_mean"]
    metrics["domain_rmse"] = float(
        np.sqrt(((model_sst - obs_sst) ** 2).mean(dim=("lat", "lon", "time")).values)
    )

    corr = metrics["spatial_corr"]
    metrics["domain_corr"] = float(corr.mean().values)

    return metrics


def _spatial_correlation(obs_arr: xr.DataArray, model_arr: xr.DataArray) -> xr.DataArray:
    cov = ((obs_arr - obs_arr.mean(dim="time")) * (model_arr - model_arr.mean(dim="time"))).mean(dim="time")
    std_obs = obs_arr.std(dim="time", ddof=1)
    std_mod = model_arr.std(dim="time", ddof=1)
    denom = std_obs * std_mod
    corr = cov / denom.where(denom > 0)
    return corr


def _monthly_climatology(arr: xr.DataArray) -> xr.DataArray:
    return arr.groupby("time.month").mean(dim="time")


# ===================================================================
#  Plotting
# ===================================================================

def _ensure_output_dir(model_name: str) -> Path:
    out_dir = OUTPUTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_mean_sst(obs_mean: xr.DataArray, model_mean: xr.DataArray, model_name: str, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        logger.warning("cartopy/matplotlib not available — skipping mean_sst_map")
        return

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        ncols=2, figsize=(14, 5),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )

    titles = ("NOAA OISST Mean SST", f"{model_name} Mean SST")
    data = (obs_mean, model_mean)
    vmin = min(d.min().values for d in data)
    vmax = max(d.max().values for d in data)

    for ax, title, da in zip(axes, titles, data):
        _draw_map(ax, da, vmin, vmax, title, "°C")

    fig.savefig(out_dir / "mean_sst_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out_dir / "mean_sst_map.png")


def plot_bias_map(bias: xr.DataArray, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
    except ImportError:
        logger.warning("cartopy unavailable — skipping bias_map")
        return

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": proj}, constrained_layout=True)

    vmax = max(abs(bias.min().values), abs(bias.max().values))
    _draw_map(ax, bias, -vmax, vmax, "SST Bias (Model − Obs)", "°C", cmap="RdBu_r")

    fig.savefig(out_dir / "bias_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out_dir / "bias_map.png")


def plot_rmse_map(rmse: xr.DataArray, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
    except ImportError:
        logger.warning("cartopy unavailable — skipping rmse_map")
        return

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": proj}, constrained_layout=True)

    _draw_map(ax, rmse, 0, rmse.max().values, "RMSE", "°C", cmap="Oranges")

    fig.savefig(out_dir / "rmse_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out_dir / "rmse_map.png")


def _draw_map(ax, da, vmin, vmax, title, cbar_label, cmap="viridis") -> None:
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    pcm = ax.pcolormesh(
        da.lon, da.lat, da.values,
        vmin=vmin, vmax=vmax, cmap=cmap, transform=ax.projection,
    )
    ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="none", zorder=2)
    ax.coastlines(linewidth=0.5, zorder=3)
    ax.set_extent([*LON_RANGE, *LAT_RANGE], crs=ax.projection)
    ax.set_title(title, fontsize=12)
    cb = plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.06, shrink=0.75)
    cb.set_label(cbar_label)


def plot_monthly_climatology(
    obs_clim: xr.DataArray, model_clim: xr.DataArray, out_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable — skipping monthly_climatology")
        return

    months = np.arange(1, 13)
    obs_spatial = obs_clim.mean(dim=("lat", "lon"))
    model_spatial = model_clim.mean(dim=("lat", "lon"))

    fig, axes = plt.subplots(ncols=2, figsize=(14, 5), constrained_layout=True)

    ax = axes[0]
    ax.plot(months, obs_spatial.values, "o-", label="NOAA OISST", color="C0")
    ax.plot(months, model_spatial.values, "s--", label="Model", color="C3")
    ax.set_xlabel("Month")
    ax.set_ylabel("SST (°C)")
    ax.set_title("Domain-average Monthly Climatology")
    ax.legend()
    ax.set_xticks(months)

    bias_clim = model_clim - obs_clim
    bias_spatial = bias_clim.mean(dim=("lat", "lon"))
    ax = axes[1]
    ax.bar(months, bias_spatial.values, color="C1", alpha=0.8)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Month")
    ax.set_ylabel("Bias (°C)")
    ax.set_title("Climatology Bias (Model − Obs)")
    ax.set_xticks(months)

    fig.savefig(out_dir / "monthly_climatology_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out_dir / "monthly_climatology_comparison.png")


def plot_timeseries(obs: xr.Dataset, model: xr.Dataset, model_name: str, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable — skipping timeseries")
        return

    obs_ts = _sst(obs).mean(dim=("lat", "lon"))
    model_ts = _sst(model).mean(dim=("lat", "lon"))

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.plot(obs_ts.time, obs_ts.values, label="NOAA OISST", linewidth=0.8, color="C0")
    ax.plot(model_ts.time, model_ts.values, label=model_name, linewidth=0.8, color="C3", alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("SST (°C)")
    ax.set_title(f"Domain-average SST — Indian Ocean  ({model_name})")
    ax.legend()

    fig.savefig(out_dir / "timeseries_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out_dir / "timeseries_comparison.png")


# ===================================================================
#  Save metrics
# ===================================================================

def save_metrics_csv(metrics: dict, out_dir: Path) -> None:
    logger.info("  Computing CSV metrics …")

    bias_data, rmse_data, corr_data, obs_mean_data, model_mean_data, obs_std_data, model_std_data = dask.compute(
        metrics["bias"],
        metrics["rmse"],
        metrics["spatial_corr"],
        metrics["obs_mean"],
        metrics["model_mean"],
        metrics["obs_std"],
        metrics["model_std"],
    )

    lat_grid = bias_data.lat.values
    lon_grid = bias_data.lon.values
    n_lat, n_lon = len(lat_grid), len(lon_grid)
    total = n_lat * n_lon

    lat_2d = np.repeat(lat_grid, n_lon)
    lon_2d = np.tile(lon_grid, n_lat)

    df = pd.DataFrame({
        "lat": lat_2d,
        "lon": lon_2d,
        "obs_mean_sst": obs_mean_data.values.ravel(),
        "model_mean_sst": model_mean_data.values.ravel(),
        "bias": bias_data.values.ravel(),
        "rmse": rmse_data.values.ravel(),
        "correlation": corr_data.values.ravel(),
        "obs_std": obs_std_data.values.ravel(),
        "model_std": model_std_data.values.ravel(),
    })
    df.to_csv(out_dir / "validation_metrics.csv", index=False)
    logger.info("  Saved %s (%d grid cells)", out_dir / "validation_metrics.csv", total)


def save_summary_txt(metrics: dict, model_name: str, out_dir: Path) -> None:
    lines = [
        "=" * 60,
        f"Validation Summary — NOAA OISST vs {model_name}",
        "=" * 60,
        f"Region        : Indian Ocean (lat {LAT_RANGE[0]}:{LAT_RANGE[1]}, "
        f"lon {LON_RANGE[0]}:{LON_RANGE[1]})",
        "",
        "Domain-averaged Metrics",
        "-" * 30,
        f"Obs mean SST      : {metrics['obs_domain_mean']:7.3f} °C",
        f"Model mean SST    : {metrics['model_domain_mean']:7.3f} °C",
        f"Mean bias         : {metrics['domain_bias']:+7.3f} °C",
        f"Domain RMSE       : {metrics['domain_rmse']:7.3f} °C",
        f"Spatial corr      : {metrics['domain_corr']:7.4f}",
        "",
    ]

    corr_data = metrics["spatial_corr"].compute().values
    lines += [
        f"Spatial correlation (median) : {float(np.nanmedian(corr_data)):7.4f}",
        f"Spatial correlation (min)    : {float(np.nanmin(corr_data)):7.4f}",
        f"Spatial correlation (max)    : {float(np.nanmax(corr_data)):7.4f}",
        "",
        "Files written",
        "-" * 30,
    ]

    for f in sorted(out_dir.glob("*")):
        if f.is_file():
            lines.append(f"  {f.name}")

    text = "\n".join(lines)
    (out_dir / "validation_summary.txt").write_text(text, encoding="utf-8")
    logger.info("  Saved %s", out_dir / "validation_summary.txt")
    logger.info("\n%s", text)


# ===================================================================
#  Cross-model ranking
# ===================================================================

def save_model_ranking(all_results: list[dict]) -> None:
    """Write a single CSV ranking all models by RMSE, bias, and correlation."""
    rows = []
    for r in all_results:
        rows.append({
            "model": r["model"],
            "domain_bias": r["domain_bias"],
            "domain_rmse": r["domain_rmse"],
            "domain_corr": r["domain_corr"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("domain_rmse")
    df["rank"] = range(1, len(df) + 1)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / "model_ranking.csv"
    df.to_csv(path, index=False)
    logger.info("")
    logger.info("=" * 60)
    logger.info("  MODEL RANKING  (by RMSE, lower is better)")
    logger.info("=" * 60)
    for _, row in df.iterrows():
        logger.info(
            "  %2d.  %-20s  bias=%+6.3f  rmse=%.3f  corr=%.4f",
            row["rank"], row["model"],
            row["domain_bias"], row["domain_rmse"], row["domain_corr"],
        )
    logger.info("Saved %s", path)


# ===================================================================
#  Per-model validation
# ===================================================================

def validate_one_model(
    model_name: str, model_path: Path, obs: xr.Dataset
) -> dict:
    """Run the full validation pipeline for a single CMIP6 model."""
    logger.info("")
    logger.info("--------------------------------------------------")
    logger.info("  Validating %s", model_name)
    logger.info("--------------------------------------------------")

    t0 = time.perf_counter()
    out_dir = _ensure_output_dir(model_name)

    model = load_model(model_path)

    obs_matched, model_matched = match_common_period(obs, model)

    metrics = compute_metrics(obs_matched, model_matched)

    plot_mean_sst(metrics["obs_mean"], metrics["model_mean"], model_name, out_dir)
    plot_bias_map(metrics["bias"], out_dir)
    plot_rmse_map(metrics["rmse"], out_dir)
    plot_monthly_climatology(metrics["obs_monthly_clim"], metrics["model_monthly_clim"], out_dir)
    plot_timeseries(obs_matched, model_matched, model_name, out_dir)

    save_metrics_csv(metrics, out_dir)
    save_summary_txt(metrics, model_name, out_dir)

    logger.info("  Metadata saved to %s", out_dir)

    elapsed = time.perf_counter() - t0
    logger.info("  Finished %s  (%.1f s)", model_name, elapsed)

    return {
        "model": model_name,
        "domain_bias": metrics["domain_bias"],
        "domain_rmse": metrics["domain_rmse"],
        "domain_corr": metrics["domain_corr"],
    }


# ===================================================================
#  Main
# ===================================================================

def main() -> None:
    overall_start = time.perf_counter()
    setup_logging()

    logger.info("=" * 60)
    logger.info("  START validation.py")
    logger.info("=" * 60)

    if not NOAA_PROCESSED.exists():
        logger.error("NOAA processed file not found — run preprocess.py first")
        return

    obs = load_noaa()

    models = discover_regridded_models()
    if not models:
        logger.warning("No regridded CMIP6 models found in %s", PROCESSED_DIR)
        logger.warning("Run preprocess.py and regrid.py first")
        return

    logger.info("Found %d CMIP6 model(s) to validate", len(models))
    all_results: list[dict] = []

    for model_name, model_path in models:
        try:
            result = validate_one_model(model_name, model_path, obs)
            all_results.append(result)
        except Exception as e:
            logger.error("Validation of %s FAILED: %s", model_name, e)
            traceback.print_exc()

    if len(all_results) > 1:
        save_model_ranking(all_results)

    logger.info("")
    logger.info("  All done  —  total time: %.1f s", time.perf_counter() - overall_start)


if __name__ == "__main__":
    main()
