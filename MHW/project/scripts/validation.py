"""
validation.py
=============
Compare NOAA OISST observations with CMIP6 historical SST simulations
over the Indian Ocean.

Workflow
--------
1. Load NOAA OISST and CMIP6 historical SST NetCDF files.
2. Subset to Indian Ocean domain (lat -40:30, lon 20:120).
3. Intersect time periods to a common window.
4. Regrid CMIP6 onto the NOAA grid with xESMF (bilinear).
5. Compute validation metrics:
   - Mean SST, monthly climatology, bias, RMSE, correlation, std.
6. Generate maps and time-series plots.
7. Save validation_metrics.csv and validation_summary.txt.

Outputs (in outputs/validation/)
--------------------------------
- mean_sst_map.png
- bias_map.png
- rmse_map.png
- monthly_climatology_comparison.png
- timeseries_comparison.png
- validation_metrics.csv
- validation_summary.txt
"""

import logging
import logging.config
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Region of interest — Indian Ocean
# ---------------------------------------------------------------------------
LAT_RANGE = (-40.0, 30.0)
LON_RANGE = (20.0, 120.0)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def setup_logging(config: dict) -> None:
    """Initialise logging from a configuration dictionary."""
    level = config.get("logging", {}).get("level", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Logging initialised at %s level", level)


def load_config(config_path: str = "config/default.yaml") -> dict:
    """Load YAML configuration from disk (returns empty dict if missing)."""
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file not found at %s — using defaults", path)
        return {}
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Configuration loaded from %s", path)
    return cfg or {}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _standardise_ds(ds: xr.Dataset) -> xr.Dataset:
    """Rename coordinates and variables to a common convention.

    Maps common OISST / CMIP6 names to 'lat', 'lon', 'sst'.
    """
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
    """Shift longitudes to [-180, 180] if needed for Indian Ocean domain."""
    if ds.lon.max() > 180.0:
        ds = ds.assign_coords(lon=(ds.lon % 360))
        ds = ds.sortby("lon")
    return ds


def load_datasets(obs_path: str, model_path: str, cfg: dict):
    """Read, standardise, and spatially subset both datasets.

    Parameters
    ----------
    obs_path : str
        Path to NOAA OISST NetCDF.
    model_path : str
        Path to CMIP6 historical SST NetCDF.
    cfg : dict
        Configuration dictionary.

    Returns
    -------
    obs : xr.Dataset
    model : xr.Dataset
    """
    logger.info("Loading NOAA OISST: %s", obs_path)
    obs = xr.open_dataset(obs_path, decode_times=True)
    obs = _standardise_ds(obs)
    obs = _ensure_lon_range(obs)

    logger.info("Loading CMIP6 historical: %s", model_path)
    model = xr.open_dataset(model_path, decode_times=True)
    model = _standardise_ds(model)
    model = _ensure_lon_range(model)

    obs = subset_indian_ocean(obs)
    model = subset_indian_ocean(model)

    return obs, model


def subset_indian_ocean(ds: xr.Dataset) -> xr.Dataset:
    """Clip dataset to the Indian Ocean domain.

    Lat: -40 to 30, Lon: 20 to 120.
    """
    ds = ds.sel(lat=slice(*LAT_RANGE))
    lon_sel = ds.sel(lon=slice(*LON_RANGE))
    if lon_sel.sizes.get("lon", 0) == 0:
        ds = ds.sortby("lon")
        ds = ds.sel(lon=slice(*LON_RANGE))
    else:
        ds = lon_sel
    logger.info(
        "Subset to Indian Ocean — shape lat=%d, lon=%d",
        ds.sizes.get("lat", -1),
        ds.sizes.get("lon", -1),
    )
    return ds


def match_common_period(obs: xr.Dataset, model: xr.Dataset):
    """Intersect the time axes of obs and model.

    Returns aligned datasets with a common time coordinate.
    """
    t0 = max(obs.time.values.min(), model.time.values.min())
    t1 = min(obs.time.values.max(), model.time.values.max())
    logger.info("Common period: %s to %s", str(t0)[:7], str(t1)[:7])
    obs = obs.sel(time=slice(t0, t1))
    model = model.sel(time=slice(t0, t1))
    return obs, model


def regrid_cmip6_to_noaa(model: xr.Dataset, obs: xr.Dataset) -> xr.Dataset:
    """Regrid CMIP6 SST onto the NOAA OISST grid using xESMF.

    Falls back to xarray's ``interp`` if xESMF is unavailable.
    """
    try:
        import xesmf as xe
    except ImportError:
        logger.warning("xESMF not available — using xarray interp (nearest neighbour)")
        return _regrid_interp(model, obs)

    regridder = xe.Regridder(
        model, obs, method="bilinear", periodic=False,
        reuse_weights=True,
    )
    logger.info("xESMF regridder created — %s", regridder)
    model_regridded = regridder(model, keep_attrs=True)
    model_regridded = model_regridded.assign_coords(
        {c: obs[c].values for c in ("lat", "lon") if c in obs.coords}
    )
    return model_regridded


def _regrid_interp(model: xr.Dataset, obs: xr.Dataset) -> xr.Dataset:
    """Fallback regridding via xarray.interp (nearest neighbour)."""
    model_regridded = model.interp(
        lat=obs.lat, lon=obs.lon, method="nearest",
        kwargs={"fill_value": np.nan},
    )
    return model_regridded


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _sst(da: xr.Dataset) -> xr.DataArray:
    """Extract the SST variable as a DataArray, squeezing extra dims."""
    if "sst" in da.data_vars:
        arr = da["sst"]
    else:
        arr = da[list(da.data_vars)[0]]
    for d in ("lev", "depth", "lev_1", "depth_1"):
        if d in arr.dims:
            arr = arr.isel({d: 0})
    return arr


def compute_metrics(obs: xr.Dataset, model: xr.Dataset) -> dict:
    """Compute all validation metrics between obs and regridded model.

    Returns a dictionary of DataArrays / scalars.
    """
    obs_sst = _sst(obs)
    model_sst = _sst(model)

    metrics = {}

    # -- mean SST --
    metrics["obs_mean"] = obs_sst.mean(dim="time")
    metrics["model_mean"] = model_sst.mean(dim="time")

    # -- bias --
    metrics["bias"] = metrics["model_mean"] - metrics["obs_mean"]

    # -- RMSE --
    se = (model_sst - obs_sst) ** 2
    metrics["rmse"] = np.sqrt(se.mean(dim="time"))

    # -- temporal std --
    metrics["obs_std"] = obs_sst.std(dim="time", ddof=1)
    metrics["model_std"] = model_sst.std(dim="time", ddof=1)
    metrics["std_ratio"] = metrics["model_std"] / metrics["obs_std"]

    # -- spatial correlation (temporal pattern) --
    metrics["spatial_corr"] = _spatial_correlation(obs_sst, model_sst)

    # -- monthly climatology --
    metrics["obs_monthly_clim"] = _monthly_climatology(obs_sst)
    metrics["model_monthly_clim"] = _monthly_climatology(model_sst)

    # -- domain averages --
    metrics["obs_domain_mean"] = float(obs_sst.mean(dim=("lat", "lon", "time")).values)
    metrics["model_domain_mean"] = float(model_sst.mean(dim=("lat", "lon", "time")).values)
    metrics["domain_bias"] = metrics["model_domain_mean"] - metrics["obs_domain_mean"]
    metrics["domain_rmse"] = float(
        np.sqrt(((model_sst - obs_sst) ** 2).mean(dim=("lat", "lon", "time")).values)
    )

    return metrics


def _spatial_correlation(obs_arr: xr.DataArray, model_arr: xr.DataArray) -> xr.DataArray:
    """Pearson correlation coefficient along time for each grid cell."""
    with np.errstate(invalid="ignore"):
        cov = ((obs_arr - obs_arr.mean(dim="time")) * (model_arr - model_arr.mean(dim="time"))).mean(dim="time")
        std_obs = obs_arr.std(dim="time", ddof=1)
        std_mod = model_arr.std(dim="time", ddof=1)
        corr = cov / (std_obs * std_mod)
    return corr


def _monthly_climatology(arr: xr.DataArray) -> xr.DataArray:
    """Compute the 12-month climatology (groupby month)."""
    return arr.groupby("time.month").mean(dim="time")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _ensure_output_dir(cfg: dict) -> Path:
    out = Path(cfg.get("paths", {}).get("validation", "outputs/validation"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_mean_sst(obs_mean: xr.DataArray, model_mean: xr.DataArray, out_dir: Path) -> None:
    """Side‑by‑side map of observed and modelled mean SST."""
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        logger.warning("cartopy / matplotlib not available — skipping plot_mean_sst")
        return

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        ncols=2, figsize=(14, 5),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )

    titles = ("NOAA OISST Mean SST", "CMIP6 Historical Mean SST")
    data = (obs_mean, model_mean)
    vmin = min(d.min().values for d in data)
    vmax = max(d.max().values for d in data)

    for ax, title, da in zip(axes, titles, data):
        _draw_map(ax, da, vmin, vmax, title, "°C")

    fig.savefig(out_dir / "mean_sst_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_dir / "mean_sst_map.png")


def plot_bias_map(bias: xr.DataArray, out_dir: Path) -> None:
    """Map of mean bias (model − obs)."""
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
    except ImportError:
        logger.warning("cartopy / matplotlib unavailable — skipping plot_bias_map")
        return

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": proj}, constrained_layout=True)

    vmax = max(abs(bias.min().values), abs(bias.max().values))
    _draw_map(ax, bias, -vmax, vmax, "SST Bias (Model − Obs)", "°C", cmap="RdBu_r")

    fig.savefig(out_dir / "bias_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_dir / "bias_map.png")


def plot_rmse_map(rmse: xr.DataArray, out_dir: Path) -> None:
    """Map of root‑mean‑squared error."""
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
    except ImportError:
        logger.warning("cartopy / matplotlib unavailable — skipping plot_rmse_map")
        return

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": proj}, constrained_layout=True)

    _draw_map(ax, rmse, 0, rmse.max().values, "RMSE", "°C", cmap="Oranges")

    fig.savefig(out_dir / "rmse_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_dir / "rmse_map.png")


def _draw_map(ax, da, vmin, vmax, title, cbar_label, cmap="viridis") -> None:
    """Add a colour‑filled contour map to an existing cartopy axis."""
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
    """Two‑panel comparison: spatial mean climatology curve + panel of maps."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable — skipping plot_monthly_climatology")
        return

    months = np.arange(1, 13)
    obs_spatial = obs_clim.mean(dim=("lat", "lon"))
    model_spatial = model_clim.mean(dim=("lat", "lon"))

    fig, axes = plt.subplots(ncols=2, figsize=(14, 5), constrained_layout=True)

    # left — time series
    ax = axes[0]
    ax.plot(months, obs_spatial.values, "o-", label="NOAA OISST", color="C0")
    ax.plot(months, model_spatial.values, "s--", label="CMIP6 Historical", color="C3")
    ax.set_xlabel("Month")
    ax.set_ylabel("SST (°C)")
    ax.set_title("Domain‑average Monthly Climatology")
    ax.legend()
    ax.set_xticks(months)

    # right — bias of climatology
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
    logger.info("Saved %s", out_dir / "monthly_climatology_comparison.png")


def plot_timeseries(obs: xr.Dataset, model: xr.Dataset, out_dir: Path) -> None:
    """Domain‑averaged SST time series for observation and model."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable — skipping plot_timeseries")
        return

    obs_ts = _sst(obs).mean(dim=("lat", "lon"))
    model_ts = _sst(model).mean(dim=("lat", "lon"))

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.plot(obs_ts.time, obs_ts.values, label="NOAA OISST", linewidth=0.8, color="C0")
    ax.plot(model_ts.time, model_ts.values, label="CMIP6 Historical", linewidth=0.8, color="C3", alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("SST (°C)")
    ax.set_title("Domain‑average SST Time Series — Indian Ocean")
    ax.legend()

    fig.savefig(out_dir / "timeseries_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_dir / "timeseries_comparison.png")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_metrics_csv(metrics: dict, out_dir: Path) -> None:
    """Write per‑grid‑cell metrics to a CSV file (long format).

    Flattens the spatial dimension into rows.
    """
    bias_flat = metrics["bias"].stack(cell=("lat", "lon")).to_dataframe(name="bias")
    rmse_flat = metrics["rmse"].stack(cell=("lat", "lon")).to_dataframe(name="rmse")
    corr_flat = metrics["spatial_corr"].stack(cell=("lat", "lon")).to_dataframe(name="correlation")
    obs_mean_flat = metrics["obs_mean"].stack(cell=("lat", "lon")).to_dataframe(name="obs_mean_sst")
    model_mean_flat = metrics["model_mean"].stack(cell=("lat", "lon")).to_dataframe(name="model_mean_sst")
    obs_std_flat = metrics["obs_std"].stack(cell=("lat", "lon")).to_dataframe(name="obs_std")
    model_std_flat = metrics["model_std"].stack(cell=("lat", "lon")).to_dataframe(name="model_std")

    df = (
        obs_mean_flat
        .join(model_mean_flat)
        .join(bias_flat)
        .join(rmse_flat)
        .join(corr_flat)
        .join(obs_std_flat)
        .join(model_std_flat)
    )
    df.index = pd.MultiIndex.from_tuples(df.index, names=["lat", "lon"])
    df = df.reset_index()
    df.to_csv(out_dir / "validation_metrics.csv", index=False)
    logger.info("Saved %s (%d grid cells)", out_dir / "validation_metrics.csv", len(df))


def save_summary_txt(metrics: dict, out_dir: Path) -> None:
    """Write a human‑readable summary of domain‑averaged metrics."""
    lines = [
        "=" * 60,
        "Validation Summary — NOAA OISST vs CMIP6 Historical",
        "=" * 60,
        f"Region        : Indian Ocean (lat {LAT_RANGE[0]}:{LAT_RANGE[1]}, "
        f"lon {LON_RANGE[0]}:{LON_RANGE[1]})",
        "",
        "Domain‑averaged Metrics",
        "-" * 30,
        f"Obs mean SST      : {metrics['obs_domain_mean']:7.3f} °C",
        f"Model mean SST    : {metrics['model_domain_mean']:7.3f} °C",
        f"Mean bias         : {metrics['domain_bias']:+7.3f} °C",
        f"Domain RMSE       : {metrics['domain_rmse']:7.3f} °C",
        "",
    ]

    corr = metrics["spatial_corr"]
    lines += [
        f"Spatial correlation (median) : {float(corr.median().values):7.4f}",
        f"Spatial correlation (min)    : {float(corr.min().values):7.4f}",
        f"Spatial correlation (max)    : {float(corr.max().values):7.4f}",
        "",
        "Files written",
        "-" * 30,
    ]

    out_dir = Path(out_dir)
    for f in sorted(out_dir.glob("*")):
        if f.is_file():
            lines.append(f"  {f.name}")

    text = "\n".join(lines)
    (out_dir / "validation_summary.txt").write_text(text, encoding="utf-8")
    logger.info("Saved %s", out_dir / "validation_summary.txt")
    logger.info("\n%s", text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    setup_logging(cfg)

    logger.info("=" * 60)
    logger.info("START validation.py")
    logger.info("=" * 60)

    out_dir = _ensure_output_dir(cfg)

    obs_path = cfg.get("paths", {}).get(
        "noaa_processed", "data/processed/noaa_oisst.nc"
    )
    model_path = cfg.get("paths", {}).get(
        "cmip6_historical_processed", "data/processed/cmip6_historical.nc"
    )

    # 1 — Load and subset
    obs, model = load_datasets(obs_path, model_path, cfg)

    # 2 — Match time periods
    obs, model = match_common_period(obs, model)

    # 3 — Regrid model onto obs grid
    model_regridded = regrid_cmip6_to_noaa(model, obs)

    # 4 — Compute metrics
    metrics = compute_metrics(obs, model_regridded)

    # 5 — Generate plots
    plot_mean_sst(metrics["obs_mean"], metrics["model_mean"], out_dir)
    plot_bias_map(metrics["bias"], out_dir)
    plot_rmse_map(metrics["rmse"], out_dir)
    plot_monthly_climatology(metrics["obs_monthly_clim"], metrics["model_monthly_clim"], out_dir)
    plot_timeseries(obs, model_regridded, out_dir)

    # 6 — Save tabular results
    save_metrics_csv(metrics, out_dir)
    save_summary_txt(metrics, out_dir)

    logger.info("=" * 60)
    logger.info("END validation.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
