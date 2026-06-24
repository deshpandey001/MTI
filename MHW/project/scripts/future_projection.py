"""
future_projection.py
=====================
Apply bias-correction factors (derived from the historical period) to
CMIP6 future projection SST fields under SSP2-4.5 and SSP5-8.5.

Climate-science rationale
-------------------------
Coupled climate models exhibit systematic biases that are assumed to be
stationary — i.e., the same additive error present in the historical
simulation also affects the future projection.  By removing the
historical bias from future runs, we obtain a more credible estimate of
the climate-change signal at regional scales.

Workflow
--------
1. Load the correction factors (``historical_bias.nc``) produced by
   ``bias_correction.py``.
2. Load raw CMIP6 SSP2-4.5 and SSP5-8.5 SST fields.
3. Apply monthly-bias correction to both scenarios.
4. Compute and plot:
   - SST trend maps (linear trend per grid cell)
   - Annual mean SST time series
   - Monthly climatology comparison across Historical / SSP245 / SSP585
5. Save corrected future datasets as NetCDF.

Outputs
-------
- outputs/bias_corrected/future_corrected_ssp245.nc
- outputs/bias_corrected/future_corrected_ssp585.nc
- outputs/bias_maps/future_*_trend.png
- outputs/bias_maps/future_*_climatology.png
- outputs/bias_maps/scenario_comparison.png
- outputs/validation/future_projection_report.txt
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain — Indian Ocean (matching the rest of the project)
# ---------------------------------------------------------------------------
LAT_RANGE = (-40.0, 30.0)
LON_RANGE = (20.0, 120.0)

# Scenario display / file naming
SCENARIOS = {
    "ssp245": {"label": "SSP2-4.5", "slug": "ssp245"},
    "ssp585": {"label": "SSP5-8.5", "slug": "ssp585"},
}


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def setup_logging(config: dict) -> None:
    level = config.get("logging", {}).get("level", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str = "config/default.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config %s not found — using defaults", path)
        return {}
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg or {}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _standardise_ds(ds: xr.Dataset) -> xr.Dataset:
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
    elif "thetao" in ds.data_vars and "sst" not in ds.data_vars:
        rename_vars["thetao"] = "sst"
    if rename_vars:
        ds = ds.rename(rename_vars)

    return ds


def _ensure_lon_range(ds: xr.Dataset) -> xr.Dataset:
    if "lon" in ds.coords and float(ds.lon.max()) > 180.0:
        ds = ds.assign_coords(lon=(ds.lon.values % 360))
        ds = ds.sortby("lon")
    return ds


def _sst(da: xr.Dataset | xr.DataArray) -> xr.DataArray:
    if isinstance(da, xr.DataArray):
        return da
    if "sst" in da.data_vars:
        arr = da["sst"]
    else:
        arr = da[list(da.data_vars)[0]]
    for d in ("lev", "depth", "lev_1", "depth_1"):
        if d in arr.dims:
            arr = arr.isel({d: 0})
    return arr


def subset_domain(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.sel(lat=slice(*LAT_RANGE))
    lon_sel = ds.sel(lon=slice(*LON_RANGE))
    if lon_sel.sizes.get("lon", 0) == 0:
        ds = ds.sortby("lon")
        ds = ds.sel(lon=slice(*LON_RANGE))
    else:
        ds = lon_sel
    return ds


def regrid_to_reference(model: xr.Dataset, ref: xr.Dataset) -> xr.Dataset:
    try:
        import xesmf as xe

        regridder = xe.Regridder(
            model, ref, method="bilinear", periodic=False, reuse_weights=True
        )
        regridded = regridder(model, keep_attrs=True)
        regridded = regridded.assign_coords(
            {c: ref[c].values for c in ("lat", "lon") if c in ref.coords}
        )
        return regridded
    except ImportError:
        logger.warning("xESMF not available — using xarray interp")
        regridded = model.interp(
            lat=ref.lat, lon=ref.lon, method="nearest",
            kwargs={"fill_value": np.nan},
        )
        return regridded


def load_scenario(
    path: str, ref_grid: xr.Dataset | None = None
) -> xr.Dataset:
    """Load, standardise, and optionally regrid a single scenario file."""
    logger.info("Loading: %s", path)
    ds = xr.open_dataset(path, decode_times=True)
    ds = _standardise_ds(ds)
    ds = _ensure_lon_range(ds)
    ds = subset_domain(ds)

    if ref_grid is not None:
        ds = regrid_to_reference(ds, ref_grid)

    return ds


# ---------------------------------------------------------------------------
# Correction-factor loading
# ---------------------------------------------------------------------------

def load_correction_factors(path: str) -> dict:
    """Load ``historical_bias.nc`` and return the correction fields.

    Returns a dict with:
      - ``monthly_bias`` : DataArray (month, lat, lon) — monthly additive bias
      - ``method``       : str — the correction method used
      - ``coords``       : dict of coordinate arrays for regridding
    """
    logger.info("Loading correction factors: %s", path)
    ds = xr.open_dataset(path, decode_times=True)

    factors: dict = {}

    if "monthly_bias" in ds.data_vars:
        factors["monthly_bias"] = ds["monthly_bias"]
    elif "bias" in ds.data_vars:
        factors["monthly_bias"] = ds["bias"]
    else:
        raise KeyError(
            "Correction file must contain 'monthly_bias' or 'bias' variable."
        )

    factors["method"] = ds.attrs.get("method", "unknown")
    factors["coords"] = {"lat": ds.lat.values, "lon": ds.lon.values}

    logger.info("Correction method: %s", factors["method"])
    return factors


# ---------------------------------------------------------------------------
# Correction application
# ---------------------------------------------------------------------------

def apply_correction(
    future_arr: xr.DataArray,
    monthly_bias: xr.DataArray,
) -> xr.DataArray:
    """Subtract the monthly climatological bias from the future SST.

    The bias field is regridded to the future grid if needed, then
    matched by calendar month and subtracted cell-wise.

    Parameters
    ----------
    future_arr : DataArray
        Raw future SST with dims (time, lat, lon).
    monthly_bias : DataArray
        Monthly bias with dims (month, lat, lon) or (lat, lon).

    Returns
    -------
    corrected_arr : DataArray
        Bias-corrected future SST.
    """
    logger.info("Applying monthly bias correction to future SST …")

    if "month" in monthly_bias.dims:
        bias_on_future_grid = monthly_bias
        if not np.allclose(
            future_arr.lat.values, monthly_bias.lat.values
        ) or not np.allclose(future_arr.lon.values, monthly_bias.lon.values):
            bias_on_future_grid = monthly_bias.interp(
                lat=future_arr.lat, lon=future_arr.lon, method="nearest",
                kwargs={"fill_value": 0.0},
            )

        corrected = future_arr.groupby("time.month") - bias_on_future_grid
    else:
        bias_on_future_grid = monthly_bias
        if not np.allclose(
            future_arr.lat.values, monthly_bias.lat.values
        ) or not np.allclose(future_arr.lon.values, monthly_bias.lon.values):
            bias_on_future_grid = monthly_bias.interp(
                lat=future_arr.lat, lon=future_arr.lon, method="nearest",
                kwargs={"fill_value": 0.0},
            )
        corrected = future_arr - bias_on_future_grid

    return corrected


# ---------------------------------------------------------------------------
# Trend computation
# ---------------------------------------------------------------------------

def compute_linear_trend(
    arr: xr.DataArray, time_dim: str = "time"
) -> xr.DataArray:
    """Least-squares linear trend (°C per year) for each grid cell.

    Fits  y = a + b * t  where t is in years since the start of the
    record.  Returns the slope *b*.
    """
    t = (arr[time_dim] - arr[time_dim].min()) / np.timedelta64(1, "Y")
    t_norm = t - t.mean()

    arr_mean = arr.mean(dim=time_dim)
    cov = ((arr - arr_mean) * t_norm).mean(dim=time_dim)
    var_t = (t_norm ** 2).mean()

    slope = cov / var_t
    slope = slope.assign_attrs(
        units="°C / year",
        long_name="Linear SST trend",
    )
    return slope


def compute_annual_mean(arr: xr.DataArray) -> xr.DataArray:
    """Resample to annual means, preserving the spatial structure."""
    return arr.resample(time="YE").mean(dim="time")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _draw_map(ax, da, vmin, vmax, title, cbar_label, cmap="RdBu_r"):
    import cartopy.feature as cfeature

    pcm = ax.pcolormesh(
        da.lon, da.lat, da.values,
        vmin=vmin, vmax=vmax, cmap=cmap, transform=ax.projection,
    )
    ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="none", zorder=2)
    ax.coastlines(linewidth=0.5, zorder=3)
    ax.set_extent([*LON_RANGE, *LAT_RANGE], crs=ax.projection)
    ax.set_title(title, fontsize=12)
    cb = ax.figure.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.06, shrink=0.75)
    cb.set_label(cbar_label)


def plot_trend_map(
    trend: xr.DataArray, scenario_label: str, out_dir: Path
) -> None:
    """Spatial map of the linear SST trend for a single scenario."""
    try:
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Plotting unavailable — skipping trend map for %s", scenario_label)
        return

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": proj})
    vmax = float(max(abs(trend.min()), abs(trend.max())))
    _draw_map(
        ax, trend, -vmax, vmax,
        f"SST Trend — {scenario_label}",
        "°C / yr", cmap="RdBu_r",
    )
    slug = scenario_label.replace(".", "").replace("-", "_").lower()
    fig.savefig(out_dir / f"future_{slug}_trend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved trend map: %s", out_dir / f"future_{slug}_trend.png")


def plot_monthly_climatology(
    clim_hist: xr.DataArray,
    clim_ssp245: xr.DataArray,
    clim_ssp585: xr.DataArray,
    out_dir: Path,
) -> None:
    """Monthly climatology comparison: Historical vs SSP245 vs SSP585.

    Left panel: domain-averaged monthly SST cycle.
    Right panel: warming relative to historical (SSP − Hist).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Plotting unavailable — skipping climatology comparison")
        return

    months = np.arange(1, 13)
    hist_spatial = clim_hist.mean(dim=("lat", "lon"))
    ssp245_spatial = clim_ssp245.mean(dim=("lat", "lon"))
    ssp585_spatial = clim_ssp585.mean(dim=("lat", "lon"))

    fig, axes = plt.subplots(ncols=2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(months, hist_spatial.values, "o-", label="Historical", color="C0")
    ax.plot(months, ssp245_spatial.values, "s--", label="SSP2-4.5", color="C1")
    ax.plot(months, ssp585_spatial.values, "D-.", label="SSP5-8.5", color="C3")
    ax.set_xlabel("Month")
    ax.set_ylabel("SST (°C)")
    ax.set_title("Domain-average Monthly Climatology")
    ax.legend()
    ax.set_xticks(months)

    ax = axes[1]
    ax.bar(months - 0.2, (ssp245_spatial - hist_spatial).values, 0.35,
           label="SSP2-4.5 − Hist", color="C1", alpha=0.8)
    ax.bar(months + 0.2, (ssp585_spatial - hist_spatial).values, 0.35,
           label="SSP5-8.5 − Hist", color="C3", alpha=0.8)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Month")
    ax.set_ylabel("Warming (°C)")
    ax.set_title("Monthly Warming Relative to Historical")
    ax.legend()
    ax.set_xticks(months)

    fig.savefig(out_dir / "future_climatology.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_dir / "future_climatology.png")


def plot_scenario_comparison(
    hist_arr: xr.DataArray,
    ssp245_arr: xr.DataArray,
    ssp585_arr: xr.DataArray,
    out_dir: Path,
) -> None:
    """Three-panel: mean SST maps for Historical, SSP245, SSP585."""
    try:
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Plotting unavailable — skipping scenario comparison")
        return

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        ncols=3, figsize=(18, 5),
        subplot_kw={"projection": proj},
    )

    titles = ["Historical", "SSP2-4.5", "SSP5-8.5"]
    data = [
        hist_arr.mean(dim="time"),
        ssp245_arr.mean(dim="time"),
        ssp585_arr.mean(dim="time"),
    ]
    vmin = float(min(d.min() for d in data))
    vmax = float(max(d.max() for d in data))

    for ax, title, da in zip(axes, titles, data):
        _draw_map(ax, da, vmin, vmax, title, "°C", cmap="viridis")

    fig.savefig(out_dir / "scenario_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_dir / "scenario_comparison.png")


def plot_annual_timeseries(
    hist_annual: xr.DataArray,
    ssp245_annual: xr.DataArray,
    ssp585_annual: xr.DataArray,
    out_dir: Path,
) -> None:
    """Domain-averaged annual SST time series for all three periods."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Plotting unavailable — skipping annual time series")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    def _label(arr):
        return float(arr.mean(dim=("lat", "lon")).values)

    for label, arr, color, ls in [
        ("Historical", hist_annual, "C0", "-"),
        ("SSP2-4.5 (corrected)", ssp245_annual, "C1", "--"),
        ("SSP5-8.5 (corrected)", ssp585_annual, "C3", "-."),
    ]:
        ts = arr.mean(dim=("lat", "lon"))
        ax.plot(ts.time, ts.values, label=label, color=color, linestyle=ls, linewidth=1.0)

    ax.set_xlabel("Year")
    ax.set_ylabel("SST (°C)")
    ax.set_title("Domain-average Annual SST — Indian Ocean")
    ax.legend()
    fig.savefig(out_dir / "future_annual_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_dir / "future_annual_timeseries.png")


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def generate_report(
    hist_arr: xr.DataArray,
    ssp245_raw: xr.DataArray,
    ssp585_raw: xr.DataArray,
    ssp245_corr: xr.DataArray,
    ssp585_corr: xr.DataArray,
    method: str,
    out_dir: Path,
) -> None:
    """Write a summary of the future projection correction."""
    hist_mean = float(hist_arr.mean(dim=("time", "lat", "lon")).values)
    ssp245_raw_mean = float(ssp245_raw.mean(dim=("time", "lat", "lon")).values)
    ssp585_raw_mean = float(ssp585_raw.mean(dim=("time", "lat", "lon")).values)
    ssp245_corr_mean = float(ssp245_corr.mean(dim=("time", "lat", "lon")).values)
    ssp585_corr_mean = float(ssp585_corr.mean(dim=("time", "lat", "lon")).values)

    hist_trend = compute_linear_trend(hist_arr)
    ssp245_trend = compute_linear_trend(ssp245_corr)
    ssp585_trend = compute_linear_trend(ssp585_corr)

    hist_trend_domain = float(hist_trend.mean(dim=("lat", "lon")).values)
    ssp245_trend_domain = float(ssp245_trend.mean(dim=("lat", "lon")).values)
    ssp585_trend_domain = float(ssp585_trend.mean(dim=("lat", "lon")).values)

    lines = [
        "=" * 70,
        "FUTURE PROJECTION — BIAS CORRECTION REPORT",
        "=" * 70,
        f"Correction method         : {method}",
        f"Domain                    : Indian Ocean"
        f"  (lat {LAT_RANGE[0]}:{LAT_RANGE[1]}, lon {LON_RANGE[0]}:{LON_RANGE[1]})",
        "",
        "Domain-averaged SST",
        "-" * 40,
        f"  Historical (raw)        : {hist_mean:>8.3f} °C",
        f"  SSP2-4.5 (raw)          : {ssp245_raw_mean:>8.3f} °C",
        f"  SSP2-4.5 (corrected)    : {ssp245_corr_mean:>8.3f} °C",
        f"  SSP5-8.5 (raw)          : {ssp585_raw_mean:>8.3f} °C",
        f"  SSP5-8.5 (corrected)    : {ssp585_corr_mean:>8.3f} °C",
        "",
        "Domain-averaged linear trend",
        "-" * 40,
        f"  Historical              : {hist_trend_domain:>+7.4f} °C / yr",
        f"  SSP2-4.5 (corrected)    : {ssp245_trend_domain:>+7.4f} °C / yr",
        f"  SSP5-8.5 (corrected)    : {ssp585_trend_domain:>+7.4f} °C / yr",
        "",
        "Scenario warming (corrected − historical)",
        "-" * 40,
        f"  SSP2-4.5                : {ssp245_corr_mean - hist_mean:>+8.3f} °C",
        f"  SSP5-8.5                : {ssp585_corr_mean - hist_mean:>+8.3f} °C",
        "",
        "Files written",
        "-" * 40,
    ]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(out_dir.glob("*")):
        if f.is_file():
            lines.append(f"  {f.name}")

    text = "\n".join(lines)
    (out_dir / "future_projection_report.txt").write_text(text, encoding="utf-8")
    logger.info("Saved %s", out_dir / "future_projection_report.txt")
    logger.info("\n%s", text)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_future_pipeline(
    correction_path: str,
    historical_path: str,
    ssp245_path: str,
    ssp585_path: str,
    out_dir_bc: str = "outputs/bias_corrected",
    out_dir_maps: str = "outputs/bias_maps",
    out_dir_val: str = "outputs/validation",
) -> dict[str, str]:
    """Apply historical bias correction to future CMIP6 scenarios.

    Parameters
    ----------
    correction_path : str
        Path to ``historical_bias.nc`` produced by bias_correction.py.
    historical_path : str
        Path to the (already bias-corrected) historical NetCDF.
    ssp245_path : str
        Path to raw CMIP6 SSP2-4.5 SST NetCDF.
    ssp585_path : str
        Path to raw CMIP6 SSP5-8.5 SST NetCDF.
    out_dir_bc : str
        Output directory for corrected future NetCDFs.
    out_dir_maps : str
        Output directory for maps.
    out_dir_val : str
        Output directory for the validation report.

    Returns
    -------
    paths : dict
        ``{"ssp245": path, "ssp585": path}`` to the saved corrected files.
    """
    logger.info("=" * 70)
    logger.info("FUTURE PROJECTION PIPELINE")
    logger.info("=" * 70)

    # 1 — Load correction factors
    factors = load_correction_factors(correction_path)
    monthly_bias = factors["monthly_bias"]
    method = factors["method"]

    # 2 — Load historical (already corrected)
    logger.info("Loading historical (corrected) SST: %s", historical_path)
    hist_ds = xr.open_dataset(historical_path, decode_times=True)
    hist_ds = _standardise_ds(hist_ds)
    hist_ds = _ensure_lon_range(hist_ds)
    hist_ds = subset_domain(hist_ds)
    hist_arr = _sst(hist_ds)

    ref_grid = hist_ds  # use historical grid as reference for regridding

    # 3 — Load future scenarios
    ssp245_ds = load_scenario(ssp245_path, ref_grid=ref_grid)
    ssp585_ds = load_scenario(ssp585_path, ref_grid=ref_grid)

    ssp245_arr = _sst(ssp245_ds)
    ssp585_arr = _sst(ssp585_ds)

    # 4 — Apply bias correction
    ssp245_corr_arr = apply_correction(ssp245_arr, monthly_bias)
    ssp585_corr_arr = apply_correction(ssp585_arr, monthly_bias)

    # 5 — Rebuild datasets
    ssp245_corr_ds = ssp245_ds.copy()
    ssp245_corr_ds["sst"] = ssp245_corr_arr
    ssp245_corr_ds.attrs["bias_correction"] = f"Applied {method} from historical period"

    ssp585_corr_ds = ssp585_ds.copy()
    ssp585_corr_ds["sst"] = ssp585_corr_arr
    ssp585_corr_ds.attrs["bias_correction"] = f"Applied {method} from historical period"

    # 6 — Compute trends and annual means
    ssp245_trend = compute_linear_trend(ssp245_corr_arr)
    ssp585_trend = compute_linear_trend(ssp585_corr_arr)

    ssp245_annual = compute_annual_mean(ssp245_corr_arr)
    ssp585_annual = compute_annual_mean(ssp585_corr_arr)
    hist_annual = compute_annual_mean(hist_arr)

    # 7 — Monthly climatologies
    hist_clim = hist_arr.groupby("time.month").mean(dim="time")
    ssp245_clim = ssp245_corr_arr.groupby("time.month").mean(dim="time")
    ssp585_clim = ssp585_corr_arr.groupby("time.month").mean(dim="time")

    # 8 — Generate maps
    maps_dir = _ensure_dir(Path(out_dir_maps))

    plot_trend_map(ssp245_trend, "SSP2-4.5", maps_dir)
    plot_trend_map(ssp585_trend, "SSP5-8.5", maps_dir)

    plot_monthly_climatology(hist_clim, ssp245_clim, ssp585_clim, maps_dir)
    plot_scenario_comparison(hist_arr, ssp245_corr_arr, ssp585_corr_arr, maps_dir)
    plot_annual_timeseries(hist_annual, ssp245_annual, ssp585_annual, maps_dir)

    # 9 — Validation report
    val_dir = _ensure_dir(Path(out_dir_val))
    generate_report(
        hist_arr,
        ssp245_arr,
        ssp585_arr,
        ssp245_corr_arr,
        ssp585_corr_arr,
        method,
        val_dir,
    )

    # 10 — Save corrected datasets
    bc_dir = _ensure_dir(Path(out_dir_bc))
    out_ssp245 = bc_dir / "future_corrected_ssp245.nc"
    out_ssp585 = bc_dir / "future_corrected_ssp585.nc"

    ssp245_corr_ds.to_netcdf(out_ssp245)
    logger.info("Saved corrected SSP2-4.5 → %s", out_ssp245)

    ssp585_corr_ds.to_netcdf(out_ssp585)
    logger.info("Saved corrected SSP5-8.5 → %s", out_ssp585)

    logger.info("=" * 70)
    logger.info("FUTURE PROJECTION PIPELINE COMPLETE")
    logger.info("=" * 70)

    return {"ssp245": str(out_ssp245), "ssp585": str(out_ssp585)}


def main() -> None:
    cfg = load_config()
    setup_logging(cfg)

    p = cfg.get("paths", {})

    correction_path = p.get(
        "historical_bias", "outputs/bias_corrected/historical_bias.nc"
    )
    historical_path = p.get(
        "historical_corrected", "outputs/bias_corrected/corrected_sst.nc"
    )
    ssp245_path = p.get(
        "cmip6_ssp245_raw", "data/cmip6_future/cmip6_ssp245.nc"
    )
    ssp585_path = p.get(
        "cmip6_ssp585_raw", "data/cmip6_future/cmip6_ssp585.nc"
    )

    out_bc = p.get("bias_corrected", "outputs/bias_corrected")
    out_maps = p.get("bias_maps", "outputs/bias_maps")
    out_val = p.get("validation", "outputs/validation")

    run_future_pipeline(
        correction_path,
        historical_path,
        ssp245_path,
        ssp585_path,
        out_bc,
        out_maps,
        out_val,
    )


if __name__ == "__main__":
    main()
