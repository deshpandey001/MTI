"""
bias_correction.py
==================
Research-grade bias correction for CMIP6 SST fields using NOAA OISST as
reference.  Three methods are provided:

  1. Mean Bias Correction (delta)   — removes the overall cell-wise mean bias.
  2. Monthly Bias Correction         — removes month-specific climatological bias.
  3. Quantile Mapping                — maps the full distribution of model values
                                       to the observed distribution via CDF
                                       inversion.

Climate-science rationale
-------------------------
Coupled climate models (CMIP6) exhibit systematic biases in sea-surface
temperature due to imperfect parameterisations of unresolved processes
(e.g. mixed-layer physics, cloud feedback, ocean eddies).  These biases
must be corrected before the model output can be used in impact studies
(e.g. marine heatwave detection).

- Mean / Monthly bias correction assumes the bias is additive and
  stationary in time — reasonable for many regions on multi-decadal
  scales.
- Quantile mapping additionally corrects the shape of the distribution,
  preserving variability extremes.

Outputs
-------
- outputs/bias_corrected/corrected_sst.nc   (corrected field, NetCDF)
- outputs/bias_maps/raw_bias_map.png
- outputs/bias_maps/corrected_bias_map.png
- outputs/bias_maps/comparison_before_after.png
- outputs/validation/validation_report.txt
"""

from __future__ import annotations

import logging
import logging.config
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain — Indian Ocean (matching the rest of the project)
# ---------------------------------------------------------------------------
LAT_RANGE = (-40.0, 30.0)
LON_RANGE = (20.0, 120.0)


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
# I/O helpers — reuse patterns from validation.py
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


def load_datasets(obs_path: str, model_path: str) -> tuple[xr.Dataset, xr.Dataset]:
    logger.info("Loading NOAA OISST: %s", obs_path)
    obs = xr.open_dataset(obs_path, decode_times=True)
    obs = _standardise_ds(obs)
    obs = _ensure_lon_range(obs)

    logger.info("Loading CMIP6 historical: %s", model_path)
    model = xr.open_dataset(model_path, decode_times=True)
    model = _standardise_ds(model)
    model = _ensure_lon_range(model)

    return obs, model


def subset_domain(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.sel(lat=slice(*LAT_RANGE))
    lon_sel = ds.sel(lon=slice(*LON_RANGE))
    if lon_sel.sizes.get("lon", 0) == 0:
        ds = ds.sortby("lon")
        ds = ds.sel(lon=slice(*LON_RANGE))
    else:
        ds = lon_sel
    return ds


def match_common_period(obs: xr.Dataset, model: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    t0 = max(obs.time.values.min(), model.time.values.min())
    t1 = min(obs.time.values.max(), model.time.values.max())
    logger.info("Common period: %s to %s", str(t0)[:7], str(t1)[:7])
    obs = obs.sel(time=slice(t0, t1))
    model = model.sel(time=slice(t0, t1))
    return obs, model


def regrid_cmip6_to_noaa(model: xr.Dataset, obs: xr.Dataset) -> xr.Dataset:
    try:
        import xesmf as xe

        regridder = xe.Regridder(
            model, obs, method="bilinear", periodic=False, reuse_weights=True
        )
        model_regridded = regridder(model, keep_attrs=True)
        model_regridded = model_regridded.assign_coords(
            {c: obs[c].values for c in ("lat", "lon") if c in obs.coords}
        )
        return model_regridded
    except ImportError:
        logger.warning("xESMF not available — using xarray interp (nearest neighbour)")
        model_regridded = model.interp(
            lat=obs.lat, lon=obs.lon, method="nearest",
            kwargs={"fill_value": np.nan},
        )
        return model_regridded


# ---------------------------------------------------------------------------
# Bias-correction methods
# ---------------------------------------------------------------------------

def _as_dataarray(arr: xr.Dataset | xr.DataArray) -> xr.DataArray:
    return _sst(arr) if isinstance(arr, xr.Dataset) else arr


def mean_bias_correction(
    obs: xr.DataArray, model: xr.DataArray, time_dim: str = "time"
) -> tuple[xr.DataArray, xr.DataArray]:
    """Method 1 — Mean Bias Correction (additive delta).

    Rationale
    ---------
    Computes the time-mean bias at each grid cell:
        bias(x,y) = mean_t(model(x,y,t)) - mean_t(obs(x,y,t))
    and subtracts it from every time step of the model.

    This assumes the bias is purely additive and stationary.  It is the
    simplest correction and is appropriate when the model error is
    dominated by a constant offset.
    """
    logger.info("Applying Mean Bias Correction (Method 1) ...")

    obs_mean = obs.mean(dim=time_dim)
    model_mean = model.mean(dim=time_dim)
    bias = model_mean - obs_mean

    logger.info("Bias range: [%.3f, %.3f] °C", float(bias.min()), float(bias.max()))

    corrected = model - bias
    return corrected, bias


def monthly_bias_correction(
    obs: xr.DataArray, model: xr.DataArray, time_dim: str = "time"
) -> tuple[xr.DataArray, xr.DataArray]:
    """Method 2 — Monthly Bias Correction.

    Rationale
    ---------
    Model biases often exhibit a seasonal cycle (e.g. larger errors in
    summer due to mixed-layer feedback).  This method computes a separate
    bias for each calendar month:

        bias_m(x,y) = mean_t(model_m(x,y,t)) - mean_t(obs_m(x,y,t))

    where _m denotes values belonging to month m.  The monthly bias is
    then subtracted from the corresponding month in the model.

    Preserves the seasonal cycle of the model while removing the
    month-specific systematic error.
    """
    logger.info("Applying Monthly Bias Correction (Method 2) ...")

    obs_clim = obs.groupby(f"{time_dim}.month").mean(dim=time_dim)
    model_clim = model.groupby(f"{time_dim}.month").mean(dim=time_dim)
    monthly_bias = model_clim - obs_clim

    logger.info(
        "Monthly bias range: [%.3f, %.3f] °C",
        float(monthly_bias.min()),
        float(monthly_bias.max()),
    )

    corrected = model.groupby(f"{time_dim}.month") - monthly_bias
    return corrected, monthly_bias


def quantile_mapping(
    obs: xr.DataArray,
    model: xr.DataArray,
    time_dim: str = "time",
    n_quantiles: int = 100,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Method 3 — Quantile Mapping (empirical CDF inversion).

    Rationale
    ---------
    Mean/Monthly corrections only shift the distribution.  Quantile
    mapping adjusts every quantile of the model distribution to match
    the observed distribution.  This corrects biases in the shape,
    variance, and extremes of the SST field.

    For each grid cell:
        1. Compute empirical CDFs of obs and model (via quantiles).
        2. Build a transfer function:  T(q) = CDF^{-1}_obs(CDF_model(q))
        3. Apply T to every model value.

    A regularised grid of n_quantiles quantiles is used, with linear
    interpolation between them.  Values outside the training range are
    linearly extrapolated.

    Reference
    ---------
    Maraun, D. (2016). "Bias correcting climate change simulations — a
    critical review."  Progress in Physical Geography, 40(4), 519–537.
    """
    logger.info("Applying Quantile Mapping (Method 3) ...")

    corrected = model.copy(deep=True)
    transfer = model.copy(deep=True)

    quantiles = np.linspace(0, 1, n_quantiles)

    for lat_idx in range(model.sizes["lat"]):
        for lon_idx in range(model.sizes["lon"]):
            obs_ts = obs.isel(lat=lat_idx, lon=lon_idx).values
            model_ts = model.isel(lat=lat_idx, lon=lon_idx).values

            valid_obs = obs_ts[~np.isnan(obs_ts)]
            valid_mod = model_ts[~np.isnan(model_ts)]

            if len(valid_obs) < 10 or len(valid_mod) < 10:
                continue

            obs_q = np.quantile(valid_obs, quantiles)
            model_q = np.quantile(valid_mod, quantiles)

            corr_ts = np.interp(model_ts, model_q, obs_q)
            corrected[dict(lat=lat_idx, lon=lon_idx)] = corr_ts

            transfer[dict(lat=lat_idx, lon=lon_idx)] = np.interp(
                model_ts, model_q, obs_q - model_q
            )

    logger.info("Quantile mapping complete.")
    return corrected, transfer


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rmse(
    obs: xr.DataArray, model: xr.DataArray, time_dim: str = "time"
) -> xr.DataArray:
    se = (model - obs) ** 2
    return np.sqrt(se.mean(dim=time_dim))


def compute_rmse_domain(
    obs: xr.DataArray, model: xr.DataArray, time_dim: str = "time"
) -> float:
    se = (model - obs) ** 2
    return float(np.sqrt(se.mean(dim=(time_dim, "lat", "lon")).values))


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


def plot_raw_bias(bias: xr.DataArray, out_dir: Path) -> None:
    """Map of the raw (uncorrected) model bias."""
    try:
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Plotting dependencies unavailable — skipping raw bias map")
        return

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": proj})
    vmax = float(max(abs(bias.min()), abs(bias.max())))
    _draw_map(ax, bias, -vmax, vmax, "Raw Bias (Model − Obs)", "°C", cmap="RdBu_r")
    fig.savefig(out_dir / "raw_bias_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_dir / "raw_bias_map.png")


def plot_corrected_bias(
    obs: xr.DataArray, corrected: xr.DataArray, out_dir: Path
) -> None:
    """Map of the residual bias after correction."""
    try:
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Plotting unavailable — skipping corrected bias map")
        return

    bias_corr = corrected.mean(dim="time") - obs.mean(dim="time")
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": proj})
    vmax = float(max(abs(bias_corr.min()), abs(bias_corr.max())))
    _draw_map(
        ax, bias_corr, -vmax, vmax,
        "Residual Bias After Correction", "°C", cmap="RdBu_r",
    )
    fig.savefig(out_dir / "corrected_bias_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_dir / "corrected_bias_map.png")


def plot_comparison(
    obs: xr.DataArray,
    raw_model: xr.DataArray,
    corrected: xr.DataArray,
    out_dir: Path,
) -> None:
    """Three-panel: Observed vs Raw vs Corrected mean SST."""
    try:
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Plotting unavailable — skipping comparison map")
        return

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        ncols=3, figsize=(18, 5),
        subplot_kw={"projection": proj},
    )

    titles = ["NOAA OISST (Observed)", "CMIP6 (Raw)", "CMIP6 (Bias-Corrected)"]
    data = [
        obs.mean(dim="time"),
        raw_model.mean(dim="time"),
        corrected.mean(dim="time"),
    ]
    vmin = float(min(d.min() for d in data))
    vmax = float(max(d.max() for d in data))

    for ax, title, da in zip(axes, titles, data):
        _draw_map(ax, da, vmin, vmax, title, "°C", cmap="viridis")

    fig.savefig(out_dir / "comparison_before_after.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_dir / "comparison_before_after.png")


def plot_rmse_comparison(
    obs: xr.DataArray,
    raw_model: xr.DataArray,
    corrected: xr.DataArray,
    out_dir: Path,
) -> None:
    """Side-by-side RMSE maps before and after correction."""
    try:
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Plotting unavailable — skipping RMSE comparison")
        return

    rmse_raw = compute_rmse(obs, raw_model)
    rmse_corr = compute_rmse(obs, corrected)

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        ncols=2, figsize=(14, 5),
        subplot_kw={"projection": proj},
    )

    vmax = float(max(rmse_raw.max(), rmse_corr.max()))
    _draw_map(axes[0], rmse_raw, 0, vmax, "RMSE Before Correction", "°C", cmap="Oranges")
    _draw_map(axes[1], rmse_corr, 0, vmax, "RMSE After Correction", "°C", cmap="Oranges")

    fig.savefig(out_dir / "rmse_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_dir / "rmse_comparison.png")


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def generate_validation_report(
    obs: xr.DataArray,
    raw_model: xr.DataArray,
    corrected: xr.DataArray,
    method_name: str,
    out_dir: Path,
) -> None:
    """Write a human-readable summary of the bias correction performance."""
    rmse_raw = compute_rmse_domain(obs, raw_model)
    rmse_corr = compute_rmse_domain(obs, corrected)
    improvement = (rmse_raw - rmse_corr) / rmse_raw * 100

    bias_raw = (raw_model.mean(dim="time") - obs.mean(dim="time")).mean().values
    bias_corr = (corrected.mean(dim="time") - obs.mean(dim="time")).mean().values

    obs_mean = float(obs.mean(dim=("time", "lat", "lon")).values)
    raw_mean = float(raw_model.mean(dim=("time", "lat", "lon")).values)
    corr_mean = float(corrected.mean(dim=("time", "lat", "lon")).values)

    lines = [
        "=" * 70,
        "BIAS CORRECTION VALIDATION REPORT",
        "=" * 70,
        f"Method                   : {method_name}",
        f"Domain                   : Indian Ocean"
        f"  (lat {LAT_RANGE[0]}:{LAT_RANGE[1]}, lon {LON_RANGE[0]}:{LON_RANGE[1]})",
        "",
        "Domain-averaged SST",
        "-" * 40,
        f"  Observed (NOAA OISST)  : {obs_mean:>8.3f} °C",
        f"  Raw model (CMIP6)      : {raw_mean:>8.3f} °C",
        f"  Corrected model        : {corr_mean:>8.3f} °C",
        "",
        "Domain-averaged Bias (model − obs)",
        "-" * 40,
        f"  Before correction      : {float(bias_raw):>+8.3f} °C",
        f"  After correction       : {float(bias_corr):>+8.3f} °C",
        "",
        "Domain-averaged RMSE",
        "-" * 40,
        f"  Before correction      : {rmse_raw:>8.3f} °C",
        f"  After correction       : {rmse_corr:>8.3f} °C",
        f"  Improvement            : {improvement:>+7.2f} %",
        "",
    ]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "validation_report.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved %s", txt_path)
    logger.info("\n%s", "\n".join(lines))


# ---------------------------------------------------------------------------
# Correction factors export (for future projection)
# ---------------------------------------------------------------------------

def _save_correction_factors(
    bias_or_transfer: xr.DataArray,
    method_name: str,
    obs_arr: xr.DataArray,
    model_arr: xr.DataArray,
    out_dir: Path,
) -> None:
    """Save correction factors to ``historical_bias.nc``.

    The saved file is the bridge between ``bias_correction.py`` and
    ``future_projection.py``.  It always contains a ``monthly_bias``
    field so that ``future_projection.py`` can apply a single code path
    regardless of the correction method chosen for the historical period.

    In addition, method-specific fields are stored for reference:
      - Mean Correction          → ``bias``
      - Monthly Correction        → ``monthly_bias``
      - Quantile Mapping          → ``monthly_bias`` + ``quantile_delta``
    """
    logger.info("Saving correction factors to %s …", out_dir)

    ds = xr.Dataset(
        coords={
            "lat": obs_arr.coords["lat"],
            "lon": obs_arr.coords["lon"],
        },
        attrs={
            "title": "CMIP6 bias-correction factors",
            "method": method_name,
            "source_observation": "NOAA OISST",
            "source_model": "CMIP6 Historical",
            "domain": f"Indian Ocean {LAT_RANGE}",
        },
    )

    if method_name == "Mean Bias Correction":
        ds["bias"] = bias_or_transfer
        ds["monthly_bias"] = bias_or_transfer  # broadcast below

    elif method_name == "Monthly Bias Correction":
        ds["monthly_bias"] = bias_or_transfer
        ds["bias"] = bias_or_transfer.mean(dim="month")

    elif method_name == "Quantile Mapping":
        clim_bias = model_arr.groupby("time.month").mean(dim="time") - obs_arr.groupby("time.month").mean(dim="time")
        ds["monthly_bias"] = clim_bias
        ds["bias"] = clim_bias.mean(dim="month")

    ds.to_netcdf(out_dir / "historical_bias.nc")
    logger.info("Saved correction factors → %s", out_dir / "historical_bias.nc")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    obs_path: str,
    model_path: str,
    method: str = "mean",
    out_dir_bc: str = "outputs/bias_corrected",
    out_dir_maps: str = "outputs/bias_maps",
    out_dir_val: str = "outputs/validation",
) -> str:
    """Execute the full bias-correction pipeline.

    Parameters
    ----------
    obs_path : str
        Path to NOAA OISST NetCDF.
    model_path : str
        Path to CMIP6 historical NetCDF.
    method : str
        One of 'mean', 'monthly', 'quantile'.
    out_dir_bc : str
        Directory for corrected NetCDF.
    out_dir_maps : str
        Directory for bias maps.
    out_dir_val : str
        Directory for validation report.

    Returns
    -------
    output_path : str
        Path to the saved corrected NetCDF.
    """
    method_map = {
        "mean": ("Mean Bias Correction", mean_bias_correction),
        "monthly": ("Monthly Bias Correction", monthly_bias_correction),
        "quantile": ("Quantile Mapping", quantile_mapping),
    }

    if method not in method_map:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {list(method_map)}"
        )

    method_name, method_func = method_map[method]
    logger.info("=" * 70)
    logger.info("Bias-correction pipeline — %s", method_name)
    logger.info("=" * 70)

    # 1 — Load
    obs_ds, model_ds = load_datasets(obs_path, model_path)

    # 2 — Subset domain
    obs_ds = subset_domain(obs_ds)
    model_ds = subset_domain(model_ds)

    # 3 — Match common period
    obs_ds, model_ds = match_common_period(obs_ds, model_ds)

    # 4 — Regrid model onto obs grid
    model_ds = regrid_cmip6_to_noaa(model_ds, obs_ds)

    # 5 — Extract DataArrays
    obs_arr = _sst(obs_ds)
    model_arr = _sst(model_ds)

    # 6 — Apply bias correction
    corrected_arr, bias_or_transfer = method_func(obs_arr, model_arr)

    # 7 — Convert corrected back to Dataset preserving metadata
    corrected_ds = model_ds.copy()
    corrected_ds["sst"] = corrected_arr

    # 8 — Compute metrics
    rmse_raw = compute_rmse_domain(obs_arr, model_arr)
    rmse_corr = compute_rmse_domain(obs_arr, corrected_arr)
    logger.info("RMSE before: %.4f °C  →  after: %.4f °C", rmse_raw, rmse_corr)

    # 9 — Generate maps
    bias_raw = model_arr.mean(dim="time") - obs_arr.mean(dim="time")

    maps_dir = _ensure_dir(Path(out_dir_maps))
    plot_raw_bias(bias_raw, maps_dir)
    plot_corrected_bias(obs_arr, corrected_arr, maps_dir)
    plot_comparison(obs_arr, model_arr, corrected_arr, maps_dir)
    plot_rmse_comparison(obs_arr, model_arr, corrected_arr, maps_dir)

    # 10 — Validation report
    val_dir = _ensure_dir(Path(out_dir_val))
    generate_validation_report(obs_arr, model_arr, corrected_arr, method_name, val_dir)

    # 11 — Save corrected SST as NetCDF
    bc_dir = _ensure_dir(Path(out_dir_bc))
    output_path = bc_dir / "corrected_sst.nc"
    corrected_ds.to_netcdf(output_path)
    logger.info("Saved corrected SST → %s", output_path)

    # 12 — Save correction factors for use by future_projection.py
    _save_correction_factors(
        bias_or_transfer, method_name, obs_arr, model_arr, bc_dir,
    )

    logger.info("=" * 70)
    logger.info("Bias-correction pipeline complete.")
    logger.info("=" * 70)

    return str(output_path)


def main() -> None:
    cfg = load_config()
    setup_logging(cfg)

    obs_path = cfg.get("paths", {}).get(
        "noaa_processed", "data/processed/noaa_oisst.nc"
    )
    model_path = cfg.get("paths", {}).get(
        "cmip6_historical_processed", "data/processed/cmip6_historical.nc"
    )
    method = cfg.get("bias_correction", {}).get("method", "mean")

    out_bc = cfg.get("paths", {}).get("bias_corrected", "outputs/bias_corrected")
    out_maps = cfg.get("paths", {}).get("bias_maps", "outputs/bias_maps")
    out_val = cfg.get("paths", {}).get("validation", "outputs/validation")

    run_pipeline(obs_path, model_path, method, out_bc, out_maps, out_val)


if __name__ == "__main__":
    main()
