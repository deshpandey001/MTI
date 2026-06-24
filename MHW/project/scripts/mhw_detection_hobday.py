"""
mhw_detection_hobday.py
=======================
Official Hobday et al. (2016) Marine Heatwave detection on daily SST data.

Methodology (Hobday et al., 2016, Progress in Oceanography)
------------------------------------------------------------
1. Compute a daily climatology from a 30-year baseline period using an
   11-day centred moving average to reduce synoptic noise.
2. Compute the 90th percentile threshold (smoothed identically).
3. An MHW event occurs when SST exceeds the threshold for ≥ 5 contiguous
   days.  Two events separated by ≤ 2 days are merged (the "join" rule).
4. Categories are defined by multiples of the threshold anomaly:
   - Category I  (Moderate):   threshold < SST ≤ threshold + T_{anom}
   - Category II (Strong):     threshold + T_{anom} < SST ≤ threshold + 2T_{anom}
   - Category III (Severe):    threshold + 2T_{anom} < SST ≤ threshold + 3T_{anom}
   - Category IV (Extreme):    SST > threshold + 3T_{anom}
   where T_{anom} = threshold − climatology.

Climate-science rationale
-------------------------
Marine heatwaves have profound impacts on ecosystem structure,
fisheries, and carbon cycling.  Consistent detection based on a
fixed, observation-based baseline enables inter-comparison across
models, scenarios, and regions.

Input
-----
- Daily SST NetCDF (1982–2025)
- Optionally a baseline period (default 1982–2011)

Outputs (outputs/mhw/)
----------------------
- mhw_metrics.nc       — per-grid-cell event statistics
- mhw_events_raw.nc    — daily boolean / category field (optional)
- *_event_frequency.png
- *_mean_duration.png
- *_mean_intensity.png
- *_max_intensity.png
- *_total_mhw_days.png
- *_cumulative_intensity.png
- *_max_category.png
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
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
# Hobday defaults
# ---------------------------------------------------------------------------
BASELINE = (1982, 2011)
MIN_DURATION = 5          # minimum consecutive days for an MHW
WINDOW = 11               # smoothing window (odd)
THRESHOLD_PCTILE = 90     # percentile for the threshold
MAX_GAP = 2               # max days between events to merge


# ---------------------------------------------------------------------------
# Config helpers
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


# ---------------------------------------------------------------------------
# Step 1 — Climatology & threshold  (Hobday Eqs. 1–3)
# ---------------------------------------------------------------------------

def compute_climatology_and_threshold(
    sst: xr.DataArray,
    baseline: tuple[int, int] = BASELINE,
    window: int = WINDOW,
    pctile: float = THRESHOLD_PCTILE,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute the Hobday daily climatology and 90th percentile threshold.

    Parameters
    ----------
    sst : DataArray
        Daily SST with dims (time, lat, lon).  Must cover at least the
        full baseline period.
    baseline : (start, end)
        Baseline years for the climatology (e.g. 1982–2011).
    window : int
        Half-window size for the centred moving average (default 11).
    pctile : float
        Percentile for the threshold (default 90).

    Returns
    -------
    clim : DataArray           (dayofyear, lat, lon) — smoothed mean SST
    thresh : DataArray         (dayofyear, lat, lon) — smoothed 90th %ile

    Notes
    -----
    Per Hobday et al. (2016):
      - The climatology is the mean SST for each day-of-year averaged
        over the baseline, then smoothed with an 11-day moving window.
      - The threshold is the 90th percentile of the same distribution,
        smoothed identically.
    """
    logger.info(
        "Computing climatology & %dth percentile (baseline %d–%d, window=%d) …",
        pctile, baseline[0], baseline[1], window,
    )

    t0 = np.datetime64(f"{baseline[0]}-01-01")
    t1 = np.datetime64(f"{baseline[1]}-12-31")
    baseline_sst = sst.sel(time=slice(t0, t1))

    baseline_sst = baseline_sst.load()  # load into memory for performance

    doy = baseline_sst.time.dt.dayofyear

    clim_raw = baseline_sst.groupby(doy).mean(dim="time")
    thresh_raw = baseline_sst.groupby(doy).quantile(pctile / 100, dim="time")

    def _smooth(da: xr.DataArray, halved_coords: bool = True) -> xr.DataArray:
        """Apply a centred moving average of length ``window``.

        Uses periodic padding so that day-of-year 366 wraps to 1.
        """
        pad = window // 2
        padded = da.pad(dayofyear=pad, mode="wrap")
        smoothed = padded.rolling(dayofyear=window, center=True).mean()
        if halved_coords:
            smoothed = smoothed.isel(dayofyear=slice(pad, -pad))
        return smoothed

    clim = _smooth(clim_raw)
    thresh = _smooth(thresh_raw)

    logger.info("Climatology range: [%.3f, %.3f] °C", float(clim.min()), float(clim.max()))
    logger.info(
        "Threshold range:   [%.3f, %.3f] °C",
        float(thresh.min()), float(thresh.max()),
    )

    return clim, thresh


# ---------------------------------------------------------------------------
# Step 2 — Event detection per grid cell
# ---------------------------------------------------------------------------

def _fill_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill short gaps (≤ ``max_gap`` days) in a boolean array.

    Implements the Hobday merge rule: if two exceedance periods are
    separated by ≤ ``max_gap`` days, the gap is filled so they are
    treated as one continuous event.
    """
    result = mask.copy()
    n = len(result)
    i = 0
    while i < n:
        if not result[i]:
            i += 1
            continue
        start_block = i
        while i < n and result[i]:
            i += 1
        end_block = i - 1

        candidate_gap_end = min(end_block + max_gap + 1, n - 1)
        next_true = -1
        for j in range(end_block + 1, candidate_gap_end + 1):
            if result[j]:
                next_true = j
                break

        if next_true != -1:
            result[end_block + 1:next_true] = True
            i = next_true
    return result


def _label_events_1d(mask: np.ndarray, min_dur: int, max_gap: int) -> np.ndarray:
    """Label contiguous MHW events in a 1-D boolean array.

    Implements the Hobday gap-filling rule (merge ≤ ``max_gap`` days),
    then assigns unique integer IDs to events that meet ``min_dur``.

    Returns an integer array where 0 = no event, and event IDs start at 1.
    """
    filled = _fill_gaps(mask, max_gap)

    n = len(filled)
    labels = np.zeros(n, dtype=np.int32)
    event_id = 0
    i = 0
    while i < n:
        if not filled[i]:
            i += 1
            continue
        start = i
        while i < n and filled[i]:
            i += 1
        end = i - 1

        duration = end - start + 1
        if duration >= min_dur:
            event_id += 1
            labels[start:end + 1] = event_id

    return labels


def _detect_events_at_cell(
    sst_ts: np.ndarray,
    clim_doy: np.ndarray,
    thresh_doy: np.ndarray,
    doy_indices: np.ndarray,
    min_dur: int = MIN_DURATION,
    max_gap: int = MAX_GAP,
) -> dict:
    """Detect MHW events at a single grid cell.

    Parameters
    ----------
    sst_ts : (T,) ndarray
        Daily SST time series at one grid cell.
    clim_doy : (366,) ndarray
        Daily climatology (smoothed).
    thresh_doy : (366,) ndarray
        Daily threshold (smoothed 90th %ile).
    doy_indices : (T,) ndarray
        Day-of-year (1–366) for each time step.
    min_dur, max_gap : int
        Hobday parameters.

    Returns
    -------
    dict with arrays of shape (T,):
        - ``is_mhw``: bool — whether each day belongs to an MHW
        - ``intensity``: float — SST anomaly rel. to climatology
        - ``category``: int8 — Hobday category (0–4)
        - ``event_id``: int32 — unique event ID per cell
    """
    T = len(sst_ts)
    clim = clim_doy[doy_indices - 1]
    thresh = thresh_doy[doy_indices - 1]

    intensity = sst_ts - clim
    exceed = sst_ts > thresh

    labels = _label_events_1d(exceed, min_dur, max_gap)
    is_event = labels > 0

    anomaly = thresh - clim
    anomaly_safe = np.where(anomaly > 1e-6, anomaly, 1e-6)

    excess = intensity - (thresh - clim)
    cat = np.zeros(T, dtype=np.int8)
    cat[is_event] = 1
    cat[(excess >= anomaly_safe) & is_event] = 2
    cat[(excess >= 2 * anomaly_safe) & is_event] = 3
    cat[(excess >= 3 * anomaly_safe) & is_event] = 4

    return {
        "is_mhw": is_event,
        "intensity": intensity,
        "category": cat,
        "event_id": labels,
    }


def detect_events(
    sst: xr.DataArray,
    clim: xr.DataArray,
    thresh: xr.DataArray,
    min_dur: int = MIN_DURATION,
    max_gap: int = MAX_GAP,
) -> xr.Dataset:
    """Detect MHW events for every grid cell.

    Returns a Dataset with daily fields:
      - ``mhw_binary``  (time, lat, lon) — 1 if MHW day, else 0
      - ``intensity``   (time, lat, lon) — SST anomaly (°C)
      - ``category``    (time, lat, lon) — Hobday category (0–4)
      - ``event_id``    (time, lat, lon) — unique event ID per cell
    """
    logger.info("Detecting MHW events …")

    doy = sst.time.dt.dayofyear.values
    clim_np = clim.values
    thresh_np = thresh.values

    T = sst.sizes["time"]
    nlat = sst.sizes["lat"]
    nlon = sst.sizes["lon"]

    mhw_binary = np.zeros((T, nlat, nlon), dtype=bool)
    intensity = np.full((T, nlat, nlon), np.nan)
    category = np.zeros((T, nlat, nlon), dtype=np.int8)
    event_id = np.zeros((T, nlat, nlon), dtype=np.int32)

    sst_np = sst.values

    for j in range(nlat):
        if j % 10 == 0:
            logger.info("  Processing lat %d / %d …", j + 1, nlat)
        for k in range(nlon):
            cell = _detect_events_at_cell(
                sst_np[:, j, k],
                clim_np[:, j, k],
                thresh_np[:, j, k],
                doy,
                min_dur=min_dur,
                max_gap=max_gap,
            )
            mhw_binary[:, j, k] = cell["is_mhw"]
            intensity[:, j, k] = cell["intensity"]
            category[:, j, k] = cell["category"]
            event_id[:, j, k] = cell["event_id"]

    ds = xr.Dataset(
        data_vars={
            "mhw_binary": (("time", "lat", "lon"), mhw_binary),
            "intensity": (("time", "lat", "lon"), intensity),
            "category": (("time", "lat", "lon"), category),
            "event_id": (("time", "lat", "lon"), event_id),
        },
        coords={
            "time": sst.time,
            "lat": sst.lat,
            "lon": sst.lon,
        },
        attrs={
            "title": "MHW detection — Hobday et al. (2016)",
            "baseline": f"{BASELINE[0]}-{BASELINE[1]}",
            "min_duration_days": min_dur,
            "max_merge_gap_days": max_gap,
            "threshold_percentile": THRESHOLD_PCTILE,
        },
    )

    n_events_total = int((event_id > 0).sum())
    logger.info("Detected ~%d event-days across all grid cells", n_events_total)
    return ds


# ---------------------------------------------------------------------------
# Step 3 — Event statistics per grid cell
# ---------------------------------------------------------------------------

def compute_event_stats(
    events_ds: xr.Dataset,
    min_dur: int = MIN_DURATION,
) -> xr.Dataset:
    """Aggregate event metrics over the full time record per grid cell.

    Returns a Dataset with:
      - ``frequency``          — total number of distinct events
      - ``mean_duration``      — mean duration (days)
      - ``max_duration``       — longest event (days)
      - ``mean_intensity``     — mean intensity across all event days (°C)
      - ``max_intensity``      — peak intensity across all events (°C)
      - ``total_mhw_days``     — total days in MHW conditions
      - ``cumulative_intensity`` — sum of daily intensities over all events (°C·days)
      - ``max_category``       — highest category reached (1–4)
      - ``strong_days``        — days in Category II+
      - ``severe_days``        — days in Category III+
      - ``extreme_days``       — days in Category IV
    """
    logger.info("Computing per-grid-cell MHW statistics …")

    mhw = events_ds["mhw_binary"].values
    intensity = events_ds["intensity"].values
    category = events_ds["category"].values
    event_id = events_ds["event_id"].values

    nlat = mhw.shape[1]
    nlon = mhw.shape[2]

    out = {}

    def _per_cell(func):
        result = np.full((nlat, nlon), np.nan)
        for j in range(nlat):
            for k in range(nlon):
                result[j, k] = func(j, k)
        return result

    def _total_mhw_days(j, k):
        return float(mhw[:, j, k].sum())

    def _frequency(j, k):
        eids = event_id[:, j, k]
        return float(len(np.unique(eids[eids > 0])))

    def _mean_duration(j, k):
        eids = event_id[:, j, k]
        ids = np.unique(eids[eids > 0])
        if len(ids) == 0:
            return np.nan
        durations = np.array([(eids == i).sum() for i in ids])
        return float(durations.mean())

    def _max_duration(j, k):
        eids = event_id[:, j, k]
        ids = np.unique(eids[eids > 0])
        if len(ids) == 0:
            return np.nan
        durations = np.array([(eids == i).sum() for i in ids])
        return float(durations.max())

    def _mean_intensity(j, k):
        mask = mhw[:, j, k]
        if not mask.any():
            return np.nan
        return float(intensity[:, j, k][mask].mean())

    def _max_intensity(j, k):
        mask = mhw[:, j, k]
        if not mask.any():
            return np.nan
        return float(intensity[:, j, k][mask].max())

    def _cumulative_intensity(j, k):
        mask = mhw[:, j, k]
        if not mask.any():
            return 0.0
        return float(intensity[:, j, k][mask].sum())

    def _max_category(j, k):
        mask = mhw[:, j, k]
        if not mask.any():
            return 0
        return int(category[:, j, k][mask].max())

    def _count_cat(j, k, cat_min):
        return float((category[:, j, k] >= cat_min).sum())

    out["frequency"] = (("lat", "lon"), _per_cell(_frequency))
    out["mean_duration"] = (("lat", "lon"), _per_cell(_mean_duration))
    out["max_duration"] = (("lat", "lon"), _per_cell(_max_duration))
    out["mean_intensity"] = (("lat", "lon"), _per_cell(_mean_intensity))
    out["max_intensity"] = (("lat", "lon"), _per_cell(_max_intensity))
    out["total_mhw_days"] = (("lat", "lon"), _per_cell(_total_mhw_days))
    out["cumulative_intensity"] = (("lat", "lon"), _per_cell(_cumulative_intensity))
    out["max_category"] = (("lat", "lon"), _per_cell(_max_category))
    out["strong_days"] = (("lat", "lon"), _per_cell(lambda j, k: _count_cat(j, k, 2)))
    out["severe_days"] = (("lat", "lon"), _per_cell(lambda j, k: _count_cat(j, k, 3)))
    out["extreme_days"] = (("lat", "lon"), _per_cell(lambda j, k: _count_cat(j, k, 4)))

    stats_ds = xr.Dataset(
        data_vars=out,
        coords={"lat": events_ds.lat, "lon": events_ds.lon},
        attrs={
            "title": "MHW summary statistics — Hobday et al. (2016)",
            "baseline": f"{BASELINE[0]}-{BASELINE[1]}",
            "min_duration_days": min_dur,
        },
    )

    logger.info("Statistics computed.")
    return stats_ds


# ---------------------------------------------------------------------------
# Step 4 — Publication-quality maps
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _draw_map(
    ax, da, vmin, vmax, title, cbar_label,
    cmap="viridis", extend="both",
    n_levels: int = 64,
):
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    levels = np.linspace(vmin, vmax, n_levels)
    pcm = ax.contourf(
        da.lon, da.lat, da.values,
        levels=levels, cmap=cmap, extend=extend,
        transform=ax.projection,
    )
    ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="none", zorder=2)
    ax.coastlines(linewidth=0.5, zorder=3)
    ax.set_extent([*LON_RANGE, *LAT_RANGE], crs=ax.projection)
    ax.set_title(title, fontsize=12, fontweight="bold")
    cb = plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.06, shrink=0.75)
    cb.set_label(cbar_label, fontsize=10)


def _draw_simple_pcolormesh(ax, da, vmin, vmax, title, cbar_label, cmap="viridis"):
    import cartopy.feature as cfeature

    pcm = ax.pcolormesh(
        da.lon, da.lat, da.values,
        vmin=vmin, vmax=vmax, cmap=cmap, transform=ax.projection,
    )
    ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="none", zorder=2)
    ax.coastlines(linewidth=0.5, zorder=3)
    ax.set_extent([*LON_RANGE, *LAT_RANGE], crs=ax.projection)
    ax.set_title(title, fontsize=12, fontweight="bold")
    cb = ax.figure.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.06, shrink=0.75)
    cb.set_label(cbar_label, fontsize=10)


def _categorical_cmap(n_cat: int):
    """Return a discrete colormap for MHW categories (0–4)."""
    import matplotlib.colors as mcolors

    colors = [
        "#ffffff",  # 0 — no MHW
        "#ffff99",  # 1 — Moderate
        "#ffcc00",  # 2 — Strong
        "#ff6600",  # 3 — Severe
        "#cc0000",  # 4 — Extreme
    ]
    return mcolors.ListedColormap(colors[:n_cat + 1])


def plot_mhw_maps(stats_ds: xr.Dataset, out_dir: Path) -> None:
    """Generate a suite of publication-quality MHW metric maps.

    Produces one figure per metric, saved as a PNG.
    """
    try:
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Plotting dependencies unavailable — skipping maps")
        return

    proj = ccrs.PlateCarree()
    out_dir = _ensure_dir(out_dir)

    metric_cfg = [
        ("frequency", "Event Frequency", "count", "YlOrRd", False),
        ("mean_duration", "Mean Duration", "days", "YlOrRd", False),
        ("max_duration", "Max Duration", "days", "YlOrRd", False),
        ("mean_intensity", "Mean Intensity", "°C", "YlOrRd", False),
        ("max_intensity", "Max Intensity", "°C", "YlOrRd", True),
        ("total_mhw_days", "Total MHW Days", "days", "YlOrRd", False),
        ("cumulative_intensity", "Cumulative Intensity", "°C·days", "YlOrRd", False),
        ("strong_days", "Strong+ MHW Days (Cat II–IV)", "days", "Oranges", False),
        ("severe_days", "Severe+ MHW Days (Cat III–IV)", "days", "Reds", False),
        ("extreme_days", "Extreme MHW Days (Cat IV)", "days", "RdPu", False),
    ]

    for key, title, unit, cmap, symmetric in metric_cfg:
        da = stats_ds[key]
        data_vals = da.values[~np.isnan(da.values)]

        if symmetric:
            vmax = max(abs(data_vals.min()), abs(data_vals.max()))
            vmin = -vmax
        else:
            vmin = 0
            vmax = float(np.percentile(data_vals, 98)) if len(data_vals) > 0 else 1

        fig, ax = plt.subplots(
            figsize=(9, 5.5), subplot_kw={"projection": proj},
        )
        _draw_simple_pcolormesh(
            ax, da, vmin, vmax, f"MHW {title}", unit, cmap=cmap,
        )
        fname = f"mhw_{key}.png"
        fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", out_dir / fname)

    # Category map (discrete)
    fig, ax = plt.subplots(figsize=(9, 5.5), subplot_kw={"projection": proj})
    cat_da = stats_ds["max_category"]
    vmax_cat = int(cat_da.max().values)
    cmap_cat = _categorical_cmap(vmax_cat)
    _draw_simple_pcolormesh(
        ax, cat_da, -0.5, vmax_cat + 0.5,
        "MHW Max Category (Hobday I–IV)", "Category", cmap=cmap_cat,
    )
    cbar = fig.axes[-1]
    cbar.set_ticks(range(vmax_cat + 1))
    cbar.set_ticklabels(["None"] + [f"Cat {i}" for i in range(1, vmax_cat + 1)])
    fig.savefig(out_dir / "mhw_max_category.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_dir / "mhw_max_category.png")

    # Four-panel composite
    _plot_composite_panel(stats_ds, out_dir, proj)


def _plot_composite_panel(stats_ds: xr.Dataset, out_dir: Path, proj) -> None:
    """Create a four-panel summary figure."""
    try:
        import matplotlib.pyplot as plt
        import cartopy.feature as cfeature
    except ImportError:
        return

    metrics = [
        ("frequency", "Frequency", "count", "YlOrRd"),
        ("mean_duration", "Mean Duration", "days", "YlOrRd"),
        ("mean_intensity", "Mean Intensity", "°C", "YlOrRd"),
        ("max_intensity", "Max Intensity", "°C", "YlOrRd"),
    ]

    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(16, 10),
        subplot_kw={"projection": proj},
    )

    for ax, (key, title, unit, cmap) in zip(axes.flat, metrics):
        da = stats_ds[key]
        data_vals = da.values[~np.isnan(da.values)]
        vmin = 0
        vmax = float(np.percentile(data_vals, 98)) if len(data_vals) > 0 else 1

        pcm = ax.pcolormesh(
            da.lon, da.lat, da.values,
            vmin=vmin, vmax=vmax, cmap=cmap, transform=proj,
        )
        ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="none", zorder=2)
        ax.coastlines(linewidth=0.5, zorder=3)
        ax.set_extent([*LON_RANGE, *LAT_RANGE], crs=proj)
        ax.set_title(f"MHW {title}", fontsize=11, fontweight="bold")
        cb = plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.06, shrink=0.7)
        cb.set_label(unit, fontsize=9)

    fig.suptitle("Marine Heatwave Statistics — Indian Ocean", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "mhw_composite_panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_dir / "mhw_composite_panel.png")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_mhw_pipeline(
    sst_path: str,
    baseline: tuple[int, int] = BASELINE,
    min_duration: int = MIN_DURATION,
    max_gap: int = MAX_GAP,
    out_dir_mhw: str = "outputs/mhw",
    out_dir_maps: str = "outputs/mhw",
) -> dict[str, str]:
    """Execute the complete Hobday MHW detection pipeline.

    Parameters
    ----------
    sst_path : str
        Path to daily SST NetCDF.
    baseline : (int, int)
        Baseline years for the climatology.
    min_duration : int
        Minimum consecutive days for an MHW event.
    max_gap : int
        Maximum gap (days) allowed between events to merge.
    out_dir_mhw : str
        Directory for NetCDF outputs.
    out_dir_maps : str
        Directory for map PNGs.

    Returns
    -------
    paths : dict
        ``{"metrics": path, "events": path}`` to the saved files.
    """
    logger.info("=" * 70)
    logger.info("MHW DETECTION — Hobday et al. (2016)")
    logger.info("=" * 70)

    # 1 — Load SST
    logger.info("Loading SST: %s", sst_path)
    ds = xr.open_dataset(sst_path, decode_times=True)
    ds = _standardise_ds(ds)
    ds = _ensure_lon_range(ds)
    ds = subset_domain(ds)
    sst = _sst(ds)

    logger.info("SST shape: time=%d, lat=%d, lon=%d", sst.sizes["time"], sst.sizes["lat"], sst.sizes["lon"])
    logger.info("Time range: %s to %s", str(sst.time.values[0])[:10], str(sst.time.values[-1])[:10])

    # 2 — Climatology & threshold
    clim, thresh = compute_climatology_and_threshold(sst, baseline)

    # 3 — Detect events
    events_ds = detect_events(sst, clim, thresh, min_dur=min_duration, max_gap=max_gap)

    # 4 — Statistics
    stats_ds = compute_event_stats(events_ds, min_dur=min_duration)

    # 5 — Save NetCDFs
    mhw_dir = _ensure_dir(Path(out_dir_mhw))

    metrics_path = mhw_dir / "mhw_metrics.nc"
    stats_ds.to_netcdf(metrics_path)
    logger.info("Saved metrics → %s", metrics_path)

    events_path = mhw_dir / "mhw_events_raw.nc"
    events_ds.to_netcdf(events_path)
    logger.info("Saved events → %s", events_path)

    # 6 — Maps
    maps_dir = _ensure_dir(Path(out_dir_maps))
    plot_mhw_maps(stats_ds, maps_dir)

    logger.info("=" * 70)
    logger.info("MHW PIPELINE COMPLETE")
    logger.info("=" * 70)

    return {
        "metrics": str(metrics_path),
        "events": str(events_path),
    }


def main() -> None:
    cfg = load_config()
    setup_logging(cfg)

    p = cfg.get("paths", {})
    mhw_cfg = cfg.get("mhw", {})

    sst_path = p.get("sst_for_mhw", "data/processed/noaa_oisst.nc")
    baseline = tuple(mhw_cfg.get("baseline_years", BASELINE))
    min_duration = mhw_cfg.get("min_duration", MIN_DURATION)
    max_gap = mhw_cfg.get("max_gap", MAX_GAP)

    out_mhw = p.get("mhw", "outputs/mhw")
    out_maps = p.get("mhw_maps", "outputs/mhw")

    run_mhw_pipeline(
        sst_path,
        baseline=baseline,
        min_duration=min_duration,
        max_gap=max_gap,
        out_dir_mhw=out_mhw,
        out_dir_maps=out_maps,
    )


if __name__ == "__main__":
    main()
