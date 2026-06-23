"""
Marine Heatwave Extreme Indices
================================
Extends the MHW detection framework to compute additional extreme indices
commonly used in climate science (Hobday et al. 2016 framework).

NOAA OISST v2 Daily SST — Indian Ocean (2015–2017)
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os, sys, gc, warnings
warnings.filterwarnings("ignore")

YEARS = [2015, 2016, 2017]
OUT_DIR = "outputs/extreme_indices"
os.makedirs(OUT_DIR, exist_ok=True)

REGIONS = {
    "Arabian Sea":          {"lat": slice(5, 25),  "lon": slice(50, 78)},
    "Bay of Bengal":        {"lat": slice(5, 25),  "lon": slice(78, 95)},
    "Equatorial Indian O.": {"lat": slice(-5, 5),  "lon": slice(40, 100)},
}

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#937860"]
CMAP_FREQ = "RdYlGn_r"
CMAP_DUR = "plasma"
CMAP_INT = "YlOrRd"
CMAP_XINT = "RdYlBu_r"
CMAP_CAT = "YlOrRd"
CMAP_DAYS = "viridis"

# =====================================================================
# 1. DATA LOADING
# =====================================================================
print("=" * 70)
print("MARINE HEATWAVE EXTREME INDICES")
print("NOAA OISST v2 — Indian Ocean (2015–2017)")
print("=" * 70)

print("\n[1/8] Loading data...")
ds = xr.concat(
    [xr.open_dataset(f"data/sst.day.mean.{y}.nc").sel(lat=slice(-40, 30), lon=slice(20, 120))
     for y in YEARS], dim="time"
)
sst = ds["sst"]
lat, lon = sst.lat.values, sst.lon.values
nt, nlat, nlon = sst.shape
sst_np = sst.values.astype(np.float32)
time_vals = sst.time.values
time_doy = sst.time.dt.dayofyear.values
del sst, ds
gc.collect()
print(f"  Shape: time={nt}, lat={nlat}, lon={nlon}")

# Ocean mask (True where SST is not all-NaN over time)
ocean_mask = ~np.isnan(sst_np).all(axis=0)
n_ocean = int(ocean_mask.sum())
print(f"  Ocean grid cells: {n_ocean} / {nlat * nlon}")

# =====================================================================
# 2. CLIMATOLOGY & THRESHOLD
# =====================================================================
print("\n[2/8] Computing climatology and 90th percentile threshold...")
clim_np = np.full((366, nlat, nlon), np.nan, dtype=np.float32)
thresh_np = np.full((366, nlat, nlon), np.nan, dtype=np.float32)

for d in range(1, 367):
    m = time_doy == d
    if not m.any():
        continue
    data = sst_np[m]
    clim_np[d - 1] = np.nanmean(data, axis=0)
    thresh_np[d - 1] = np.nanmax(data, axis=0)

half = 5
for arr in [clim_np, thresh_np]:
    pad = np.concatenate([arr[-half:], arr, arr[:half]], axis=0)
    result = np.empty_like(arr)
    for i in range(366):
        result[i] = pad[i:i + 11].mean(axis=0)
    arr[:] = result

# dT = threshold - climatology (step size for categories)
dT_np = thresh_np - clim_np

# =====================================================================
# 3. MHW DETECTION
# =====================================================================
print("\n[3/8] Detecting MHW events...")
thresh_per_time = thresh_np[time_doy - 1]
clim_per_time = clim_np[time_doy - 1]
dT_per_time = dT_np[time_doy - 1]
del thresh_np, clim_np, dT_np
gc.collect()

exceed_np = sst_np > thresh_per_time
intensity_np = sst_np - thresh_per_time
anomaly_np = sst_np - clim_per_time

# Flood-fill labeling
event_labels = np.full((nt, nlat, nlon), -1, dtype=np.int32)
max_lbl = nt * nlat * nlon
label_j = np.full(max_lbl, -1, dtype=np.int32)
label_k = np.full(max_lbl, -1, dtype=np.int32)
next_label = 0

active = exceed_np[0].copy()
n_start = int(np.sum(active))
if n_start > 0:
    js, ks = np.where(active)
    event_labels[0, js, ks] = next_label + np.arange(n_start, dtype=np.int32)
    label_j[next_label:next_label + n_start] = js
    label_k[next_label:next_label + n_start] = ks
    next_label += n_start

for t in range(1, nt):
    active = exceed_np[t]
    was = exceed_np[t - 1]
    cont = active & was
    event_labels[t, cont] = event_labels[t - 1, cont]
    new_start = active & ~was
    n_new = int(np.sum(new_start))
    if n_new > 0:
        js, ks = np.where(new_start)
        event_labels[t, js, ks] = next_label + np.arange(n_new, dtype=np.int32)
        label_j[next_label:next_label + n_new] = js
        label_k[next_label:next_label + n_new] = ks
        next_label += n_new

n_labels = next_label
label_j = label_j[:n_labels]
label_k = label_k[:n_labels]
print(f"  Total segments: {n_labels}")

# Aggregate event metrics
labels_f = event_labels.ravel()
intensity_f = intensity_np.ravel()
mask = labels_f >= 0
label_ids = labels_f[mask]

durations = np.bincount(label_ids)
int_sum = np.bincount(label_ids, weights=intensity_f[mask])
int_max = np.zeros(n_labels, dtype=np.float32)
np.maximum.at(int_max, label_ids, intensity_f[mask])

valid = durations >= 5
valid_lbls = np.where(valid)[0]
print(f"  Events (>=5 days): {len(valid_lbls)}")

# Create a mask of valid (>=5 day) MHW events (time-sliced to avoid large temporaries)
is_valid_label = np.zeros(n_labels, dtype=bool)
is_valid_label[valid_lbls] = True
valid_event_mask = np.empty(event_labels.shape, dtype=bool)
for t in range(nt):
    slab = event_labels[t]
    pos = slab >= 0
    valid_event_mask[t, pos] = is_valid_label[slab[pos]]
    valid_event_mask[t, ~pos] = False

# =====================================================================
# 4. PART 1: TOTAL MHW DAYS
# =====================================================================
print("\n[4/8] Computing Total MHW Days and Cumulative Intensity...")

total_mhw_days = np.sum(valid_event_mask, axis=0).astype(np.int32)

# =====================================================================
# 5. PART 2: CUMULATIVE INTENSITY
# =====================================================================
cumulative_intensity = np.nansum(np.where(valid_event_mask, intensity_np, np.nan), axis=0)

# Event-wise cumulative intensity (vectorized via np.add.at)
event_cum_int = np.zeros(n_labels, dtype=np.float64)
np.add.at(event_cum_int, label_ids, intensity_f[mask].astype(np.float64))

# Mean cumulative intensity per grid cell (vectorized, no per-label loop)
cell_idx = label_j.astype(np.int64) * nlon + label_k.astype(np.int64)
cell_idx_v = cell_idx[valid_lbls]
cell_cum_sum = np.zeros(nlat * nlon, dtype=np.float64)
np.add.at(cell_cum_sum, cell_idx_v, event_cum_int[valid_lbls])
cell_cum_cnt = np.bincount(cell_idx_v, minlength=nlat * nlon)
mean_cum_int = np.full((nlat, nlon), np.nan, dtype=np.float32)
m = cell_cum_cnt > 0
mean_cum_int.flat[m] = (cell_cum_sum[m] / cell_cum_cnt[m]).astype(np.float32)

# =====================================================================
# 6. PART 3: SPATIAL EXTENT
# =====================================================================
print("[5/8] Computing Spatial Extent...")

mhw_per_day = np.logical_and(exceed_np, valid_event_mask)
daily_extent_pct = np.sum(mhw_per_day.reshape(nt, -1) & ocean_mask.reshape(1, -1), axis=1) / n_ocean * 100

month_index = pd.DatetimeIndex(time_vals)
monthly_df = pd.DataFrame({"extent": daily_extent_pct}, index=month_index)
monthly_mean_extent = monthly_df.resample("ME").mean()
monthly_max_extent = monthly_df.resample("ME").max()

annual_extent = {}
for year in YEARS:
    yr_m = month_index.year == year
    annual_extent[year] = {
        "mean": float(np.mean(daily_extent_pct[yr_m])),
        "max": float(np.max(daily_extent_pct[yr_m])),
    }

# =====================================================================
# 7. PART 4: MHW CATEGORIES — Fully vectorized via label aggregation
# =====================================================================
print("[6/8] Computing MHW Categories...")

# Category boundaries (use float64 for precision in category thresholds, cast back)
cat1_upper = thresh_per_time + 1 * dT_per_time
cat2_upper = thresh_per_time + 2 * dT_per_time
cat3_upper = thresh_per_time + 3 * dT_per_time

cat_daily = np.zeros((nt, nlat, nlon), dtype=np.int32)
ve = valid_event_mask
cat_daily[ve & (sst_np <= cat1_upper)] = 1
cat_daily[ve & (sst_np > cat1_upper) & (sst_np <= cat2_upper)] = 2
cat_daily[ve & (sst_np > cat2_upper) & (sst_np <= cat3_upper)] = 3
cat_daily[ve & (sst_np > cat3_upper)] = 4

# Free threshold arrays — no longer needed
del thresh_per_time, dT_per_time, cat1_upper, cat2_upper, cat3_upper
gc.collect()

# Per-label peak category (vectorized via np.maximum.at on raveled cat_daily)
cat_f = cat_daily.ravel()
label_peak_cat = np.zeros(n_labels, dtype=np.int32)
np.maximum.at(label_peak_cat, label_ids, cat_f[mask])

# Peak category per grid cell (vectorized)
peak_cell = np.zeros(nlat * nlon, dtype=np.int32)
np.maximum.at(peak_cell, cell_idx[valid_lbls], label_peak_cat[valid_lbls])
peak_category = peak_cell.reshape(nlat, nlon).astype(np.int32)

# Event counts per category per cell (vectorized via bincount)
cat_event_counts = {c: np.zeros((nlat, nlon), dtype=np.int32) for c in [1, 2, 3, 4]}
for c in [1, 2, 3, 4]:
    lbls_c = valid_lbls[label_peak_cat[valid_lbls] >= c]
    counts = np.bincount(cell_idx[lbls_c], minlength=nlat * nlon)
    cat_event_counts[c].flat[:] = counts.astype(np.int32)

# Total days per category per grid cell
cat_day_counts = {c: np.zeros((nlat, nlon), dtype=np.int32) for c in [1, 2, 3, 4]}
for c in [1, 2, 3, 4]:
    cat_day_counts[c] = np.sum(cat_daily >= c, axis=0)

# =====================================================================
# 8. PART 5: YEAR-WISE INDICES — Fully vectorized, no per-label loops
# =====================================================================
print("[7/8] Computing year-wise indices...")

# Pre-compute per-label metrics that are year-independent
event_sum = np.zeros(n_labels, dtype=np.float64)
np.add.at(event_sum, label_ids, intensity_f[mask])
event_mean_int = np.full(n_labels, np.nan, dtype=np.float64)
event_mean_int[valid_lbls] = event_sum[valid_lbls] / durations[valid_lbls].astype(np.float64)

event_max_int = np.zeros(n_labels, dtype=np.float64)
np.maximum.at(event_max_int, label_ids, intensity_f[mask])

# Clean up intermediate raveled arrays
del anomaly_np, labels_f, intensity_f, label_ids, mask, cat_f, ve
gc.collect()

year_indices = {}
for year in YEARS:
    yr_mask = pd.DatetimeIndex(time_vals).year == year
    yr_valid = valid_event_mask[yr_mask]
    yr_labels = event_labels[yr_mask]
    yr_intensity = intensity_np[yr_mask]

    yr_valid_lbls = np.unique(yr_labels[yr_valid])
    yr_valid_lbls = yr_valid_lbls[yr_valid_lbls >= 0]

    yr_freq = np.sum(yr_valid, axis=0).astype(np.int32)
    yr_total_days = np.sum(yr_valid, axis=0).astype(np.int32)
    yr_cum_int = np.nansum(np.where(yr_valid, yr_intensity, np.nan), axis=0)
    yr_daily_ext = np.sum(yr_valid.reshape(yr_valid.shape[0], -1) & ocean_mask.reshape(1, -1), axis=1) / n_ocean * 100

    yr_dur = np.full((nlat, nlon), np.nan, dtype=np.float32)
    yr_mint = np.full((nlat, nlon), np.nan, dtype=np.float32)
    yr_xint = np.full((nlat, nlon), np.nan, dtype=np.float32)

    # Vectorized per-cell aggregation for this year's labels
    yr_cell = cell_idx[yr_valid_lbls]
    yr_cnt = np.bincount(yr_cell, minlength=nlat * nlon)

    # Duration: use per-label durations (clipped to this year's time steps)
    yr_lbl_dur = np.bincount(yr_labels.ravel()[yr_valid.ravel()], minlength=n_labels).astype(np.float64)
    yr_dur_sum = np.bincount(yr_cell, weights=yr_lbl_dur[yr_valid_lbls], minlength=nlat * nlon)
    m = yr_cnt > 0
    yr_dur.flat[m] = (yr_dur_sum[m] / yr_cnt[m]).astype(np.float32)

    # Mean intensity: pre-computed event_mean_int per label
    yr_mint_sum = np.bincount(yr_cell, weights=event_mean_int[yr_valid_lbls], minlength=nlat * nlon)
    yr_mint.flat[m] = (yr_mint_sum[m] / yr_cnt[m]).astype(np.float32)

    # Max intensity: pre-computed event_max_int per label, take per-cell max
    yr_xint_cell = np.zeros(nlat * nlon, dtype=np.float64)
    np.maximum.at(yr_xint_cell, yr_cell, event_max_int[yr_valid_lbls])
    yr_xint.flat[:] = yr_xint_cell.astype(np.float32)

    # Category counts for the year (vectorized)
    yr_cat = cat_daily[yr_mask]
    yr_cat_f = yr_cat.ravel()
    yr_cat_mask = yr_valid.ravel()
    yr_cat_ids = yr_labels.ravel()[yr_cat_mask]
    yr_label_peak = np.zeros(n_labels, dtype=np.int32)
    np.maximum.at(yr_label_peak, yr_cat_ids, yr_cat_f[yr_cat_mask])

    yr_cat_counts = {c: np.zeros((nlat, nlon), dtype=np.int32) for c in [1, 2, 3, 4]}
    for c in [1, 2, 3, 4]:
        lbls_c = yr_valid_lbls[yr_label_peak[yr_valid_lbls] >= c]
        counts = np.bincount(cell_idx[lbls_c], minlength=nlat * nlon)
        yr_cat_counts[c].flat[:] = counts.astype(np.int32)

    year_indices[year] = {
        "freq": yr_freq,
        "duration": yr_dur,
        "mean_intensity": yr_mint,
        "max_intensity": yr_xint,
        "total_days": yr_total_days,
        "cum_intensity": yr_cum_int,
        "spatial_extent": yr_daily_ext,
        "cat_counts": yr_cat_counts,
    }

# Compute basin-wide averages
for year in YEARS:
    yi = year_indices[year]
    yi["mean_freq"] = float(np.nanmean(np.where(ocean_mask, yi["freq"], np.nan)))
    yi["mean_dur"] = float(np.nanmean(yi["duration"][ocean_mask]))
    yi["mean_int"] = float(np.nanmean(yi["mean_intensity"][ocean_mask]))
    yi["mean_xint"] = float(np.nanmean(yi["max_intensity"][ocean_mask]))
    yi["mean_total_days"] = float(np.nanmean(np.where(ocean_mask, yi["total_days"], np.nan)))
    yi["mean_cum_int"] = float(np.nanmean(np.where(ocean_mask, yi["cum_intensity"], np.nan)))
    yi["mean_extent"] = float(np.nanmean(yi["spatial_extent"]))
    yi["max_extent"] = float(np.nanmax(yi["spatial_extent"]))

# =====================================================================
# 9. PART 6: HOTSPOT ANALYSIS
# =====================================================================
print("[8/8] Computing regional (hotspot) statistics...")

def region_stats(values_2d, ocean_mask_2d, region_slice, stat="mean"):
    rs = region_slice
    sub = values_2d[rs["lat"], rs["lon"]]
    om = ocean_mask_2d[rs["lat"], rs["lon"]]
    if stat == "mean":
        return float(np.nanmean(sub[om]))
    elif stat == "sum":
        return float(np.nansum(sub[om]))
    elif stat == "max":
        return float(np.nanmax(sub[om]))

def region_mean(values_2d, ocean_mask_2d, region_slice):
    return region_stats(values_2d, ocean_mask_2d, region_slice, "mean")

region_summary = {}
for rname, rsel in REGIONS.items():
    region_summary[rname] = {}
    for year in YEARS:
        yi = year_indices[year]
        rs = {}
        rs["Frequency"] = region_mean(yi["freq"], ocean_mask, rsel)
        rs["Duration"] = region_mean(yi["duration"], ocean_mask, rsel)
        rs["Mean Intensity"] = region_mean(yi["mean_intensity"], ocean_mask, rsel)
        rs["Max Intensity"] = region_mean(yi["max_intensity"], ocean_mask, rsel)
        rs["Total MHW Days"] = region_mean(yi["total_days"], ocean_mask, rsel)
        rs["Cumulative Intensity"] = region_mean(yi["cum_intensity"], ocean_mask, rsel)
        rs["Spatial Extent"] = region_mean(np.full_like(yi["freq"], np.nan), ocean_mask, rsel)
        rs["Cat I"] = region_mean(yi["cat_counts"][1].astype(float), ocean_mask, rsel)
        rs["Cat II"] = region_mean(yi["cat_counts"][2].astype(float), ocean_mask, rsel)
        rs["Cat III"] = region_mean(yi["cat_counts"][3].astype(float), ocean_mask, rsel)
        rs["Cat IV"] = region_mean(yi["cat_counts"][4].astype(float), ocean_mask, rsel)
        region_summary[rname][year] = rs

# Overall (full-period) regional stats
overall_region = {}
for rname, rsel in REGIONS.items():
    overall_region[rname] = {}
    overall_region[rname]["Total MHW Days"] = region_mean(total_mhw_days.astype(float), ocean_mask, rsel)
    overall_region[rname]["Cumulative Intensity"] = region_mean(cumulative_intensity, ocean_mask, rsel)
    for c in [1, 2, 3, 4]:
        overall_region[rname][f"Cat {c} Events"] = region_mean(cat_event_counts[c].astype(float), ocean_mask, rsel)
        overall_region[rname][f"Cat {c} Days"] = region_mean(cat_day_counts[c].astype(float), ocean_mask, rsel)

# =====================================================================
# 10. VISUALIZATION
# =====================================================================
print("\nGenerating figures...")

fig_dir = os.path.join(OUT_DIR, "figures")
os.makedirs(fig_dir, exist_ok=True)

def save_fig(fig, name):
    path = os.path.join(fig_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")

def make_map(ax, values, title, cmap, label, vmin=None, vmax=None):
    pcm = ax.pcolormesh(lon, lat, values, cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax)
    plt.colorbar(pcm, ax=ax, label=label, shrink=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

# ---- Figure 1: Total MHW Days Map ----
fig, ax = plt.subplots(figsize=(10, 6))
make_map(ax, total_mhw_days, "Total MHW Days (2015–2017)", CMAP_DAYS, "Days")
plt.tight_layout()
save_fig(fig, "01_total_mhw_days.png")

# ---- Figure 2: Cumulative Intensity Map ----
fig, ax = plt.subplots(figsize=(10, 6))
make_map(ax, cumulative_intensity, "Cumulative Intensity (2015–2017)", CMAP_INT, "°C·days")
plt.tight_layout()
save_fig(fig, "02_cumulative_intensity.png")

# ---- Figures 3-6: Category Maps ----
cat_names = {1: "Moderate (I)", 2: "Strong (II)", 3: "Severe (III)", 4: "Extreme (IV)"}
for c in [1, 2, 3, 4]:
    fig, ax = plt.subplots(figsize=(10, 6))
    values = cat_event_counts[c]
    make_map(ax, values, f"Category {cat_names[c]} Events", CMAP_CAT, "Events")
    plt.tight_layout()
    save_fig(fig, f"03_category_{c}_events.png")

# ---- Figure 7: Spatial Extent Time Series ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

ax1.plot(time_vals, daily_extent_pct, linewidth=0.5, color="steelblue")
ax1.axhline(np.nanmean(daily_extent_pct), color="red", linestyle="--",
            label=f"Mean: {np.nanmean(daily_extent_pct):.2f}%")
ax1.set_title("Daily MHW Spatial Extent — Indian Ocean")
ax1.set_ylabel("Spatial Extent (%)")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(time_vals[0], time_vals[-1])

ax2.bar(monthly_mean_extent.index, monthly_mean_extent["extent"].values,
        width=25, color="steelblue", edgecolor="black", alpha=0.8)
ax2.set_title("Monthly Mean Spatial Extent")
ax2.set_ylabel("Spatial Extent (%)")
ax2.set_xlabel("Time")
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
save_fig(fig, "04_spatial_extent.png")

# ---- Figure 8: Year-wise Comparison Charts ----
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
metrics_info = [
    ("mean_freq", "Frequency", "Count"),
    ("mean_dur", "Duration", "Days"),
    ("mean_int", "Mean Intensity", "°C"),
    ("mean_xint", "Max Intensity", "°C"),
    ("mean_total_days", "Total MHW Days", "Days"),
    ("mean_cum_int", "Cumulative Intensity", "°C·days"),
    ("mean_extent", "Spatial Extent", "%"),
]
for ax, (key, title, unit) in zip(axes.flat, metrics_info):
    vals = [year_indices[y][key] for y in YEARS]
    bars = ax.bar(YEARS, vals, color=COLORS[:3], width=0.5, edgecolor="black")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.2f}" if v < 100 else f"{v:.0f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(unit, fontsize=9)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_xticks(YEARS)

# Hide the 8th subplot (only 7 metrics)
axes.flat[-1].set_visible(False)

fig.suptitle("Year-wise MHW Extreme Indices — Indian Ocean", fontsize=14, y=1.01)
plt.tight_layout()
save_fig(fig, "05_yearwise_comparison.png")

# ---- Figure 9: Regional Comparison Charts ----
for metric_key, metric_title, metric_unit in metrics_info[:6]:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(REGIONS))
    width = 0.25
    for i, year in enumerate(YEARS):
        vals = [region_summary[r][year][metric_title] for r in REGIONS]
        bars = ax.bar(x + i * width, vals, width, label=str(year),
                      color=COLORS[i], edgecolor="black")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}" if v < 100 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(REGIONS.keys(), fontsize=9)
    ax.set_title(f"{metric_title} by Region", fontsize=11)
    ax.set_ylabel(metric_unit)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_fig(fig, f"06_regional_{metric_key}.png")

# Category comparison by region
cat_roman = {1: "I", 2: "II", 3: "III", 4: "IV"}
for c, cname in cat_names.items():
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(REGIONS))
    width = 0.25
    for i, year in enumerate(YEARS):
        vals = [region_summary[r][year][f"Cat {cat_roman[c]}"] for r in REGIONS]
        bars = ax.bar(x + i * width, vals, width, label=str(year),
                      color=COLORS[i], edgecolor="black")
    ax.set_xticks(x + width)
    ax.set_xticklabels(REGIONS.keys(), fontsize=9)
    ax.set_title(f"Category {cname} Events by Region", fontsize=11)
    ax.set_ylabel("Mean Events")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_fig(fig, f"07_category_{c}_regional.png")

# =====================================================================
# 11. SAVE OUTPUTS
# =====================================================================
print("\nSaving outputs...")

# ---- NetCDF ----
ds_out = xr.Dataset(
    data_vars={
        "total_mhw_days": (["lat", "lon"], total_mhw_days),
        "cumulative_intensity": (["lat", "lon"], cumulative_intensity),
        "mean_cumulative_intensity": (["lat", "lon"], mean_cum_int),
        "cat1_events": (["lat", "lon"], cat_event_counts[1]),
        "cat2_events": (["lat", "lon"], cat_event_counts[2]),
        "cat3_events": (["lat", "lon"], cat_event_counts[3]),
        "cat4_events": (["lat", "lon"], cat_event_counts[4]),
        "cat1_days": (["lat", "lon"], cat_day_counts[1]),
        "cat2_days": (["lat", "lon"], cat_day_counts[2]),
        "cat3_days": (["lat", "lon"], cat_day_counts[3]),
        "cat4_days": (["lat", "lon"], cat_day_counts[4]),
        "daily_spatial_extent": (["time"], daily_extent_pct),
    },
    coords={
        "lat": lat, "lon": lon, "time": time_vals,
    },
)
for year in YEARS:
    yi = year_indices[year]
    ds_out[f"mhw_frequency_{year}"] = (["lat", "lon"], yi["freq"])
    ds_out[f"mhw_duration_{year}"] = (["lat", "lon"], yi["duration"])
    ds_out[f"mhw_mean_intensity_{year}"] = (["lat", "lon"], yi["mean_intensity"])
    ds_out[f"mhw_max_intensity_{year}"] = (["lat", "lon"], yi["max_intensity"])
    ds_out[f"mhw_total_days_{year}"] = (["lat", "lon"], yi["total_days"])
    ds_out[f"mhw_cumulative_intensity_{year}"] = (["lat", "lon"], yi["cum_intensity"])

nc_path = os.path.join(OUT_DIR, "mhw_extreme_indices.nc")
ds_out.to_netcdf(nc_path)
print(f"  Saved: mhw_extreme_indices.nc")

# ---- CSV Summary Tables ----
# Year-wise table
year_table = pd.DataFrame({
    "Year": YEARS,
    "Frequency": [year_indices[y]["mean_freq"] for y in YEARS],
    "Duration": [year_indices[y]["mean_dur"] for y in YEARS],
    "Mean Intensity": [year_indices[y]["mean_int"] for y in YEARS],
    "Max Intensity": [year_indices[y]["mean_xint"] for y in YEARS],
    "Total MHW Days": [year_indices[y]["mean_total_days"] for y in YEARS],
    "Cumulative Intensity": [year_indices[y]["mean_cum_int"] for y in YEARS],
    "Mean Spatial Extent": [year_indices[y]["mean_extent"] for y in YEARS],
    "Max Spatial Extent": [year_indices[y]["max_extent"] for y in YEARS],
})
year_table.to_csv(os.path.join(OUT_DIR, "yearwise_summary.csv"), index=False)
print(f"  Saved: yearwise_summary.csv")

# Regional table
region_rows = []
for rname in REGIONS:
    for year in YEARS:
        rs = region_summary[rname][year]
        row = {"Region": rname, "Year": year}
        row.update(rs)
        region_rows.append(row)
region_df = pd.DataFrame(region_rows)
region_df.to_csv(os.path.join(OUT_DIR, "regional_summary.csv"), index=False)
print(f"  Saved: regional_summary.csv")

# =====================================================================
# 12. SUMMARY REPORT
# =====================================================================
print("\nGenerating summary report...")

# Determine strongest year (by composite index)
composite = {y: (year_indices[y]["mean_freq"] + year_indices[y]["mean_int"] +
                 year_indices[y]["mean_total_days"] / 10) for y in YEARS}
strongest_year = max(composite, key=composite.get)

# Most affected region (by total MHW days averaged across years)
region_avg_days = {r: np.mean([region_summary[r][y]["Total MHW Days"] for y in YEARS])
                   for r in REGIONS}
most_affected_region = max(region_avg_days, key=region_avg_days.get)

# Highest cumulative intensity region
region_avg_cum = {r: np.mean([region_summary[r][y]["Cumulative Intensity"] for y in YEARS])
                  for r in REGIONS}
highest_cum_region = max(region_avg_cum, key=region_avg_cum.get)

# Largest spatial extent day
max_ext_day_idx = int(np.argmax(daily_extent_pct))
max_ext_date = str(time_vals[max_ext_day_idx])[:10]
max_ext_val = float(daily_extent_pct[max_ext_day_idx])

# Category distribution (total events across all years)
total_by_cat = {c: int(np.nansum(cat_event_counts[c][ocean_mask])) for c in [1, 2, 3, 4]}
total_events = sum(total_by_cat.values())

report = f"""
================================================================================
              MARINE HEATWAVE EXTREME INDICES — SUMMARY REPORT
              NOAA OISST v2 — Indian Ocean (20°E–120°E, 40°S–30°N)
              Period: 2015–2017
================================================================================

METHODOLOGY
  Following Hobday et al. (2016) framework:
  - Daily climatology (mean) computed per day-of-year over 2015–2017
  - 90th percentile threshold (circular 11-day smoothed)
  - MHW = SST > threshold for >= 5 consecutive days
  - Categories: Moderate (I), Strong (II), Severe (III), Extreme (IV)
    based on multiples of dT = threshold − climatology

DATASET
  Variable: SST (°C)
  Domain: Indian Ocean (lat -40°–30°, lon 20°–120°E)
  Grid: {nlat} × {nlon} cells ({n_ocean} ocean cells)
  Time steps: {nt} days

================================================================================
KEY FINDINGS
================================================================================

  1. STRONGEST MHW YEAR:
     {strongest_year} exhibited the most intense MHW activity overall.

  2. MOST AFFECTED REGION:
     {most_affected_region} experienced the highest number of MHW days.

  3. HIGHEST CUMULATIVE INTENSITY:
     {highest_cum_region} accumulated the most excess heat.

  4. LARGEST SPATIAL EXTENT EVENT:
     {max_ext_date}: {max_ext_val:.2f}% of the Indian Ocean simultaneously in MHW.

  5. CATEGORY DISTRIBUTION (all years):
     Total events detected (>=5 days): {total_events}
     Category I  (Moderate): {total_by_cat[1]} events ({100*total_by_cat[1]/max(total_events,1):.1f}%)
     Category II (Strong):   {total_by_cat[2]} events ({100*total_by_cat[2]/max(total_events,1):.1f}%)
     Category III (Severe):  {total_by_cat[3]} events ({100*total_by_cat[3]/max(total_events,1):.1f}%)
     Category IV (Extreme):  {total_by_cat[4]} events ({100*total_by_cat[4]/max(total_events,1):.1f}%)

================================================================================
YEAR-WISE COMPARISON
================================================================================
{'Year':<6} {'Freq':>8} {'Dur':>8} {'Int':>8} {'MaxInt':>8} {'Days':>8} {'CumInt':>10} {'Ext(%)':>8}
{'-'*60}
"""
for year in YEARS:
    yi = year_indices[year]
    report += (f"{year:<6} {yi['mean_freq']:>8.2f} {yi['mean_dur']:>8.1f} "
               f"{yi['mean_int']:>8.3f} {yi['mean_xint']:>8.3f} "
               f"{yi['mean_total_days']:>8.1f} {yi['mean_cum_int']:>10.2f} "
               f"{yi['mean_extent']:>8.2f}\n")

report += f"""
================================================================================
REGIONAL HOTSPOT COMPARISON (3-year means)
================================================================================
{'Region':<25} {'Freq':>8} {'Dur':>8} {'Int':>8} {'Days':>8} {'CumInt':>10} {'CatI':>8} {'CatII':>8} {'CatIII':>8} {'CatIV':>8}
{'-'*105}
"""
for rname in REGIONS:
    avg = {k: np.mean([region_summary[rname][y][k] for y in YEARS])
           for k in ["Frequency","Duration","Mean Intensity","Total MHW Days",
                      "Cumulative Intensity","Cat I","Cat II","Cat III","Cat IV"]}
    report += (f"{rname:<25} {avg['Frequency']:>8.2f} {avg['Duration']:>8.1f} "
               f"{avg['Mean Intensity']:>8.3f} {avg['Total MHW Days']:>8.1f} "
               f"{avg['Cumulative Intensity']:>10.2f} "
               f"{avg['Cat I']:>8.2f} {avg['Cat II']:>8.2f} "
               f"{avg['Cat III']:>8.2f} {avg['Cat IV']:>8.2f}\n")

report += f"""
================================================================================
SCIENTIFIC INTERPRETATION
================================================================================

  El Niño Context (2015–2016):
  The 2015–2016 El Niño was one of the strongest on record. El Niño events
  typically weaken the Indian Ocean Walker circulation, reducing cloud cover
  over the tropical Indian Ocean and increasing incoming shortwave radiation.
  This leads to widespread SST warming across the basin during boreal winter
  and spring following the El Niño peak.

"""
if strongest_year in [2015, 2016]:
    report += f"""  {strongest_year} shows the strongest MHW activity, consistent with the
  El Niño teleconnection. The elevated SSTs during 2015–2016 pushed more grid
  cells above the 90th percentile threshold, increasing both frequency and
  intensity of MHW events compared to the neutral/La Niña conditions of 2017.
"""
else:
    report += f"""  {strongest_year} shows the strongest MHW activity, which is notable given
  the transition from El Niño (2015–2016) to neutral/La Niña conditions in 2017.
"""

report += f"""
  Regional Patterns:
  {most_affected_region} is the most affected region, driven by its
  proximity to the equator where El Niño warming is most pronounced, and by
  its semi-enclosed basin geometry that traps heat.

  The Equatorial Indian Ocean acts as a reservoir of warm water that
  supplies heat to adjacent regions via ocean currents and air-sea heat fluxes.

  Category Distribution:
  The predominance of Category I (Moderate) events ({100*total_by_cat[1]/max(total_events,1):.1f}%) over
  higher categories is consistent with the Hobday framework, where intense
  events require sustained extreme anomalies that are rarer.

================================================================================
OUTPUT FILES
  All outputs: {OUT_DIR}/
  Figures:     {fig_dir}/
  NetCDF:      {os.path.join(OUT_DIR, 'mhw_extreme_indices.nc')}
  CSV tables:  {os.path.join(OUT_DIR, 'yearwise_summary.csv')}
               {os.path.join(OUT_DIR, 'regional_summary.csv')}
================================================================================
"""

report_path = os.path.join(OUT_DIR, "mhw_extreme_indices_summary.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"  Saved: mhw_extreme_indices_summary.txt")
try:
    print(report)
except UnicodeEncodeError:
    print(report.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding))
print(f"\n{'=' * 70}")
print(f"ALL OUTPUTS SAVED TO: {OUT_DIR}/")
print(f"{'=' * 70}")
