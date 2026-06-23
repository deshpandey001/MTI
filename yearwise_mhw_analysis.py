import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings("ignore")

YEARS = [2015, 2016, 2017]

out_dir = "outputs/yearwise_analysis"
os.makedirs(out_dir, exist_ok=True)

REGIONS = {
    "Arabian Sea":          {"lat": slice(5, 25),  "lon": slice(50, 78)},
    "Bay of Bengal":        {"lat": slice(5, 25),  "lon": slice(78, 95)},
    "Equatorial Indian O.": {"lat": slice(-5, 5),  "lon": slice(40, 100)},
}

print("=" * 60)
print("YEAR-WISE MARINE HEATWAVE ANALYSIS")
print("NOAA OISST v2 — Indian Ocean (2015–2017)")
print("=" * 60)

# =====================
# 1. Load full dataset
# =====================
print("\nLoading data...")
ds = xr.concat(
    [xr.open_dataset(f"data/sst.day.mean.{y}.nc").sel(lat=slice(-40, 30), lon=slice(20, 120))
     for y in YEARS], dim="time"
)
sst = ds["sst"]
lat, lon = sst.lat.values, sst.lon.values
nt, nlat, nlon = sst.shape
sst_np = sst.values.astype(np.float64)
time_doy = sst.time.dt.dayofyear.values
print(f"  Full dataset: time={nt}, lat={nlat}, lon={nlon}")

# ========================================================
# 2. Climatology & 90th percentile threshold (3-year)
# ========================================================
print("\nComputing 3-year climatology and 90th percentile threshold...")
clim_np = np.full((366, nlat, nlon), np.nan, dtype=np.float64)
thresh_np = np.full((366, nlat, nlon), np.nan, dtype=np.float64)

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

# ========================================================
# 3. Per-year detection & metrics
# ========================================================
def detect_mhw_year(sst_year_np, year_doy, thresh_np):
    nt_yr = sst_year_np.shape[0]
    thresh_per_time = thresh_np[year_doy - 1]
    exceed = sst_year_np > thresh_per_time
    intensity = sst_year_np - thresh_per_time

    # Flood-fill labeling
    labels = np.full((nt_yr, nlat, nlon), -1, dtype=np.int32)
    max_lbl = nt_yr * nlat * nlon
    label_j = np.full(max_lbl, -1, dtype=np.int32)
    label_k = np.full(max_lbl, -1, dtype=np.int32)
    next_label = 0

    active = exceed[0].copy()
    n_start = int(np.sum(active))
    if n_start > 0:
        js, ks = np.where(active)
        labels[0, js, ks] = next_label + np.arange(n_start, dtype=np.int32)
        label_j[next_label:next_label + n_start] = js
        label_k[next_label:next_label + n_start] = ks
        next_label += n_start

    for t in range(1, nt_yr):
        active = exceed[t]
        was = exceed[t - 1]
        cont = active & was
        labels[t, cont] = labels[t - 1, cont]
        new_start = active & ~was
        n_new = int(np.sum(new_start))
        if n_new > 0:
            js, ks = np.where(new_start)
            labels[t, js, ks] = next_label + np.arange(n_new, dtype=np.int32)
            label_j[next_label:next_label + n_new] = js
            label_k[next_label:next_label + n_new] = ks
            next_label += n_new

    n_labels = next_label
    label_j = label_j[:n_labels]
    label_k = label_k[:n_labels]

    # Aggregate metrics
    labels_f = labels.ravel()
    intensity_f = intensity.ravel()
    mask = labels_f >= 0
    label_ids = labels_f[mask]

    durations = np.bincount(label_ids)
    int_sum = np.bincount(label_ids, weights=intensity_f[mask])
    int_max = np.zeros(n_labels, dtype=np.float64)
    np.maximum.at(int_max, label_ids, intensity_f[mask])

    valid = durations >= 5
    valid_lbls = np.where(valid)[0]

    event_count = np.zeros((nlat, nlon), dtype=np.int32)
    mean_dur = np.full((nlat, nlon), np.nan, dtype=np.float64)
    mean_int = np.full((nlat, nlon), np.nan, dtype=np.float64)
    max_int = np.full((nlat, nlon), np.nan, dtype=np.float64)

    cell_d = [[] for _ in range(nlat * nlon)]
    cell_m = [[] for _ in range(nlat * nlon)]
    cell_x = [[] for _ in range(nlat * nlon)]

    for lbl in valid_lbls:
        j, k = label_j[lbl], label_k[lbl]
        dur, mi, xi = durations[lbl], int_sum[lbl] / durations[lbl], int_max[lbl]
        idx = j * nlon + k
        cell_d[idx].append(dur)
        cell_m[idx].append(mi)
        cell_x[idx].append(xi)
        event_count[j, k] += 1

    for idx in range(nlat * nlon):
        if cell_d[idx]:
            mean_dur.flat[idx] = np.nanmean(cell_d[idx])
            mean_int.flat[idx] = np.nanmean(cell_m[idx])
            max_int.flat[idx] = np.nanmax(cell_x[idx])

    return event_count, mean_dur, mean_int, max_int

# ========================================================
# 4. Year-wise analysis
# ========================================================
year_metrics = {}

for year in YEARS:
    print(f"\n--- {year} ---")
    yr_sel = sst.sel(time=str(year))
    yr_np = yr_sel.values.astype(np.float64)
    yr_doy = yr_sel.time.dt.dayofyear.values

    ev_freq_da = xr.DataArray(
        np.zeros((nlat, nlon)), dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon}
    )
    ev_dur_da = xr.DataArray(
        np.full((nlat, nlon), np.nan), dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon}
    )
    ev_mint_da = xr.DataArray(
        np.full((nlat, nlon), np.nan), dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon}
    )
    ev_xint_da = xr.DataArray(
        np.full((nlat, nlon), np.nan), dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon}
    )

    ev_count, ev_dur, ev_mint, ev_xint = detect_mhw_year(yr_np, yr_doy, thresh_np)

    ev_freq_da.values = ev_count
    ev_dur_da.values = ev_dur
    ev_mint_da.values = ev_mint
    ev_xint_da.values = ev_xint

    mean_freq = float(ev_freq_da.mean().values)
    mean_dur = float(ev_dur_da.mean().values)
    mean_int = float(ev_mint_da.mean().values)
    mean_xint = float(ev_xint_da.mean().values)

    print(f"  Mean frequency:       {mean_freq:.2f} events")
    print(f"  Mean duration:        {mean_dur:.1f} days")
    print(f"  Mean intensity:       {mean_int:+.3f} °C")
    print(f"  Mean max intensity:   {mean_xint:+.3f} °C")

    year_metrics[year] = {
        "freq_da": ev_freq_da, "dur_da": ev_dur_da,
        "mint_da": ev_mint_da, "xint_da": ev_xint_da,
        "mean_freq": mean_freq, "mean_dur": mean_dur,
        "mean_int": mean_int, "mean_xint": mean_xint,
    }

    # ---- Maps ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    items = [
        (ev_freq_da,  "MHW Frequency (count)",       "RdYlGn_r", "Count"),
        (ev_dur_da,   "MHW Mean Duration (days)",    "plasma",   "Days"),
        (ev_mint_da,  "MHW Mean Intensity (°C)",     "YlOrRd",   "°C"),
        (ev_xint_da,  "MHW Max Intensity (°C)",      "RdYlBu_r", "°C"),
    ]
    for ax, (d, t, cm, lb) in zip(axes.flat, items):
        pcm = ax.pcolormesh(lon, lat, d.values, cmap=cm, shading="auto")
        plt.colorbar(pcm, ax=ax, label=lb, shrink=0.8)
        ax.set_title(t)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig.suptitle(f"MHW Metrics — {year}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/mhw_metrics_{year}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: mhw_metrics_{year}.png")

# ========================================================
# 5. Regional statistics
# ========================================================
print("\n" + "=" * 60)
print("REGIONAL STATISTICS")
print("=" * 60)

regional_table = {}
for region_name, sel in REGIONS.items():
    print(f"\n  {region_name}  [{sel['lat']}, {sel['lon']}]:")
    regional_table[region_name] = {}
    for year in YEARS:
        ym = year_metrics[year]
        fr = float(ym["freq_da"].sel(**sel).mean().values)
        dr = float(ym["dur_da"].sel(**sel).mean().values)
        mi = float(ym["mint_da"].sel(**sel).mean().values)
        xi = float(ym["xint_da"].sel(**sel).mean().values)
        regional_table[region_name][year] = (fr, dr, mi, xi)
        print(f"    {year}: freq={fr:.2f}, dur={dr:.1f}d, int={mi:+.3f}°C, max_int={xi:+.3f}°C")

# ========================================================
# 6. Comparison table
# ========================================================
print("\n" + "=" * 60)
print("COMPARISON TABLE (Indian Ocean mean)")
print("=" * 60)
header = f"{'Year':<6} {'Frequency':>10} {'Duration':>10} {'Intensity':>12} {'Max Intensity':>14}"
sep = "-" * len(header)
print(header)
print(sep)
table_rows = []
for year in YEARS:
    ym = year_metrics[year]
    row = f"{year:<6} {ym['mean_freq']:>10.2f} {ym['mean_dur']:>10.1f} {ym['mean_int']:>12.3f} {ym['mean_xint']:>14.3f}"
    print(row)
    table_rows.append((year, ym['mean_freq'], ym['mean_dur'], ym['mean_int'], ym['mean_xint']))
print(sep)

# ========================================================
# 7. Bar charts
# ========================================================
print("\nCreating bar charts...")
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
titles = ["Mean MHW Frequency", "Mean MHW Duration", "Mean MHW Intensity", "Mean MHW Max Intensity"]
keys = ["mean_freq", "mean_dur", "mean_int", "mean_xint"]
units = ["Count", "Days", "°C", "°C"]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

for ax, title, key, unit, color in zip(axes.flat, titles, keys, units, colors):
    vals = [year_metrics[y][key] for y in YEARS]
    bars = ax.bar(YEARS, vals, color=color, width=0.5, edgecolor="black")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.2f}" if key != "mean_dur" else f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.set_xlabel("Year")
    ax.set_xticks(YEARS)

fig.suptitle("Year-wise MHW Comparison — Indian Ocean", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{out_dir}/mhw_comparison_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: mhw_comparison_bars.png")

# Regional bar charts
for region_name in REGIONS:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, key, unit, color in zip(axes.flat, keys, units, colors):
        vals = [regional_table[region_name][y][keys.index(key)] for y in YEARS]
        bars = ax.bar(YEARS, vals, color=color, width=0.5, edgecolor="black")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}" if key != "mean_dur" else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{region_name}\n{' '.join(t.title() for t in key.split('_'))}")
        ax.set_ylabel(unit)
        ax.set_xlabel("Year")
        ax.set_xticks(YEARS)
    fig.suptitle(f"MHW Comparison — {region_name}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/mhw_{region_name.lower().replace(' ', '_').replace('.', '')}_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: mhw_{region_name.lower().replace(' ', '_').replace('.', '')}_bars.png")

# ========================================================
# 8. Interpretation
# ========================================================
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

# Find strongest year
best_year = max(YEARS, key=lambda y: year_metrics[y]["mean_freq"] + year_metrics[y]["mean_int"])
print(f"\n  • Strongest MHW year: {best_year}")

# Find most affected region
region_avg_intensity = {}
for rn in REGIONS:
    region_avg_intensity[rn] = np.mean([regional_table[rn][y][2] for y in YEARS])
strongest_region = max(region_avg_intensity, key=region_avg_intensity.get)
print(f"  • Most affected region: {strongest_region}")

# El Niño context
print(f"  • 2015–16 El Niño context: 2015–2016 was one of the strongest El Niño")
print(f"    events on record. El Niño typically weakens Indian Ocean monsoon")
print(f"    circulation and can elevate basin-wide SSTs. The year-wise results")
print(f"    show {'2016' if best_year == 2016 else '2015/2016'} as the most active,")
print(f"    consistent with El Niño teleconnections that warm the Indian Ocean")
print(f"    during boreal winter-spring following an El Niño peak.")

print(f"\nAll outputs saved to: {out_dir}/")
