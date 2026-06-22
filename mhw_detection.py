import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. Load data (Indian Ocean subset)
# =============================================================================
print("Loading data and subsetting to Indian Ocean...")
ds = xr.concat(
    [xr.open_dataset(f"data/sst.day.mean.{y}.nc").sel(lat=slice(-40, 30), lon=slice(20, 120))
     for y in [2015, 2016, 2017]], dim="time"
)
sst = ds["sst"]
lat, lon = sst.lat.values, sst.lon.values
nt, nlat, nlon = sst.shape
sst_np = sst.values.astype(np.float64)
time_doy = sst.time.dt.dayofyear.values
print(f"  Shape: time={nt}, lat={nlat}, lon={nlon}")

# =============================================================================
# 2. Climatology (mean) & 90th percentile threshold
#    Use groupby + circular 11-day rolling smooth for computational efficiency
# =============================================================================
print("Computing climatology and 90th percentile threshold...")
clim_np = np.full((366, nlat, nlon), np.nan, dtype=np.float64)
thresh_np = np.full((366, nlat, nlon), np.nan, dtype=np.float64)

for d in range(1, 367):
    m = time_doy == d
    if not m.any():
        continue
    data = sst_np[m]
    clim_np[d - 1] = np.nanmean(data, axis=0)
    # For n <= 3 values, 90th pct ≈ maximum (fast approximation)
    thresh_np[d - 1] = np.nanmax(data, axis=0)

# Apply 11-day circular rolling smooth
half = 5
for arr in [clim_np, thresh_np]:
    pad = np.concatenate([arr[-half:], arr, arr[:half]], axis=0)
    result = np.empty_like(arr)
    for i in range(366):
        result[i] = pad[i:i + 11].mean(axis=0)
    arr[:] = result

# =============================================================================
# 3. Identify exceedances (SST > threshold)
# =============================================================================
print("Identifying exceedances...")
thresh_per_time = thresh_np[time_doy - 1]
exceed_np = sst_np > thresh_per_time
sst_intensity = sst_np - thresh_per_time  # positive when exceeding threshold

# =============================================================================
# 4. Event labeling via flood-fill (vectorized per time step)
# =============================================================================
print("Labeling events (flood-fill)...")
event_labels = np.full((nt, nlat, nlon), -1, dtype=np.int32)
max_lbl = nt * nlat * nlon + 1
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
print(f"  Total event segments: {n_labels}")

# =============================================================================
# 5. Compute event metrics with grouped aggregation
# =============================================================================
print("Computing event metrics...")
labels_f = event_labels.ravel()
intensity_f = sst_intensity.ravel()
mask = labels_f >= 0
label_ids = labels_f[mask]

durations = np.bincount(label_ids)
int_sum = np.bincount(label_ids, weights=intensity_f[mask])
int_max = np.zeros(n_labels, dtype=np.float64)
np.maximum.at(int_max, label_ids, intensity_f[mask])

valid = durations >= 5
valid_lbls = np.where(valid)[0]
print(f"  Events (>=5 days): {len(valid_lbls)}")

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
    cell_d[idx].append(dur); cell_m[idx].append(mi); cell_x[idx].append(xi)
    event_count[j, k] += 1

for idx in range(nlat * nlon):
    if cell_d[idx]:
        mean_dur.flat[idx] = np.nanmean(cell_d[idx])
        mean_int.flat[idx] = np.nanmean(cell_m[idx])
        max_int.flat[idx] = np.nanmax(cell_x[idx])

# Package into xarray
ev_freq = xr.DataArray(event_count, dims=["lat", "lon"],
                        coords={"lat": lat, "lon": lon}, name="mhw_frequency")
ev_dur = xr.DataArray(mean_dur, dims=["lat", "lon"],
                       coords={"lat": lat, "lon": lon}, name="mhw_mean_duration")
ev_mint = xr.DataArray(mean_int, dims=["lat", "lon"],
                        coords={"lat": lat, "lon": lon}, name="mhw_mean_intensity")
ev_xint = xr.DataArray(max_int, dims=["lat", "lon"],
                        coords={"lat": lat, "lon": lon}, name="mhw_max_intensity")
clim_da = xr.DataArray(clim_np, dims=["dayofyear", "lat", "lon"],
                        coords={"dayofyear": np.arange(1, 367), "lat": lat, "lon": lon})
thr_da = xr.DataArray(thresh_np, dims=["dayofyear", "lat", "lon"],
                       coords={"dayofyear": np.arange(1, 367), "lat": lat, "lon": lon})

# =============================================================================
# 6. Maps
# =============================================================================
os.makedirs("outputs", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
items = [
    (ev_freq,  "MHW Frequency (count)",       "RdYlGn_r", "Count"),
    (ev_dur,   "MHW Mean Duration (days)",    "plasma",   "Days"),
    (ev_mint,  "MHW Mean Intensity (°C)",     "YlOrRd",   "°C"),
    (ev_xint,  "MHW Max Intensity (°C)",      "RdYlBu_r", "°C"),
]
for ax, (d, t, cm, lb) in zip(axes.flat, items):
    pcm = ax.pcolormesh(lon, lat, d.values, cmap=cm, shading="auto")
    plt.colorbar(pcm, ax=ax, label=lb, shrink=0.8)
    ax.set_title(t); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
plt.tight_layout()
plt.savefig("outputs/mhw_metrics_maps.png", dpi=150)
plt.close()
print("Saved: outputs/mhw_metrics_maps.png")

# =============================================================================
# 7. Save NetCDF
# =============================================================================
xr.Dataset({
    "mhw_frequency": ev_freq, "mhw_mean_duration": ev_dur,
    "mhw_mean_intensity": ev_mint, "mhw_max_intensity": ev_xint,
    "climatology": clim_da, "threshold_90pct": thr_da,
}).to_netcdf("outputs/mhw_metrics.nc")
print("Saved: outputs/mhw_metrics.nc")

# =============================================================================
# 8. Hotspot regions
# =============================================================================
print("\n" + "=" * 60)
print("HOTSPOT REGION ANALYSIS")
print("=" * 60)
regions = {
    "Arabian Sea":          {"lat": slice(5, 25),  "lon": slice(50, 78)},
    "Bay of Bengal":        {"lat": slice(5, 25),  "lon": slice(78, 95)},
    "Equatorial Indian O.": {"lat": slice(-5, 5),  "lon": slice(40, 100)},
}

for name, sel in regions.items():
    fr = float(ev_freq.sel(**sel).mean().values)
    dr = float(ev_dur.sel(**sel).mean().values)
    mi = float(ev_mint.sel(**sel).mean().values)
    xi = float(ev_xint.sel(**sel).mean().values)
    print(f"\n  {name}  [{sel['lat']}, {sel['lon']}]:")
    print(f"    Mean frequency:     {fr:.2f} events")
    print(f"    Mean duration:      {dr:.1f} days")
    print(f"    Mean intensity:     {mi:+.3f} °C")
    print(f"    Mean max intensity: {xi:+.3f} °C")

# Regional frequency map
fig, axes2 = plt.subplots(2, 2, figsize=(12, 8))
for idx, (name, sel) in enumerate(regions.items()):
    ax = plt.subplot(2, 2, idx + 1)
    fr = ev_freq.sel(**sel)
    pcm = ax.pcolormesh(fr.lon, fr.lat, fr.values, cmap="RdYlGn_r", shading="auto")
    plt.colorbar(pcm, ax=ax, label="Events", shrink=0.7)
    ax.set_title(name); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
plt.suptitle("MHW Frequency by Region", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/mhw_regional_frequency.png", dpi=150)
plt.close()
print("Saved: outputs/mhw_regional_frequency.png")

# =============================================================================
# 9. Summary
# =============================================================================
print("\n" + "=" * 60)
print("OVERALL MHW SUMMARY (Indian Ocean, 2015-2017)")
print("=" * 60)
print(f"  Total events (>=5 days):           {int(np.sum(ev_freq.values))}")
print(f"  Mean frequency per grid cell:      {float(ev_freq.mean().values):.2f}")
print(f"  Mean event duration:               {float(ev_dur.mean().values):.1f} days")
print(f"  Mean intensity (grid-cell avg):    {float(ev_mint.mean().values):.3f} °C")
print(f"  Mean max intensity:                {float(ev_xint.mean().values):.3f} °C")
print("\nAll outputs saved in 'outputs/' folder.")
