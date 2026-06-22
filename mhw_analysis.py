import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. Load and combine all three NetCDF files (Indian Ocean subset only)
#    to reduce memory usage, we select the region per-file before combining
# =============================================================================
print("Loading datasets and subsetting to Indian Ocean...")
files = [
    "data/sst.day.mean.2015.nc",
    "data/sst.day.mean.2016.nc",
    "data/sst.day.mean.2017.nc",
]

# Open each file, immediately subset to Indian Ocean, then combine
datasets = []
for f in files:
    ds = xr.open_dataset(f)
    io = ds.sel(lat=slice(-40, 30), lon=slice(20, 120))
    datasets.append(io)
    ds.close()

ds = xr.concat(datasets, dim="time")
del datasets  # free intermediate references

print(f"Combined dataset (Indian Ocean only):\n{ds}\n")

# =============================================================================
# 2. Print dataset information
# =============================================================================
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(f"\nDimensions: {dict(ds.sizes)}")
print(f"\nCoordinates: {list(ds.coords)}")
print(f"\nData variables: {list(ds.data_vars)}")
print(f"\nSST variable attributes:")
for key, value in ds["sst"].attrs.items():
    print(f"  {key}: {value}")

missing_count = int(np.isnan(ds["sst"].values).sum())
total_count = ds["sst"].size
print(f"\nMissing values (NaN): {missing_count} out of {total_count} "
      f"({100 * missing_count / total_count:.3f}%)")

print(f"\nTime range: {ds.time.values[0]} to {ds.time.values[-1]}")
print(f"Number of time steps: {ds.sizes['time']}")

# =============================================================================
# 3. Indian Ocean region is already extracted above; store reference
# =============================================================================
indian_ocean = ds

# =============================================================================
# 4. Plotting
# =============================================================================
os.makedirs("outputs", exist_ok=True)

# ---- 4a. Mean SST map for the Indian Ocean ----
print("\nComputing mean SST map...")
sst_mean_map = indian_ocean["sst"].mean(dim="time")

fig, ax = plt.subplots(figsize=(10, 6))
pcm = ax.pcolormesh(
    indian_ocean.lon, indian_ocean.lat, sst_mean_map,
    cmap="RdYlBu_r", shading="auto"
)
cb = plt.colorbar(pcm, ax=ax, label="SST (°C)")
ax.set_title("Mean SST - Indian Ocean (2015–2017)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.savefig("outputs/mean_sst_map.png", dpi=150)
plt.close()
print("Saved: outputs/mean_sst_map.png")

# ---- 4b. SST time series averaged over the Indian Ocean ----
print("Computing SST time series...")
sst_ts = indian_ocean["sst"].mean(dim=["lat", "lon"])

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(sst_ts.time, sst_ts.values, linewidth=0.8, color="steelblue")
ax.set_title("Daily SST Averaged over Indian Ocean (2015–2017)")
ax.set_xlabel("Time")
ax.set_ylabel("SST (°C)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/sst_timeseries.png", dpi=150)
plt.close()
print("Saved: outputs/sst_timeseries.png")

# =============================================================================
# 5. Calculate annual mean SST for each year
# =============================================================================
print("\n" + "=" * 60)
print("ANNUAL MEAN SST - INDIAN OCEAN")
print("=" * 60)
annual_means = {}
for year in [2015, 2016, 2017]:
    annual_mean = indian_ocean["sst"].sel(time=str(year)).mean(dim=["time", "lat", "lon"])
    value = float(annual_mean.values)
    annual_means[year] = value
    print(f"  {year}: {value:.3f} °C")

# =============================================================================
# 6. Compare SST differences between years
# =============================================================================
print("\n" + "=" * 60)
print("SST DIFFERENCES BETWEEN YEARS")
print("=" * 60)

# Compute spatial annual means for difference maps
sst_by_year = {}
for year in [2015, 2016, 2017]:
    sst_by_year[str(year)] = indian_ocean["sst"].sel(time=str(year)).mean(dim="time")

pairs = [("2015", "2016"), ("2015", "2017"), ("2016", "2017")]
titles = ["2016 – 2015", "2017 – 2015", "2017 – 2016"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for idx, ((y1, y2), title) in enumerate(zip(pairs, titles)):
    diff = sst_by_year[y2] - sst_by_year[y1]

    # Row 0: difference maps
    ax = axes[0, idx]
    pcm = ax.pcolormesh(
        indian_ocean.lon, indian_ocean.lat, diff,
        cmap="RdBu_r", shading="auto"
    )
    plt.colorbar(pcm, ax=ax, label="SST diff (°C)")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Row 1: histograms of differences
    ax = axes[1, idx]
    diff_flat = diff.values[~np.isnan(diff.values)]
    ax.hist(diff_flat, bins=80, color="gray", edgecolor="none")
    mean_diff = float(np.mean(diff_flat))
    ax.axvline(mean_diff, color="red", linestyle="--",
               label=f"Mean: {mean_diff:.3f}°C")
    ax.set_title(f"Histogram of {title}")
    ax.set_xlabel("SST difference (°C)")
    ax.set_ylabel("Grid cells")
    ax.legend(fontsize=8)

    # Print numeric summary
    std_diff = float(np.std(diff_flat))
    print(f"  {title}: mean = {mean_diff:.3f} °C, std = {std_diff:.3f} °C")

fig.suptitle("Inter-annual SST Differences - Indian Ocean", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/sst_differences.png", dpi=150)
plt.close()
print("Saved: outputs/sst_differences.png")

print("\nAll outputs saved in the 'outputs/' folder.")
