import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. Load dataset (Indian Ocean subset)
# =============================================================================
print("Loading datasets and subsetting to Indian Ocean...")
files = [
    "data/sst.day.mean.2015.nc",
    "data/sst.day.mean.2016.nc",
    "data/sst.day.mean.2017.nc",
]

datasets = []
for f in files:
    ds = xr.open_dataset(f, chunks={"time": -1})
    io = ds.sel(lat=slice(-40, 30), lon=slice(20, 120))
    datasets.append(io)
    ds.close()

ds = xr.concat(datasets, dim="time")
# Rechunk so each timestep is a single chunk, aligned with stored chunk layout
ds = ds.chunk({"time": 1})
print(f"Combined dataset (Indian Ocean): {ds.sizes}")

indian_ocean = ds
sst = indian_ocean["sst"]

# =============================================================================
# 2. Compute daily climatology (group by day-of-year)
# =============================================================================
print("\nComputing daily climatology (2015-2017)...")
climatology = sst.groupby("time.dayofyear").mean(dim="time")
climatology = climatology.chunk({"lat": -1, "lon": -1})
print(f"Climatology shape: {climatology.sizes}")

# =============================================================================
# 3. Compute SST anomalies
# =============================================================================
print("Computing SST anomalies...")
anomalies = sst.groupby("time.dayofyear") - climatology
anomalies = anomalies.rename("sst_anomaly")

# =============================================================================
# 4. Anomaly maps for selected months
# =============================================================================
print("\nCreating anomaly maps for selected months...")
os.makedirs("outputs", exist_ok=True)

monthly_anomalies = anomalies.resample(time="1ME").mean(dim="time")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
selections = [
    ("2015-01", "January 2015"),
    ("2016-07", "July 2016"),
    ("2017-12", "December 2017"),
]

for ax, (time_slice, title) in zip(axes, selections):
    anom_map = monthly_anomalies.sel(time=time_slice).squeeze("time")
    vmax = max(abs(float(anom_map.min())), abs(float(anom_map.max())))
    pcm = ax.pcolormesh(
        indian_ocean.lon, indian_ocean.lat, anom_map,
        cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax
    )
    plt.colorbar(pcm, ax=ax, label="SST anomaly (°C)")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

fig.suptitle("Monthly SST Anomalies - Indian Ocean", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/anomaly_maps.png", dpi=150)
plt.close()
print("Saved: outputs/anomaly_maps.png")

# =============================================================================
# 5. Anomaly time series averaged over the Indian Ocean
# =============================================================================
print("Plotting anomaly time series...")
anomaly_ts = anomalies.mean(dim=["lat", "lon"])

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(anomaly_ts.time, anomaly_ts.values, linewidth=0.8, color="crimson")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("Daily SST Anomaly Averaged over Indian Ocean (2015–2017)")
ax.set_xlabel("Time")
ax.set_ylabel("SST anomaly (°C)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/anomaly_timeseries.png", dpi=150)
plt.close()
print("Saved: outputs/anomaly_timeseries.png")

# =============================================================================
# 6. Identify the warmest anomaly periods
# =============================================================================
print("\nIdentifying warmest anomaly periods...")
anomaly_vals = anomaly_ts.values
anomaly_times = anomaly_ts.time.values

top_n = 10
top_idx = np.argsort(anomaly_vals)[-top_n:][::-1]
print(f"\nTop {top_n} warmest anomaly days:")
print(f"{'Date':<20} {'Anomaly (°C)':>15}")
print("-" * 35)
for idx in top_idx:
    print(f"{str(anomaly_times[idx])[:10]:<20} {anomaly_vals[idx]:>15.3f}")

# =============================================================================
# 7. Calculate summary statistics
# =============================================================================
anom_flat = anomaly_ts.values
max_anom = float(np.max(anom_flat))
mean_anom = float(np.mean(anom_flat))
std_anom = float(np.std(anom_flat))
print(f"\nSummary anomaly statistics (Indian Ocean spatial average):")
print(f"  Maximum anomaly:  {max_anom:+.3f} °C")
print(f"  Mean anomaly:     {mean_anom:+.3f} °C")
print(f"  Std deviation:    {std_anom:.3f} °C")

for year in [2015, 2016, 2017]:
    yr_anom = monthly_anomalies.sel(time=str(year))
    yr_mean = yr_anom.mean(dim=["lat", "lon"])
    warmest_idx = int(np.argmax(yr_mean.values))
    warmest_month = yr_mean.time.values[warmest_idx]
    print(f"  Warmest month in {year}: {str(warmest_month)[:7]} "
          f"({float(yr_mean.values[warmest_idx]):+.3f} °C)")

# =============================================================================
# 8. Generate a summary report
# =============================================================================
report = f"""
===============================================================================
                  MARINE HEATWAVE (MHW) ANALYSIS SUMMARY
                  NOAA OISST v2 — Indian Ocean (20°E–120°E, 40°S–30°N)
                  Period: 2015–2017
===============================================================================

CLIMATOLOGY:
  Computed as daily mean SST for each calendar day over 2015–2017.
  Leap year (2016) included; day 366 (Feb 29) uses a single-year value.

SST ANOMALIES:
  Anomaly = SST - daily climatology
  Units: °C

SPATIAL AVERAGE ANOMALY STATISTICS (Indian Ocean):
  Mean anomaly:       {mean_anom:+.3f} °C
  Maximum anomaly:    {max_anom:+.3f} °C
  Standard deviation: {std_anom:.3f} °C

TOP 10 WARMEST ANOMALY DAYS:
"""
for idx in top_idx:
    report += f"    {str(anomaly_times[idx])[:10]}   {anomaly_vals[idx]:+.3f} °C\n"

report += f"""
WARMEST MONTH PER YEAR:
"""
for year in [2015, 2016, 2017]:
    yr_anom = monthly_anomalies.sel(time=str(year))
    yr_mean = yr_anom.mean(dim=["lat", "lon"])
    warmest_idx = int(np.argmax(yr_mean.values))
    warmest_month = yr_mean.time.values[warmest_idx]
    report += f"  {year}: {str(warmest_month)[:7]} ({float(yr_mean.values[warmest_idx]):+.3f} °C)\n"

report += f"""
GENERATED FIGURES:
  1. outputs/mean_sst_map.png         — Mean SST map (from earlier analysis)
  2. outputs/sst_timeseries.png       — SST time series (from earlier analysis)
  3. outputs/sst_differences.png      — Inter-annual SST differences
  4. outputs/anomaly_maps.png         — Anomaly maps for Jan 2015, Jul 2016, Dec 2017
  5. outputs/anomaly_timeseries.png   — Daily anomaly time series
===============================================================================
"""

with open("outputs/summary_report.txt", "w") as f:
    f.write(report)

print("\n" + report)
print("Summary report saved to: outputs/summary_report.txt")
