# Preprocessing Pipeline — Design Decisions

## 1. Coordinate Standardization

**What we did:** Map all coordinate names (`latitude`, `longitude`, `nav_lat`, `nav_lon`) → canonical `lat`/`lon`.

**Why this method:** CMIP6 models from different centers name coordinates differently:
- GFDL uses `lat`/`lon` (1D)
- IPSL uses `nav_lat`/`nav_lon` (2D)
- EC-Earth3 uses `latitude`/`longitude` (2D)

A simple rename dictionary is the most straightforward approach. Xarray's `.rename()` handles both 1D and 2D coordinates identically — it just renames the variable without touching the data or dimensions.

**Alternatives & tradeoffs:**
- **Don't rename, use the original names everywhere** — would require every downstream function to check for all possible names. Extremely fragile, lots of `if/elif` chains.
- **Use xarray's `swap_dims()`** — only works when you want to replace dimension names, not aux coordinate names. Not applicable here since `nav_lat` is an aux coord with dims `(y, x)`, not a dimension itself.
- **Don't drop the original coords** — leaving `latitude`/`nav_lat` alongside renamed `lat` causes `sel(lat=...)` to fail with "multiple values" during spatial subsetting.

---

## 2. Longitude Conversion (0–360 → -180..180)

**What we did:** `((lon + 180) % 360) - 180` for both 1D and 2D, then `sortby("lon")` for 1D.

**Why this method:** The Indian Ocean is defined as 20°E–120°E. In a 0–360 system, this is just `20–120`. In -180..180, it's still `20–120`. Same slice works for both. But many plotting libraries require -180..180, and CMIP6 models are mixed (GFDL: 0–360, EC-Earth3: 0–360, NOAA: -180..180). Converting everything to -180..180 makes them consistent.

**Alternatives & tradeoffs:**
- **Keep 0–360, adjust the slice** — would mean splitting the slice at 360 for datasets already in -180..180. The slice `lon=slice(20, 120)` works identically in both systems for this specific Indian Ocean region, so it's fine either way. But if you change the region to cross the date line (e.g., Pacific), -180..180 is necessary.
- **Use xarray's `roll()`** — `ds.roll(lon=180, roll_coords=True)` shifts the data so that 0° wraps to the center. This keeps the values as 0–360 but rearranges the array. Problem: the coordinate values themselves remain 0–360, so `sel(lon=slice(20, 120))` still works, but numerical computations (e.g., `lon.max()`) still show 360.
- **Don't convert** — works for the Indian Ocean subset since the slice is unambiguous, but fails for climate indices that require -180..180 convention. Also causes issues when concatenating NOAA (-180..180) with CMIP6 models.

**For 2D grids**, we use `xr.DataArray(new_lon, dims=ds.lon.dims)` instead of bare numpy because xarray's `assign_coords` requires explicit dimension names for multi-dimensional coordinates. Without this, you get `MissingDimensionsError`.

---

## 3. Spatial Subsetting (Indian Ocean)

**What we did:** For 1D: `sel(lat=slice(-40, 30), lon=slice(20, 120))` with latitude ascending check. For 2D: boolean mask on flattened lat/lon, compute, then `.where(mask, drop=True)`.

**Why this method:** 1D regular grids can use xarray's optimized `sel()` which internally creates an index, enabling O(log N) lookup. 2D curvilinear grids have no index — lat/lon are auxiliary coordinates on grid dimensions, so we must build a mask manually.

**Why `mask.compute()` before `where(drop=True)`:** Xarray refuses to `drop=True` on a dask boolean mask because it cannot determine the output shape without computing. This is a fundamental xarray limitation (known issue since 2019). Computing the mask is cheap since lat/lon are small coordinate arrays (e.g., 105×362 = 38K elements for EC-Earth3).

**Alternatives & tradeoffs:**
- **`sel(method="nearest")` on 2D coords** — xarray doesn't support `sel()` with method for 2D coordinates. It only works for 1D indexed coordinates.
- **Don't drop, use `.where(mask)` without `drop=True`** — keeps all original grid points but sets non-Indian-Ocean to NaN. Wasteful: the dataset carries the full global grid with 70% NaN for the Indian Ocean. This severely impacts memory and performance.
- **Use `regionmask` or other third-party libraries** — adds dependencies and complexity. The manual mask approach does exactly the same thing internally.
- **Regrid first, then subset with `sel()`** — would mean regridding the entire globe, then subsetting. Doing it the other way (subset → regrid) processes only the Indian Ocean region through the expensive regridding step.

**Descending latitude handling:** Some CMIP models store latitude from North to South (90 to -90). The slice `lat=slice(-40, 30)` would return nothing because `slice(start, stop)` on a descending axis interprets `-40` as "start from -40 and go downward" which immediately stops. We detect this by comparing the first and last values, then `sortby("lat")` to make it ascending.

---

## 4. Kelvin → Celsius Conversion

**What we did:** Check `sst.attrs["units"]` for "kelvin"/"K"/"degK", subtract 273.15 if found.

**Why this method:** NOAA OISST is already in °C. CMIP6 models typically store SST in Kelvin (units="K"). The function checks the metadata attribute rather than the actual data values, which is more robust — you can't reliably auto-detect Kelvin vs Celsius from values alone (25°C = 298K, both plausible).

**Alternatives & tradeoffs:**
- **Always subtract 273.15** — breaks already-in-°C data (e.g., NOAA would become -248°C).
- **Detect from data values** — `if sst.mean() > 100` would work since ocean SST is 0-40°C or 273-313K. But it requires a `.compute()` call and could misclassify cold polar regions.
- **Never convert, handle Kelvin downstream** — all analysis would need to track units. Cleaner to standardize early.

---

## 5. Quality Control (Impossible SST Removal)

**What we did:** `ds["sst"] = ds.sst.where((sst >= -2) & (sst <= 40), other=np.nan)`, targeting only the `sst` variable.

**Why targeting `sst` only:** The original code did `ds.where(cond, other=np.nan)` on the entire dataset. This fails when the dataset has non-float variables (e.g., `time_bnds` with datetime dtype, `area` with float64) because xarray tries to replace False values with NaN in all variables, including datetime-typed ones. Targeting `sst` only avoids this type-promotion error.

**Alternatives & tradeoffs:**
- **Don't filter** — land grid cells and bad data (e.g., SST > 40°C) would corrupt statistics. Land cells in the Indian Ocean naturally occur near coastlines in the 0.25° NOAA data.
- **Filter after concat** — works but means carrying invalid data through the intermediate pipeline. Filtering early is memory-safe since bad values are set to NaN early, potentially improving compression.
- **Use `sst.where(cond)` without `other=np.nan`** — defaults to NaN anyway. Explicit is clearer.

---

## 6. CMIP6 Regridding (Common Grid)

**What we did:** For 1D lat/lon → `xr.interp()`. For 2D curvilinear → `scipy.interpolate.griddata(method="linear")` on each time step in chunks of 50.

**Why regrid at all:** Five CMIP6 models have incompatible native grids:
- EC-Earth3: 105×362 curvilinear
- GFDL-ESM4: 70×100 regular 1D (already Indian Ocean subset)
- IITM-ESM: 98×360 curvilinear
- IPSL-CM6A-LR: 104×361 curvilinear
- NorCPM1: 203×89 curvilinear

These cannot be concatenated along time because they have different spatial dimensions. `xr.concat(dim="time")` requires identical non-concat dimensions (or you use `join="override"` which creates huge NaN-filled arrays).

**Why 0.5° resolution:**
- NOAA is 0.25° (finer than needed for CMIP6)
- CMIP6 models are typically 1°–2° resolution
- 0.5° preserves most model detail without creating an excessively large grid (141×201 = 28K grid points)
- If we used 0.25° (matching NOAA), the CMIP6 output would be 4× larger (276×401 = 110K points) with no real information gain since CMIP6 models can't resolve 0.25° features
- If we used 1°, we'd lose finer features from higher-res models like GFDL-ESM4 (~1°)

**Why `method="linear"` over `"nearest"`:**
- `linear` produces smooth physical fields — the interpolated SST varies continuously between source grid points
- `nearest` produces blocky artifacts — each target point gets the value of the nearest source grid cell, creating visible grid-cell boundaries in the output
- For climate research, smooth fields are preferred for spatial analysis, EOFs, and gradient computations

**Why not `cubic`:** Much slower, and for sparse curvilinear grids, can produce unrealistic overshoots (SST > 40°C or < -2°C).

**Why process in chunks of 50:** The individual `griddata` calls per time step are slow but memory-efficient. Processing all 396 time steps in one vectorized call would require holding 396 2D arrays in memory simultaneously — ~400 MB for 105×362 grids. With chunking, peak memory is ~50 MB.

**Alternatives & tradeoffs:**
- **xESMF (ESMF regridding)** — the proper tool for climate model regridding. It supports conservative, bilinear, and patch methods. But it requires `esmpy` which is notoriously difficult to install on Windows (Fortran compiler needed). Not available in this environment.
- **CDO (Climate Data Operators)** — command-line tool for climate data processing. Not a Python library, requires separate installation.
- **Don't regrid, save each model separately** — cleaner but incompatible with the pipeline design that produces one file per source. User would need a separate regridding step before any multi-model analysis.
- **Use `xr.interp()` for 1D and `xr.interp()` on stacked dimensions for 2D** — xarray's interp doesn't support 2D coordinates (only 1D). You'd need to convert the 2D grid to 1D first, which is essentially what `griddata` does.

---

## 7. Calendar Standardization

**What we did:** Try `pd.Timestamp(time[0])`, if it fails, iterate over time values converting each `cftime.datetime` → `pd.Timestamp`.

**Why this method:** CMIP6 models use different calendars:
- GFDL-ESM4: `noleap` (no leap years) → `cftime.DatetimeNoLeap`
- IITM-ESM: `standard` → `cftime.DatetimeGregorian`
- EC-Earth3-CC: `proleptic_gregorian` → already standard
- IPSL-CM6A-LR: `gregorian` → already standard
- NorCPM1: `noleap` → `cftime.DatetimeNoLeap`

Mixed calendar types cannot be compared or sorted because `cftime.DatetimeNoLeap < pd.Timestamp` raises `TypeError`. Standardizing all to `pd.Timestamp` uses the Proleptic Gregorian calendar (what we use in daily life) and enables sorting, diff, and date arithmetic.

**Alternatives & tradeoffs:**
- **Keep cftime, compare using `cftime.date2num`** — would require converting timestamps to numeric for every comparison. Impractical for general use.
- **Use `cftime.num2date` with a standard calendar** — requires knowing the original units. More complex.
- **Skip calendar conversion, catch sort errors** — we do this as a fallback (`sortby_time_safe`).
- **List comprehension vs vectorized** — looping in Python over 2000 time values takes ~0.1 seconds, negligible compared to regridding. `pd.to_datetime()` doesn't handle `cftime` arrays, so vectorization isn't possible.

---

## 8. Lazy Evaluation & Dask Chunking

**What we did:** Open with `chunks={}`, concat lazily, rechunk to `{time: 100, lat: -1, lon: -1}`.

**Why chunks={}:** Tells xarray to use dask but auto-chunk each dimension. This keeps operations lazy — opening 33 NOAA files doesn't load any data, just creates a task graph.

**Why rechunk to {time: 100, lat: -1, lon: -1}:** After concatenating 33 separate years (each with its own chunks), the task graph has 33 separate chains. Rechunking consolidates them into ~121 chunks along time (12053/100) with full lat/lon chunks. This simplifies the task graph dramatically, reducing scheduler overhead.

**Alternatives & tradeoffs:**
- **No dask, load eagerly** — would require ~5 GB RAM for NOAA and ~100 MB for CMIP6. Exceeds available memory on most laptops.
- **Chunk only time** — `{time: 100}` means each chunk loads all 280×400 = 112K spatial points × 100 time steps = ~45 MB per chunk. Fine for NOAA.
- **Chunk everything** — `{time: 50, lat: 50, lon: 50}` creates many small chunks (thousands), increasing task graph overhead.
- **Use `chunks="auto"`** — xarray's auto-chunking adapts to data size but can create suboptimal chunk structures for concatenated data.

**Why dataset_summary uses a single `dask.compute()`:** Computing 5 statistics separately triggers 5 full passes through the data (each `sst.min().compute()` starts from scratch). With `dask.compute(sst.min(), sst.max(), ..., sst.isnull().sum())`, all 5 are fused into a single task graph and computed in one pass. This is ~5x faster.

---

## 9. Figure Generation

**What we did:** Compute mean SST, time series, and subsample together in one `dask.compute()`, then plot separately.

**Why compute together:** Same reason as dataset_summary — fusing multiple reductions into one compute pass avoids re-reading the data multiple times.

**Why subsample every 100th time step for histogram:** The full NOAA dataset has 12,053 time steps × 280 × 400 = 1.35 billion values. A histogram needs a representative sample, not the full distribution. Every 100th step gives ~120 time steps × 112K points = 13 million values, which is more than enough for a smooth histogram. If still > 1M, we randomly downsample to 1M.

**Why clim.dims not sst.dims in climatology:** After `groupby("time.month").mean("time")`, the resulting DataArray has dimensions `('month', 'lat', 'lon')`, not `('time', 'lat', 'lon')`. Using `sst.dims` would filter for dimensions not equal to "time", giving `['lat', 'lon']`, but then `clim.mean(dim=['lat', 'lon'])` needs the "month" dimension. The corrected code uses `clim.dims` which is `['month', 'lat', 'lon']`, so filtering out "month" gives `['lat', 'lon']` — the correct spatial dimensions.

**Alternatives & tradeoffs:**
- **Compute each figure independently** — simpler code but 4 separate full data scans instead of 1. For the 5 GB NOAA dataset, this would be ~20 minutes vs ~5 minutes.
- **Use cartopy for map projections** — would produce publication-quality maps with coastlines. But adds a heavy dependency (cartopy requires GEOS, complex to install on Windows). The simple pcolormesh is sufficient for quick-look figures.
- **Save as PDF/SVG** — vector formats would be huge for 280×400 grid cells. PNG is appropriate.

---

## 10. Safe Saving (Temp File + Rename)

**What we did:** Write to a temporary file in the same directory, then `os.replace()` atomically.

**Why this method:** If the script crashes during `to_netcdf()` (power loss, OOM, disk full, encoding error), the output file would be left in a corrupt state. The temp file isolates this risk:
1. If `to_netcdf` fails mid-write, only the temp file is corrupt
2. `os.replace()` is atomic on POSIX and near-atomic on Windows — the destination is either fully the old file or fully the new file, never a partial write

**Why not `mode="w"`:** We tried this earlier but xarray's `to_netcdf` with `mode="w"` on an existing file that was opened by a previous dask worker can cause `PermissionError` due to stale file handles. The temp file avoids this entirely since it's always a new file.

**Why zlib level 1:** Level 1 provides ~2× compression (5 GB → 1.8 GB) at ~5× the speed of level 5. Level 9 would compress to ~1.5 GB but take 20× longer. For a 5 GB dataset, the tradeoff of level 1 is optimal — good compression ratio with acceptable speed.

---

## 11. The "1576 duplicate time steps" in CMIP6

**What happened:** 5 models × 396-420 time steps = 2004 total, but only 428 unique. The remaining 1576 are duplicates.

**Why:** The CMIP6 models cover overlapping periods (1980–2014 for NorCPM1, 1982–2014 for others). When concatenated along time, time steps from different models that fall on the same date are stacked. Since we concatenated along time (not a model dimension), the same date appears multiple times — once per model.

**Is this correct?** Yes and no. The pipeline treats all CMIP6 data as one time series, which effectively interleaves different models. For a proper multi-model ensemble analysis, you'd typically keep a model dimension. But the current design treats CMIP6 as one file per source. The duplicate removal (`np.unique`) keeps only the first occurrence per date.

**Better approach:** Concatenate along a new `model` dimension instead of `time`. But this changes the downstream analysis fundamentally — all statistics would be per-model rather than per-date. For quick-look figures, the current approach is acceptable.

---

## 12. Summary Table

| Component | Chosen Method | Why Not Other |
|-----------|--------------|---------------|
| Coord rename | Dict map + `rename()` | Covers all CMIP6 conventions, no fragile if/elif |
| Longitude | `(lon+180)%360-180` | Works for both 1D and 2D, correct modulo for negative values |
| Subset (1D) | `sel(lat=..., lon=...)` | O(log N) via index, no computation |
| Subset (2D) | Boolean mask + `compute()` + `where(drop=True)` | Xarray requires computed mask for `drop=True` |
| Temp filter | `ds["sst"] = sst.where(cond)` | Avoids dtype promotion errors on non-float variables |
| Regrid (1D) | `xr.interp()` | Native xarray, vectorized, fast |
| Regrid (2D) | `scipy.griddata(method="linear", chunks=50)` | xESMF not available; linear > nearest for smooth fields |
| Calendar | `pd.Timestamp(cftime_obj)` loop | Necessary for cross-calendar comparison |
| Chunking | `{time:100, lat:-1, lon:-1}` | Balances graph complexity vs chunk overhead |
| Stats | Single `dask.compute()` | 5× faster than sequential `.compute()` |
| Save | Temp file + `os.replace()` | Crash-safe, avoids stale-handle PermissionError |
| Compression | zlib level 1 | Best speed/size tradeoff for archival |
| Figures | Batch compute + `clim.dims` | Avoids 4× full scans; `clim.dims` fixes the bug |
