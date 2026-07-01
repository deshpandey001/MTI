import xarray as xr
import pandas as pd

ds = xr.open_dataset("../data/processed/cmip6_historical_regridded.nc")

print("Variables:", list(ds.data_vars))
print("Dimensions:", ds.dims)

print("\nFirst 10 dates:")
print(ds.time.values[:10])

print("\nLast 10 dates:")
print(ds.time.values[-10:])

# Check duplicates
time = pd.to_datetime(ds.time.values)
print("\nDuplicate timestamps:", time.duplicated().sum())

# Time difference statistics
diff = pd.Series(time).diff().value_counts().sort_index()
print("\nTime intervals:")
print(diff)