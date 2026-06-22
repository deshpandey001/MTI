import xarray as xr

for yr in [2016, 2017]:
    ds = xr.open_dataset(f'data/sst.day.mean.{yr}.nc')
    print(f'{yr}: time={ds.dims["time"]}, sst attrs={ds["sst"].attrs}')
    print()
