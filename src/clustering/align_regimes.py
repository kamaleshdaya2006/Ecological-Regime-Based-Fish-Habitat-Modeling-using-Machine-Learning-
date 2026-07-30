import xarray as xr
import numpy as np

ds = xr.open_dataset("ensemble_output/aligned_regimes.nc")

r1 = ds.aligned_regime[0].values
r2 = ds.aligned_regime[1].values

print(np.array_equal(r1, r2))