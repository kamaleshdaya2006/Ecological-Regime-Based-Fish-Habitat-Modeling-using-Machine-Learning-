import xarray as xr
import os

folder = "processed_monthly_0p25"

ds = xr.open_dataset(os.path.join(folder, "marine_2015_2024_monthly_0p25.nc"))

# Group by calendar month
climatology = ds.groupby("time.month").mean("time")

print("Climatology dims:")
print(climatology.dims)

climatology.to_netcdf(os.path.join(folder, "marine_climatology_2015_2024.nc"))
# Compute anomalies
anomalies = ds.groupby("time.month") - climatology

print("Anomaly dims:")
print(anomalies.dims)

anomalies.to_netcdf(os.path.join(folder, "marine_anomalies_2015_2024.nc"))
for var in anomalies.data_vars:
    print(var, float(anomalies[var].mean()))
import numpy as np

anom = anomalies.copy()

for var in anom.data_vars:
    mean = anom[var].mean()
    std = anom[var].std()
    anom[var] = (anom[var] - mean) / std

print("Standardized anomaly check:")
for var in anom.data_vars:
    print(var, float(anom[var].mean()), float(anom[var].std()))

anom.to_netcdf(os.path.join(folder, "marine_anomalies_standardized.nc"))