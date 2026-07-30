import xarray as xr
import os

folder = "processed_monthly_0p25"

# ===============================
# Load datasets
# ===============================

sst = xr.open_dataset(os.path.join(folder, "sst_2015_2024_monthly_0p25.nc"))
ssh = xr.open_dataset(os.path.join(folder, "ssh_2015_2024_monthly_0p25.nc"))
sal = xr.open_dataset(os.path.join(folder, "salinity_2015_2024_monthly_0p25.nc"))
o2  = xr.open_dataset(os.path.join(folder, "do2015_2024_monthly_0p25.nc"))
chl = xr.open_dataset(os.path.join(folder, "chlorophyll_2015_2024_monthly_0p25.nc"))

# ===============================
# Remove depth coordinate safely
# ===============================

def remove_depth(ds):
    if "depth" in ds.dims:
        ds = ds.isel(depth=0)
    if "depth" in ds.coords:
        ds = ds.drop_vars("depth")
    return ds

sst = remove_depth(sst)
ssh = remove_depth(ssh)
sal = remove_depth(sal)
o2  = remove_depth(o2)
chl = remove_depth(chl)

# ===============================
# Standardize time to month-start
# ===============================

def standardize_time(ds):
    ds = ds.copy()
    ds["time"] = ds.indexes["time"].to_period("M").to_timestamp()
    return ds

sst = standardize_time(sst)
ssh = standardize_time(ssh)
sal = standardize_time(sal)
o2  = standardize_time(o2)
chl = standardize_time(chl)

# ===============================
# Force coordinate alignment
# ===============================

# Make sure all share identical coordinates
sst, ssh, sal, o2, chl = xr.align(
    sst, ssh, sal, o2, chl,
    join="inner"
)

# ===============================
# Merge datasets
# ===============================

merged = xr.merge([sst, ssh, sal, o2, chl])

print("\nFINAL MERGED DIMENSIONS:")
print(merged.dims)

print("\nVARIABLES:")
print(list(merged.data_vars))

# ===============================
# Save final dataset
# ===============================

output_path = os.path.join(folder, "marine_2015_2024_monthly_0p25.nc")
merged.to_netcdf(output_path)

print("\nSaved final merged dataset to:")
print(output_path)