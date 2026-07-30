import xarray as xr
import numpy as np
import os
import glob

data_folder =r"D:\Project\fish density prediction\NoAA\marine_data\All_environment_data"
output_folder = "processed_monthly_0p25"
os.makedirs(output_folder, exist_ok=True)

lat_min, lat_max = -27.5, 27.5
lon_min, lon_max = 42.5, 97.5

# Regular 0.25 grid
new_lats = np.arange(lat_min, lat_max + 0.25, 0.25)
new_lons = np.arange(lon_min, lon_max + 0.25, 0.25)

def process_file(file_path):

    print("\nProcessing:", os.path.basename(file_path))

    ds = xr.open_dataset(file_path, chunks={"time": 30})

    # Remove depth if exists
    if "depth" in ds.dims:
        ds = ds.isel(depth=0)

    varname = list(ds.data_vars)[0]

    # Convert SST from Kelvin to Celsius
    if "sst" in file_path.lower():
        ds[varname] = ds[varname] - 273.15

    # Daily → Monthly
    ds_monthly = ds.resample(time="1M").mean()

    # Interpolate to 0.25 grid if not already 0.25
    lat_res = float(abs(ds.latitude[1] - ds.latitude[0]))

    if abs(lat_res - 0.25) > 0.01:
        ds_monthly = ds_monthly.interp(
            latitude=new_lats,
            longitude=new_lons,
            method="linear"
        )

    # Save new file
    out_name = os.path.basename(file_path).replace(".nc", "_monthly_0p25.nc")
    out_path = os.path.join(output_folder, out_name)

    ds_monthly.to_netcdf(out_path)

    # Print summary
    print("Saved:", out_name)
    print("Time:", str(ds_monthly.time.values[0]), 
          "to", str(ds_monthly.time.values[-1]))
    print("Lat:", float(ds_monthly.latitude.min()), 
          "to", float(ds_monthly.latitude.max()))
    print("Lon:", float(ds_monthly.longitude.min()), 
          "to", float(ds_monthly.longitude.max()))
    print("First 10 grid points:")
    for i in range(10):
        print(f"({new_lats[i]}, {new_lons[i]})")

    ds.close()


# Run on all files
files = glob.glob(os.path.join(data_folder, "*.nc"))

for f in files:
    process_file(f)