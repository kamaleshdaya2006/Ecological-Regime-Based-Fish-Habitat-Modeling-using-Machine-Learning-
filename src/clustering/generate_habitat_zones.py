import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# LOAD DATA
# ==============================

regime_ds = xr.open_dataset(
    "global_raw_regime_output/global_regimes_raw.nc"
)

regimes = regime_ds["global_regime"]

# Put your favorable regimes here manually
habitat_regimes = [1, 3]   # <-- replace with output from previous script

# ==============================
# CREATE HABITAT MASK
# ==============================

habitat_mask = regimes.isin(habitat_regimes)

habitat_mask.to_netcdf("ecological_habitat_zones.nc")

print("Habitat zone dataset saved.")