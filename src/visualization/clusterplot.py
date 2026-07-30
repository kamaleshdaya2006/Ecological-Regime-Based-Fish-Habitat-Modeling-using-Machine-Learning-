import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import ListedColormap

# ===================================
# Load regime map
# ===================================

regime_ds = xr.open_dataset("processed_monthly_0p25/ecological_regimes_map.nc")
