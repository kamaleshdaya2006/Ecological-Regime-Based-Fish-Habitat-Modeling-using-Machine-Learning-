import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==============================
# LOAD DATA
# ==============================

regime_ds = xr.open_dataset(
    "global_raw_regime_output/global_regimes_raw.nc"
)

marine = xr.open_dataset(
    "processed_monthly_0p25/marine_2015_2024_monthly_0p25.nc"
)

regimes = regime_ds["global_regime"]

month_index = 0

regime_month = regimes.isel(time=month_index)
chl = marine["chl"].isel(time=month_index)
o2  = marine["o2"].isel(time=month_index)

# ==============================
# REGIME CPUE EFFECT (NOT normalized)
# ==============================

regime_effect = {
    0: 1.00,
    1: 0.73,
    2: 0.72,
    3: 0.75,
    4: 0.60,
    5: 0.52
}

hsi = xr.zeros_like(regime_month)

for k in regime_effect:
    hsi = hsi.where(regime_month != k, regime_effect[k])

# ==============================
# CHL PERCENTILE (robust scaling)
# ==============================

chl_flat = chl.values.flatten()
chl_flat = chl_flat[~np.isnan(chl_flat)]

chl_p90 = np.percentile(chl_flat, 90)
chl_p10 = np.percentile(chl_flat, 10)

chl_scaled = (chl - chl_p10) / (chl_p90 - chl_p10)
chl_scaled = chl_scaled.clip(0,1)

# ==============================
# OXYGEN PENALTY
# ==============================

o2_threshold = np.percentile(o2.values[~np.isnan(o2.values)], 20)

o2_penalty = xr.where(o2 < o2_threshold, 0.5, 1.0)

# ==============================
# FINAL HSI
# ==============================

hsi = hsi * 0.5 + chl_scaled * 0.5
hsi = hsi * o2_penalty

# Normalize 0–1
hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min())

# ==============================
# PERCENTILE-BASED CLASSIFICATION
# ==============================

hsi_flat = hsi.values.flatten()
hsi_flat = hsi_flat[~np.isnan(hsi_flat)]

p20 = np.percentile(hsi_flat, 20)
p40 = np.percentile(hsi_flat, 40)
p60 = np.percentile(hsi_flat, 60)
p80 = np.percentile(hsi_flat, 80)

zones = xr.full_like(hsi, 1)

zones = xr.where(hsi >= p20, 2, zones)
zones = xr.where(hsi >= p40, 3, zones)
zones = xr.where(hsi >= p60, 4, zones)
zones = xr.where(hsi >= p80, 5, zones)

# ==============================
# PLOT
# ==============================

fig = plt.figure(figsize=(12,6))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.coastlines()
ax.add_feature(cfeature.BORDERS)
ax.set_extent([42.5, 97.5, -27.5, 27.5])

zones.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="RdYlGn",
    vmin=1,
    vmax=5,
    cbar_kwargs={"label": "Fish Habitat Suitability Zone"}
)

plt.title("Fish Habitat Suitability Zones (Robust)")
plt.show()