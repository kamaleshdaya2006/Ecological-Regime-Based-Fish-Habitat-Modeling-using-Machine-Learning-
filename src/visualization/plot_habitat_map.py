import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==============================
# LOAD HABITAT DATA
# ==============================

ds = xr.open_dataset("ecological_habitat_zones.nc")
mask = ds["global_regime"]

# Choose month index (0 = Jan 2015)
month_index = 0

data = mask.isel(time=month_index)

# ==============================
# PLOT
# ==============================

fig = plt.figure(figsize=(12,6))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.coastlines()
ax.add_feature(cfeature.BORDERS)
ax.set_extent([42.5, 97.5, -27.5, 27.5], crs=ccrs.PlateCarree())

data.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="YlGn",
    add_colorbar=False
)

plt.title("Ecological Fish Habitat Zone")
plt.show()