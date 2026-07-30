import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap

# =====================================
# LOAD REGIME DATA
# =====================================

ds = xr.open_dataset("pca_ecological_regimes.nc")

regimes = ds["ecological_regime"]

lats = ds.latitude.values
lons = ds.longitude.values

# =====================================
# MODE FUNCTION
# =====================================

def mode_func(x, axis=None):

    x = np.moveaxis(x, axis, 0)

    out = np.full(x.shape[1:], np.nan)

    for i in range(x.shape[1]):
        for j in range(x.shape[2]):

            vals = x[:, i, j]
            vals = vals[~np.isnan(vals)]

            if len(vals) == 0:
                continue

            vals = vals.astype(int)

            out[i, j] = np.bincount(vals).argmax()

    return out

# =====================================
# MONTHLY CLIMATOLOGY (MODE)
# =====================================

regimes_clim = regimes.groupby("time.month").reduce(mode_func)

# =====================================
# MONTH NAMES
# =====================================

month_names = [
    "January","February","March","April",
    "May","June","July","August",
    "September","October","November","December"
]

# =====================================
# COLOR MAP
# =====================================

colors = [
    "#b50000",
    "#fe8c01",
    "#f0fc08",
    "#7bff00",
    "#04ff00",
    "#005603"
]

cmap = ListedColormap(colors)

# =====================================
# PLOT
# =====================================

for m in range(12):

    data = regimes_clim.isel(month=m)

    fig = plt.figure(figsize=(12,8))

    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([42.5, 97.5, -27.5, 27.5])

    mesh = ax.pcolormesh(
        lons,
        lats,
        data,
        cmap=cmap,
        shading="auto",
        transform=ccrs.PlateCarree()
    )

    ax.coastlines(resolution="10m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(mesh)
    cbar.set_label("Ecological Regime ID")

    plt.title(f"PCA Ecological Regimes — {month_names[m]}")

    plt.show()