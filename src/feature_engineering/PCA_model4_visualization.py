import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap

ds = xr.open_dataset("pca_habitat_zones_ensemble.nc")

zones = ds["fish_habitat_zone"]

lats = ds.latitude
lons = ds.longitude

colors = ["#8B0000","#FF4500","#FFD700","#7CFC00","#006400"]

cmap = ListedColormap(colors)

for m in range(12):

    data = zones.groupby("time.month").median("time").isel(month=m)

    fig = plt.figure(figsize=(10,7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([42.5,97.5,-27.5,27.5])

    mesh = ax.pcolormesh(
        lons,
        lats,
        data,
        cmap=cmap,
        shading="auto",
        transform=ccrs.PlateCarree()
    )

    ax.coastlines()

    ax.add_feature(cfeature.BORDERS)

    plt.title(f"Fish Habitat Zones Month {m+1}")

    plt.colorbar(mesh)

    plt.show()