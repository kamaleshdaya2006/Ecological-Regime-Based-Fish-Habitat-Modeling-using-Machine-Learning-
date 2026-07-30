import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
from sklearn.mixture import GaussianMixture

# =========================================================
# SETTINGS
# =========================================================

folder = "processed_monthly_0p25"
best_k = 6

# =========================================================
# LOAD STANDARDIZED ANOMALY DATA
# =========================================================

anom = xr.open_dataset(os.path.join(folder, "marine_anomalies_standardized.nc"))

lats = anom.latitude.values
lons = anom.longitude.values
n_lat = len(lats)
n_lon = len(lons)

# =========================================================
# BUILD BASE FEATURE MATRIX (for GMM training)
# =========================================================

feature_list = []
for var in anom.data_vars:
    feature_list.append(anom[var].mean("time"))
    feature_list.append(anom[var].std("time"))

feature_stack = xr.concat(feature_list, dim="feature")

X = feature_stack.values.reshape(len(feature_list), -1).T
valid_mask = ~np.isnan(X).any(axis=1)
X_valid = X[valid_mask]

# =========================================================
# FIT GMM ON FULL DATA (ONCE)
# =========================================================

gmm = GaussianMixture(n_components=best_k, random_state=42)
gmm.fit(X_valid)

# =========================================================
# CREATE DISCRETE COLORMAP
# =========================================================

cmap = ListedColormap(plt.cm.tab10.colors[:best_k])

# =========================================================
# LOOP THROUGH 12 MONTHS
# =========================================================

for m in range(1, 13):

    # ---- Get all years of this calendar month ----
    month_data = anom.sel(time=anom["time.month"] == m)

    # Mean anomaly for that month
    month_mean = month_data.mean("time")

    # Build monthly feature matrix
    feature_list_month = []
    for var in month_mean.data_vars:
        feature_list_month.append(month_mean[var])
        feature_list_month.append(month_data[var].std("time"))

    feature_stack_month = xr.concat(feature_list_month, dim="feature")
    X_month = feature_stack_month.values.reshape(len(feature_list_month), -1).T

    valid_mask_month = ~np.isnan(X_month).any(axis=1)

    labels_full = np.full(X_month.shape[0], np.nan)
    labels_full[valid_mask_month] = gmm.predict(X_month[valid_mask_month])

    cluster_map_month = labels_full.reshape(n_lat, n_lon)

    # =========================================================
    # PLOT USING CARTOPY (YOUR STYLE)
    # =========================================================

    fig = plt.figure(figsize=(12,8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([42.5, 97.5, -27.5, 27.5], crs=ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        lons,
        lats,
        cluster_map_month,
        cmap=cmap,
        shading="auto",
        transform=ccrs.PlateCarree()
    )

    ax.coastlines(resolution="10m")
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(mesh, orientation="vertical", pad=0.02)
    cbar.set_label("Ecological Regime ID")

    month_name = month_data.time.dt.strftime('%B').values[0]
    plt.title(f"Ecological Regimes — {month_name}")
    plt.show()