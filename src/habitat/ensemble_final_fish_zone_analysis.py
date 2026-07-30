import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union
import shapely.prepared as sprepared

# ======================================================
# SETTINGS
# ======================================================

folder = "processed_monthly_0p25"
best_k = 6

# ======================================================
# LOAD DATA
# ======================================================

anom = xr.open_dataset(os.path.join(folder, "marine_anomalies_standardized.nc"))

lats = anom.latitude.values
lons = anom.longitude.values
n_lat = len(lats)
n_lon = len(lons)

# ======================================================
# BUILD OCEAN MASK (IMPORTANT FIX)
# ======================================================

print("Building land mask...")

land_shp = shpreader.natural_earth(
    resolution='110m',
    category='physical',
    name='land'
)

reader = shpreader.Reader(land_shp)
land_geom = unary_union([geom for geom in reader.geometries()])
land_geom_prepared = sprepared.prep(land_geom)

land_mask = np.zeros((n_lat, n_lon), dtype=bool)

for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        if land_geom_prepared.contains(sgeom.Point(lon, lat)):
            land_mask[i, j] = True

print("Land mask complete.")

# ======================================================
# BUILD GLOBAL FEATURE MATRIX
# ======================================================

feature_list = []
for var in anom.data_vars:
    feature_list.append(anom[var].mean("time"))
    feature_list.append(anom[var].std("time"))

feature_stack = xr.concat(feature_list, dim="feature")

X_global = feature_stack.values.reshape(len(feature_list), -1).T
valid_mask = ~np.isnan(X_global).any(axis=1)
X_valid = X_global[valid_mask]

# ======================================================
# FIT GLOBAL GMM
# ======================================================

gmm = GaussianMixture(n_components=best_k, random_state=42)
gmm.fit(X_valid)

# ======================================================
# DEFINE CLUSTER SUITABILITY SCORES
# (Replace with your real ones)
# ======================================================

cluster_scores = {
    0: 0.85,
    1: 0.32,
    2: 0.61,
    3: 0.12,
    4: 0.74,
    5: 0.45
}

vals = np.array(list(cluster_scores.values()))
minv, maxv = vals.min(), vals.max()

for k in cluster_scores:
    cluster_scores[k] = (cluster_scores[k] - minv) / (maxv - minv)

# ======================================================
# DISCRETE COLORMAP
# ======================================================

zone_colors = [
    "#8B0000",  # 1 Very Low
    "#FF4500",  # 2 Low
    "#FFD700",  # 3 Moderate
    "#7CFC00",  # 4 High
    "#006400"   # 5 Very High
]

cmap = ListedColormap(zone_colors)
bounds = [1,2,3,4,5,6]
norm = BoundaryNorm(bounds, cmap.N)

# ======================================================
# MONTHLY LOOP
# ======================================================

for m in range(1, 13):

    print(f"Processing month {m}")

    month_data = anom.sel(time=anom["time.month"] == m)
    month_mean = month_data.mean("time")

    feature_list_month = []
    for var in month_mean.data_vars:
        feature_list_month.append(month_mean[var])
        feature_list_month.append(month_data[var].std("time"))

    feature_stack_month = xr.concat(feature_list_month, dim="feature")

    X_month = feature_stack_month.values.reshape(len(feature_list_month), -1).T
    valid_mask_month = ~np.isnan(X_month).any(axis=1)

    probs = np.zeros((X_month.shape[0], best_k))
    probs[valid_mask_month] = gmm.predict_proba(X_month[valid_mask_month])

    # ==================================================
    # HSI
    # ==================================================

    hsi_flat = np.zeros(X_month.shape[0])

    for k in range(best_k):
        hsi_flat += probs[:, k] * cluster_scores[k]

    hsi_map = hsi_flat.reshape(n_lat, n_lon)

    # Normalize within month
    valid_vals = hsi_map[~np.isnan(hsi_map)]
    hsi_map = (hsi_map - valid_vals.min()) / (valid_vals.max() - valid_vals.min())

    # ==================================================
    # 5-ZONE CLASSIFICATION
    # ==================================================

    p20 = np.percentile(valid_vals, 20)
    p40 = np.percentile(valid_vals, 40)
    p60 = np.percentile(valid_vals, 60)
    p80 = np.percentile(valid_vals, 80)

    zones = np.full_like(hsi_map, 1)
    zones[hsi_map >= p20] = 2
    zones[hsi_map >= p40] = 3
    zones[hsi_map >= p60] = 4
    zones[hsi_map >= p80] = 5

    # ==================================================
    # FORCE LAND = ZONE 1
    # ==================================================

    zones[land_mask] = 1

    # ==================================================
    # PLOT
    # ==================================================

    fig = plt.figure(figsize=(12,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([42.5, 97.5, -27.5, 27.5])

    mesh = ax.pcolormesh(
        lons,
        lats,
        zones,
        cmap=cmap,
        norm=norm,
        shading="nearest",
        transform=ccrs.PlateCarree()
    )

    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    cbar = plt.colorbar(mesh, ticks=[1.5,2.5,3.5,4.5,5.5])
    cbar.ax.set_yticklabels([
        "Very Low",
        "Low",
        "Moderate",
        "High",
        "Very High"
    ])

    plt.title(f"Ensemble Fish Habitat Zones — Month {m}")
    plt.show()

print("All months plotted successfully.")