import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy

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
# DEFINE CLUSTER ECOLOGICAL SCORES
# (Replace with your computed cluster_scores if needed)
# ======================================================

cluster_scores = {
    0: 0.85,
    1: 0.32,
    2: 0.61,
    3: 0.12,
    4: 0.74,
    5: 0.45
}

# Normalize 0–1
vals = np.array(list(cluster_scores.values()))
minv, maxv = vals.min(), vals.max()
for k in cluster_scores:
    cluster_scores[k] = (cluster_scores[k] - minv) / (maxv - minv)

# ======================================================
# STEP 1 — COMPUTE GLOBAL HSI RANGE
# ======================================================

all_hsi_values = []

for m in range(1, 13):

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

    hsi_flat = np.zeros(X_month.shape[0])
    for k in range(best_k):
        hsi_flat += probs[:, k] * cluster_scores[k]

    valid_vals = hsi_flat[~np.isnan(hsi_flat)]
    all_hsi_values.extend(valid_vals)

all_hsi_values = np.array(all_hsi_values)

global_min = np.min(all_hsi_values)
global_max = np.max(all_hsi_values)

p20 = np.percentile(all_hsi_values, 20)
p40 = np.percentile(all_hsi_values, 40)
p60 = np.percentile(all_hsi_values, 60)
p80 = np.percentile(all_hsi_values, 80)

print("Global HSI range:", global_min, global_max)

# ======================================================
# STORAGE
# ======================================================

monthly_hsi = []
monthly_zones = []
monthly_entropy = []
area_stats = []
centroid_stats = []

# ======================================================
# STEP 2 — MONTHLY PROCESSING
# ======================================================

for m in range(1, 13):

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

    # Continuous HSI
    hsi_flat = np.zeros(X_month.shape[0])
    for k in range(best_k):
        hsi_flat += probs[:, k] * cluster_scores[k]

    hsi_map = hsi_flat.reshape(n_lat, n_lon)

    # GLOBAL scaling
    hsi_map = (hsi_map - global_min) / (global_max - global_min)

    monthly_hsi.append(hsi_map)

    # Global percentile classification
    zones = np.full_like(hsi_map, 1)
    zones[hsi_map >= p20] = 2
    zones[hsi_map >= p40] = 3
    zones[hsi_map >= p60] = 4
    zones[hsi_map >= p80] = 5

    monthly_zones.append(zones)

    # Entropy (uncertainty)
    ent = np.zeros(X_month.shape[0])
    ent[valid_mask_month] = entropy(probs[valid_mask_month].T)
    entropy_map = ent.reshape(n_lat, n_lon)
    monthly_entropy.append(entropy_map)

    # Area stats
    total_cells = np.sum(~np.isnan(hsi_map))
    zone_area = [(zones == i).sum() / total_cells for i in range(1,6)]
    area_stats.append(zone_area)

    # Centroid of very high zone
    mask = zones == 5
    if np.any(mask):
        lat_indices, lon_indices = np.where(mask)
        lat_cent = np.mean(lats[lat_indices])
        lon_cent = np.mean(lons[lon_indices])
    else:
        lat_cent, lon_cent = np.nan, np.nan

    centroid_stats.append([lat_cent, lon_cent])

    # ==================================================
    # PLOT
    # ==================================================

    fig = plt.figure(figsize=(12,8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([42.5, 97.5, -27.5, 27.5])
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.BORDERS)

    mesh = ax.pcolormesh(
        lons,
        lats,
        zones,
        cmap="RdYlGn",
        shading="auto",
        transform=ccrs.PlateCarree(),
        vmin=1,
        vmax=5
    )

    plt.title(f"Global-Scaled Ensemble Fish Habitat Zones — Month {m}")
    plt.colorbar(mesh, label="Fish Habitat Zone (1=Very Low → 5=Very High)")
    plt.show()

# ======================================================
# SAVE OUTPUTS
# ======================================================

xr.Dataset(
    {"HSI": (["month","latitude","longitude"], np.array(monthly_hsi))},
    coords={"month": np.arange(1,13), "latitude": lats, "longitude": lons}
).to_netcdf("ensemble_monthly_HSI_global_scaled.nc")

xr.Dataset(
    {"Zones": (["month","latitude","longitude"], np.array(monthly_zones))},
    coords={"month": np.arange(1,13), "latitude": lats, "longitude": lons}
).to_netcdf("ensemble_monthly_Zones_global_scaled.nc")

xr.Dataset(
    {"Entropy": (["month","latitude","longitude"], np.array(monthly_entropy))},
    coords={"month": np.arange(1,13), "latitude": lats, "longitude": lons}
).to_netcdf("ensemble_monthly_entropy.nc")

print("\nAll global-scaled ensemble outputs saved.")

# ======================================================
# PRINT SUMMARY
# ======================================================

print("\nSeasonal Area Fraction per Zone:")
for m, stats in enumerate(area_stats, start=1):
    print(f"Month {m}: {stats}")

print("\nVery High Zone Centroid Migration:")
for m, cent in enumerate(centroid_stats, start=1):
    print(f"Month {m}: Lat={cent[0]:.2f}, Lon={cent[1]:.2f}")