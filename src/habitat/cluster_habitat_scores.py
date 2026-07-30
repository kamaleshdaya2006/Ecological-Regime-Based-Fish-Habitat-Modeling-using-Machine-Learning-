import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.mixture import GaussianMixture

# =====================================================
# SETTINGS
# =====================================================

folder = "processed_monthly_0p25"
best_k = 6

# =====================================================
# LOAD STANDARDIZED ANOMALIES
# =====================================================

anom = xr.open_dataset(os.path.join(folder, "marine_anomalies_standardized.nc"))

lats = anom.latitude.values
lons = anom.longitude.values
n_lat = len(lats)
n_lon = len(lons)

# =====================================================
# BUILD FEATURE MATRIX (ANNUAL CLUSTERING FEATURES)
# =====================================================

feature_list = []
feature_names = []

for var in anom.data_vars:
    mean_field = anom[var].mean("time")
    std_field  = anom[var].std("time")

    feature_list.append(mean_field)
    feature_list.append(std_field)

    feature_names.append(f"{var}_mean")
    feature_names.append(f"{var}_std")

feature_stack = xr.concat(feature_list, dim="feature")

X = feature_stack.values.reshape(len(feature_list), -1).T
valid_mask = ~np.isnan(X).any(axis=1)
X_valid = X[valid_mask]

print("Total valid spatial cells:", X_valid.shape[0])

# =====================================================
# FIT GMM
# =====================================================

gmm = GaussianMixture(n_components=best_k, random_state=42)
labels_valid = gmm.fit_predict(X_valid)

# =====================================================
# COMPUTE ECOLOGICAL SUITABILITY SCORE PER CLUSTER
# =====================================================

cluster_scores = {}

for cid in range(best_k):

    centroid = X_valid[labels_valid == cid].mean(axis=0)
    centroid_dict = dict(zip(feature_names, centroid))

    # -------- ECOLOGICAL LOGIC --------
    # Positive: chlorophyll mean
    # Positive: oxygen mean
    # Negative: SST variability

    chl_score = centroid_dict.get("chl_mean", 0)
    o2_score  = centroid_dict.get("o2_mean", 0)
    sst_var   = -abs(centroid_dict.get("adjusted_sea_surface_temperature_std", 0))

    score = 0.5 * chl_score + 0.3 * o2_score + 0.2 * sst_var
    cluster_scores[cid] = score

# Normalize 0–1
vals = np.array(list(cluster_scores.values()))
minv, maxv = vals.min(), vals.max()

for k in cluster_scores:
    cluster_scores[k] = (cluster_scores[k] - minv) / (maxv - minv)

print("\nCluster Ecological Suitability Scores:")
print(cluster_scores)

# =====================================================
# CREATE ANNUAL CLUSTER MAP
# =====================================================

labels_full = np.full(X.shape[0], np.nan)
labels_full[valid_mask] = labels_valid

cluster_map = labels_full.reshape(n_lat, n_lon)

# Convert cluster → ecological score
zone_map = np.full_like(cluster_map, np.nan)

for cid in cluster_scores:
    zone_map[cluster_map == cid] = cluster_scores[cid]

# =====================================================
# CLASSIFY INTO 5 FISH ZONES (ANNUAL)
# =====================================================

valid_vals = zone_map[~np.isnan(zone_map)]

p20 = np.percentile(valid_vals, 20)
p40 = np.percentile(valid_vals, 40)
p60 = np.percentile(valid_vals, 60)
p80 = np.percentile(valid_vals, 80)

zones_annual = np.full_like(zone_map, 1)

zones_annual[zone_map >= p20] = 2
zones_annual[zone_map >= p40] = 3
zones_annual[zone_map >= p60] = 4
zones_annual[zone_map >= p80] = 5

# Save annual map
annual_ds = xr.Dataset(
    {"fish_zone": (["latitude", "longitude"], zones_annual)},
    coords={"latitude": lats, "longitude": lons}
)

annual_ds.to_netcdf("annual_cluster_fish_zones.nc")

print("Annual fish zone map saved.")

# =====================================================
# PLOT ANNUAL MAP
# =====================================================

fig = plt.figure(figsize=(12,8))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([42.5, 97.5, -27.5, 27.5], crs=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.BORDERS)

mesh = ax.pcolormesh(
    lons,
    lats,
    zones_annual,
    cmap="RdYlGn",
    shading="auto",
    transform=ccrs.PlateCarree(),
    vmin=1,
    vmax=5
)

cbar = plt.colorbar(mesh, orientation="vertical", pad=0.02)
cbar.set_label("Fish Habitat Zone (1=Very Low → 5=Very High)")

plt.title("Annual Fish Habitat Zones (Cluster-Based)")
plt.show()

# =====================================================
# MONTHLY FISH ZONES
# =====================================================

monthly_zones = []

for m in range(1, 13):

    month_data = anom.sel(time=anom["time.month"] == m)

    month_mean = month_data.mean("time")

    # Build monthly feature matrix
    feature_list_month = []
    for var in month_mean.data_vars:
        feature_list_month.append(month_mean[var])
        feature_list_month.append(month_data[var].std("time"))

    feature_stack_month = xr.concat(feature_list_month, dim="feature")

    X_month = feature_stack_month.values.reshape(len(feature_list_month), -1).T
    valid_mask_month = ~np.isnan(X_month).any(axis=1)

    labels_full_month = np.full(X_month.shape[0], np.nan)
    labels_full_month[valid_mask_month] = gmm.predict(X_month[valid_mask_month])

    cluster_map_month = labels_full_month.reshape(n_lat, n_lon)

    # Convert cluster → ecological score
    zone_month = np.full_like(cluster_map_month, np.nan)
    for cid in cluster_scores:
        zone_month[cluster_map_month == cid] = cluster_scores[cid]

    # Percentile classification
    vals = zone_month[~np.isnan(zone_month)]
    p20 = np.percentile(vals, 20)
    p40 = np.percentile(vals, 40)
    p60 = np.percentile(vals, 60)
    p80 = np.percentile(vals, 80)

    zones_month = np.full_like(zone_month, 1)
    zones_month[zone_month >= p20] = 2
    zones_month[zone_month >= p40] = 3
    zones_month[zone_month >= p60] = 4
    zones_month[zone_month >= p80] = 5

    monthly_zones.append(zones_month)

    # Plot
    fig = plt.figure(figsize=(12,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([42.5, 97.5, -27.5, 27.5])
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.BORDERS)

    mesh = ax.pcolormesh(
        lons,
        lats,
        zones_month,
        cmap="RdYlGn",
        shading="auto",
        transform=ccrs.PlateCarree(),
        vmin=1,
        vmax=5
    )

    plt.title(f"Fish Habitat Zones — Month {m}")
    plt.colorbar(mesh, label="Fish Habitat Zone")
    plt.show()

# Save monthly dataset
monthly_zones = np.array(monthly_zones)

monthly_ds = xr.Dataset(
    {"fish_zone": (["month", "latitude", "longitude"], monthly_zones)},
    coords={
        "month": np.arange(1,13),
        "latitude": lats,
        "longitude": lons
    }
)

monthly_ds.to_netcdf("monthly_cluster_fish_zones.nc")

print("Monthly fish zone maps saved.")