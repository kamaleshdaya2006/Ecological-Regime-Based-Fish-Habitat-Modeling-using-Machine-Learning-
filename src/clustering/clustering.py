import xarray as xr
import numpy as np
import os

folder = "processed_monthly_0p25"

anom = xr.open_dataset(os.path.join(folder, "marine_anomalies_standardized.nc"))

feature_list = []
feature_names = []

for var in anom.data_vars:
    mean_field = anom[var].mean("time")
    std_field  = anom[var].std("time")

    feature_list.append(mean_field)
    feature_list.append(std_field)

    feature_names.append(f"{var}_mean")
    feature_names.append(f"{var}_std")

# Stack features
feature_stack = xr.concat(feature_list, dim="feature")
feature_stack["feature"] = feature_names

# Reshape to (n_cells, n_features)
n_features = len(feature_names)
n_lat = feature_stack.latitude.size
n_lon = feature_stack.longitude.size

X = feature_stack.values.reshape(n_features, -1).T

print("Feature matrix shape:", X.shape)
valid_mask = ~np.isnan(X).any(axis=1)
X_valid = X[valid_mask]

print("Valid samples:", X_valid.shape)
from sklearn.mixture import GaussianMixture

bic_scores = []

for k in range(4, 9):
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
    gmm.fit(X_valid)
    bic = gmm.bic(X_valid)
    bic_scores.append((k, bic))
    print("Clusters:", k, "BIC:", bic)

print("Best model:", min(bic_scores, key=lambda x: x[1]))
best_k = 6

gmm = GaussianMixture(n_components=best_k, covariance_type="full", random_state=42)
labels_valid = gmm.fit_predict(X_valid)

print("Cluster label count:")
print(np.bincount(labels_valid))
labels_full = np.full(X.shape[0], np.nan)
labels_full[valid_mask] = labels_valid

cluster_map = labels_full.reshape(n_lat, n_lon)

print("Cluster map shape:", cluster_map.shape)
cluster_ds = xr.Dataset(
    {"cluster": (["latitude", "longitude"], cluster_map)},
    coords={
        "latitude": anom.latitude,
        "longitude": anom.longitude
    }
)

cluster_ds.to_netcdf(os.path.join(folder, "ecological_regimes_map.nc"))
cluster_ids = np.unique(labels_valid)

for cid in cluster_ids:
    cluster_mean = X_valid[labels_valid == cid].mean(axis=0)
    print("Cluster", cid)
    for name, val in zip(feature_names, cluster_mean):
        print("  ", name, round(val, 3))
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
plt.pcolormesh(
    anom.longitude,
    anom.latitude,
    cluster_map,
    shading='auto'
)
plt.colorbar(label="Cluster ID")
plt.title("Ecological Regimes (GMM)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()