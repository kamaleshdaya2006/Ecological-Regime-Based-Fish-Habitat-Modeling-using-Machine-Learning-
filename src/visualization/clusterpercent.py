import xarray as xr
import numpy as np
import os
from sklearn.mixture import GaussianMixture

# =========================================
# SETTINGS
# =========================================

folder = "processed_monthly_0p25"
best_k = 6   # change if you want 8

# =========================================
# LOAD STANDARDIZED ANOMALY DATA
# =========================================

anom = xr.open_dataset(os.path.join(folder, "marine_anomalies_standardized.nc"))

# =========================================
# BUILD FEATURE MATRIX (mean + std per grid cell)
# =========================================

feature_list = []

for var in anom.data_vars:
    mean_field = anom[var].mean("time")
    std_field  = anom[var].std("time")

    feature_list.append(mean_field)
    feature_list.append(std_field)

feature_stack = xr.concat(feature_list, dim="feature")

n_features = len(feature_list)
n_lat = feature_stack.latitude.size
n_lon = feature_stack.longitude.size

X = feature_stack.values.reshape(n_features, -1).T

print("Feature matrix shape:", X.shape)

# =========================================
# REMOVE NaN CELLS
# =========================================

valid_mask = ~np.isnan(X).any(axis=1)
X_valid = X[valid_mask]

print("Valid samples:", X_valid.shape)

# =========================================
# RUN GMM
# =========================================

gmm = GaussianMixture(n_components=best_k, covariance_type="full", random_state=42)
labels_valid = gmm.fit_predict(X_valid)

# =========================================
# PRINT CLUSTER AREA PERCENTAGES
# =========================================

total = len(labels_valid)

print("\nCluster Area Percentages:")
for i, count in enumerate(np.bincount(labels_valid)):
    percent = round(count / total * 100, 2)
    print(f"Cluster {i}: {percent}%")

# =========================================
# OPTIONAL: BUILD CLUSTER MAP AGAIN
# =========================================

labels_full = np.full(X.shape[0], np.nan)
labels_full[valid_mask] = labels_valid
cluster_map = labels_full.reshape(n_lat, n_lon)

print("\nCluster map shape:", cluster_map.shape)