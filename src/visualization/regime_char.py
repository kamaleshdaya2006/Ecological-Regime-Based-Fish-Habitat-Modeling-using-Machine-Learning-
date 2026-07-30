import xarray as xr
import numpy as np
import os

folder = "processed_monthly_0p25"

# Load original (non-anomaly) dataset
ds = xr.open_dataset(os.path.join(folder, "marine_2015_2024_monthly_0p25.nc"))

# Load anomaly dataset again to rebuild cluster map
anom = xr.open_dataset(os.path.join(folder, "marine_anomalies_standardized.nc"))

# --- Rebuild feature matrix ---
feature_list = []
for var in anom.data_vars:
    feature_list.append(anom[var].mean("time"))
    feature_list.append(anom[var].std("time"))

feature_stack = xr.concat(feature_list, dim="feature")
n_features = len(feature_list)
n_lat = feature_stack.latitude.size
n_lon = feature_stack.longitude.size

X = feature_stack.values.reshape(n_features, -1).T
valid_mask = ~np.isnan(X).any(axis=1)

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=6, random_state=42)
labels_valid = gmm.fit_predict(X[valid_mask])

labels_full = np.full(X.shape[0], np.nan)
labels_full[valid_mask] = labels_valid
cluster_map = labels_full.reshape(n_lat, n_lon)

# --- Compute long-term mean physical state ---
mean_fields = ds.mean("time")

cluster_ids = np.unique(labels_valid)

print("\n=== Physical Characteristics of Each Regime ===")

for cid in cluster_ids:
    mask = (cluster_map == cid)

    print(f"\nCluster {cid}")
    for var in mean_fields.data_vars:
        val = mean_fields[var].values[mask].mean()
        print(f"{var}: {round(float(val),3)}")