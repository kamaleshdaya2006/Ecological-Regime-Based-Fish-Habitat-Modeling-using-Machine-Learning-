import xarray as xr
import numpy as np
import os
from sklearn.mixture import GaussianMixture

# ===============================
# SETTINGS
# ===============================

folder = "processed_monthly_0p25"
best_k = 6

# ===============================
# LOAD DATA
# ===============================

anom = xr.open_dataset(os.path.join(folder, "marine_anomalies_standardized.nc"))

lats = anom.latitude.values
lons = anom.longitude.values
n_lat = len(lats)
n_lon = len(lons)

# ===============================
# REBUILD GLOBAL FEATURE MATRIX
# ===============================

feature_list = []
for var in anom.data_vars:
    feature_list.append(anom[var].mean("time"))
    feature_list.append(anom[var].std("time"))

feature_stack = xr.concat(feature_list, dim="feature")

X_global = feature_stack.values.reshape(len(feature_list), -1).T
valid_mask = ~np.isnan(X_global).any(axis=1)
X_valid = X_global[valid_mask]

# ===============================
# FIT GLOBAL ENSEMBLE GMM
# (Or load if already saved)
# ===============================

gmm = GaussianMixture(n_components=best_k, random_state=42)
gmm.fit(X_valid)

# ===============================
# DEFINE CLUSTER SUITABILITY
# ===============================

cluster_scores = {
    0: 0.85,
    1: 0.32,
    2: 0.61,
    3: 0.12,
    4: 0.74,
    5: 0.45
}

# ===============================
# MONTHLY FISH ZONES (SOFT PROBABILITY)
# ===============================

monthly_hsi = []

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

    probs = np.zeros((X_month.shape[0], best_k))

    probs[valid_mask_month] = gmm.predict_proba(X_month[valid_mask_month])

    # Compute continuous habitat suitability
    hsi_flat = np.zeros(X_month.shape[0])

    for k in range(best_k):
        hsi_flat += probs[:, k] * cluster_scores[k]

    hsi_map = hsi_flat.reshape(n_lat, n_lon)

    monthly_hsi.append(hsi_map)

monthly_hsi = np.array(monthly_hsi)

# ===============================
# SAVE DATASET
# ===============================

ds_out = xr.Dataset(
    {"ensemble_habitat_index": (["month", "latitude", "longitude"], monthly_hsi)},
    coords={
        "month": np.arange(1,13),
        "latitude": lats,
        "longitude": lons
    }
)

ds_out.to_netcdf("ensemble_monthly_habitat_index.nc")

print("Ensemble monthly habitat index saved.")