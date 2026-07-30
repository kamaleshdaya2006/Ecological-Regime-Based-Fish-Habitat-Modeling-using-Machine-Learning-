import xarray as xr
import numpy as np
import os
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

folder = "processed_monthly_0p25"

# Load standardized anomalies
anom = xr.open_dataset(os.path.join(folder, "marine_anomalies_standardized.nc"))

# -------------------------------
# Rebuild feature matrix (same as before)
# -------------------------------

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
X_valid = X[valid_mask]

# -------------------------------
# Fit GMM once
# -------------------------------

best_k = 6
gmm = GaussianMixture(n_components=best_k, random_state=42)
gmm.fit(X_valid)

# -------------------------------
# Now build monthly regime maps
# -------------------------------

monthly_regimes = {}

for m in range(1, 13):

    # Get all years for this month
    month_data = anom.sel(time=anom["time.month"] == m)

    # Mean anomaly for that calendar month
    month_mean = month_data.mean("time")

    # Build feature matrix (same feature design)
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

    monthly_regimes[m] = cluster_map_month

print("Monthly regime maps created for 12 months.")
import pandas as pd

area_table = []

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

    labels_full = np.full(X_month.shape[0], np.nan)
    labels_full[valid_mask_month] = gmm.predict(X_month[valid_mask_month])

    cluster_map_month = labels_full.reshape(n_lat, n_lon)

    labels_flat = labels_full[valid_mask_month]
    total = len(labels_flat)

    month_stats = []
    for r in range(best_k):
        percent = np.sum(labels_flat == r) / total * 100
        month_stats.append(percent)

    area_table.append(month_stats)

area_df = pd.DataFrame(area_table,
                       columns=[f"Regime_{i}" for i in range(best_k)],
                       index=["Jan","Feb","Mar","Apr","May","Jun",
                              "Jul","Aug","Sep","Oct","Nov","Dec"])

print(area_df)
