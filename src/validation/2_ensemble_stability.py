import xarray as xr
import numpy as np
import os
from sklearn.mixture import GaussianMixture

# -----------------------
# LOAD DATA
# -----------------------
folder = "processed_monthly_0p25"

anom = xr.open_dataset(os.path.join(folder, "marine_anomalies_standardized.nc"))

# -----------------------
# BUILD FEATURE MATRIX (SAME AS TRAINING)
# -----------------------
feature_list = []

for var in anom.data_vars:
    feature_list.append(anom[var].mean("time"))
    feature_list.append(anom[var].std("time"))

feature_stack = xr.concat(feature_list, dim="feature")

X = feature_stack.values.reshape(len(feature_list), -1).T

valid_mask = ~np.isnan(X).any(axis=1)
X_valid = X[valid_mask]

print("Total valid samples:", X_valid.shape)

# -----------------------
# ENSEMBLE GMM
# -----------------------
best_k = 6
n_runs = 20

all_labels = []

for seed in range(n_runs):
    gmm = GaussianMixture(n_components=best_k, random_state=seed)
    labels = gmm.fit_predict(X_valid)
    all_labels.append(labels)

all_labels = np.array(all_labels)

# -----------------------
# COMPUTE STABILITY
# -----------------------
stability = []

for i in range(all_labels.shape[1]):
    counts = np.bincount(all_labels[:, i])
    stability.append(np.max(counts) / n_runs)

stability = np.array(stability)

print("Mean Stability:", stability.mean())
print("Min Stability:", stability.min())
print("Max Stability:", stability.max())