import xarray as xr
import numpy as np
import pandas as pd
import os
from sklearn.mixture import GaussianMixture

# -----------------------
# LOAD DATA
# -----------------------
folder = "processed_monthly_0p25"

anom = xr.open_dataset(os.path.join(folder, "marine_anomalies_standardized.nc"))

# -----------------------
# BUILD FEATURE MATRIX
# -----------------------
feature_list = []
feature_names = []

for var in anom.data_vars:
    feature_list.append(anom[var].mean("time"))
    feature_names.append(f"{var}_mean")

    feature_list.append(anom[var].std("time"))
    feature_names.append(f"{var}_std")

feature_stack = xr.concat(feature_list, dim="feature")

X = feature_stack.values.reshape(len(feature_list), -1).T

valid_mask = ~np.isnan(X).any(axis=1)
X_valid = X[valid_mask]

print("Valid samples:", X_valid.shape)

# -----------------------
# FIT GMM (same as model)
# -----------------------
best_k = 6

gmm = GaussianMixture(n_components=best_k, random_state=42)
labels_valid = gmm.fit_predict(X_valid)

# -----------------------
# REBUILD FULL LABEL ARRAY
# -----------------------
labels_full = np.full(X.shape[0], -1)
labels_full[valid_mask] = labels_valid

# -----------------------
# CREATE DATAFRAME
# -----------------------
df = pd.DataFrame(X, columns=feature_names)
df["cluster"] = labels_full

df = df[df["cluster"] != -1]

# -----------------------
# COMPUTE CLUSTER MEANS
# -----------------------
cluster_means = df.groupby("cluster").mean()

print("\nCluster-wise Feature Means:\n")
print(cluster_means)

# -----------------------
# SAVE OUTPUT
# -----------------------
cluster_means.to_csv("cluster_physical_characteristics.csv")

print("\nSaved: cluster_physical_characteristics.csv")