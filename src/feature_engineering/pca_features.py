import xarray as xr
import numpy as np
import os
from sklearn.decomposition import PCA

folder = "processed_monthly_0p25"

anom = xr.open_dataset(os.path.join(folder, "marine_anomalies_standardized.nc"))

# -----------------------
# BUILD ORIGINAL FEATURES
# -----------------------
feature_list = []

for var in anom.data_vars:
    feature_list.append(anom[var].mean("time"))
    feature_list.append(anom[var].std("time"))

feature_stack = xr.concat(feature_list, dim="feature")

X = feature_stack.values.reshape(len(feature_list), -1).T

valid_mask = ~np.isnan(X).any(axis=1)
X_valid = X[valid_mask]

# -----------------------
# PCA
# -----------------------
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_valid)

print("Explained variance:", pca.explained_variance_ratio_)
print("Total variance:", np.sum(pca.explained_variance_ratio_))

# save
np.save("X_pca.npy", X_pca)
np.save("valid_mask.npy", valid_mask)

print("Saved PCA features")