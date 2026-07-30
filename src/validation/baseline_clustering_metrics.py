import xarray as xr
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

# -----------------------
# LOAD STANDARDIZED DATA
# -----------------------
ds = xr.open_dataset(r"D:\Project\fish density prediction\NoAA\marine_data\All_environment_data\processed_monthly_0p25\marine_anomalies_standardized.nc")

# -----------------------
# FEATURE MATRIX (same as your clustering)
# -----------------------
features = []
for var in ds.data_vars:
    features.append(ds[var].mean("time"))
    features.append(ds[var].std("time"))

X = xr.concat(features, dim="feature").values
X = X.reshape(len(features), -1).T

# Remove NaNs
mask = ~np.isnan(X).any(axis=1)
X_valid = X[mask]

# -----------------------
# GMM CLUSTERING
# -----------------------
gmm = GaussianMixture(n_components=6, random_state=42)
labels = gmm.fit_predict(X_valid)

# -----------------------
# METRICS
# -----------------------
sil = silhouette_score(X_valid, labels)
db = davies_bouldin_score(X_valid, labels)
bic = gmm.bic(X_valid)

print("\n PURE CLUSTERING RESULTS")
print("Silhouette Score:", sil)
print("Davies-Bouldin Index:", db)
print("BIC:", bic)