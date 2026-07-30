import xarray as xr
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

folder = "processed_monthly_0p25"

anom = xr.open_dataset(os.path.join(folder, "marine_anomalies_standardized.nc"))

# -----------------------
# BUILD FEATURE MATRIX
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
# BIC CURVE
# -----------------------
bic_scores = []
k_range = range(2, 9)

for k in k_range:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_valid)
    bic_scores.append(gmm.bic(X_valid))

plt.plot(k_range, bic_scores, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("BIC")
plt.title("BIC vs Number of Clusters")
plt.savefig("bic_curve.png")
plt.show()

# -----------------------
# FINAL MODEL (K=6)
# -----------------------
gmm = GaussianMixture(n_components=6, random_state=42)
labels = gmm.fit_predict(X_valid)

# -----------------------
# SILHOUETTE SCORE
# -----------------------
sil_score = silhouette_score(X_valid, labels)

# -----------------------
# DAVIES BOULDIN
# -----------------------
db_score = davies_bouldin_score(X_valid, labels)

print("Silhouette Score:", sil_score)
print("Davies-Bouldin Index:", db_score)