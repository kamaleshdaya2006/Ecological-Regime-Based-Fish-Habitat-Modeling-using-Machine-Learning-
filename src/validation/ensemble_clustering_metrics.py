import xarray as xr
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score

# -----------------------
# LOAD DATA
# -----------------------
ds = xr.open_dataset(r"D:\Project\fish density prediction\NoAA\marine_data\All_environment_data\processed_monthly_0p25\marine_anomalies_standardized.nc")

# -----------------------
# FEATURE MATRIX
# -----------------------
features = []
for var in ds.data_vars:
    features.append(ds[var].mean("time"))
    features.append(ds[var].std("time"))

X = xr.concat(features, dim="feature").values
X = X.reshape(len(features), -1).T

mask = ~np.isnan(X).any(axis=1)
X_valid = X[mask]

# -----------------------
# ENSEMBLE GMM
# -----------------------
n_runs = 20
labels_list = []

for i in range(n_runs):
    idx = np.random.choice(len(X_valid), int(0.8 * len(X_valid)), replace=False)
    X_sample = X_valid[idx]

    gmm = GaussianMixture(n_components=6, random_state=i)
    gmm.fit(X_sample)

    labels_full = gmm.predict(X_valid)
    labels_list.append(labels_full)

labels_array = np.array(labels_list)

# -----------------------
# CONSENSUS LABEL
# -----------------------
from scipy.stats import mode
consensus_labels = mode(labels_array, axis=0)[0].flatten()

# -----------------------
# METRICS
# -----------------------
sil = silhouette_score(X_valid, consensus_labels)
db = davies_bouldin_score(X_valid, consensus_labels)

# ARI between runs
ari_scores = []
for i in range(n_runs):
    for j in range(i+1, n_runs):
        ari = adjusted_rand_score(labels_array[i], labels_array[j])
        ari_scores.append(ari)

print("\n PURE ENSEMBLE RESULTS")
print("Silhouette Score:", sil)
print("Davies-Bouldin Index:", db)
print("Mean ARI:", np.mean(ari_scores))
print("Min ARI:", np.min(ari_scores))
print("Max ARI:", np.max(ari_scores))