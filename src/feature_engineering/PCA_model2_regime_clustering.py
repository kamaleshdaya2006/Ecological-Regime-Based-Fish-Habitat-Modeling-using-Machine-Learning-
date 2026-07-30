import numpy as np
import xarray as xr
from sklearn.mixture import GaussianMixture

# =====================================
# LOAD PCA FEATURES
# =====================================

X_cluster = np.load("pca_features.npy")
valid_mask = np.load("valid_mask.npy")

best_k = 6
n_ensemble = 20

# =====================================
# LOAD ORIGINAL GRID
# =====================================

ds = xr.open_dataset("processed_monthly_0p25/marine_anomalies_standardized.nc")

n_time = ds.time.size
n_lat = ds.latitude.size
n_lon = ds.longitude.size

# =====================================
# ENSEMBLE CLUSTERING
# =====================================

labels_ensemble = []

for seed in range(n_ensemble):

    print("Running ensemble member:", seed)

    gmm = GaussianMixture(
        n_components=best_k,
        covariance_type="full",
        random_state=seed
    )

    labels = gmm.fit_predict(X_cluster)

    labels_ensemble.append(labels)

labels_ensemble = np.array(labels_ensemble)

# =====================================
# MAJORITY VOTE ACROSS ENSEMBLES
# =====================================

from scipy.stats import mode

labels_final = mode(labels_ensemble, axis=0)[0][0]

# =====================================
# REBUILD FULL GRID
# =====================================

labels_full = np.full(valid_mask.shape, np.nan)

labels_full[valid_mask] = labels_final

cluster_map = labels_full.reshape(
    n_time,
    n_lat,
    n_lon
)

# =====================================
# SAVE DATASET
# =====================================

cluster_ds = xr.Dataset(
    {"ecological_regime": (["time","latitude","longitude"], cluster_map)},
    coords={
        "time": ds.time,
        "latitude": ds.latitude,
        "longitude": ds.longitude
    }
)

cluster_ds.to_netcdf("pca_ecological_regimes_ensemble.nc")

print("Ensemble regime map saved")