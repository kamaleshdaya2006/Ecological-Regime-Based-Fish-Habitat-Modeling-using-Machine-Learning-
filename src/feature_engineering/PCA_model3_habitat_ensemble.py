import numpy as np
import xarray as xr
from sklearn.mixture import GaussianMixture

# =====================================
# LOAD DATA
# =====================================

X_cluster = np.load("pca_features.npy")   # (N_valid, 3)
valid_mask = np.load("valid_mask.npy")    # (time*lat*lon,)
grid_shape = np.load("grid_shape.npy")

n_time, n_lat, n_lon = grid_shape

best_k = 6
n_ensemble = 20

print("X_cluster:", X_cluster.shape)
print("Valid mask:", valid_mask.shape)
print("Grid shape:", (n_time, n_lat, n_lon))

# =====================================
# SAFETY CHECK (MANDATORY)
# =====================================

if X_cluster.shape[0] != np.sum(valid_mask):
    raise ValueError("❌ Data mismatch — PCA and mask are not aligned")

# =====================================
# USE PC1 AS ECOLOGICAL AXIS
# =====================================

pc1 = X_cluster[:, 0]

# normalize safely
pc1_min, pc1_max = pc1.min(), pc1.max()
pc1 = (pc1 - pc1_min) / (pc1_max - pc1_min + 1e-10)

# =====================================
# ENSEMBLE GMM + IMPROVED HSI
# =====================================

hsi_ensemble = []

for seed in range(n_ensemble):

    print("Ensemble member:", seed)

    gmm = GaussianMixture(
        n_components=best_k,
        covariance_type="full",
        random_state=seed
    )

    gmm.fit(X_cluster)

    probs = gmm.predict_proba(X_cluster)   # (N_valid, k)

    # =====================================
    # IMPROVED HSI (CLUSTER-AWARE)
    # =====================================

    # take PC1 mean of each cluster
    cluster_means = gmm.means_[:, 0]

    # normalize cluster means
    cm_min, cm_max = cluster_means.min(), cluster_means.max()
    cluster_means = (cluster_means - cm_min) / (cm_max - cm_min + 1e-10)

    # compute weighted HSI
    hsi = np.zeros(X_cluster.shape[0])

    for k in range(best_k):
        hsi += probs[:, k] * cluster_means[k]

    hsi_ensemble.append(hsi)

hsi_ensemble = np.array(hsi_ensemble)   # (ensemble, N_valid)

# =====================================
# ENSEMBLE MEAN
# =====================================

hsi_mean = np.mean(hsi_ensemble, axis=0)

# =====================================
# REBUILD FULL GRID (CORRECT WAY)
# =====================================

hsi_full = np.full(valid_mask.shape, np.nan)

hsi_full[valid_mask] = hsi_mean

hsi_map = hsi_full.reshape(n_time, n_lat, n_lon)

# =====================================
# NORMALIZE FINAL MAP
# =====================================

hsi_min = np.nanmin(hsi_map)
hsi_max = np.nanmax(hsi_map)

hsi_map = (hsi_map - hsi_min) / (hsi_max - hsi_min + 1e-10)

# =====================================
# HABITAT ZONES (QUANTILES)
# =====================================

vals = hsi_map[~np.isnan(hsi_map)]

p20 = np.percentile(vals, 20)
p40 = np.percentile(vals, 40)
p60 = np.percentile(vals, 60)
p80 = np.percentile(vals, 80)

zones = np.full_like(hsi_map, 1)

zones[hsi_map >= p20] = 2
zones[hsi_map >= p40] = 3
zones[hsi_map >= p60] = 4
zones[hsi_map >= p80] = 5

# =====================================
# SAVE OUTPUT
# =====================================

ds = xr.open_dataset("processed_monthly_0p25/marine_anomalies_standardized.nc")

zone_ds = xr.Dataset(
    {
        "fish_habitat_zone": (["time", "latitude", "longitude"], zones),
        "hsi": (["time", "latitude", "longitude"], hsi_map)
    },
    coords={
        "time": ds.time,
        "latitude": ds.latitude,
        "longitude": ds.longitude
    }
)

zone_ds.to_netcdf("pca_habitat_zones_ensemble.nc")

print("✅ FINAL OUTPUT SAVED SUCCESSFULLY")
