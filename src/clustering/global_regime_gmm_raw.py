import xarray as xr
import numpy as np
from sklearn.mixture import GaussianMixture
import os

# ==========================
# CONFIG
# ==========================

K = 6
INPUT_FILE = "processed_monthly_0p25/marine_2015_2024_monthly_0p25.nc"
OUTPUT_FOLDER = "global_raw_regime_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================
# LOAD DATA
# ==========================

ds = xr.open_dataset(INPUT_FILE)

lat = ds.latitude.values
lon = ds.longitude.values
time = ds.time.values

vars_list = list(ds.data_vars)

print("Variables:", vars_list)
# Compute persistent spatial mask
valid_space_mask = None

for t in time:
    ds_month = ds.sel(time=t)
    X_month = np.column_stack([
        ds_month[var].values.flatten()
        for var in vars_list
    ])
    valid = ~np.isnan(X_month).any(axis=1)

    if valid_space_mask is None:
        valid_space_mask = valid
    else:
        valid_space_mask = valid_space_mask & valid

print("Persistent spatial cells:", np.sum(valid_space_mask))
# ==========================
# STACK ALL MONTHS
# ==========================
# ==========================
# STACK ALL MONTHS USING PERSISTENT MASK
# ==========================

X_list = []

for t in time:
    ds_month = ds.sel(time=t)
    X_month = np.column_stack([
        ds_month[var].values.flatten()
        for var in vars_list
    ])
    X_list.append(X_month[valid_space_mask])

X_all = np.vstack(X_list)

print("Total samples after stacking:", X_all.shape)

# ==========================
# GLOBAL SCALING
# ==========================

mean = X_all.mean(axis=0)
std = X_all.std(axis=0)

X_scaled = (X_all - mean) / std

print("Scaled mean (should be ~0):", X_scaled.mean(axis=0))
print("Scaled std (should be ~1):", X_scaled.std(axis=0))

# ==========================
# FIT GMM
# ==========================

print("Fitting GMM...")

gmm = GaussianMixture(
    n_components=K,
    covariance_type="full",
    random_state=42,
    max_iter=200
)

gmm.fit(X_scaled)

labels = gmm.predict(X_scaled)

print("GMM complete.")

# ==========================
# CHECK CLUSTER DISTRIBUTION
# ==========================

unique, counts = np.unique(labels, return_counts=True)
print("Cluster counts:", dict(zip(unique, counts)))

# ==========================
# RECONSTRUCT FULL GRID
# ==========================

n_time = len(time)
n_lat = len(lat)
n_lon = len(lon)

# Prepare full 3D array
labels_reshaped = np.full((n_time, n_lat * n_lon), np.nan)

# Fill persistent spatial indices for each month
start = 0
cells_per_month = np.sum(valid_space_mask)

for t_idx in range(n_time):
    end = start + cells_per_month
    labels_reshaped[t_idx, valid_space_mask] = labels[start:end]
    start = end

labels_reshaped = labels_reshaped.reshape(n_time, n_lat, n_lon)

# ==========================
# SAVE RESULTS
# ==========================

regime_ds = xr.Dataset(
    {
        "global_regime": (["time", "latitude", "longitude"], labels_reshaped),
    },
    coords={
        "time": time,
        "latitude": lat,
        "longitude": lon,
    },
)

regime_ds.to_netcdf(
    os.path.join(OUTPUT_FOLDER, "global_regimes_raw.nc")
)

np.save(
    os.path.join(OUTPUT_FOLDER, "gmm_means_scaled.npy"),
    gmm.means_
)

print("Global raw regime model saved.")