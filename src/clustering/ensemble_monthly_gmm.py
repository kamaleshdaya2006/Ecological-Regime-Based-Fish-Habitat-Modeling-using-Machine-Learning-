import xarray as xr
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import mode
import os

# ==============================
# CONFIG
# ==============================

K = 6
N_RUNS = 20
BOOTSTRAP_FRAC = 0.8
INPUT_FILE = "processed_monthly_0p25/marine_anomalies_standardized.nc"
OUTPUT_FOLDER = "ensemble_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================

ds = xr.open_dataset(INPUT_FILE)

lat = ds.latitude.values
lon = ds.longitude.values
time = ds.time.values

vars_list = list(ds.data_vars)

# ==============================
# FEATURE EXTRACTION
# ==============================

def get_feature_matrix(ds_month):
    X = np.column_stack([
        ds_month[var].values.flatten()
        for var in vars_list
    ])
    valid_mask = ~np.isnan(X).any(axis=1)
    return X, valid_mask

# ==============================
# ENSEMBLE FUNCTION
# ==============================

def ensemble_month(ds_month):

    X, valid_mask = get_feature_matrix(ds_month)
    X_valid = X[valid_mask]
    n_cells = X.shape[0]

    all_labels = []
    reference_means = None

    for seed in range(N_RUNS):

        n_boot = int(len(X_valid) * BOOTSTRAP_FRAC)
        idx = np.random.choice(len(X_valid), n_boot, replace=True)
        X_boot = X_valid[idx]

        gmm = GaussianMixture(
            n_components=K,
            covariance_type="full",
            random_state=seed
        )
        gmm.fit(X_boot)

        labels_full = gmm.predict(X_valid)
        all_labels.append(labels_full)

        # Save means only from first run
        if seed == 0:
            reference_means = gmm.means_

    all_labels = np.array(all_labels)

    consensus_labels = mode(all_labels, axis=0).mode[0]
    stability = np.mean(all_labels == consensus_labels, axis=0)

    full_labels = np.full(n_cells, np.nan)
    full_stability = np.full(n_cells, np.nan)

    full_labels[valid_mask] = consensus_labels
    full_stability[valid_mask] = stability

    return full_labels, full_stability, reference_means

# ==============================
# RUN FOR ALL MONTHS
# ==============================

all_regimes = []
all_stability = []
all_means = []

for t in time:
    print("Processing:", str(t))
    ds_month = ds.sel(time=t)

    labels, stability, means = ensemble_month(ds_month)

    all_regimes.append(labels.reshape(len(lat), len(lon)))
    all_stability.append(stability.reshape(len(lat), len(lon)))
    all_means.append(means)

all_regimes = np.array(all_regimes)
all_stability = np.array(all_stability)
all_means = np.array(all_means)

# ==============================
# SAVE RESULTS
# ==============================

regime_ds = xr.Dataset(
    {
        "ensemble_regime": (["time", "latitude", "longitude"], all_regimes),
        "stability": (["time", "latitude", "longitude"], all_stability),
    },
    coords={
        "time": time,
        "latitude": lat,
        "longitude": lon,
    },
)

regime_ds.to_netcdf(os.path.join(OUTPUT_FOLDER, "ensemble_regimes.nc"))

# Save means separately
np.save(
    os.path.join(OUTPUT_FOLDER, "ensemble_means.npy"),
    all_means
)

print("Ensemble complete and means saved.")