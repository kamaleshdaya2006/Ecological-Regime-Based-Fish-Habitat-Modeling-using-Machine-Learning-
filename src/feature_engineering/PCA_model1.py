import xarray as xr
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =====================================
# LOAD DATA
# =====================================

print("="*60)
print("Loading dataset...")
print("="*60)

folder = r"D:\Project\fish density prediction\NoAA\marine_data\All_environment_data\processed_monthly_0p25"
ds = xr.open_dataset(
    os.path.join(folder, "marine_anomalies_standardized.nc")
)

vars_list = list(ds.data_vars)

n_time = ds.time.size
n_lat = ds.latitude.size
n_lon = ds.longitude.size

print(f"Grid Size : {n_time} × {n_lat} × {n_lon}")
print(f"Variables : {len(vars_list)}")
print()

# =====================================
# BUILD FEATURE MATRIX
# =====================================

print("="*60)
print("Building feature matrix...")
print("="*60)

feature_arrays = []

for i, v in enumerate(vars_list, start=1):

    print(f"[{i}/{len(vars_list)}] Processing variable: {v}")

    data = ds[v].values
    feature_arrays.append(data.reshape(-1))

X = np.vstack(feature_arrays).T

print("\nFeature matrix created.")
print("Raw feature matrix shape:", X.shape)
print()

# =====================================
# REMOVE NAN CELLS
# =====================================

print("="*60)
print("Removing invalid samples...")
print("="*60)

valid_mask = ~np.isnan(X).any(axis=1)

X_valid = X[valid_mask]

print(f"Total samples      : {X.shape[0]:,}")
print(f"Valid samples      : {X_valid.shape[0]:,}")
print(f"Removed (NaN)      : {X.shape[0]-X_valid.shape[0]:,}")
print()

# =====================================
# STANDARDIZATION
# =====================================

print("="*60)
print("Standardizing features...")
print("="*60)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_valid)

print("Standardization completed.")
print()

# =====================================
# PCA
# =====================================

print("="*60)
print("Running Principal Component Analysis...")
print("="*60)

pca = PCA()

X_pca = pca.fit_transform(X_scaled)

print("PCA completed.\n")

print("Explained Variance Ratio")

for i, var in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"PC{i:2d} : {var:.6f}")

print()

# =====================================
# KEEP FIRST 3 PCs
# =====================================

X_cluster = X_pca[:, :3]

print("Selected Principal Components : 3")
print("Reduced Feature Matrix Shape  :", X_cluster.shape)
print()

# =====================================
# SAVE FILES
# =====================================

print("="*60)
print("Saving outputs...")
print("="*60)

np.save("pca_features.npy", X_cluster)
np.save("valid_mask.npy", valid_mask)
np.save("grid_shape.npy", np.array([n_time, n_lat, n_lon]))

np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)

print("Saved:")
print("  ✓ pca_features.npy")
print("  ✓ valid_mask.npy")
print("  ✓ grid_shape.npy")
print("  ✓ scaler_mean.npy")
print("  ✓ scaler_scale.npy")

print("\n" + "="*60)
print("PCA pipeline completed successfully!")
print("="*60)