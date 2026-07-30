import xarray as xr
import numpy as np

K = 6

# IMPORTANT: Use aligned regimes
ds = xr.open_dataset("ensemble_output/aligned_regimes.nc")
regimes = ds["aligned_regime"].values

# Check first two months
r1 = regimes[0].flatten()
r2 = regimes[1].flatten()

valid = ~np.isnan(r1) & ~np.isnan(r2)
change_rate = np.mean(r1[valid] != r2[valid])

print("Fraction changed:", change_rate)

# Transition matrix
transition = np.zeros((K, K))

for t in range(len(regimes) - 1):
    r1 = regimes[t].flatten()
    r2 = regimes[t+1].flatten()

    valid = ~np.isnan(r1) & ~np.isnan(r2)

    for i in range(K):
        for j in range(K):
            transition[i, j] += np.sum(
                (r1[valid] == i) & (r2[valid] == j)
            )

transition = transition / transition.sum(axis=1, keepdims=True)

print("Transition matrix:")
print(transition)