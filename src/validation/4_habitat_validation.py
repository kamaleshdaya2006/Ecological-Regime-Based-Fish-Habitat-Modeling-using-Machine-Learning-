import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# LOAD DATA
# -----------------------
ds = xr.open_dataset("ensemble_monthly_habitat_index.nc")

hsi = ds["ensemble_habitat_index"].values  # (month, lat, lon)

print("HSI shape:", hsi.shape)

# -----------------------
# 1️⃣ AREA FRACTION
# -----------------------
area_fraction = []

for m in range(12):
    month = hsi[m]

    high = np.sum(month > 0.6)
    total = np.sum(~np.isnan(month))

    frac = high / total
    area_fraction.append(frac)

# -----------------------
# PLOT AREA FRACTION
# -----------------------
plt.figure()
plt.plot(range(1, 13), area_fraction, marker='o')
plt.xlabel("Month")
plt.ylabel("High Habitat Fraction")
plt.title("Seasonal Habitat Area")
plt.grid()

plt.savefig("habitat_area.png")
plt.show()

print("Saved: habitat_area.png")

# -----------------------
# 2️⃣ HABITAT PERSISTENCE
# -----------------------
# how often each pixel is high habitat

high_mask = hsi > 0.6
persistence = np.sum(high_mask, axis=0) / 12

# -----------------------
# PLOT PERSISTENCE MAP
# -----------------------
plt.figure()
plt.imshow(persistence, origin="lower")
plt.colorbar(label="Persistence (fraction of year)")
plt.title("Habitat Persistence Map")

plt.savefig("habitat_persistence.png")
plt.show()

print("Saved: habitat_persistence.png")