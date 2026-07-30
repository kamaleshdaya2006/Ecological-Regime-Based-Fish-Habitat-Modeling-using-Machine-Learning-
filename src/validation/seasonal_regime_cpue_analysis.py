import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# FILE PATHS
# =========================================================

regime_path = "processed_monthly_0p25/ecological_regimes_map.nc"
cpue_path = r"D:\Project\fish density prediction\CPUE\CPUE_2015_2024_5deg_monthly.csv"

# =========================================================
# LOAD REGIME MAP
# =========================================================

regime_ds = xr.open_dataset(regime_path)
cluster_map = regime_ds["cluster"]

# =========================================================
# LOAD CPUE
# =========================================================

cpue = pd.read_csv(cpue_path)

print("Columns in CPUE file:", cpue.columns)

# Use correct column name
cpue = cpue[["YEAR","MONTH_START","lat","lon","CPUE","EFFORT"]]

# Rename MONTH_START to MONTH for clarity
cpue = cpue.rename(columns={"MONTH_START": "MONTH"})

# Remove negative CPUE
cpue["CPUE"] = cpue["CPUE"].clip(lower=0)

print("Years available:", sorted(cpue["YEAR"].unique()))
print("Months available:", sorted(cpue["MONTH"].unique()))

# =========================================================
# FUNCTION TO GET DOMINANT REGIME
# =========================================================

def dominant_regime(lat_center, lon_center):
    lat_min = lat_center - 2.5
    lat_max = lat_center + 2.5
    lon_min = lon_center - 2.5
    lon_max = lon_center + 2.5

    sub = cluster_map.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max)
    )

    values = sub.values.flatten()
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan

    return int(pd.Series(values).mode()[0])

# =========================================================
# ASSIGN REGIME
# =========================================================

cpue["regime"] = cpue.apply(
    lambda row: dominant_regime(row["lat"], row["lon"]),
    axis=1
)

cpue = cpue.dropna(subset=["regime"])
cpue["regime"] = cpue["regime"].astype(int)

print("Assigned regimes successfully.")

# =========================================================
# SEASONAL CPUE
# =========================================================

season_cpue = cpue.groupby(["MONTH","regime"])["CPUE"].median().unstack()

print("\n=== Seasonal Mean CPUE by Regime ===")
print(season_cpue)

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(10,6))

for r in season_cpue.columns:
    plt.plot(season_cpue.index,
             season_cpue[r],
             marker='o',
             label=f"Regime {r}")

plt.xlabel("Month")
plt.ylabel("Mean CPUE")
plt.title("Seasonal CPUE Variation by Ecological Regime")
plt.legend()
plt.xticks(range(1,13))
plt.grid(True, alpha=0.3)
plt.show()

# Log version
cpue["log_CPUE"] = np.log1p(cpue["CPUE"])
season_log = cpue.groupby(["MONTH","regime"])["log_CPUE"].mean().unstack()

plt.figure(figsize=(10,6))

for r in season_log.columns:
    plt.plot(season_log.index,
             season_log[r],
             marker='o',
             label=f"Regime {r}")

plt.xlabel("Month")
plt.ylabel("Mean log(1+CPUE)")
plt.title("Seasonal log-CPUE Variation by Ecological Regime")
plt.legend()
plt.xticks(range(1,13))
plt.grid(True, alpha=0.3)
plt.show()
effort_table = cpue.groupby(["MONTH","regime"])["EFFORT"].sum().unstack()