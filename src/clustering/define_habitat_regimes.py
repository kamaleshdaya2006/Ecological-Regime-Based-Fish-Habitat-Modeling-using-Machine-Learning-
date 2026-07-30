import pandas as pd

df = pd.read_csv("regime_physical_centroids.csv", index_col=0)

# Compute basin medians
chl_median = df["chl"].median()
o2_median = df["o2"].median()

habitat_regimes = []

for idx, row in df.iterrows():

    if (row["chl"] > chl_median) and (row["o2"] > o2_median):
        habitat_regimes.append(idx)

print("Ecologically Favorable Regimes:", habitat_regimes)