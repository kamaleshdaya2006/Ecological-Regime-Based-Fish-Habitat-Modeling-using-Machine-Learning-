import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

X_pca = np.load("X_pca.npy")

n_runs = 10
all_labels = []

for seed in range(n_runs):
    gmm = GaussianMixture(n_components=6, random_state=seed)
    labels = gmm.fit_predict(X_pca)
    all_labels.append(labels)

aris = []

for i in range(n_runs):
    for j in range(i+1, n_runs):
        aris.append(adjusted_rand_score(all_labels[i], all_labels[j]))

print("PCA Mean ARI:", np.mean(aris))