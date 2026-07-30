import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

X_pca = np.load("X_pca.npy")

gmm = GaussianMixture(n_components=6, random_state=42)
labels = gmm.fit_predict(X_pca)

sil = silhouette_score(X_pca, labels)
db = davies_bouldin_score(X_pca, labels)

print("PCA Silhouette:", sil)
print("PCA DB Index:", db)