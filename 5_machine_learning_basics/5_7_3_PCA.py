import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

n_samples = 500
random_state = 127
X, y = make_blobs(n_samples=n_samples, random_state=random_state, centers=1)

# Anisotropicly distributed data
transformation = [[0.60834549, -0.63667341], [-0.40887718, 1.85253229]]
X = np.dot(X, transformation)
X[:,1] = X[:,1] - np.mean(X[:,1])
#y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

plt.scatter(X[:, 0], X[:, 1])
plt.title("Anisotropicly Distributed Blobs")
plt.show()

eigval, eigvect = np.linalg.eig(X.T@X)

D = eigvect
X_pca = X @ D
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("Reconstructed - 2 eigenvencors")
plt.show()

D = eigvect[0]
X_pca = X @ D
plt.scatter(list(range(len(X_pca))), X_pca)
plt.title(f"Reconstructed - 1 eigenvencors (eigenvalue: {eigval[0]:.1f})")
plt.show()

D = eigvect[1]
X_pca = X @ D
plt.scatter(list(range(len(X_pca))), X_pca)
plt.title(f"Reconstructed - 1 eigenvencors (eigenvalue: {eigval[1]:.1f})")
plt.show()

