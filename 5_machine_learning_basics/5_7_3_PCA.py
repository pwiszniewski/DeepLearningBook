import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

n_samples = 500
random_state = 12

# X, y = make_blobs(n_samples=n_samples, random_state=random_state, centers=1)
# transformation = [[0.60834549, -0.63667341], [-0.40887718, 1.85253229]]
# X = np.dot(X, transformation)
# X[:,1] = X[:,1] - np.mean(X[:,1])

rng = np.random.RandomState(random_state)
cov = [[3, 3],
        [3, 4]]
X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)

plt.scatter(X[:, 0], X[:, 1])
plt.title("Anisotropicly Distributed Blobs")
plt.show()

eigval, eigvect = np.linalg.eig(X.T@X)

D = eigvect
print(D[0]@D[1])
X_enc = X @ D
X_dec = X_enc @ D.T
plt.scatter(X_dec[:, 0], X_dec[:, 1])
plt.title("Reconstructed - 2 eigenvencors")
plt.show()

D = eigvect[0].reshape(-1,1)
X_enc = X @ D
X_dec = X @ D @ D.T
plt.scatter(X_dec[:, 0], X_dec[:, 1])
plt.title(f"Reconstructed - 1 eigenvencors (eigenvalue: {eigval[0]:.1f})")
plt.show()


D = eigvect[1].reshape(-1,1)
X_enc = X @ D
X_dec = X @ D @ D.T
plt.scatter(X_dec[:, 0], X_dec[:, 1])
plt.title(f"Reconstructed - 1 eigenvencors (eigenvalue: {eigval[1]:.1f})")
plt.show()

