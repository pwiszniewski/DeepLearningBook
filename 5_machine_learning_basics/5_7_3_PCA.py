# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.datasets import make_blobs

# n_samples = 500
# random_state = 12

# # X, y = make_blobs(n_samples=n_samples, random_state=random_state, centers=1)
# # transformation = [[0.60834549, -0.63667341], [-0.40887718, 1.85253229]]
# # X = np.dot(X, transformation)
# # X[:,1] = X[:,1] - np.mean(X[:,1])

# rng = np.random.RandomState(random_state)
# cov = [[3, 3],
#         [3, 4]]
# X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)

# plt.scatter(X[:, 0], X[:, 1])
# plt.title("Anisotropicly Distributed Blobs")
# plt.show()

# eigval, eigvect = np.linalg.eig(X.T@X)

# D = eigvect
# print(D[0]@D[1])
# X_enc = X @ D
# X_dec = X_enc @ D.T
# plt.scatter(X_dec[:, 0], X_dec[:, 1])
# plt.title("Reconstructed - 2 eigenvencors")
# plt.show()

# D = eigvect[0].reshape(-1,1)
# X_enc = X @ D
# X_dec = X @ D @ D.T
# plt.scatter(X_dec[:, 0], X_dec[:, 1])
# plt.title(f"Reconstructed - 1 eigenvencors (eigenvalue: {eigval[0]:.1f})")
# plt.show()


# D = eigvect[1].reshape(-1,1)
# X_enc = X @ D
# X_dec = X @ D @ D.T
# plt.scatter(X_dec[:, 0], X_dec[:, 1])
# plt.title(f"Reconstructed - 1 eigenvencors (eigenvalue: {eigval[1]:.1f})")
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

rng = np.random.RandomState(0)
n_samples = 500
cov = [[3, 3],
       [3, 4]]
X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)
pca = PCA(n_components=2).fit(X)


plt.scatter(X[:, 0], X[:, 1], alpha=.3, label='samples')
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    comp = comp * var  # scale component by its variance explanation power
    plt.plot([0, comp[0]], [0, comp[1]], label=f"Component {i}", linewidth=5,
             color=f"C{i + 2}")
plt.gca().set(aspect='equal',
              title="2-dimensional dataset with principal components",
              xlabel='first feature', ylabel='second feature')
plt.legend()
plt.show()

y = X.dot(pca.components_[1]) + rng.normal(size=n_samples) / 2

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

axes[0].scatter(X.dot(pca.components_[0]), y, alpha=.3)
axes[0].set(xlabel='Projected data onto first PCA component', ylabel='y')
axes[1].scatter(X.dot(pca.components_[1]), y, alpha=.3)
axes[1].set(xlabel='Projected data onto second PCA component', ylabel='y')
plt.tight_layout()
plt.show()


eigval, eigvect = np.linalg.eig(X.T@X)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].scatter(X.dot(eigvect[0]), y, alpha=.3)
axes[0].set(xlabel='Projected data onto first PCA component', ylabel='y')
axes[1].scatter(X.dot(eigvect[1]), y, alpha=.3)
axes[1].set(xlabel='Projected data onto second PCA component', ylabel='y')
plt.tight_layout()
plt.show()
