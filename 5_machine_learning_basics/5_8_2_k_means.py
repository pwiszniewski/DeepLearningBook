from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import random 
import numpy as np

seed = 0
np.random.seed(seed=seed)
random.seed(seed)

# Generate sample data
n_samples = 4000
n_components = 4

X, y_true = make_blobs(n_samples=n_samples,
                       centers=n_components,
                       cluster_std=0.60,
                       random_state=0)

plt.scatter(X[:,0], X[:,1], c='b', s=5)
plt.xticks([])
plt.yticks([])
plt.show()

#init centers random
centers_idx = random.sample(range(0, n_samples), n_components)
centers = X[centers_idx,:]

dist = np.diag((X-centers[0]) @ (X-centers[0]).T)
dist2 = np.einsum('ijk,ilm->ijkm',X-centers[0],X-centers[0])