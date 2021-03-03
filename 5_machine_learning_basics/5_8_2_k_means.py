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
plt.scatter(X[:,0], X[:,1], c=y_true, s=5)
plt.xticks([])
plt.yticks([])
plt.show()

#init random centers 
centers_idx = random.sample(range(0, n_samples), n_components)
centers = X[centers_idx,:]

# calculate distances
eps = 1e-5
max_iter = 100
dists = np.zeros((n_samples, n_components))

for it in range(max_iter):
    for i in range(n_components):
        dists[:,i] = np.diag((X-centers[i]) @ (X-centers[i]).T)
        
    y_pred = np.argmin(dists, axis=1)
    
    new_centers = np.zeros_like(centers)
    for i in range(n_components):
        new_centers[i] =  np.mean(X[y_pred == i], axis=0)
        
    if np.linalg.norm(centers-new_centers) < eps:
        centers = new_centers 
        break
    centers = new_centers
    
print(f'iters: {it}')

# plot results
plt.scatter(X[:,0], X[:,1], c=y_pred, s=5)
plt.xticks([])
plt.yticks([])
plt.show()