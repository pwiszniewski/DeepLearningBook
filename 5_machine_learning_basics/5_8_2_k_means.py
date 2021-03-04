from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import random 
import numpy as np
from sklearn.cluster import kmeans_plusplus

seed = 3
np.random.seed(seed=seed)
random.seed(seed)

# Generate sample data
n_samples = 4000
n_clusters = 4

X, y_true = make_blobs(n_samples=n_samples,
                       centers=n_clusters,
                       cluster_std=0.60,
                       random_state=seed)
plt.scatter(X[:,0], X[:,1], c=y_true, s=5)
plt.xticks([])
plt.yticks([])
plt.show()

#init random centers 
centers_idx = random.sample(range(0, n_samples), n_clusters)
centers = X[centers_idx,:]

# calculate distances
eps = 1e-5
max_iter = 100
dists = np.zeros((n_samples, n_clusters))

for it in range(max_iter):
    for i in range(n_clusters):
        dists[:,i] = np.diag((X-centers[i]) @ (X-centers[i]).T)
        
    y_pred = np.argmin(dists, axis=1)
    
    new_centers = np.zeros_like(centers)
    for i in range(n_clusters):
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


# init centers with kmeans++
centers, indices = kmeans_plusplus(X, n_clusters=4,
                                        random_state=seed)
# calculate distances
eps = 1e-5
max_iter = 100
dists = np.zeros((n_samples, n_clusters))

for it in range(max_iter):
    for i in range(n_clusters):
        dists[:,i] = np.diag((X-centers[i]) @ (X-centers[i]).T)
        
    y_pred = np.argmin(dists, axis=1)
    
    new_centers = np.zeros_like(centers)
    for i in range(n_clusters):
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