import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

n_samples = 500
random_state = 127
X, y = make_blobs(n_samples=n_samples, random_state=random_state, centers=1)

# Anisotropicly distributed data
transformation = [[0.60834549, -0.63667341], [-0.40887718, 1.85253229]]
X_aniso = np.dot(X, transformation)
X_aniso[:,1] = X_aniso[:,1] - np.mean(X_aniso[:,1])
#y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

plt.scatter(X_aniso[:, 0], X_aniso[:, 1])
plt.title("Anisotropicly Distributed Blobs")

