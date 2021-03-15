import matplotlib.pyplot as plt
import numpy as np


X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

w = np.random.rand(3, 1) # [w0, w1, b]
X = np.hstack((X, np.ones((X.shape[0], 1)))) # b constant

f = X @ w

J = np.mean(f)

w_opt = np.linalg.inv(X.T @ X) @ (X.T @ y)

# plot
plt.scatter(X[:,0], X[:,1], c=y)
xmin, xmax = plt.xlim()
plt.plot((xmin, xmax), (.5, .5))