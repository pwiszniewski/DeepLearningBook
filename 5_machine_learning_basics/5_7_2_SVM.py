import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import torch

from sklearn.datasets import make_blobs
from scipy.optimize import minimize

from scipy.optimize import NonlinearConstraint


N = 100
X, y = make_blobs(n_samples=N, centers=2, n_features=2, random_state=1)
y[y==0] = -1

# plot examples
y_plot = y.reshape(-1, 1)
dataframe = pd.DataFrame(data=np.hstack([X,y_plot]), columns=("x", "y", "label"))
sn.FacetGrid(dataframe, hue="label", size=6).map(plt.scatter, 'x', 'y').add_legend()
plt.show()

alphs0 = np.ones_like(y)

def func(alphs, X, y):
    sum1 = sum(alphs)
    N = len(alphs)
    sum2 = 0
    for i in range(N):
        for j in range(N):
            sum2 += alphs[i]*alphs[j]*y[i]*y[j]*X[i]@X[j]
    sum2 /= 2
    ret = sum1 - sum2
    print(ret)
    return - ret

bounds = [[0, np.inf] for _ in range(N)]

def con(alphs):
    return alphs.T @ y
# nlc = NonlinearConstraint(con, 0, 0)
cons = {'type':'eq', 'fun': con}

# res = minimize(func, alphs0, (X, y), tol=1e-2, bounds=bounds)
res = minimize(func, alphs0, (X, y), tol=1e-5, bounds=bounds, constraints=cons)
# res = minimize(func, alphs0, (X, y), tol=1e-2)

xx = np.arange(-12, 2, 0.1)
yy = np.arange(-7, 7, 0.1)

xxx, yyy = np.meshgrid(xx, yy, sparse=False)

arr = np.zeros((140,140))

alphs_opt = res.x.reshape((-1,1))

for i in range(len(y)):
    for j in range(len(y)):
        arr[i][j] = (alphs_opt[i]*y[i]*X[i]@np.array([xxx[i][j], yyy[i][j]])) > 0

dataframe = pd.DataFrame(data=np.hstack([np.vstack((xxx.ravel(), yyy.ravel())).T, arr.ravel().reshape(-1, 1)]), columns=("x", "y", "label"))
sn.FacetGrid(dataframe, hue="label", size=6).map(plt.scatter, 'x', 'y').add_legend()
plt.show()

# y_hat = []

# for i in range(len(y)):
#     print(alphs[i]*y[i]*X[[i]]@X[i])

# for i in range(len(y)):
#     y_hat.append(alphs[i]*y[i]*X[[i]]@X[i])