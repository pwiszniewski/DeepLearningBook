import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from sklearn.datasets import make_blobs
from scipy.optimize import minimize

from cvxopt import matrix, solvers

## make blobs
N = 50
X, y = make_blobs(n_samples=N, centers=2, n_features=2, random_state=1)
y[y==0] = -1

# plot examples
y_plot = y.reshape(-1, 1)
dataframe = pd.DataFrame(data=np.hstack([X,y_plot]), columns=("x", "y", "label"))
sn.FacetGrid(dataframe, hue="label", size=6).map(plt.scatter, 'x', 'y').add_legend()
plt.show()

## optimization
alphs0 = np.ones_like(y)

Q = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        Q[i][j] = y[i]*y[j]*X[i]@X[j]


Q = matrix(Q)
# p = -matrix(np.ones((N,1))/N)
p = -matrix(np.ones((N,1)))
G = -matrix(np.eye(N))
h = -matrix(np.zeros((N,1)))
A = matrix(y.reshape(1,N).astype('float64') )
b = matrix(0.0)
sol = solvers.qp(Q, p, G, h, A, b)
alphs_opt = np.array(sol['x'])

## draw decision boundary

x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1

xx = np.arange(x_min, x_max, .1)
yy = np.arange(y_min, y_max, .1)


xxx, yyy = np.meshgrid(xx, yy, sparse=False)

arr = np.zeros((len(yy),len(xx)))

# alphs_opt = np.round(res.x.reshape((-1,1)), 2)

w = np.zeros((2,1))
for i in range(len(y)):
    w += (alphs_opt[i]*y[i]*X[i]).reshape(2,1)

sup_vects = []  
for i in np.nonzero(np.round(alphs_opt, 2))[0]:
    sup_vects.append((X[i], y[i]))
    
if sup_vects[0][1] != sup_vects[1][1]:
    b = -(w.T@sup_vects[0][0] + w.T@sup_vects[1][0])[0]/2
else:
    b = -(w.T@sup_vects[0][0]  + w.T@sup_vects[2][0])[0]/2

for i in range(len(yy)):
    for j in range(len(xx)):
        arr[i][j] = (w.T@np.array([xxx[i][j], yyy[i][j]]) + b) > 0
        # arr[i][j] = ((alphs_opt[i]*y[i]*X[i]@np.array([xxx[i][j], yyy[i][j]])) + b) > 0
    
    
dataframe = pd.DataFrame(data=np.hstack([np.vstack((xxx.ravel(), yyy.ravel())).T, arr.ravel().reshape(-1, 1)]), columns=("x", "y", "label"))
sn.FacetGrid(dataframe, hue="label", palette='pastel', size=6).map(plt.scatter, 'x', 'y').add_legend()
dataframe = pd.DataFrame(data=np.hstack([X,y_plot]), columns=("x", "y", "label"))
sn.FacetGrid(dataframe, hue="label", size=6).map(plt.scatter, 'x', 'y').add_legend()

for sp in sup_vects:
    plt.scatter(sp[0][0], sp[0][1], color='r')

# plt.scatter(w[0], w[1], color='green', s=100)
# x_min, x_max = plt.xlim()
# y_min, y_max = plt.ylim()
plt.plot((x_min, x_max), ((-b-w[0]*x_min)/w[1], (-b-w[0]*x_max)/w[1]), color='red')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

