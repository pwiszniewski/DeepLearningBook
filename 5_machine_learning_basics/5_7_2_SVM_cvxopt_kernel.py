import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from sklearn.datasets import make_blobs
from scipy.optimize import minimize

from cvxopt import matrix, solvers

## make blobs
N = 20
X, y = make_blobs(n_samples=N, centers=2, n_features=2, random_state=0)
y[y==0] = -1

# plot examples
y_plot = y.reshape(-1, 1)
dataframe = pd.DataFrame(data=np.hstack([X,y_plot]), columns=("x", "y", "label"))
sn.FacetGrid(dataframe, hue="label", size=6).map(plt.scatter, 'x', 'y').add_legend()
plt.show()

## optimization
alphs0 = np.ones_like(y)

Q = np.zeros((N, N))

def normal_kernel(x1, x2):
    return x1@x2

def poly_kernel(x1, x2, d=2):
    return (x1@x2 + 1)**d




## optimization
for i in range(N):
    for j in range(N):
        Q[i][j] = y[i]*y[j]*poly_kernel(X[i], X[j])


Q = matrix(Q)
p = -matrix(np.ones((N,1)))
G = -matrix(np.eye(N))
h = -matrix(np.zeros((N,1)))
A = matrix(y.reshape(1,N).astype('float64') )
b = matrix(0.0)
sol = solvers.qp(Q, p, G, h, A, b)
alphs_opt = np.array(sol['x'])


## get support vectors
sup_vects = []  
for i in np.nonzero(np.round(alphs_opt, 4))[0]:
    sup_vects.append((X[i], y[i]))
    
    
## calculate bias
sums = [0, 0]
bs = [0, 0]
for j in range(2):
    for k in range(len(y)):
        sums[j] += alphs_opt[k]*y[k]*poly_kernel(sup_vects[j][0], X[k])
        bs[j] = sup_vects[j][1] - sums[j]
        
if np.round(bs[1] - bs[0], 2):
    print('something wrong')

## classify points to visualisation
xx = np.arange(-7, 7, .1)
yy = np.arange(-12, 2, .1)
xxx, yyy = np.meshgrid(xx, yy, sparse=False)
arr = np.zeros((len(xx),len(yy)))

b = bs[0]
for i in range(len(xx)):
    for j in range(len(yy)):
        for k in range(len(y)):
            arr[i][j] += alphs_opt[k]*y[k]*poly_kernel(np.array([xxx[i][j], yyy[i][j]]), X[k])
        arr[i][j] = (arr[i][j] + b) > 0

## visualisation of decision boundary
dataframe = pd.DataFrame(data=np.hstack([np.vstack((xxx.ravel(), yyy.ravel())).T, arr.ravel().reshape(-1, 1)]), columns=("x", "y", "label"))
sn.FacetGrid(dataframe, hue="label", palette='pastel', size=6).map(plt.scatter, 'x', 'y').add_legend()
dataframe = pd.DataFrame(data=np.hstack([X,y_plot]), columns=("x", "y", "label"))
sn.FacetGrid(dataframe, hue="label", size=6).map(plt.scatter, 'x', 'y').add_legend()

plt.show()

