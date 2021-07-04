import matplotlib.pyplot as plt
import numpy as np

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

''' Linear regression '''

w = np.random.rand(3, 1) # [w0, w1, b]
X_ext = np.hstack((X, np.ones((X.shape[0], 1)))) # b constant

f = X_ext @ w

J = np.mean(f)

w_opt = np.linalg.inv(X_ext.T @ X_ext) @ (X_ext.T @ y)
y_hat_lr = X_ext @ w_opt

# plot
plt.scatter(X_ext[:,0], X_ext[:,1], c=y)
xmin, xmax = plt.xlim()
plt.plot((xmin, xmax), (.5, .5))
plt.title('Linear regression')
plt.show()

''' MLP '''
# simple MLP with 1 hidden layer consisting of 2 hidden units
# function describing MKP is: f = w.T @ max{0, W.T@x + c} + b
W = np.array([[1, 1], 
              [1, 1]])
c = np.array([0, -1])
w = np.array([1, -2])
b = 0

h = X @ W + c

# all points in hidden space lie on one line with slope 1
# and cannot be separated linearly
plt.scatter(h[:,0], h[:,1], c=y)
xmin, xmax = plt.xlim()
plt.title('MLP')
plt.show()

# after ReLU the points are linearly separeble
def ReLU(x):
    return np.maximum(x, 0)
h_rect = ReLU(h)
plt.scatter(h_rect[:,0], h_rect[:,1], c=y)
xmin, xmax = plt.xlim()
plt.show()

y_hat_lr = h_rect @ w