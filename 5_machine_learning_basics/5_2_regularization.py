import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(size=5) - .5
x = x.reshape(-1,1)
y = -x**2

# no regularization
X = np.concatenate([np.ones((len(x), 1)), x], axis=1).reshape(-1,2)
w_lin = np.linalg.inv(X.T @ X) @ X.T @ y
num = 500
X_lin = np.concatenate([np.ones((num, 1)), np.linspace(-.5, .5, num=500).reshape(-1,1)], axis=1).reshape(-1,2)
y_hat_lin = X_lin @ w_lin

X_sq = np.concatenate([X, X[:,1].reshape(-1,1)**2], axis=1).reshape(-1,3)
w_sq = np.linalg.inv(X_sq.T @ X_sq) @ X_sq.T @ y
X_sq_app = np.concatenate([X_lin, X_lin[:,1].reshape(-1,1)**2], axis=1).reshape(-1,3)
y_hat_sq = X_sq_app @ w_sq

X_ninth = X_sq
X_ninth_app = X_sq_app

for i in range(3, 9):
    X_ninth = np.concatenate([X_ninth, X_ninth[:,1].reshape(-1,1)**i], axis=1).reshape(-1,i+1)
    X_ninth_app = np.concatenate([X_ninth_app, X_ninth_app[:,1].reshape(-1,1)**i], axis=1).reshape(-1,i+1)

w_ninth = np.linalg.inv(X_ninth.T @ X_ninth) @ X_ninth.T @ y
y_hat_ninth = X_ninth_app @ w_ninth

ax1=plt.subplot(3, 2, 1)
ax2=plt.subplot(3, 2, 3)
ax3=plt.subplot(3, 2, 5)

ax1.scatter(x,y)
ax1.scatter(X_lin[:,1],y_hat_lin, s=1)

ax2.scatter(x,y)
ax2.scatter(X_lin[:,1],y_hat_sq, s=1)

ax3.scatter(x,y)
ax3.scatter(X_lin[:,1],y_hat_ninth, s=1)

#plt.xlim(min(x), max(x))

# with regularization
lmbd = .001 # regularization coeff
X = np.concatenate([np.ones((len(x), 1)), x], axis=1).reshape(-1,2)
w_lin = np.linalg.pinv(X.T @ X + lmbd * np.eye(X.shape[1])) @ X.T @ y
num = 500
X_lin = np.concatenate([np.ones((num, 1)), np.linspace(-.5, .5, num=500).reshape(-1,1)], axis=1).reshape(-1,2)
y_hat_lin = X_lin @ w_lin

X_sq = np.concatenate([X, X[:,1].reshape(-1,1)**2], axis=1).reshape(-1,3)
w_sq = np.linalg.pinv(X_sq.T @ X_sq + lmbd * np.eye(X_sq.shape[1])) @ X_sq.T @ y
X_sq_app = np.concatenate([X_lin, X_lin[:,1].reshape(-1,1)**2], axis=1).reshape(-1,3)
y_hat_sq = X_sq_app @ w_sq

X_ninth = X_sq
X_ninth_app = X_sq_app

for i in range(3, 9):
    X_ninth = np.concatenate([X_ninth, X_ninth[:,1].reshape(-1,1)**i], axis=1).reshape(-1,i+1)
    X_ninth_app = np.concatenate([X_ninth_app, X_ninth_app[:,1].reshape(-1,1)**i], axis=1).reshape(-1,i+1)

w_ninth = np.linalg.pinv(X_ninth.T @ X_ninth + lmbd * np.eye(X_ninth
                                                            .shape[1])) @ X_ninth.T @ y
y_hat_ninth = X_ninth_app @ w_ninth

ax1=plt.subplot(3, 2, 2)
ax2=plt.subplot(3, 2, 4)
ax3=plt.subplot(3, 2
                , 6)

ax1.scatter(x,y)
ax1.scatter(X_lin[:,1],y_hat_lin, s=1)

ax2.scatter(x,y)
ax2.scatter(X_lin[:,1],y_hat_sq, s=1)

ax3.scatter(x,y)
ax3.scatter(X_lin[:,1],y_hat_ninth, s=1)

plt.show()
