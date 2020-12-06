import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(size=5) - .5
x = x.reshape(-1,1)
y = -x**2

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

ax1=plt.subplot(3, 1, 1)
ax2=plt.subplot(3, 1, 2)
ax3=plt.subplot(3, 1, 3)

ax1.scatter(x,y)
ax1.scatter(X_lin[:,1],y_hat_lin, s=1)

ax2.scatter(x,y)
ax2.scatter(X_lin[:,1],y_hat_sq, s=1)

ax3.scatter(x,y)
ax3.scatter(X_lin[:,1],y_hat_ninth, s=1)

#plt.xlim(min(x), max(x))
plt.show()
