import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')


##A = np.array([[7, 4], [3, 4]])
A = np.array([[1, 4], [3, -1]])
##A = np.array([[1, 4], [3, -2]])
# b = np.array([1, 2]).reshape(2, 1)
b = np.array([2, 2]).reshape(2, 1)

# solution with matrix invertion
A_inv = np.linalg.inv(A)
x_linear = A_inv @ b


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
x1 = np.arange(x_linear[0] - 1, x_linear[0] + 1, 0.05)
x2 = np.arange(x_linear[1] - 1, x_linear[1] + 1, 0.05)

num = len(x1)

x1, x2 = np.meshgrid(x1, x2)
Z = np.zeros((num, num))
for i in range(num):
    for j in range(num):
        x_vect = np.array([x1[i][j], x2[i][j]]).reshape(2, 1)
        Z[i][j] = np.linalg.norm(A.T @ A @ x_vect - A.T @ b)

# Plot the surface.
surf = ax.plot_surface(x1, x2, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

ax.scatter(x_linear[0], x_linear[1], zs=np.linalg.norm(A.T @ A @ x_linear - A.T @ b), s=100, c='r')




# solution with gradient minimalization
x = np.random.rand(2, 1)

alpha = 0.001
delta = 1e-5

cnt = 0
while np.linalg.norm(A.T @ A @ x - A.T @ b) > delta:
    x = x - alpha * (A.T @ A @ x - A.T @ b)
    if cnt % 30 == 0:
        #ax.scatter(x[0], x[1], zs=np.linalg.norm(A.T @ A @ x - A.T @ b), s=50, c='b')
        ax.plot(x[0], x[1], zs=np.linalg.norm(A.T @ A @ x - A.T @ b), c='b', marker='o')
    cnt += 1


##    
# constrained optimization
##
x = np.random.rand(2, 1)
alpha_x = 0.001
alpha_lmd = 0.1
delta = 1e-5
norm_contr = .5

# plot edge of constrain
theta = np.linspace(0, 2 * np.pi, 201)
xy = np.array([norm_contr**.5*np.cos(theta), norm_contr**.5*np.sin(theta)])
z = np.linalg.norm(A.T @ A @ xy - A.T @ b, axis=0)
ax.plot(xy[0], xy[1], z, c='r')


'''find x which satisfies constrain'''
max_iter = 5e3
delta = 1e-5
cnt = 0
lmbd = 1e1 #np.rand.random()


if np.linalg.norm(x_linear.T @ x_linear) > norm_contr:
    while abs(np.linalg.norm(x.T @ x) - norm_contr) > delta:
        lmbd += alpha_lmd * (np.linalg.norm(x.T @ x) - norm_contr)
        x = np.linalg.inv(A.T @ A + 2 * lmbd * np.eye(len(x))) @ A.T @ b
        
        print(cnt, 'lmbd:', lmbd, 'norm:', np.linalg.norm(x.T @ x), 'delta:', abs(np.linalg.norm(x.T @ x) - norm_contr))
        if cnt == 0:
            ax.plot(x[0], x[1], zs=np.linalg.norm(A.T @ A @ x - A.T @ b),c='y', marker='o')
        if cnt > 0 and cnt % 30 == 0:
            ax.plot(x[0], x[1], zs=np.linalg.norm(A.T @ A @ x - A.T @ b), c='g', marker='o')
            alpha_lmd *= 1.3
        cnt += 1
        if cnt == max_iter:
            break
else:
    # global minimum lie within contrain
    x = x_linear

ax.plot(x[0], x[1], zs=np.linalg.norm(A.T @ A @ x - A.T @ b), c='r', marker='o')

# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

































