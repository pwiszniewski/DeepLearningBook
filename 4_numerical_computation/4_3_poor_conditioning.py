import numpy as np

# A = np.array([[4.1, 2.8], [9.7, 6.6]])
              
# b = A[:,0]
# x,resid,rank,s = np.linalg.lstsq(A,b, rcond=1)

# b1 = A@x

# b2 = [4.11, 9.7]
# x2 = np.linalg.lstsq(A,b2, rcond=1)[0]

# cond = np.linalg.cond(A)

# max_change = cond*np.linalg.norm(b - b2) / np.linalg.norm(b)
# act_change = np.linalg.norm(x - x2) / np.linalg.norm(x)

print(20*'-')

A = np.array([[4.1, 2.79], [9.7, 6.6]])
              
b = A[:,0]
x,resid,rank,s = np.linalg.lstsq(A,b, rcond=1)

b1 = A@x

b2 = [4.11, 9.7]
x2 = np.linalg.lstsq(A,b2, rcond=1)[0]

cond = np.linalg.cond(A)

max_change = cond*np.linalg.norm(b - b2) / np.linalg.norm(b)
act_change = np.linalg.norm(x - x2) / np.linalg.norm(x)

print(20*'-')

# A = np.array([[4.1, 2.8], [9.676, 6.608]])
# A_cond = np.linalg.cond(A)
# A_inv = np.linalg.inv(A)
# print(A_inv, A_cond)