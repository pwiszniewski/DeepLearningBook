import matplotlib.pyplot as plt
import numpy as np


x = np.array([[0,0], [0,1], [1,0], [1,1]])
w = np.random.rand(2, 1)

b = np.random.randn(1)
f = x @ w + b

J = np.mean(f)