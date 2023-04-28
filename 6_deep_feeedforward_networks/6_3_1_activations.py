import torch
import matplotlib.pyplot as plt
import numpy as np

x = torch.linspace(-10, 10, 100, requires_grad=True)

def relu(x):
    return torch.maximum(torch.zeros_like(x), x)

def sigmoid(x):
    return torch.sigmoid(x)

y = relu(x)
# y = sigmoid(x)

grad = torch.ones_like(x)
print(y)
y.backward(gradient=grad)
print(x.grad)
plt.plot(x.detach().numpy(), y.detach().numpy())
plt.plot(x.detach().numpy(), x.grad)
plt.show()