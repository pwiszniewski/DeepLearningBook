import torch

# var = torch.randn((1, 1), requires_grad=True)

# x = torch.normal(mean=0, std=1)

import numpy as np
from torch import tensor, nn
def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
labels = [0, 2, 1, 3]
logits =  np.array([
    [-3.5, -3.45, 0.23, 1.25],
    [-2.14, 0.54, 2.67, -5.23],
    [-1.34, 5.01, -1.54, -1.17],
    [ -2.98, -1.37, 1.54,5.23]
])
probs = softmax(logits)
log_probs = np.log(softmax(logits))
nll = -(log_probs[range(len(labels)), labels])
pass