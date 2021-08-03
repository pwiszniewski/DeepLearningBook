import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import torch

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1)


# plot examples
y_plot = y.reshape(-1, 1)
dataframe = pd.DataFrame(data=np.hstack([X,y_plot]), columns=("1st_principal", "2nd_principal", "label"))
sn.FacetGrid(dataframe, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin = nn.Linear(2, 2)

    def forward(self, x):
        x = self.lin(x)
        #x = F.relu(x)
        return x

X = torch.tensor(X)
y = torch.tensor(y)
split_rate = 0.8
wholeset = torch.utils.data.TensorDataset(X, y)
train_len = int(.8 * len(wholeset))
test_len = len(wholeset) - train_len
trainset, testset = torch.utils.data.random_split(wholeset, [train_len, test_len])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=0)

net = Net()

import torch.optim as optim

# train 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        #outputs = torch.max(outputs, 1)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# test
X_hats, lbls = [], []
net.eval()
with torch.no_grad():
    running_loss = 0.0
    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        X_hats.append(inputs.numpy())
        lbls.append(labels.numpy().reshape(-1,1))
        
lbls = np.vstack(lbls)
X_hats = np.vstack(X_hats)
# plot classification
y_plot = y.reshape(-1, 1)
dataframe = pd.DataFrame(data=np.hstack([X_hats,lbls]), columns=("1st_principal", "2nd_principal", "label"))
sn.FacetGrid(dataframe, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()
