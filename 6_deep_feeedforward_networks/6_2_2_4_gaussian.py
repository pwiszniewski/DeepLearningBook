import torch
import matplotlib.pyplot as plt
import numpy as np

# neural network with a mixture density output layer.
class MixtureDensityNetwork(torch.nn.Module):
    def __init__(self, n_hidden, n_gaussians):
        super(MixtureDensityNetwork, self).__init__()
        self.n_gaussians = n_gaussians
        self.n_hidden = n_hidden
        self.fc1 = torch.nn.Linear(1, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_hidden)
        self.fc3 = torch.nn.Linear(n_hidden, n_gaussians * 3) # 3 parameters per gaussian
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_gaussian_params(self, x):
        # get the parameters of the mixture density output layer
        y = self.forward(x)
        pi, mu, sigma = torch.split(y, self.n_gaussians, dim=1)
        pi = torch.nn.functional.softmax(pi, dim=1)
        sigma = torch.exp(sigma)
        return pi, mu, sigma
    
    def get_loss(self, x, y):
        # get the loss for a batch of data
        pi, mu, sigma = self.get_gaussian_params(x)
        # compute the loss
        loss = 0
        for i in range(self.n_gaussians):
            loss += pi[:, i] * torch.exp(-(y - mu[:, i])**2 / (2 * sigma[:, i]**2)) / (sigma[:, i] * np.sqrt(2 * np.pi))
        loss = -torch.log(loss)
        return loss.mean()
    
    def get_prediction(self, x):
        # get the prediction for a batch of data
        pi, mu, sigma = self.get_gaussian_params(x)
        # compute the prediction
        y = 0
        for i in range(self.n_gaussians):
            y += pi[:, i] * torch.exp(-(x - mu[:, i])**2 / (2 * sigma[:, i]**2)) / (sigma[:, i] * np.sqrt(2 * np.pi))
        return y
    

# generate data
def generate_data(n_samples):
    x = torch.rand(n_samples, 1) * 10 - 5
    y = torch.sin(x) + torch.randn(n_samples, 1) * 0.2
    return x, y

# train the network
def train_network(n_hidden, n_gaussians, n_epochs, learning_rate):
    # create the network
    net = MixtureDensityNetwork(n_hidden, n_gaussians)
    # create the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # train the network
    for epoch in range(n_epochs):
        # generate data
        x, y = generate_data(100)
        # compute the loss
        loss = net.get_loss(x, y)
        # backpropagate the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print the loss
        if epoch % 100 == 0:
            print('epoch: {}, loss: {}'.format(epoch, loss.item()))
    return net

# plot the prediction
def plot_prediction(net):
    # generate data
    x, y = generate_data(100)
    # get the prediction
    y_pred = net.get_prediction(x)
    # plot the prediction
    plt.scatter(x, y, c='b', label='data')
    plt.plot(x, y_pred.detach(), c='r', label='prediction')
    plt.legend()
    plt.show()

# main
if __name__ == '__main__':
    # train the network
    net = train_network(50, 5, 1000, 0.01)
    # plot the prediction
    plot_prediction(net)

    
