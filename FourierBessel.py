import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from scipy.special import j0, y0, j1, y1, jv, yn

# in this file we wish to approximate the Bessel function with a Fourier Series network
# in a supervised way

# define the network
# define the network
class NN_Fourier(nn.Module):

    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=32):
        """Initializer method.
        """
        super(NN_Fourier, self).__init__()
        self.sin_part_lin = nn.Linear(n_input_units, n_hidden_units)
        self.cos_part_lin = nn.Linear(n_input_units, n_hidden_units)
        self.output = nn.Linear(2*n_hidden_units, n_output_units)

    def forward(self, t):
        sin_lin = self.sin_part_lin(t)
        sin = torch.sin(sin_lin)
        cos_lin = self.cos_part_lin(t)
        cos = torch.cos(cos_lin)
        combined = torch.cat((sin, cos), 1)
        output = self.output(combined)
        return output

x = torch.unsqueeze(torch.linspace(0.2, 20, 50), dim=1)
bessels = [j0, y0, j1, y1, jv, yn]
labels = ['j0', 'y0', 'j1', 'y1', 'j2', 'y2']
epochs = 1000

fig, ax = plt.subplots(6, 3, figsize=(15,20))

for i, func in enumerate(bessels):
    if i > 3:
        y = func(2,x)
        y = torch.tensor(y.numpy(), dtype = torch.float)
    else:
        y = func(x)
    Fourier_net = NN_Fourier(n_hidden_units=8)
    optimizer = optim.Adam(Fourier_net.parameters(), lr=0.005)
    loss_func = nn.MSELoss()
    losses = []

    for t in range(epochs):
        prediction = Fourier_net(x)     # input x and predict based on x
        loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
        losses.append(loss.detach().item())
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

    # plot solution
    ax[i,0].set_title('{}'.format(labels[i]),fontsize = 16)
    ax[i,0].set_xlabel('x',fontsize = 16)
    ax[i,0].set_ylabel('y',fontsize = 16)
    ax[i,0].scatter(x.data.numpy(), y.data.numpy(), label = 'Data', c = 'green')
    ax[i,0].plot(x.data.numpy(), prediction.data.numpy(), label = 'Prediction', c = 'red')
    ax[i,0].legend(fontsize = 16)

    # plot hist of freqencies
    weights_biases = np.array([p.detach().numpy() for p in Fourier_net.parameters()])
    weights_biases_sin = weights_biases[0][:,0]
    weights_biases_cos = weights_biases[2][:,0]

    ax[i,1].hist(weights_biases_sin, bins = 16, alpha = 0.5, label = 'Sin')
    ax[i,1].hist(weights_biases_cos, bins = 16, alpha = 0.5, label = 'Cos')
    ax[i,1].legend(fontsize = 16)
    ax[i,1].set_title('Histogram of frequencies', fontsize = 16)

    # plot loss
    ax[i,2].loglog(range(epochs), losses)
    ax[i,2].set_xlabel('Epochs',fontsize = 16)
    ax[i,2].set_ylabel('Loss',fontsize = 16)
    ax[i,2].set_title('Loss',fontsize = 16)
plt.tight_layout()
plt.savefig('PlotsApril27/BesselFourier.png')

