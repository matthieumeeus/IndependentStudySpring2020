import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from scipy.special import j0, y0, j1, y1, jv, yn

# in this file we wish to approximate the Bessel function with a Fourier Series network
# in a supervised way
# and compare one sine layer with bias with cos and sine together without bias

# define the network
# define the network
class NN_Fourier_wo_bias(nn.Module):

    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=32):
        """Initializer method.
        """
        super(NN_Fourier_wo_bias, self).__init__()
        self.sin_part_lin = nn.Linear(n_input_units, n_hidden_units, bias = False)
        self.cos_part_lin = nn.Linear(n_input_units, n_hidden_units, bias = False)
        self.output = nn.Linear(2*n_hidden_units, n_output_units)

    def forward(self, t):
        sin_lin = self.sin_part_lin(t)
        sin = torch.sin(sin_lin)
        cos_lin = self.cos_part_lin(t)
        cos = torch.cos(cos_lin)
        combined = torch.cat((sin, cos), 1)
        output = self.output(combined)
        return output

class NN_Fourier_onlySine(nn.Module):

    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=32):
        """Initializer method.
        """
        super(NN_Fourier_onlySine, self).__init__()
        self.sin_part_lin = nn.Linear(n_input_units, 2*n_hidden_units, bias = True)
        self.output = nn.Linear(2*n_hidden_units, n_output_units)

    def forward(self, t):
        sin_lin = self.sin_part_lin(t)
        sin = torch.sin(sin_lin)
        output = self.output(sin)
        return output

x = torch.unsqueeze(torch.linspace(0.2, 20, 50), dim=1)
bessels = [j0, y0, j1, y1, jv, yn]
labels = ['j0', 'y0', 'j1', 'y1', 'j2', 'y2']
epochs = 50000
n_hidden_units = 8
fig, ax = plt.subplots(4, 2, figsize=(20,20))

for i, func in enumerate(bessels[:4]):
    ### PART 1: SINE + COSINE  & WITHOUT BIAS
    if i > 3:
        y = func(2,x)
        y = torch.tensor(y.numpy(), dtype = torch.float)
    else:
        y = func(x)
    Fourier_net = NN_Fourier_wo_bias(n_hidden_units=n_hidden_units)
    optimizer = optim.Adam(Fourier_net.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
    losses = []

    for _ in range(epochs):
        prediction = Fourier_net(x)     # input x and predict based on x
        loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
        losses.append(loss.detach().item())
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

    # plot hist of freqencies
    weights_biases = np.array([p.detach().numpy() for p in Fourier_net.parameters()])
    freq_sin = weights_biases[0][:,0]
    freq_cos = weights_biases[1][:,0]
    contributions_sin = weights_biases[2][0,:n_hidden_units]
    contributions_cos = weights_biases[2][0,n_hidden_units:]

    ax[i,1].bar(freq_sin,contributions_sin, width=0.08, alpha = 0.7, label = 'Network 1 - sin')
    ax[i,1].bar(freq_cos, contributions_cos, width=0.08, alpha = 0.7, label = 'Network 1 - cos')
    ax[i,1].set_xlabel('Frequency weight', fontsize = 18)
    ax[i,1].set_ylabel('Output weight', fontsize = 18)

    # plot loss
    ax[i,0].loglog(range(epochs), losses, label = 'Sin + Cos & No bias', c = 'red')
    ax[i,0].set_xlabel('Epochs',fontsize = 18)
    ax[i,0].set_ylabel('Loss',fontsize = 18)
    ax[i,0].set_title('Loss for {}'.format(labels[i]),fontsize = 20)

    ### PART 2: SINE WITH BIAS
    if i > 3:
        y = func(2,x)
        y = torch.tensor(y.numpy(), dtype = torch.float)
    else:
        y = func(x)
    Fourier_net = NN_Fourier_onlySine(n_hidden_units=n_hidden_units)
    optimizer = optim.Adam(Fourier_net.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
    losses = []

    for _ in range(epochs):
        prediction = Fourier_net(x)     # input x and predict based on x
        loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
        losses.append(loss.detach().item())
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

    # plot hist of freqencies
    weights_biases = np.array([p.detach().numpy() for p in Fourier_net.parameters()])
    freq_sin = weights_biases[0][:,0]
    contributions_sin = weights_biases[2][0]

    ax[i,1].bar(freq_sin,contributions_sin, width=0.08, alpha = 0.7, label = 'Network 2 - only sin')
    ax[i,1].legend(fontsize = 15)

    # plot loss
    ax[i,0].loglog(range(epochs), losses, label = 'Sin & Bias', c = 'green')
    ax[i,0].legend(fontsize = 18)

plt.tight_layout()
plt.savefig('PlotsApril27/BesselFourier.png')

