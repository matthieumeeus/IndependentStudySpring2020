import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from scipy.special import j0, y0, j1, y1, jv, yn

# in this file we wish to approximate the Bessel function with a Fourier Series network
# in a supervised way
# and compare integer weight initialization vs none

# define the network
class NN_Fourier_wo_init(nn.Module):

    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=32):
        """Initializer method.
        """
        super(NN_Fourier_wo_init, self).__init__()
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
        print(output.shape)
        return output

class NN_Fourier_with_init(nn.Module):

    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=32):
        """Initializer method.
        """
        super(NN_Fourier_with_init, self).__init__()
        self.sin_part_lin = nn.Linear(n_input_units, n_hidden_units, bias = False)
        self.cos_part_lin = nn.Linear(n_input_units, n_hidden_units, bias = False)
        K = torch.Tensor(np.array([2*np.pi*k/20.0 for k in range(n_hidden_units)]))
        K = torch.unsqueeze(K,1)
        self.sin_part_lin.weight.data = self.sin_part_lin.weight.data + K
        self.cos_part_lin.weight.data = self.cos_part_lin.weight.data + K
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
epochs = 10000
n_hidden_units = 16
fig, ax = plt.subplots(6, 3, figsize=(20,20))

for i, func in enumerate(bessels):
    ### PART 1: SINE + COSINE  & WITHOUT INIT
    if i > 3:
        y = func(2,x)
        y = torch.tensor(y.numpy(), dtype = torch.float)
    else:
        y = func(x)
    Fourier_net = NN_Fourier_wo_init(n_hidden_units=n_hidden_units)
    optimizer = optim.Adam(Fourier_net.parameters(), lr=0.003)
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

    ax[i,1].bar(freq_sin,contributions_sin, width=0.08, alpha = 0.7, label = 'Sin')
    ax[i,1].bar(freq_cos, contributions_cos, width=0.08, alpha = 0.7, label = 'Cos')
    ax[i,1].legend(fontsize = 15)
    ax[i,1].set_xlabel('Frequency weight', fontsize = 18)
    ax[i,1].set_title('Without weight init', fontsize = 18)
    ax[i,1].set_ylabel('Output weight', fontsize = 18)

    # plot loss
    ax[i,0].loglog(range(epochs), losses, label = 'No Weight init', c = 'red')
    ax[i,0].set_xlabel('Epochs',fontsize = 18)
    ax[i,0].set_ylabel('Loss',fontsize = 18)
    ax[i,0].set_title('Loss for {}'.format(labels[i]),fontsize = 20)

    ### PART 2: SINE WITH BIAS
    if i > 3:
        y = func(2,x)
        y = torch.tensor(y.numpy(), dtype = torch.float)
    else:
        y = func(x)
    Fourier_net = NN_Fourier_with_init(n_hidden_units=n_hidden_units)
    optimizer = optim.Adam(Fourier_net.parameters(), lr=0.003)
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

    ax[i,2].bar(freq_sin,contributions_sin, width=0.08, alpha = 0.7, label = 'Sin')
    ax[i,2].bar(freq_cos, contributions_cos, width=0.08, alpha = 0.7, label = 'Cos')
    ax[i,2].set_xlabel('Frequency weight', fontsize = 18)
    ax[i,2].set_ylabel('Output weight', fontsize = 18)
    ax[i,2].set_title('With weight init', fontsize = 18)
    ax[i,2].legend(fontsize = 15)

    # plot loss
    ax[i,0].loglog(range(epochs), losses, label = 'Weight init', c = 'green')
    ax[i,0].legend(fontsize = 18)

plt.tight_layout()
plt.savefig('PlotsApril27/BesselFourierInitAll.png')

