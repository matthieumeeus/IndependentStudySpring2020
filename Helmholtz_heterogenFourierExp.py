from neurodiffeq.pde_polar import DirichletBVPPolar, solve_polar
import torch
from neurodiffeq.networks import FCNN
import numpy as np
from neurodiffeq import diff
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from scipy.special import j0, y0, j1, y1

# In this script, I will try to compare the sine with bias
# vs the sin/cos network without bias with equal amount of parameters

helmholtz = lambda u, r, theta: diff(u, r, order=2) + (1/r)*diff(u, r) \
                                + (1/r**2)*diff(u, theta, order = 2) + u

bc = DirichletBVPPolar(
    r_0 = 1.0,
    f = lambda theta : 1.0,
    r_1 = 5.0,
    g = lambda theta : torch.sin(3*theta))

class FCNN_FourierSinCos(nn.Module):

    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=32):
        r"""Initializer method.
        """
        super(FCNN_FourierSinCos, self).__init__()
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

class FCNN_FourierSin(nn.Module):

    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=32):
        r"""Initializer method.
        """
        super(FCNN_FourierSin, self).__init__()
        self.sin_part_lin = nn.Linear(n_input_units, n_hidden_units)
        self.output = nn.Linear(n_hidden_units, n_output_units)

    def forward(self, t):
        sin_lin = self.sin_part_lin(t)
        sin = torch.sin(sin_lin)
        output = self.output(sin)
        return output

n_experiments = 20
n_hidden_units = 32
n_epochs = 5000
last_loss_range = int(0.1*n_epochs)
last_losses_SinCos = []
last_losses_Sin = []

for i in range(n_experiments):
    print('Experiment {} has begun'.format(i))
    ########################################
    # Train network 1
    # with 32 Sin and 32 Cos but no bias
    ########################################

    Fourier_net = FCNN_FourierSinCos(n_input_units=2, n_hidden_units=n_hidden_units)
    adam = optim.Adam(Fourier_net.parameters(), lr=0.005)
    solution_NN_helmholtz_FourierSinCos, loss_helmholtz_FourierSinCos = solve_polar(
            pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
            net=Fourier_net, max_epochs=n_epochs, optimizer = adam)

    last_losses_SinCos.append(loss_helmholtz_FourierSinCos['valid'][-last_loss_range:])

    ########################################
    # Train network 2
    # with 32 Sin only with bias
    ########################################

    Fourier_net = FCNN_FourierSin(n_input_units=2, n_hidden_units=n_hidden_units)
    adam = optim.Adam(Fourier_net.parameters(), lr=0.005)
    solution_NN_helmholtz_FourierSin, loss_helmholtz_FourierSin = solve_polar(
            pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
            net=Fourier_net, max_epochs=n_epochs, optimizer = adam)

    last_losses_Sin.append(loss_helmholtz_FourierSin['valid'][-last_loss_range:])

    if i%2 == 0:
        np.savetxt('PlotsMay4/Exp2SinLosses{}.txt'.format(i), np.array(last_losses_Sin),delimiter=',')
        np.savetxt('PlotsMay4/Exp2SinCosLosses{}.txt'.format(i), np.array(last_losses_SinCos),delimiter=',')

np.savetxt('PlotsMay4/Exp2SinLosses.txt', np.array(last_losses_Sin),delimiter=',')
np.savetxt('PlotsMay4/Exp2SinCosLosses.txt', np.array(last_losses_SinCos),delimiter=',')

epochs = range(n_epochs)
plt.figure(figsize = (12,8))
plt.loglog(epochs, loss_helmholtz_FourierSin['valid'], label = 'Valid - SinCos & weights')
plt.loglog(epochs, loss_helmholtz_FourierSinCos['valid'], label = 'Valid - Sine & no weights')
plt.legend(fontsize = 18)
plt.xlabel('Epochs', fontsize = 20)
plt.ylabel('Loss', fontsize = 20)
plt.title('Loss within training domain - evaluating Fourier layer', fontsize = 18)
plt.legend(fontsize = 18)
plt.savefig('PlotsMay4/HelmHoltzFourierLossExp.png')
