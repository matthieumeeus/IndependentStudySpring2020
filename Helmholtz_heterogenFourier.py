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

def plt_surf(xx, yy, zz, z_label='u', x_label='x', y_label='y', title='', save='PlotsApril9/Fourier/SolutionLargeLR'):
    fig  = plt.figure(figsize=(16, 8))
    ax   = Axes3D(fig)
    surf = ax.plot_surface(xx, yy, zz)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    ax.set_proj_type('ortho')
    plt.savefig(save)
    #plt.show()

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

class FCNN_FourierSinCosInit(nn.Module):

    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=32):
        r"""Initializer method.
        """
        super(FCNN_FourierSinCosInit, self).__init__()
        self.sin_part_lin = nn.Linear(n_input_units, n_hidden_units, bias = False)
        self.cos_part_lin = nn.Linear(n_input_units, n_hidden_units, bias = False)

        K_theta = np.array(range(n_hidden_units))
        K_r = np.array([2*np.pi*k/4.0 for k in range(n_hidden_units)])
        K = torch.tensor(np.array([K_r, K_theta]).T)
        K = K.float()
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


class FCNN_FourierSin(nn.Module):

    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=32):
        r"""Initializer method.
        """
        super(FCNN_FourierSin, self).__init__()
        self.sin_part_lin = nn.Linear(n_input_units, 2*n_hidden_units)
        self.output = nn.Linear(2*n_hidden_units, n_output_units)

    def forward(self, t):
        sin_lin = self.sin_part_lin(t)
        sin = torch.sin(sin_lin)
        output = self.output(sin)
        return output

########################################
# Train network 1
# with Sin and Cos but no bias wo weight init
########################################
n_hidden_units = 32
n_epochs = 500
Fourier_net = FCNN_FourierSinCos(n_input_units=2, n_hidden_units=n_hidden_units)
adam = optim.Adam(Fourier_net.parameters(), lr=0.05)
solution_NN_helmholtz_Fourier1, loss_helmholtz_Fourier1 = solve_polar(
        pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
        net=Fourier_net, max_epochs=n_epochs, optimizer = adam)

weights_biases = np.array([p.detach().numpy() for p in Fourier_net.parameters()])
print([p.shape for p in weights_biases])
freq_sin_r = weights_biases[0][:,0]
freq_sin_theta = weights_biases[0][:,1]
freq_cos_r = weights_biases[1][:,0]
freq_cos_theta = weights_biases[1][:,1]
relev_sin = weights_biases[2][0,:n_hidden_units]
relev_cos = weights_biases[2][0,n_hidden_units:]

plt.figure()
plt.bar(freq_sin_theta, relev_sin, width = 0.05, alpha = 0.7, label = 'Sin')
plt.bar(freq_cos_theta, relev_cos, width = 0.05, alpha = 0.7, label = 'Cos')
plt.xlabel('Frequency weight', fontsize = 16)
plt.ylabel('Output weight', fontsize = 16)
plt.legend(fontsize = 16)
plt.title('Theta - No Weight init', fontsize = 16)
plt.savefig('PlotsApril27/theta_weights_SinCos_NOINIT{}.png'.format(n_hidden_units))

plt.figure()
plt.bar(freq_sin_r, relev_sin, width = 0.05,  alpha = 0.5, label = 'Sin')
plt.bar(freq_cos_r, relev_cos, width = 0.05, alpha = 0.5, label = 'Cos')
plt.xlabel('Frequency weight', fontsize = 16)
plt.ylabel('Output weight', fontsize = 16)
plt.legend(fontsize = 16)
plt.title('Radius - No weight init', fontsize = 16)
plt.savefig('PlotsApril27/r_weights_SinCos_NOINIT{}.png'.format(n_hidden_units))

########################################
# Train network 2
# with Sin and Cos but no bias with weight init
########################################

Fourier_net = FCNN_FourierSinCosInit(n_input_units=2, n_hidden_units=n_hidden_units)
adam = optim.Adam(Fourier_net.parameters(), lr=0.05)
solution_NN_helmholtz_Fourier3, loss_helmholtz_Fourier3 = solve_polar(
        pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
        net=Fourier_net, max_epochs=n_epochs, optimizer = adam)

weights_biases = np.array([p.detach().numpy() for p in Fourier_net.parameters()])
print([p.shape for p in weights_biases])
freq_sin_r = weights_biases[0][:,0]
freq_sin_theta = weights_biases[0][:,1]
freq_cos_r = weights_biases[1][:,0]
freq_cos_theta = weights_biases[1][:,1]
relev_sin = weights_biases[2][0,:n_hidden_units]
relev_cos = weights_biases[2][0,n_hidden_units:]

plt.figure()
plt.bar(freq_sin_theta, relev_sin, width = 0.1, alpha = 0.7, label = 'Sin')
plt.bar(freq_cos_theta, relev_cos, width = 0.1, alpha = 0.7, label = 'Cos')
plt.xlabel('Frequency weight', fontsize = 16)
plt.ylabel('Output weight', fontsize = 16)
plt.legend(fontsize = 16)
plt.title('Theta - Weight init', fontsize = 16)
plt.savefig('PlotsApril27/theta_weights_SinCos_INIT{}.png'.format(n_hidden_units))

plt.figure()
plt.bar(freq_sin_r, relev_sin, width = 0.05,  alpha = 0.5, label = 'Sin')
plt.bar(freq_cos_r, relev_cos, width = 0.05, alpha = 0.5, label = 'Cos')
plt.xlabel('Frequency weight', fontsize = 16)
plt.ylabel('Output weight', fontsize = 16)
plt.legend(fontsize = 16)
plt.title('Radius - Weight init', fontsize = 16)
plt.savefig('PlotsApril27/r_weights_SinCos_INIT{}.png'.format(n_hidden_units))

########################################
# Train network 2
# Only Sin with bias
########################################

# Fourier_net = FCNN_FourierSin(n_input_units=2, n_hidden_units=16)
# adam = optim.Adam(Fourier_net.parameters(), lr=0.005)
# solution_NN_helmholtz_Fourier2, loss_helmholtz_Fourier2 = solve_polar(
#         pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
#         net=Fourier_net, max_epochs=n_epochs, optimizer = adam)
#
# weights_biases = np.array([p.detach().numpy() for p in Fourier_net.parameters()])
# print([p.shape for p in weights_biases])
# freq_sin_r = weights_biases[0][:,0]
# freq_sin_theta = weights_biases[0][:,1]
# relev_sin = weights_biases[2][0,:]
# biases = weights_biases[1]

# plt.figure()
# plt.bar(freq_sin_theta, relev_sin, width = 0.05, alpha = 0.7, color = 'green')
# plt.xlabel('Frequency weight', fontsize = 16)
# plt.ylabel('Output weight', fontsize = 16)
# plt.title('Contribution of trained modes of theta', fontsize = 16)
# plt.savefig('PlotsApril27/theta_weights_SinOnly_{}.png'.format(n_hidden_units))
#
# plt.figure()
# plt.bar(freq_sin_r, relev_sin, width = 0.05, alpha = 0.7, color = 'green')
# plt.xlabel('Frequency weight', fontsize = 16)
# plt.ylabel('Output weight', fontsize = 16)
# plt.title('Contribution of trained modes of radius', fontsize = 16)
# plt.savefig('PlotsApril27/r_weights_SinOnly_{}.png'.format(n_hidden_units))
#
# plt.figure()
# plt.hist(biases, bins = 20, alpha = 0.7, color = 'green')
# plt.title('Histogram of the biases', fontsize = 16)
# plt.savefig('PlotsApril27/biasesSinOnly_{}.png'.format(n_hidden_units))

epochs = range(n_epochs)
plt.figure(figsize = (12,8))
plt.loglog(epochs, loss_helmholtz_Fourier1['train'], label = 'Train - No weight init')
plt.loglog(epochs, loss_helmholtz_Fourier1['valid'], label = 'Valid - No weight init')
plt.loglog(epochs, loss_helmholtz_Fourier3['train'], label = 'Train - Weight init')
plt.loglog(epochs, loss_helmholtz_Fourier3['valid'], label = 'Valid - Weight init')
plt.legend(fontsize = 18)
plt.xlabel('Epochs', fontsize = 20)
plt.ylabel('Loss', fontsize = 20)
plt.title('Loss within training domain - evaluating weight init', fontsize = 18)
plt.legend(fontsize = 18)
plt.savefig('PlotsApril27/HelmHoltzFourierLossInit.png')

# contourplot
# plt.figure()
# plt.contourf(X,Y,Z, 30)
# plt.colorbar()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.savefig('PlotsApril9/Fourier/GeneralSolContourLargeLR.png')

#
# #let's plot the boundary conditions
# rs_bc = solution_NN_helmholtz(np.ones(101)*5.0, thetas, as_type='np')
# plt.figure()
# plt.plot(thetas, rs_bc, label = 'Solution at boundary')
# mse = np.sum([(rs_bc[i] - np.sin(3*thetas)[i])**2 for i in range(101)])
# plt.plot(thetas, np.sin(3*thetas), label = 'Difference, MSE = {}'.format(mse))
# plt.legend()
# plt.xlabel('Theta', fontsize = 17)
# plt.ylabel('u(r,theta)', fontsize = 17)
# plt.title('Solution at the boundary for r = 5', fontsize = 17)
# plt.savefig('PlotsApril2/BoundaryR5.png')


#
# epochs = range(3000)
# plt.figure(figsize = (12,8))
# plt.loglog(epochs, loss_helmholtz['train'], label = 'Normal - train - Tanh')
# plt.loglog(epochs, loss_helmholtz['valid'], label = 'Normal - valid - Tanh')
# plt.loglog(epochs, loss_helmholtz_sin['train'], label = 'Normal - train - Sin')
# plt.loglog(epochs, loss_helmholtz_sin['valid'], label = 'Normal - valid - Sin')
# plt.legend(fontsize = 18)
# plt.xlabel('Epochs', fontsize = 20)
# plt.ylabel('Loss', fontsize = 20)
# plt.title('Loss within training domain', fontsize = 18)
# plt.legend(fontsize = 18)
# plt.savefig('PlotsMarch19/HelmholtzLossComparisonLargeLR.png')
