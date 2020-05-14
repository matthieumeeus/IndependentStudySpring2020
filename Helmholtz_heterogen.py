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

########################################
# 8 Fourier (8 cos and 8 sine)
########################################

class FCNN_Fourier(nn.Module):

    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=32):
        r"""Initializer method.
        """
        super(FCNN_Fourier, self).__init__()
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

Fourier_net = FCNN_Fourier(n_input_units=2, n_hidden_units=8)
adam = optim.Adam(Fourier_net.parameters(), lr=0.005)
solution_NN_helmholtz_Fourier_8, loss_helmholtz_Fourier_8 = solve_polar(
        pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
        net=Fourier_net, max_epochs=5000, optimizer = adam)

########################################
# 16 Fourier (16 cos and 16 sine)
########################################

Fourier_net = FCNN_Fourier(n_input_units=2, n_hidden_units=16)
adam = optim.Adam(Fourier_net.parameters(), lr=0.005)
solution_NN_helmholtz_Fourier_16, loss_helmholtz_Fourier_16 = solve_polar(
        pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
        net=Fourier_net, max_epochs=5000, optimizer = adam)

########################################
# 32 Fourier (32 cos and 32 sine)
########################################

Fourier_net = FCNN_Fourier(n_input_units=2, n_hidden_units=32)
adam = optim.Adam(Fourier_net.parameters(), lr=0.005)
solution_NN_helmholtz_Fourier_32, loss_helmholtz_Fourier_32 = solve_polar(
        pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
        net=Fourier_net, max_epochs=5000, optimizer = adam)

########################################
# 64 Fourier (64 cos and 64 sine)
########################################

Fourier_net = FCNN_Fourier(n_input_units=2, n_hidden_units=64)
adam = optim.Adam(Fourier_net.parameters(), lr=0.005)
solution_NN_helmholtz_Fourier_64, loss_helmholtz_Fourier_64 = solve_polar(
        pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
        net=Fourier_net, max_epochs=5000, optimizer = adam)


epochs = range(5000)
plt.figure(figsize = (12,8))
plt.loglog(epochs, loss_helmholtz_Fourier_8['valid'], label = 'Valid - 8 sin + 8 cos')
plt.loglog(epochs, loss_helmholtz_Fourier_16['valid'], label = 'Valid - 16 sin + 16 cos')
plt.loglog(epochs, loss_helmholtz_Fourier_32['valid'], label = 'Valid - 32 sin + 32 cos')
plt.loglog(epochs, loss_helmholtz_Fourier_64['valid'], label = 'Valid - 64 sin + 64 cos')
plt.legend(fontsize = 18)
plt.xlabel('Epochs', fontsize = 20)
plt.ylabel('Loss', fontsize = 20)
plt.title('Loss within training domain', fontsize = 18)
plt.legend(fontsize = 18)
plt.savefig('PlotsApril20/Fourier/LossComparison.png')

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
