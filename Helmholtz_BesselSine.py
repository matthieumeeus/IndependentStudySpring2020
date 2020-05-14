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

def plt_surf(xx, yy, zz, z_label='u', x_label='x', y_label='y', title='', save='PlotsApril2/Gekkie.png'):
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
# Bessel Sine net
########################################

class SineFunction(nn.Module):

        def forward(self, x):
            return torch.sin(x)

class BesselFunction(nn.Module):

    def forward(self, x):
        return torch.from_numpy(j1(x.detach().numpy()))

class FCNN_BesselSine(nn.Module):

    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=32, n_hidden_layers=1,
                     actv=nn.Tanh):
        r"""Initializer method.
        """
        super(FCNN_BesselSine, self).__init__()

        layers = []
        layers.append(nn.Linear(n_input_units, n_hidden_units))
        layers.append(BesselFunction())
        layers.append(nn.Linear(n_hidden_units, n_hidden_units))
        layers.append(SineFunction())
        layers.append(nn.Linear(n_hidden_units, n_output_units))
        self.NN = torch.nn.Sequential(*layers)

    def forward(self, t):
        x = self.NN(t)
        return x

net_Bessel_sine = FCNN_BesselSine(n_input_units=2, n_hidden_units=32)
solution_NN_helmholtz_BesselSine, loss_helmholtz_BesselSine = solve_polar(
        pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
        net=net_Bessel_sine, max_epochs=10000)

########################################
# Sine Bessel net
########################################
#
# class FCNN_SineBessel(nn.Module):
#
#     def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=32, n_hidden_layers=1,
#                      actv=nn.Tanh):
#         r"""Initializer method.
#         """
#         super(FCNN_SineBessel, self).__init__()
#
#         layers = []
#         layers.append(nn.Linear(n_input_units, n_hidden_units))
#         layers.append(SineFunction())
#         layers.append(nn.Linear(n_hidden_units, n_hidden_units))
#         layers.append(BesselFunction())
#         layers.append(nn.Linear(n_hidden_units, n_output_units))
#         self.NN = torch.nn.Sequential(*layers)
#
#     def forward(self, t):
#         x = self.NN(t)
#         return x
#
# net_sine_bessel = FCNN_SineBessel(n_input_units=2, n_hidden_units=32)
# solution_NN_helmholtz_SineBessel, loss_helmholtz_SineBessel = solve_polar(
#         pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
#         net=net_sine_bessel, max_epochs=5000)


epochs = range(10000)
plt.figure(figsize = (12,8))
plt.loglog(epochs, loss_helmholtz_BesselSine['train'], label = 'Train - Bessel -Sine')
plt.loglog(epochs, loss_helmholtz_BesselSine['valid'], label = 'Valid - Bessel -Sine')
# plt.loglog(epochs, loss_helmholtz_SineBessel['train'], label = 'Train - Sine - Bessel')
# plt.loglog(epochs, loss_helmholtz_SineBessel['valid'], label = 'Valid - Sine - Bessel')
plt.legend(fontsize = 18)
plt.xlabel('Epochs', fontsize = 20)
plt.ylabel('Loss', fontsize = 20)
plt.title('Loss within training domain', fontsize = 18)
plt.legend(fontsize = 18)
plt.savefig('PlotsApril2/LossBesselSineAlone.png')

# # contourplot
# plt.figure()
# plt.contourf(X,Y,Z, 30)
# plt.colorbar()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.savefig('PlotsApril2/GeneralSolContour.png')
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
