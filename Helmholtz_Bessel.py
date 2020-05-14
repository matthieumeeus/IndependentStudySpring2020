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
# Bessel net
########################################
print('Computing Besssel network j01')
class BesselFunction(nn.Module):

    def forward(self, x):
        return torch.from_numpy(j0(x.detach().numpy()))

net_Bessel = FCNN(n_input_units=2, n_hidden_units=32, n_hidden_layers=1, actv=BesselFunction)
solution_NN_helmholtz_Besselj01, loss_helmholtz_Besselj01 = solve_polar(
        pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
        net=net_Bessel, max_epochs=5000)

print('Computing Besssel network j11')
class BesselFunction(nn.Module):

    def forward(self, x):
        return torch.from_numpy(j1(x.detach().numpy()))

net_Bessel = FCNN(n_input_units=2, n_hidden_units=32, n_hidden_layers=1, actv=BesselFunction)
solution_NN_helmholtz_Besselj11, loss_helmholtz_Besselj11 = solve_polar(
        pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
        net=net_Bessel, max_epochs=5000)

print('Computing Besssel network j02')
class BesselFunction(nn.Module):

    def forward(self, x):
        return torch.from_numpy(j0(x.detach().numpy()))

net_Bessel = FCNN(n_input_units=2, n_hidden_units=32, n_hidden_layers=2, actv=BesselFunction)
solution_NN_helmholtz_Besselj02, loss_helmholtz_Besselj02 = solve_polar(
        pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
        net=net_Bessel, max_epochs=5000)

print('Computing Besssel network j12')
class BesselFunction(nn.Module):

    def forward(self, x):
        return torch.from_numpy(j1(x.detach().numpy()))

net_Bessel = FCNN(n_input_units=2, n_hidden_units=32, n_hidden_layers=2, actv=BesselFunction)
solution_NN_helmholtz_Besselj12, loss_helmholtz_Besselj12 = solve_polar(
        pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
        net=net_Bessel, max_epochs=5000)

epochs = range(5000)
plt.figure(figsize = (12,8))
plt.loglog(epochs, loss_helmholtz_Besselj01['valid'], label = 'Valid - j0 - 1 layer')
plt.loglog(epochs, loss_helmholtz_Besselj11['valid'], label = 'Valid - j1 - 1 layer')
plt.loglog(epochs, loss_helmholtz_Besselj02['valid'], label = 'Valid - j0 - 2 layers')
plt.loglog(epochs, loss_helmholtz_Besselj12['valid'], label = 'Valid - j1 - 2 layers')
plt.legend(fontsize = 18)
plt.xlabel('Epochs', fontsize = 20)
plt.ylabel('Loss', fontsize = 20)
plt.title('Loss within training domain', fontsize = 18)
plt.legend(fontsize = 18)
plt.savefig('PlotsApril2/LossBesselAll.png')

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
