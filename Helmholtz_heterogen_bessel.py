from neurodiffeq.pde_polar import DirichletBVPPolar, solve_polar
import torch
from neurodiffeq.networks import FCNN
import numpy as np
from neurodiffeq import diff
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from scipy.special import jv, yn

def plt_surf(xx, yy, zz, z_label='u', x_label='x', y_label='y', title='',
             save='PlotsApril9/heteroBessel/SolutionAlone'):
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
    f = lambda theta : 0.0,
    r_1 = 5.0,
    g = lambda theta : torch.sin(3*theta)*torch.cos(theta))

########################################
# Normal network with Tanh
########################################

# print('Computing Tanh network')
# net = FCNN(n_input_units=2, n_hidden_units=32, n_hidden_layers=2)
# adam1 = optim.Adam(net.parameters(), lr=0.001)
# solution_NN_helmholtz, loss_helmholtz = solve_polar(
#         pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
#         net=net, max_epochs=5, optimizer = adam1)

########################################
# Heterogeneous network with Bessel/sin
########################################
print('Computing heteroBessel network')

class BesselFunction(nn.Module):

    def forward(self, x, type):
        if type == 'j2':
            return torch.from_numpy(jv(2,x.detach().numpy()))
        if type == 'j4':
            return torch.from_numpy(jv(4,x.detach().numpy()))
        if type == 'y2':
            return torch.from_numpy(yn(2,x.detach().numpy()))
        if type == 'y4':
            return torch.from_numpy(yn(4,x.detach().numpy()))

class FCNN_heteroBessel(nn.Module):

    def __init__(self, n_input_units=2, n_output_units=1, n_hidden_units=32):
        r"""Initializer method.
        """
        super(FCNN_heteroBessel, self).__init__()
        self.sin_part_lin = nn.Linear(1, n_hidden_units)
        self.j2_part_lin = nn.Linear(1, n_hidden_units)
        self.y2_part_lin = nn.Linear(1, n_hidden_units)
        self.j4_part_lin = nn.Linear(1, n_hidden_units)
        self.y4_part_lin = nn.Linear(1, n_hidden_units)
        self.output = nn.Linear(2*n_hidden_units, n_output_units)

    def forward(self, t):
        r = t.narrow(1,0,1).reshape(t.shape[0])
        theta = t.narrow(1,1,1).reshape(t.shape[0])
        sin_lin = self.sin_part_lin(theta)
        sin = torch.sin(sin_lin)
        j2_lin = self.j2_part_lin(r)
        j2 = torch.from_numpy(jv(2, j2_lin.detach().numpy())).float()
        # y2_lin = self.y2_part_lin(t)
        # y2 = torch.from_numpy(yn(2, y2_lin.detach().numpy())).float()
        j4_lin = self.j4_part_lin(r)
        j4 = torch.from_numpy(jv(4, j4_lin.detach().numpy())).float()
        # y4_lin = self.y4_part_lin(t)
        # y4 = torch.from_numpy(yn(4, y4_lin.detach().numpy())).float()

        r_combined = torch.cat((j2, j4))
        tensor_prod = torch.ger(sin, r_combined)
        summed = tensor_prod.sum(dim = 0)
        print(summed.shape)
        output = self.output(summed)
        return output

# turns out that this doesn't work
# the code is not able to predict when you separate the

heteroBessel_net = FCNN_heteroBessel(n_input_units=2, n_hidden_units=8)
adam = optim.Adam(heteroBessel_net.parameters(), lr=0.001)
solution_NN_helmholtz_heterobessel, loss_helmholtz_heterobessel = solve_polar(
        pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
        net=heteroBessel_net, max_epochs=2, optimizer = adam)

for p in heteroBessel_net.parameters():
    print(p)

rs = np.linspace(1, 5, 101)
thetas = np.linspace(0, 2*np.pi, 101)
circle_r, circle_theta = np.meshgrid(rs, thetas)

X, Y = circle_r*np.cos(circle_theta), circle_r*np.sin(circle_theta)

sol_net = solution_NN_helmholtz_heterobessel(circle_r, circle_theta, as_type='np')
Z = sol_net.reshape((101,101))

plt_surf(X, Y, Z)

# contourplot
plt.figure()
plt.contourf(X,Y,Z, 30)
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('PlotsApril20/TrialTensorProd.png')

epochs = range(10)
plt.figure(figsize = (12,8))
# plt.loglog(epochs, loss_helmholtz['train'], label = 'Train - Classic Tanh')
# plt.loglog(epochs, loss_helmholtz['valid'], label = 'Valid - Classic Tanh')
plt.loglog(epochs, loss_helmholtz_heterobessel['train'], label = 'Train - 1 layer sine,j2,j4')
plt.loglog(epochs, loss_helmholtz_heterobessel['valid'], label = 'Valid - 1 layer sine,j2,j4')
plt.legend(fontsize = 18)
plt.xlabel('Epochs', fontsize = 20)
plt.ylabel('Loss', fontsize = 20)
plt.title('Loss within training domain', fontsize = 18)
plt.legend(fontsize = 18)
plt.savefig('PlotsApril20/TrialTensorProdLoss.png')


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
