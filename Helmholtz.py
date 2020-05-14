from neurodiffeq.pde_polar import DirichletBVPPolar, solve_polar
import torch
from neurodiffeq.networks import FCNN
import numpy as np
from neurodiffeq import diff
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn

def plt_surf(xx, yy, zz, z_label='u', x_label='x', y_label='y', title=''):
    fig  = plt.figure(figsize=(16, 8))
    ax   = Axes3D(fig)
    surf = ax.plot_surface(xx, yy, zz)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    ax.set_proj_type('ortho')
    plt.savefig('PlotsApril2/GeneralSol.png')
    #plt.show()

helmholtz = lambda u, r, theta: diff(u, r, order=2) + (1/r)*diff(u, r) \
                                + (1/r**2)*diff(u, theta, order = 2) + u

bc = DirichletBVPPolar(
    r_0 = 1.0,
    f = lambda theta : 1.0,
    r_1 = 5.0,
    g = lambda theta : torch.sin(3*theta))

net = FCNN(n_input_units=2, n_hidden_units=32, n_hidden_layers=5)
solution_NN_helmholtz, loss_helmholtz = solve_polar(
        pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
        net=net, max_epochs=5000)

rs = np.linspace(1, 5, 101)
thetas = np.linspace(0, 2*np.pi, 101)
circle_r, circle_theta = np.meshgrid(rs, thetas)

X, Y = circle_r*np.cos(circle_theta), circle_r*np.sin(circle_theta)

# theta_transf = np.sin(thetas)
# circle_r_transf, circle_theta_transf = np.meshgrid(rs, theta_transf)
# sol_net = solution_NN_helmholtz(circle_r_transf, circle_theta_transf, as_type='np')
sol_net = solution_NN_helmholtz(circle_r, circle_theta, as_type='np')
Z = sol_net.reshape((101,101))

plt_surf(X, Y, Z)

# contourplot
plt.figure()
plt.contourf(X,Y,Z, 30)
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('PlotsApril2/GeneralSolContour.png')

#let's plot the boundary conditions
rs_bc = solution_NN_helmholtz(np.ones(101)*5.0, thetas, as_type='np')
plt.figure()
plt.plot(thetas, rs_bc, label = 'Solution at boundary')
mse = np.sum([(rs_bc[i] - np.sin(3*thetas)[i])**2 for i in range(101)])
plt.plot(thetas, np.sin(3*thetas), label = 'Difference, MSE = {}'.format(mse))
plt.legend()
plt.xlabel('Theta', fontsize = 17)
plt.ylabel('u(r,theta)', fontsize = 17)
plt.title('Solution at the boundary for r = 5', fontsize = 17)
plt.savefig('PlotsApril2/BoundaryR5.png')

epochs = range(5000)
plt.figure(figsize = (12,8))
plt.loglog(epochs, loss_helmholtz['train'], label = 'Normal - train - Tanh')
plt.loglog(epochs, loss_helmholtz['valid'], label = 'Normal - valid - Tanh')
# plt.loglog(epochs, loss_helmholtz_sin['train'], label = 'Normal - train - Sin')
# plt.loglog(epochs, loss_helmholtz_sin['valid'], label = 'Normal - valid - Sin')
plt.legend(fontsize = 18)
plt.xlabel('Epochs', fontsize = 20)
plt.ylabel('Loss', fontsize = 20)
plt.title('Loss within training domain', fontsize = 18)
plt.legend(fontsize = 18)
plt.savefig('PlotsApril2/GeneralSolLoss.png')
#
# class SineFunction(nn.Module):
#
#     def forward(self, x):
#         return torch.sin(x)
#
# net_sin = FCNN(n_input_units=3, n_hidden_units=32, n_hidden_layers=3, actv=SineFunction)
# adam2 = optim.Adam(net_sin.parameters(), lr=0.005)
# solution_NN_helmholtz_sin, loss_helmholtz_sin = solve_spherical(
#         pde = helmholtz, condition = bc, r_min = 0.0, r_max = 2.0,
#         net=net, max_epochs=1000, optimizer = adam2)
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
