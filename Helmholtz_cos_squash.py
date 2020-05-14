from neurodiffeq.pde_polar import DirichletBVPPolar, solve_polar
import torch
from neurodiffeq.networks import FCNN
import numpy as np
from neurodiffeq import diff
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn

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
# Sine net
########################################
print('Computing Sine network')
class SineFunction(nn.Module):

    def forward(self, x):
        return torch.sin(x)

net_sin = FCNN(n_input_units=2, n_hidden_units=32, n_hidden_layers=3, actv=SineFunction)
adam2 = optim.Adam(net_sin.parameters(), lr=0.0005)
solution_NN_helmholtz_sin, loss_helmholtz_sin = solve_polar(
        pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
        net=net_sin, max_epochs=5000, optimizer = adam2)

########################################
# Cosine squash net
########################################
print('Computing Cosine network')
class SquashCos(nn.Module):

    def forward(self, x):
        return 0.5*(torch.cos(torch.clamp(x, min = -np.pi/2, max = np.pi/2) + 3*np.pi/2.0) + 1.0)

net_sin = FCNN(n_input_units=2, n_hidden_units=32, n_hidden_layers=3, actv=SquashCos)
adam2 = optim.Adam(net_sin.parameters(), lr=0.0005)
solution_NN_helmholtz_squash, loss_helmholtz_squash = solve_polar(
        pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
        net=net_sin, max_epochs=5000, optimizer = adam2)

epochs = range(5000)
plt.figure(figsize = (12,8))
plt.loglog(epochs, loss_helmholtz_sin['train'], label = 'Train - Sin')
plt.loglog(epochs, loss_helmholtz_sin['valid'], label = 'Valid - Sin')
plt.loglog(epochs, loss_helmholtz_squash['train'], label = 'Train - Cos Squash')
plt.loglog(epochs, loss_helmholtz_squash['valid'], label = 'Valid - Cos Squash')
plt.legend(fontsize = 18)
plt.xlabel('Epochs', fontsize = 20)
plt.ylabel('Loss', fontsize = 20)
plt.title('Loss within training domain', fontsize = 18)
plt.legend(fontsize = 18)
plt.savefig('PlotsApril20/LossSineAndSquash.png')