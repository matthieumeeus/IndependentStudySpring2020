from neurodiffeq.pde import DirichletBVP2D, solve2D, ExampleGenerator2D, Monitor2D
import torch
from neurodiffeq.networks import FCNN
from neurodiffeq.pde import IBVP1D, make_animation
import numpy as np
from neurodiffeq import diff
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plt_surf(xx, yy, zz, z_label='u', x_label='x', y_label='y', title=''):
    fig  = plt.figure(figsize=(16, 8))
    ax   = Axes3D(fig)
    surf = ax.plot_surface(xx, yy, zz, rstride=2, cstride=1, alpha=0.8, cmap='hot')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    ax.set_proj_type('ortho')
    plt.show()
    plt.savefig('PlotsMarch11/2DLaplace.png')

laplace = lambda u, x, y: diff(u, x, order=2) + diff(u, y, order=2)
bc = DirichletBVP2D(
    x_min=0, x_min_val=lambda y: torch.sin(np.pi*y),
    x_max=1, x_max_val=lambda y: 0,
    y_min=0, y_min_val=lambda x: 0,
    y_max=1, y_max_val=lambda x: 0
)
net = FCNN(n_input_units=2, n_hidden_units=32, n_hidden_layers=1)

solution_neural_net_laplace, loss_laplace = solve2D(
    pde=laplace, condition=bc, xy_min=(0, 0), xy_max=(1, 1),
    net=net, max_epochs=200, train_generator=ExampleGenerator2D(
        (32, 32), (0, 0), (1, 1), method='equally-spaced-noisy'
    ),
)


xs, ys = np.linspace(0, 1, 101), np.linspace(0, 1, 101)
xx, yy = np.meshgrid(xs, ys)
sol_net = solution_neural_net_laplace(xx, yy, as_type='np')
plt_surf(xx, yy, sol_net, title='u(x, y) as solved by neural network')


epochs = range(200)
plt.figure(figsize = (12,8))
plt.loglog(epochs, loss_laplace['train_loss'], label = 'Normal - train')
plt.loglog(epochs, loss_laplace['valid_loss'], label = 'Normal - valid')
plt.legend(fontsize = 18)
plt.xlabel('Epochs', fontsize = 20)
plt.ylabel('Loss', fontsize = 20)
plt.title('Loss within training domain', fontsize = 18)
plt.legend(fontsize = 18)
plt.savefig('PlotsMarch11/LaplaceLoss.png')
