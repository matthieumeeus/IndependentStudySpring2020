import torch
import numpy as np
import matplotlib.pyplot as plt
from neurodiffeq import ode
from neurodiffeq import diff
from neurodiffeq.ode import solve_system
import torch.nn as nn
from neurodiffeq import ode_masternode
from neurodiffeq.networks import FCNN
from scipy import linspace
from scipy.integrate import solve_ivp

# Let's try to predict the Vanderpol equation
# define problem
mu = 2

ts = np.linspace(0, 20, 200)
oscillator = lambda x, t: diff(x, t, order=2) + mu*(1-x*x)*diff(x, t) + x
init_val_ho = ode.IVP(t_0=0.0, x_0=0.0, x_0_prime=1.0)

# let's find ground truth first
def vdp(t, z):
    x, y = z
    return [y, mu*(1 - x**2)*y - x]

a, b = 0, 20

t = np.linspace(a, b, 500)

sol = solve_ivp(vdp, [a, b], [0, 1], t_eval=t)

print('Ground truth has been computed.')

# solve the ODE with normal code
train_gen = ode.ExampleGenerator(size=200,  t_min=0.0, t_max=20, method='equally-spaced-noisy')
valid_gen = ode.ExampleGenerator(size=200, t_min=0.0, t_max=20, method='uniform')
net_ho = FCNN(
    n_hidden_layers=10, n_hidden_units=32, actv=nn.Tanh
)
solution_ho, loss_normal = ode.solve(ode=oscillator, condition=init_val_ho, net=net_ho,
                       max_epochs=5000,t_min=0.0, t_max=20,
                       train_generator=train_gen, valid_generator=valid_gen)

pred_normal = solution_ho(ts, as_type = 'np')

print('Normal NN prediction has been computed.')

# implement the network needed for the masternode
class SineFunction(nn.Module):

    def forward(self, x):
        return torch.sin(x)

class FCNN_masternode(nn.Module):
    """A fully connected neural network.
    :param n_input_units: number of units in the input layer, defaults to 1.
    :type n_input_units: int
    :param n_input_units: number of units in the output layer, defaults to 1.
    :type n_input_units: int
    :param n_hidden_units: number of hidden units in each hidden layer, defaults to 32.
    :type n_hidden_units: int
    :param n_hidden_layers: number of hidden layers, defaults to 1.
    :type n_hidden_layers: int
    :param actv: the activation layer used in each hidden layer, defaults to `torch.nn.Tanh`.
    :type actv: `torch.nn.Module`
    """
    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=32, n_hidden_layers=1,
                 actv=nn.Tanh):
        r"""Initializer method.
        """
        super(FCNN_masternode, self).__init__()

        layers = []
        # add masternode
        layers.append(nn.Linear(n_input_units, n_input_units, bias = False))
        layers.append(SineFunction())
        layers.append(nn.Linear(n_input_units, n_hidden_units))
        layers.append(actv())
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(n_hidden_units, n_hidden_units))
            layers.append(actv())
        layers.append(nn.Linear(n_hidden_units, n_output_units))
        self.NN = torch.nn.Sequential(*layers)

    def forward(self, t):
        x = self.NN(t)
        return x


net_ho = FCNN_masternode(n_hidden_layers=10, n_hidden_units=32, actv=nn.Tanh)
train_gen = ode.ExampleGenerator(size=200,  t_min=0.0, t_max=20, method='equally-spaced-noisy')
valid_gen = ode.ExampleGenerator(size=200, t_min=0.0, t_max=20, method='uniform')
init_val_ho = ode_masternode.IVP(t_0=0.0, x_0=0.0, x_0_prime=1.0)
# solve the ODE
solution_ho, loss_per_enforce = ode_masternode.solve(
    ode=oscillator, condition=init_val_ho,  max_epochs=5000,t_min=0.0, t_max=20,
    net=net_ho,train_generator=train_gen, valid_generator=valid_gen
)

print('Masternode NN prediction has been computed.')

pred_per_enforce = solution_ho(ts, as_type = 'np')

plt.figure(figsize = (12,8))
plt.plot(ts, pred_normal, label = 'NN prediction - Normal')
plt.plot(ts, pred_per_enforce, label = 'NN prediction - Masternode')
plt.scatter(sol.t, sol.y[0], label = 'Scipy SolveIVP', c = 'red')
plt.xlabel('T', fontsize = 25)
plt.ylabel('X', fontsize = 25)
plt.title('Solutions to the Van der Pol equation with mu = {}'.format(mu), fontsize = 18)
plt.legend(fontsize = 18)
plt.savefig('PlotsMarch19/Vanderpol_compNN.png')
plt.show()

epochs = range(5000)
plt.figure(figsize = (12,8))
plt.loglog(epochs, loss_normal['train_loss'], label = 'Normal - train')
plt.loglog(epochs, loss_normal['valid_loss'], label = 'Normal - valid')
plt.loglog(epochs, loss_per_enforce['train_loss'], label = 'Masternode - train')
plt.loglog(epochs, loss_per_enforce['valid_loss'], label = 'Masternode - valid')
plt.legend(fontsize = 18)
plt.xlabel('Epochs', fontsize = 20)
plt.ylabel('Loss', fontsize = 20)
plt.title('Loss within training domain', fontsize = 18)
plt.legend(fontsize = 18)
plt.savefig('PlotsMarch19/VanderpolLoss_compNN.png')
