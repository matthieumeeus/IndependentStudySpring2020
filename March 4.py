import torch
import numpy as np
import matplotlib.pyplot as plt
from neurodiffeq import ode
from neurodiffeq import diff
from neurodiffeq.ode import solve_system
import torch.nn as nn
from neurodiffeq import ode_masternode
from neurodiffeq.networks import FCNN

#omega_list = [0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]

omega = 2.5

for i in range(1):
    # define problem
    print(omega)
    oscillator = lambda x, t: diff(x, t, order=2) + (omega**2)*x
    init_val_ho = ode.IVP(t_0=0.0, x_0=0.0, x_0_prime=1.0)

    # define domain and solution
    ts = np.linspace(0, 4*np.pi, 200)
    ts_ana = np.linspace(0, 4*np.pi, 50)
    analyt = 1/omega*np.sin(omega*ts_ana)

    # solve the ODE with normal code
    init_val_ho = ode.IVP(t_0=0.0, x_0=0.0, x_0_prime=1.0)
    net_ho = FCNN(
        n_hidden_layers=2, n_hidden_units=40, actv=nn.Tanh
    )
    solution_ho, loss_normal = ode.solve(ode=oscillator, condition=init_val_ho, net=net_ho,
                           max_epochs=9000,t_min=0.0, t_max=2*np.pi)

    pred_normal = solution_ho(ts, as_type = 'np')

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


    # # solve the ODE with master node,
    # # without changing the enforcing
    # net_ho = FCNN_masternode(n_hidden_layers=2, n_hidden_units=32, actv=nn.Tanh)
    # solution_ho, loss_wo_per_enforce = ode.solve(
    #     ode=oscillator, condition=init_val_ho,  t_min=0.0, t_max=2*np.pi,
    #     net=net_ho,max_epochs=3000,
    # )
    #
    # pred_wo_per_enforce = solution_ho(ts, as_type = 'np')
    #
    #

    net_ho = FCNN_masternode(n_hidden_layers=2, n_hidden_units=40, actv=nn.Tanh)

    # define problem
    oscillator = lambda x, t: diff(x, t, order=2) + (omega**2)*x
    init_val_ho = ode_masternode.IVP(t_0=0.0, x_0=0.0, x_0_prime=1.0)

    # solve the ODE
    solution_ho, loss_per_enforce = ode_masternode.solve(
        ode=oscillator, condition=init_val_ho,  t_min=0.0, t_max=2*np.pi,
        net=net_ho,max_epochs=9000,
    )

    ps = [p for p in net_ho.parameters()]
    freq = ps[0]
    print('Frequency: {}'.format(freq.item()))
    print('Loss at the end: {}'.format(loss_per_enforce['valid_loss'][-1]))

    pred_per_enforce = solution_ho(ts, as_type = 'np')

    plt.figure(figsize = (12,8))
    plt.plot(ts, pred_normal, label = 'NN - Normal')
    plt.plot(ts, pred_per_enforce, label = 'NN - Master + periodic enforcing')
    plt.scatter(ts_ana, analyt, label = 'Analytical', color = 'red')
    plt.legend(fontsize = 18)
    plt.xlabel('T', fontsize = 25)
    plt.ylabel('X', fontsize = 25)
    plt.title('NN predictions for HO inside and outside training domain', fontsize = 18)
    plt.savefig('PlotsMarch11/FreqGuesses/MasternodeOmega{}.png'.format(omega))

    epochs = range(9000)
    plt.figure(figsize = (12,8))
    plt.loglog(epochs, loss_normal['train_loss'], label = 'Normal - train')
    plt.loglog(epochs, loss_normal['valid_loss'], label = 'Normal - valid')
    plt.loglog(epochs, loss_per_enforce['train_loss'], label = 'Periodic enforcing - train')
    plt.loglog(epochs, loss_per_enforce['valid_loss'], label = 'Periodic enforcing - valid')
    plt.legend(fontsize = 18)
    plt.xlabel('Epochs', fontsize = 20)
    plt.ylabel('Loss', fontsize = 20)
    plt.title('Loss within training domain', fontsize = 18)
    plt.savefig('PlotsMarch11/FreqGuesses/MasternodeLossOmega{}.png'.format(omega))
    
    print('Done!')
    print('---------------------------------------------------')

# # plot the error over time for the periodic model
# net_ho = FCNN_masternode(n_hidden_layers=2, n_hidden_units=32, actv=nn.Tanh)
#
# # define problem
# oscillator = lambda x, t: diff(x, t, order=2) + (omega**2)*x
# init_val_ho = ode_masternode.IVP(t_0=0.0, x_0=0.0, x_0_prime=1.0)
#
# # solve the ODE
# solution_ho, _ = ode_masternode.solve(
#     ode=oscillator, condition=init_val_ho,  t_min=0.0, t_max=2*np.pi,
#     net=net_ho,max_epochs=5000,
# )
#
# ts_ext = np.linspace(0, 6*np.pi, 500)
# x_ana = 1/omega*np.sin(omega*ts_ext)
# pred_per_enforce = solution_ho(ts_ext, as_type = 'np')
#
# plt.figure(figsize = (12,8))
# plt.plot(ts_ext, abs(x_ana - pred_per_enforce), color = 'red')
# plt.xlabel('T', fontsize = 25)
# plt.ylabel('X', fontsize = 25)
# plt.title('Errors inside and outside training domain', fontsize = 18)
# plt.savefig('PlotsMarch11/MasternodeErrors.png')




