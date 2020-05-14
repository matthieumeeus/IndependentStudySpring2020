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

helmholtz = lambda u, r, theta: diff(u, r, order=2) + (1/r)*diff(u, r) \
                                + (1/r**2)*diff(u, theta, order = 2) + u

bc = DirichletBVPPolar(
    r_0 = 1.0,
    f = lambda theta : 0.0,
    r_1 = 5.0,
    g = lambda theta : torch.sin(3*theta)*torch.cos(theta))

def plt_surf(xx, yy, zz, z_label='u', x_label='x', y_label='y', title='',
             save='PlotsMay11/SolutionAloneN=8'):
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

########################################
# Heterogeneous network with Bessel/sin
########################################
print('Computing heteroBessel network')

# for j2
class BesselFunction_j2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return torch.from_numpy(jv(2,input.detach().numpy()))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        gradient = (torch.from_numpy(jv(1,input.detach().numpy())) - torch.from_numpy(jv(3,input.detach().numpy())))/2
        grad_input = grad_output.clone()
        return gradient*grad_input

# for j4
class BesselFunction_j4(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return torch.from_numpy(jv(4,input.detach().numpy()))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        gradient = (torch.from_numpy(jv(3,input.detach().numpy())) - torch.from_numpy(jv(5,input.detach().numpy())))/2
        grad_input = grad_output.clone()
        return gradient*grad_input


# for y2
class BesselFunction_y2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        input = torch.clamp(input, min = 1.0)
        ctx.save_for_backward(input)
        bessel =  torch.from_numpy(yn(2, input.detach().numpy()))
        return torch.where(torch.isnan(bessel), torch.zeros_like(bessel), bessel)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        gradient = (torch.from_numpy(yn(1,input.detach().numpy())) - torch.from_numpy(yn(3,input.detach().numpy())))/2
        gradient = torch.where(torch.isnan(gradient), torch.zeros_like(gradient), gradient)
        grad_input = grad_output.clone()
        grad_input = torch.where(torch.isnan(grad_input), torch.zeros_like(grad_input), grad_input)
        returning = gradient*grad_input
        return returning.float()

# for y4
class BesselFunction_y4(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        input = torch.clamp(input, min = 1.0)
        ctx.save_for_backward(input)
        bessel =  torch.from_numpy(yn(4, input.detach().numpy()))
        return torch.where(torch.isnan(bessel), torch.zeros_like(bessel), bessel)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        gradient = (torch.from_numpy(yn(3,input.detach().numpy())) - torch.from_numpy(yn(5,input.detach().numpy())))/2
        gradient = torch.where(torch.isnan(gradient), torch.zeros_like(gradient), gradient)
        grad_input = grad_output.clone()
        grad_input = torch.where(torch.isnan(grad_input), torch.zeros_like(grad_input), grad_input)
        returning = gradient*grad_input
        return returning.float()

class FCNN_heteroBessel(nn.Module):

    def __init__(self, n_input_units=2, n_output_units=1, n_hidden_units=32):
        """Initializer method.
        """
        super(FCNN_heteroBessel, self).__init__()
        self.sin_part_lin = nn.Linear(1, 4*n_hidden_units)
        self.j2_part_lin = nn.Linear(1, n_hidden_units)
        self.j4_part_lin = nn.Linear(1, n_hidden_units)
        self.y2_part_lin = nn.Linear(1, n_hidden_units)
        self.y4_part_lin = nn.Linear(1, n_hidden_units)
        self.output = nn.Linear(8*n_hidden_units, n_output_units)

    def forward(self, t):
        r = t.narrow(1,0,1)
        theta = t.narrow(1,1,1)
        sin_lin = self.sin_part_lin(theta)
        sin = torch.sin(sin_lin)
        j2_lin = self.j2_part_lin(r)
        j2_func = BesselFunction_j2.apply
        j2 = j2_func(j2_lin)
        j4_lin = self.j4_part_lin(r)
        j4_func = BesselFunction_j4.apply
        j4 = j4_func(j4_lin)
        y2_lin = self.y2_part_lin(r)
        y2_func = BesselFunction_y2.apply
        y2 = y2_func(y2_lin)
        y4_lin = self.y4_part_lin(r)
        y4_func = BesselFunction_y4.apply
        y4 = y4_func(y4_lin)
        combined = torch.cat((sin, j2, j4, y2.float(), y4.float()), 1)
        output = self.output(combined)
        return output


# let's do one layer
# n_epochs = 5000
#
# heteroBessel_net = FCNN_heteroBessel(n_input_units=2, n_hidden_units=16)
# adam = optim.Adam(heteroBessel_net.parameters(), lr=0.001)
# solution_NN_helmholtz_heterobessel, loss_helmholtz_heterobessel = solve_polar(
#         pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
#         net=heteroBessel_net, max_epochs=n_epochs, optimizer = adam)
#
# for p in heteroBessel_net.parameters():
#     print(p.shape)
#
# rs = np.linspace(1, 5, 101)
# thetas = np.linspace(0, 2*np.pi, 101)
# circle_r, circle_theta = np.meshgrid(rs, thetas)
#
# X, Y = circle_r*np.cos(circle_theta), circle_r*np.sin(circle_theta)
#
# sol_net = solution_NN_helmholtz_heterobessel(circle_r, circle_theta, as_type='np')
# Z = sol_net.reshape((101,101))
# plt_surf(X, Y, Z)
#
# # contourplot
# plt.figure()
# plt.contourf(X,Y,Z, 30)
# plt.colorbar()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.savefig('PlotsMay4/TrialBessel_J&Y.png')
# print(loss_helmholtz_heterobessel['valid'])
# epochs = range(n_epochs)
# plt.figure(figsize = (12,8))
# # plt.loglog(epochs, loss_helmholtz['train'], label = 'Train - Classic Tanh')
# # plt.loglog(epochs, loss_helmholtz['valid'], label = 'Valid - Classic Tanh')
# plt.loglog(epochs, loss_helmholtz_heterobessel['train'], label = 'Train - 1 layer sine,j2,j4')
# plt.loglog(epochs, loss_helmholtz_heterobessel['valid'], label = 'Valid - 1 layer sine,j2,j4')
# plt.legend(fontsize = 18)
# plt.xlabel('Epochs', fontsize = 20)
# plt.ylabel('Loss', fontsize = 20)
# plt.title('Loss within training domain', fontsize = 18)
# plt.legend(fontsize = 18)
# plt.savefig('PlotsMay4/TrialLoss_J&Y.png')

# let's do the tensor product one!

class FCNN_heteroBessel_prod(nn.Module):

    def __init__(self, n_input_units=2, n_output_units=1, n_hidden_units=32):
        """Initializer method.
        """
        super(FCNN_heteroBessel_prod, self).__init__()
        self.sin_part_lin = nn.Linear(1, 4*n_hidden_units, bias = False)
        self.j2_part_lin = nn.Linear(1, n_hidden_units, bias = False)
        self.j4_part_lin = nn.Linear(1, n_hidden_units, bias = False)
        self.y2_part_lin = nn.Linear(1, n_hidden_units, bias = False)
        self.y4_part_lin = nn.Linear(1, n_hidden_units, bias = False)
        self.output = nn.Linear((4*n_hidden_units)**2, n_output_units)

    def forward(self, t):
        r = t.narrow(1,0,1)
        theta = t.narrow(1,1,1)
        sin_lin = self.sin_part_lin(theta)
        sin = torch.sin(sin_lin)
        j2_lin = self.j2_part_lin(r)
        j2_func = BesselFunction_j2.apply
        j2 = j2_func(j2_lin)
        j4_lin = self.j4_part_lin(r)
        j4_func = BesselFunction_j4.apply
        j4 = j4_func(j4_lin)
        y2_lin = self.y2_part_lin(r)
        y2_func = BesselFunction_y2.apply
        y2 = y2_func(y2_lin)
        y4_lin = self.y4_part_lin(r)
        y4_func = BesselFunction_y4.apply
        y4 = y4_func(y4_lin)
        r_combined = torch.cat((j2, j4, y2.float(), y4.float()), 1).unsqueeze(0)
        r_combined = r_combined.view(r_combined.size(1),r_combined.size(0),r_combined.size(2))
        #print('R combo : {}'.format(r_combined.shape))
        sin_ready = sin.unsqueeze(0)
        sin_ready = sin_ready.view(sin_ready.size(1), sin_ready.size(0), sin_ready.size(2))
        #print('Sin ready : {}'.format(sin_ready.shape))
        r_theta_product = torch.bmm(torch.transpose(sin_ready, -2, -1), r_combined)
        #print('R theta prod : {}'.format(r_theta_product.shape))
        last_layer = r_theta_product.view(r_theta_product.size(0),
                                          r_theta_product.size(1)*r_theta_product.size(2))
        #print('last : {}'.format(last_layer.shape))
        output = self.output(last_layer)
        #print('Output: {}'.format(output.shape))
        return output

# let's do one layer
n_epochs = 20000

heteroBessel_net = FCNN_heteroBessel_prod(n_input_units=2, n_hidden_units=8)
adam = optim.Adam(heteroBessel_net.parameters(), lr=0.005)
solution_NN_helmholtz_heterobessel, loss_helmholtz_heterobessel = solve_polar(
        pde = helmholtz, condition = bc, r_min = 1.0, r_max = 5.0,
        net=heteroBessel_net, max_epochs=n_epochs, optimizer = adam)

for p in heteroBessel_net.parameters():
    print(p.shape)

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
plt.savefig('PlotsMay11/TrialBessel_prodN=8.png')
print(loss_helmholtz_heterobessel['valid'])
epochs = range(n_epochs)
plt.figure(figsize = (12,8))
# plt.loglog(epochs, loss_helmholtz['train'], label = 'Train - Classic Tanh')
# plt.loglog(epochs, loss_helmholtz['valid'], label = 'Valid - Classic Tanh')
plt.loglog(epochs, loss_helmholtz_heterobessel['train'], label = 'Train')
plt.loglog(epochs, loss_helmholtz_heterobessel['valid'], label = 'Valid')
plt.legend(fontsize = 18)
plt.xlabel('Epochs', fontsize = 20)
plt.ylabel('Loss', fontsize = 20)
plt.title('Loss within training domain', fontsize = 18)
plt.legend(fontsize = 18)
plt.savefig('PlotsMay11/TrialLoss_TensorProdN=8.png')