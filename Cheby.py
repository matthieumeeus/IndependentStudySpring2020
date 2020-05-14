import torch
import numpy as np
import matplotlib.pyplot as plt
from neurodiffeq import ode
from neurodiffeq import diff
from neurodiffeq.ode import solve_system
import torch.nn as nn
from neurodiffeq.networks import FCNN

x0 = 0.0
v0 = 1.0
# omega = 2.0
#
# # ## Let's try Chebyshev
#
#
# # define problem
# omega = 2
# oscillator = lambda x, t: diff(x, t, order=2) + (omega**2)*x
# init_val_ho = ode.IVP(t_0=0.0, x_0=0.0, x_0_prime=1.0)
#
# # solve with default uniform
# train_gen = ode.ExampleGenerator(size=50,  t_min=0.0, t_max=2*np.pi, method='uniform')
# valid_gen = ode.ExampleGenerator(size=100, t_min=0.0, t_max=2*np.pi, method='equally-spaced')
# solution_ho_uni, loss_uniform = ode.solve(
#     ode=oscillator, condition=init_val_ho, t_min=0.0, t_max=2*np.pi,
#     train_generator=train_gen, valid_generator=valid_gen,
#     max_epochs=3000
# )
#
# # solve with chebyshev
# train_gen = ode.ExampleGenerator(size=50,  t_min=0.0, t_max=2*np.pi, method='chebyshev')
# valid_gen = ode.ExampleGenerator(size=100, t_min=0.0, t_max=2*np.pi, method='equally-spaced')
# solution_ho_cheby, loss_cheby = ode.solve(
#     ode=oscillator, condition=init_val_ho, t_min=0.0, t_max=2*np.pi,
#     train_generator=train_gen, valid_generator=valid_gen,
#     max_epochs=3000
# )
#
#
#
# # In[33]:
#
#
# epochs = range(len(loss_cheby['train_loss']))
# plt.figure(figsize = (12,8))
# plt.loglog(epochs, loss_uniform['train_loss'], label = 'Train uniform')
# plt.loglog(epochs, loss_uniform['valid_loss'], label = 'Validation uniform')
# plt.loglog(epochs, loss_cheby['train_loss'], label = 'Train Chebyshev')
# plt.loglog(epochs, loss_cheby['valid_loss'], label = 'Validation Chebyshev')
# plt.legend(fontsize = 18)
# plt.xlabel('Epochs', fontsize = 20)
# plt.ylabel('Loss', fontsize = 20)
# plt.title('Loss vs epochs [loglog] for HO with omega = 2', fontsize = 18)
# plt.savefig('Plots/Cheby1_less_pts.png')



# define problem
# omega = 4
# oscillator = lambda x, t: diff(x, t, order=2) + (omega**2)*x
# init_val_ho = ode.IVP(t_0=0.0, x_0=0.0, x_0_prime=1.0)
#
# # solve with default uniform
# train_gen = ode.ExampleGenerator(size=100,  t_min=0.0, t_max=2*np.pi, method='uniform')
# valid_gen = ode.ExampleGenerator(size=100, t_min=0.0, t_max=2*np.pi, method='equally-spaced')
# solution_ho, loss_uniform = ode.solve(
#     ode=oscillator, condition=init_val_ho, t_min=0.0, t_max=2*np.pi,
#     train_generator=train_gen, valid_generator=valid_gen,
#     max_epochs=4000
# )
#
# # solve with chebyshev
# train_gen = ode.ExampleGenerator(size=100,  t_min=0.0, t_max=2*np.pi, method='chebyshev')
# valid_gen = ode.ExampleGenerator(size=100, t_min=0.0, t_max=2*np.pi, method='equally-spaced')
# solution_ho, loss_cheby = ode.solve(
#     ode=oscillator, condition=init_val_ho, t_min=0.0, t_max=2*np.pi,
#     train_generator=train_gen, valid_generator=valid_gen,
#     max_epochs=4000
# )
#
# epochs = range(len(loss_cheby['train_loss']))
# plt.figure(figsize = (12,8))
# plt.loglog(epochs, loss_uniform['train_loss'], label = 'Train uniform')
# plt.loglog(epochs, loss_uniform['valid_loss'], label = 'Validation uniform')
# plt.loglog(epochs, loss_cheby['train_loss'], label = 'Train Chebyshev')
# plt.loglog(epochs, loss_cheby['valid_loss'], label = 'Validation Chebyshev')
# plt.legend(fontsize = 18)
# plt.xlabel('Epochs', fontsize = 20)
# plt.ylabel('Loss', fontsize = 20)
# plt.title('Loss vs epochs [loglog] for HO with omega = 4', fontsize = 18)
# plt.savefig('Plots/Cheby21.png')


# # In[9]:
#
#
# define problem
exponential = lambda x, t: diff(x, t) + x
init_val_ho = ode.IVP(t_0=0.0, x_0=1)

# solve with default uniform
train_gen = ode.ExampleGenerator(size=50,  t_min=0.0, t_max=2*np.pi, method='uniform')
valid_gen = ode.ExampleGenerator(size=100, t_min=0.0, t_max=2*np.pi, method='equally-spaced')
solution_ho, loss_uniform = ode.solve(
    ode=exponential, condition=init_val_ho, t_min=0.0, t_max=2*np.pi,
    train_generator=train_gen, valid_generator=valid_gen,
    max_epochs=5000
)

# solve with chebyshev
train_gen = ode.ExampleGenerator(size=50,  t_min=0.0, t_max=2*np.pi, method='chebyshev-noisy')
valid_gen = ode.ExampleGenerator(size=100, t_min=0.0, t_max=2*np.pi, method='equally-spaced')
solution_ho, loss_cheby = ode.solve(
    ode=exponential, condition=init_val_ho, t_min=0.0, t_max=2*np.pi,
    train_generator=train_gen, valid_generator=valid_gen,
    max_epochs=5000
)

epochs = range(len(loss_cheby['train_loss']))
plt.figure(figsize = (12,8))
plt.loglog(epochs, loss_uniform['train_loss'], label = 'Train uniform')
plt.loglog(epochs, loss_uniform['valid_loss'], label = 'Validation uniform')
plt.loglog(epochs, loss_cheby['train_loss'], label = 'Train Chebyshev')
plt.loglog(epochs, loss_cheby['valid_loss'], label = 'Validation Chebyshev')
plt.legend(fontsize = 18)
plt.xlabel('Epochs', fontsize = 20)
plt.ylabel('Loss', fontsize = 20)
plt.title('Loss vs epochs [loglog] for EXPONENTIAL', fontsize = 18)
plt.savefig('Plots/Expon.png')
#
#
# # In[11]:
#
#
# # let's plot inside and outside domain for both interpolation methods
#
# # define problem
# omega = 2
# oscillator = lambda x, t: diff(x, t, order=2) + (omega**2)*x
# init_val_ho = ode.IVP(t_0=0.0, x_0=0.0, x_0_prime=1.0)
#
# # solve with default uniform
# train_gen = ode.ExampleGenerator(size=50,  t_min=0.0, t_max=2*np.pi, method='uniform')
# valid_gen = ode.ExampleGenerator(size=100, t_min=0.0, t_max=2*np.pi, method='equally-spaced')
# solution_ho_uni, _ = ode.solve(
#     ode=oscillator, condition=init_val_ho, t_min=0.0, t_max=2*np.pi,
#     train_generator=train_gen, valid_generator=valid_gen,
#     max_epochs=3000
# )
#
# # solve with chebyshev
# train_gen = ode.ExampleGenerator(size=50,  t_min=0.0, t_max=2*np.pi, method='chebyshev')
# valid_gen = ode.ExampleGenerator(size=100, t_min=0.0, t_max=2*np.pi, method='equally-spaced')
# solution_ho_cheby, _ = ode.solve(
#     ode=oscillator, condition=init_val_ho, t_min=0.0, t_max=2*np.pi,
#     train_generator=train_gen, valid_generator=valid_gen,
#     max_epochs=3000
# )







