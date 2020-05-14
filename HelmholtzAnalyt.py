import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import jv, yn
import scipy.integrate as integrate

def plt_surf(xx, yy, zz, z_label='u', x_label='x', y_label='y', title=''):
    fig  = plt.figure(figsize=(16, 8))
    ax   = Axes3D(fig)
    surf = ax.plot_surface(xx, yy, zz)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    ax.set_proj_type('ortho')
    plt.savefig('PlotsApril9/Crazy_g/AnalytSolN=5.png')
    plt.show()

r_0 = 1
r_1 = 5

rs = np.linspace(1, 5, 100)
thetas = np.linspace(0, 2*np.pi, 100)
circle_r, circle_theta = np.meshgrid(rs, thetas)

X, Y = circle_r*np.cos(circle_theta), circle_r*np.sin(circle_theta)

analyt_sol = np.zeros((100,100))

A_0 = 1.0/np.pi*integrate.quad(lambda theta: np.sin(3*theta)*np.cos(theta), 0, 2*np.pi)[0]
kappa_ns = []
A_ns =[]
B_ns = []
ns = range(6)

for n in ns:
    kappa_n = - jv(n,r_0)/yn(n, r_0)
    kappa_ns.append(kappa_n)
    A_n_prime = 1.0/np.pi*integrate.quad(lambda theta: (np.sin(3*theta)*np.cos(theta))*np.cos(n*theta), 0, 2*np.pi)[0]
    A_n = A_n_prime/(jv(n,r_1) + kappa_n*yn(n,r_1))
    A_ns.append(A_n)
    B_n_prime = 1.0/np.pi*integrate.quad(lambda theta: (np.sin(3*theta)*np.cos(theta))*np.sin(n*theta), 0, 2*np.pi)[0]
    B_n = B_n_prime/(jv(n,r_1) + kappa_n*yn(n,r_1))
    B_ns.append(B_n)

print('Kappas:')
print(kappa_ns)
print('-----------')
print('Ans:')
print(A_ns)
print('-----------')
print('Bns:')
print(B_ns)
print('-----------')

# for p, r in enumerate(rs):
#     for q, theta in enumerate(thetas):
#         sol = A_0
#         for i,n in enumerate(ns):
#             sol += A_ns[i]*(jv(n, r) + kappa_ns[i]*yn(n,r))*np.cos(n*theta)
#             sol += B_ns[i]*(jv(n, r) + kappa_ns[i]*yn(n,r))*np.sin(n*theta)
#         analyt_sol[q,p] = sol

abs_sums = []
abs_diff = []

analyt_sol = np.ones((100,100))*A_0

for i, n in enumerate(ns):
    abs_sum = 0
    for p, r in enumerate(rs):
        for q, theta in enumerate(thetas):
            sol = analyt_sol[q,p]
            sol += A_ns[i]*(jv(n, r) + kappa_ns[i]*yn(n,r))*np.cos(n*theta)
            sol += B_ns[i]*(jv(n, r) + kappa_ns[i]*yn(n,r))*np.sin(n*theta)
            analyt_sol[q,p] = sol
            abs_sum += abs(sol)
    abs_sums.append(abs_sum)
    if i >= 1:
        abs_diff.append(abs(abs_sum - abs_sums[-2]))

# plt.figure()
# plt.plot(ns, abs_sums)
# plt.xlabel('Value for N')
# plt.ylabel('Absolute sum')
# plt.title('Absolute sum in entire solution for increasing amount of terms')
# plt.savefig('PlotsApril9/Crazy_g/AbsSumAnalyt.png')
#
#
# plt.figure()
# plt.plot(ns[1:], abs_diff)
# plt.xlabel('Value for N')
# plt.ylabel('Absolute difference')
# plt.title('Absolute Difference in entire solution for increasing amount of terms')
# plt.savefig('PlotsApril9/Crazy_g/AbsDiffAnalyt.png')

plt.figure()
plt.plot(thetas, analyt_sol[:,-1], label = 'Boundary computed', marker = 'o')
plt.plot(thetas, [np.sin(3*theta)*np.cos(theta) for theta in thetas], label = 'Exact boundary')
plt.legend()
plt.savefig('PlotsApril9/Crazy_g/BoundaryN=5.png')

Z = analyt_sol

plt_surf(X, Y, Z)

plt.figure()
plt.contourf(X,Y,Z, 30)
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('PlotsApril9/Crazy_g/AnalytSolContourN=5.png')