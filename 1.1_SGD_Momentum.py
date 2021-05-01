import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import torch


def Rastrigin(X, Y):
    A = 1
    Z = 2 * A + X ** 2 - A * torch.cos(2 * np.pi * X) + Y ** 2 - A * torch.cos(2 * np.pi * Y)
    return Z


def plot_Rastrigin(X, Y):
    A = 1
    Z = 2 * A + X ** 2 - A * np.cos(2 * np.pi * X) + Y ** 2 - A * np.cos(2 * np.pi * Y)
    return Z

def partial_derivative_0(X):
    return 2 * X + 2 * np.pi * np.sin(2 * np.pi * X)


def partial_derivative_1(Y):
    return 2 * Y + 2 * np.pi * np.cos(2 * np.pi * Y)


def optimization_route(X):
    opt = torch.optim.SGD([X], lr=0.01, momentum=0.9)
    epochs = 1000
    opt_path = np.empty((2, 0))
    opt_path = np.append(opt_path, X.data.numpy(), axis=1)
    for i in range(epochs):
        opt.zero_grad()
        Z_temp = Rastrigin(X[0], X[1])
        Z_temp.backward()
        opt.step()
        opt_path = np.append(opt_path, X.data.numpy(), axis=1)
    return opt_path


def plot_contour(x_path, y_path):
    xmin, xmax, xstep = -5, 5, .2
    ymin, ymax, ystep = -5, 5, .2
    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
    z = plot_Rastrigin(x, y)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contourf(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.gray)
    ax.plot(x_path, y_path, color='red', label='SGD', linewidth=2)
    plt.show()



z = optimization_route(torch.tensor([[5.], [5.]], requires_grad=True))
print(type(z[0]))
plot_contour(z[0], z[1])
