import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import torch


def Rastrigin(X, Y):
    A = 1
    Z = 2 * A + X ** 2 - A * np.cos(2 * np.pi * X) + Y ** 2 - A * np.cos(2 * np.pi * Y)
    return Z


def plot_Rastrigin(X, Y):
    A = 1
    Z = 2 * A + X ** 2 - A * np.cos(2 * np.pi * X) + Y ** 2 - A * np.cos(2 * np.pi * Y)
    return Z

def partial_derivative_0(X):
    return 2 * X + 2 * np.pi * np.sin(2 * np.pi * X)


def partial_derivative_1(Y):
    return 2 * Y + 2 * np.pi * np.cos(2 * np.pi * Y)


def function(X):
    maxIter, lr = 1000, 0.01
    x, y = X[0], X[1]
    ex = np.zeros(maxIter)
    GD_x, GD_y = [x], [y]
    GD_z = [Rastrigin(x, y)]
    z_change = []
    for i in range(maxIter):
        x_temp = x - lr * partial_derivative_0(x)
        y_temp = x - lr * partial_derivative_0(y)
        z_temp = Rastrigin(x_temp, y_temp)
        z_change.append(np.absolute(z_temp - Rastrigin(x_temp, y_temp)))
        GD_x.append(x_temp)
        GD_y.append(y_temp)
        GD_z.append(z_temp)
        x = x_temp
        y = y_temp
        z = z_temp
    return GD_x, GD_y, GD_z, z_change


def plot_contour(x_path, y_path):
    xmin, xmax, xstep = -5, 5, .2
    ymin, ymax, ystep = -5, 5, .2
    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
    z = plot_Rastrigin(x, y)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contourf(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.gray)
    ax.plot(x_path, y_path, color='red', label='SGD', linewidth=2)
    plt.show()



x_series, y_series, z_series, z_change = function(torch.tensor([5.0, 5.0]))
print("x_convergence:", x_series)
print("y_convergence:", y_series)
print("z_convergence:", z_series)
plot_contour(x_series, y_series)




