from typing import Tuple
import torch
import pandas as pd
import matplotlib.pyplot as plt


def gd_factorise_ad(A: torch.Tensor, rank: int, num_epochs=1000, lr=0.01):
    U, V = torch.tensor(torch.rand(A.shape[0], rank), requires_grad=True), torch.tensor(torch.rand(A.shape[1], rank), requires_grad=True)
    Eplots = []
    for epoch in range(num_epochs):
        J = torch.nn.functional.mse_loss(U @ V.T, A.float(), reduction='sum')
        Eplots.append(J.item())
        J.backward()
        U.data -= lr * U.grad
        V.data -= lr * V.grad
        U.grad.fill_(0)
        V.grad.fill_(0)
    return U.data, V.data



def plot_two_scatter(U, data):
    df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    data = torch.tensor(df.iloc[:, [0, 1, 2, 3]].values).float()
    data_mean = data.mean(dim=0)
    data = data - data_mean
    U2, V2, = gd_factorise_ad(data, 2)
    MD_axe_1 = V2[:, 0]
    MD_axe_2 = V2[:, 1]
    projection_MD_axe_1 = (data @ MD_axe_1).data.detach().numpy()
    projection_MD_axe_2 = (data @ MD_axe_2).data.detach().numpy()


    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax[0].scatter(projection_MD_axe_1, projection_MD_axe_2)
    ax[0].grid(True)
    ax[0].set_xlabel("project_axes1", fontsize=14)
    ax[0].set_ylabel("project_axes2", fontsize=14)
    ax[0].set_title("data projected onto the first two principle axes", fontsize=12)
    ax[1].scatter(U[:, 0], U[:, 1])
    ax[1].grid(True)
    ax[1].set_xlabel("U[0]", fontsize=14)
    ax[1].set_ylabel("U[1]", fontsize=14)
    ax[1].set_title("Matrix U", fontsize=12)
    plt.savefig("pca.png")
    plt.show()













if __name__ == '__main__':
    df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    data = torch.tensor(df.iloc[:, [0, 1, 2, 3]].values)
    data_mean = data.mean(dim=0)
    data = data - data_mean
    U, V = gd_factorise_ad(data, 2)
    print(torch.nn.functional.mse_loss(U @ V.T, data, reduction='sum'))
    plot_two_scatter(U, data)


