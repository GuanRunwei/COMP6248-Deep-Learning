import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def truncated_svd(basic_tensor):
    U1, S1, V1 = torch.svd(basic_tensor)
    print("U1:\n", U1)
    print("S1:\n", S1)
    print("V1:\n", V1)
    S1[-1] = 0
    S_new = S1
    print("S_new:\n", S_new)

    S_new_matrix, count = torch.empty(3, 3), 0
    for i in range(S_new_matrix.shape[0]):
        for j in range(S_new_matrix.shape[1]):
            if i == j:
                S_new_matrix[i][j] = S_new[count]
                count += 1

    A_estimate = U1 @ S_new_matrix @ V1.t()
    print(A_estimate)
    loss1 = torch.nn.functional.mse_loss(reduction='sum', input=basic_tensor, target=A_estimate)
    return loss1



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
    loss1 = torch.nn.functional.mse_loss(reduction='sum', input=U @ V.T, target=A)
    return loss1



if __name__ == '__main__':
    lack_array = [[0.3374, 0.6005, 0.1735], [3.3359, 0.0492, 1.8374], [2.9407, 0.5301, 2.02620]]
    lack_matrix = torch.Tensor(lack_array)
    print("loss of truncated_svd:", truncated_svd(lack_matrix))
    print("loss of gd_factorise_ad:", gd_factorise_ad(lack_matrix, 2))