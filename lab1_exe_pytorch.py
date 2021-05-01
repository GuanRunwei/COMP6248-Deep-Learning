import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def sgd_factorise(A: torch.Tensor, rank: int, num_epochs=1000, lr=0.01):
    U, V = torch.rand(A.shape[0], rank), torch.rand(A.shape[1], rank)
    for epoch in range(num_epochs):
        for r in range(A.shape[0]):
            for c in range(A.shape[1]):
                e = A[r][c] - U[r] @ V[c].t()
                U[r] = U[r] + lr * e * V[c]
                V[c] = V[c] + lr * e * U[r]
    return U, V


def sgd_factorise_masked(A: torch.Tensor, M: torch.Tensor, rank: int, num_epochs=1000, lr=0.01):
    U, V = torch.rand(A.shape[0], rank), torch.rand(A.shape[1], rank)
    for epoch in range(num_epochs):
        for r in range(A.shape[0]):
            for c in range(A.shape[1]):
                if M[r][c] is not 0:
                    e = A[r][c] - U[r] @ V[c].t()
                    U[r] = U[r] + lr * e * V[c]
                    V[c] = V[c] + lr * e * U[r]

    return U, V








if __name__ == '__main__':
    # section 1
    basic_array = [[0.3374, 0.6005, 0.1735], [3.3359, 0.0492, 1.8374], [2.9407, 0.5301, 2.02620]]
    basic_tensor = torch.Tensor(basic_array)
    print("A:\n", basic_tensor)
    U, V = sgd_factorise(basic_tensor, 2)
    print("U_hat:\n", U)
    print("V_hat:\n", V)
    A_hat = U @ V.t()
    print("A_hat:\n", A_hat)
    loss = torch.nn.functional.mse_loss(reduction='sum', input=basic_tensor, target=A_hat)
    print(loss)

    # section 2
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
    print(loss1)


    # section 3
    lack_array = [[0.3374, 0.6005, 0.1735], [0, 0.0492, 1.8374], [2.9407, 0, 2.02620]]
    lack_matrix = torch.Tensor(lack_array)
    binary_array = [[1, 1, 1], [0, 1, 1], [1, 0, 1]]
    binary_matrix = torch.Tensor(binary_array)
    U_lack, V_lack = sgd_factorise_masked(lack_matrix, binary_matrix, 2)
    print("U_lack:\n", U_lack)
    print("V_lack:\n", V_lack)
    A_lack_estimate = U @ V.T
    print("A_lack_estimate:\n", A_lack_estimate)
    loss_lack = torch.nn.functional.mse_loss(reduction='sum', input=basic_tensor, target=A_lack_estimate)
    print("loss_lack:\n", loss_lack)




