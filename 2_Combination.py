import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import torch
import pandas as pd
from celluloid import Camera
from IPython.display import HTML



df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
df = df.sample(frac=1, random_state=0)  # shuffle
df = df[df[4].isin(['Iris-virginica', 'Iris-versicolor'])]  # filter

# add label indices column
mapping = {k: v for v, k in enumerate(df[4].unique())}
df[5] = (2 * df[4].map(mapping)) - 1  # labels in {-1, 1}

#  normalise data
alldata = torch.tensor(df.iloc[:, [0, 1, 2, 3]].values, dtype=torch.float)
alldata = (alldata - alldata.mean(dim=0)) / alldata.var(dim=0)

# create datasets
targets_tr = torch.tensor(df.iloc[:75, 5].values, dtype=torch.long)
targets_va = torch.tensor(df.iloc[75:, 5].values, dtype=torch.long)

data_tr = alldata[:75]
data_va = alldata[75:]

# Set up drawing
fig = plt.figure(figsize=(5, 5))
camera = Camera(fig)

#  训练
w = torch.randn(1, 4, requires_grad=True)
b = torch.randn(1, requires_grad=True)

w1 = torch.randn(1, 4, requires_grad=True)
b1 = torch.randn(1, requires_grad=True)


def hinge_loss(y_pred, y_true):
    return torch.mean(torch.clamp(1-y_pred.t() * y_true, min=0))



def svm(x, w, b):
    h = (w*x).sum(1) + b
    return h


def SGD_training():
    opt = torch.optim.SGD([w, b], lr=0.01, weight_decay=0.0001)
    loss_record, error_record = [], []
    error_record.append(real_time_error_rate(svm(data_tr, w, b)))
    for epoch in range(100):
        loss = None
        for i in range(3):
            try:
                opt.zero_grad()
                pred = svm(data_tr[25 * i: 25 * (i + 1)], w, b)
                loss = hinge_loss(pred, targets_tr[25 * i: 25 * (i + 1)])
                loss.backward()
                opt.step()
            except:
                break
        loss_record.append(loss)
        error_record.append(real_time_error_rate(svm(data_tr, w, b)))
        correct_rate = total_correct_rate_test(w, b)
    return loss_record, error_record, correct_rate


def Adam_training():
    opt = torch.optim.Adam([w1, b1], lr=0.01, weight_decay=0.0001)
    loss_record, error_record = [], []
    error_record.append(real_time_error_rate(svm(data_tr, w1, b1)))
    for epoch in range(100):
        loss = None
        for i in range(3):
            try:
                opt.zero_grad()
                pred = svm(data_tr[25 * i: 25 * (i + 1)], w1, b1)
                loss = hinge_loss(pred, targets_tr[25 * i: 25 * (i + 1)])
                loss.backward()
                opt.step()
            except:
                break
        loss_record.append(loss)
        error_record.append(real_time_error_rate(svm(data_tr, w1, b1)))
        correct_rate = total_correct_rate_test(w1, b1)
    return loss_record, error_record, correct_rate


def real_time_error_rate(pred):
    # 测试
    error_percentage, error_number = float, 0

    for i in range(len(data_tr)):
        pred_bool = 1 if pred[i] > 0 else -1
        if pred_bool != targets_tr[i]:
            error_number += 1
        error_percentage = error_number / len(data_tr) * 1.0
    return error_percentage


def total_correct_rate_test(w, b):
    result = svm(data_va, w, b)
    correct_num = 0
    for i in range(len(data_va)):
        result_bool = 1 if result[i] > 0 else -1
        if result_bool == targets_va[i]:
            correct_num += 1
    return correct_num / len(data_va) * 1.0





if __name__ == '__main__':
    sgd_loss, sgd_error, correct_rate_sgd = SGD_training()
    adam_loss, adam_error, correct_rate_adam = Adam_training()
    print("correct_rate_sgd:", correct_rate_sgd)
    print("correct_rate_adam:", correct_rate_adam)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax[0].plot(sgd_loss)
    ax[0].grid(True)
    ax[0].set_xlabel("Iter", fontsize=14)
    ax[0].set_ylabel("Hinge Loss", fontsize=14)
    ax[0].set_title("SGD", fontsize=12)
    ax[1].plot(adam_loss)
    ax[1].grid(True)
    ax[1].set_xlabel("Iter", fontsize=14)
    ax[1].set_ylabel("Hinge Loss", fontsize=14)
    ax[1].set_title("Adam", fontsize=12)
    plt.savefig("hinge loss.png")

    fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax[0].plot(sgd_error)
    ax[0].grid(True)
    ax[0].set_xlabel("Iter", fontsize=14)
    ax[0].set_ylabel("Prediction Error", fontsize=14)
    ax[0].set_title("SGD", fontsize=12)
    ax[1].plot(adam_error)
    ax[1].grid(True)
    ax[1].set_xlabel("Iter", fontsize=14)
    ax[1].set_ylabel("Prediction Error", fontsize=14)
    ax[1].set_title("Adam", fontsize=12)
    plt.savefig("prediction error.png")
    plt.show()

