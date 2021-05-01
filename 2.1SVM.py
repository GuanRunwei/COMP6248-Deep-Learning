import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import torch
import pandas as pd
from celluloid import Camera
from IPython.display import HTML


def hinge_loss(y_pred, y_true):
    return torch.mean(torch.clamp(1-y_pred.t() * y_true, min=0))



def svm(x, w, b):
    h = (w*x).sum(1) + b
    return h


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
print(len(targets_tr))
data_tr = alldata[:75]
data_va = alldata[75:]

# Set up drawing
fig = plt.figure(figsize=(5, 5))
camera = Camera(fig)


#  训练
w = torch.randn(1, 4, requires_grad=True)
b = torch.randn(1, requires_grad=True)

opt = torch.optim.SGD([w, b], lr=0.01, weight_decay=0.0001)
loss_record = []
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


# 测试
test_loss_record = []
correct_percentage, correct_number = float, 0

for i in range(len(data_va)):
    pred = svm(data_va[i], w, b)
    pred_bool = 1 if pred > 0 else -1
    if pred_bool == targets_va[i]:
        correct_number += 1
correct_percentage = correct_number / len(data_va) * 1.0
print("correct percentage:", correct_percentage)



plt.plot(loss_record)
plt.grid(True)
plt.xlabel("iter", fontsize=14)
plt.ylabel("loss value", fontsize=14)
plt.title("SGD(training set)", fontsize=12)

plt.show()



