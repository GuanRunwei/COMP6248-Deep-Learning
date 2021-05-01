from typing import Tuple
import torch
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
df = df.sample(frac=1)
# add label indices column
mapping = {k: v for v, k in enumerate(df[4].unique())}
df[5] = df[4].map(mapping)

# normalise data
alldata = torch.tensor(df.iloc[:, [0, 1, 2, 3]].values, dtype=torch.float)
alldata = (alldata - alldata.mean(dim=0)) / alldata.var(dim=0)

# create datasets
targets_tr = torch.tensor(df.iloc[:100, 5].values, dtype=torch.long)
targets_va = torch.tensor(df.iloc[100:, 5].values, dtype=torch.long)
data_tr = alldata[:100]
data_va = alldata[100:]

W1, W2 = torch.tensor(torch.rand(4, 12), requires_grad=True), torch.tensor(torch.rand(12, 3), requires_grad=True)
b1, b2 = torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)
lr = 0.1
Eplot_training = []
Eplot_test = []
for epoch in range(1000):
    logits = torch.matmul(torch.relu(torch.matmul(data_tr, W1) + b1), W2) + b2
    cross_entropy = torch.nn.functional.cross_entropy(logits, targets_tr)
    logits_test = torch.matmul(torch.relu(torch.matmul(data_va, W1) + b1), W2) + b2
    cross_entropy_test = torch.nn.functional.cross_entropy(logits_test, targets_va)
    cross_entropy.backward()
    cross_entropy_test.backward()
    W1.data -= lr * W1.grad
    W2.data -= lr * W2.grad
    b1.data -= lr * b1.grad
    b2.data -= lr * b2.grad
    W1.grad.fill_(0)
    W2.grad.fill_(0)
    b1.grad.fill_(0)
    b2.grad.fill_(0)
    Eplot_training.append(cross_entropy.item())
    Eplot_test.append(cross_entropy_test.item())



print("cross_entropy for training:", Eplot_training[-1])
test_targets = torch.matmul(torch.relu(torch.matmul(data_va, W1) + b1), W2) + b2
cross_entropy_test = torch.nn.functional.cross_entropy(test_targets, targets_va)
print("cross_entropy for test:", cross_entropy_test.item())

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
ax[0].plot(Eplot_training)
ax[0].grid(True)
ax[0].set_xlabel("Iter", fontsize=14)
ax[0].set_ylabel("Loss", fontsize=14)
ax[0].set_title("Training Set", fontsize=12)
ax[1].plot(Eplot_test)
ax[1].grid(True)
ax[1].set_xlabel("Iter", fontsize=14)
ax[1].set_ylabel("Loss", fontsize=14)
ax[1].set_title("Test Set", fontsize=12)
plt.savefig("MLP_test.png")
plt.show()


