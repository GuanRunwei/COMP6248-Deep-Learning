import torch
import torchbearer
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


# fix random seed for reproducibility
seed = 7
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(seed)


# flatten 28*28 images to a 784 vector for each image
transform = transforms.Compose([
    transforms.ToTensor(),  # convert to tensor
    transforms.Lambda(lambda x: x.view(-1))  # flatten into vector
])


trainset = MNIST(".", train=True, download=True, transform=transform)
testset = MNIST(".", train=False, download=True, transform=transform)


# create data loaders
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=True)


class_counts = torch.zeros(10, dtype=torch.int32)
for (images, labels) in trainloader:
    for item in images:
      class_counts += 1



# define baseline model
class BaselineModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BaselineModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        if not self.training:
            out = F.softmax(out, dim=1)
        return out



# build the model
model = BaselineModel(784, 784, 10)

# define the loss function and the optimiser
loss_function = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters())

# the epoch loop
for epoch in range(10):
    running_loss = 0.0
    for data in trainloader:
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimiser.zero_grad()

        # forward + loss + backward + optimise (update weights)
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimiser.step()

        # keep track of the loss this epoch
        running_loss += loss.item()
    print("Epoch %d, loss %4.2f" % (epoch, running_loss))
print('**** Finished Training ****')

model.eval()

# Compute the model accuracy on the test set
correct = 0
total = 0

# YOUR CODE HERE
for images, labels in testloader:
    output = model(images)

    _, pred = output.max(1)
    pred_correct_num = (pred == labels).sum().item()
    correct += pred_correct_num
    total += len(images)

print('Test Accuracy: %2.2f %%' % ((100.0 * correct) / total))

# Compute the model accuracy on the test set
class_correct = torch.zeros(10)
class_total = torch.zeros(10)
confusion_matrix = torch.zeros(10, 10)

# YOUR CODE HERE
for images, labels in testloader:
    outputs = model(images)
    outputs = F.softmax(outputs)
    results = []
    for i in outputs:
        results.append(int(torch.argmax(i)))
    results = torch.tensor(results)
    for t, p in zip(labels.view(-1), results.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

class_correct = confusion_matrix.diag()
class_total = confusion_matrix.sum(1)

for i in range(10):
    print('Class %d accuracy: %2.2f %%' % (i, 100.0 * class_correct[i] / class_total[i]))