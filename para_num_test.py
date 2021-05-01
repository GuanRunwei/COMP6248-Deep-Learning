import torch.nn as nn

class QuizNet(nn.Module):
  def __init__(self):
    super().__init__()

    self.convs = nn.Sequential(
        nn.Conv2d(3, 16, (3, 3), padding=1, stride=3),
        nn.ReLU(),
        nn.Conv2d(16, 32, (3, 3), padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.ReLU(),
    )

    self.fcs = nn.Sequential(
        nn.Linear(4*4*32, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )

  def forward(self, x):
    x = self.convs(x)
    x = x.view(x.shape[0], -1)
    x = self.fcs(x)
    return x
model = QuizNet()
print(sum([i.numel() for i in model.parameters()]))