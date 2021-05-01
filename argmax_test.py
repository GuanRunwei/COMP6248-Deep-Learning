import numpy as np
import torch

b = torch.randn(4, 5)
result = torch.argmax(b, 1)
print(b)
print(result)



