import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms


def set_env(seed = 3407) -> None:
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

set_env(42)

a = torch.rand((2, 3))
a = torch.sin(a)
b = torch.fft.fft2(a)
c = b.real - b.imag

mean = c.mean()
std = c.std()
normalize = transforms.Normalize(mean=[mean], std=[std])
c = c.unsqueeze(0)
d = normalize(c).squeeze(0)
d_min = d.min()
d_max = d.max()
d = 2 * (d - d_min) / (d_max - d_min) - 1

print(d.type())