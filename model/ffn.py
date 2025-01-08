import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class BinaryGatingUnit(nn.Module):
    def __init__(self) -> None:
        super(BinaryGatingUnit, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        p = torch.sigmoid(input)
        bern = torch.bernoulli(p)
        eps = torch.where(bern == 1, 1 - p, -p)
        gate = p + eps
        bgu = input * gate

        return bgu
    

class SignGatingUnit(nn.Module):
    def __init__(self) -> None:
        super(SignGatingUnit, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, input: Tensor) -> Tensor:
        temp = torch.tanh(input)
        p = 1 - temp ** 2
        bern = torch.bernoulli(p)
        eps = torch.where(bern == 1, 1 - temp, -1 - temp)
        gate = temp + eps
        sgu = self.relu(input) * gate

        return sgu


class GatedFeedForward(nn.Module):
    def __init__(self, feat_dim: int, hid_dim: int, gffn_drop_prob: float = 0.1) -> None:
        super(GatedFeedForward, self).__init__()
        self.linear1_real = nn.Linear(feat_dim, hid_dim, bias=True)
        self.linear1_imag = nn.Linear(feat_dim, hid_dim, bias=True)
        self.linear2 = nn.Linear(hid_dim, feat_dim, bias=True)
        self.softplus = nn.Softplus(beta=1.0, threshold=5.0)
        self.gffn_dropout = nn.Dropout(p=gffn_drop_prob)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.linear1_real.weight, gain=1.0)
        init.xavier_uniform_(self.linear1_imag.weight, gain=1.0)
        init.xavier_uniform_(self.linear2.weight, gain=1.0)
        init.zeros_(self.linear1_real.bias)
        init.zeros_(self.linear1_imag.bias)
        init.zeros_(self.linear2.bias)

    def forward(self, input: Tensor) -> Tensor:
        if input.is_complex():
            input_real, input_imag = input.real, input.imag
        else:
            input_real, input_imag = input, input

        linear1_real = self.linear1_real(input_real)
        linear1_imag = self.linear1_imag(input_imag)
        linear = self.softplus(linear1_real) * torch.tanh(linear1_imag)
        linear = self.gffn_dropout(linear)
        ffn = self.linear2(linear)

        return ffn