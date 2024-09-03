import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class BilinearFeedForward(nn.Module):
    def __init__(self, length, feat_dim, bffn_drop_prob) -> None:
        super(BilinearFeedForward, self).__init__()
        self.length = length
        self.query_real_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.query_imag_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.key_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.value_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.bffn_dropout = nn.Dropout(p=bffn_drop_prob)
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.query_real_linear.weight, gain=1.0)
        init.xavier_uniform_(self.query_imag_linear.weight, gain=1.0)
        init.xavier_uniform_(self.key_linear.weight, gain=1.0)
        init.xavier_uniform_(self.value_linear.weight, gain=1.0)
        init.zeros_(self.query_real_linear.bias)
        init.zeros_(self.query_imag_linear.bias)
        init.zeros_(self.key_linear.bias)
        init.zeros_(self.value_linear.bias)

    def forward(self, input: Tensor) -> Tensor:
        if torch.is_complex(input):
            input_real, input_imag = input.real, input.imag
        else:
            input_real, input_imag = input, input
        
        query_real = self.query_real_linear(input_real)
        query_imag = self.query_imag_linear(input_imag)

        query = torch.mul(query_real, torch.tanh(self.softplus(query_imag)))
        query = self.bffn_dropout(query)

        key = self.key_linear(input_real)
        value = self.value_linear(input_imag)

        key_norm = key / (torch.norm(key, dim=1, keepdim=True) + 1e-5)
        value_norm = value / (torch.norm(value, dim=1, keepdim=True) + 1e-5)

        kv_attn = torch.einsum('bnd,bne->bde', key_norm, value_norm)
        kv_attn = self.relu(kv_attn / math.sqrt(self.length))

        bffn = torch.einsum('bnd,bde->bne', query, kv_attn)

        return bffn