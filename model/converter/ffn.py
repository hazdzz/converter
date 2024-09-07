import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class GatedFeedForward(nn.Module):
    def __init__(self, length, feat_dim, act_func: str = 'glu', bffn_drop_prob: float = 0.1) -> None:
        super(GatedFeedForward, self).__init__()
        assert act_func in ['glu', 'gtu', 'bilinear']
        
        self.length = length
        self.query_real_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.query_imag_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.key_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.value_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.bffn_dropout = nn.Dropout(p=bffn_drop_prob)
        self.act_func = act_func
        self.softplus = nn.Softplus()
        self.mish = nn.Mish()
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

        if self.act_func == 'glu':
            # tanh(softplus(x)) is a sigmoid-like function.
            query = torch.mul(query_real, torch.tanh(self.softplus(query_imag)))
        elif self.act_func == 'gtu':
            # We replace sigmoid in GTU with softplus.
            # Unlike Dauphin et al. who think tanh hinders back-propagation,
            # we found that sigmoid is what hinders back-propagation.
            query = torch.mul(torch.tanh(query_real), self.softplus(query_imag))
        else:
            query = self.mish(torch.mul(query_real, query_imag))

        query = self.bffn_dropout(query)

        key = self.key_linear(input_real)
        value = self.value_linear(input_imag)

        key_norm = key / (torch.norm(key, dim=1, keepdim=True) + 1e-5)
        value_norm = value / (torch.norm(value, dim=1, keepdim=True) + 1e-5)

        kv_attn = torch.einsum('bnd,bne->bde', key_norm, value_norm)
        kv_attn = self.relu(kv_attn)
        # kv_attn = self.softplus(kv_attn)
        gffn = torch.einsum('bnd,bde->bne', query, kv_attn)

        return gffn