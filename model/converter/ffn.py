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
        self.value_real_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.value_imag_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.key_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.query_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.bffn_dropout = nn.Dropout(p=bffn_drop_prob)
        self.act_func = act_func
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.value_real_linear.weight, gain=1.0)
        init.xavier_uniform_(self.value_imag_linear.weight, gain=1.0)
        init.xavier_uniform_(self.key_linear.weight, gain=1.0)
        init.xavier_uniform_(self.query_linear.weight, gain=1.0)
        init.zeros_(self.value_real_linear.bias)
        init.zeros_(self.value_imag_linear.bias)
        init.zeros_(self.key_linear.bias)
        init.zeros_(self.query_linear.bias)

    def forward(self, input: Tensor) -> Tensor:
        if torch.is_complex(input):
            input_real, input_imag = input.real, input.imag
        else:
            input_real, input_imag = input, input
        
        value_real = self.value_real_linear(input_real)
        value_imag = self.value_imag_linear(input_imag)

        if self.act_func == 'glu':
            # tanh(softplus(x)) is a sigmoid-like function.
            value = torch.mul(value_real, torch.tanh(self.softplus(value_imag)))
        elif self.act_func == 'gtu':
            # We replace sigmoid in GTU with softplus.
            # Unlike Dauphin et al. who think tanh hinders back-propagation,
            # we found that sigmoid is what hinders back-propagation.
            value = torch.mul(torch.tanh(value_real), self.softplus(value_imag))
        else:
            value = torch.mul(value_real, value_imag)

        value = self.bffn_dropout(value)

        key = self.key_linear(input_real)
        query = self.query_linear(input_imag)

        key_norm = key / (torch.norm(key, dim=1, keepdim=True) + 1e-5)
        query_norm = query / (torch.norm(query, dim=1, keepdim=True) + 1e-5)

        feature_attn = torch.einsum('bnd,bne->bde', key_norm, query_norm)
        feature_attn = self.relu(feature_attn)
        
        gffn = torch.einsum('bnd,bde->bne', value, feature_attn)

        return gffn