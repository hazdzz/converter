import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class BilinearFeedForward(nn.Module):
    def __init__(self, feat_dim, bffn_drop_prob) -> None:
        super(BilinearFeedForward, self).__init__()
        self.weight_query_real = nn.Parameter(torch.empty((feat_dim, feat_dim)))
        self.weight_query_imag = nn.Parameter(torch.empty((feat_dim, feat_dim)))
        self.weight_key = nn.Parameter(torch.empty((feat_dim, feat_dim)))
        self.weight_value = nn.Parameter(torch.empty((feat_dim, feat_dim)))
        self.relu = nn.ReLU()
        self.bffn_dropout = nn.Dropout(p=bffn_drop_prob)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.weight_query_real, gain=1.0)
        init.xavier_uniform_(self.weight_query_imag, gain=1.0)
        init.xavier_uniform_(self.weight_key, gain=1.0)
        init.xavier_uniform_(self.weight_value, gain=1.0)

    def forward(self, x) -> Tensor:
        if torch.is_complex(x):
            x_real, x_imag = x.real, x.imag
        else:
            x_real, x_imag = x, x
        
        query_real = torch.einsum('bnd,de->bne', x_real, self.weight_query_real)
        query_imag = torch.einsum('bnd,de->bne', x_imag, self.weight_query_imag)
        # query = query_real * query_imag
        query = self.relu(query_real) * self.relu(query_imag)
        query = self.bffn_dropout(query)

        key = torch.einsum('bnd,de->bne', x_real, self.weight_key)
        value = torch.einsum('bnd,de->bne', x_imag, self.weight_value)
        key = key / (torch.norm(key, dim=1, keepdim=True) + 1e-5)
        value = value / (torch.norm(value, dim=1, keepdim=True) + 1e-5)
        kv_attn = torch.einsum('bnd,bne->bde', key, value)
        kv_attn = self.relu(kv_attn)

        bffn = torch.einsum('bnd,bde->bne', query, kv_attn)

        return bffn