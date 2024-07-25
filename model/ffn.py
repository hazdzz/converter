import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class BilinearFeedForward(nn.Module):
    def __init__(self, length, feat_dim, bffn_drop_prob) -> None:
        super(BilinearFeedForward, self).__init__()
        self.length = length
        self.feat_dim = feat_dim
        self.weight_query_real = nn.Parameter(torch.empty((feat_dim, feat_dim)))
        self.weight_query_imag = nn.Parameter(torch.empty((feat_dim, feat_dim)))
        self.weight_key = nn.Parameter(torch.empty((feat_dim, feat_dim)))
        self.weight_value = nn.Parameter(torch.empty((feat_dim, feat_dim)))
        self.bias = nn.Parameter(torch.empty(feat_dim))
        self.bffn_dropout = nn.Dropout(p=bffn_drop_prob)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.weight_query_real, gain=1.0)
        init.xavier_uniform_(self.weight_query_imag, gain=1.0)
        init.xavier_uniform_(self.weight_key, gain=1.0)
        init.xavier_uniform_(self.weight_value, gain=1.0)
        init.zeros_(self.bias)
    
    def forward(self, x) -> Tensor:
        query_real = torch.einsum('bnd,de->bne', x.real, self.weight_query_real)
        query_imag = torch.einsum('bnd,de->bne', x.imag, self.weight_query_imag)
        query = query_real * query_imag
        query = self.bffn_dropout(query)

        key = torch.einsum('bnd,de->bne', x.real, self.weight_key)
        value = torch.einsum('bnd,de->bne', x.imag, self.weight_value)
        key = key / (torch.norm(key, dim=1, keepdim=True)).clamp(min=1e-5)
        value = value / (torch.norm(value, dim=1, keepdim=True)).clamp(min=1e-5)
        kv_attn = torch.einsum('bnd,bne->bde', key, value)

        bffn = torch.einsum('bnd,bde->bne', query, kv_attn)
        bffn = bffn + self.bias

        return bffn