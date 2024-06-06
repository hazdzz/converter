import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class BFFN(nn.Module):
    def __init__(self, feat_dim, hid_dim, ffn_drop_prob) -> None:
        super(BFFN, self).__init__()
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim # hid_dim >= feat_dim
        self.weight_key_real = nn.Parameter(torch.empty((feat_dim, hid_dim)))
        self.weight_key_imag = nn.Parameter(torch.empty((feat_dim, hid_dim)))
        self.value_trs = nn.Parameter(torch.empty((hid_dim, feat_dim)))
        self.bffn_dropout = nn.Dropout(p=ffn_drop_prob)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.weight_key_real, mean=0, std=1 / self.hid_dim)
        init.normal_(self.weight_key_imag, mean=0, std=1 / self.hid_dim)
        init.normal_(self.value_trs, mean=0, std=1 / self.hid_dim)

    def forward(self, x) -> Tensor:
        if torch.is_complex(x):
            x_real, x_imag = x.real, x.imag
        else:
            x_real, x_imag = x, -x
        key_real = torch.einsum('bnd,de->bne', x_real, self.weight_key_real)
        key_imag = torch.einsum('bnd,de->bne', x_imag, self.weight_key_imag)
        key = key_real * key_imag
        key = self.bffn_dropout(key)
        bffn = torch.einsum('bne,ed->bnd', key, self.value_trs)

        return bffn