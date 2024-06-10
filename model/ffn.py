import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class FFN(nn.Module):
    def __init__(self, feat_dim, hid_dim, ffn_drop_prob) -> None:
        super(FFN, self).__init__()
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim # hid_dim >= feat_dim
        self.key = nn.Parameter(torch.empty((feat_dim, hid_dim)))
        self.value_trs = nn.Parameter(torch.empty((hid_dim, feat_dim)))
        self.gelu = nn.GELU()
        self.ffn_dropout = nn.Dropout(p=ffn_drop_prob)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.key, mean=0, std=1 / self.hid_dim)
        init.normal_(self.value_trs, mean=0, std=1 / self.hid_dim)

    def forward(self, x) -> Tensor:
        key = torch.einsum('bnd,de->bne', x, self.key)
        key = self.gelu(key)
        key = self.ffn_dropout(key)
        ffn = torch.einsum('bne,ed->bnd', key, self.value_trs)

        return ffn


class BFFN(nn.Module):
    def __init__(self, feat_dim, bffn_drop_prob) -> None:
        super(BFFN, self).__init__()
        self.feat_dim = feat_dim
        self.weight_query_real = nn.Parameter(torch.empty((feat_dim, feat_dim)))
        self.weight_query_imag = nn.Parameter(torch.empty((feat_dim, feat_dim)))
        self.weight_key = nn.Parameter(torch.empty((feat_dim, feat_dim)))
        self.weight_value = nn.Parameter(torch.empty((feat_dim, feat_dim)))
        self.bffn_dropout = nn.Dropout(p=bffn_drop_prob)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.weight_query_real, mean=0, std=1 / self.feat_dim)
        init.normal_(self.weight_query_imag, mean=0, std=1 / self.feat_dim)
        init.normal_(self.weight_key, mean=0, std=1 / self.feat_dim)
        init.normal_(self.weight_value, mean=0, std=1 / self.feat_dim)
    
    def forward(self, x) -> Tensor:
        query_real = torch.einsum('bnd,de->bne', x.real, self.weight_query_real)
        query_imag = torch.einsum('bnd,de->bne', x.imag, self.weight_query_imag)
        query = query_real * query_imag

        key = torch.einsum('bnd,de->bne', x.real, self.weight_key)
        value = torch.einsum('bnd,de->bne', x.imag, self.weight_value)

        kv_attn = torch.einsum('bnd,bne->bde', key, value)
        kv_attn = self.bffn_dropout(kv_attn)

        bffn = torch.einsum('bnd,bde->bne', query, kv_attn)

        return bffn