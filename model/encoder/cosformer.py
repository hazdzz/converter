import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from .. import embedding


class CosformerMultiHeadAttention(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        feat_dim: int,
        num_head: int,
        act_fun: str = "relu",
        value_drop_prob: float = 0.1
    ):
        super(CosformerMultiHeadAttention, self).__init__()
        assert act_fun in ['relu', 'elu', 'softplus']

        self.max_seq_len = max_seq_len
        self.feat_dim = feat_dim
        self.num_head = num_head
        self.head_dim = feat_dim // num_head
        self.act_fun = act_fun

        self.query_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.key_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.value_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.output_linear = nn.Linear(feat_dim, feat_dim, bias=False)

        self.value_dropout = nn.Dropout(p=value_drop_prob)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.query_linear.weight)
        init.xavier_uniform_(self.key_linear.weight)
        init.xavier_uniform_(self.value_linear.weight)
        init.xavier_uniform_(self.output_linear.weight)

    def get_index(self, seq_len):
        index = math.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)

    def forward(self, input: Tensor) -> Tensor:
        b, _, _ = input.size()

        query = self.query_linear(input)
        key = self.key_linear(input)
        value = self.value_linear(input)
        value = self.value_dropout(value)

        if self.act_fun == 'elu':
            query = F.elu(query) + 1
            key = F.elu(key) + 1
        elif self.act_fun == 'relu':
            query = F.relu(query)
            key = F.relu(key)
        elif self.act_fun == 'softplus':
            query = F.softplus(query)
            key = F.softplus(key)

        weight_index = self.get_index(self.max_seq_len).to(query)

        query_ = torch.cat([query * torch.sin(weight_index / self.max_seq_len), query * torch.cos(weight_index / self.max_seq_len)], dim=-1)
        key_ = torch.cat([key * torch.sin(weight_index / self.max_seq_len), key * torch.cos(weight_index / self.max_seq_len)], dim=-1)

        kv_ = torch.einsum('bnd,bnm->bdm', key_, value)
        z_ = 1 / torch.clamp_min(torch.einsum('bnd,bd->bn', query_, torch.sum(key_, axis=1)), 1e-6)
        attn_output = torch.einsum('bnd,bdm,bn->bnm', query_, kv_, z_)
        attn_output = attn_output.reshape(b, self.num_head, self.max_seq_len, self.head_dim).transpose(1, 2).reshape(b, self.max_seq_len, self.feat_dim)

        return attn_output
    

class PostLayerNorm(nn.Module):
    def __init__(self, dim, func) -> None:
        super(PostLayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.func = func
    
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.layernorm(self.func(input, **kwargs) + input)
    

class PreLayerNorm(nn.Module):
    def __init__(self, dim, func):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.func = func

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.layer_norm(self.func(x, **kwargs)) + x
    

class FeedForwardNetwork(nn.Module):
    def __init__(self, feat_dim: int, hid_dim: int, ffn_drop_prob: float = 0.1) -> None:
        super(FeedForwardNetwork, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_features=feat_dim, out_features=hid_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=ffn_drop_prob),
            nn.Linear(in_features=hid_dim, out_features=feat_dim, bias=True)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.ffn[0].weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_normal_(self.ffn[3].weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, input: Tensor) -> Tensor:
        ffn_output = self.ffn(input)

        return ffn_output
    

class CosformerEncoder(nn.Module):
    def __init__(self, args) -> None:
        super(CosformerEncoder, self).__init__()
        self.embedding = embedding.Embedding(args.pe_type, 
                                             args.pooling_type, 
                                             args.vocab_size, 
                                             args.max_seq_len, 
                                             args.embed_dim, 
                                             args.pe_drop_prob, 
                                             args.embed_drop_prob)
        self.cosformer_encoder_block = nn.ModuleList([])
        for _ in range(args.num_block):
            self.cosformer_encoder_block.append(nn.ModuleList([
                PostLayerNorm(args.embed_dim, CosformerMultiHeadAttention(args.max_seq_len, args.embed_dim, args.num_head, args.xformer.cosformer.act_func, args.value_drop_prob)),
                PostLayerNorm(args.embed_dim, FeedForwardNetwork(args.embed_dim, args.hidden_dim, args.ffn_drop_prob))
            ]))
    
    def forward(self, input: Tensor) -> Tensor:
        input = self.embedding(input)

        for mhca, ffn in self.cosformer_encoder_block:
            input = mhca(input)
            input = ffn(input)
        
        return input