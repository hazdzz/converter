import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import Tensor
from typing import Optional
from .. import embedding


def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z


class NystromMultiHeadAttention(nn.Module):
    def __init__(self, seq_len, feat_dim, num_head, num_landmark, num_iter, conv_kernel_size) -> None:
        super(NystromMultiHeadAttention, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.num_head = num_head
        self.num_landmark = num_landmark
        self.num_iter = num_iter
        self.head_dim = feat_dim // num_head
        self.tau = math.sqrt(self.head_dim)
        self.query_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.key_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.value_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.output_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.conv = nn.Conv2d(in_channels = num_head, out_channels = num_head, 
                              kernel_size = (conv_kernel_size, 1), padding = (conv_kernel_size // 2, 0), 
                              bias = False, groups = num_head)
        
    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.query_linear.weight, gain=1.0)
        init.xavier_uniform_(self.key_linear.weight, gain=1.0)
        init.xavier_uniform_(self.value_linear.weight, gain=1.0)
        init.xavier_uniform_(self.output_linear.weight, gain=1.0)
        init.zeros_(self.query_linear.bias)
        init.zeros_(self.key_linear.bias)
        init.zeros_(self.value_linear.bias)
        init.zeros_(self.output_linear.bias)

    def forward(self, input: Tensor) -> Tensor:
        _, n, _, h, m, iters = *input.shape, self.num_head, self.num_landmark, self.num_iter

        # padding
        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            input = F.pad(input, (0, 0, padding, 0), value=0)

        query, key, value = self.query_linear(input), self.key_linear(input), self.value_linear(input)
        query, key, value = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (query, key, value))
        query = query / self.tau

        l = math.ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        query_landmarks = reduce(query, landmark_einops_eq, 'sum', l = l)
        key_landmarks = reduce(key, landmark_einops_eq, 'sum', l = l)

        query_landmarks = query_landmarks / l
        key_landmarks = key_landmarks / l

        einops_eq = '... i d, ... j d -> ... i j'
        kernel1 = torch.einsum(einops_eq, query, key_landmarks)
        kernel2 = torch.einsum(einops_eq, query_landmarks, key_landmarks)
        kernel3 = torch.einsum(einops_eq, query_landmarks, key)

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (kernel1, kernel2, kernel3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)
        nystrom_attn = (attn1 @ attn2_inv) @ (attn3 @ value)
        nystrom_attn = nystrom_attn + self.conv(value)

        nystrom_attn = rearrange(nystrom_attn, 'b h n d -> b n (h d)', h = h)
        nystrom_attn_out = self.output_linear(nystrom_attn)
        nystrom_attn_out = nystrom_attn_out[:, -n:]

        return nystrom_attn_out


class PreLayerNorm(nn.Module):
    def __init__(self, dim, func):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.func = func

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.layer_norm(self.func(x, **kwargs)) + x


class FeedForwardNetwork(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, ffn_drop_prob: float = 0.1) -> None:
        super(FeedForwardNetwork, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_features=feat_dim, out_features=hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=ffn_drop_prob),
            nn.Linear(in_features=hidden_dim, out_features=feat_dim, bias=True)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.ffn[0].weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_normal_(self.ffn[3].weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, input: Tensor) -> Tensor:
        ffn_output = self.ffn(input)

        return ffn_output


class NystromformerEncoder(nn.Module):
    def __init__(self, args):
        super(NystromformerEncoder, self).__init__()
        self.embedding = embedding.Embedding(args.pe_type, 
                                             args.pooling_type, 
                                             args.vocab_size, 
                                             args.max_seq_len, 
                                             args.embed_dim, 
                                             args.pe_drop_prob, 
                                             args.embed_drop_prob)
        self.nystromformer_encoder_block = nn.ModuleList([])
        for _ in range(args.num_block):
            self.nystromformer_encoder_block.append(nn.ModuleList([
                PreLayerNorm(args.embed_dim, NystromMultiHeadAttention(args.max_seq_len, args.embed_dim, args.num_head, args.xformer.nystromformer.num_landmark, args.xformer.nystromformer.num_iter, args.xformer.nystromformer.conv_kernel_size)),
                PreLayerNorm(args.embed_dim, FeedForwardNetwork(args.embed_dim, args.hidden_dim, args.ffn_drop_prob))
            ]))

    def forward(self, input: Tensor) -> Tensor:
        input = self.embedding(input)

        for mhna, ffn in self.nystromformer_encoder_block:
            input = mhna(input)
            input = ffn(input)
        
        return input