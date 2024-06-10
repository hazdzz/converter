import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .norm import ScaleNorm
from torch import Tensor


class Conv1DPositionEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, 
                 bias=True, padding_mode='zeros') -> None:
        super(Conv1DPositionEmbedding, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride, padding, dilation, groups, 
                              bias, padding_mode
                            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.in_channels >= self.out_channels:
            init.normal_(self.conv.weight, mean=0, std=1 / self.in_channels)
        else:
            init.normal_(self.conv.weight, mean=0, std=1 / self.out_channels)
        if self.bias:
            init.zeros_(self.conv.bias)

    def forward(self, x) -> Tensor:
        x = x.permute(0, 2, 1)

        if self.kernel_size % 2 == 0:
            x = F.pad(x, (self.kernel_size // 2 - 1, self.kernel_size // 2), mode='constant', value=0)
        else:
            x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2), mode='constant', value=0)
        
        x = self.conv(x)

        x = x.permute(0, 2, 1)

        return x

class ConverterEmbedding(nn.Module):
    def __init__(self, pe_type, pooling_type, vocab_size, max_seq_len, embed_dim, 
                 embed_drop_prob) -> None:
        super(ConverterEmbedding, self).__init__()
        assert pe_type in ['nope', 'spe', 'ape', 'cpe']
        self.pe_type = pe_type
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        if pooling_type == "CLS":
            padding_idx = vocab_size - 2
        else:
            padding_idx = None
        self.token_embed = nn.Embedding(
                            vocab_size,
                            embed_dim,
                            padding_idx
                        )
        self.pos_embed = nn.Embedding(
                            max_seq_len, 
                            embed_dim
                        )
        self.conv1d = Conv1DPositionEmbedding(in_channels=embed_dim, 
                                              out_channels=embed_dim, 
                                              kernel_size=embed_dim)
        self.gelu = nn.GELU()
        self.embed_dropout = nn.Dropout(p=embed_drop_prob)
        self.token_embed_norm = ScaleNorm(embed_dim, eps=1e-8)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.token_embed.weight, mean=0, std=1 / self.max_seq_len)
        init.normal_(self.pos_embed.weight, mean=0, std=1 / self.max_seq_len)

    def sinusoidal_positional_encoding(self, d_model, length, device) -> Tensor:
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding \
                             with odd dim (got dim={:d})".format(d_model)
                            )

        pe = torch.zeros(length, d_model)

        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2) \
                              * (-math.log(10000.0) / d_model)))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.to(device)
    
    def forward(self, input) -> Tensor:
        token_embed = self.token_embed(input)
        if self.pe_type == 'nope':
            embed = token_embed
        elif self.pe_type == 'spe':
            pos_embed = self.sinusoidal_positional_encoding(self.embed_dim, self.max_seq_len, input.device)
            embed = token_embed + pos_embed
        elif self.pe_type == 'ape':
            pos_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=input.device)
            pos_ids = pos_ids.expand(input.size(0), self.max_seq_len)
            pos_embed = self.pos_embed(pos_ids)
            embed = token_embed + pos_embed
        elif self.pe_type == 'cpe':
            token_embed_norm = self.token_embed_norm(token_embed)
            pos_embed = self.conv1d(token_embed_norm)
            embed = pos_embed + token_embed
        else:
            raise ValueError(f'ERROR: The Position Embedding {self.pe} is not implemented yet.')
        embed_output = self.embed_dropout(embed)

        return embed_output