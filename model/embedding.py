import math
import torch
import torch.nn as nn
import torch.nn.init as init
from .norm import ScaleNorm
from torch import Tensor


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, length: int = 512, d_model: int = 64) -> Tensor:
        super(SinusoidalPositionEmbedding, self).__init__()
        pe = torch.zeros(length, d_model)
        
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input: Tensor) -> Tensor:
        return self.pe[:, :input.size(1)].to(input.device)


# class Conv1DPositionEmbedding(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, 
#                  stride=1, padding='same', dilation=1, groups=1, 
#                  bias=False, padding_mode='zeros') -> None:
#         super(Conv1DPositionEmbedding, self).__init__()
#         self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, 
#                                 stride, padding, dilation, groups, 
#                                 bias, padding_mode
#                                 )

#     def forward(self, input: Tensor) -> Tensor:
#         return self.conv1d(input.permute(0, 2, 1)).permute(0, 2, 1)

class Conv1DPositionEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding='same', dilation=1, 
                 bias=False, padding_mode='zeros') -> None:
        super(Conv1DPositionEmbedding, self).__init__()

        self.depthwise_conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                stride, padding, dilation, groups=in_channels, 
                                bias=bias, padding_mode=padding_mode
                                )
        self.pointwise_conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                                          stride=1, padding=0, dilation=1, groups=1, 
                                          bias=bias, padding_mode=padding_mode)
        self.relu = nn.ReLU()

    def forward(self, input: Tensor) -> Tensor:
        input_dwconv1d = self.depthwise_conv1d(input.permute(0, 2, 1))
        input_dwconv1d = self.relu(input_dwconv1d)
        output = self.pointwise_conv1d(input_dwconv1d).permute(0, 2, 1)

        return output

class ConverterEmbedding(nn.Module):
    def __init__(self, pe_type, pooling_type, vocab_size, max_seq_len, embed_dim, 
                 embed_drop_prob) -> None:
        super(ConverterEmbedding, self).__init__()
        assert pe_type in ['nope', 'spe', 'ape', 'cope']

        self.pe_type = pe_type
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        if pooling_type == "CLS":
            padding_idx = vocab_size - 2
        else:
            padding_idx = None
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.sin_pos_embed = SinusoidalPositionEmbedding(max_seq_len, embed_dim)
        self.conv1d = Conv1DPositionEmbedding(in_channels=embed_dim, 
                                              out_channels=embed_dim, 
                                              kernel_size=embed_dim, 
                                              dilation=1
                                              )
        self.embed_dropout = nn.Dropout(p=embed_drop_prob)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.token_embed.weight, mean=0, std=math.sqrt(1 / self.max_seq_len))
        init.normal_(self.pos_embed.weight, mean=0, std=math.sqrt(1 / self.max_seq_len))

    def forward(self, input: Tensor) -> Tensor:
        token_embed = self.token_embed(input)
        if self.pe_type == 'nope':
            # No Position Embedding
            embed = token_embed
        elif self.pe_type == 'spe':
            # Sinusoidal Positional Encoding
            pos_embed = self.sin_pos_embed(token_embed)
            embed = token_embed + pos_embed
        elif self.pe_type == 'ape':
            # Absolute Learnable Position Embedding
            pos_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=input.device)
            pos_ids = pos_ids.expand(input.size(0), self.max_seq_len)
            pos_embed = self.pos_embed(pos_ids)
            embed = token_embed + pos_embed
        elif self.pe_type == 'cope':
            # Convolutional Position Embedding
            pos_embed = self.conv1d(token_embed)
            embed = token_embed + pos_embed
        else:
            raise ValueError(f'ERROR: The Position Embedding {self.pe} is not implemented yet.')

        embed = self.embed_dropout(embed)

        return embed