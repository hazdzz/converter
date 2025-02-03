import math
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Union, List, Optional, Tuple
from torch import Size, Tensor
from .. import embedding


class MultiHeadAttention(nn.Module):
    def __init__(self, feat_dim, num_head, value_drop_prob) -> None:
        super(MultiHeadAttention, self).__init__()
        assert num_head >= 1
        assert feat_dim % num_head == 0, 'feat_dim should be divisible by num_head'

        self.feat_dim = feat_dim
        self.num_head = num_head
        self.head_dim = feat_dim // num_head
        self.tau = math.sqrt(self.head_dim)

        self.query_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.key_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.value_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.output_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        
        self.softmax = nn.Softmax(dim=-1)
        self.value_dropout = nn.Dropout(p=value_drop_prob)
    
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.query_linear.weight)
        init.xavier_uniform_(self.key_linear.weight)
        init.xavier_uniform_(self.value_linear.weight)
        init.xavier_uniform_(self.output_linear.weight)

    def split_head(self, input: Tensor) -> Tensor:
        batch_size, seq_len, _ = input.size()
        return input.contiguous().view(batch_size, seq_len, self.num_head, self.head_dim).permute(0, 2, 1, 3)
    
    def concat_head(self, input: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = input.size()
        return input.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.feat_dim)

    def forward(self, input: Tensor) -> Tensor:
        query = self.query_linear(input)
        key = self.key_linear(input)
        value = self.value_linear(input)
        value = self.value_dropout(value)
        
        multihead_query = self.split_head(query)
        multihead_key = self.split_head(key)
        multihead_value = self.split_head(value)
        
        multihead_attn = torch.einsum('bhnd,bhmd->bhnm', multihead_query, multihead_key)
        multihead_attn = multihead_attn / self.tau
        multihead_attn_score = self.softmax(multihead_attn)

        multihead_attn_score = torch.einsum('bhnm,bhmd->bhnd', multihead_attn_score, multihead_value)
        multihead_attn_score_concat = self.concat_head(multihead_attn_score)
        multihead_attn_output = self.output_linear(multihead_attn_score_concat)

        return multihead_attn_output
    

class FeedForward(nn.Module):
    def __init__(self, feat_dim, hid_dim, ffn_drop_prob: float = 0.1) -> None:
        super(FeedForward, self).__init__()
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
        init.zeros_(self.ffn[0].bias)
        init.zeros_(self.ffn[3].bias)

    def forward(self, input: Tensor) -> Tensor:
        return self.ffn(input)
    

class PostLayerNorm(nn.Module):
    def __init__(self, dim, func) -> None:
        super(PostLayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.func = func
    
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.layernorm(self.func(input, **kwargs) + input)
        

class TransformerEncoder(nn.Module):
    def __init__(self, args) -> None:
        super(TransformerEncoder, self).__init__()
        self.embedding = embedding.Embedding(args.pe_type, 
                                             args.pooling_type, 
                                             args.vocab_size, 
                                             args.max_seq_len, 
                                             args.embed_dim, 
                                             args.pe_drop_prob, 
                                             args.embed_drop_prob)
        self.transformer_encoder_block = nn.ModuleList([])
        for _ in range(args.num_block):
            self.transformer_encoder_block.append(nn.ModuleList([
                PostLayerNorm(args.embed_dim, MultiHeadAttention(args.embed_dim, args.num_head, args.value_drop_prob)),
                PostLayerNorm(args.embed_dim, FeedForward(args.embed_dim, args.hidden_dim, args.ffn_drop_prob))
            ]))
    
    def forward(self, input: Tensor) -> Tensor:
        input = self.embedding(input)

        for mhsa, ffn in self.transformer_encoder_block:
            input = mhsa(input)
            input = ffn(input)
        
        return input