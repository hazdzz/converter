import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from typing import Optional
from .. import embedding


class LinformerMultiHeadSelfAttention(nn.Module):
    def __init__(self, seq_len, feat_dim, num_head, proj_dim, value_drop_prob, para_share_schema) -> None:
        super(LinformerMultiHeadSelfAttention, self).__init__()
        assert proj_dim <= feat_dim
        assert num_head >= 1
        assert para_share_schema in ['head', 'kv']

        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.num_head = num_head
        self.head_dim = feat_dim // num_head
        self.tau = math.sqrt(self.head_dim)
        self.proj_dim = proj_dim
        self.para_share_schema = para_share_schema
        self.value_dropout = nn.Dropout(p=value_drop_prob)
        self.query_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.key_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.value_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        if para_share_schema == 'head':
            self.proj_weight_e = nn.Parameter(torch.empty(self.seq_len, proj_dim), requires_grad=False)
            self.proj_weight_f = nn.Parameter(torch.empty(self.seq_len, proj_dim), requires_grad=False)
        else:
            self.proj_weight_kv = nn.Parameter(torch.empty(self.seq_len, proj_dim), requires_grad=False)
        self.output_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
    
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.query_linear.weight, gain=1.0)
        init.xavier_uniform_(self.key_linear.weight, gain=1.0)
        init.xavier_uniform_(self.value_linear.weight, gain=1.0)
        init.xavier_uniform_(self.output_linear.weight, gain=1.0)

        if self.para_share_schema == 'head':
            init.normal_(self.proj_weight_e, mean=0, std=math.sqrt(1 / self.seq_len))
            init.normal_(self.proj_weight_f, mean=0, std=math.sqrt(1 / self.seq_len))
        elif self.para_share_schema == 'kv':
            init.normal_(self.proj_weight_kv, mean=0, std=math.sqrt(1 / self.seq_len))

    def split(self, batch_size, input: Tensor) -> Tensor:
        return input.view(batch_size, self.seq_len, self.num_head, self.head_dim).transpose(1, 2)
    
    def concat(self, batch_size, input: Tensor) -> Tensor:
        return input.transpose(1, 2).contiguous().view(batch_size, self.seq_len, self.feat_dim)

    def forward(self, input: Tensor) -> Tensor:
        bs, _, _ = input.size()

        query = self.query_linear(input)
        key = self.key_linear(input)
        value = self.value_linear(input)

        query_multihead = self.split(bs, query)
        key_multihead = self.split(bs, key)
        value_multihead = self.split(bs, value)

        if self.para_share_schema == 'head':
            key_multihead_proj = torch.einsum('bhnd,ne->bhed', key_multihead, self.proj_weight_e)
            value_multihead_proj = torch.einsum('bhnd,nf->bhfd', value_multihead, self.proj_weight_f)
        else:
            key_multihead_proj = torch.einsum('bhnd,ne->bhed', key_multihead, self.proj_weight_kv)
            value_multihead_proj = torch.einsum('bhnd,ne->bhed', value_multihead, self.proj_weight_kv)

        value_multihead_proj = self.value_dropout(value_multihead_proj)
        
        qk_attn_multihead_proj = torch.einsum('bhnd,bhed->bhne', query_multihead, key_multihead_proj)
        qk_attn_multihead_proj_score = self.softmax(qk_attn_multihead_proj / self.tau)

        linformer_multihead_attn_score = torch.einsum('bhne,bhed->bhnd', qk_attn_multihead_proj_score, value_multihead_proj)
        linformer_multihead_attn_score_concat = self.concat(bs, linformer_multihead_attn_score)
        linformer_multihead_attn_output = self.output_linear(linformer_multihead_attn_score_concat)

        return linformer_multihead_attn_output


class LinformerMultiHeadSelfAttentionProjectLayerwise(nn.Module):
    def __init__(self, seq_len, feat_dim, num_head, proj_dim, value_drop_prob, para_share_schema) -> None:
        super(LinformerMultiHeadSelfAttentionProjectLayerwise, self).__init__()
        assert proj_dim <= feat_dim
        assert num_head >= 1
        assert para_share_schema == 'layer'

        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.num_head = num_head
        self.head_dim = feat_dim // num_head
        self.proj_dim = proj_dim
        self.para_share_schema = para_share_schema
        self.value_dropout = nn.Dropout(p=value_drop_prob)
        self.query_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.key_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.value_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.output_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
    
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.query_linear.weight, gain=1.0)
        init.xavier_uniform_(self.key_linear.weight, gain=1.0)
        init.xavier_uniform_(self.value_linear.weight, gain=1.0)
        init.xavier_uniform_(self.output_linear.weight, gain=1.0)

    def split(self, batch_size, input: Tensor) -> Tensor:
        return input.view(batch_size, self.seq_len, self.num_head, self.head_dim).transpose(1, 2)
    
    def concat(self, batch_size, input: Tensor) -> Tensor:
        return input.transpose(1, 2).contiguous().view(batch_size, self.seq_len, self.feat_dim)

    def forward(self, input: Tensor, proj_weight_all: Tensor) -> Tensor:
        bs, _, _ = input.size()

        query = self.query_linear(input)
        key = self.key_linear(input)
        value = self.value_linear(input)

        query_multihead = self.split(bs, query)
        key_multihead = self.split(bs, key)
        value_multihead = self.split(bs, value)

        key_multihead_proj = torch.einsum('bhnd,ne->bhed', key_multihead, proj_weight_all)
        value_multihead_proj = torch.einsum('bhnd,nf->bhfd', value_multihead, proj_weight_all)
        value_multihead_proj = self.value_dropout(value_multihead_proj)
        qk_attn_multihead_proj = torch.einsum('bhnd,bhed->bhne', query_multihead, key_multihead_proj)
        qk_attn_multihead_proj_score = self.softmax(qk_attn_multihead_proj / math.sqrt(self.head_dim))
        linformer_multihead_attn_score = torch.einsum('bhne,bhed->bhnd', qk_attn_multihead_proj_score, value_multihead_proj)
        linformer_multihead_attn_score_concat = self.concat(bs, linformer_multihead_attn_score)
        linformer_multihead_attn_output = self.output_linear(linformer_multihead_attn_score_concat)

        return linformer_multihead_attn_output


class FeedForwardNetwork(nn.Module):
    def __init__(self, feat_dim, hid_dim, ffn_drop_prob) -> None:
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


class PostLayerNormAttention(nn.Module):
    def __init__(self, dim, func) -> None:
        super(PostLayerNormAttention, self).__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.func = func
    
    def forward(self, input1: Tensor, input2: Optional[Tensor] = None, **kwargs) -> Tensor:
        return self.layernorm(self.func(input1, input2, **kwargs) + input1)
    

class PostLayerNormFFN(nn.Module):
    def __init__(self, dim, func) -> None:
        super(PostLayerNormFFN, self).__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.func = func
    
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.layernorm(self.func(input, **kwargs) + input)


class LinformerEncoder(nn.Module):
    def __init__(self, args) -> None:
        super(LinformerEncoder, self).__init__()
        assert args.xformer.linformer.para_share_schema in ['head', 'kv', 'layer']

        self.num_block = args.num_block
        self.max_seq_len = args.max_seq_len
        self.head_dim = args.embed_dim // args.num_head
        self.para_share_schema = args.xformer.linformer.para_share_schema

        self.embedding = embedding.Embedding(args.pe_type, 
                                             args.pooling_type, 
                                             args.vocab_size, 
                                             args.max_seq_len, 
                                             args.embed_dim, 
                                             args.pe_drop_prob, 
                                             args.embed_drop_prob)
        self.linformer_encoder_block = nn.ModuleList([])
        if self.para_share_schema == 'head' or self.para_share_schema == 'kv':
            for _ in range(args.num_block):
                self.linformer_encoder_block.append(nn.ModuleList([
                    PostLayerNormAttention(args.embed_dim, LinformerMultiHeadSelfAttention(args.max_seq_len, 
                                                                                  args.embed_dim, 
                                                                                  args.num_head, 
                                                                                  args.xformer.linformer.proj_dim,  
                                                                                  args.value_drop_prob, 
                                                                                  args.xformer.linformer.para_share_schema)),
                    PostLayerNormFFN(args.embed_dim, FeedForwardNetwork(args.embed_dim, args.hidden_dim, args.ffn_drop_prob))
            ]))
        else:
            self.proj_weight_all = nn.Parameter(torch.empty(args.max_seq_len, args.xformer.linformer.proj_dim), requires_grad=False)
            self.reset_parameters()

            for _ in range(args.num_block):
                self.linformer_encoder_block.append(nn.ModuleList([
                    PostLayerNormAttention(args.embed_dim, LinformerMultiHeadSelfAttentionProjectLayerwise(args.max_seq_len, 
                                                                                                   args.embed_dim, 
                                                                                                   args.num_head, 
                                                                                                   args.xformer.linformer.proj_dim, 
                                                                                                   args.value_drop_prob, 
                                                                                                   args.xformer.linformer.para_share_schema)),
                    PostLayerNormFFN(args.embed_dim, FeedForwardNetwork(args.embed_dim, args.hidden_dim, args.ffn_drop_prob))
            ]))

    def reset_parameters(self) -> None:
        if self.para_share_schema == 'layer':
            init.normal_(self.proj_weight_all, mean=0, std=math.sqrt(1 / self.max_seq_len))

    def forward(self, input: Tensor) -> Tensor:
        input = self.embedding(input)

        if self.para_share_schema == 'head' or self.para_share_schema == 'kv':
            for mhla, ffn in self.linformer_encoder_block:
                input = mhla(input)
                input = ffn(input)
        else:
            for mhla, ffn in self.linformer_encoder_block:
                input = mhla(input, self.proj_weight_all)
                input = ffn(input)

        return input