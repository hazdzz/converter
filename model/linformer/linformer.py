import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


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
        self.proj_dim = proj_dim
        self.para_share_schema = para_share_schema
        self.value_dropout = nn.Dropout(p=value_drop_prob)
        self.query_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.key_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.value_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        if para_share_schema == 'head':
            self.proj_weight_e = nn.Parameter(torch.empty(self.seq_len, proj_dim), requires_grad=False)
            self.proj_weight_f = nn.Parameter(torch.empty(self.seq_len, proj_dim), requires_grad=False)
        else:
            self.proj_weight_kv = nn.Parameter(torch.empty(self.seq_len, proj_dim), requires_grad=False)
        self.softmax = nn.Softmax(dim=-1)
    
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.query_linear.weight, gain=1.0)
        init.xavier_uniform_(self.key_linear.weight, gain=1.0)
        init.xavier_uniform_(self.value_linear.weight, gain=1.0)
        init.zeros_(self.query_linear.bias)
        init.zeros_(self.key_linear.bias)
        init.zeros_(self.value_linear.bias)

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
        qk_attn_multihead_proj_score = self.softmax(qk_attn_multihead_proj / math.sqrt(self.head_dim))

        linformer_multihead_attn = torch.einsum('bhne,bhed->bhnd', qk_attn_multihead_proj_score, value_multihead_proj)
        linformer_multihead_attn_output = self.concat(bs, linformer_multihead_attn)

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
        self.query_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.key_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.value_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)
    
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.query_linear.weight, gain=1.0)
        init.xavier_uniform_(self.key_linear.weight, gain=1.0)
        init.xavier_uniform_(self.value_linear.weight, gain=1.0)
        init.zeros_(self.query_linear.bias)
        init.zeros_(self.key_linear.bias)
        init.zeros_(self.value_linear.bias)

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
        linformer_multihead_attn = torch.einsum('bhne,bhed->bhnd', qk_attn_multihead_proj_score, value_multihead_proj)
        linformer_multihead_attn_output = self.concat(bs, linformer_multihead_attn)

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


class PostLayerNorm(nn.Module):
    def __init__(self, dim, func) -> None:
        super(PostLayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.func = func
    
    def forward(self, input, **kwargs) -> Tensor:
        return self.layernorm(self.func(input, **kwargs) + input)


class Linformer(nn.Module):
    def __init__(self, args) -> None:
        super(Linformer, self).__init__()
        assert args.para_share_schema in ['head', 'kv', 'layer']

        self.num_block = args.num_block
        self.head_dim = args.embed_size // args.num_head
        self.para_share_schema = args.para_share_schema

        self.linformer_block = nn.ModuleList([])
        if args.para_share_schema == 'head' or args.para_share_schema == 'kv':
            for _ in range(args.num_block):
                self.linformer_block.append(nn.ModuleList([
                    PostLayerNorm(args.embed_size, LinformerMultiHeadSelfAttention(args.max_seq_len, 
                                                                                   args.embed_size, 
                                                                                   args.proj_size, 
                                                                                   args.num_head, 
                                                                                   args.value_drop_prob, 
                                                                                   args.para_share_schema)),
                    PostLayerNorm(args.embed_size, FeedForwardNetwork(args.embed_size, args.hidden_size, args.ffn_drop_prob))
            ]))
        else:
            self.proj_weight_all = nn.Parameter(torch.empty(args.max_seq_len, args.proj_dim), requires_grad=False)
            self.reset_parameters()

            for _ in range(args.num_block):
                self.linformer_block.append(nn.ModuleList([
                    PostLayerNorm(args.embed_size, LinformerMultiHeadSelfAttentionProjectLayerwise(args.max_seq_len, 
                                                                                                   args.embed_size, 
                                                                                                   args.proj_size, 
                                                                                                   args.num_head, 
                                                                                                   args.value_drop_prob, 
                                                                                                   args.para_share_schema)),
                    PostLayerNorm(args.embed_size, FeedForwardNetwork(args.embed_size, args.hidden_size, args.ffn_drop_prob))
            ]))

    def reset_parameters(self) -> None:
        if self.para_share_schema == 'layer':
            init.normal_(self.proj_weight_all, mean=0, std=math.sqrt(1 / self.seq_len))

    def forward(self, input: Tensor) -> Tensor:
        if self.para_share_schema == 'head' or self.para_share_schema == 'kv':
            for attn, ffn in self.linformer_block:
                input = attn(input)
                input = ffn(input)
        else:
            for attn, ffn in self.linformer_block:
                input = attn(input, self.proj_weight_all)
                input = ffn(input)

        return input