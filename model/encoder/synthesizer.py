import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import Tensor
from .. import embedding


class MultiHeadRandomAttention(nn.Module):
    def __init__(self, batch_size: int, max_seq_len: int, feat_dim: int, num_head: int, value_drop_prob: float) -> None:
        super(MultiHeadRandomAttention, self).__init__()
        self.num_head = num_head
        self.feat_dim = feat_dim
        self.head_dim = feat_dim // num_head

        self.multihead_random_attn = nn.Parameter(torch.empty(batch_size, num_head, max_seq_len, max_seq_len))
        self.value_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.output_linear = nn.Linear(feat_dim, feat_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.value_dropout = nn.Dropout(p=value_drop_prob)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.multihead_random_attn)
        init.xavier_uniform_(self.value_linear.weight)
        init.xavier_uniform_(self.output_linear.weight)

    def split_head(self, input: Tensor) -> Tensor:
        batch_size, seq_len, _ = input.size()
        return input.contiguous().view(batch_size, seq_len, self.num_head, self.head_dim).permute(0, 2, 1, 3)
    
    def concat_head(self, input: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = input.size()
        return input.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.feat_dim)

    def forward(self, input: Tensor) -> Tensor:
        value = self.value_linear(input)
        value = self.value_dropout(value)
        multihead_value = self.split_head(value)

        multihead_random_attn_score = self.softmax(self.multihead_random_attn)
        multihead_random_attn_score = torch.einsum('bhnm,bhmd->bhnd', multihead_random_attn_score, multihead_value)
        multihead_random_attn_score_concat = self.concat_head(multihead_random_attn_score)
        multihead_random_attn_output = self.output_linear(multihead_random_attn_score_concat)
        
        return multihead_random_attn_output
    

class MultiHeadFactorizedRandomAttention(nn.Module):
    def __init__(self, batch_size: int, max_seq_len: int, feat_dim: int, num_head: int, rank: int, value_drop_prob: float) -> None:
        super(MultiHeadFactorizedRandomAttention, self).__init__()
        self.num_head = num_head
        self.feat_dim = feat_dim
        self.head_dim = feat_dim // num_head

        self.random_attn_factor_l = nn.Parameter(torch.empty(batch_size, num_head, max_seq_len, rank))
        self.random_attn_factor_r = nn.Parameter(torch.empty(batch_size, num_head, max_seq_len, rank))
        self.value_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.output_linear = nn.Linear(feat_dim, feat_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.value_dropout = nn.Dropout(p=value_drop_prob)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.random_attn_factor_l)
        init.xavier_uniform_(self.random_attn_factor_r)
        init.xavier_uniform_(self.value_linear.weight)
        init.xavier_uniform_(self.output_linear.weight)

    def split_head(self, input: Tensor) -> Tensor:
        batch_size, seq_len, _ = input.size()
        return input.contiguous().view(batch_size, seq_len, self.num_head, self.head_dim).permute(0, 2, 1, 3)
    
    def concat_head(self, input: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = input.size()
        return input.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.feat_dim)

    def forward(self, input: Tensor) -> Tensor:
        value = self.value_linear(input)
        value = self.value_dropout(value)
        multihead_value = self.split_head(value)

        multihead_random_attn = torch.einsum('bhnk,bhnk->bhnn', self.random_attn_factor_l, self.random_attn_factor_r)
        multihead_random_attn_score = self.softmax(multihead_random_attn)
        multihead_random_attn_score = torch.einsum('bhnm,bhmd->bhnd', multihead_random_attn_score, multihead_value)
        multihead_random_attn_score_concat = self.concat_head(multihead_random_attn_score)
        multihead_random_attn_output = self.output_linear(multihead_random_attn_score_concat)
        
        return multihead_random_attn_output
    

class MultiHeadDenseAttention(nn.Module):
    def __init__(self, max_seq_len: int, feat_dim: int, num_head: int, value_drop_prob: float) -> None:
        super(MultiHeadDenseAttention, self).__init__()
        self.num_head = num_head
        self.feat_dim = feat_dim
        self.head_dim = feat_dim // num_head

        self.attn_mlp = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.head_dim, max_seq_len, bias=True)
        )
        self.value_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.output_linear = nn.Linear(feat_dim, feat_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.value_dropout = nn.Dropout(p=value_drop_prob)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.attn_mlp[0].weight)
        init.xavier_uniform_(self.attn_mlp[2].weight)
        init.xavier_uniform_(self.value_linear.weight)

        init.zeros_(self.attn_mlp[0].bias)
        init.zeros_(self.attn_mlp[2].bias)

    def split_head(self, input: Tensor) -> Tensor:
        batch_size, seq_len, _ = input.size()
        return input.contiguous().view(batch_size, seq_len, self.num_head, self.head_dim).permute(0, 2, 1, 3)
    
    def concat_head(self, input: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = input.size()
        return input.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.feat_dim)

    def forward(self, input: Tensor) -> Tensor:
        value = self.value_linear(input)
        value = self.value_dropout(value)
        multihead_value = self.split_head(value)

        multihead_input = self.split_head(input)
        multihead_dense_attn = self.attn_mlp(multihead_input)
        multihead_dense_attn_score = self.softmax(multihead_dense_attn)
        multihead_dense_attn_score = torch.einsum('bhnm,bhmd->bhnd', multihead_dense_attn_score, multihead_value)
        multihead_dense_attn_score_concat = self.concat_head(multihead_dense_attn_score)
        multihead_dense_attn_output = self.output_linear(multihead_dense_attn_score_concat)
        
        return multihead_dense_attn_output
    

class MultiHeadFactorizedDenseAttention(nn.Module):
    def __init__(self, max_seq_len: int, feat_dim: int, num_head: int, proj_dim: int, value_drop_prob: float) -> None:
        super(MultiHeadFactorizedDenseAttention, self).__init__()
        assert max_seq_len % num_head == 0

        self.num_head = num_head
        self.feat_dim = feat_dim
        self.head_dim = feat_dim // num_head
        self.proj_dim_l = proj_dim
        self.proj_dim_r = max_seq_len // proj_dim

        self.attn_factor_l_linear = nn.Linear(feat_dim, proj_dim, bias=False)
        self.attn_factor_r_linear = nn.Linear(feat_dim, max_seq_len // proj_dim, bias=False)
        self.value_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.output_linear = nn.Linear(feat_dim, feat_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.value_dropout = nn.Dropout(p=value_drop_prob)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.attn_factor_l_linear)
        init.xavier_uniform_(self.attn_factor_r_linear)
        init.xavier_uniform_(self.value_linear.weight)
        init.xavier_uniform_(self.output_linear.weight)

    def split_head(self, input: Tensor) -> Tensor:
        batch_size, seq_len, _ = input.size()
        return input.contiguous().view(batch_size, seq_len, self.num_head, self.head_dim).permute(0, 2, 1, 3)
    
    def concat_head(self, input: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = input.size()
        return input.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.feat_dim)

    def forward(self, input: Tensor) -> Tensor:
        b, _, _ = input.size()

        value = self.value_linear(input)
        value = self.value_dropout(value)
        multihead_value = self.split_head(value)

        attn_factor_l = self.attn_factor_l_linear(input)
        multihead_attn_factor_l = self.split_head(attn_factor_l)
        multihead_attn_factor_l = torch.tile(multihead_attn_factor_l.unsqueeze(-1), (1,1,1,1,self.proj_dim_r))

        attn_factor_r = self.attn_factor_r_linear(input)
        multihead_attn_factor_r = self.split_head(attn_factor_r)
        multihead_attn_factor_r = torch.tile(multihead_attn_factor_r.unsqueeze(-1), (1,1,1,1,self.proj_dim_l))

        multihead_attn = (multihead_attn_factor_l * multihead_attn_factor_r.transpose(-1, -2)).view(b, self.num_head, self.max_seq_len, self.max_seq_len)
        multihead_attn_score = self.softmax(multihead_attn)
        multihead_attn_score = torch.einsum('bhnm,bhmd->bhnd', multihead_attn_score, multihead_value)
        multihead_attn_score_concat = self.concat_head(multihead_attn_score)
        multihead_attn_output = self.output_linear(multihead_attn_score_concat)
        
        return multihead_attn_output
    

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
    

class SynthesizerEncoder(nn.Module):
    def __init__(self, args):
        super(SynthesizerEncoder, self).__init__()
        assert args.xformer.synthesizer.attn_type in ['random', 'dense', 'fac_random', 'fac_dense']

        self.embedding = embedding.Embedding(args.pe_type, 
                                             args.pooling_type, 
                                             args.vocab_size, 
                                             args.max_seq_len, 
                                             args.embed_dim, 
                                             args.pe_drop_prob, 
                                             args.embed_drop_prob)
        self.synthesizer_encoder_block = nn.ModuleList([])
        if args.xformer.synthesizer.attn_type == 'random':
            for _ in range(args.num_block):
                self.synthesizer_encoder_block.append(nn.ModuleList([
                    PostLayerNorm(args.embed_dim, MultiHeadRandomAttention(args.batch_size, args.max_seq_len, args.embed_dim, args.num_head, args.value_drop_prob)),
                    PostLayerNorm(args.embed_dim, FeedForwardNetwork(args.embed_dim, args.hidden_dim, args.ffn_drop_prob))
                ]))
        elif args.xformer.synthesizer.attn_type == 'dense':
            for _ in range(args.num_block):
                self.synthesizer_encoder_block.append(nn.ModuleList([
                    PostLayerNorm(args.embed_dim, MultiHeadDenseAttention(args.max_seq_len, args.embed_dim, args.num_head, args.value_drop_prob)),
                    PostLayerNorm(args.embed_dim, FeedForwardNetwork(args.embed_dim, args.hidden_dim, args.ffn_drop_prob))
                ]))
        elif args.xformer.synthesizer.attn_type == 'fac_random':
            for _ in range(args.num_block):
                self.synthesizer_encoder_block.append(nn.ModuleList([
                    PostLayerNorm(args.embed_dim, MultiHeadFactorizedRandomAttention(args.batch_size, args.max_seq_len, args.embed_dim, args.num_head, args.xformer.synthesizer.proj_dim, args.value_drop_prob)),
                    PostLayerNorm(args.embed_dim, FeedForwardNetwork(args.embed_dim, args.hidden_dim, args.ffn_drop_prob))
                ]))
        elif args.xformer.synthesizer.attn_type == 'fac_dense':
            for _ in range(args.num_block):
                self.synthesizer_encoder_block.append(nn.ModuleList([
                    PostLayerNorm(args.embed_dim, MultiHeadFactorizedDenseAttention(args.max_seq_len, args.embed_dim, args.num_head, args.xformer.synthesizer.proj_dim, args.value_drop_prob)),
                    PostLayerNorm(args.embed_dim, FeedForwardNetwork(args.embed_dim, args.hidden_dim, args.ffn_drop_prob))
                ]))

    def forward(self, input: Tensor) -> Tensor:
        input = self.embedding(input)

        for mhsa, ffn in self.synthesizer_encoder_block:
            input = mhsa(input)
            input = ffn(input)
        
        return input