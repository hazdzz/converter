import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import Tensor
from .. import embedding


def deterministic_dropout(x: Tensor, seed=0, dropout=0):
    torch.manual_seed(seed)
    return F.dropout(x, p=dropout, training=True)


def look_back(input_tensor: Tensor) -> Tensor:
    '''
    Looks back one bucket
    '''
    shift = torch.cat([input_tensor[:, -1:], input_tensor[:, :-1]], dim=1)
    # [batch * head, n_buckets, bucket_length, d_k, rounds]
    concat = torch.cat([shift, input_tensor], dim=2)
    # [batch * head, n_buckets, bucket_length * 2, d_k, rounds]
    return concat


def reverse_sort(indice: Tensor, dim: int) -> Tensor:
    '''
    Unsorts sorted indice
    '''
    new_size = [1] * indice.dim()
    new_size[dim] = indice.size(dim)
    arange = indice.new_empty(size=new_size)
    torch.arange(new_size[dim], out=arange)
    arange = arange.expand_as(indice)
    new_indice = torch.empty_like(indice)
    new_indice.scatter_(dim=dim, index=indice, src=arange)
    return new_indice


def expand(input_tensor: Tensor, dim=0, num=1) -> Tensor:
    '''
    Shortcut for unsqueeze + expand
    '''
    new_size = [-1] * (input_tensor.dim() + 1)
    new_size[dim] = num
    return input_tensor.unsqueeze(dim=dim).expand(new_size)


def expand_gather(input_tensor: Tensor, dim: int, index: Tensor, expand_dim=0, num=1) -> Tensor:
    expanded_index = expand(index, dim=expand_dim, num=num)
    return input_tensor.gather(dim=dim, index=expanded_index)


def get_dup_keys(input_tensor: Tensor, rounds=0) -> Tensor:
    sorted_flat_key, flat_key_indice = torch.sort(input_tensor, dim=-1)
    # [batch * head, length, bucket_length * 2 * rounds]
    count_shift_keys = torch.ones_like(sorted_flat_key)
    # [batch * head, length, bucket_length * 2 * rounds]
    for i in range(1, rounds):
        equiv_flat_key = (sorted_flat_key[..., i:] == sorted_flat_key[..., :-i]).int()
        count_shift_keys[..., i:] += equiv_flat_key
        count_shift_keys[..., :-i] += equiv_flat_key
    count_key_indice = reverse_sort(flat_key_indice, dim=2)
    # [batch * head, length, bucket_length * 2 * rounds]
    return torch.gather(count_shift_keys, dim=-1, index=count_key_indice)


def top_p_sample(prob: Tensor, perc=0.5) -> np.array:
    sorted_prob, sorted_indices = torch.sort(prob, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_prob, dim=-1)
    mask = cumsum < perc
    one_more_indice = mask.long().sum(dim=-1, keepdim=True)
    mask.scatter_(dim=-1, index=one_more_indice, value=True)
    sorted_prob.masked_fill_(~mask, value=0.0)
    masked_prob = sorted_prob.gather(dim=-1, index=reverse_sort(sorted_indices, dim=-1))
    return torch.multinomial(masked_prob, num_samples=1)


class LocalitySensitiveHash(nn.Module):
    '''
    Implements Locality Sensitive Hash
    class is used to save random matrix used for hashing
    '''
    def __init__(self, d_model, head, rounds):
        super(LocalitySensitiveHash, self).__init__()
        self.d_k = d_model // head
        self.rounds = rounds
        self.rand_matrix = None

    def forward(self, inp: Tensor, n_buckets=0, random=True):
        batch_size = inp.size(0)
        length = inp.size(1)
        inp = F.normalize(inp, p=2, dim=-1)
        # [batch * head, length, d_k]
        if random:
            self.rand_matrix = torch.randn(
                [batch_size, self.d_k, self.rounds, n_buckets // 2],
                device=inp.get_device()
            )
            # [batch * head, d_k, rounds, n_buckets // 2]
            self.rand_matrix /= torch.norm(self.rand_matrix, dim=1, keepdim=True)
            # [batch * head, d_k, rounds, n_buckets // 2]
        matmul = torch.einsum('...ij,...jkl->...ikl', inp, self.rand_matrix)
        # [batch * head, length, rounds, n_buckets // 2]
        hashes = torch.argmax(torch.cat([matmul, -matmul], dim=-1), dim=-1).int()
        # [batch * head, length, rounds]
        arange = hashes.new_empty((1, length, 1))
        # [1, length, 1]
        hashes = hashes * length + torch.arange(length, out=arange).expand_as(hashes)
        # [batch * head, length, rounds]
        return hashes

class LSHAttention(nn.Module):
    '''
    Implements LSHAttention
    class is used to save LocalitySensitiveHash
    '''
    def __init__(self, d_model, head, rounds, droprate, bucket_length):
        super(LSHAttention, self).__init__()
        self.d_k = d_model // head
        self.rounds = rounds
        self.dropout = nn.Dropout(p=droprate)
        self.bucket_length = bucket_length
        self.lsh = LocalitySensitiveHash(d_model, head, rounds)

    def forward(self, query, value, seed, random=True):
        length = query.size(1)
        n_buckets = length // self.bucket_length

        sorted_hashes, hash_indice = torch.sort(self.lsh(query, n_buckets, random), dim=1)
        # [batch * head, length, rounds]
        original_indice = reverse_sort(hash_indice, dim=1)
        # [batch * head, length, rounds]

        reordered_query = expand_gather(
            expand(query, dim=3, num=self.rounds), dim=1,\
            index=hash_indice, expand_dim=2, num=self.d_k
        )
        # [batch * head, length, d_k, rounds]
        reordered_query = reordered_query.reshape(
            -1, n_buckets, self.bucket_length, self.d_k, self.rounds
        )
        # [batch * head, n_buckets, bucket_length, d_k, rounds]
        lookback_key = F.normalize(look_back(reordered_query), p=2, dim=-2)
        # [batch * head, n_buckets, bucket_length * 2, d_k, rounds]
        matmul_qk = torch.einsum(
            '...ijk,...ljk->...ilk', reordered_query, lookback_key
        ) / math.sqrt(self.d_k)
        # [batch * head, n_buckets, bucket_length, bucket_length * 2, rounds]

        sorted_hashes = sorted_hashes.reshape(
            -1, n_buckets, self.bucket_length, self.rounds
        ) // length
        # [batch * head, n_buckets, bucket_length, rounds]
        matmul_qk.masked_fill_(
            mask=(sorted_hashes[..., None, :] != look_back(sorted_hashes)[..., None, :, :]),\
            value=-1e9
        )

        query_indice = hash_indice.reshape(
            -1, n_buckets, self.bucket_length, self.rounds
        ).int()
        # [batch * head, n_buckets, bucket_length, rounds]
        key_indice = look_back(query_indice)
        # [batch * head, n_buckets, bucket_length * 2, rounds]
        matmul_qk.masked_fill_(
            mask=(query_indice[..., None, :] < key_indice[..., None, :, :]), value=-1e9
        )
        matmul_qk.masked_fill_(
            mask=(query_indice[..., None, :] == key_indice[..., None, :, :]), value=-1e5
        )

        key_indice = expand(key_indice, dim=2, num=self.bucket_length).flatten(1, 2)
        # [batch * head, length, bucket_length * 2, rounds]
        key_indice = expand_gather(
            key_indice,
            dim=1, index=original_indice,
            expand_dim=2, num=self.bucket_length * 2
        )
        # [batch * head, length, bucket_length * 2, rounds]
        count_key = get_dup_keys(
            key_indice.flatten(-2, -1), self.rounds
        ).reshape(-1, length, self.bucket_length * 2, self.rounds)
        # [batch * head, length, bucket_length * 2, rounds]
        count_key = expand_gather(
            count_key, dim=1, index=hash_indice, expand_dim=2, num=self.bucket_length * 2
        )
        # [batch * head, length, bucket_length * 2, rounds]
        matmul_qk = matmul_qk.flatten(1, 2)
        # [batch * head, length, bucket_length * 2, rounds]
        logsumexp_qk = torch.logsumexp(matmul_qk, dim=2)
        # [batch * head, length, rounds]
        softmax_qk = torch.exp(matmul_qk - count_key.float().log_() - logsumexp_qk[..., None, :])
        # [batch * head, length, bucket_length * 2, rounds]

        if self.training:
            softmax_qk = self.dropout(softmax_qk)
            # [batch * head, length, bucket_length * 2, rounds]

        reordered_value = expand_gather(
            expand(value, dim=3, num=self.rounds), dim=1,\
            index=hash_indice, expand_dim=2, num=self.d_k
        )
        # [batch * head, length, d_k, rounds]
        reordered_value = reordered_value.reshape(
            -1, n_buckets, self.bucket_length, self.d_k, self.rounds
        )
        # [batch * head, n_buckets, bucket_length, d_k, rounds]

        softmax_qk = softmax_qk.reshape(
            -1, n_buckets, self.bucket_length, self.bucket_length * 2, self.rounds
        )
        # [batch * head, n_buckets, bucket_length, bucket_length * 2, rounds]

        attention = torch.einsum('...ijl,...jkl->...ikl', softmax_qk, look_back(reordered_value))
        # [batch * head, n_buckets, bucket_length, d_k, rounds]
        attention = attention.flatten(1, 2)
        # [batch * head, length, d_k, rounds]
        attention = expand_gather(
            attention, dim=1, index=original_indice, expand_dim=2, num=self.d_k
        )
        # [batch * head, length, d_k, rounds]
        logsumexp_qk = torch.gather(logsumexp_qk, dim=1, index=original_indice)
        # [batch * head, length, rounds]
        logsumexp_qk = F.softmax(logsumexp_qk, dim=1)
        # [batch * head, length, rounds]
        attention = torch.einsum('...ij,...j->...i', attention, logsumexp_qk)
        # [batch * head, length, d_k]

        return attention

class MultiRoundLSHAttention(nn.Module):
    '''
    Implements Multi Round LSH Attention
    class is defined to save LSHAttention
    '''
    def __init__(self, d_model, head, chunk, rounds, droprate, bucket_length):
        super(MultiRoundLSHAttention, self).__init__()
        self.d_k = d_model // head
        self.head = head
        self.chunk = chunk
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.lshattention = LSHAttention(d_model, head, rounds, droprate, bucket_length)

    def forward(self, input):
        seed = 42
        random = True

        length = input.size(1)

        query = self.linear_query(input).reshape(-1, length, self.head, self.d_k).transpose_(1, 2)
        # [batch, head, length, d_k]
        value = self.linear_value(input).reshape(-1, length, self.head, self.d_k).transpose_(1, 2)
        # [batch, head, length, d_k]

        chunked_query = torch.chunk(query.flatten(0, 1), chunks=self.chunk, dim=0)
        # [batch * head // chunk, length, d_k]
        chunked_value = torch.chunk(value.flatten(0, 1), chunks=self.chunk, dim=0)
        # [batch * head // chunk, length, d_k]

        attention = torch.cat([
            self.lshattention(q, v, seed + i, random) for q, v, i\
                in zip(chunked_query, chunked_value, range(self.chunk))
        ], dim=0).reshape(-1, self.head, length, self.d_k)
        # [batch, head, length, d_k]

        attention = attention.transpose(1, 2).flatten(-2, -1)
        # [batch, length, d_model]

        return self.linear_out(attention)


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
    

class ReformerEncoder(nn.Module):
    def __init__(self, args):
        super(ReformerEncoder, self).__init__()
        self.embedding = embedding.Embedding(args.pe_type, 
                                             args.pooling_type, 
                                             args.vocab_size, 
                                             args.max_seq_len, 
                                             args.embed_dim, 
                                             args.pe_drop_prob, 
                                             args.embed_drop_prob)
        self.reformer_encoder_block = nn.ModuleList([])
        for _ in range(args.num_block):
            self.reformer_encoder_block.append(nn.ModuleList([
                PreLayerNorm(args.embed_dim, MultiRoundLSHAttention(args.embed_dim, args.num_head, args.xformer.reformer.num_chunk, args.xformer.reformer.rounds, args.xformer.reformer.drop_prob, args.xformer.reformer.bucket_length)),
                PreLayerNorm(args.embed_dim, FeedForwardNetwork(args.embed_dim, args.hidden_dim, args.ffn_drop_prob))
            ]))

    def forward(self, input: Tensor) -> Tensor:
        input = self.embedding(input)

        for mhra, ffn in self.reformer_encoder_block:
            input = mhra(input)
            input = ffn(input)
        
        return input