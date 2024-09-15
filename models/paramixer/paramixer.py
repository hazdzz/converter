import itertools
import math
import torch
import torch.nn as nn
import torch_sparse
from typing import List, Tuple, Union
from torch import Tensor


def get_chord_indices_assym(n_vec, n_link) -> Tuple[List[int], List[int]]:
    """
    Generates position indicies, based on the Chord protocol (incl. itself).

    :param n_vec: sequence length
    :param n_link: number of links in the Chord protocol
    :return: target indices in two lists, each is of size n_vec * n_link
    """

    rows = list(
        itertools.chain(
            *[
                [i for j in range(n_link)] for i in range(n_vec)
            ]
        )
    )

    cols = list(
        itertools.chain(
            *[
                [i] + [(i + 2 ** k) % n_vec for k in range(n_link - 1)] for i in range(n_vec)
            ]
        )
    )

    return rows, cols


def get_dil_indices_assym(n_vec, n_link, n_layer) -> Tuple[List[int], List[int]]:
    """
    Generates the position indicies, based on the symmetric Chord protocol (incl. itself).
    So n_link is an odd number
    """

    dil_ws = []
    for n in range(n_layer):
        dilation = 2 ** n
        half_link = int((n_link - 1) / 2)

        rows = list(
            itertools.chain(
                *[
                    [r for _ in range(n_link)] for r in range(n_vec)
                ]
            )
        )
        cols = list(
            itertools.chain(
                *[
                    [i] + [(i + k * dilation) % n_vec for k in range(1, 1 + half_link)] +
                    [(i - k * dilation) % n_vec for k in range(1, 1 + half_link)] for i in range(n_vec)
                ]
            )
        )

        rc_tensor = torch.tensor([rows, cols])
        rc_list = rc_tensor.tolist()
        dil_ws.append(rc_list)

    return dil_ws


def BuildMLPBlock(cfg: List[Union[str, int]], in_channels: int, out_channels: int) -> nn.Sequential:
    """
    Constructs an MLP based on a given structural config.
    """
    layers: List[nn.Module] = []
    for i in cfg:
        if isinstance(i, int):
            layers += [nn.Linear(in_channels, i)]
            in_channels = i
        else:
            layers += [nn.GELU()]
    layers += [nn.Linear(in_channels, out_channels)]

    return nn.Sequential(*layers)


class MLPBlock(nn.Module):
    """
    Constructs a MLP with the specified structure.

    """

    def __init__(self, cfg: List[Union[str, int]], in_channels: int, out_channels: int) -> None:
        super(MLPBlock, self).__init__()
        self.mlp_block = BuildMLPBlock(cfg, in_channels, out_channels)

    def forward(self, input: Tensor) -> Tensor:
        return self.mlp_block(input)


class AttentionModule(nn.Module):
    def __init__(self,
                 embed_size,
                 max_seq_len,
                 protocol,
                 drop_prob,
                 hidden_size, 
                 device
                 ) -> None:
        super(AttentionModule, self).__init__()
        self.max_seq_len = max_seq_len
        self.protocol = protocol
        self.n_W = math.ceil(math.log2(max_seq_len))
        self.n_links = self.n_W + 1
        self.embedding_each_head = embed_size

        if protocol == "dil":
            self.n_links = 9
            self.protocol_indicies = torch.tensor(
                                        get_dil_indices_assym(max_seq_len, self.n_links, self.n_W)
                                    ).to(device)
        elif protocol == "chord":
            self.protocol_indicies = torch.tensor(
                                        get_chord_indices_assym(max_seq_len, self.n_links)
                                    ).to(device)

        # Init Ws
        self.fs = nn.ModuleList(
            [
                MLPBlock(
                    [hidden_size, 'GELU'],
                    embed_size,
                    self.n_links
                )
                for _ in range(self.n_W)
            ]
        )

        # Init V
        self.g = MLPBlock(
            [hidden_size, 'GELU'],
            embed_size,
            embed_size
        )

        self.attn_dropout = nn.Dropout(drop_prob)

    def forward(self, V: Tensor, input: Tensor) -> Tensor:
        # Iterate over all heads
        # Get V
        V = self.g(V)
        w_index = 0
        for m in range(self.n_W):
            # Init residual connection
            res_conn = V
            # Get W_m
            W = self.fs[m](input)
            # Multiply W_m and V, get new V

            if self.protocol == "dil":
                V = torch_sparse.spmm(
                    self.protocol_indicies[w_index],
                    W.view(W.size(0), -1),
                    self.max_seq_len,
                    self.max_seq_len,
                    V
                )
            else:
                V = torch_sparse.spmm(
                    self.protocol_indicies,
                    W.view(W.size(0), -1),
                    self.max_seq_len,
                    self.max_seq_len,
                    V
                )
            w_index += 1
            V = V + res_conn

        V = self.attn_dropout(V)
        return V


class Paramixer(nn.Module):
    def __init__(self, args, device) -> None:
        super(Paramixer, self).__init__()
        self.n_layers = args.n_layers
        self.attention = nn.ModuleList(
            [
                AttentionModule(args.embed_size, 
                                args.max_seq_len, 
                                args.protocol, 
                                args.attn_drop_prob,
                                args.hidden_size, 
                                device
                                )
                for _ in range(args.n_layers)
            ]

        )

    def forward(self, embed: Tensor) -> Tensor:
        V = embed

        for l in range(self.n_layers):
            V = self.attention[l](V, embed)

        return V