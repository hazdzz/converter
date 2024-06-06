import math
import torch
import torch.nn as nn
from model import chsyconv, embedding, ffn, norm
from torch import Tensor


class Converter(nn.Module):
    def __init__(self, args) -> None:
        super(Converter, self).__init__()
        self.embed_dim = args.embed_dim
        self.embedding = embedding.ConverterEmbedding(args.pe_type, 
                                                      args.pooling_type, 
                                                      args.vocab_size,
                                                      args.max_seq_len, 
                                                      args.embed_dim, 
                                                      args.embed_drop_prob
                                                      )
        self.chsyconv = chsyconv.ChsyConv(args.batch_size, 
                                          args.max_seq_len, 
                                          args.embed_dim, 
                                          args.eigenvalue_drop_prob, 
                                          args.value_drop_prob, 
                                          args.kernel_type, 
                                          args.max_order, 
                                          args.mu, 
                                          args.xi, 
                                          args.stigma, 
                                          args.heta
                                          )
        self.bffn = ffn.BFFN(args.embed_dim, args.hid_dim, args.bffn_drop_prob)
        self.embed_norm = norm.ScaleNorm(args.embed_dim, eps=1e-8)
        self.chsyconv_norm = norm.ScaleNorm(args.embed_dim, eps=1e-8)

    def forward(self, input) -> Tensor:
        embed= self.embedding(input)

        embed_norm = self.embed_norm(embed)
        chsyconv = self.chsyconv(embed_norm) + embed

        chsyconv_normed = self.chsyconv_norm(chsyconv)
        bffn = self.bffn(chsyconv_normed)

        return bffn