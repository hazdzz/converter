import torch
import torch.nn as nn
from . import chsyconv, embedding, ffn, norm
from torch import Tensor


class Converter(nn.Module):
    def __init__(self, args) -> None:
        super(Converter, self).__init__()
        self.embedding = embedding.Embedding(args.pe_type, 
                                             args.pooling_type, 
                                             args.vocab_size, 
                                             args.max_seq_len, 
                                             args.embed_dim, 
                                             args.pe_drop_prob, 
                                             args.embed_drop_prob)
        self.chsyconv = chsyconv.ChsyConv(args.batch_size, 
                                          args.max_seq_len, 
                                          args.embed_dim, 
                                          args.eigenvalue_drop_prob, 
                                          args.eigenvector_drop_prob, 
                                          args.value_drop_prob, 
                                          args.enable_kpm, 
                                          args.q, 
                                          args.kernel_type, 
                                          args.max_order, 
                                          args.mu, 
                                          args.xi, 
                                          args.stigma, 
                                          args.heta)
        self.gffn = ffn.GatedFeedForward(args.embed_dim, args.hid_dim, args.gffn_drop_prob)
        self.chsyconv_norm = norm.ScaleNorm(args.embed_dim)
        self.gffn_norm = norm.ScaleNorm(args.embed_dim)

    def forward(self, input: Tensor) -> Tensor:
        embed = self.embedding(input)

        chsyconv = self.chsyconv(embed)
        chsyconv_normed = self.chsyconv_norm(chsyconv)

        gffn = self.gffn(chsyconv_normed)
        encoder = self.gffn_norm(gffn)

        return encoder