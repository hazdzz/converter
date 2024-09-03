import torch
import torch.nn as nn
from model.converter import chsyconv, embedding, ffn, norm
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
                                          args.value_drop_prob, 
                                          args.enable_kpm, 
                                          args.kernel_type, 
                                          args.max_order, 
                                          args.mu, 
                                          args.xi, 
                                          args.stigma, 
                                          args.heta)
        self.bffn = ffn.BilinearFeedForward(args.max_seq_len, args.embed_dim, args.bffn_drop_prob)
        self.chsyconv_norm = norm.ScaleNorm(args.embed_dim)
        self.bffn_norm = norm.ScaleNorm(args.embed_dim)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, input: Tensor) -> Tensor:
        alpha = torch.clamp(self.alpha, min=0.0, max=1.0).to(input.device)

        embed = self.embedding(input)

        chsyconv = self.chsyconv(embed) + embed
        chsyconv_normed = self.chsyconv_norm(chsyconv)

        bffn = self.bffn(chsyconv_normed) + alpha * chsyconv_normed.real + (1.0 - alpha) * chsyconv_normed.imag
        bffn_norm = self.bffn_norm(bffn)

        return bffn_norm