import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from .. import embedding


class FourierTransform2D(nn.Module):
    def __init__(self) -> None:
        super(FourierTransform2D, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return torch.fft.fft2(input, dim=(-2, -1)).real
    

class FeedForwardNetwork(nn.Module):
    def __init__(self, feat_dim: int, hid_dim: int, ffn_drop_prob: float = 0.1) -> None:
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
    
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.layernorm(self.func(input, **kwargs) + input)
    

class FNetEncoder(nn.Module):
    def __init__(self, args) -> None:
        super(FNetEncoder, self).__init__()
        self.embedding = embedding.Embedding(args.pe_type, 
                                             args.pooling_type, 
                                             args.vocab_size, 
                                             args.max_seq_len, 
                                             args.embed_dim, 
                                             args.pe_drop_prob, 
                                             args.embed_drop_prob)
        self.fnet_encoder_block = nn.ModuleList([])
        for _ in range(args.num_block):
            self.fnet_encoder_block.append(nn.ModuleList([
                PostLayerNorm(args.embed_dim, FourierTransform2D()),
                PostLayerNorm(args.embed_dim, FeedForwardNetwork(args.embed_dim, args.hidden_dim, args.ffn_drop_prob))
            ]))
    
    def forward(self, input: Tensor) -> Tensor:
        input = self.embedding(input)

        for fourier, ffn in self.fnet_encoder_block:
            input = fourier(input)
            input = ffn(input)
        
        return input