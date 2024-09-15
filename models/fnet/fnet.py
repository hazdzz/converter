import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class IdentityFFTConv2D(nn.Module):
    def __init__(self) -> None:
        super(IdentityFFTConv2D, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return torch.fft.fft2(input, dim=(-2, -1)).real
    

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
    
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.layernorm(self.func(input, **kwargs) + input)
    

class FNet(nn.Module):
    def __init__(self, args) -> None:
        super(FNet, self).__init__()
        self.fnet_block = nn.ModuleList([])
        for _ in range(args.num_block):
            self.fnet_block.append(nn.ModuleList([
                PostLayerNorm(args.embed_size, IdentityFFTConv2D()),
                PostLayerNorm(args.embed_size, FeedForwardNetwork(args.embed_size, args.hidden_size, args.ffn_drop_prob))
            ]))
    
    def forward(self, input: Tensor) -> Tensor:
        for fourier, ffn in self.fnet_block:
            input = fourier(input)
            input = ffn(input)
        
        return input