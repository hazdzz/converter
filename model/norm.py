import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


# Nguyen, T., & Salazar, J. (2019). 
# Transformers without Tears: Improving the Normalization of Self-Attention. 
# 16th International Conference on Spoken Language Translation. 
# Association for Computational Linguistics. 
class ScaleNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-8, bias: bool = False) -> None:
        super(ScaleNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.empty(dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.scale)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        scalenorm = self.scale * input / (torch.norm(input, dim=-1, keepdim=True) + self.eps)

        if self.bias is not None:
            scalenorm = scalenorm + self.bias

        return scalenorm


# Zhang, B., & Sennrich, R. (2019). 
# Root Mean Square Layer Normalization. 
# Advances in Neural Information Processing Systems (pp. 12360–12371).
class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-8, bias: bool = False) -> None:
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.empty(dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.scale)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        var = input.pow(2).mean(dim=-1, keepdim=True) + self.eps
        input_norm = input * torch.rsqrt(var)

        rmsnorm = self.scale * input_norm
        
        if self.bias is not None:
            rmsnorm = rmsnorm + self.bias

        return rmsnorm