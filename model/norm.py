import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor


# Nguyen, T., & Chiang, D. (2018). 
# Improving Lexical Choice in Neural Machine Translation. 
# 2018 Conference of the North American Chapter of 
# the Association for Computational Linguistics: Human Language Technologies, 
# Volume 1 (Long Papers) (pp. 334–343). 
# Association for Computational Linguistics.
class FixNorm(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super(FixNorm, self).__init__()
        self.eps = eps

    def forward(self, input: Tensor) -> Tensor:
        fixnorm = input / (torch.norm(input, dim=-1, keepdim=True) + self.eps)

        return fixnorm


# Nguyen, T., & Salazar, J. (2019). 
# Transformers without Tears: Improving the Normalization of Self-Attention. 
# 16th International Conference on Spoken Language Translation. 
# Association for Computational Linguistics. 
class ScaleNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-5, bias: bool = False) -> None:
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
    def __init__(self, dim, eps: float = 1e-5, bias: bool = False) -> None:
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
    

# Xu, J., Sun, X., Zhang, Z., Zhao, G., & Lin, J. (2019). 
# Understanding and Improving Layer Normalization. 
# Advances in Neural Information Processing Systems (pp. 4383–4393).
class AdaNorm(nn.Module):
    def __init__(self, dim, k: float = 0.1, eps: float = 1e-5) -> None:
        super(AdaNorm, self).__init__()
        self.k = k
        self.eps = eps
        self.scale = nn.Parameter(torch.empty(dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.scale)

    def forward(self, input: Tensor) -> Tensor:
        mean = torch.mean(input, dim=-1, keepdim=True)
        var = (input - mean).pow(2).mean(dim=-1, keepdim=True) + self.eps
    
        input_norm = (input - mean) * torch.rsqrt(var)
        
        adanorm = self.scale * (1 - self.k * input_norm) * input_norm
    
        return adanorm
    

# Jiang, Z., Gu, J., Zhu, H., & Pan, D. (2023). 
# Pre-RMSNorm and Pre-CRMSNorm Transformers: Equivalent and Efficient Pre-LN Transformers.
# Advances in Neural Information Processing Systems (pp. 45777–45793).
class CRMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-5, bias: bool = False) -> None:
        super(CRMSNorm, self).__init__()
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
        discarded_element = input.sum(dim=-1, keepdim=True)
        var = (input.pow(2).mean(dim=-1, keepdim=True) + discarded_element.pow(2)) / (input.size(-1) + 1) + self.eps
        input_norm = input * torch.rsqrt(var)

        crmsnorm = self.scale * input_norm
        
        if self.bias is not None:
            crmsnorm = crmsnorm + self.bias

        return crmsnorm