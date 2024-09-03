import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.functional as cF
from typing import Callable, List, Optional, Tuple
from torch import Tensor

class dSiLU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return cF.d_silu(input)

class CSigmoid(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return cF.c_sigmoid(input)

class CTanh(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return cF.c_tanh(input)

class modTanh(nn.Module):
    __constants__ = ['rounding_mode']
    rounding_mode: str

    def __init__(self, rounding_mode: str = None):
        super(modTanh, self).__init__()
        self.rounding_mode = rounding_mode

    def forward(self, input: Tensor, rounding_mode: str = None) -> Tensor:
        return cF.mod_tanh(input, rounding_mode=rounding_mode)

    def extra_repr(self) -> str:
        return 'rounding_mode={}'.format(self.rounding_mode)

class Hirose(nn.Module):
    __constants__ = ['m', 'inplace']
    m: float
    inplace: bool

    def __init__(self, m: float = 1., inplace: bool = False):
        super(Hirose, self).__init__()
        self.m = m
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return cF.hirose(input, m=self.m, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return 'm={}{}'.format(self.m, inplace_str)

class Siglog(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return cF.siglog(input)

class CCardioid(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return cF.c_cardioid(input)

class CReLU(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(CReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return cF.c_relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class zReLU(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(zReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return cF.z_relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class modReLU(nn.Module):
    __constants__ = ['bias', 'rounding_mode', 'inplace']
    bias: float
    rounding_mode: str
    inplace: bool

    def __init__(self, bias: float = -math.sqrt(2), rounding_mode: str = None, inplace: bool = False):
        super(modReLU, self).__init__()
        self.bias = bias
        self.rounding_mode = rounding_mode
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return cF.mod_relu(input, bias=self.bias, rounding_mode=self.rounding_mode, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return 'bias={}, rounding_mode={}, inplace_str={}'.format(self.bias, self.rounding_mode, inplace_str)

class CLeakyReLU(nn.Module):
    __constants__ = ['negative_slope', 'inplace']
    negative_slope: float
    inplace: bool

    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        super(CLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return cF.c_leaky_relu(input, self.negative_slope, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'negative_slope={}{}'.format(self.negative_slope, inplace_str)

class CGELU(nn.Module):
    __constants__ = ['approximate']
    approximate: str

    def __init__(self, approximate: str = 'none'):
        super(CGELU, self).__init__()
        self.approximate = approximate

    def forward(self, input: Tensor) -> Tensor:
        return cF.c_gelu(input, approximate=self.approximate)

    def extra_repr(self) -> str:
        return f'approximate={repr(self.approximate)}'

class TanhExp(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(TanhExp, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return cF.tanhexp(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class SinSig(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(SinSig, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return cF.sinsig(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class SquaredReLU(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(SquaredReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return cF.sqrrelu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
    
class Squareplus(nn.Module):
    __constants__ = ['bias', 'threshold', 'inplace']
    bias: float
    threshold: float
    inplace: bool

    def __init__(self, bias: float = 4 * (math.log(2) ** 2), threshold: float = 20.0, inplace: bool = False):
        super(Squareplus, self).__init__()
        self.bias = bias
        self.threshold = threshold
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return cF.squareplus(input, self.bias, self.threshold, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f'bias={self.bias}, threshold={self.threshold}, inplace={self.inplace}'
    
class DiracReLU(nn.Module):
    __constants__ = ['beta', 'inplace']
    beta: float
    inplace: bool

    def __init__(self, bias: float = 1.0, inplace: bool = False):
        super(DiracReLU, self).__init__()
        self.bias = bias
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return cF.diracrelu(input, self.bias, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'bias={}{}'.format(self.bias, inplace_str)

class LipGELU(nn.Module):
    __constants__ = ['approximate']
    approximate: str

    def __init__(self, approximate: str = 'none'):
        super(LipGELU, self).__init__()
        self.approximate = approximate

    def forward(self, input: Tensor) -> Tensor:
        return cF.lip_gelu(input, approximate=self.approximate)

    def extra_repr(self) -> str:
        return f'approximate={repr(self.approximate)}'

class LipSiLU(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(LipSiLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return cF.lip_silu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class LipMish(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(LipMish, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return cF.lip_mish(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class LipTanhExp(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(LipTanhExp, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return cF.lip_tanhexp(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class LipSinSig(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(LipSinSig, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return cF.lip_sinsig(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
    
class LipSReLU(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(LipSReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return cF.lip_srelu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class modSoftmax(nn.Module):
    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super(modSoftmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        return cF.mod_softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self) -> str:
        return 'dim={dim}'.format(dim=self.dim)

class modLogSoftmax(nn.Module):
    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super(modLogSoftmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        return cF.mod_log_softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self) -> str:
        return 'dim={dim}'.format(dim=self.dim)

class rSoftmax(nn.Module):
    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super(rSoftmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        return cF.r_softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self) -> str:
        return 'dim={dim}'.format(dim=self.dim)

class rLogSoftmax(nn.Module):
    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super(rLogSoftmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        return cF.r_log_softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self) -> str:
        return 'dim={dim}'.format(dim=self.dim)
