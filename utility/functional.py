from typing import Callable, List, Optional, Tuple
import math
import warnings
import torch
import torch.nn.functional as F
from torch import Tensor


def complex_fcaller(funtional_handle, *args):
    return torch.complex(funtional_handle(args[0].real, *args[1:]), funtional_handle(args[0].imag, *args[1:]))

def d_silu(input: Tensor):
    return torch.sigmoid(input) * (1 + input * (1 - torch.sigmoid(input)))

def c_sigmoid(input: Tensor):
    if input.is_complex():
        return torch.complex(torch.sigmoid(input.real), torch.sigmoid(input.imag))
    else:
        return torch.sigmoid(input)

def c_tanh(input: Tensor):
    if input.is_complex():
        return torch.complex(torch.tanh(input.real), torch.tanh(input.imag))
    else:
        return torch.tanh(input)

def mod_tanh(input: Tensor, rounding_mode: str = None) -> Tensor:
    if input.is_complex():
        magnitude = torch.abs(input)
        euler_phase = torch.div(input=input, other=magnitude, rounding_mode=rounding_mode)
        return torch.mul(torch.tanh(magnitude), euler_phase).type(input.type())
    else:
        return torch.tanh(input)

def hirose(input: Tensor, m: float = 1., rounding_mode: str = None, inplace: bool = False) -> Tensor:
    if input.is_complex():
        magnitude = torch.abs(input)
        euler_phase = torch.div(input, magnitude)
        if inplace:
            input = torch.mul(torch.tanh(torch.div(input=magnitude, other=torch.pow(m, 2), rounding_mode=rounding_mode)), euler_phase).type(input.type())
            return input
        else:
            hirose = torch.mul(torch.tanh(torch.div(input=magnitude, other=torch.pow(m, 2), rounding_mode=rounding_mode)), euler_phase).type(input.type())
            return hirose
    else:
        if inplace:
            input = torch.tanh(torch.div(input=input, other=torch.pow(m, 2), rounding_mode=rounding_mode)).type(input.type())
            return input
        else:
            hirose = torch.tanh(torch.div(input=input, other=torch.pow(m, 2), rounding_mode=rounding_mode)).type(input.type())
            return hirose

def siglog(input: Tensor):
    return torch.div(input, 1 + torch.abs(input))

def c_cardioid(input: Tensor):
    phase = torch.angle(input)
    return 0.5 * torch.mm(1 + torch.cos(phase), input)

def c_relu(input: Tensor, inplace: bool = False) -> Tensor:
    if input.is_complex():
        return torch.complex(F.relu(input.real, inplace=inplace), F.relu(input.imag, inplace=inplace))
    else:
        return F.relu(input, inplace=inplace)

def mod_relu(input: Tensor, bias: float = -math.sqrt(2), rounding_mode: str = None, inplace: bool = False) -> Tensor:
    if input.is_complex():
        magnitude = torch.abs(input)
        euler_phase = torch.div(input=input, other=magnitude, rounding_mode=rounding_mode)
        if inplace:
            input = torch.mul(F.relu(magnitude + bias, inplace=False), euler_phase).type(input.type())
            return input
        else:
            mod_relu = torch.mul(F.relu(magnitude + bias, inplace=inplace), euler_phase).type(input.type())
            return mod_relu
    else:
        return F.relu(input, inplace=inplace)

def z_relu(input: Tensor, inplace: bool = False) -> Tensor:
    if input.is_complex():
        if inplace:
            mask = torch.zeros_like(input)
            input = torch.where(torch.angle(input) < 0, mask, input)
            input = torch.where(torch.angle(input) > (math.pi / 2), mask, input)
            return input
        else:
            mask = torch.zeros_like(input)
            z_relu = torch.where(torch.angle(input) < 0, mask, input)
            z_relu = torch.where(torch.angle(z_relu) > (math.pi / 2), mask, z_relu)
            return z_relu
    else:
        return F.relu(input, inplace=inplace)

def c_leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    if input.is_complex():
        return torch.complex(F.leaky_relu(input=input.real, negative_slope=negative_slope, inplace=inplace), \
                            F.leaky_relu(input=input.imag, negative_slope=negative_slope, inplace=inplace))
    else:
        return F.leaky_relu(input=input, negative_slope=negative_slope, inplace=inplace)

def tanhexp(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        input = torch.mul(input, torch.tanh(torch.exp(input))).type(input.type())
        return input
    else:
        tanhexp = torch.mul(input, torch.tanh(torch.exp(input))).type(input.type())
        return tanhexp

def sinsig(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        input = torch.mul(input, torch.sin(torch.pi/2 * torch.sigmoid(input))).type(input.type())
        return input
    else:
        sinsig = torch.mul(input, torch.sin(torch.pi/2 * torch.sigmoid(input))).type(input.type())
        return sinsig

def sqr_relu(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        input = F.relu(input=input, inplace=inplace) ** 2
        return input
    else:
        sqr_relu = F.relu(input=input, inplace=inplace) ** 2
        return sqr_relu

def lip_gelu(input: Tensor, approximate = 'none') -> Tensor:
    lip_gelu = F.gelu(input=input, approximate=approximate) / 1.129
    return lip_gelu

def lip_silu(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        input = F.silu(input=input, inplace=inplace) / 1.1
        return input
    else:
        lip_silu = F.silu(input=input, inplace=inplace) / 1.1
        return lip_silu

def lip_mish(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        input = F.mish(input=input, inplace=inplace) / 1.008
        return input
    else:
        lip_mish = F.mish(input=input, inplace=inplace) / 1.008
        return lip_mish

def lip_tanhexp(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        input = tanhexp(input=input, inplace=inplace) / 1.062
        return input
    else:
        lip_tanhexp = tanhexp(input=input, inplace=inplace) / 1.062
        return lip_tanhexp

def lip_sinsig(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        input = torch.mul(input, torch.sin(torch.pi/2 * torch.sigmoid(input))).type(input.type()) / 1.059
        return input
    else:
        lip_sinsig = torch.mul(input, torch.sin(torch.pi/2 * torch.sigmoid(input))).type(input.type()) / 1.059
        return lip_sinsig
    
def lip_srelu(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        input = torch.mul(input, F.relu(input)).type(input.type())
        return input
    else:
        lip_srelu = torch.mul(input, F.relu(input)).type(input.type())
        return lip_srelu

def mod_softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    if input.is_complex():
        return F.softmax(torch.abs(input), dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    else:
        return F.softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

def mod_log_softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    if input.is_complex():
        return F.log_softmax(torch.abs(input), dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    else:
        return F.log_softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

def r_softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    if input.is_complex():
        return F.softmax(input.real, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    else:
        return F.softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

def r_log_softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    if input.is_complex():
        return F.log_softmax(input.real, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    else:
        return F.log_softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)