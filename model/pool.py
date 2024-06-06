import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HartleySpectralPool1d(nn.Module):
    def __init__(self, pool_dim, target_size: int) -> None:
        super(HartleySpectralPool1d, self).__init__()
        assert pool_dim == -1 or pool_dim == -2 or pool_dim == 1 or pool_dim == 2
        self.pool_dim = pool_dim
        self.target_size = target_size

    def fast_hartley_transform_1d(self, input: Tensor, pool_dim: int = -1) -> Tensor:
        input_fourier = torch.fft.fft(input, dim=pool_dim)
        input_hartley = input_fourier.real - input_fourier.imag

        return input_hartley

    def spectral_crop_1d(self, input: Tensor, dim: int = -1) -> Tensor:
        if dim == -1 or dim == 2:
            output = input[:, :, :self.target_size]
        else:
            output = input[:, :self.target_size, :]
        
        return output
        
    def forward(self, input: Tensor) -> Tensor:
        if input.dim() <= 1:
            raise Exception("HartleySpectralPool1D is not supported for dimensions less than 2.")
        elif input.dim() == 2: 
            b, n = input.size()
            input = input.view(b, 1, n)
        elif input.dim() >= 4:
            raise Exception("HartleySpectralPool1D is not supported for dimensions greater than 3.")
        
        input_hartley = self.fast_hartley_transform_1d(input, self.pool_dim)
        
        hartley_pool_1d = self.spectral_crop_1d(input_hartley, self.pool_dim)

        return hartley_pool_1d