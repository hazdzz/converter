import math
import torch
import torch.nn as nn
from torch import Tensor


class KernelPolynomialLoss(nn.Module):
    def __init__(self, batch_size: int = 1, 
                 max_order: int = 2, 
                 eta: float = 1e-6, 
                 enable_simplified: bool = False
                 ) -> None:
        super(KernelPolynomialLoss, self).__init__()
        self.batch_size = batch_size
        self.max_order = max_order
        self.eta = eta
        self.enable_simplified = enable_simplified

    def forward(self, cheb_coef: Tensor) -> Tensor:
        order = torch.arange(0, self.max_order + 1, device=cheb_coef.device, dtype=cheb_coef.dtype)

        if self.enable_simplified:
            loss = torch.sum(cheb_coef.pow(2) * order.pow(2)) / math.pow(self.max_order + 1, 2)
        else:
            loss = torch.sum(cheb_coef.pow(2) * order.pow(2) * math.pi * self.eta)
        
        loss = loss / self.batch_size
        
        return loss