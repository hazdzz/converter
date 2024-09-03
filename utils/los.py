import math
import torch
import torch.nn as nn
from torch import Tensor


class KernelPolynomialLoss(nn.Module):
    def __init__(self, batch_size: int = 1, 
                 max_order: int = 2, 
                 eta: float = 1e-2
                 ) -> None:
        super(KernelPolynomialLoss, self).__init__()
        self.batch_size = batch_size
        self.max_order = max_order
        self.eta = eta

    def forward(self, cheb_coef: Tensor) -> Tensor:
        order = torch.arange(0, self.max_order + 1, device=cheb_coef.device, dtype=cheb_coef.dtype).unsqueeze(0).repeat(self.batch_size, 1)

        loss = torch.sum(cheb_coef.pow(2) * order.pow(2), dim=-1) * math.pi * self.eta
        
        return loss.mean()