import math
import torch
import torch.nn as nn
from torch import Tensor


class KernelPolynomialLoss(nn.Module):
    def __init__(self, batch_size: int = 2, 
                 max_order: int = 2, 
                 eta: float = 0.1, 
                 enable_simplified: bool = True
                 ) -> None:
        super(KernelPolynomialLoss, self).__init__()
        self.batch_size = batch_size
        self.max_order = max_order
        self.eta = eta
        self.enable_simplified = enable_simplified

    def forward(self, cheb_coef) -> Tensor:
        order = torch.arange(0, self.max_order + 1, device=cheb_coef.device)

        if self.enable_simplified:
            loss = torch.sum(torch.pow(cheb_coef, 2) * torch.pow(order, 2) \
                             / math.pow(self.max_order + 1, 2))
        else:
            loss = torch.sum(torch.pow(cheb_coef, 2) * torch.pow(order, 2) \
                             * math.pi * self.eta)
        
        if self.batch_size > 1:
            loss = loss / self.batch_size
        
        return loss