import torch.nn.functional as F
import utils.functional as cF
from torch.nn.modules import Module
from torch import Tensor


class _ComplexDropoutNd(Module):
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(_ComplexDropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, \
                             but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)


class ComplexDropout(_ComplexDropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return cF.complex_fcaller(F.dropout, input, self.p, self.training, self.inplace)


class ComplexDropout2d(_ComplexDropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return cF.complex_fcaller(F.dropout2d, input, self.p, self.training, self.inplace)


class ComplexDropout3d(_ComplexDropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return cF.complex_fcaller(F.dropout3d, input, self.p, self.training, self.inplace)


class ComplexAlphaDropout(_ComplexDropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return cF.complex_fcaller(F.alpha_dropout, input, self.p, self.training)


class ComplexFeatureAlphaDropout(_ComplexDropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return cF.complex_fcaller(F.feature_alpha_dropout, input, input, self.p, self.training)