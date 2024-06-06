import math
import torch
import torch.nn as nn
import torch.nn.init as init
from utility import activation as act
from .pool import HartleySpectralPool1d
from torch import Tensor


class Eigenvalue(nn.Module):
    def __init__(self, batch_size, length, feat_dim, pool_dim, target_size, eigenvalue_drop_prob) -> None:
        super(Eigenvalue, self).__init__()
        self.batch_size = batch_size
        self.length = length
        self.feat_dim = feat_dim
        self.pool_dim = pool_dim
        self.weight_eigenvalue = nn.Parameter(torch.empty(feat_dim, feat_dim))
        self.unitary_transform_1d = HartleySpectralPool1d(pool_dim, target_size)
        self.eigenvalue_dropout = nn.Dropout(p=eigenvalue_drop_prob)
        self.gelu = nn.GELU()
        self.softsign = nn.Softsign()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.uniform_(self.weight_eigenvalue, a=-math.sqrt(3 / self.feat_dim), b=math.sqrt(3 / self.feat_dim))

    def forward(self, input) -> Tensor:
        input_linear = torch.einsum('bnd,de->bne', input, self.weight_eigenvalue)
        input_linear = self.eigenvalue_dropout(input_linear)
        input_linear = self.gelu(input_linear)

        pooled = self.unitary_transform_1d(input_linear)
        if self.pool_dim == -1 or self.pool_dim == 2:
            pooled = pooled.view(self.batch_size, self.length)
        else:
            pooled = pooled.view(self.batch_size, self.feat_dim)

        # eigenvalue = torch.tanh(pooled)
        # eigenvalue = torch.arctan(pooled)
        # eigenvalue = self.softsign(pooled)
        
        eigenvalue_min = pooled.min(dim=1, keepdim=True)[0]
        eigenvalue_max = pooled.max(dim=1, keepdim=True)[0]
        eigenvalue = (pooled - eigenvalue_min) / (eigenvalue_max - eigenvalue_min) * 2 - 1
        
        return eigenvalue
    
class KernelPolynomial(nn.Module):
    def __init__(self, batch_size, kernel_type: str, max_order: int, 
                 mu: int = 3, xi: float = 4.0, 
                 stigma: float = 0.5, heta: int = 2) -> None:
        super(KernelPolynomial, self).__init__()
        assert kernel_type in ['none', 'dirichlet', 'fejer', 'jackson', 
                               'lanczos', 'lorentz', 'vekic', 'wang']
        assert max_order >= 0
        assert mu >= 1

        self.batch_size = batch_size
        self.kernel_type = kernel_type
        self.max_order = max_order
        self.mu = mu
        self.xi = xi
        self.stigma = stigma
        self.heta = heta
        self.cheb_coef = nn.Parameter(torch.empty(batch_size, max_order + 1))
        self.gibbs_damp = torch.empty(batch_size, max_order + 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.cheb_coef)

        if self.kernel_type == 'none' or self.kernel_type == 'dirichlet':
            self.gibbs_damp = torch.ones(self.batch_size, self.max_order + 1)
        elif self.kernel_type == 'fejer':
            for k in range(1, self.max_order + 1):
                self.gibbs_damp[:, k] = torch.tensor(1 - k / (self.max_order + 1))
        elif self.kernel_type == 'jackson':
            # Weiße, A., Wellein, G., Alvermann, A., & Fehske, H. (2006). 
            # The kernel polynomial method. 
            # Reviews of Modern Physics, 78, 275–306.
            # Weiße, A., & Fehske, H. (2008). Chebyshev Expansion Techniques. 
            # In Computational Many-Particle Physics (pp. 545–577). 
            # Springer Berlin Heidelberg.
            c = torch.tensor(torch.pi / (self.max_order + 2))
            for k in range(1, self.max_order + 1):
                self.gibbs_damp[:, k] = ((self.max_order + 2 - k) * torch.sin(c) * \
                                    torch.cos(k * c) + torch.cos(c) * \
                                    torch.sin(k * c)) / ((self.max_order + 2) * \
                                    torch.sin(c))
        elif self.kernel_type == 'lanczos':
            for k in range(1, self.max_order + 1):
                self.gibbs_damp[:, k] = torch.sinc(torch.tensor(k / (self.max_order + 1)))
            self.gibbs_damp = torch.pow(self.gibbs_damp, self.mu)
        elif self.kernel_type == 'lorentz':
            # Vijay, A., Kouri, D., & Hoffman, D. (2004). 
            # Scattering and Bound States: 
            # A Lorentzian Function-Based Spectral Filter Approach. 
            # The Journal of Physical Chemistry A, 108(41), 8987-9003.
            for k in range(1, self.max_order + 1):
                self.gibbs_damp[:, k] = torch.sinh(self.xi * torch.tensor(1 - k / (self.max_order + 1))) / \
                                    torch.sinh(self.xi)
        elif self.kernel_type == 'vekic':
            # M. Vekić, & S. R. White (1993). Smooth boundary 
            # conditions for quantum lattice systems. 
            # Physical Review Letters, 71, 4283–4286.
            for k in range(1, self.max_order + 1):
                self.gibbs_damp[:, k] = torch.tensor(k / (self.max_order + 1))
                self.gibbs_damp[:, k] = 0.5 * (1 - torch.tanh((self.gibbs_damp[k] - 0.5) / \
                                    (self.gibbs_damp[k] * (1 - self.gibbs_damp[k]))))
        elif self.kernel_type == 'wang':
            # Wang, L.W. (1994). Calculating the density of 
            # states and optical-absorption spectra of 
            # large quantum systems by the plane-wave moments method. 
            # Physical Review B, 49, 10154–10158.
            for k in range(1, self.max_order + 1):
                self.gibbs_damp[:, k] = torch.tensor(k / (self.stigma * (self.max_order + 1)))
            self.gibbs_damp = -torch.pow(self.gibbs_damp, self.heta)
            self.gibbs_damp = torch.exp(self.gibbs_damp)

    def forward(self, seq) -> Tensor:
        self.gibbs_damp = self.gibbs_damp.to(seq.device)

        Tx_0 = torch.ones_like(seq)
        # Tx_0 = torch.ones_like(seq) * self.cheb_coef[:, 0].unsqueeze(1)
        ChebGibbs = Tx_0

        Tx_1 = seq 
        ChebGibbs = ChebGibbs + Tx_1 * self.gibbs_damp[:, 1].unsqueeze(1)
        # ChebGibbs = ChebGibbs + Tx_1 * self.cheb_coef[:, 1].unsqueeze(1) * self.gibbs_damp[:, 1].unsqueeze(1)

        if self.max_order >= 2:
            for k in range(2, self.max_order + 1):
                Tx_2 = seq * Tx_1
                Tx_2 = 2. * Tx_2 - Tx_0
                ChebGibbs = ChebGibbs + Tx_2 * self.gibbs_damp[:, k].unsqueeze(1)
                # ChebGibbs = ChebGibbs + Tx_2 * self.cheb_coef[:, k].unsqueeze(1) * self.gibbs_damp[:, k].unsqueeze(1)
                Tx_0, Tx_1 = Tx_1, Tx_2

        return ChebGibbs

class ChsyConv(nn.Module):
    def __init__(self, batch_size, length, feat_dim, 
                 eigenvalue_drop_prob, value_drop_prob,  
                 kernel_type, max_order, mu, xi, stigma, heta) -> None:
        super(ChsyConv, self).__init__()
        self.batch_size = batch_size
        self.length = length
        self.feat_dim = feat_dim
        seq_pool_dim = 2
        target_size = 1
        self.seq_eigenvalue = Eigenvalue(batch_size, length, feat_dim, seq_pool_dim, target_size, eigenvalue_drop_prob)
        self.seq_kernel_poly = KernelPolynomial(batch_size, kernel_type, max_order, mu, xi, stigma, heta)
        self.weight_value = nn.Parameter(torch.empty(feat_dim, feat_dim))
        self.gelu = nn.GELU()
        self.value_dropout = nn.Dropout(p=value_drop_prob)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.uniform_(self.weight_value, a=-math.sqrt(3 / self.feat_dim), b=math.sqrt(3 / self.feat_dim))

    def forward(self, embed) -> Tensor:
        seq_eigenvalue = self.seq_eigenvalue(embed)
        seq_cheb_eigenvalue = self.seq_kernel_poly(seq_eigenvalue)
 
        value = torch.einsum('bnd,de->bne', embed, self.weight_value)
        value = self.value_dropout(value)
        value = self.gelu(value)
        
        chsyconv_input = torch.fft.fft(value, dim=-2)
        chsyconv_real = torch.einsum('bn,bnd->bnd', seq_cheb_eigenvalue, chsyconv_input.real)
        chsyconv_imag = torch.einsum('bn,bnd->bnd', seq_cheb_eigenvalue, chsyconv_input.imag)
        chsyconv = chsyconv_real + 1j * chsyconv_imag
        chsyconv_output = torch.fft.ifft(chsyconv, dim=-2)

        return chsyconv_output