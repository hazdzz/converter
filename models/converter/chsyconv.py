import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class Sine(nn.Module):
    def __init__(self) -> None:
        super(Sine, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


class GenerateEigenvalue(nn.Module):
    def __init__(self, batch_size, length, feat_dim, pool_dim, target_size, drop_prob) -> None:
        super(GenerateEigenvalue, self).__init__()
        self.batch_size = batch_size
        self.length = length
        self.feat_dim = feat_dim
        self.pool_dim = pool_dim
        self.siren = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=True),
            Sine(),
            nn.Dropout(p=drop_prob),
            nn.Linear(feat_dim, feat_dim, bias=True),
            Sine()
        )
        self.avgpool1d = nn.AvgPool1d(kernel_size=target_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = math.sqrt(6 / self.feat_dim)
        init.uniform_(self.siren[0].weight, a=-bound, b=bound)
        init.uniform_(self.siren[3].weight, a=-bound, b=bound)
        init.zeros_(self.siren[0].bias)
        init.zeros_(self.siren[3].bias)

    def forward(self, input: Tensor) -> Tensor:
        input_linear = self.siren(input)

        if self.pool_dim == -1 or self.pool_dim == 2:
            eigenvalue = self.avgpool1d(input_linear).view(self.batch_size, self.length)
        else:
            eigenvalue = self.avgpool1d(input_linear.permute(0, 2 ,1)).view(self.batch_size, self.feat_dim)
        
        eigenvalue = torch.where(eigenvalue == 0, eigenvalue + 1e-8, eigenvalue)

        return eigenvalue


class KernelPolynomial(nn.Module):
    def __init__(self, batch_size, kernel_type: str = 'none', max_order: int = 2, 
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
        init.constant_(self.cheb_coef, 2 / (self.max_order + 1))
        init.constant_(self.cheb_coef[:, 0], 1 / (self.max_order + 1))
        init.ones_(self.gibbs_damp)

        if self.kernel_type == 'fejer':
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

    def forward(self, seq: Tensor) -> Tensor:
        gibbs_damp = self.gibbs_damp.to(seq.device)

        # Tx_0 = 1
        Tx_0 = torch.ones_like(seq)
        ChebGibbs = self.cheb_coef[:, 0].unsqueeze(1)
        if self.max_order == 0:
            return ChebGibbs

        # Tx_1 = x
        Tx_1 = seq
        ChebGibbs = ChebGibbs + Tx_1 * self.cheb_coef[:, 1].unsqueeze(1) * gibbs_damp[:, 1].unsqueeze(1)
        if self.max_order == 1:
            return ChebGibbs

        if self.max_order >= 2:
            for k in range(2, self.max_order + 1):
                # Tx_2 = 2x * Tx_1 - Tx_0 
                Tx_2 = 2.0 * seq * Tx_1 - Tx_0
                ChebGibbs = ChebGibbs + Tx_2 * self.cheb_coef[:, k].unsqueeze(1) * gibbs_damp[:, k].unsqueeze(1)
                Tx_0, Tx_1 = Tx_1, Tx_2

        return ChebGibbs
    

class ChsyConv(nn.Module):
    def __init__(self, batch_size, length, feat_dim, 
                 eigenvalue_drop_prob, value_drop_prob, enable_kpm, 
                 kernel_type: str = 'none', max_order: int = 2, 
                 mu: int = 3, xi: float = 4.0, 
                 stigma: float = 0.5, heta: int = 2) -> None:
        super(ChsyConv, self).__init__()
        self.feat_dim = feat_dim
        self.enable_kpm = enable_kpm
        seq_pool_dim = 2
        # feat_pool_dim = 1
        self.seq_eigenvalue = GenerateEigenvalue(batch_size, length, feat_dim, seq_pool_dim, feat_dim, eigenvalue_drop_prob)
        self.seq_kernel_poly = KernelPolynomial(batch_size, kernel_type, max_order, mu, xi, stigma, heta)
        # self.feat_eigenvalue = GenerateEigenvalue(batch_size, length, feat_dim, feat_pool_dim, length, eigenvalue_drop_prob)
        # self.feat_kernel_poly = KernelPolynomial(batch_size, kernel_type, max_order, mu, xi, stigma, heta)
        self.value_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        self.value_dropout = nn.Dropout(p=value_drop_prob)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.value_linear.weight, gain=1.0)
        init.zeros_(self.value_linear.bias)

    def forward(self, input: Tensor) -> Tensor:
        seq_eigenvalue = self.seq_eigenvalue(input)
        # feat_eigenvalue = self.feat_eigenvalue(input)

        if self.enable_kpm:
            seq_cheb_eigenvalue = self.seq_kernel_poly(seq_eigenvalue)
            # feat_cheb_eigenvalue = self.feat_kernel_poly(feat_eigenvalue)
        else:
            seq_cheb_eigenvalue = seq_eigenvalue
            # feat_cheb_eigenvalue = feat_eigenvalue

        value = self.value_linear(input)
        value = self.value_dropout(value)

        chsyconv1d_input = torch.fft.fft(value, dim=-2)
        chsyconv1d_real = torch.einsum('bn,bnd->bnd', seq_cheb_eigenvalue, chsyconv1d_input.real)
        chsyconv1d_imag = torch.einsum('bn,bnd->bnd', seq_cheb_eigenvalue, chsyconv1d_input.imag)
        chsyconv1d = torch.complex(chsyconv1d_real, chsyconv1d_imag)
        chsyconv_output = torch.fft.ifft(chsyconv1d, dim=-2)

        return chsyconv_output
