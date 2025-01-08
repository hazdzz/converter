import math
import torch
import torch.nn as nn
import torch.nn.init as init
from . import norm
from torch import Tensor
from typing import Optional


class Sine(nn.Module):
    def __init__(self) -> None:
        super(Sine, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return torch.sin(input)
    

class ComplexDropout(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super(ComplexDropout, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, input1: Tensor, input2: Optional[Tensor] = None) -> Tensor:
        if input1.is_complex():
            input_real, input_imag = input1.real, input1.imag
        else:
            input_real, input_imag = input1, input2

        input = torch.cat([input_real, input_imag], dim=-1)
        dropped = self.dropout(input)
        dropped_real, dropped_imag = dropped.chunk(2, dim=-1)
        output = torch.complex(dropped_real, dropped_imag)

        return output


class GenerateEigenvalue(nn.Module):
    def __init__(self, batch_size, length, feat_dim, pool_dim, drop_prob) -> None:
        super(GenerateEigenvalue, self).__init__()
        self.batch_size = batch_size
        self.length = length
        self.feat_dim = feat_dim
        self.pool_dim = pool_dim
        self.siren = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=True),
            Sine(),
            norm.ScaleNorm(feat_dim),
            nn.Dropout(p=drop_prob),
            nn.Linear(feat_dim, feat_dim, bias=True),
            Sine()
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = math.sqrt(6 / self.feat_dim)
        init.uniform_(self.siren[0].weight, a=-bound, b=bound)
        init.uniform_(self.siren[4].weight, a=-bound, b=bound)
        init.zeros_(self.siren[0].bias)
        init.zeros_(self.siren[4].bias)

    def forward(self, input: Tensor) -> Tensor:
        input_linear = self.siren(input)
        eigenvalue = torch.mean(input_linear, dim=self.pool_dim, keepdim=False)

        return eigenvalue
    

class GenerateHyperparameters(nn.Module):
    def __init__(self, length, feat_dim, drop_prob) -> None:
        super(GenerateHyperparameters, self).__init__()
        self.length = length
        self.feat_dim = feat_dim
        self.siren = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=True),
            Sine(),
            norm.ScaleNorm(feat_dim),
            nn.Dropout(p=drop_prob),
            nn.Linear(feat_dim, feat_dim, bias=True), 
            Sine()
        )
        self.pool1d = nn.AdaptiveAvgPool1d(6)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = math.sqrt(6 / self.feat_dim)
        init.uniform_(self.siren[0].weight, a=-bound, b=bound)
        init.uniform_(self.siren[4].weight, a=-bound, b=bound)
        init.zeros_(self.siren[0].bias)
        init.zeros_(self.siren[4].bias)
    
    def forward(self, input: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        b, n, _ = input.size()
        parameters = self.siren(input)    
        pooled = self.pool1d(parameters).view(b, 6 * n)
        Alpha_l, Beta_l, Gamma_l, Alpha_u, Beta_u, Gamma_u = pooled.chunk(6, dim=1)

        return Alpha_l[:,1:], Beta_l[:,1:], Gamma_l[:,1:], Alpha_u[:,:-1], Beta_u[:,:-1], Gamma_u[:,:-1]
    

def gengerate_dhhp_parameters(alpha: Tensor, beta: Tensor, gamma: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    rad_add = (alpha + beta) / 2 * math.pi
    rad_sub = (alpha - beta) / 2 * math.pi
    gamma = gamma / 2 * math.pi

    g_ii = torch.exp(-1j * rad_add) * torch.cos(gamma)
    g_ij = torch.exp(-1j * rad_sub) * torch.sin(gamma)
    g_ji = -torch.exp(1j * rad_sub) * torch.sin(gamma)
    g_jj = torch.exp(1j * rad_add) * torch.cos(gamma)

    g_ii_conj_trs = g_jj
    g_ij_conj_trs = -g_ij
    g_ji_conj_trs = -g_ji
    g_jj_conj_trs = g_ii

    return g_ii, g_ij, g_ji, g_jj, g_ii_conj_trs, g_ij_conj_trs, g_ji_conj_trs, g_jj_conj_trs


class DHHPTransform(nn.Module):
    def __init__(self) -> None:
        super(DHHPTransform, self).__init__()

    def forward(self, transform: bool, input: Tensor, G_l_ii: Tensor, G_l_ij: Tensor, G_l_ji: Tensor, G_l_jj: Tensor, \
                G_u_ii: Tensor, G_u_ij: Tensor, G_u_ji: Tensor, G_u_jj: Tensor, Diag: Optional[Tensor] = None) -> Tensor:
        B, N, D = input.size()
        Y = torch.zeros_like(input)
        Z = torch.zeros_like(input)

        if (transform is False) and (Diag is not None):
            input = torch.einsum('bn,bnd->bnd', Diag, input)
    
        # Compute Y
        # First row (j = 0)
        Y[:,0,:] = G_u_ii[:,0].unsqueeze(-1) * input[:,0,:] + G_u_ij[:,0].unsqueeze(-1) * input[:,1,:]
    
        # Middle rows (j in [1, N-2])
        Y[:,1:N-1,:] = (G_u_ji[:,:-1].unsqueeze(-1) * input[:,:-2,:] + \
                        (G_u_jj[:,:-1] * G_u_ii[:,1:]).unsqueeze(-1) * input[:,1:N-1,:] + \
                        (G_u_jj[:,:-1] * G_u_ij[:,1:]).unsqueeze(-1) * input[:,2:,:])
    
        # Last row (j = N-1)
        Y[:,N-1,:] = G_u_ji[:,N-2].unsqueeze(-1) * input[:,N-2,:] + G_u_jj[:,N-2].unsqueeze(-1) * input[:,N-1,:]
    
        # Compute Z
        # First row (i = 0)
        Z[:,0,:] = G_l_ii[:,0].unsqueeze(-1) * Y[:,0,:] + G_l_ij[:,0].unsqueeze(-1) * Y[:,1,:]
    
        # Middle rows (i in [1, N-2])
        Z[:,1:N-1,:] = ((G_l_ii[:,1:] * G_l_ji[:,:-1]).unsqueeze(-1) * Y[:,:-2,:] + \
                        (G_l_ii[:,1:] * G_l_jj[:,:-1]).unsqueeze(-1) * Y[:,1:N-1,:] + \
                        G_l_ij[:,1:].unsqueeze(-1) * Y[:,2:,:])
    
        # Last row (i = N-1)
        Z[:,N-1,:] = G_l_ji[:,N-2].unsqueeze(-1) * Y[:,N-2,:] + G_l_jj[:,N-2].unsqueeze(-1) * Y[:,N-1,:]

        if (transform is True) and (Diag is not None):
            Z = torch.einsum('bn,bnd->bnd', Diag, Z)

        return Z
    

class InverseDHHPTransform(nn.Module):
    def __init__(self) -> None:
        super(InverseDHHPTransform, self).__init__()
        self.inverse_dhhp_transform = DHHPTransform()

    def forward(self, transform: bool, input: Tensor, G_l_ii_conj_trs: Tensor, G_l_ij_conj_trs: Tensor, G_l_ji_conj_trs: Tensor, G_l_jj_conj_trs: Tensor, \
                G_u_ii_conj_trs: Tensor, G_u_ij_conj_trs: Tensor, G_u_ji_conj_trs: Tensor, G_u_jj_conj_trs: Tensor, Diag_conj_trs: Optional[Tensor] = None):
        return self.inverse_dhhp_transform(transform, input, G_l_ii_conj_trs, G_l_ij_conj_trs, G_l_ji_conj_trs, G_l_jj_conj_trs, \
                                    G_u_ii_conj_trs, G_u_ij_conj_trs, G_u_ji_conj_trs, G_u_jj_conj_trs, Diag_conj_trs)


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
                                    math.sinh(self.xi)
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
    def __init__(self, batch_size: int, length: int, feat_dim: int, 
                 eigenvalue_drop_prob: float, eigenvector_drop_prob: float, value_drop_prob: float, 
                 enable_kpm: bool, q: float = 0.5, kernel_type: str = 'none', 
                 max_order: int = 2, mu: int = 3, xi: float = 4.0, 
                 stigma: float = 0.5, heta: int = 2) -> None:
        super(ChsyConv, self).__init__()

        self.batch_size = batch_size
        self.length = length
        self.feat_dim = feat_dim
        self.q = q # [0, 0.5]
        self.enable_kpm = enable_kpm
        seq_pool_dim = 2
        self.seq_eigenvalue = GenerateEigenvalue(batch_size, length, feat_dim, seq_pool_dim, eigenvalue_drop_prob)
        self.seq_kernel_poly = KernelPolynomial(batch_size, kernel_type, max_order, mu, xi, stigma, heta)
        self.hyperparameters = GenerateHyperparameters(length, feat_dim, eigenvector_drop_prob)
        self.Theta = nn.Parameter(torch.empty(batch_size, length))
        self.value_linear_real = nn.Linear(feat_dim, feat_dim, bias=False)
        self.value_linear_imag = nn.Linear(feat_dim, feat_dim, bias=False)
        self.value_dropout = ComplexDropout(p=value_drop_prob)
        self.dhhp_transform = DHHPTransform()
        self.inverse_dhhp_transform = InverseDHHPTransform()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.value_linear_real.weight, mean=0.0, std=math.sqrt(0.5))
        init.normal_(self.value_linear_imag.weight, mean=0.0, std=math.sqrt(0.5))
        init.uniform_(self.Theta, a=0.0, b=2.0)

    def forward(self, input: Tensor) -> Tensor:
        Alpha_l, Beta_l, Gamma_l, Alpha_u, Beta_u, Gamma_u = self.hyperparameters(input)

        G_l_ii, G_l_ij, G_l_ji, G_l_jj, G_l_ii_conj_trs, G_l_ij_conj_trs, G_l_ji_conj_trs, G_l_jj_conj_trs = gengerate_dhhp_parameters(Alpha_l, Beta_l, Gamma_l)
        G_u_ii, G_u_ij, G_u_ji, G_u_jj, G_u_ii_conj_trs, G_u_ij_conj_trs, G_u_ji_conj_trs, G_u_jj_conj_trs = gengerate_dhhp_parameters(Alpha_u, Beta_u, Gamma_u)

        Diag = torch.exp(1j * math.pi * self.Theta)
        Diag_conj_trs = Diag.conj()

        seq_eigenvalue = self.seq_eigenvalue(input)
        if self.enable_kpm:
            seq_cheb_eigenvalue = self.seq_kernel_poly(seq_eigenvalue)
        else:
            seq_cheb_eigenvalue = seq_eigenvalue

        digraph_conv_eigenvalue = torch.exp(2j * math.pi * self.q * seq_cheb_eigenvalue)

        value_real = self.value_linear_real(input)
        value_imag = self.value_linear_imag(input)
        value = self.value_dropout(value_real, value_imag)

        unitary_conv_1d_forward = self.dhhp_transform(True, value, G_l_ii, G_l_ij, G_l_ji, G_l_jj, G_u_ii, G_u_ij, G_u_ji, G_u_jj, Diag)
        unitary_conv_1d = torch.einsum('bn,bnd->bnd', digraph_conv_eigenvalue, unitary_conv_1d_forward)
        unitary_conv_1d_inverse = self.inverse_dhhp_transform(False, unitary_conv_1d, G_u_ii_conj_trs, G_u_ij_conj_trs, G_u_ji_conj_trs, G_u_jj_conj_trs, \
                                                            G_l_ii_conj_trs, G_l_ij_conj_trs, G_l_ji_conj_trs, G_l_jj_conj_trs, Diag_conj_trs)
        
        return unitary_conv_1d_inverse