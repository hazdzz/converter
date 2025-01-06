import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class Sine(nn.Module):
    def __init__(self) -> None:
        super(Sine, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return torch.sin(input)


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

        return eigenvalue
    

class GenerateHyperparameters(nn.Module):
    def __init__(self, length, feat_dim, drop_prob) -> None:
        super(GenerateHyperparameters, self).__init__()
        self.length = length
        self.feat_dim = feat_dim
        self.siren = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=True),
            Sine(),
            nn.Dropout(p=drop_prob),
            nn.Linear(feat_dim, feat_dim, bias=True),
            Sine()
        )
        self.pool1d = nn.AdaptiveAvgPool1d(7)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = math.sqrt(6 / self.feat_dim)
        init.uniform_(self.siren[0].weight, a=-bound, b=bound)
        init.uniform_(self.siren[3].weight, a=-bound, b=bound)
        init.zeros_(self.siren[0].bias)
        init.zeros_(self.siren[3].bias)
    
    def forward(self, input: Tensor) -> Tensor:
        b, n, _ = input.size()
        parameters = self.siren(input)
        parameters = self.pool1d(parameters).view(b, 7 * n)
        Alpha_l, Beta_l, Gamma_l, Alpha_u, Beta_u, Gamma_u, Theta = parameters.chunk(7, dim=1)

        return Alpha_l[:,1:], Beta_l[:,1:], Gamma_l[:,1:], Alpha_u[:,:-1], Beta_u[:,:-1], Gamma_u[:,:-1], Theta


class GenerateDHHPParameters(nn.Module):
    def __init__(self) -> None:
        super(GenerateDHHPParameters, self).__init__()

    def _compute_givens_rotation(self, alpha: Tensor, beta: Tensor, gamma: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rad_add = (alpha + beta) / 2 * math.pi
        rad_sub = (alpha - beta) / 2 * math.pi
        gamma = gamma / 2 * math.pi

        g_ii = torch.complex(-torch.cos(rad_add), -torch.sin(rad_add)) * torch.cos(gamma)
        g_ij = torch.complex(-torch.cos(rad_sub), -torch.sin(rad_sub)) * torch.sin(gamma)
        g_ji = -torch.complex(torch.cos(rad_sub), torch.sin(rad_sub)) * torch.sin(gamma)
        g_jj = torch.complex(torch.cos(rad_add), torch.sin(rad_add)) * torch.cos(gamma)

        return g_ii, g_ij, g_ji, g_jj
    
    def forward(self, Alpha, Beta, Gamma) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        G_ii, G_ij, G_ji, G_jj = self._compute_givens_rotation(Alpha, Beta, Gamma)

        return G_ii, G_ij, G_ji, G_jj
    

class DHHPTransform(nn.Module):
    def __init__(self, M) -> None:
        super(DHHPTransform, self).__init__()
        self.M = M

    def forward(self, input: Tensor, G_l_ii: Tensor, G_l_ij: Tensor, G_l_ji: Tensor, G_l_jj: Tensor, \
                G_u_ii: Tensor, G_u_ij: Tensor, G_u_ji: Tensor, G_u_jj: Tensor, Diag: Tensor, transform: bool) -> Tensor:
        B, N, D = input.size()
        Y = torch.zeros_like(input)
        Z = torch.zeros_like(input)

        if transform is True:
            if N % 2 == 0:
                # Permutation
                input = input.reshape(B, N // self.M, self.M, D).transpose(1, 2).reshape(B, N, D)
            else:
                pass
        else:
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

        if transform is True:
            Z = torch.einsum('bn,bnd->bnd', Diag, Z)
        else:
            if N % 2 == 0:
                # Permutation
                Z = Z.reshape(B, N // self.M, self.M, D).transpose(1, 2).reshape(B, N, D)
            else:
                pass

        return Z
    

class InverseDHHPTransform(nn.Module):
    def __init__(self, M) -> None:
        super(InverseDHHPTransform, self).__init__()
        self.inverse_dhhp_transform = DHHPTransform(M)

    def forward(self, input: Tensor, G_l_ii_conj_trs: Tensor, G_l_ij_conj_trs: Tensor, G_l_ji_conj_trs: Tensor, G_l_jj_conj_trs: Tensor, \
                G_u_ii_conj_trs: Tensor, G_u_ij_conj_trs: Tensor, G_u_ji_conj_trs: Tensor, G_u_jj_conj_trs: Tensor, Diag_conj_trs: Tensor, transform: bool):
        return self.inverse_dhhp_transform(input, G_l_ii_conj_trs, G_l_ij_conj_trs, G_l_ji_conj_trs, G_l_jj_conj_trs, \
                                    G_u_ii_conj_trs, G_u_ij_conj_trs, G_u_ji_conj_trs, G_u_jj_conj_trs, Diag_conj_trs, transform)


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
        self.seq_eigenvalue = GenerateEigenvalue(batch_size, length, feat_dim, \
                            seq_pool_dim, feat_dim, eigenvalue_drop_prob)
        self.seq_kernel_poly = KernelPolynomial(batch_size, kernel_type, max_order, mu, xi, stigma, heta)
        self.hyperparameters = GenerateHyperparameters(length, feat_dim, eigenvector_drop_prob)
        self.givens_parameters_l = GenerateDHHPParameters()
        self.givens_parameters_u = GenerateDHHPParameters()
        self.value_linear_real = nn.Linear(feat_dim, feat_dim, bias=True)
        self.value_linear_imag = nn.Linear(feat_dim, feat_dim, bias=True)
        self.value_dropout = nn.Dropout(p=value_drop_prob)
        permutation_dim = 16
        self.dhhp_transform = DHHPTransform(permutation_dim)
        self.inverse_dhhp_transform = InverseDHHPTransform(permutation_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.value_linear_real.weight, gain=1.0)
        init.xavier_uniform_(self.value_linear_imag.weight, gain=1.0)
        init.zeros_(self.value_linear_real.bias)
        init.zeros_(self.value_linear_imag.bias)

    def forward(self, input: Tensor) -> Tensor:
        Alpha_l, Beta_l, Gamma_l, Alpha_u, Beta_u, Gamma_u, Theta = self.hyperparameters(input)
        G_l_ii, G_l_ij, G_l_ji, G_l_jj = self.givens_parameters_l(Alpha_l, Beta_l, Gamma_l)
        G_u_ii, G_u_ij, G_u_ji, G_u_jj = self.givens_parameters_u(Alpha_u, Beta_u, Gamma_u)
        Theta = Theta * math.pi
        Diag = torch.complex(torch.cos(Theta), torch.sin(Theta))

        G_l_ii_conj_trs, G_l_ij_conj_trs, G_l_ji_conj_trs, G_l_jj_conj_trs = G_l_jj, -G_l_ij, -G_l_ji, G_l_ii
        G_u_ii_conj_trs, G_u_ij_conj_trs, G_u_ji_conj_trs, G_u_jj_conj_trs = G_u_jj, -G_u_ij, -G_u_ji, G_u_ii
        Diag_conj_trs = Diag.conj()

        seq_eigenvalue = self.seq_eigenvalue(input)
        if self.enable_kpm:
            seq_cheb_eigenvalue = self.seq_kernel_poly(seq_eigenvalue)
        else:
            seq_cheb_eigenvalue = seq_eigenvalue

        seq_cheb_eigenvalue = 2.0 * math.pi * self.q * seq_cheb_eigenvalue
        seq_cheb_eigenvalue = torch.complex(torch.cos(seq_cheb_eigenvalue), torch.sin(seq_cheb_eigenvalue))

        value_real = self.value_linear_real(input)
        value_imag = self.value_linear_imag(input)
        value_real = self.value_dropout(value_real)
        value_imag = self.value_dropout(value_imag)
        value = torch.complex(value_real, value_imag)

        unitary_conv_1d = self.dhhp_transform(value, G_l_ii, G_l_ij, G_l_ji, G_l_jj, G_u_ii, G_u_ij, G_u_ji, G_u_jj, Diag, True)
        unitary_conv_1d = torch.einsum('bn,bnd->bnd', seq_cheb_eigenvalue, unitary_conv_1d)
        unitary_conv_1d = self.inverse_dhhp_transform(unitary_conv_1d, G_l_ii_conj_trs, G_l_ij_conj_trs, G_l_ji_conj_trs, G_l_jj_conj_trs, \
                                                      G_u_ii_conj_trs, G_u_ij_conj_trs, G_u_ji_conj_trs, G_u_jj_conj_trs, Diag_conj_trs, False)
        
        return unitary_conv_1d