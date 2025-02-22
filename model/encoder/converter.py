import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from typing import Optional, Union
from .. import embedding, norm


def complex_fcaller(funtional_handle, *args):
    return torch.complex(funtional_handle(args[0].real, *args[1:]), funtional_handle(args[0].imag, *args[1:]))


class _ComplexDropoutNd(nn.Module):
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
        return complex_fcaller(F.dropout, input, self.p, self.training, self.inplace)
    

class SeparateComplexDropout(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super(SeparateComplexDropout, self).__init__()
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


class Sine(nn.Module):
    def __init__(self) -> None:
        super(Sine, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return torch.sin(input)


class GenerateEigenvalue(nn.Module):
    def __init__(self, feat_dim: int, pool_dim: int, drop_prob: float = 0.1) -> None:
        super(GenerateEigenvalue, self).__init__()
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
    

class GenerateParameters(nn.Module):
    def __init__(self, feat_dim: int, drop_prob: float = 0.1) -> None:
        super(GenerateParameters, self).__init__()
        self.feat_dim = feat_dim
        self.siren = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=True),
            Sine(),
            norm.ScaleNorm(feat_dim),
            nn.Dropout(p=drop_prob),
            nn.Linear(feat_dim, feat_dim, bias=True), 
            Sine(),
            norm.FixNorm(feat_dim)
        )
        self.pool = nn.AdaptiveAvgPool1d(7)

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
        Alpha_l, Beta_l, Gamma_l, Alpha_u, Beta_u, Gamma_u, Theta = self.pool(parameters).view(b, 7 * n).chunk(7, dim=1)
        
        return Alpha_l[:,0:-1], Beta_l[:,0:-1], Gamma_l[:,0:-1], Alpha_u[:,1:n], Beta_u[:,1:n], Gamma_u[:,1:n], Theta
    

def gengerate_dhhp_parameters(alpha: Tensor, beta: Tensor, gamma: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    A = (alpha + beta) * math.pi
    B = (alpha - beta) * math.pi
    C = gamma * math.pi

    g_ii = torch.exp(-1j * A) * torch.cos(C)
    g_ij = -torch.exp(1j * B) * torch.sin(C)
    g_ji = torch.exp(-1j * B) * torch.sin(C)
    g_jj = torch.exp(1j * A) * torch.cos(C)

    g_ii_conj_trs = g_jj
    g_ij_conj_trs = -g_ij
    g_ji_conj_trs = -g_ji
    g_jj_conj_trs = g_ii

    return g_ii, g_ij, g_ji, g_jj, g_ii_conj_trs, g_ij_conj_trs, g_ji_conj_trs, g_jj_conj_trs


class DHHPTransform(nn.Module):
    def __init__(self, transform: bool = True, permutation_dim: Optional[int] = None) -> None:
        super(DHHPTransform, self).__init__()
        self.transform = transform
        self.M = permutation_dim

    def forward(self, X: Tensor, G_l_ii: Tensor, G_l_ij: Tensor, G_l_ji: Tensor, G_l_jj: Tensor, \
                G_u_ii: Tensor, G_u_ij: Tensor, G_u_ji: Tensor, G_u_jj: Tensor, Diag: Optional[Tensor] = None) -> Tensor:
        B, N, D = X.size()
        Y = torch.zeros_like(X)
        Z = torch.zeros_like(X)

        if (self.transform is True) and (self.M is not None):
            X = X.reshape(B, N // self.M, self.M, D).transpose(1, 2).reshape(B, N, D)
        elif (self.transform is False) and (Diag is not None):
            X = torch.einsum('bn,bnd->bnd', Diag, X)

        Y[:, -1, :] = G_u_ji[:, N-2].unsqueeze(-1) * X[:, -2, :] + G_u_jj[:, N-2].unsqueeze(-1) * X[:, -1, :]

        Y[:, 1:-1, :] = G_u_ji[:, :N-2].unsqueeze(-1) * X[:, :-2, :] \
                    + (G_u_jj[:, :N-2] * G_u_ii[:, 1:N-1]).unsqueeze(-1) * X[:, 1:-1, :] \
                    + (G_u_jj[:, :N-2] * G_u_ij[:, 1:N-1]).unsqueeze(-1) * X[:, 2:, :]

        Y[:, 0, :] = G_u_ii[:, 0].unsqueeze(-1) * X[:, 0, :] + G_u_ij[:, 0].unsqueeze(-1) * X[:, 1, :]

        Z[:, 0, :] = G_l_ii[:, 0].unsqueeze(-1) * Y[:, 0, :] + G_l_ij[:, 0].unsqueeze(-1) * Y[:, 1, :]

        Z[:, 1:-1, :] = (G_l_ii[:, 1:N-1] * G_l_ji[:, 0:N-2]).unsqueeze(-1) * Y[:, :-2, :] \
                    + (G_l_ii[:, 1:N-1] * G_l_jj[:, 0:N-2]).unsqueeze(-1) * Y[:, 1:-1, :] \
                    + G_l_ij[:, 1:N-1].unsqueeze(-1) * Y[:, 2:, :]

        Z[:, -1, :] = G_l_ji[:, N-2].unsqueeze(-1) * Y[:, -2, :] + G_l_jj[:, N-2].unsqueeze(-1) * Y[:, -1, :]

        if (self.transform is True) and (Diag is not None):
            Z = torch.einsum('bn,bnd->bnd', Diag, Z)
        elif (self.transform is False) and (self.M is not None):
            Z = Z.reshape(B, self.M, N // self.M, D).transpose(1, 2).reshape(B, N, D)

        return Z
    

class InverseDHHPTransform(nn.Module):
    def __init__(self, transform: bool = False, permutation_dim: Optional[int] = None) -> None:
        super(InverseDHHPTransform, self).__init__()
        self.inverse_dhhp_transform = DHHPTransform(transform, permutation_dim)

    def forward(self, X: Tensor, G_l_ii_conj_trs: Tensor, G_l_ij_conj_trs: Tensor, G_l_ji_conj_trs: Tensor, G_l_jj_conj_trs: Tensor, \
                G_u_ii_conj_trs: Tensor, G_u_ij_conj_trs: Tensor, G_u_ji_conj_trs: Tensor, G_u_jj_conj_trs: Tensor, Diag_conj_trs: Optional[Tensor] = None):
        return self.inverse_dhhp_transform(X, G_u_ii_conj_trs, G_u_ij_conj_trs, G_u_ji_conj_trs, G_u_jj_conj_trs, \
                                           G_l_ii_conj_trs, G_l_ij_conj_trs, G_l_ji_conj_trs, G_l_jj_conj_trs, Diag_conj_trs)


class KernelPolynomial(nn.Module):
    def __init__(self, batch_size: int, kernel_type: str = 'none', max_order: int = 2, 
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
        ChebGibbs = Tx_0 * self.cheb_coef[:, 0].unsqueeze(1)
        if self.max_order == 0:
            return ChebGibbs

        # Tx_1 = x
        Tx_1 = seq
        ChebGibbs = ChebGibbs + Tx_1 * self.cheb_coef[:, 1].unsqueeze(1) * gibbs_damp[:, 1].unsqueeze(1)
        if self.max_order == 1:
            return ChebGibbs

        if self.max_order >= 2:
            for k in range(2, self.max_order + 1):
                # Tx_2 = 2 * x * Tx_1 - Tx_0 
                Tx_2 = 2.0 * seq * Tx_1 - Tx_0
                ChebGibbs = ChebGibbs + Tx_2 * self.cheb_coef[:, k].unsqueeze(1) * gibbs_damp[:, k].unsqueeze(1)
                Tx_0, Tx_1 = Tx_1, Tx_2

        return ChebGibbs
    

class Kernelution(nn.Module):
    def __init__(self, batch_size: int, length: int, feat_dim: int, 
                 eigenvalue_drop_prob: float = 0.1, eigenvector_drop_prob: float = 0.1, value_drop_prob: float = 0.1, 
                 permutation_dim: Union[int, str, None] = None, 
                 enable_kpm: bool = True, kernel_type: str = 'none', 
                 max_order: int = 2, mu: int = 3, xi: float = 4.0, 
                 stigma: float = 0.5, heta: int = 2) -> None:
        super(Kernelution, self).__init__()
        self.batch_size = batch_size
        self.length = length
        self.feat_dim = feat_dim
        self.enable_kpm = enable_kpm
        seq_pool_dim = 2
        self.seq_eigenvalue = GenerateEigenvalue(feat_dim, seq_pool_dim, eigenvalue_drop_prob)
        self.seq_kernel_poly = KernelPolynomial(batch_size, kernel_type, max_order, mu, xi, stigma, heta)
        self.givens_parameters = GenerateParameters(feat_dim, eigenvector_drop_prob)
        self.value_linear_real = nn.Linear(feat_dim, feat_dim, bias=False)
        self.value_linear_imag = nn.Linear(feat_dim, feat_dim, bias=False)
        # self.value_dropout = ComplexDropout(p=value_drop_prob)
        self.value_dropout = SeparateComplexDropout(p=value_drop_prob)
        if (permutation_dim is None) or (permutation_dim == 'none') or (permutation_dim == 0):
            permutation_dim = None
        self.dhhp_transform = DHHPTransform(True, permutation_dim)
        self.inverse_dhhp_transform = InverseDHHPTransform(False, permutation_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.value_linear_real.weight, mean=0.0, std=math.sqrt(0.5))
        init.normal_(self.value_linear_imag.weight, mean=0.0, std=math.sqrt(0.5))

    def forward(self, input: Tensor) -> Tensor:
        # Hyperparameters for 1-DHHP
        Alpha_l, Beta_l, Gamma_l, Alpha_u, Beta_u, Gamma_u, Theta = self.givens_parameters(input)
        G_l_ii, G_l_ij, G_l_ji, G_l_jj, G_l_ii_conj_trs, G_l_ij_conj_trs, G_l_ji_conj_trs, G_l_jj_conj_trs = gengerate_dhhp_parameters(Alpha_l, Beta_l, Gamma_l)
        G_u_ii, G_u_ij, G_u_ji, G_u_jj, G_u_ii_conj_trs, G_u_ij_conj_trs, G_u_ji_conj_trs, G_u_jj_conj_trs = gengerate_dhhp_parameters(Alpha_u, Beta_u, Gamma_u)
        Diag = torch.exp(2j * math.pi * Theta)
        Diag_conj_trs = Diag.conj()

        # Eigenvalues
        seq_eigenvalue = self.seq_eigenvalue(input)
        if self.enable_kpm is True:
            seq_cheb_eigenvalue = self.seq_kernel_poly(seq_eigenvalue)
        else:
            seq_cheb_eigenvalue = seq_eigenvalue
        digraph_conv_eigenvalue = torch.exp(1j * seq_cheb_eigenvalue)

        # Value
        value_real = self.value_linear_real(input)
        value_imag = self.value_linear_imag(input)
        value = torch.complex(value_real, value_imag)
        value = self.value_dropout(value)

        # Kernerlution
        unitary_conv_1d_forward = self.dhhp_transform(value, G_l_ii, G_l_ij, G_l_ji, G_l_jj, G_u_ii, G_u_ij, G_u_ji, G_u_jj, Diag)
        unitary_conv_1d = torch.einsum('bn,bnd->bnd', digraph_conv_eigenvalue, unitary_conv_1d_forward)
        unitary_conv_1d_inverse = self.inverse_dhhp_transform(unitary_conv_1d, G_u_ii_conj_trs, G_u_ij_conj_trs, G_u_ji_conj_trs, G_u_jj_conj_trs, \
                                                            G_l_ii_conj_trs, G_l_ij_conj_trs, G_l_ji_conj_trs, G_l_jj_conj_trs, Diag_conj_trs)
        
        return unitary_conv_1d_inverse
    

class GatedFeedForward(nn.Module):
    def __init__(self, feat_dim: int, hid_dim: int, gffn_drop_prob: float = 0.1) -> None:
        super(GatedFeedForward, self).__init__()
        self.linear1_real = nn.Linear(feat_dim, hid_dim, bias=True)
        self.linear1_imag = nn.Linear(feat_dim, hid_dim, bias=True)
        self.linear2 = nn.Linear(hid_dim, feat_dim, bias=True)
        self.softplus = nn.Softplus(beta=1.0, threshold=5.0)
        self.gffn_dropout = nn.Dropout(p=gffn_drop_prob)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.linear1_real.weight, gain=1.0)
        init.xavier_uniform_(self.linear1_imag.weight, gain=1.0)
        init.xavier_uniform_(self.linear2.weight, gain=1.0)
        init.zeros_(self.linear1_real.bias)
        init.zeros_(self.linear1_imag.bias)
        init.zeros_(self.linear2.bias)

    def forward(self, input: Tensor) -> Tensor:
        if input.is_complex():
            input_real, input_imag = input.real, input.imag
        else:
            input_real, input_imag = input, input

        linear1_real = self.linear1_real(input_real)
        linear1_imag = self.linear1_imag(input_imag)
        linear = self.softplus(linear1_real) * torch.tanh(linear1_imag)
        linear = self.gffn_dropout(linear)
        ffn = self.linear2(linear)

        return ffn


class ConverterEncoder(nn.Module):
    def __init__(self, args) -> None:
        super(ConverterEncoder, self).__init__()
        self.embedding = embedding.Embedding(args.pe_type, 
                                             args.pooling_type, 
                                             args.vocab_size, 
                                             args.max_seq_len, 
                                             args.embed_dim, 
                                             args.pe_drop_prob, 
                                             args.embed_drop_prob)
        self.kernelution = Kernelution(args.batch_size, 
                                       args.max_seq_len, 
                                       args.embed_dim, 
                                       args.xformer.converter.eigenvalue_drop_prob, 
                                       args.xformer.converter.eigenvector_drop_prob, 
                                       args.value_drop_prob, 
                                       args.xformer.converter.permutation_dim, 
                                       args.xformer.converter.enable_kpm, 
                                       args.xformer.converter.kernel_type, 
                                       args.xformer.converter.max_order, 
                                       args.xformer.converter.mu, 
                                       args.xformer.converter.xi, 
                                       args.xformer.converter.stigma, 
                                       args.xformer.converter.heta)
        self.gffn = GatedFeedForward(args.embed_dim, args.hidden_dim, args.ffn_drop_prob)
        self.kernelution_norm = norm.ScaleNorm(args.embed_dim)
        self.gffn_norm = norm.ScaleNorm(args.embed_dim)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, input: Tensor) -> Tensor:
        alpha = torch.clamp(self.alpha, min=0.0, max=1.0).to(input.device)
        
        embed = self.embedding(input)

        kernelution = self.kernelution(embed) + embed
        kernelution_normed = self.kernelution_norm(kernelution)

        gffn = self.gffn(kernelution_normed) + alpha * kernelution_normed.real + (1.0 - alpha) * kernelution_normed.imag
        converter_encoder = self.gffn_norm(gffn)

        return converter_encoder
