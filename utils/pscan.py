# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/
# Written by Francois Fleuret <francois@fleuret.org>


import os
import random
import torch


class PScan(torch.autograd.Function):
    @staticmethod
    def expand_(A, X):
        if A.size(1) == 1:
            return
        T = 2 * (A.size(1) // 2)
        Aa = A[:, :T].view(A.size(0), T//2, 2, -1)
        Xa = X[:, :T].view(X.size(0), T//2, 2, -1)
        Xa[:, :, 1].add_(Aa[:, :, 1] * Xa[:, :, 0])
        Aa[:, :, 1].mul_(Aa[:, :, 0])
        PScan.expand_(Aa[:, :, 1], Xa[:, :, 1])
        Xa[:, 1:, 0].add_(Aa[:, 1:, 0] * Xa[:, :-1, 1])
        Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])
        if T < A.size(1):
            X[:, -1].add_(A[:, -1] * X[:, -2])
            A[:, -1].mul_(A[:, -2])


    @staticmethod
    def acc_rev_(A, X):
        if X.size(1) == 1:
            return
        T = 2 * (X.size(1) // 2)
        Aa = A[:, -T:].view(A.size(0), T//2, 2, -1)
        Xa = X[:, -T:].view(X.size(0), T//2, 2, -1)
        Xa[:, :, 0].add_(Aa[:, :, 1].conj() * Xa[:, :, 1])
        B = Aa[:, :, 0].clone()
        B[:, 1:].mul_(Aa[:, :-1, 1].conj())
        PScan.acc_rev_(B, Xa[:, :, 0])
        Xa[:, :-1, 1].add_(Aa[:, 1:, 0].conj() * Xa[:, 1:, 0])
        if T < A.size(1):
            X[:, 0].add_(A[:, 1].conj() * X[:, 1])


    @staticmethod
    def forward(ctx, A, X, Y_init):
        ctx.A = A[:, :, None].clone()
        ctx.Y_init = Y_init[:, None, :].clone()
        ctx.A_star = ctx.A.clone()
        ctx.X_star = X.clone()
        PScan.expand_(ctx.A_star, ctx.X_star)
        return ctx.A_star * ctx.Y_init + ctx.X_star


    @staticmethod
    def backward(ctx, grad_output):
        U = grad_output * ctx.A_star.conj()
        A = ctx.A.clone()
        R = grad_output.clone()
        PScan.acc_rev_(A, R)
        Q = ctx.Y_init.expand_as(ctx.X_star).clone()
        Q[:, 1:].mul_(ctx.A_star[:, :-1].conj()).add_(ctx.X_star[:, :-1])
        grad_A = (Q.conj() * R).sum(-1)
        return grad_A, R, U.sum(dim=1)


pscan = PScan.apply


def set_env(seed=42) -> None:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    device = torch.device('cuda')
    set_env(42)

    b, n, d = 1, 8, 4
    x = torch.rand(b, n, d) + 1j * torch.rand(b, n, d)
    ii = torch.rand(b, n-1) + 1j * torch.rand(b, n-1)
    ij = torch.rand(b, n-1) + 1j * torch.rand(b, n-1)

    h = torch.zeros_like(x)
    h_ = torch.zeros_like(x)
    x_ = torch.zeros_like(x)

    x, ii, ij, h, x_ = x.to(device), ii.to(device), ij.to(device), h.to(device), x_.to(device)

    x_[:, :-1, :] = ii.unsqueeze(-1) * x[:, :-1, :]

    x_ = x_.flip(1)
    ii, ij = ii.flip(1), ij.flip(1)
    ij = torch.cat([ij, torch.zeros_like(ij[:, :1])], dim=1)

    h[:, 0, :] = x_[:, 0, :]

    for j in range(1, n):
        h[:, j, :] = x_[:, j, :] + ij[:, j-1].unsqueeze(-1) * h[:, j-1, :]

    h = h.flip(1)

    h_init = x_[:, 0, :].clone()
    p = torch.zeros_like(ij)
    p[:, 1:] = ij[:, :-1].clone() 
    h_ = pscan(p, x_, h_init)
    h_ = h_.flip(1)

    print(h)
    print(h_)