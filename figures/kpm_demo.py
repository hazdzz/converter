import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

np.random.seed(42)

def f(x):
    return np.where(x < 0, 0, 1)

def f_plot(x):
    y = f(x).astype(float)
    y[np.isclose(x, 0)] = np.nan
    return y

K = 50

k = np.arange(0, K+1)
xk = np.cos((2*k + 1)/(2*(K + 1)) * np.pi)
yk = f(xk)

cheb_coef = np.polynomial.chebyshev.chebfit(xk, yk, K)
cheb_coef_fejer = (K + 1 - k) / (K + 1) * cheb_coef
cheb_coef_jackson = ((K + 2 - k) * np.cos(k * np.pi / (K + 2)) + np.sin(k * np.pi / (K + 2)) / np.tan(np.pi / (K + 2))) / (K + 2) * cheb_coef
M = 3
cheb_coef_lanczos = np.power(np.sinc(k * np.pi / (K + 1)), M) * cheb_coef
xi = 4
cheb_coef_lorentz = np.sinh(xi * (K + 1 - k) / (K + 1)) / np.sinh(xi) * cheb_coef
temp = k / (K + 1)
numerator = temp - 0.5
denominator = temp * (1 - temp)
ratio = np.empty_like(temp)
valid = denominator != 0
ratio[valid] = numerator[valid] / denominator[valid]
ratio[~valid] = np.where(numerator[~valid] > 0, np.inf, -np.inf)
cheb_coef_vekic = (0.5 - 0.5 * np.tanh(ratio)) * cheb_coef

x_plot = np.linspace(-1, 1, 1001)

p = np.polynomial.chebyshev.chebval(x_plot, cheb_coef)
p_fejer = np.polynomial.chebyshev.chebval(x_plot, cheb_coef_fejer)
p_jackson = np.polynomial.chebyshev.chebval(x_plot, cheb_coef_jackson)
p_lanczos = np.polynomial.chebyshev.chebval(x_plot, cheb_coef_lanczos)
p_lorentz = np.polynomial.chebyshev.chebval(x_plot, cheb_coef_lorentz)
p_vekic = np.polynomial.chebyshev.chebval(x_plot, cheb_coef_vekic)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, f_plot(x_plot), label=r'target function $f(x)$', linewidth=2)
plt.plot(x_plot, p, label=r'KPM w/ Dirichlet kernel', linestyle='-')
plt.plot(x_plot, p_fejer, label=r'KPM w/ Fej\'{e}r kernel', linestyle='--')
plt.plot(x_plot, p_jackson, label=r'KPM w/ Jackson kernel', linestyle='-.')
plt.plot(x_plot, p_lanczos, label=r'KPM w/ Lanczos kernel ($M = 3$)', linestyle=':')
plt.plot(x_plot, p_lorentz, label=r'KPM w/ Lorentz kernel ($\xi = 4$)', linestyle='solid')
plt.plot(x_plot, p_vekic, label=r'KPM w/ Veki\'{c} kernel', linestyle='dashdot')
plt.legend()
plt.grid(True)

ax = plt.gca()

ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_tick_params(direction='inout')
ax.yaxis.set_tick_params(direction='inout')

ax.set_xlim([-1, 1])

ax.plot((1), (0), ls="", marker=">", ms=10, color="k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot((0), (1), ls="", marker="^", ms=10, color="k", transform=ax.get_xaxis_transform(), clip_on=False)

xticks = ax.get_xticks()
yticks = ax.get_yticks()

ax.set_xticks(xticks)
ax.set_xticklabels([f'{tick:.1f}' if tick != 0 else '0.0' for tick in xticks])

ax.set_yticks(yticks)
ax.set_yticklabels([f'{tick:.1f}' if tick != 0 else '' for tick in yticks])

ax.set_xlabel('')
ax.set_ylabel('')

ax.text(-0.1, 1.4, r'$y$', fontsize=14, va='center', ha='center')
ax.text(0.98, -0.1, r'$x$', fontsize=14, va='top', ha='left')

plt.savefig('kpm_step_func.png')
plt.show()