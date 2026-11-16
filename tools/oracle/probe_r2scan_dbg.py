# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Debug r2scan-x negative-alpha branch: my formula vs libxc over a tau scan.
import numpy as np
from pyscf.dft import libxc

X2S = 0.5 / (6 * np.pi**2) ** (1 / 3)
KF = 0.3 * (6 * np.pi**2) ** (2 / 3)
MU = 10.0 / 81.0
ETA = 0.001
DP2 = 0.361
C = [1.0, -0.667, -0.4445555, -0.663086601049, 1.451297044490,
     -0.887998041597, 0.234528941479, -0.023185843322]
C1, C2P, D, K1, H0X, A1 = 0.667, 0.8, 1.24, 0.065, 1.174, 4.9479
CXS = -0.75 * (3 / np.pi) ** (1 / 3) * 2 ** (1 / 3)


def fx_channel(rho, sig, tau, negmode):
    xs2 = sig / rho ** (8 / 3)
    ts = tau / rho ** (5 / 3)
    a = (ts - xs2 / 8) / (KF + ETA * xs2 / 8)
    p = X2S**2 * xs2
    ssum = sum((k + 1) * c for k, c in enumerate(C))
    c2 = -ssum * (1 - H0X)
    cn = 20 / 27 + ETA * 5 / 3
    x = (cn * c2 * np.exp(-(p**2) / DP2**4) + MU) * p
    h1 = 1 + K1 - K1**2 / (K1 + x)
    if a <= 0:
        if negmode == "exp":
            fa = np.exp(-C1 * a / (1 - a))
        elif negmode == "poly":
            fa = sum(c * a**k for k, c in enumerate(C))
        else:
            fa = 1.0
    elif a <= 2.5:
        fa = sum(c * a**k for k, c in enumerate(C))
    else:
        fa = -D * np.exp(C2P / (1 - a))
    s = X2S * np.sqrt(xs2)
    gx = 1 - np.exp(-A1 / np.sqrt(s))
    return CXS * rho ** (4 / 3) * (h1 + fa * (H0X - h1)) * gx


rho_s, sig_s = 0.5, 0.1
for tau_s in [0.6, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.0005]:
    va = np.vstack([[rho_s], [np.sqrt(sig_s)], [0.0], [0.0], [0.0], [tau_s]])
    ref = libxc.eval_xc("mgga_x_r2scan,", (va, va), spin=1)[0][0] * 2 * rho_s
    ts = tau_s / rho_s ** (5 / 3)
    xs2 = sig_s / rho_s ** (8 / 3)
    a = (ts - xs2 / 8) / (KF + ETA * xs2 / 8)
    # extract the implied f(alpha) from ref
    p = X2S**2 * xs2
    ssum = sum((k + 1) * c for k, c in enumerate(C))
    c2v = -ssum * (1 - H0X)
    cn = 20 / 27 + ETA * 5 / 3
    x = (cn * c2v * np.exp(-(p**2) / DP2**4) + MU) * p
    h1 = 1 + K1 - K1**2 / (K1 + x)
    s = X2S * np.sqrt(xs2)
    gx = 1 - np.exp(-A1 / np.sqrt(s))
    fa_ref = (ref / (2 * CXS * rho_s ** (4 / 3) * gx) - h1) / (H0X - h1)
    fa_exp = np.exp(-C1 * a / (1 - a)) if a <= 0 else None
    print(f"tau={tau_s:7} a={a:+.6f} fa_ref={fa_ref:.10f} "
          f"fa_exp={fa_exp if fa_exp is None else round(fa_exp, 10)} "
          f"lnfa_ref={np.log(fa_ref):+.6e}")
