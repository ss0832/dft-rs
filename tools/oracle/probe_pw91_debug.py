# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Debug PW91-C conventions: side-by-side vs libxc at zeta=0 over (rho, sig)
# with variant constants.
import numpy as np
from pyscf.dft import libxc

PW_MOD = [
    (0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294),
    (0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517),
    (0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671),
]

def pw_g(rs, P):
    A, a1, b1, b2, b3, b4 = P
    s = np.sqrt(rs)
    den = 2 * A * (b1 * s + b2 * rs + b3 * rs * s + b4 * rs * rs)
    return -2 * A * (1 + a1 * rs) * np.log(1 + 1 / den)

def c_rs(rs):
    return 0.001667 + (0.002568 + 0.023266 * rs + 7.389e-6 * rs**2) / (
        1 + 8.723 * rs + 0.472 * rs**2 + 1e4 * 7.389e-6 * rs**3
    )

def mine(rho, sig, beta, nu, cc0, cx, hundred):
    eps = pw_g((3 / (4 * np.pi * rho)) ** (1 / 3), PW_MOD[0])
    kf = (3 * np.pi**2 * rho) ** (1 / 3)
    ks = np.sqrt(4 * kf / np.pi)
    t2 = sig / (4 * ks**2 * rho**2)  # g = 1 at zeta = 0
    rs = (3 / (4 * np.pi * rho)) ** (1 / 3)
    alpha = 0.09
    A = (2 * alpha / beta) / (np.exp(-2 * alpha * eps / (beta**2)) - 1)
    h0 = (beta**2 / (2 * alpha)) * np.log(
        1 + (2 * alpha / beta) * t2 * (1 + A * t2) / (1 + A * t2 + A**2 * t2**2)
    )
    h1 = nu * (c_rs(rs) - cc0 - 3 * cx / 7) * t2 * np.exp(-hundred * (ks**2 / kf**2) * t2)
    return rho * (eps + h0 + h1)

nu0 = (16 / np.pi) * (3 * np.pi**2) ** (1 / 3)
for rho in [0.05, 0.2, 1.0, 4.0]:
    for sig in [1e-4, 0.01, 0.2, 2.0]:
        r = rho / 2
        rho_a = np.array([[r], [0.0], [0.0], [np.sqrt(sig) / 2]])
        ref = libxc.eval_xc(",gga_c_pw91", (rho_a, rho_a), spin=1)[0][0] * rho
        base = mine(rho, sig, nu0 * 0.004235, nu0, 0.004235, -0.001667, 100.0)
        print(f"rho={rho:5} sig={sig:6}: ref {ref:.10e}  diff {base - ref:+.2e}")
