# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Decompose libxc gga_c_pw91's gradient term H = exc - eps_local at zeta=0
# and compare H/t^2 against my H0/t^2 + H1/t^2 over sigma, to locate the
# structural difference (small-t coefficient vs A / screening shape).
import numpy as np
from pyscf.dft import libxc

PAPER0 = (0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)

def pw_g(rs, P):
    A, a1, b1, b2, b3, b4 = P
    s = np.sqrt(rs)
    den = 2 * A * (b1 * s + b2 * rs + b3 * rs * s + b4 * rs * rs)
    return -2 * A * (1 + a1 * rs) * np.log(1 + 1 / den)

def c_rs(rs):
    return 0.001667 + (0.002568 + 0.023266 * rs + 7.389e-6 * rs**2) / (
        1 + 8.723 * rs + 0.472 * rs**2 + 1e4 * 7.389e-6 * rs**3
    )

NU = (16 / np.pi) * (3 * np.pi**2) ** (1 / 3)
BETA = NU * 0.004235
ALPHA = 0.09

for rho in [0.4, 1.0, 2.1]:
    rs = (3 / (4 * np.pi * rho)) ** (1 / 3)
    eps = pw_g(rs, PAPER0)
    kf = (3 * np.pi**2 * rho) ** (1 / 3)
    ks = np.sqrt(4 * kf / np.pi)
    A = (2 * ALPHA / BETA) / (np.exp(-2 * ALPHA * eps / BETA**2) - 1)
    dc = c_rs(rs) - 0.004235 - 3 * (-0.001667) / 7
    print(f"rho={rho}: rs={rs:.6f} eps={eps:.10f} A={A:.8f} "
          f"nu*dc={NU*dc:.10f} beta={BETA:.10f}")
    for sig in [1e-6, 1e-3, 0.03, 0.3, 1.0, 4.0]:
        r = rho / 2
        gr = np.sqrt(sig) / 2
        va = np.array([[r], [gr], [0.0], [0.0]])
        ref = libxc.eval_xc(",gga_c_pw91", (va, va), spin=1)[0][0]
        t2 = sig / (4 * ks**2 * rho**2)
        h_ref = ref - eps
        h0 = (BETA**2 / (2 * ALPHA)) * np.log(
            1 + (2 * ALPHA / BETA) * t2 * (1 + A * t2) / (1 + A * t2 + A**2 * t2**2)
        )
        h1 = NU * dc * t2 * np.exp(-100 * (ks**2 / kf**2) * t2)
        print(f"  sig={sig:8}: t2={t2:.5e} Href/t2={h_ref/t2:.8f} "
              f"h0/t2={h0/t2:.8f} h1/t2={h1/t2:.8f} "
              f"diff(H)={h0+h1-h_ref:+.3e}")
