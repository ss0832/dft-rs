# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Identify PW91-C's local base: libxc pw91(sig->0) vs lda_c_pw variants.
import numpy as np
from pyscf.dft import libxc

PAPER = [
    (0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294),
    (0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517),
    (0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671),
]
MOD = [
    (0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294),
    (0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517),
    (0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671),
]

def pw_g(rs, P):
    A, a1, b1, b2, b3, b4 = P
    s = np.sqrt(rs)
    den = 2 * A * (b1 * s + b2 * rs + b3 * rs * s + b4 * rs * rs)
    return -2 * A * (1 + a1 * rs) * np.log(1 + 1 / den)

for rho, za in [(4.0, 0.0), (4.0, 0.5), (0.2, 0.5)]:
    ra = rho * (1 + za) / 2
    rb = rho * (1 - za) / 2
    va = np.array([[ra], [0.0], [0.0], [1e-8]])
    vb = np.array([[rb], [0.0], [0.0], [0.0]])
    pw91 = libxc.eval_xc(",gga_c_pw91", (va, vb), spin=1)[0][0]
    ldapw = libxc.eval_xc(",lda_c_pw", (va[:1], vb[:1]), spin=1)[0][0]
    ldapwmod = libxc.eval_xc(",lda_c_pw_mod", (va[:1], vb[:1]), spin=1)[0][0]
    rs = (3 / (4 * np.pi * rho)) ** (1 / 3)
    z4 = za**4
    fz = ((1 + za) ** (4 / 3) + (1 - za) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)
    fpp0 = 8.0 / (9.0 * (2 ** (4 / 3) - 2))
    def eps(P):
        e0 = pw_g(rs, P[0]); e1 = pw_g(rs, P[1]); ac = -pw_g(rs, P[2])
        return e0 + ac * fz / fpp0 * (1 - z4) + (e1 - e0) * fz * z4
    print(f"rho={rho} z={za}: pw91(s->0) {pw91:.12f}  lda_pw {ldapw:.12f} "
          f"lda_pw_mod {ldapwmod:.12f}  mine_paper {eps(PAPER):.12f} mine_mod {eps(MOD):.12f}")
