# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Reference points for gga_x_hjs_pbe at omega = 0.4, extracted as
# HYB_GGA_XC_LC_WPBE_WHS (= 1.0 hjs_pbe(0.4) + 1.0 gga_c_pbe) minus
# gga_c_pbe, evaluated pointwise. Energies reported per volume (x rho).
import numpy as np
from pyscf.dft import libxc

CASES = [
    ([0.3, 0.1], [0.0525, 0.0301, 0.0233]),
    ([1.2, 0.9], [0.29, 0.055, 0.1725]),
    ([0.02, 0.05], [0.0005, 0.0006, 0.0009]),
    ([0.5, 0.5], [1e-12, 1e-12, 1e-12]),
]

for (r, sg) in CASES:
    saa, sab, sbb = sg
    # eval_xc spin-polarized input: per-spin rho and gradient components.
    # Choose gradients realizing the requested (saa, sab, sbb):
    ga = np.array([np.sqrt(saa), 0.0, 0.0])
    # gb with gb.ga = sab and gb.gb = sbb
    gbx = sab / np.sqrt(saa)
    gby = np.sqrt(max(sbb - gbx * gbx, 0.0))
    gb = np.array([gbx, gby, 0.0])
    va = np.array([[r[0]], [ga[0]], [ga[1]], [ga[2]]])
    vb = np.array([[r[1]], [gb[0]], [gb[1]], [gb[2]]])
    rho = r[0] + r[1]
    e_full = libxc.eval_xc("HYB_GGA_XC_LC_WPBE_WHS", (va, vb), spin=1)[0][0]
    e_c = libxc.eval_xc(",gga_c_pbe", (va, vb), spin=1)[0][0]
    print(f"{r} {sg}: hjs*rho = {(e_full - e_c) * rho:.12f}")
