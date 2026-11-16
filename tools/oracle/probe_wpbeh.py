# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Reference points for gga_x_wpbeh at omega = 0 (direct; libxc default),
# omega = 0.11 (extracted from HSE06 = wpbeh(0) - 0.25 wpbeh(0.11) + pbe_c),
# and omega = 0.4 (extracted from HYB_GGA_XC_LC_WPBE = wpbeh(0.4) + pbe_c).
# Energies reported per volume (x rho). Points span the numeric branches:
# s < 0.08 (EG polynomial), moderate s, s > 8.57 saturation, and w > 14
# (low-density large-w branch for omega = 0.4).
import numpy as np
from pyscf.dft import libxc

CASES = [
    ([0.3, 0.1], [0.0525, 0.0301, 0.0233]),
    ([1.2, 0.9], [0.29, 0.055, 0.1725]),
    ([0.02, 0.05], [0.0005, 0.0006, 0.0009]),
    ([0.5, 0.5], [1e-12, 1e-12, 1e-12]),
    # large s (deep tail): s ~ 20 before rescaling
    ([1e-4, 2e-4], [1.2e-7, 0.5e-7, 4.1e-7]),
    # very low density: w = 0.4/kF > 14
    ([2e-7, 1e-7], [1e-16, 0.3e-16, 0.6e-16]),
]


def grids(r, sg):
    saa, sab, sbb = sg
    ga = np.array([np.sqrt(saa), 0.0, 0.0])
    gbx = sab / np.sqrt(saa)
    gby = np.sqrt(max(sbb - gbx * gbx, 0.0))
    gb = np.array([gbx, gby, 0.0])
    va = np.array([[r[0]], [ga[0]], [ga[1]], [ga[2]]])
    vb = np.array([[r[1]], [gb[0]], [gb[1]], [gb[2]]])
    return va, vb


for (r, sg) in CASES:
    va, vb = grids(r, sg)
    rho = r[0] + r[1]
    w0 = libxc.eval_xc("gga_x_wpbeh,", (va, vb), spin=1)[0][0]
    e_c = libxc.eval_xc(",gga_c_pbe", (va, vb), spin=1)[0][0]
    hse06 = libxc.eval_xc("HSE06", (va, vb), spin=1)[0][0]
    lcwpbe = libxc.eval_xc("HYB_GGA_XC_LC_WPBE", (va, vb), spin=1)[0][0]
    w011 = (w0 + e_c - hse06) / 0.25
    w04 = lcwpbe - e_c
    print(
        f"{r} {sg}:\n"
        f"  wpbeh(0.00)*rho = {w0 * rho:.12f}\n"
        f"  wpbeh(0.11)*rho = {w011 * rho:.12f}\n"
        f"  wpbeh(0.40)*rho = {w04 * rho:.12f}"
    )
