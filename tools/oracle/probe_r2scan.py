# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Reference points for mgga_x_r2scan / mgga_c_r2scan (energy per volume).
import numpy as np
from pyscf.dft import libxc

CASES = [
    [0.3, 0.1, 0.0425, 0.031, 0.0233, 0.35, 0.12],
    [1.2, 0.9, 0.29, 0.055, 0.1725, 2.2, 1.6],
    [0.5, 0.4, 0.1, 0.05, 0.08, 3.0, 2.5],
    [0.5, 0.4, 0.1, 0.05, 0.08, 0.01, 0.008],
    [0.4, 0.0, 0.09, 0.0, 0.0, 0.5, 0.0],
]

for p in CASES:
    saa, sab, sbb = p[2], p[3], p[4]
    ga = np.array([np.sqrt(saa), 0.0, 0.0])
    if saa > 0:
        gbx = sab / np.sqrt(saa)
        gby = np.sqrt(max(sbb - gbx * gbx, 0.0))
    else:
        gbx, gby = np.sqrt(sbb), 0.0
    va = np.vstack([[p[0]], [ga[0]], [ga[1]], [ga[2]], [0.0], [p[5]]])
    vb = np.vstack([[p[1]], [gbx], [gby], [0.0], [0.0], [p[6]]])
    rho = p[0] + p[1]
    ex = libxc.eval_xc("mgga_x_r2scan,", (va, vb), spin=1)[0][0]
    ec = libxc.eval_xc(",mgga_c_r2scan", (va, vb), spin=1)[0][0]
    print(f"({p}, {ex * rho:.12f}, {ec * rho:.12f}),")
