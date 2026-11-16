# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Pinned pointwise references for TPSS (X, C, XC) at several points.
import numpy as np
from pyscf.dft import libxc

POINTS = [
    # (ra, rb, ga, gb, ta, tb)
    (0.3, 0.1, (0.1, 0.1, 0.15), (0.08, 0.05, 0.12), 0.35, 0.12),
    (1.2, 0.9, (0.3, -0.2, 0.4), (0.25, 0.1, -0.3), 1.9, 1.4),
    (0.05, 0.4, (0.02, 0.0, 0.05), (0.3, 0.2, -0.1), 0.06, 0.9),
]
for (ra, rb, ga, gb, ta, tb) in POINTS:
    rho_a = np.array([[ra], [ga[0]], [ga[1]], [ga[2]], [0.0], [ta]])
    rho_b = np.array([[rb], [gb[0]], [gb[1]], [gb[2]], [0.0], [tb]])
    ex = libxc.eval_xc("mgga_x_tpss,", (rho_a, rho_b), spin=1)[0][0] * (ra + rb)
    ec = libxc.eval_xc(",mgga_c_tpss", (rho_a, rho_b), spin=1)[0][0] * (ra + rb)
    print(f"({ra},{rb}): X {ex:.12f}  C {ec:.12f}")
