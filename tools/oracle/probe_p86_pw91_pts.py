# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Pinned pointwise references for P86 and PW91 correlation.
import numpy as np
from pyscf.dft import libxc

POINTS = [
    (0.3, 0.1, (0.1, 0.1, 0.15), (0.08, 0.05, 0.12)),
    (1.2, 0.9, (0.3, -0.2, 0.4), (0.25, 0.1, -0.3)),
    (0.02, 0.05, (0.01, 0.0, 0.02), (0.02, 0.01, -0.01)),
]
for (ra, rb, ga, gb) in POINTS:
    rho_a = np.array([[ra], [ga[0]], [ga[1]], [ga[2]]])
    rho_b = np.array([[rb], [gb[0]], [gb[1]], [gb[2]]])
    p86 = libxc.eval_xc(",gga_c_p86", (rho_a, rho_b), spin=1)[0][0] * (ra + rb)
    pw91 = libxc.eval_xc(",gga_c_pw91", (rho_a, rho_b), spin=1)[0][0] * (ra + rb)
    print(f"({ra},{rb}): P86 {p86:.12f}  PW91 {pw91:.12f}")
