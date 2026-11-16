# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Pointwise reference values for the M06-family pinned tests.
import numpy as np
from pyscf.dft import libxc

ra, rb = 0.3, 0.1
ga = [0.1, 0.1, 0.15]
gb = [0.08, 0.05, 0.12]
ta, tb = 0.35, 0.12  # physical (above vW bound)
rho_a = np.array([[ra], [ga[0]], [ga[1]], [ga[2]], [0.0], [ta]])
rho_b = np.array([[rb], [gb[0]], [gb[1]], [gb[2]], [0.0], [tb]])
print("saa,sab,sbb:", np.dot(ga, ga), np.dot(ga, gb), np.dot(gb, gb))
for name in ["mgga_x_revm06_l,mgga_c_revm06_l", "hyb_mgga_x_revm06,mgga_c_revm06"]:
    exc = libxc.eval_xc(name, (rho_a, rho_b), spin=1)[0][0] * (ra + rb)
    print(name, format(exc, ".12f"))
