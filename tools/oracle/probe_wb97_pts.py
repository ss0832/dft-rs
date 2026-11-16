# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Pointwise reference values for the wB97 family pinned tests.
import numpy as np
from pyscf.dft import libxc

ra, rb = 0.3, 0.1
rho_a = np.array([[ra], [0.1], [0.1], [0.15]])
rho_b = np.array([[rb], [0.08], [0.05], [0.12]])
for name in ["wb97x_d", "wb97x_v", "wb97", "wb97x"]:
    exc = libxc.eval_xc(name, (rho_a, rho_b), spin=1)[0][0] * (ra + rb)
    print(name, format(exc, ".12f"))
