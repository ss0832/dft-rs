# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Pinned pointwise references for B97M-V and revPBE.
import numpy as np
from pyscf.dft import libxc

ra, rb = 0.3, 0.1
rho_a = np.array([[ra], [0.1], [0.1], [0.15], [0.0], [0.35]])
rho_b = np.array([[rb], [0.08], [0.05], [0.12], [0.0], [0.12]])
e = libxc.eval_xc("b97m_v", (rho_a, rho_b), spin=1)[0][0] * (ra + rb)
print("b97m_v pinned:", format(e, ".12f"))
e2 = libxc.eval_xc("gga_x_pbe_r,gga_c_pbe", (rho_a[:4], rho_b[:4]), spin=1)[0][0] * (ra + rb)
print("revpbe pinned:", format(e2, ".12f"))
