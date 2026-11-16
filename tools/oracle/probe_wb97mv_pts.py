# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# wB97M-V: RSH metadata + pointwise reference for the pinned Rust test.
import numpy as np
from pyscf.dft import libxc

print("rsh (omega, alpha, beta):", libxc.rsh_coeff("wb97m_v"))
print("is_nlc:", libxc.is_nlc("wb97m_v"))
ra, rb = 0.3, 0.1
rho_a = np.array([[ra], [0.1], [0.1], [0.15], [0.0], [0.35]])
rho_b = np.array([[rb], [0.08], [0.05], [0.12], [0.0], [0.12]])
exc = libxc.eval_xc("wb97m_v", (rho_a, rho_b), spin=1)[0][0] * (ra + rb)
print("pinned exc:", format(exc, ".12f"))
