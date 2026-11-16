# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Throwaway probe: libxc RSH conventions (coefficients + SR-B88 pointwise).
import numpy as np
from pyscf.dft import libxc

for name in ["camb3lyp", "lc-blyp"]:
    try:
        print(name, "rsh_coeff (omega, alpha, beta) =", libxc.rsh_coeff(name))
        print(name, "  -> is (omega, hyb_at_sr..)? hybrid_coeff:", libxc.hybrid_coeff(name))
    except Exception as e:
        print(name, "error:", e)

# Pointwise SR-B88 (ITYH) with omega=0.33 against our formula.
ra, rb = 0.3, 0.1
saa, sab, sbb = 0.05, 0.02, 0.04
rho = np.array(
    [
        [ra, 0.1, 0.1, 0.15],  # rho_a, grad_ax, ...
        [rb, 0.08, 0.05, 0.12],
    ]
)
# PySCF eval_xc for GGA spin case wants rho = ((den_a, gx, gy, gz), (den_b,...)).
rho_a = np.array([[ra], [0.1], [0.1], [0.15]])
rho_b = np.array([[rb], [0.08], [0.05], [0.12]])
saa = (0.1**2 + 0.1**2 + 0.15**2)
sab = 0.1 * 0.08 + 0.1 * 0.05 + 0.15 * 0.12
sbb = 0.08**2 + 0.05**2 + 0.12**2
print("sigma:", saa, sab, sbb)
for name in ["ityh", "gga_x_ityh"]:
    try:
        exc = libxc.eval_xc(f"{name}", (rho_a, rho_b), spin=1, omega=0.33)[0]
        print(f"{name}: e = {exc[0] * (ra + rb):.12f}")
    except Exception as e:
        print(f"{name}: {e}")
