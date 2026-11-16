# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Does the reference freeze exactly at tau = tau_W (PySCF/libxc-side clamp)?
import numpy as np
from pyscf.dft import libxc

rho_s, sig_s = 0.5, 0.1
tau_w = sig_s / (8 * rho_s)  # per-spin von Weizsaecker = 0.025
for tau_s in [0.03, 0.026, 0.0251, 0.025, 0.0249, 0.024, 0.02, 0.01, 0.001]:
    va = np.vstack([[rho_s], [np.sqrt(sig_s)], [0.0], [0.0], [0.0], [tau_s]])
    ref = libxc.eval_xc("mgga_x_r2scan,", (va, va), spin=1)[0][0]
    print(f"tau={tau_s:7}: exc={ref:.14f}")
print(f"tau_w = {tau_w}")
