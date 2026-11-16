# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# One extra wpbeh point at s ~ 0.05 (EG polynomial branch, nonzero s).
import numpy as np
from pyscf.dft import libxc

va = np.array([[3.0], [1.69], [0.0], [0.0]])
vb = np.array([[3.0], [0.0], [1.69], [0.0]])
rho = 6.0
w0 = libxc.eval_xc("gga_x_wpbeh,", (va, vb), spin=1)[0][0]
ec = libxc.eval_xc(",gga_c_pbe", (va, vb), spin=1)[0][0]
hse = libxc.eval_xc("HSE06", (va, vb), spin=1)[0][0]
lc = libxc.eval_xc("HYB_GGA_XC_LC_WPBE", (va, vb), spin=1)[0][0]
print("%.12f %.12f %.12f" % (w0 * rho, (w0 + ec - hse) / 0.25 * rho, (lc - ec) * rho))
