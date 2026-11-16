# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Reference point for the M06-SX semilocal part (exchange + correlation)
# at the family's pinned spin-polarized meta-GGA point.
import numpy as np
from pyscf.dft import libxc

# [ra, rb, saa, sab, sbb, ta, tb]
p = [0.3, 0.1, 0.0425, 0.031, 0.0233, 0.35, 0.12]
saa, sab, sbb = p[2], p[3], p[4]
ga = np.array([np.sqrt(saa), 0.0, 0.0])
gbx = sab / np.sqrt(saa)
gby = np.sqrt(max(sbb - gbx * gbx, 0.0))
va = np.vstack([[p[0]], [ga[0]], [ga[1]], [ga[2]], [0.0], [p[5]]])
vb = np.vstack([[p[1]], [gbx], [gby], [0.0], [0.0], [p[6]]])
exc = libxc.eval_xc("hyb_mgga_x_m06_sx,mgga_c_m06_sx", (va, vb), spin=1)[0][0]
print("%.12f" % (exc * (p[0] + p[1])))
