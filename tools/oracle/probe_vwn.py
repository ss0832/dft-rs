# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Throwaway probe: libxc VWN variants at spin-polarized densities.
import numpy as np
from pyscf.dft import libxc

ra, rb = 0.3, 0.1
rho = np.array([[ra], [rb]])
for name in ["lda_c_vwn", "lda_c_vwn_rpa", "lda_c_vwn_3", "lda_c_vwn_4"]:
    try:
        exc, vxc = libxc.eval_xc(name, rho, spin=1)[:2]
        # exc is energy per particle; energy density = exc * (ra+rb)
        print(f"{name:16s} e = {exc[0] * (ra + rb):.12f}")
    except Exception as e:
        print(f"{name}: {e}")
