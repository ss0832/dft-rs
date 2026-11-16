# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
from pyscf.dft import libxc

print("revm06 hyb:", libxc.hybrid_coeff("hyb_mgga_x_revm06,mgga_c_revm06"))
print("m06sx rsh:", libxc.rsh_coeff("hyb_mgga_x_m06_sx,mgga_c_m06_sx"))
print("hse06 rsh:", libxc.rsh_coeff("hse06"))
