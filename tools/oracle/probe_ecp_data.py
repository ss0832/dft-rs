# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Print PySCF's built-in LANL2DZ ECP data for Na/Cl to check numerical
# identity with our bundled BSE lanl2dz.g94 (values-only comparison).
from pyscf import gto

for el in ["Na", "Cl"]:
    ecp = gto.basis.load_ecp("lanl2dz", el)
    print(el, "ncore:", ecp[0])
    for l, chans in ecp[1]:
        print("  l =", l)
        for power, terms in enumerate(chans):
            for e, c in terms:
                print(f"    r^{power - 2}: exp {e}  coef {c}")
