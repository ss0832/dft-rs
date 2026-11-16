# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Unit bridges of the atomic-unit interface.

The native layer and every driver report use atomic units (Hartree, Bohr).
These are the conversion factors of the Rust core (``dft_rs::constants``,
CODATA 2018), so that a caller converting at its own boundary uses the same
numbers as the XYZ reader, ASE adapter and reports. The adapter uses these
factors independently of the installed ASE release's default CODATA edition.
``X_IN_Y`` is the value of one X
expressed in Y.
"""

BOHR_IN_ANGSTROM = 0.529177210903
ANGSTROM_IN_BOHR = 1.0 / BOHR_IN_ANGSTROM
BOHR_IN_METER = 0.529177210903e-10
HARTREE_IN_EV = 27.211386245988
HARTREE_IN_KCAL_MOL = 627.509474063
HARTREE_IN_KJ_MOL = 2625.4996394799
HARTREE_IN_CM1 = 219474.63136320
HARTREE_IN_JOULE = 4.3597447222071e-18
HARTREE_PER_BOHR3_IN_GPA = HARTREE_IN_JOULE / (BOHR_IN_METER * BOHR_IN_METER * BOHR_IN_METER) * 1e-9
AU_DIPOLE_IN_DEBYE = 2.541746473

__all__ = [
    "BOHR_IN_ANGSTROM",
    "ANGSTROM_IN_BOHR",
    "BOHR_IN_METER",
    "HARTREE_IN_EV",
    "HARTREE_IN_KCAL_MOL",
    "HARTREE_IN_KJ_MOL",
    "HARTREE_IN_CM1",
    "HARTREE_IN_JOULE",
    "HARTREE_PER_BOHR3_IN_GPA",
    "AU_DIPOLE_IN_DEBYE",
]
