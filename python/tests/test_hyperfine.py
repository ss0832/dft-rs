# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Hyperfine couplings agree on native Python and ASE."""
import numpy as np
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr

import dft_rs
from dft_rs import DftRs

METHYL_NUMBERS = [6, 1, 1, 1]
METHYL = [[0.0, 0.0, 0.0], [2.04, 0.0, 0.0], [-1.02, 1.76669, 0.0], [-1.02, -1.76669, 0.0]]
NUCLEAR_G = [1.4048236, 5.5856946893, 5.5856946893, 5.5856946893]
CONTROLS = dict(basis="sto-3g", multiplicity=2, unrestricted=True, conv_tol=1e-12,
                conv_tol_dp=1e-10, max_iter=300, nuclear_g=NUCLEAR_G)


def methyl_atoms():
    return Atoms("CH3", positions=np.asarray(METHYL) * Bohr)


def calculator():
    return DftRs(units="atomic", basis="sto-3g", conv_tol=1e-12, conv_tol_dp=1e-10, max_iter=300)


def test_hyperfine_couplings_agree_on_native_and_ase():
    native = dft_rs.hyperfine(METHYL_NUMBERS, METHYL, **CONTROLS)
    adapted = calculator().get_hyperfine(methyl_atoms(), **CONTROLS, units="atomic")
    assert native["feature"] == "hyperfine" and native["converged"]
    np.testing.assert_allclose(adapted["isotropic_coupling_mhz"],
                               native["isotropic_coupling_mhz"], rtol=1e-6)
    hydrogen = np.asarray(native["isotropic_coupling_mhz"])[1:]
    # The three hydrogens are equivalent up to the orientation of the
    # quadrature grid: measured spread 1.7e-6 relative (-152.27801,
    # -152.277756, -152.277756 MHz per unit nuclear g factor).
    np.testing.assert_allclose(hydrogen, hydrogen[0], rtol=1e-5)
