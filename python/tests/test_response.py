# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

import math
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, dynamic_polarizability, electric_field_taylor, mixed_electric_field_response

H2 = [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]
OPTIONS = dict(conv_tol=1e-13, conv_tol_grad=1e-10, conv_tol_dp=1e-12)


def test_python_and_ase_use_the_same_response_driver_and_units():
    atoms = Atoms("H2", positions=np.asarray(H2)*Bohr)
    calc = DftRs(units="atomic", **OPTIONS)
    native = dynamic_polarizability([1,1], H2, omega=0.2, damping=0.03, **OPTIONS)
    adapted = calc.get_dynamic_polarizability(atoms, omega=0.2, damping=0.03)
    for key in ("alpha_real", "alpha_imag", "residuals"):
        np.testing.assert_allclose(native[key], adapted[key], atol=1e-10)
    assert native["units"] == adapted["units"] == "bohr^3"
    assert native["frequency_units"] == "hartree"
    assert native["converged"]
    assert max(native["residuals"]) <= 1e-10
    assert native["alpha_imag"][2][2] > 0

    native = electric_field_taylor([1,1], H2, (0,0,1), order=6, **OPTIONS)
    adapted = calc.get_electric_field_taylor(atoms, order=6)
    np.testing.assert_allclose(native["dipole_coefficients"], adapted["dipole_coefficients"], atol=1e-9)
    assert max(native["projector_defects"]) < 1e-8
    directions = [(0,0,1)]*3
    mixed = mixed_electric_field_response([1,1], H2, directions, **OPTIONS)
    adapted = calc.get_mixed_electric_field_response(atoms, directions=directions)
    np.testing.assert_allclose(mixed["dipole_derivative"], adapted["dipole_derivative"], atol=1e-9)
    np.testing.assert_allclose(mixed["dipole_derivative"], np.asarray(native["dipole_coefficients"][3])*6, atol=1e-9)
    # Centered cubic polarization uses -3d/2, -d/2, d/2 and 3d/2.
    # All four paths are nonzero, including for repeated directions.
    assert mixed["directional_evaluations"] == adapted["directional_evaluations"] == 4
    assert all(math.isfinite(x) for x in mixed["accumulation_roundoff"])


def test_response_boundary_inputs_and_frequency_reversal():
    with pytest.raises(ValueError):
        dynamic_polarizability([1,1], H2, damping=-0.1)
    with pytest.raises(ValueError):
        dynamic_polarizability([1,1], H2, imaginary=True, omega=-0.1)
    with pytest.raises(ValueError):
        electric_field_taylor([1,1], H2, (0,0,float("nan")))
    with pytest.raises(ValueError):
        mixed_electric_field_response([1,1], H2, [(0,0,float("inf"))])
    positive = dynamic_polarizability([1,1], H2, omega=0.2, damping=0.01, **OPTIONS)
    negative = dynamic_polarizability([1,1], H2, omega=-0.2, damping=0.01, **OPTIONS)
    np.testing.assert_allclose(positive["alpha_real"], negative["alpha_real"], atol=1e-10)
    np.testing.assert_allclose(positive["alpha_imag"], -np.asarray(negative["alpha_imag"]), atol=1e-10)
