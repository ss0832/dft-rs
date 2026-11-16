# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, molecular_taylor, mixed_molecular_response, static_spectroscopic_derivative

NUMBERS = [2, 1]
GEOMETRY = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]
NUCLEAR = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
ZERO = [[0.0, 0.0, 0.0]] * 2
OPTIONS = dict(charge=1, xc="hf", conv_tol=1e-13, conv_tol_grad=1e-12,
               conv_tol_dp=1e-12, max_iter=500)


def test_joint_mixed_and_spectroscopy_native_ase_parity_with_reference_field():
    atoms = Atoms("HeH", positions=np.asarray(GEOMETRY) * Bohr)
    calc = DftRs(units="atomic", **OPTIONS)
    field = [0.0, 0.0, 0.02]
    direct = molecular_taylor(NUMBERS, GEOMETRY, NUCLEAR, [0.1, -0.2, 0.3], reference_field=field, **OPTIONS)
    adapted = calc.get_molecular_taylor(atoms, nuclear_direction=NUCLEAR, electric_direction=[0.1, -0.2, 0.3], reference_field=field)
    assert direct["converged"] and adapted["converged"]
    for key in ("energy_coefficients", "dipole_coefficients", "density_coefficients", "metric_defects", "residuals"):
        np.testing.assert_allclose(direct[key], adapted[key], atol=1e-10)
    np.testing.assert_allclose(sum(np.asarray(v) for v in direct["energy_components"].values()), direct["energy_coefficients"], atol=1e-12)
    assert np.max(np.abs(direct["energy_components"]["nuclear_field"])) > 0.01
    mixed = mixed_molecular_response(NUMBERS, GEOMETRY, [NUCLEAR, ZERO], [[0, 0, 0], [0, 0, 1]], reference_field=field, **OPTIONS)
    adapted = calc.get_mixed_molecular_response(atoms, nuclear_directions=[NUCLEAR, ZERO], electric_directions=[[0, 0, 0], [0, 0, 1]], reference_field=field)
    assert mixed["converged"] and adapted["converged"]
    assert mixed["energy_derivative"] == pytest.approx(adapted["energy_derivative"], abs=1e-10)
    np.testing.assert_allclose(mixed["dipole_derivative"], adapted["dipole_derivative"], atol=1e-10)
    for rank in (1, 2, 3):
        direct = static_spectroscopic_derivative(NUMBERS, GEOMETRY, NUCLEAR, electric_order=rank, reference_field=field, **OPTIONS)
        adapted = calc.get_static_spectroscopic_derivative(atoms, nuclear_direction=NUCLEAR, electric_order=rank, reference_field=field)
        assert direct["converged"] and adapted["converged"]
        assert direct["shape"] == [3] * rank
        assert direct["units"] == adapted["units"]
        np.testing.assert_allclose(direct["tensor"], adapted["tensor"], atol=1e-10)
        if rank == 1:
            assert mixed["energy_derivative"] == pytest.approx(-direct["tensor"][2], abs=1e-10)
        if rank == 3:
            assert direct["permutation_comparisons"] == 36
            # Six distinct-axis pairs need 2^3 centered corners each;
            # three repeated-axis pairs need (1+1)*(2+1) corners each.
            # No corner is zero because the independent nuclear leg is +/-1/2.
            assert direct["directional_evaluations"] == 6 * 2**3 + 3 * (2 * 3)
            assert direct["permutation_defect"] < 1e-10


def test_molecular_boundaries_and_electric_energy_dipole_relation():
    electric = molecular_taylor(NUMBERS, GEOMETRY, ZERO, [0, 0, 0.5], **OPTIONS)
    for n in range(4):
        assert (n+1)*electric["energy_coefficients"][n+1] == pytest.approx(-0.5*electric["dipole_coefficients"][n][2], abs=1e-10)
    with pytest.raises(ValueError):
        mixed_molecular_response(NUMBERS, GEOMETRY, [NUCLEAR], [[0, 0, 1]] * 2, **OPTIONS)
    with pytest.raises(ValueError):
        molecular_taylor(NUMBERS, GEOMETRY, NUCLEAR, [0, 0, float("nan")], **OPTIONS)
    with pytest.raises(ValueError):
        static_spectroscopic_derivative(NUMBERS, GEOMETRY, NUCLEAR, electric_order=4, **OPTIONS)


def test_cli_python_all_spectroscopic_slots_and_mixed_observables(tmp_path):
    executable = os.environ.get("DFT_RS_CLI")
    if not executable:
        pytest.skip("DFT_RS_CLI is required to test the independent CLI surface")
    path = tmp_path / "mixed.xyz"
    path.write_text(f"2\nmixed route validation\nHe 0 0 0\nH 0 0 {1.5*0.529177210903:.16f}\n")
    def run(command, controls):
        return subprocess.run([executable, command, str(path), "--charge", "1", "--xc", "hf",
            "--conv-tol", "1e-13", "--conv-tol-grad", "1e-12", "--conv-tol-dp", "1e-12",
            "--max-iter", "500", "--log-mode", "none", *controls], check=True, capture_output=True, text=True).stdout
    text = run("static-spectroscopy", ["--nuclear-direction", "0,0,0;0,0,1", "--electric-order", "3"])
    tensor = [float(line.split()[2]) for line in text.splitlines() if line.startswith("tensor ")]
    direct = static_spectroscopic_derivative(NUMBERS, GEOMETRY, NUCLEAR, electric_order=3, **OPTIONS)
    np.testing.assert_allclose(tensor, direct["tensor"], atol=1e-10)
    assert len(tensor) == 27 and "permutation_comparisons 36" in text
    text = run("molecular-mixed", ["--nuclear-directions", "0,0,0;0,0,1|0,0,0;0,0,0", "--field-directions", "0,0,0;0,0,1"])
    direct = mixed_molecular_response(NUMBERS, GEOMETRY, [NUCLEAR, ZERO], [[0, 0, 0], [0, 0, 1]], **OPTIONS)
    energy = float(next(line for line in text.splitlines() if line.startswith("energy_derivative ")).split()[1])
    dipole = [float(x) for x in next(line for line in text.splitlines() if line.startswith("dipole_derivative ")).split()[1:]]
    assert energy == pytest.approx(direct["energy_derivative"], abs=1e-10)
    np.testing.assert_allclose(dipole, direct["dipole_derivative"], atol=1e-10)
