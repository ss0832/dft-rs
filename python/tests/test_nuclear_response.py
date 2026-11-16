# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, nuclear_taylor, mixed_nuclear_response

GEOMETRY = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]
DIRECTION = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
OPTIONS = dict(charge=1, xc="hf", conv_tol=1e-13, conv_tol_dp=1e-12,
               conv_tol_grad=1e-12, max_iter=500)


def test_native_and_ase_share_cubic_quartic_coefficients_and_moving_metric():
    atoms = Atoms("HeH", positions=np.asarray(GEOMETRY)*Bohr)
    calc = DftRs(units="atomic", **OPTIONS)
    direct = nuclear_taylor([2, 1], GEOMETRY, DIRECTION, **OPTIONS)
    adapted = calc.get_nuclear_taylor(atoms, direction=DIRECTION)
    assert direct["converged"] and adapted["converged"]
    assert direct["shape"] == [5]
    assert direct["units"] == adapted["units"]
    assert "Hartree" in direct["units"] and "n!" in direct["units"]
    for key in ("energy_coefficients", "density_coefficients", "overlap_coefficients",
                "metric_defects", "particle_number_coefficients", "residuals"):
        np.testing.assert_allclose(direct[key], adapted[key], atol=1e-10)
    component_sum = sum(np.asarray(v) for v in direct["energy_components"].values())
    np.testing.assert_allclose(component_sum, direct["energy_coefficients"], atol=1e-13)
    p = np.asarray(direct["density_coefficients"])
    s = np.asarray(direct["overlap_coefficients"])
    for n in range(1, 5):
        number = sum(np.einsum("sij,ij->s", p[k], s[n-k]) for k in range(n+1))
        np.testing.assert_allclose(number, direct["particle_number_coefficients"][n-1], atol=1e-13)
        np.testing.assert_allclose(number, 0, atol=1e-10)
    for n, factor in [(3, 6), (4, 24)]:
        directions = [DIRECTION]*n
        mixed = mixed_nuclear_response([2, 1], GEOMETRY, directions, **OPTIONS)
        adapted = calc.get_mixed_nuclear_response(atoms, directions=directions)
        assert mixed["converged"] and adapted["converged"]
        assert mixed["energy_derivative"] == pytest.approx(factor*direct["energy_coefficients"][n], abs=1e-9)
        assert adapted["energy_derivative"] == pytest.approx(mixed["energy_derivative"], abs=1e-9)


def test_python_nuclear_boundary_validation_and_model_refusals():
    for direction in ([[0, 0, 1]], [[0, 0, 0], [0, 0, float("nan")]]):
        with pytest.raises(ValueError):
            nuclear_taylor([2, 1], GEOMETRY, direction, **OPTIONS)
    with pytest.raises(ValueError):
        nuclear_taylor([2, 1], GEOMETRY, DIRECTION, response_tol=0, **OPTIONS)
    with pytest.raises(ValueError):
        nuclear_taylor([2, 1], GEOMETRY, DIRECTION, dispersion="d3bj", **OPTIONS)
    with pytest.raises(ValueError):
        mixed_nuclear_response([2, 1], GEOMETRY, [], **OPTIONS)
    with pytest.raises(TypeError):
        mixed_nuclear_response([2, 1], GEOMETRY, [DIRECTION]*3, order=3, **OPTIONS)


def test_cli_and_python_quartic_reports_match(tmp_path):
    executable = os.environ.get("DFT_RS_CLI")
    if not executable:
        pytest.skip("DFT_RS_CLI is required for the independent CLI route")
    path = tmp_path / "nuclear.xyz"
    path.write_text(f"2\nquartic route validation\nHe 0 0 0\nH 0 0 {1.5*0.529177210903:.16f}\n")
    result = subprocess.run([executable, "nuclear-taylor", str(path), "--charge", "1", "--xc", "hf",
                             "--nuclear-direction", "0,0,0;0,0,1", "--conv-tol", "1e-13",
                             "--conv-tol-grad", "1e-12", "--conv-tol-dp", "1e-12", "--log-mode", "none"],
                            check=True, capture_output=True, text=True)
    report = nuclear_taylor([2, 1], GEOMETRY, DIRECTION, **OPTIONS)
    rows = [line.split() for line in result.stdout.splitlines() if line.startswith("energy_coefficient ")]
    np.testing.assert_allclose([float(row[2]) for row in rows], report["energy_coefficients"], atol=1e-10)
    assert "Converged: true" in result.stdout and "particle_number_coefficients" in result.stdout
