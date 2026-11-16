# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, analytic_force_constants, nuclear_taylor

GEOMETRY = [[0, 0, 0], [0, 0, 1.5]]
OPTIONS = dict(charge=1, xc="hf", conv_tol=1e-13, conv_tol_dp=1e-12,
               conv_tol_grad=1e-12, max_iter=500)


def test_full_block_pattern_force_constant_reports_share_native_and_ase_values():
    calc = DftRs(units="atomic", **OPTIONS)
    atoms = Atoms("HeH", positions=np.asarray(GEOMETRY) * Bohr)
    full = analytic_force_constants([2, 1], GEOMETRY, order=3, **OPTIONS)
    adapted = calc.get_analytic_force_constants(atoms, order=3)
    assert full["converged"] and adapted["converged"]
    assert full["shape"] == [6, 6, 6] and full["independent_components"] == 56
    assert full["permutation_comparisons"] == 50
    assert full["translation_comparisons"] == 3 * 3 * 6**2
    assert full["translation_residual"] < 1e-10
    np.testing.assert_allclose(full["tensor"], adapted["tensor"], atol=1e-10)
    for order, factor in [(3, 6), (4, 24)]:
        direct = analytic_force_constants([2, 1], GEOMETRY, order=order, coordinates=[2, 5], **OPTIONS)
        adapted = calc.get_analytic_force_constants(atoms, order=order, coordinates=[2, 5])
        assert direct["converged"] and adapted["converged"]
        assert direct["shape"] == [2] * order
        assert direct["units"] == adapted["units"] and "Bohr^order" in direct["units"]
        assert direct["translation_residual"] is None and direct["translation_comparisons"] == 0
        np.testing.assert_allclose(direct["tensor"], adapted["tensor"], atol=1e-10)
        if order == 3:
            ix = np.ix_([2, 5], [2, 5], [2, 5])
            np.testing.assert_allclose(np.asarray(direct["tensor"]).reshape([2]*3), np.asarray(full["tensor"]).reshape([6]*3)[ix], atol=1e-10)
        stretch = [[0, 0, 0], [0, 0, 1]]
        radial = nuclear_taylor([2, 1], GEOMETRY, stretch, order=order, **OPTIONS)
        np.testing.assert_allclose(direct["tensor"][-1], factor * radial["energy_coefficients"][order], atol=1e-10)
        pattern = analytic_force_constants([2, 1], GEOMETRY, order=order, patterns=[stretch], **OPTIONS)
        assert pattern["permutation_comparisons"] == 0 and pattern["translation_residual"] is None
        np.testing.assert_allclose(pattern["tensor"][0], direct["tensor"][-1], atol=1e-10)


def test_force_constant_request_boundaries_are_explicit():
    for request in (dict(order=2), dict(order=5), dict(coordinates=[]), dict(coordinates=[6]),
                    dict(coordinates=[2, 2]), dict(patterns=[]),
                    dict(patterns=[[[0, 0, 1]]]), dict(patterns=[GEOMETRY], coordinates=[5])):
        with pytest.raises(ValueError):
            analytic_force_constants([2, 1], GEOMETRY, **request, **OPTIONS)


def test_force_constant_cli_and_python_quartic_block_match(tmp_path):
    executable = os.environ.get("DFT_RS_CLI")
    if not executable:
        pytest.skip("DFT_RS_CLI is required for the independent CLI route")
    path = tmp_path / "force_constants.xyz"
    path.write_text(f"2\nquartic block validation\nHe 0 0 0\nH 0 0 {1.5*0.529177210903:.16f}\n")
    result = subprocess.run([executable, "force-constants", str(path), "--response-order", "4",
        "--force-coordinates", "2,5", "--charge", "1", "--xc", "hf", "--conv-tol", "1e-13",
        "--conv-tol-grad", "1e-12", "--conv-tol-dp", "1e-12", "--log-mode", "none"],
        check=True, capture_output=True, text=True)
    tensor = [float(line.split()[2]) for line in result.stdout.splitlines() if line.startswith("tensor ")]
    direct = analytic_force_constants([2, 1], GEOMETRY, order=4, coordinates=[2, 5], **OPTIONS)
    np.testing.assert_allclose(tensor, direct["tensor"], atol=1e-10)
    assert "translation_residual None" in result.stdout and "permutation_comparisons 3" in result.stdout
