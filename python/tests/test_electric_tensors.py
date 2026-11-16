# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, beta, gamma, static_electric_tensor, electric_field_taylor

GEOMETRY = [[0, 0, 0], [1.75 / 3, 3.5 / 3, 3.5 / 3]]
OPTIONS = dict(charge=1, xc="hf", conv_tol=1e-14, conv_tol_dp=1e-13,
               conv_tol_grad=1e-12, max_iter=250)


@pytest.mark.parametrize("order,entry,method", [(2, beta, "get_beta"), (3, gamma, "get_gamma")])
def test_every_raw_tensor_entry_and_invariant_agree_on_native_and_ase(order, entry, method):
    direct = entry([2, 1], GEOMETRY, **OPTIONS)
    atoms = Atoms("HeH", positions=np.asarray(GEOMETRY) * Bohr)
    calc = DftRs(units="atomic", **OPTIONS)
    adapted = getattr(calc, method)(atoms)
    assert direct["converged"] and adapted["converged"]
    assert direct["shape"] == [3] * (order + 1)
    assert len(direct["tensor"]) == 3 ** (order + 1)
    assert direct["units"] == adapted["units"]
    assert "actual derivatives" in direct["axis_convention"]
    assert direct["ordered_field_evaluations"] == 3 ** order
    assert direct["directional_evaluations"] == (24 if order == 2 else 141)
    assert direct["intrinsic_permutation_comparisons"] == (9 if order == 2 else 81)
    assert direct["energy_permutation_comparisons"] == (18 if order == 2 else 81)
    for key in ("tensor", "accumulation_roundoff", "residuals", "frobenius_norm",
                "projector_defect", "stationarity_defect", "particle_number_defect",
                "intrinsic_permutation_defect", "energy_permutation_defect"):
        np.testing.assert_allclose(direct[key], adapted[key], atol=1e-9)
    tensor = np.asarray(direct["tensor"]).reshape(direct["shape"])
    for axis in range(1, order + 1):
        np.testing.assert_allclose(tensor, tensor.swapaxes(0, axis), atol=1e-8)
    direction = np.asarray([0.2, -0.3, 0.4])
    contracted = tensor.copy()
    for _ in range(order):
        contracted = np.tensordot(contracted, direction, axes=(-1, 0))
    series = electric_field_taylor([2, 1], GEOMETRY, direction, order=order, **OPTIONS)
    np.testing.assert_allclose(contracted, np.asarray(series["dipole_coefficients"][order]) * (2 if order == 2 else 6), atol=1e-8)
    if order == 2:
        np.testing.assert_allclose(direct["beta_vector"], adapted["beta_vector"], atol=1e-9)
        assert direct["gamma_isotropic"] is None
        assert np.isclose(direct["beta_vector_norm"], np.linalg.norm(direct["beta_vector"]))
    else:
        assert direct["beta_vector"] is None and direct["beta_vector_norm"] is None
        assert np.isclose(direct["gamma_isotropic"], adapted["gamma_isotropic"])


def test_biased_field_and_invalid_requests_use_the_shared_driver():
    direct = beta([2, 1], GEOMETRY, reference_field=[0, 0, 0.03], **OPTIONS)
    adapted = DftRs(units="atomic", **OPTIONS).get_beta(Atoms("HeH", positions=np.asarray(GEOMETRY) * Bohr), reference_field=[0, 0, 0.03])
    np.testing.assert_allclose(direct["tensor"], adapted["tensor"], atol=1e-8)
    assert direct["reference_field"] == [0, 0, 0.03]
    for order in (0, 1, 4):
        with pytest.raises(ValueError):
            static_electric_tensor([2, 1], GEOMETRY, order=order, **OPTIONS)
    with pytest.raises(ValueError):
        gamma([2, 1], GEOMETRY, response_max_iter=0, **OPTIONS)
    with pytest.raises(ValueError):
        beta([2, 1], GEOMETRY, reference_field=[0, 0, float("nan")], **OPTIONS)
    with pytest.raises(ValueError):
        beta([2, 1], GEOMETRY, assembly="unknown", **OPTIONS)
    with pytest.raises(ValueError):
        gamma([2, 1], GEOMETRY, assembly="variational", **OPTIONS)


def test_variational_beta_uses_three_first_order_responses_on_every_surface(tmp_path):
    direct = beta([2, 1], GEOMETRY, assembly="variational", **OPTIONS)
    projector = beta([2, 1], GEOMETRY, **OPTIONS)
    adapted = DftRs(units="atomic", **OPTIONS).get_beta(Atoms("HeH", positions=np.asarray(GEOMETRY) * Bohr), assembly="variational")
    assert direct["converged"] and adapted["converged"]
    assert "2n+1" in direct["model"]
    assert len(direct["residuals"]) == 3
    np.testing.assert_allclose(direct["tensor"], projector["tensor"], atol=1e-9)
    np.testing.assert_allclose(direct["tensor"], adapted["tensor"], atol=1e-9)
    executable = os.environ.get("DFT_RS_CLI")
    if not executable:
        pytest.skip("DFT_RS_CLI is required for the independent CLI route")
    path = tmp_path / "variational_beta.xyz"
    path.write_text("2\nvariational beta validation\nHe 0 0 0\nH " + " ".join(f"{v * 0.529177210903:.16f}" for v in GEOMETRY[1]) + "\n")
    cli = subprocess.run([executable, "beta", str(path), "--electric-assembly", "variational",
        "--charge", "1", "--xc", "hf", "--conv-tol", "1e-14", "--conv-tol-dp", "1e-13",
        "--conv-tol-grad", "1e-12", "--log-mode", "none"], check=True, capture_output=True, text=True)
    tensor = [float(line.split()[-1]) for line in cli.stdout.splitlines() if line.startswith("beta ")]
    np.testing.assert_allclose(tensor, direct["tensor"], atol=1e-9)
    assert "2n+1" in cli.stdout and "converged true" in cli.stdout


@pytest.mark.parametrize("order,entry,command", [(2, beta, "beta"), (3, gamma, "gamma")])
def test_cli_matches_native_full_tensors_and_diagnostics(tmp_path, order, entry, command):
    executable = os.environ.get("DFT_RS_CLI")
    if not executable:
        pytest.skip("DFT_RS_CLI is required for the independent CLI route")
    path = tmp_path / "electric_tensor.xyz"
    path.write_text("2\nelectric tensor route validation\nHe 0 0 0\nH " + " ".join(f"{v * 0.529177210903:.16f}" for v in GEOMETRY[1]) + "\n")
    result = subprocess.run([executable, command, str(path), "--charge", "1", "--xc", "hf",
        "--conv-tol", "1e-14", "--conv-tol-grad", "1e-12", "--conv-tol-dp", "1e-13",
        "--max-iter", "250", "--log-mode", "none"], check=True, capture_output=True, text=True)
    direct = entry([2, 1], GEOMETRY, **OPTIONS)
    values = [float(line.split()[-1]) for line in result.stdout.splitlines() if line.startswith(command + " ")]
    np.testing.assert_allclose(values, direct["tensor"], atol=2e-8)
    assert "converged true" in result.stdout
    assert direct["units"] in result.stdout
    assert "raw components preserved" in result.stdout
    for key in ("frobenius_norm", "intrinsic_permutation_defect", "energy_permutation_defect"):
        row = next(line for line in result.stdout.splitlines() if line.startswith(key + " "))
        assert np.isclose(float(row.split()[1]), direct[key], atol=1e-9)
