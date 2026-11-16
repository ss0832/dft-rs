# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Harmonic postprocessing on the common constrained-reference driver.

The complete Cartesian/model derivative oracles are in p311. These tests
check public routes and units, without inferring another functional class.
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs import DftRs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree


@pytest.mark.parametrize("reference", ["rohf", "ghf"])
def test_c1_harmonic_modes_on_native_ase_and_canonical_cli(reference, tmp_path):
    angstrom = np.array([[0.13, -0.21, 0.19], [1.57, 0.16, -0.08],
                         [-0.19, 1.93, 0.37], [0.42, 0.29, 2.28]]) * Bohr
    positions = angstrom / Bohr
    options = dict(reference=reference, xc="hf", basis="sto-3g", charge=1,
                   multiplicity=2, grid_level=0, grid_weight_cutoff=0.0,
                   conv_tol=1e-14, conv_tol_grad=1e-12, conv_tol_dp=1e-13,
                   response_tolerance=1e-12, max_iter=600, guess="core",
                   dispersion="none", temperature=0.0,
                   vibrational_projection="eckart")
    if reference == "ghf":
        real, imaginary = np.zeros((8, 8)), np.zeros((8, 8))
        for i in range(4):
            real[i, i], real[4+i, 4+i] = -0.013 * (i+1), 0.013 * (i+1)
            real[i, 4+i] = real[4+i, i] = 0.007 * (i+1)
            imaginary[i, 4+i], imaginary[4+i, i] = 0.009 * (4-i), -0.009 * (4-i)
        options.update(gks_operator_real=real.tolist(),
                       gks_operator_imaginary=imaginary.tolist(), spinor_xc_tolerance=1e-12)
    common = dft_rs.frequencies_report([1]*4, positions.tolist(), **options)
    assert common["converged"] and common["provenance"] == "analytic"
    assert np.shape(common["hessian"]) == (12, 12)
    assert np.shape(common["normal_modes_cartesian"]) == (6, 4, 3)
    assert len(common["frequencies_cm1"]) == 6 and common["n_projected"] == 6
    assert common["thermochemistry"] is None
    if reference == "ghf":
        assert common["ir_intensities_km_mol"] is None
    atoms = Atoms("H4", positions=angstrom)
    adapted = DftRs(units="atomic", **options).get_frequencies_report(atoms)
    for key in ("hessian", "frequencies_cm1", "normal_modes_cartesian"):
        np.testing.assert_array_equal(adapted[key], common[key])
    assert adapted["units"] == common["units"]
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for canonical route acceptance"
    xyz = tmp_path / "reference_harmonic.xyz"
    xyz.write_text("4\nC1 constrained harmonic modes\n" + "".join(
        "H " + " ".join(f"{x:.17g}" for x in row) + "\n" for row in angstrom),
        encoding="utf-8")
    command = [executable, "frequencies", str(xyz), "--format", "json"]
    for key, value in options.items():
        command += ["--" + key.replace("_", "-"),
                    json.dumps(value) if isinstance(value, list) else str(value).lower()]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "SCF" in result.stderr or "[spinor-scf]" in result.stderr
    actual = json.loads(result.stdout)
    expected = dft_rs.frequencies_report(
        [1]*4, (angstrom * (1.0 / Bohr)).tolist(), **options)
    np.testing.assert_allclose(actual["frequencies_cm1"], expected["frequencies_cm1"],
                               rtol=0, atol=1e-8)


@pytest.mark.parametrize("reference", ["rohf", "ghf"])
def test_atomic_reference_gibbs_and_electronic_degeneracy_units(reference, tmp_path):
    options = dict(reference=reference, xc="hf", basis="sto-3g", multiplicity=2,
                   conv_tol=1e-14, response_tolerance=1e-12, guess="core",
                   temperature=317.0, electronic_degeneracy=2)
    common = dft_rs.gibbs_free_energy([1], [[0.0, 0.0, 0.0]], **options)
    assert common["converged"] and common["frequencies_cm1"] == []
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atomic = DftRs(units="atomic", **options).get_gibbs_free_energy(atoms)
    physical_options = dict(options, conv_tol=options["conv_tol"] * Hartree)
    physical = DftRs(**physical_options).get_gibbs_free_energy(atoms)
    assert atomic["gibbs_free_energy"] == common["gibbs_free_energy"]
    np.testing.assert_allclose(physical["gibbs_free_energy"],
                               common["gibbs_free_energy"] * Hartree, rtol=0, atol=1e-12)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for canonical route acceptance"
    xyz = tmp_path / "reference_atomic_gibbs.xyz"
    xyz.write_text("1\nAtomic reference Gibbs energy\nH 0 0 0\n", encoding="utf-8")
    command = [executable, "gibbs-free-energy", str(xyz), "--format", "json"]
    for key, value in options.items():
        command += ["--" + key.replace("_", "-"), str(value).lower()]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    np.testing.assert_allclose(json.loads(result.stdout)["gibbs_free_energy"],
                               common["gibbs_free_energy"], rtol=0, atol=1e-12)


def test_double_hybrid_h2_keeps_full_hessian_and_residuals_on_public_routes(tmp_path):
    angstrom = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]) * Bohr
    positions = angstrom * (1.0 / Bohr)
    options = dict(reference="rks", xc="b2plyp", basis="sto-3g", grid_level=0,
                   grid_weight_cutoff=0.0, guess="core", conv_tol=1e-13,
                   response_tolerance=1e-12, temperature=0.0, dispersion="none")
    common = dft_rs.frequencies_report([1, 1], positions.tolist(), **options)
    report_directory = os.environ.get("DFT_RS_ACCEPTANCE_REPORT_DIR")
    if report_directory:
        directory = Path(report_directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "double_hybrid_h2_native.json").write_text(
            json.dumps(common, indent=2, allow_nan=False), encoding="utf-8")
    assert common["converged"] and common["provenance"] == "analytic"
    assert np.shape(common["hessian"]) == (6, 6)
    assert common["n_projected"] == 5 and len(common["frequencies_cm1"]) == 1
    assert np.any(np.abs(common["pt2_hessian"]) > 1e-10)
    np.testing.assert_allclose(
        common["hessian"],
        np.asarray(common["parent_hessian"]) + np.asarray(common["pt2_hessian"])
        + np.asarray(common["dispersion_hessian"]), rtol=0, atol=1e-12)
    for key in ("response_residuals", "amplitude_response_residuals", "projector_defects",
                "nuclear_stationarity_defects", "metric_defects", "response_number_defects",
                "orbital_frame_defects", "reference_energy_defects", "reference_fock_defects"):
        values = np.asarray(common[key])
        assert np.all(np.isfinite(values)) and np.max(np.abs(values)) <= 1e-12
    equations = common["coefficient_equations"]
    assert len(equations) == len(common["response_residuals"])
    for equation, returned in zip(equations, common["response_residuals"]):
        assert equation["linear_relative_residual"] == returned
        assert np.isfinite(equation["inner_linear_relative_residual"])
        assert equation["inner_linear_relative_residual"] <= 1e-12
        assert np.isfinite(equation["source_norm"]) and equation["source_norm"] >= 0
        assert np.isfinite(equation["reconstructed_norm"])
        assert equation["reconstructed_norm"] <= 1e-12
    adapted = DftRs(units="atomic", **options).get_frequencies_report(
        Atoms("H2", positions=angstrom))
    for key in ("hessian", "parent_hessian", "pt2_hessian", "response_residuals"):
        np.testing.assert_array_equal(adapted[key], common[key])
    np.testing.assert_allclose(adapted["frequencies_cm1"], common["frequencies_cm1"],
                               rtol=0, atol=1e-8)
    assert adapted["units"] == common["units"]
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for canonical route acceptance"
    xyz = tmp_path / "double_hybrid_h2.xyz"
    xyz.write_text("2\nDouble-hybrid harmonic contract\n" + "".join(
        "H " + " ".join(f"{x:.17g}" for x in row) + "\n" for row in angstrom),
        encoding="utf-8")
    command = [executable, "frequencies", str(xyz), "--format", "json"]
    for key, value in options.items():
        command += ["--" + key.replace("_", "-"), str(value).lower()]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "SCF" in result.stderr
    actual = json.loads(result.stdout)
    assert actual["converged"]
    for key in ("hessian", "parent_hessian", "pt2_hessian", "response_residuals"):
        np.testing.assert_allclose(actual[key], common[key], rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual["frequencies_cm1"], common["frequencies_cm1"],
                               rtol=0, atol=1e-8)
