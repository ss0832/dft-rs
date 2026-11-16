# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Shared molecular reference selection on every polarizability surface."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr

import dft_rs
from dft_rs import DftRs
from dft_rs.units import BOHR_IN_ANGSTROM

AMMONIA = np.array([[0.17, -0.11, 0.23], [1.77, 0.18, -0.53],
                    [-0.31, 1.74, -0.45], [-1.20, -1.31, -0.61]])
HYDROGEN = np.array([[0.13, -0.21, 0.19], [1.57, 0.16, -0.08],
                     [-0.19, 1.93, 0.37], [0.42, 0.29, 2.28]])
OPTIONS = dict(basis="sto-3g", grid_level=0, grid_weight_cutoff=0.0,
               conv_tol=1e-13, conv_tol_grad=1e-12, conv_tol_dp=1e-12,
               response_tol=1e-11, max_iter=300)


def case(reference):
    if reference in {"rohf", "uhf"}:
        return [7, 1, 1, 1], AMMONIA, dict(charge=1, multiplicity=2)
    if reference == "ghf":
        real = np.zeros((8, 8))
        imag = np.zeros((8, 8))
        for i in range(4):
            real[i, i], real[i + 4, i + 4] = -0.013 * (i + 1), 0.013 * (i + 1)
            real[i, i + 4] = real[i + 4, i] = 0.007 * (i + 1)
            imag[i, i + 4], imag[i + 4, i] = 0.009 * (4 - i), -0.009 * (4 - i)
        return [1] * 4, HYDROGEN, dict(gks_operator_real=real.tolist(),
                                     gks_operator_imaginary=imag.tolist())
    return [1] * 4, HYDROGEN, {}


@pytest.mark.parametrize("reference", ["rhf", "uhf", "rohf", "ghf"])
def test_reference_frequency_response_native_ase_and_common_driver(reference):
    numbers, positions, extra = case(reference)
    options = dict(OPTIONS, reference=reference, **extra)
    report = dft_rs.dynamic_polarizability(numbers, positions.tolist(), omega=0.11,
                                         damping=0.02, **options)
    common = dft_rs.run_feature("dynamic-polarizability", numbers, positions.tolist(),
                               frequency=0.11, frequency_damping=0.02, **options)
    atoms = Atoms(numbers=numbers, positions=positions * Bohr)
    calc = DftRs(units="atomic", unrestricted=reference == "uhf", **{k: v for k, v in OPTIONS.items()
                                   if k not in {"response_tol", "grid_weight_cutoff"}})
    adapted = calc.get_dynamic_polarizability(atoms, omega=0.11, damping=0.02,
                                             reference=reference, grid_weight_cutoff=0.0,
                                             **extra)
    assert report["converged"] and report["provenance"] == "analytic"
    assert max(report["residuals"]) <= 1e-10
    for key in ["alpha_real", "alpha_imag"]:
        assert np.asarray(report[key]).shape == (3, 3)
        np.testing.assert_allclose(report[key], common[key], rtol=0, atol=1e-10)
        np.testing.assert_allclose(report[key], adapted[key], rtol=0, atol=1e-9)
        assert report["unit_labels"][key] == "e^2*bohr^2/hartree"
    assert report["frequency_real"] == 0.11 and report["frequency_imag"] == 0.02


@pytest.mark.parametrize("reference", ["uhf", "rohf", "ghf"])
def test_canonical_cli_reference_response_and_default_scf_progress(reference, tmp_path):
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for public route acceptance"
    numbers, positions, extra = case(reference)
    symbols = ["N" if z == 7 else "H" for z in numbers]
    xyz = tmp_path / "reference_alpha.xyz"
    xyz.write_text("4\ncomplex and common-orbital response\n" + "".join(
        symbol + " " + " ".join(f"{x:.17g}" for x in point) + "\n"
        for symbol, point in zip(symbols, positions * BOHR_IN_ANGSTROM)), encoding="utf-8")
    controls = dict(OPTIONS, reference=reference, frequency=0.11, frequency_damping=0.02,
                    **extra)
    args = [executable, "dynamic-polarizability", str(xyz), "--format", "json"]
    for key, value in controls.items():
        args += ["--" + key.replace("_", "-"), json.dumps(value) if not isinstance(value, str)
                 else value]
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "iter" in result.stderr and "residual" in result.stderr
    report = json.loads(result.stdout)
    native = dft_rs.run_feature("dynamic-polarizability", numbers, positions.tolist(), **controls)
    for key in ["alpha_real", "alpha_imag", "residuals"]:
        np.testing.assert_allclose(report[key], native[key], rtol=0, atol=1e-9)


def test_complex_static_alpha_uses_the_same_generalized_reference():
    numbers, positions, extra = case("ghf")
    options = dict(OPTIONS, reference="ghf", **extra)
    static = dft_rs.polarizability_report(numbers, positions.tolist(), **options)
    zero = dft_rs.dynamic_polarizability(numbers, positions.tolist(), **options)
    np.testing.assert_allclose(static["polarizability"], zero["alpha_real"], rtol=0, atol=1e-10)
    assert max(zero["residuals"]) < 1e-10


def test_static_correlated_response_uses_the_same_standard_dynamic_entry():
    positions = HYDROGEN.tolist() + [[1.41, 1.83, 1.59]]
    options = dict(OPTIONS, reference="uks", multiplicity=2,
                   pt2_same_spin=0.2, pt2_opposite_spin=0.3,
                   double_hybrid_target_xc="pbe")
    static = dft_rs.polarizability_report([1] * 5, positions, **options)
    zero = dft_rs.dynamic_polarizability([1] * 5, positions, **options)
    np.testing.assert_allclose(static["polarizability"], zero["alpha_real"], rtol=0, atol=1e-10)
    for key in ["pt2_same_spin_alpha_polarizability", "pt2_same_spin_beta_polarizability",
                "pt2_opposite_spin_polarizability"]:
        np.testing.assert_allclose(static[key], zero[key], rtol=0, atol=1e-10)
    assert zero["frequency_real"] == zero["frequency_imag"] == 0.0


def test_native_response_control_aliases_do_not_conflict_with_adapter_defaults():
    options = {key: value for key, value in OPTIONS.items() if key != "response_tol"}
    report = dft_rs.dynamic_polarizability([1] * 4, HYDROGEN.tolist(),
                                         response_tolerance=1e-11,
                                         response_max_iterations=500, **options)
    assert max(report["residuals"]) < 1e-10
    with pytest.raises(ValueError, match="use either"):
        dft_rs.dynamic_polarizability([1] * 4, HYDROGEN.tolist(),
                                     response_tol=1e-11, response_tolerance=1e-12)
