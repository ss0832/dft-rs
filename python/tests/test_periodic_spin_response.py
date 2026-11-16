# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Actual longitudinal spin-only response through native, ASE and canonical CLI."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms

from dft_rs import DftRs, periodic_spin_response
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree

POSITIONS = np.array([[.1, -.2, .3], [1.35, .1, .2], [-.3, 3.2, .9], [.2, 3.4, 2.2]])
CELL = np.array([[8.7, .31, .13], [.2, 9.2, .43], [.11, .27, 9.7]])
Q = [.043, .017, -.021]


def fixture(tmp_path):
    basis = tmp_path / "spin-primitive.g94"
    basis.write_text("H 0\nS 1 1.0\n0.91 1.0\n****\n")
    controls = dict(basis=str(basis), overlap_cutoff=4.5, grid_level=0, grid_shell=1,
                    max_iter=250, conv_tol=1e-12, conv_tol_dp=1e-11, conv_tol_grad=1e-10,
                    guess="core", grid_weight_cutoff=0.)
    atoms = Atoms("HHHH", positions=POSITIONS * Bohr, cell=CELL * Bohr, pbc=True)
    return atoms, controls


def assert_same_response(a, b):
    for key in ("q", "frequency", "spin_probe_field", "spin_g_factor", "spin_numbers",
                "total_number", "magnetic_moment", "spin_susceptibility",
                "response_temperature_hartree", "response_reciprocal_cutoff"):
        np.testing.assert_allclose(a[key], b[key], atol=1e-9, rtol=0)
    for spin_a, spin_b in zip(a["density"], b["density"]):
        for matrix_a, matrix_b in zip(spin_a, spin_b):
            for part in ("real", "imag"):
                np.testing.assert_allclose(matrix_a[part], matrix_b[part], atol=1e-9, rtol=0)
    assert a["converged"] and b["converged"]
    assert max(a["response_residuals"]) <= 1e-10
    assert max(b["response_residuals"]) <= 1e-10
    assert a["occupation_consistency_defect"] < 1e-9
    assert b["occupation_consistency_defect"] < 1e-9


@pytest.mark.parametrize("xc,omega,damping,imaginary,temperature", [
    ("pbe", .13, .04, False, 0.),
    ("tpss", .17, 0., True, 0.),
    ("pbe0", .13, .04, False, 0.),
    ("pbe", .13, .04, False, 40000.),
])
def test_native_ase_and_canonical_cli_longitudinal_spin_match(
        tmp_path, xc, omega, damping, imaginary, temperature):
    atoms, controls = fixture(tmp_path)
    controls.update(xc=xc, electronic_temperature=temperature)
    probe = dict(response_q=Q, frequency=omega, frequency_damping=damping,
                 imaginary=imaginary, spin_probe_field=[.31, -.23], spin_g_factor=2.1,
                 response_reciprocal_cutoff=1.)
    native = periodic_spin_response([1] * 4, POSITIONS, CELL, **controls, **probe)
    ase = DftRs(units="atomic", **controls).get_periodic_spin_response(atoms, **probe)
    assert_same_response(native, ase)
    assert "spin-Zeeman" in native["model"]
    assert "no orbital" in native["model"]
    numbers = np.asarray(native["spin_numbers"])
    np.testing.assert_allclose(native["total_number"], numbers.sum(axis=0), atol=1e-13, rtol=0)
    np.testing.assert_allclose(native["magnetic_moment"], -2.1 / 4 * (numbers[0] - numbers[1]),
                               atol=1e-13, rtol=0)
    moment = complex(*native["magnetic_moment"])
    susceptibility = complex(*native["spin_susceptibility"])
    assert abs(susceptibility - moment / complex(.31, -.23)) < 1e-12

    xyz = tmp_path / "spin.xyz"
    xyz.write_text("4\nC1 periodic longitudinal spin probe\n" + "".join(
        "H " + " ".join(f"{x * Bohr:.17g}" for x in row) + "\n" for row in POSITIONS))
    args = [os.environ["DFT_RS_CLI"], "periodic-spin-response", str(xyz),
            "--cell", ";".join(",".join(f"{x * Bohr:.17g}" for x in row) for row in CELL),
            "--format", "json"]
    for key, value in {**controls, **probe}.items():
        args += ["--" + key.replace("_", "-"),
                 str(value).lower() if isinstance(value, bool) else
                 ",".join(map(str, value)) if isinstance(value, list) else str(value)]
    result = subprocess.run(args, capture_output=True, text=True, cwd=tmp_path, check=False)
    assert result.returncode == 0, result.stderr
    assert "pbc-scf" in result.stderr and "residual" in result.stderr
    cli = json.loads(result.stdout)
    assert_same_response(native, cli)


def test_physical_ase_units_and_frequency_alias_preserve_the_spin_model(tmp_path):
    atoms, controls = fixture(tmp_path)
    controls["xc"] = "pbe"
    probe = dict(response_q=Q, response_omega=.13, response_damping=.04,
                 spin_probe_field=[1., 0.], response_reciprocal_cutoff=1.)
    native = periodic_spin_response([1] * 4, POSITIONS, CELL, **controls, **probe)
    physical_probe = dict(probe, response_q=(np.array(Q) / Bohr).tolist(),
                          response_omega=.13 * Hartree, response_damping=.04 * Hartree,
                          response_reciprocal_cutoff=1. / Bohr)
    physical_controls = dict(controls, overlap_cutoff=controls["overlap_cutoff"] * Bohr,
                             conv_tol=controls["conv_tol"] * Hartree,
                             conv_tol_grad=controls["conv_tol_grad"] * Hartree)
    physical = DftRs(**physical_controls).get_periodic_spin_response(atoms, **physical_probe)
    np.testing.assert_allclose(physical["energy"], native["energy"] * Hartree,
                               atol=1e-10, rtol=0)
    np.testing.assert_allclose(physical["q"], np.array(native["q"]) / Bohr, atol=1e-11)
    np.testing.assert_allclose(physical["frequency"], np.array(native["frequency"]) * Hartree,
                               atol=1e-11)
    np.testing.assert_allclose(physical["magnetic_moment"], native["magnetic_moment"], atol=1e-9)
    np.testing.assert_allclose(physical["spin_susceptibility"], native["spin_susceptibility"], atol=1e-9)


def test_static_ensemble_and_invalid_spin_probe_are_not_silently_substituted(tmp_path):
    _, controls = fixture(tmp_path)
    controls["xc"] = "pbe"
    with pytest.raises((ValueError, RuntimeError), match="common-total-number"):
        periodic_spin_response([1] * 4, POSITIONS, CELL, **controls)
    with pytest.raises((ValueError, RuntimeError), match="positive finite spin g"):
        periodic_spin_response([1] * 4, POSITIONS, CELL, response_q=Q,
                               spin_g_factor=0., **controls)
    with pytest.raises((ValueError, RuntimeError), match="requires.*real.*imaginary"):
        periodic_spin_response([1] * 4, POSITIONS, CELL, response_q=Q,
                               spin_probe_field=[1.], **controls)
    zero = periodic_spin_response([1] * 4, POSITIONS, CELL, response_q=Q, frequency=.13,
                                  spin_probe_field=[0., 0.], response_reciprocal_cutoff=1., **controls)
    assert zero["converged"] and zero["spin_susceptibility"] is None
    np.testing.assert_allclose(zero["magnetic_moment"], [0., 0.], atol=0, rtol=0)
