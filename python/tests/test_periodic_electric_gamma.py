# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Periodic fourth electric-field derivatives on every common public route."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM
from test_periodic_gks_stress import cli_command


POSITIONS = [[-0.1, -0.05, -0.7], [0.1, 0.05, 0.7]]
CELL = np.eye(3) * 9.0
BASE = dict(xc="hf", basis="sto-3g", k_mesh=[1, 1, 1], overlap_cutoff=7.0,
            grid_level=0, grid_shell=1, conv_tol=1e-12, conv_tol_grad=1e-12,
            conv_tol_dp=1e-12, max_iter=250, guess="core", response_tolerance=1e-12)


@pytest.mark.parametrize("bias", [[0.0, 0.0, 0.0], [0.004, -0.006, 0.014]])
def test_periodic_gamma_native_generic_ase_cli_and_units(tmp_path, bias):
    controls = dict(BASE, field=bias)
    native = dft_rs.periodic_gamma([1, 1], POSITIONS, CELL.tolist(), **controls)
    generic = dft_rs.run_feature("periodic-gamma", [1, 1], POSITIONS, cell=CELL.tolist(),
                                 use_symmetry=False, **controls)
    atoms = Atoms("H2", positions=np.asarray(POSITIONS) * Bohr, cell=CELL * Bohr, pbc=True)
    calc = dft_rs.DftRs(units="atomic", **BASE)
    adapted = calc.get_periodic_gamma(atoms, units="atomic", field=bias)
    adapted_generic = calc.get_feature_report("periodic-gamma", atoms, units="atomic", field=bias)
    xyz = tmp_path / "hydrogen.xyz"
    xyz.write_text("2\nperiodic electric tensor\n" + "".join(
        f"H {x*BOHR_IN_ANGSTROM:.16f} {y*BOHR_IN_ANGSTROM:.16f} {z*BOHR_IN_ANGSTROM:.16f}\n"
        for x, y, z in POSITIONS), encoding="utf-8")
    args = [os.environ["DFT_RS_CLI"], "periodic-gamma", str(xyz), "--cell",
            ";".join(",".join(f"{value*BOHR_IN_ANGSTROM:.16f}" for value in row) for row in CELL),
            "--format", "json"]
    for key, value in controls.items():
        args.extend(["--" + key.replace("_", "-"),
                     ",".join(map(str, value)) if isinstance(value, list) else str(value)])
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cli = json.loads(result.stdout)
    assert native["converged"]
    assert native["tensor_shape"] == [3] * 4
    assert native["polarization_derivative_order"] == 3
    assert native["response_order"] == 2
    np.testing.assert_array_equal(native["reference_field"], bias)
    assert native["provenance"] == "analytic"
    tensor = np.asarray(native["tensor"])
    assert tensor.shape == (81,) and np.isfinite(tensor).all()
    assert np.max(np.abs(tensor)) > 1e-8
    np.testing.assert_array_equal(native["third_susceptibility"], tensor)
    np.testing.assert_allclose(np.asarray(native["chi3"]) * 6, tensor, rtol=1e-13, atol=1e-14)
    assert native["use_symmetry"] is True
    assert native["independent_components"] == 15
    assert native["raw_permutation_residual"] is None
    assert generic["use_symmetry"] is False
    assert generic["independent_components"] == 81
    assert generic["raw_permutation_residual"] < 1e-10
    assert max(native["response_residuals"]) <= 1e-12
    for other in [generic, adapted, adapted_generic, cli]:
        assert other["units"] == native["units"]
        assert other["model"] == native["model"]
        assert other["tensor_shape"] == native["tensor_shape"]
        np.testing.assert_allclose(other["tensor"], tensor, rtol=1e-9, atol=1e-12)
    converted = calc.get_periodic_gamma(atoms, units="ase", field=np.asarray(bias) * Hartree / Bohr)
    np.testing.assert_allclose(converted["tensor"], tensor * Bohr**2 / Hartree**2,
                               rtol=1e-11, atol=1e-14)
    assert "eV" in converted["units"]["tensor"]
    np.testing.assert_allclose(converted["reference_field"], np.asarray(bias) * Hartree / Bohr,
                               rtol=1e-12, atol=1e-14)
    print("periodic gamma public tensor", tensor.tolist(), "raw permutation",
          generic["raw_permutation_residual"], "response", native["response_residuals"])


@pytest.mark.parametrize("xc", ["hf", "tpss"])
def test_periodic_gamma_closed_inert_orbital_limit(xc):
    result = dft_rs.periodic_gamma([2], [[0, 0, 0]], CELL.tolist(), **dict(BASE, xc=xc))
    assert result["converged"]
    np.testing.assert_array_equal(result["tensor"], np.zeros(81))
    assert result["raw_permutation_residual"] is None
    assert result["independent_components"] == 15


@pytest.mark.parametrize("feature,field_key,nuclear_key", [
    ("periodic-efield", "field", "forces"),
    ("phonons", "reference_field", "force_constants"),
])
def test_periodic_implicit_chg_finite_field_on_public_routes(tmp_path, feature, field_key, nuclear_key):
    numbers = [3, 1]
    positions = [[0.17, -0.25, 0.08], [2.89, 0.44, -0.57]]
    cell = [[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]]
    field = [0.001, -0.0015, 0.002]
    controls = dict(xc="wb97x-d", basis="sto-3g", k_mesh=[1, 1, 1], field=field,
                    disp_cutoff=12.3, exchange_cutoff=5.6, overlap_cutoff=6.0,
                    grid_shell=1, grid_level=0, grid_weight_cutoff=0.0, conv_tol=1e-13,
                    conv_tol_dp=1e-11, conv_tol_grad=1e-11, cpu_threads=1)
    if feature == "periodic-efield":
        controls["forces"] = True
    name = feature.replace("-", "_")
    native = getattr(dft_rs, name)(numbers, positions, cell, **controls)
    atoms = Atoms(numbers, positions=np.array(positions) * Bohr,
                  cell=np.array(cell) * Bohr, pbc=True)
    adapted = getattr(dft_rs.DftRs(), f"get_{name}")(atoms, units="atomic", **controls)
    args = cli_command(os.environ["DFT_RS_CLI"], tmp_path, numbers, positions, cell,
                       controls, feature=feature)
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    assert "residual" in result.stderr.lower()
    assert native["dispersion_energy"] < 0.0
    assert native["units"]["dispersion_energy_per_primitive_cell"] == "hartree per primitive cell"
    assert np.max(np.abs(native[nuclear_key])) > 1e-6
    for report in (native, adapted, json.loads(result.stdout)):
        assert report["converged"] and report["units"] == native["units"]
        np.testing.assert_allclose(report[field_key], field, rtol=0, atol=1e-14)
        assert report["dispersion_energy"] == report["dispersion_energy_per_primitive_cell"]
        for key in ("energy", "field_energy", "dispersion_energy", nuclear_key):
            np.testing.assert_allclose(report[key], native[key], rtol=1e-9, atol=1e-10)
        assert report["series_reference_residual"] < 1e-10
    print(f"{feature} E={native['field_energy']:.17e} CHG={native['dispersion_energy']:.17e} "
          f"max_{nuclear_key}={np.max(np.abs(native[nuclear_key])):.17e} "
          f"residual={native['series_reference_residual']:.17e}")
