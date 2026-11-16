# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Full double-hybrid Cartesian tensors and spin channels on public routes."""
import json
import os
import re
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree

import dft_rs
from dft_rs import DftRs
from dft_rs.units import BOHR_IN_ANGSTROM

POSITIONS = np.array([[0.13, -0.21, 0.19], [1.57, 0.16, -0.08],
                      [-0.19, 1.93, 0.37], [0.42, 0.29, 2.28]])
CONTROLS = dict(basis="sto-3g", grid_level=0, grid_weight_cutoff=0.0,
                conv_tol=1e-13, conv_tol_grad=1e-12, conv_tol_dp=1e-12,
                response_tolerance=1e-12, max_iter=300, guess="core")
CHANNELS = ("", "parent_", "target_", "pt2_", "pt2_same_spin_alpha_",
            "pt2_opposite_spin_", "pt2_same_spin_beta_", "dispersion_")


def validate(report, atom_count=4):
    assert report["provenance"] == "analytic" and report["converged"]
    for prefix in CHANNELS:
        for tensor, shape, unit in [
            ("gradient", (atom_count, 3), "hartree/bohr"),
            ("hessian", (3 * atom_count, 3 * atom_count), "hartree/bohr^2"),
        ]:
            key = prefix + tensor
            values = np.asarray(report[key])
            assert values.shape == shape and np.isfinite(values).all(), key
            assert report["units"][key] == unit
    for tensor in ["gradient", "hessian"]:
        pt2 = (report["pt2_same_spin_coefficient"] * (
            np.asarray(report["pt2_same_spin_alpha_" + tensor]) +
            np.asarray(report["pt2_same_spin_beta_" + tensor])) +
            report["pt2_opposite_spin_coefficient"] *
            np.asarray(report["pt2_opposite_spin_" + tensor]))
        np.testing.assert_allclose(report["pt2_" + tensor], pt2, rtol=0, atol=1e-10)
        total = (np.asarray(report["target_" + tensor]) + pt2 +
                 np.asarray(report["dispersion_" + tensor]))
        np.testing.assert_allclose(report[tensor], total, rtol=0, atol=1e-10)
    np.testing.assert_allclose(report["forces"], -np.asarray(report["gradient"]), rtol=0, atol=0)
    for key in ("response_residuals", "amplitude_response_residuals",
                "projector_defects", "nuclear_stationarity_defects",
                "metric_defects", "response_number_defects", "orbital_frame_defects",
                "reference_energy_defects", "reference_fock_defects"):
        values = np.asarray(report[key])
        assert values.size and np.isfinite(values).all(), key
        assert np.max(np.abs(values)) < 1e-10, key
    print("correlated nuclear raw orbital/amplitude residuals:",
          max(report["response_residuals"]), max(report["amplitude_response_residuals"]))


@pytest.mark.parametrize("parent", ["hf", "pbe", "tpss", "pbe0"])
def test_unrestricted_correlated_nuclear_all_channels_on_public_routes(parent, tmp_path):
    # Five asymmetric hydrogen centers give three alpha and two beta electrons.
    # Both same-spin channels are nonzero, unlike a one-beta-electron fixture.
    positions = np.vstack([POSITIONS, [1.41, 1.83, 1.59]])
    controls = dict(CONTROLS, xc=parent, reference="uks", multiplicity=2,
                    double_hybrid_model="custom", pt2_same_spin=0.37,
                    pt2_opposite_spin=0.61)
    native = dft_rs.hessian_report([1] * 5, positions.tolist(), **controls)
    atoms = Atoms("H5", positions=positions * Bohr)
    calculator = DftRs(**controls)
    adapted = calculator.get_feature_report("hessian", atoms, units="atomic")

    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for route acceptance"
    xyz = tmp_path / "unrestricted_correlated_nuclear.xyz"
    xyz.write_text("5\nunrestricted correlated nuclear tensors\n" + "".join(
        "H " + " ".join(f"{v:.17g}" for v in row) + "\n"
        for row in positions * Bohr), encoding="utf-8")
    arguments = [executable, "hessian", str(xyz), "--format", "json"]
    for name, value in controls.items():
        arguments += ["--" + name.replace("_", "-"), str(value).lower()]
    result = subprocess.run(arguments, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert re.search(r"\bE\s*=", result.stderr) and "[scf] iter" in result.stderr
    cli = json.loads(result.stdout)
    for report in (native, adapted, cli):
        validate(report, atom_count=5)
        assert report["reference_type"] == "uks" and report["unrestricted"]
        assert (report["n_alpha"], report["n_beta"]) == (3, 2)
        for key in ("pt2_same_spin_alpha", "pt2_opposite_spin", "pt2_same_spin_beta"):
            assert np.isfinite(report[key]) and abs(report[key]) > 1e-12, key
        for prefix in CHANNELS:
            for tensor in ("nuclear_energy", "gradient", "hessian"):
                key = prefix + tensor
                np.testing.assert_allclose(report[key], native[key], rtol=0, atol=1e-9)
    # The ASE convenience getter must convert every Cartesian Hessian entry.
    np.testing.assert_allclose(calculator.get_hessian(atoms),
                               np.asarray(native["hessian"]) * Hartree / Bohr**2,
                               rtol=0, atol=1e-8)


@pytest.mark.parametrize("model", ["b2plyp", "dsd-blyp", "dsd-pbep86", "pwpb95"])
def test_named_double_hybrid_nuclear_tensors_on_native_and_ase(model):
    controls = dict(CONTROLS, xc=model)
    report = dft_rs.hessian([1] * 4, POSITIONS.tolist(), **controls)
    validate(report)
    calc = DftRs(xc=model, basis="sto-3g")
    adapted = calc.get_feature_report("hessian", Atoms("H4", positions=POSITIONS * Bohr),
                                      units="atomic", **controls)
    validate(adapted)
    for key in ["gradient", "hessian", "pt2_gradient", "pt2_hessian"]:
        np.testing.assert_allclose(report[key], adapted[key], rtol=0, atol=1e-9)
    atoms = Atoms("H4", positions=POSITIONS * Bohr)
    atoms.calc = DftRs(xc=model, units="atomic", **CONTROLS)
    np.testing.assert_allclose(atoms.get_forces(), -np.asarray(report["gradient"]) * Hartree / Bohr,
                               rtol=0, atol=1e-8)


def test_double_hybrid_canonical_cli_returns_every_nuclear_tensor_entry(tmp_path):
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for route acceptance"
    xyz = tmp_path / "correlated_nuclear.xyz"
    xyz.write_text("4\ncorrelated nuclear tensors\n" + "".join(
        "H " + " ".join(f"{v:.17g}" for v in row) + "\n"
        for row in POSITIONS * BOHR_IN_ANGSTROM), encoding="utf-8")
    controls = dict(CONTROLS, xc="b2plyp")
    args = [executable, "hessian", str(xyz), "--format", "json"]
    for name, value in controls.items():
        args += ["--" + name.replace("_", "-"), str(value).lower()]
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    validate(report)
    native = dft_rs.hessian([1] * 4, POSITIONS.tolist(), **controls)
    for key in ["gradient", "hessian", "pt2_gradient", "pt2_hessian"]:
        np.testing.assert_allclose(report[key], native[key], rtol=0, atol=1e-9)
    plain = args.copy()
    index = plain.index("--format")
    del plain[index:index + 2]
    result = subprocess.run(plain, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "pt2_hessian" in result.stdout


def test_legacy_native_hessian_and_calculate_use_correlated_nuclear_response():
    controls = dict(CONTROLS, xc="b2plyp")
    common = dft_rs.hessian_report([1] * 4, POSITIONS.tolist(), **controls)
    legacy = {key: value for key, value in controls.items()
              if key not in {"grid_weight_cutoff", "response_tolerance", "guess"}}
    hessian = dft_rs.hessian([1] * 4, POSITIONS.tolist(), **legacy)
    calculation = dft_rs.calculate([1] * 4, POSITIONS.tolist(), gradient=True, **legacy)
    assert hessian["method"] == "analytic"
    np.testing.assert_allclose(hessian["hessian"], common["hessian"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(calculation["gradient"], common["gradient"], rtol=0, atol=1e-8)


def test_custom_distinct_target_on_native_ase_and_canonical_cli(tmp_path):
    controls = dict(CONTROLS, xc="hf", double_hybrid_model="custom",
                    double_hybrid_target_xc="pbe", pt2_same_spin=0.37,
                    pt2_opposite_spin=0.61)
    native = dft_rs.hessian_report([1] * 4, POSITIONS.tolist(), **controls)
    validate(native)
    atoms = Atoms("H4", positions=POSITIONS * Bohr)
    adapted = DftRs(**controls).get_feature_report("hessian", atoms, units="atomic")
    validate(adapted)
    for key in ["gradient", "hessian", "target_gradient", "target_hessian"]:
        np.testing.assert_allclose(native[key], adapted[key], rtol=0, atol=1e-9)
    atoms.calc = DftRs(units="atomic", **controls)
    np.testing.assert_allclose(atoms.calc.get_hessian(atoms), np.asarray(native["hessian"]) * Hartree / Bohr**2,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(atoms.get_forces(), -np.asarray(native["gradient"]) * Hartree / Bohr,
                               rtol=0, atol=1e-8)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for route acceptance"
    xyz = tmp_path / "distinct_target.xyz"
    xyz.write_text("4\ndistinct target nuclear tensors\n" + "".join(
        "H " + " ".join(f"{v:.17g}" for v in row) + "\n"
        for row in POSITIONS * BOHR_IN_ANGSTROM), encoding="utf-8")
    args = [executable, "hessian", str(xyz), "--format", "json"]
    for name, value in controls.items():
        args += ["--" + name.replace("_", "-"), str(value).lower()]
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert re.search(r"\bE\s*=", result.stderr) and "[scf] iter" in result.stderr
    cli = json.loads(result.stdout)
    validate(cli)
    for key in ["gradient", "hessian", "target_gradient", "target_hessian"]:
        np.testing.assert_allclose(native[key], cli[key], rtol=0, atol=1e-9)
    with pytest.raises(ValueError, match="requires double_hybrid_model=custom"):
        dft_rs.energy([1] * 4, POSITIONS.tolist(), xc="b2plyp",
                      double_hybrid_target_xc="pbe")
