# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Bounded T017/T139 public-route acceptance with strict report units."""

import json
import os
from pathlib import Path
import subprocess

from ase import Atoms
import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM
import numpy as np
import pytest


def cli_report(feature, numbers, positions, controls, tmp_path):
    executable = os.environ.get("DFT_RS_CLI")
    assert executable and Path(executable).is_file(), "DFT_RS_CLI selects the verified build"
    symbols = {1: "H", 7: "N", 8: "O"}
    xyz = tmp_path / "geometry.xyz"
    xyz.write_text(str(len(numbers)) + "\nvalidation\n" + "".join(
        symbols[z] + " " + " ".join(f"{v:.17g}" for v in row) + "\n"
        for z, row in zip(numbers, positions * BOHR_IN_ANGSTROM)
    ), encoding="utf-8")
    args = [executable, feature, str(xyz), "--format", "json"]
    for name, value in controls.items():
        args += ["--" + name.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)]
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", check=False)
    assert result.returncode == 0, result.stderr
    assert "SCF" in result.stderr
    return json.loads(result.stdout)


def controls(xc):
    return dict(xc=xc, basis="sto-3g", grid_weight_cutoff=0.0,
                conv_tol=1e-13, conv_tol_dp=1e-12, conv_tol_grad=1e-11)


@pytest.mark.parametrize("xc", ["hf", "pbe"])
def test_streamed_gradient_native_ase_cli_and_stored_gradient(xc, tmp_path):
    numbers = [7, 1, 1, 1]
    positions = np.array([[0.17, -0.11, 0.23], [1.43, 0.81, -0.62],
                          [-0.63, 1.59, -0.43], [-1.12, -1.36, -0.71]])
    options = controls(xc)
    streamed = dft_rs.streamed_gradient(numbers, positions.tolist(), **options)
    stored = dft_rs.gradient(numbers, positions.tolist(), **options)
    generic = dft_rs.run_feature("streamed-gradient", numbers, positions.tolist(), **options)
    atoms = Atoms(numbers=numbers, positions=positions * BOHR_IN_ANGSTROM)
    adapted = dft_rs.DftRs(units="atomic", **options).get_streamed_gradient(atoms, **options)
    cli = cli_report("streamed-gradient", numbers, positions, options, tmp_path)
    for name, report in [("stored", stored), ("generic", generic), ("ASE", adapted), ("CLI", cli)]:
        np.testing.assert_allclose(streamed["gradient"], report["gradient"], rtol=0, atol=1e-9)
        assert abs(streamed["energy"] - report["energy"]) < 1e-10
        assert report["converged"]
        defect = np.max(np.abs(np.asarray(streamed["gradient"]) - report["gradient"]))
        print(f"T139 {xc} {name}: gradient defect={defect:.12e} Hartree/Bohr; "
              f"energy defect={abs(streamed['energy']-report['energy']):.12e} Hartree")
    for report in [streamed, generic, adapted, cli]:
        assert report["integral_storage"] == "none"
        assert report["translation_residual"] < 1e-8


def test_dynamic_raman_native_ase_cli_with_normal_mode_phase_alignment(tmp_path):
    numbers = [8, 1, 1]
    positions = np.array([[0.0, 0.0, 0.2217], [0.0, 1.4309, -0.8867],
                          [0.13, -1.4309, -0.8867]])
    options = dict(controls("hf"), raman_frequency=0.1)
    native = dft_rs.run_feature("ir-raman", numbers, positions.tolist(), **options)
    atoms = Atoms(numbers=numbers, positions=positions * BOHR_IN_ANGSTROM)
    adapted = dft_rs.DftRs(units="atomic", **options).get_feature_report("ir-raman", atoms, **options)
    cli = cli_report("ir-raman", numbers, positions, options, tmp_path)
    modes = np.asarray(native["normal_modes_cartesian"]).reshape(-1, 9)
    derivatives = np.asarray(native["dynamic_polarizability_derivatives"]).reshape(len(modes), 9)
    for name, report in [("ASE", adapted), ("CLI", cli)]:
        other_modes = np.asarray(report["normal_modes_cartesian"]).reshape(modes.shape)
        overlap = np.sum(modes * other_modes, axis=1)
        cosine = np.abs(overlap) / np.linalg.norm(modes, axis=1) / np.linalg.norm(other_modes, axis=1)
        assert np.all(cosine > 1-1e-10)
        phase = np.where(overlap > 0, 1.0, -1.0)
        other = np.asarray(report["dynamic_polarizability_derivatives"]).reshape(derivatives.shape)
        aligned = phase[:, None] * other
        np.testing.assert_allclose(derivatives, aligned, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(native["dynamic_raman_activities"], report["dynamic_raman_activities"],
                                   rtol=1e-9, atol=1e-9)
        assert report["dynamic_response_converged"]
        print(f"T017 {name}: aligned derivative defect={np.max(np.abs(derivatives-aligned)):.12e}; "
              f"residual={report['dynamic_response_residual']:.12e}")
    assert native["dynamic_response_converged"]
