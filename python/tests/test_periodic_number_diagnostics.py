# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Particle counts remain dimensionless through the common periodic routes."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms

from dft_rs import DftRs, run_feature
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree


@pytest.mark.parametrize("xc", ["hf", "pbe"])
def test_periodic_number_report_native_ase_cli(tmp_path, xc):
    positions = np.array([[.11, -.17, .23], [1.24, .19, .57]])
    cell = np.array([[7.3, .21, .17], [.35, 7.9, .41], [.13, .29, 8.6]])
    basis = tmp_path / "number-report.g94"
    basis.write_text("H 0\nS 1 1.0\n1.2 1.0\n****\nHe 0\nS 1 1.0\n1.6 1.0\n****\n")
    controls = dict(basis=str(basis), xc=xc, charge=1, k_mesh=[2, 1, 1],
                    overlap_cutoff=3., grid_level=1, grid_shell=1,
                    grid_weight_cutoff=0., conv_tol=1e-13,
                    conv_tol_dp=1e-12, conv_tol_grad=1e-11)
    native = run_feature("periodic-energy", [1, 2], positions,
                         cell=cell, pbc=[True] * 3, **controls)
    atoms = Atoms("HHe", positions=positions * Bohr, cell=cell * Bohr, pbc=True)
    ase_controls = dict(controls, overlap_cutoff=controls["overlap_cutoff"] * Bohr,
                        conv_tol=controls["conv_tol"] * Hartree,
                        conv_tol_grad=controls["conv_tol_grad"] * Hartree)
    ase = DftRs(units="ase", **ase_controls).get_periodic_energy_report(atoms)
    xyz = tmp_path / "number-report.xyz"
    xyz.write_text("2\nC1 particle-number report\n" + "".join(
        symbol + " " + " ".join(f"{x * Bohr:.17g}" for x in row) + "\n"
        for symbol, row in zip(("H", "He"), positions)))
    args = [os.environ["DFT_RS_CLI"], "periodic-energy", str(xyz), "--cell",
            ";".join(",".join(f"{x * Bohr:.17g}" for x in row) for row in cell),
            "--format", "json"]
    for key, value in controls.items():
        args += ["--" + key.replace("_", "-"),
                 ",".join(map(str, value)) if isinstance(value, list) else str(value)]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "pbc-scf" in result.stderr and "residual" in result.stderr
    cli = json.loads(result.stdout)
    reference = native["particle_number_diagnostics"]
    for report in (native, ase, cli):
        number = report["particle_number_diagnostics"]
        assert "dimensionless" in report["units"]["particle_number_diagnostics"]
        assert number["occupied_electrons"] == 2
        np.testing.assert_allclose(number["overlap_metric_electrons"], 2., atol=1e-11, rtol=0.)
        for key in ["overlap_metric_electrons", "overlap_metric_number_error",
                    "grid_electrons", "grid_number_error", "relative_grid_number_error"]:
            if reference[key] is None:
                assert number[key] is None
            else:
                np.testing.assert_allclose(number[key], reference[key], atol=1e-11, rtol=1e-10)
        if xc == "hf":
            assert number["grid_electrons"] is None
            assert number["large_grid_discrepancy"] is None
        else:
            np.testing.assert_allclose(number["grid_number_error"],
                                       number["grid_electrons"] - 2., atol=1e-14, rtol=0.)
            np.testing.assert_allclose(number["relative_grid_number_error"],
                                       abs(number["grid_electrons"] - 2.) / 2., atol=1e-14, rtol=0.)
            assert number["large_grid_discrepancy"] == (number["relative_grid_number_error"] > .01)
        assert report["converged"]
