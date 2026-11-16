# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Explicit analytic triplet optimization path across public driver routes."""

import json
import os
from pathlib import Path
import subprocess

from ase import Atoms
import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM
import numpy as np


def test_triplet_optimization_path_has_consistent_public_reports(tmp_path):
    numbers = [1] * 4
    positions = np.asarray([[0.13, 0.21, -0.07], [1.48, -0.19, 0.32],
                            [-0.43, 1.71, 0.58], [0.36, -0.81, 2.05]])
    options = dict(xc="hf", basis="sto-3g", charge=2, td_spin="triplet", nstates=3,
                   td_gradient_route="path", max_steps=1, gtol=1e-9,
                   conv_tol=1e-13, conv_tol_dp=1e-12, conv_tol_grad=1e-11,
                   td_tolerance=1e-11, grid_weight_cutoff=0.0)
    native = dft_rs.excited_optimization(numbers, positions.tolist(), **options)
    atoms = Atoms(numbers=numbers, positions=positions * BOHR_IN_ANGSTROM)
    adapted = dft_rs.DftRs(units="atomic", **options).get_excited_optimization(atoms, **options)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable and Path(executable).is_file(), "DFT_RS_CLI selects the fixed rebuilt artifact"
    xyz = tmp_path / "c1_four_center.xyz"
    xyz.write_text("4\nC1 two-electron four-center model\n" + "".join(
        "H " + " ".join(f"{v:.17g}" for v in row) + "\n"
        for row in positions * BOHR_IN_ANGSTROM
    ), encoding="utf-8")
    args = [executable, "excited-optimization", str(xyz), "--format", "json"]
    for key, value in options.items():
        args += ["--" + key.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)]
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", check=False)
    assert result.returncode == 0, result.stderr
    assert "SCF" in result.stderr
    for report in [native, adapted, json.loads(result.stdout)]:
        assert report["td_spin"] == "triplet" and report["td_method"] == "tda"
        assert report["td_gradient_route"] == "path"
        assert report["response_residual"] < 1e-8
        assert report["provenance"] == "analytic"
        assert len(report["trajectory_energies"]) == 2
        assert report["trajectory_energies"][1] < report["trajectory_energies"][0]
        assert abs(report["excited_state_energy"] - native["excited_state_energy"]) < 1e-9
        np.testing.assert_allclose(report["positions"], native["positions"], rtol=0, atol=1e-8)
        np.testing.assert_allclose(report["gradient"], native["gradient"], rtol=0, atol=1e-8)
