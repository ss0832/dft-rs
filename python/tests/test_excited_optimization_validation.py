# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""T120 isolated HF singlet TDA optimization on native, ASE and CLI routes."""

import json
import math
import os
from pathlib import Path
import subprocess

from ase import Atoms
import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM
import numpy as np


def geometry():
    cz, sz = math.cos(0.41), math.sin(0.41)
    cx, sx = math.cos(0.63), math.sin(0.63)
    points = []
    for x, y, z in [[0.02, 0.01, -0.04], [0.04, -0.03, 2.30],
                    [0.17, 1.83, -1.06], [0.11, -1.68, -1.19]]:
        a, b = cz*x-sz*y, sz*x+cz*y
        points.append([a+0.17, cx*b-sx*z-0.23, sx*b+cx*z+0.31])
    return [6, 8, 1, 1], np.asarray(points)


def test_isolated_excited_optimization_has_consistent_public_endpoints(tmp_path):
    numbers, positions = geometry()
    options = dict(xc="hf", basis="sto-3g", grid_weight_cutoff=0.0,
                   conv_tol=1e-13, conv_tol_dp=1e-12, conv_tol_grad=1e-11,
                   td_tolerance=1e-11, nstates=4, root=1, max_steps=200, gtol=1e-5)
    native = dft_rs.excited_optimization(numbers, positions.tolist(), **options)
    atoms = Atoms(numbers=numbers, positions=positions*BOHR_IN_ANGSTROM)
    adapted = dft_rs.DftRs(units="atomic", **options).get_excited_optimization(atoms, **options)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable and Path(executable).is_file(), "DFT_RS_CLI selects the fixed build"
    xyz = tmp_path / "c1_formaldehyde.xyz"
    xyz.write_text("4\nC1 formaldehyde\n" + "".join(
        symbol + " " + " ".join(f"{v:.17g}" for v in row) + "\n"
        for symbol, row in zip(["C", "O", "H", "H"], positions*BOHR_IN_ANGSTROM)
    ), encoding="utf-8")
    args = [executable, "excited-optimization", str(xyz), "--format", "json"]
    for key, value in options.items():
        args += ["--" + key.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)]
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", check=False)
    assert result.returncode == 0, result.stderr
    assert "SCF" in result.stderr
    cli = json.loads(result.stdout)
    final_distances = []
    for name, report in [("native", native), ("ASE", adapted), ("CLI", cli)]:
        assert report["optimization_converged"] and report["gradient_max"] < 1e-5
        energies = np.asarray(report["trajectory_energies"])
        assert len(energies) > 1 and np.all(np.diff(energies) <= 1e-12)
        assert energies[-1] < energies[0]
        final = np.asarray(report["positions"])
        fresh_options = {key: value for key, value in options.items() if key not in ["max_steps", "gtol"]}
        fresh = dft_rs.td_state_gradient(numbers, final.tolist(), **fresh_options)
        assert abs(fresh["excited_state_energy"] - report["excited_state_energy"]) < 1e-9
        np.testing.assert_allclose(fresh["gradient"], report["gradient"], rtol=0, atol=1e-8)
        assert abs(report["excited_state_energy"] - native["excited_state_energy"]) < 1e-9
        final_distances.append(np.linalg.norm(final[:, None, :] - final[None, :, :], axis=2))
        print(f"T120 {name}: iterations={report['iterations']} energy={report['excited_state_energy']:.12f} "
              f"gradient_max={report['gradient_max']:.12e} fresh_gradient_defect="
              f"{np.max(np.abs(np.asarray(fresh['gradient'])-report['gradient'])):.12e}")
    for distances in final_distances[1:]:
        np.testing.assert_allclose(distances, final_distances[0], rtol=0, atol=1e-5)
