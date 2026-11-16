# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Common optical-rotation routes, gauge diagnostics and mirror parity."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
import dft_rs

POSITIONS = np.asarray([[-0.7, 0, 0], [0.7, 0, 0], [0.2, 0.8, 1.3], [-0.4, -0.8, 1.8]])
ORIGIN = [0.1, 0.2, -0.1]


@pytest.mark.parametrize("xc", ["hf", "pbe0"])
@pytest.mark.parametrize("gauge", ["length", "velocity"])
def test_optical_rotation_public_gauges_units_and_mirror_parity(xc, gauge, tmp_path):
    controls = dict(xc=xc, basis="sto-3g", grid_level=0, frequencies=[0.05],
                    gauge=gauge, gauge_origin=ORIGIN, conv_tol=1e-13,
                    conv_tol_grad=1e-12, conv_tol_dp=1e-12, response_tolerance=1e-12)
    native = dft_rs.optical_activity([1] * 4, POSITIONS.tolist(), **controls)
    generic = dft_rs.run_feature("optical-activity", [1] * 4, POSITIONS.tolist(), **controls)
    atoms = Atoms("H4", positions=POSITIONS * Bohr)
    calc = dft_rs.DftRs(units="atomic")
    adapted = calc.get_optical_activity(atoms, units="atomic", **controls)
    xyz = tmp_path / "hydrogen_cluster.xyz"
    xyz.write_text("4\noptical rotation\n" + "".join(
        f"H {x*Bohr:.16f} {y*Bohr:.16f} {z*Bohr:.16f}\n" for x, y, z in POSITIONS), encoding="utf-8")
    args = [os.environ["DFT_RS_CLI"], "optical-activity", str(xyz), "--format", "json"]
    for key, value in controls.items():
        args.extend(["--" + key.replace("_", "-"),
                     ",".join(map(str, value)) if isinstance(value, list) else str(value)])
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cli = json.loads(result.stdout)
    tensor = np.asarray(native["rotation_tensors"])
    assert tensor.shape == (1, 3, 3) and np.isfinite(tensor).all()
    assert np.max(np.abs(tensor)) > 1e-12
    assert ("modified velocity gauge" if gauge == "velocity" else "length gauge") in native["model"]
    assert max(native["response_residuals"]) < 1e-10
    for report in (generic, adapted, cli):
        np.testing.assert_allclose(report["rotation_tensors"], tensor, rtol=1e-8, atol=1e-10)
        assert report["model"] == native["model"] and report["units"] == native["units"]
    reflection = np.diag([-1.0, 1.0, 1.0])
    mirror = dft_rs.optical_activity([1] * 4, (POSITIONS @ reflection).tolist(),
                                     **dict(controls, gauge_origin=(np.asarray(ORIGIN) @ reflection).tolist()))
    expected = -reflection @ tensor[0] @ reflection
    np.testing.assert_allclose(mirror["rotation_tensors"][0], expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(mirror["beta"], -np.asarray(native["beta"]), rtol=1e-8, atol=1e-10)
    ase_controls = dict(controls, frequencies=[0.05 * Hartree], gauge_origin=np.asarray(ORIGIN) * Bohr)
    converted = calc.get_optical_activity(atoms, units="ase", **ase_controls)
    np.testing.assert_allclose(converted["rotation_tensors"], tensor * Bohr**3, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(converted["beta"], np.asarray(native["beta"]) * Bohr**3 / Hartree,
                               rtol=1e-10, atol=1e-12)
    print(xc, gauge, "optical tensor", tensor.tolist(), "residual", native["response_residuals"])
