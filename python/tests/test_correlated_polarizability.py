# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Static correlated alpha through the existing common property API."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
import dft_rs

POSITIONS = [[-0.1, -0.05, -0.7], [0.1, 0.05, 0.7]]
BASE = dict(basis="sto-3g", grid_level=0, vv10_grid_level=0,
            conv_tol=1e-13, conv_tol_grad=1e-12, conv_tol_dp=1e-12,
            response_tolerance=1e-11, field=[0.0004, -0.0006, 0.0014])


@pytest.mark.parametrize("model", ["b2plyp", "dsd-pbep86", "dsd-blyp", "pwpb95", "wb97m2",
                                    "pbe", "tpss", "pbe0"])
def test_correlated_alpha_native_ase_cli_and_units(model, tmp_path):
    controls = dict(BASE, xc=model)
    if model in ("pbe", "tpss", "pbe0"):
        controls.update(pt2_same_spin=0.2, pt2_opposite_spin=0.3)
    native = dft_rs.polarizability_report([1, 1], POSITIONS, **controls)
    generic = dft_rs.run_feature("polarizability", [1, 1], POSITIONS, **controls)
    atoms = Atoms("H2", positions=np.asarray(POSITIONS) * Bohr)
    calc = dft_rs.DftRs(units="atomic")
    adapted = calc.get_polarizability_report(atoms, units="atomic", **controls)
    xyz = tmp_path / "hydrogen.xyz"
    xyz.write_text("2\ncorrelated static response\n" + "".join(
        f"H {x*Bohr:.16f} {y*Bohr:.16f} {z*Bohr:.16f}\n" for x, y, z in POSITIONS), encoding="utf-8")
    args = [os.environ["DFT_RS_CLI"], "polarizability", str(xyz), "--format", "json"]
    for key, value in controls.items():
        args.extend(["--" + key.replace("_", "-"),
                     ",".join(map(str, value)) if isinstance(value, list) else str(value)])
    completed = subprocess.run(args, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    cli = json.loads(completed.stdout)
    tensor = np.asarray(native["polarizability"])
    assert native["converged"] and native["response_order"] == 2
    assert native["independent_components"] == 6
    assert native["raw_asymmetry"] is None
    assert tensor.shape == (3, 3) and np.isfinite(tensor).all()
    assert np.max(np.abs(native["pt2_polarizability_correction"])) > 1e-9
    for result in (generic, adapted, cli):
        np.testing.assert_allclose(result["polarizability"], tensor, rtol=1e-8, atol=1e-10)
        assert result["units"] == native["units"]
        assert result["model"] == native["model"]
    np.testing.assert_allclose(tensor, np.asarray(native["parent_polarizability"])
                               + np.asarray(native["target_polarizability_correction"])
                               + np.asarray(native["pt2_polarizability_correction"]), atol=1e-12)
    assert np.max(np.abs(native["response_residuals"])) < 1e-9
    assert np.max(np.abs(native["amplitude_response_residuals"])) < 1e-9
    converted_controls = dict(controls, field=np.asarray(controls["field"]) * Hartree / Bohr)
    converted = calc.get_polarizability_report(atoms, units="ase", **converted_controls)
    np.testing.assert_allclose(converted["polarizability"], tensor * Bohr**2 / Hartree,
                               rtol=1e-10, atol=1e-12)
    print(model, "correlated alpha", tensor.tolist(), "response", native["response_residuals"],
          "pair", native["amplitude_response_residuals"])
