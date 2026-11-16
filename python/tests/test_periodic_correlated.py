# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Periodic pair/orbital response through the existing property surfaces."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
import dft_rs

POSITIONS = [[-0.1, -0.05, -0.7], [0.1, 0.05, 0.7]]
CELL = np.eye(3) * 9.0
BASE = dict(basis="sto-3g", k_mesh=[1, 1, 1], overlap_cutoff=7.0,
            grid_level=0, grid_shell=1, conv_tol=1e-12, conv_tol_grad=1e-12,
            conv_tol_dp=1e-12, max_iter=250, response_tolerance=1e-12)


@pytest.mark.parametrize("model", ["b2plyp", "pbe", "tpss", "pbe0", "wb97m2"])
def test_periodic_correlated_field_native_ase_cli_units(model, tmp_path):
    controls = dict(BASE, xc=model, field=[0.004, -0.006, 0.014],
                    response_order=2, field_direction=[0.2, -0.3, 0.7])
    if model=="wb97m2":
        controls["vv10_cutoff"]=11.3
    if model in ("pbe", "tpss", "pbe0"):
        controls.update(pt2_same_spin=0.2, pt2_opposite_spin=0.3)
    native = dft_rs.periodic_efield([1, 1], POSITIONS, CELL.tolist(), **controls)
    generic = dft_rs.run_feature("periodic-efield", [1, 1], POSITIONS,
                                 cell=CELL.tolist(), **controls)
    atoms = Atoms("H2", positions=np.asarray(POSITIONS)*Bohr, cell=CELL*Bohr, pbc=True)
    calc = dft_rs.DftRs(units="atomic")
    adapted = calc.get_periodic_efield(atoms, units="atomic", **controls)
    xyz = tmp_path / "hydrogen.xyz"
    xyz.write_text("2\nperiodic correlated response\n" + "".join(
        f"H {x*Bohr:.16f} {y*Bohr:.16f} {z*Bohr:.16f}\n" for x,y,z in POSITIONS), encoding="utf-8")
    args = [os.environ["DFT_RS_CLI"], "periodic-efield", str(xyz), "--cell",
            ";".join(",".join(f"{v*Bohr:.16f}" for v in row) for row in CELL), "--format", "json"]
    for key, value in controls.items():
        args.extend(["--"+key.replace("_", "-"), ",".join(map(str,value)) if isinstance(value,list) else str(value)])
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cli = json.loads(result.stdout)
    assert native["converged"] and native["provenance"] == "analytic"
    assert native["response_order"] == 2
    coefficients = np.asarray(native["field_energy_coefficients"])
    assert coefficients.shape == (3,) and np.isfinite(coefficients).all()
    np.testing.assert_allclose(coefficients, np.asarray(native["target_energy_coefficients"])
                               + np.asarray(native["pt2_energy_coefficients"]), atol=1e-12)
    assert abs(native["pt2_energy_coefficients"][2]) > 1e-9
    assert sum(native["energy_components"].values())==pytest.approx(native["energy"],abs=1e-11)
    if model=="wb97m2":
        assert native["vv10_b"]==6.0 and native["target_nonlocal_b"]==10.0
        assert native["target_nonlocal_coefficient"]==0.65904
        assert abs(native["target_nonlocal_coefficients"][2])>1e-12
    for report in (generic, adapted, cli):
        np.testing.assert_allclose(report["field_energy_coefficients"], coefficients, rtol=1e-8, atol=1e-11)
        np.testing.assert_allclose(report["dipole"], native["dipole"], rtol=1e-8, atol=1e-11)
        assert report["units"] == native["units"]
        assert report["model"] == native["model"]
    for key in ("response_residuals", "amplitude_response_residuals", "projector_residuals",
                "stationarity_residuals", "metric_residuals", "response_number_residuals",
                "orbital_frame_residuals"):
        assert np.max(np.abs(native[key])) < 1e-9
    converted_controls = dict(controls, field=np.asarray(controls["field"])*Hartree/Bohr,
                              field_direction=np.asarray(controls["field_direction"])*Hartree/Bohr,
                              overlap_cutoff=controls["overlap_cutoff"]*Bohr,
                              conv_tol=controls["conv_tol"]*Hartree,
                              conv_tol_grad=controls["conv_tol_grad"]*Hartree)
    if "vv10_cutoff" in controls:
        converted_controls["vv10_cutoff"]=controls["vv10_cutoff"]*Bohr
    # The direction is a field increment per dimensionless Taylor parameter.
    converted = calc.get_periodic_efield(atoms, units="ase", **converted_controls)
    np.testing.assert_allclose(converted["field_energy_coefficients"], coefficients*Hartree, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(converted["dipole"], np.asarray(native["dipole"])*Bohr, rtol=1e-8, atol=1e-10)
    assert not any(label == "unspecified" for label in native["units"].values())
