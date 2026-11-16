# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""High-order dispersion contributions share all existing public routes.

This one-atom route fixture is the exact rigid-translation limit. Nonzero
two/three-atom tensors have separate independent four-step Rust acceptance.
"""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM

CELL = [[8.0, 0.0, 0.0], [0.4, 8.2, 0.0], [0.2, 0.3, 8.4]]
BASE = dict(xc="hf", basis="sto-3g", k_mesh=[1, 1, 1], overlap_cutoff=5.0,
            grid_level=0, grid_shell=1, conv_tol=1e-12, conv_tol_dp=1e-11,
            dispersion="d3bj", disp_cutoff=10.0, cn_cutoff=10.0)


@pytest.mark.parametrize("feature,order,along", [
    ("periodic-cubic-fc-along", 3, True), ("periodic-quartic-fc-along", 4, True),
    ("periodic-cubic-fc", 3, False), ("periodic-quartic-fc", 4, False)])
def test_periodic_dispersion_high_order_common_routes(tmp_path, feature, order, along):
    controls = dict(BASE)
    if along:
        controls["probe"] = [0.3, -0.2, 0.1]
        if order == 4:
            controls["second_probe"] = [-0.1, 0.2, 0.3]
    function = getattr(dft_rs, feature.replace("-", "_"))
    native = function([2], [[0.0, 0.0, 0.0]], CELL, **controls)
    generic = dft_rs.run_feature(feature, [2], [[0.0, 0.0, 0.0]], cell=CELL, **controls)
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=np.asarray(CELL)*Bohr, pbc=True)
    calculator = dft_rs.DftRs(units="atomic", **BASE)
    patterns = {k: v for k, v in controls.items() if k in ["probe", "second_probe"]}
    adapted = getattr(calculator, "get_" + feature.replace("-", "_"))(
        atoms, units="atomic", **patterns)
    xyz = tmp_path / "helium.xyz"
    xyz.write_text("1\nperiodic dispersion\nHe 0 0 0\n", encoding="utf-8")
    command = [os.environ["DFT_RS_CLI"], feature, str(xyz), "--cell",
        ";".join(",".join(f"{x*BOHR_IN_ANGSTROM:.16f}" for x in row) for row in CELL),
        "--xc", "hf", "--basis", "sto-3g", "--overlap-cutoff", "5", "--grid-level", "0",
        "--grid-shell", "1", "--conv-tol", "1e-12", "--conv-tol-dp", "1e-11",
        "--dispersion", "d3bj", "--disp-cutoff", "10", "--cn-cutoff", "10", "--format", "json"]
    if along:
        command.extend(["--probe", "0.3,-0.2,0.1"])
        if order == 4:
            command.extend(["--second-probe=-0.1,0.2,0.3"])
    cli = subprocess.run(command, capture_output=True, text=True)
    assert cli.returncode == 0, cli.stderr
    key = "contracted_force_constants" if along else "force_constant_tensor"
    extra = "dispersion_force_constants" if along else "dispersion_force_constant_tensor"
    assert native["dispersion_energy"] < 0.0
    assert native["dispersion_energy_per_primitive_cell"] == native["dispersion_energy"]
    reports = [native, generic, adapted, json.loads(cli.stdout)]
    if order == 3 and along:
        reports.append(calculator.get_feature_report(feature, atoms, units="atomic", **patterns))
    for report in reports:
        for name in [key, extra]:
            assert np.max(np.abs(report[name])) < 1e-10
            np.testing.assert_allclose(report[name], native[name], atol=1e-12, rtol=0)
        np.testing.assert_allclose(report["energy"], native["energy"], atol=1e-11, rtol=0)
        assert report["dispersion_acoustic_residual"] < 1e-11
        assert report["units"] == native["units"]
        assert report["units"][extra].startswith(f"hartree/bohr^{order}")
