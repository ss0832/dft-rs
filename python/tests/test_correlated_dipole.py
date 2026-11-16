# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""The existing dipole API includes the correlated electric derivative."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM

NUMBERS = [8, 1, 1]
POSITIONS = [[0.0, 0.0, 0.2217], [0.0, 1.4309, -0.8867], [0.0, -1.4309, -0.8867]]
# Route equality uses a compact quadrature. The independent derivative
# acceptance uses the finer grids documented in response.md.
BASE = dict(basis="sto-3g", grid_level=3, vv10_grid_level=0, conv_tol=1e-13,
            conv_tol_grad=1e-11, conv_tol_dp=1e-12)


@pytest.mark.parametrize("xc", ["b2plyp", "dsd-pbep86", "dsd-blyp", "pwpb95", "wb97m2"])
def test_correlated_dipole_agrees_on_native_generic_ase_and_cli(tmp_path, xc):
    controls = dict(BASE, xc=xc)
    native = dft_rs.dipole(NUMBERS, POSITIONS, **controls)
    generic = dft_rs.run_feature("dipole", NUMBERS, POSITIONS, **controls)
    atoms = Atoms("OH2", positions=np.asarray(POSITIONS) * Bohr)
    atoms.calc = dft_rs.DftRs(units="atomic", **controls)
    adapted = atoms.calc.get_dipole_report(atoms, units="atomic")
    xyz = tmp_path / "water.xyz"
    xyz.write_text("3\ncorrelated dipole\n" + "".join(
        f"{symbol} " + " ".join(f"{x*BOHR_IN_ANGSTROM:.16f}" for x in p) + "\n"
        for symbol, p in zip(["O", "H", "H"], POSITIONS)), encoding="utf-8")
    command = [os.environ["DFT_RS_CLI"], "dipole", str(xyz), "--xc", xc,
               "--basis", "sto-3g", "--grid-level", "3", "--vv10-grid-level", "0",
               "--conv-tol", "1e-13", "--conv-tol-grad", "1e-11", "--conv-tol-dp", "1e-12",
               "--format", "json"]
    cli = subprocess.run(command, text=True, capture_output=True)
    assert cli.returncode == 0, cli.stderr
    for other in [generic, adapted, json.loads(cli.stdout)]:
        assert other["units"] == native["units"]
        for key in ["dipole", "dipole_norm", "dipole_debye", "dipole_norm_debye",
                    "parent_dipole", "pt2_dipole_correction", "target_dipole_correction",
                    "response_residuals", "amplitude_response_residuals", "response_number_residuals"]:
            np.testing.assert_allclose(other[key], native[key], atol=2e-10, rtol=1e-10)
    total = sum(np.asarray(native[k]) for k in ["parent_dipole", "pt2_dipole_correction", "target_dipole_correction"])
    np.testing.assert_allclose(native["dipole"], total, atol=1e-14, rtol=0)
    np.testing.assert_allclose(atoms.get_dipole_moment(), np.asarray(native["dipole"]) * Bohr, atol=2e-10, rtol=1e-10)
    converted = atoms.calc.get_dipole_report(atoms, units="ase")
    np.testing.assert_allclose(converted["dipole"], np.asarray(native["dipole"]) * Bohr, atol=2e-10, rtol=1e-10)
    for key in ["response_residuals", "amplitude_response_residuals", "response_number_residuals"]:
        assert max(abs(x) for x in native[key]) < 1e-9
    if xc == "wb97m2":
        assert abs(native["target_dipole_correction"][2]) > 1e-5
    else:
        assert native["target_dipole_correction"] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("xc", ["pbe", "tpss", "pbe0"])
def test_zero_correlation_dipole_and_fixed_number_origin_limit(xc):
    # The water-dication fixture did not converge under these SCF controls.
    # This bound two-electron cation has a virtual orbital in both channels.
    numbers, positions = [2, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]
    controls = dict(BASE, xc=xc, charge=1)
    parent = dft_rs.dipole(numbers, positions, origin="zero", **controls)
    correlated = dft_rs.dipole(numbers, positions, origin="zero", pt2_same_spin=0.0,
                              pt2_opposite_spin=0.0, **controls)
    np.testing.assert_allclose(correlated["dipole"], parent["dipole"], atol=1e-11, rtol=0)
    shift = np.asarray([0.2, -0.3, 0.4])
    shifted = dft_rs.dipole(numbers, positions, origin=shift.tolist(), pt2_same_spin=0.2,
                           pt2_opposite_spin=0.3, **controls)
    unshifted = dft_rs.dipole(numbers, positions, origin="zero", pt2_same_spin=0.2,
                             pt2_opposite_spin=0.3, **controls)
    np.testing.assert_allclose(shifted["dipole"], np.asarray(unshifted["dipole"]) - shift,
                               atol=1e-11, rtol=0)
    assert any("charged" in warning for warning in shifted["warnings"])
