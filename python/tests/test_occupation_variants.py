# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""The common occupation controls preserve energies, forces and diagnostics."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree

import dft_rs


def cli_report(tmp_path, symbols, positions, feature, controls):
    path = tmp_path / "selected.xyz"
    lines = [str(len(symbols)), "selected determinant"]
    lines += ["%s %.16f %.16f %.16f" % (symbol, *(np.asarray(position) * Bohr))
              for symbol, position in zip(symbols, positions)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    arguments = [os.environ["DFT_RS_CLI"], feature, str(path), "--format", "json"]
    for key, value in controls.items():
        if isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, list):
            value = json.dumps(value)
        arguments += ["--" + key.replace("_", "-"), str(value)]
    completed = subprocess.run(arguments, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "[scf] iter" in completed.stderr and "SCF residual" in completed.stderr
    return json.loads(completed.stdout)


@pytest.mark.parametrize("xc,mode,unrestricted", [
    ("hf", "mom", False), ("pbe", "imom", True), ("tpss", "imom", False),
    ("pbe0", "mom", True), ("cam-b3lyp", "imom", False),
])
def test_selected_energy_and_gradient_on_all_surfaces(tmp_path, xc, mode, unrestricted):
    positions = [[0, 0, -0.7], [0, 0, 0.7]]
    controls = dict(xc=xc, basis="sto-3g", grid_level=4, dispersion="none",
                    conv_tol=1e-13, conv_tol_grad=1e-11, conv_tol_dp=1e-12,
                    max_iter=500, occupation_mode=mode, occupied_alpha=[1],
                    occupied_beta=[0 if unrestricted else 1], unrestricted=unrestricted,
                    field=[0, 0, 0.0003])
    native = dft_rs.gradient([1, 1], positions, **controls)
    assert native["converged"]
    tracking = native["occupation_tracking"]
    assert tracking["mode"] == mode
    assert tracking["density_residual"] < 1e-11
    assert tracking["energy_difference"] > 0.01
    assert native["units"]["initial_energy"] == "hartree"
    assert abs(native["energy"] - tracking["initial_energy"] - tracking["energy_difference"]) < 1e-12
    atoms = Atoms("H2", positions=np.asarray(positions) * Bohr)
    atoms.calc = dft_rs.DftRs(units="atomic", **controls)
    np.testing.assert_allclose(atoms.get_forces(), -np.asarray(native["gradient"]) * Hartree / Bohr, atol=1e-8)
    np.testing.assert_allclose(atoms.get_potential_energy(), native["energy"] * Hartree, atol=1e-9)
    assert atoms.calc.results["scf"]["occupation_tracking"]["density_residual"] < 1e-11
    adapted = dft_rs.DftRs(units="atomic").get_gradient_report(atoms, **controls)
    physical = dft_rs.DftRs().get_gradient_report(atoms, **dict(controls, field=np.asarray(controls["field"]) * Hartree / Bohr))
    np.testing.assert_allclose(adapted["gradient"], native["gradient"], atol=1e-9)
    np.testing.assert_allclose(np.asarray(physical["gradient"]) / (Hartree / Bohr), native["gradient"], atol=1e-9)
    np.testing.assert_allclose(physical["occupation_tracking"]["initial_energy"] / Hartree, tracking["initial_energy"], atol=1e-10)
    np.testing.assert_allclose(physical["occupation_tracking"]["overlap_weights"], tracking["overlap_weights"], atol=1e-10)
    cli = cli_report(tmp_path, ["H", "H"], positions, "gradient", controls)
    np.testing.assert_allclose(cli["gradient"], native["gradient"], atol=1e-9)
    np.testing.assert_allclose(cli["occupation_tracking"]["energy_difference"], tracking["energy_difference"], atol=1e-10)
    print("occupation public", xc, mode, unrestricted, "map", tracking["density_residual"], "energy_difference", tracking["energy_difference"])


def test_water_selected_state_and_invalid_constraints(tmp_path):
    positions = [[0, 0, 0.1173], [0, 1.4332, -0.9314], [0, -1.4332, -0.9314]]
    controls = dict(xc="pbe0", basis="sto-3g", grid_level=4, conv_tol=1e-12,
                    conv_tol_grad=1e-9, conv_tol_dp=1e-10, max_iter=500, dispersion="none",
                    occupation_mode="imom", occupied_alpha=[0, 1, 2, 3, 5], occupied_beta=[0, 1, 2, 3, 4], unrestricted=True)
    report = dft_rs.energy([8, 1, 1], positions, **controls)
    assert report["converged"] and report["occupation_tracking"]["energy_difference"] > 0.01
    assert report["occupation_tracking"]["density_residual"] < 1e-10
    atoms = Atoms("OH2", positions=np.asarray(positions) * Bohr)
    atomic = dft_rs.DftRs(units="atomic").get_energy_report(atoms, **controls)
    physical = dft_rs.DftRs().get_energy_report(atoms, **controls)
    cli = cli_report(tmp_path, ["O", "H", "H"], positions, "energy", controls)
    for other in (atomic, cli):
        np.testing.assert_allclose(other["energy"], report["energy"], atol=1e-9)
        np.testing.assert_allclose(other["occupation_tracking"]["overlap_weights"], report["occupation_tracking"]["overlap_weights"], atol=1e-8)
    np.testing.assert_allclose(physical["energy"] / Hartree, report["energy"], atol=1e-9)
    print("water selected", report["energy"], report["occupation_tracking"])
    for override, pattern in [
        ({"unrestricted": False}, "unrestricted reference"),
        ({"occupied_alpha": [0, 0, 2, 3, 5]}, "repeated"),
        ({"occupied_alpha": [0, 1, 2, 3, 5.5]}, "non-negative integers"),
        ({"electronic_temperature": 1000}, "Fermi smearing"),
    ]:
        with pytest.raises(ValueError, match=pattern):
            dft_rs.energy([8, 1, 1], positions, **dict(controls, **override))
