# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Prescribed populations use the shared energy/force driver on every surface."""
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree

import dft_rs

ORACLES = json.loads((Path(__file__).resolve().parents[2] /
                     "tests/oracle_data/fixed_populations.json").read_text(encoding="utf-8"))


def common(**extra):
    return dict(basis="sto-3g", dispersion="none", grid_level=4, max_iter=500,
                conv_tol=1e-13, conv_tol_grad=1e-10, conv_tol_dp=1e-11, **extra)


def cli_report(tmp_path, feature, atoms, controls, plain=False, assignment=False, *, spinor=False):
    path = tmp_path / "populations.xyz"
    lines = [str(len(atoms)), "prescribed populations"] + [
        "%s %.16f %.16f %.16f" % (symbol, *position)
        for symbol, position in zip(atoms.get_chemical_symbols(), atoms.positions)]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    arguments = [os.environ["DFT_RS_CLI"], feature, str(path)]
    if not plain:
        arguments += ["--format", "json"]
    for key, value in controls.items():
        flag = "--"+key.replace("_", "-")
        value = json.dumps(value) if isinstance(value, (list, bool)) else str(value)
        arguments += [flag+"="+value] if assignment else [flag, value]
    result = subprocess.run(arguments, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    if spinor:
        assert "[spinor-scf]" in result.stderr and "residual" in result.stderr
    else:
        assert "[scf] iter" in result.stderr and "SCF residual" in result.stderr
    return result.stdout if plain else json.loads(result.stdout)


@pytest.mark.parametrize("xc,unrestricted", [
    ("hf", False), ("svwn5", True), ("pbe", False),
    ("tpss", True), ("pbe0", False), ("cam-b3lyp", True),
])
def test_fixed_population_all_surfaces_and_units(tmp_path, xc, unrestricted):
    positions = [[0., 0., -0.7], [0., 0., 0.7]]
    controls = common(xc=xc, unrestricted=unrestricted, occupation_mode="fixed",
                      occupations_alpha=[0.8, 0.2], occupations_beta=[0.7, 0.3] if unrestricted else [0.8, 0.2], field=[0., 0., 0.0003])
    native = dft_rs.gradient([1, 1], positions, **controls)
    assert native["fixed_occupation_residual"] < 1e-11
    assert native["units"]["fixed_occupation_residual"] == "dimensionless"
    np.testing.assert_array_equal(native["occupations"], [controls["occupations_alpha"], controls["occupations_beta"]])
    assert "no thermal" in native["occupation_model"]
    atoms = Atoms("H2", positions=np.asarray(positions)*Bohr)
    atoms.calc = dft_rs.DftRs(units="atomic", **controls)
    np.testing.assert_allclose(atoms.get_forces(), -np.asarray(native["gradient"])*Hartree/Bohr, atol=1e-8, rtol=0.)
    np.testing.assert_allclose(atoms.get_potential_energy()/Hartree, native["energy"], atol=1e-10, rtol=0.)
    assert atoms.calc.results["scf"]["fixed_occupation_residual"] < 1e-11
    assert abs(atoms.get_charges().sum()) < 1e-10
    atomic = dft_rs.DftRs(units="atomic").get_gradient_report(atoms, **controls)
    physical = dft_rs.DftRs().get_gradient_report(atoms, **dict(controls, field=np.asarray(controls["field"])*Hartree/Bohr))
    cli = cli_report(tmp_path, "gradient", atoms, controls)
    text = cli_report(tmp_path, "energy", atoms, controls, plain=True)
    text_energy = re.search(r"^energy\s+([-+0-9.eE]+)\s*$", text, re.M)
    assert text_energy is not None, text
    np.testing.assert_allclose(float(text_energy.group(1)), native["energy"], atol=1e-10, rtol=0.)
    assert "fixed_occupation_residual" in text and "occupation_model" in text
    for report in [atomic, cli]:
        np.testing.assert_allclose(report["energy"], native["energy"], atol=1e-10, rtol=0.)
        np.testing.assert_allclose(report["gradient"], native["gradient"], atol=1e-9, rtol=0.)
        np.testing.assert_array_equal(report["occupations"], native["occupations"])
    np.testing.assert_allclose(np.asarray(physical["gradient"])/(Hartree/Bohr), native["gradient"], atol=1e-9, rtol=0.)
    np.testing.assert_allclose(physical["energy"]/Hartree, native["energy"], atol=1e-10, rtol=0.)
    np.testing.assert_allclose(physical["fixed_occupation_gap"]/Hartree, native["fixed_occupation_gap"], atol=1e-10, rtol=0.)
    orbitals = dft_rs.orbitals([1, 1], positions, **controls)
    assert all(channel["homo"] is None and channel["lumo"] is None and channel["gap"] is None for channel in orbitals["channels"])
    print("fixed public", xc, unrestricted, "energy", native["energy"], "map", native["fixed_occupation_residual"])


@pytest.mark.parametrize("mode", ["mom", "imom"])
def test_selected_determinant_plain_text_cli_keeps_occupation_controls(tmp_path, mode):
    positions = [[0., 0., -0.7], [0., 0., 0.7]]
    controls = common(xc="hf", occupation_mode=mode, occupied_alpha=[1], occupied_beta=[1])
    native = dft_rs.energy([1, 1], positions, **controls)
    atoms = Atoms("H2", positions=np.asarray(positions)*Bohr)
    text = cli_report(tmp_path, "energy", atoms, controls, plain=True, assignment=(mode == "imom"))
    matched = re.search(r"^energy\s+([-+0-9.eE]+)\s*$", text, re.M)
    assert matched is not None, text
    np.testing.assert_allclose(float(matched.group(1)), native["energy"], atol=1e-10, rtol=0.)
    assert "occupation_tracking.mode "+mode in text


@pytest.mark.parametrize("case", ORACLES["records"], ids=lambda case: case["name"])
def test_five_molecule_fixed_pbe_energy_components_and_dipole_oracles(case):
    controls = common(xc=case["functional"], occupation_mode="fixed", occupations_alpha=case["populations"], occupations_beta=case["populations"])
    numbers, positions = case["numbers"], case["positions"]
    moment = dft_rs.dipole(numbers, positions, **controls)
    ground = dft_rs.dipole(numbers, positions, **common(xc=case["functional"]))
    exact = case["fixed"]
    energy_error = abs(moment["energy"]-exact["energy"])
    dipole_error = float(np.max(np.abs(np.asarray(moment["dipole"])-exact["dipole"])))
    assert moment["fixed_occupation_residual"] < 1e-11
    assert energy_error < 2e-8
    assert dipole_error < 1e-8
    components = moment["energy_components"]
    np.testing.assert_allclose(components["exchange_correlation"], exact["exchange_correlation"], atol=2e-8, rtol=0.)
    np.testing.assert_allclose(components["nuclear_repulsion"], exact["nuclear"], atol=1e-10, rtol=0.)
    np.testing.assert_allclose(components["electronic"], exact["one_body"]+exact["hartree"]+exact["exchange_correlation"], atol=2e-8, rtol=0.)
    np.testing.assert_allclose(ground["energy"], case["ground"]["energy"], atol=2e-8, rtol=0.)
    np.testing.assert_allclose(ground["dipole"], case["ground"]["dipole"], atol=1e-8, rtol=0.)
    assert abs(sum(moment["occupations"][0])-sum(numbers)/2) < 1e-10
    print("five-molecule fixed oracle", case["name"], "energy", moment["energy"],
          "energy_error", energy_error, "dipole_error", dipole_error,
          "map", moment["fixed_occupation_residual"], "ground_dipole", ground["dipole"])


def test_fixed_population_limits_and_incompatible_constraints():
    positions = [[0., 0., -0.7], [0., 0., 0.7]]
    controls = common(xc="pbe0", occupation_mode="fixed", occupations_alpha=[1.], occupations_beta=[1., 0.])
    fixed = dft_rs.gradient([1, 1], positions, **controls)
    ground = dft_rs.gradient([1, 1], positions, **common(xc="pbe0"))
    np.testing.assert_allclose(fixed["gradient"], ground["gradient"], atol=1e-9, rtol=0.)
    np.testing.assert_allclose(fixed["energy"], ground["energy"], atol=1e-10, rtol=0.)
    fraction = dict(controls, occupations_alpha=[0.8, 0.2], occupations_beta=[0.8, 0.2])
    for override, pattern in [
        ({"occupations_alpha": [0.8, 0.1]}, "sum to"),
        ({"occupations_alpha": [1.1, -0.1]}, r"\[0,1\]"),
        ({"occupations_alpha": [0.8, 0.2, 0.]}, "retained MO rank"),
        ({"occupations_beta": [0.7, 0.3]}, "unrestricted"),
        ({"electronic_temperature": 1000}, "Fermi smearing"),
        ({"stabilize": True}, "stability descent"),
        ({"occupation_mode": "aufbau"}, "occupation_mode=fixed"),
        ({"occupied_alpha": [0], "occupied_beta": [0]}, "explicit occupied lists"),
        ({"xc": "b2plyp"}, "integer zero-temperature"),
    ]:
        with pytest.raises(ValueError, match=pattern):
            dft_rs.energy([1, 1], positions, **dict(fraction, **override))
    with pytest.raises(ValueError, match="thermal equilibrium"):
        dft_rs.mermin_free_energy([1, 1], positions, **fraction)
    response = dft_rs.polarizability_report([1, 1], positions, **fraction)
    assert "df = 0" in response["response_population_model"]
    assert max(response["response_residuals"]) < 1e-10
    atom = dft_rs.energy([1], [[0., 0., 0.]], **common(xc="hf", multiplicity=2,
        unrestricted=True, occupation_mode="fixed", occupations_alpha=[1.], occupations_beta=[]))
    assert atom["n_alpha"] == 1 and atom["n_beta"] == 0
    np.testing.assert_allclose(atom["s_squared"], 0.75, atol=1e-12, rtol=0.)
    assert atom["fixed_occupation_gap"] is None
