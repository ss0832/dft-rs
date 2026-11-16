# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Structure objects, calculator objects and reports of the native layer."""

import copy
import json
import pickle

import numpy as np
import pytest

import dft_rs
from dft_rs import Crystal, MolecularCalculator, Molecule, PeriodicCalculator, Report

WATER = [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]
METHOD = dict(xc="hf", basis="sto-3g")


def test_report_is_a_dictionary_with_attribute_access():
    water = Molecule.from_angstrom([8, 1, 1], WATER)
    report = dft_rs.energy(water.numbers, water.positions, **METHOD)
    assert isinstance(report, dict) and isinstance(report, Report)
    assert report.energy == report["energy"]
    assert report.units is report["units"]
    assert "energy" in dir(report)
    assert report.unit("not an entry") is None
    assert report.require_converged() is report
    with pytest.raises(AttributeError):
        report.not_an_entry
    assert copy.deepcopy(report) == report
    assert pickle.loads(pickle.dumps(report)) == report
    assert json.loads(json.dumps(report))["energy"] == report.energy
    with pytest.raises(RuntimeError):
        Report(converged=False, feature="energy").require_converged()


def test_molecule_and_calculator_match_the_function_interface(tmp_path):
    water = Molecule.from_angstrom([8, 1, 1], WATER)
    assert water.natoms == 3
    calc = MolecularCalculator(water, **METHOD)
    molecular = MolecularCalculator.features()
    periodic = PeriodicCalculator.features()
    assert molecular["energy"] == "energy" and "phonons" not in molecular
    assert "phonons" in periodic and "gradient" not in periodic
    assert "gradient" in dir(calc)
    direct = dft_rs.energy(water.numbers, water.positions, **METHOD)
    assert abs(calc.energy().energy - direct["energy"]) < 1e-12
    assert abs(dft_rs.run_feature("energy", water, **METHOD).energy - direct["energy"]) < 1e-12
    gradient = np.asarray(calc.gradient().gradient)
    assert gradient.shape == (3, 3)
    assert np.abs(gradient.sum(axis=0)).max() < 1e-8
    # One control overridden for one call, the stored controls unchanged.
    assert calc.energy(basis="3-21g").basis == "3-21g" and calc.controls["basis"] == "sto-3g"
    path = tmp_path / "water.xyz"
    path.write_text("3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\n1 0 -0.7572 -0.4692\n")
    read = Molecule.from_xyz(path)
    assert read.numbers == (8, 1, 1)
    np.testing.assert_allclose(read.positions, water.positions, atol=1e-12)
    np.testing.assert_allclose(Molecule.from_ase(water.to_ase()).positions, water.positions, atol=1e-12)
    moved = water.displaced(1, 2, 0.01)
    assert abs(moved.positions[1][2] - water.positions[1][2] - 0.01) < 1e-15
    assert isinstance(dft_rs.calculator(water), MolecularCalculator)
    with pytest.raises(TypeError):
        MolecularCalculator(Crystal([2], [[0.0, 0.0, 0.0]], np.eye(3) * 6.0))
    with pytest.raises(TypeError):
        dft_rs.calculator(([8, 1, 1], WATER))
    with pytest.raises(AttributeError):
        calc.phonons


def test_crystal_reads_extended_xyz_and_the_registry_is_listed(tmp_path):
    path = tmp_path / "helium.xyz"
    path.write_text(
        '1\nLattice="0.0 1.5875316327 1.5875316327 1.5875316327 0.0 1.5875316327 '
        '1.5875316327 1.5875316327 0.0" pbc="T T F"\nHe 0 0 0\n'
    )
    crystal = Crystal.from_xyz(path)
    assert crystal.pbc == (True, True, False)
    np.testing.assert_allclose(crystal.cell, [[0.0, 3.0, 3.0], [3.0, 0.0, 3.0], [3.0, 3.0, 0.0]], atol=1e-8)
    plain = tmp_path / "plain.xyz"
    plain.write_text("1\nhelium\nHe 0 0 0\n")
    with pytest.raises(ValueError):
        Crystal.from_xyz(plain)
    assert Molecule.from_xyz(plain).numbers == (2,)
    assert isinstance(dft_rs.calculator(crystal, xc="svwn5"), PeriodicCalculator)
    periodic = dft_rs.features("periodic")
    assert {"id", "title", "space", "route", "python"} <= set(periodic[0])
    assert "phonons" in {entry["id"] for entry in periodic}
    assert "gradient" not in {entry["id"] for entry in periodic}
    assert len(dft_rs.features()) > len(periodic)
    assert abs(dft_rs.units.HARTREE_IN_EV - dft_rs.native.HARTREE_IN_EV) < 1e-15
    assert abs(dft_rs.units.BOHR_IN_ANGSTROM * dft_rs.native.BOHR_PER_ANGSTROM - 1.0) < 1e-15
    assert abs(dft_rs.units.HARTREE_PER_BOHR3_IN_GPA - 29421.0156965221) < 1e-6


def test_dedicated_functions_return_reports_and_a_molecule_relaxes():
    stretched = Molecule([1, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]])
    single = dft_rs.calculate(stretched.numbers, stretched.positions, **METHOD)
    assert isinstance(single, Report) and single.energy == single["energy"]
    assert dft_rs.calculate.__name__ == "calculate"
    calc = MolecularCalculator(stretched, **METHOD)
    assert calc.molecule is stretched
    relaxed, result = calc.relax(gtol=1e-5)
    assert isinstance(relaxed, Molecule) and isinstance(result, Report)
    assert result.converged
    bond = np.linalg.norm(np.subtract(relaxed.positions[1], relaxed.positions[0]))
    assert 1.30 < bond < 1.40
    assert result.energy < single.energy
    after = calc.with_molecule(relaxed).gradient().gradient
    assert np.abs(after).max() < 5e-5
    # Lists and other results are passed through unchanged.
    assert isinstance(dft_rs.basis_sets(), list)


@pytest.mark.parametrize("number", [1.9, "1", 0, -1, 119])
@pytest.mark.parametrize("structure", [Molecule, Crystal])
def test_structures_reject_invalid_atomic_numbers_without_changing_element(number, structure):
    extra = (np.eye(3) * 6.0,) if structure is Crystal else ()
    with pytest.raises(ValueError, match="atomic numbers"):
        structure([number], [[0.0, 0.0, 0.0]], *extra)


@pytest.mark.parametrize("repeats", [(1.9, 1, 1), ("2", 1, 1), (0, 1, 1), (-1, 1, 1)])
def test_supercell_rejects_invalid_repeats_without_truncation(repeats):
    crystal = Crystal([2], [[0.0, 0.0, 0.0]], np.eye(3) * 6.0)
    with pytest.raises(ValueError, match="three positive integers"):
        crystal.supercell(repeats)


def test_numpy_integer_atomic_numbers_and_repeats_preserve_atom_order():
    crystal = Crystal(np.array([2, 1]), [[0.0, 0.0, 0.0], [0.2, 0.3, 0.4]], np.eye(3) * 6.0)
    repeated = crystal.supercell(np.array([2, 1, 1]))
    assert repeated.numbers == (2, 1, 2, 1)
    np.testing.assert_allclose(repeated.positions[2], [6.0, 0.0, 0.0])


@pytest.mark.parametrize("pbc", [["False", False, False], [2, 0, 0], [None, False, False]])
def test_crystal_rejects_ambiguous_periodicity(pbc):
    with pytest.raises(ValueError, match="periodicity"):
        Crystal([2], [[0.0, 0.0, 0.0]], np.eye(3), pbc=pbc)


@pytest.mark.parametrize("fractional", [[0.1, 0.2], [0.1, 0.2, 0.3, 9.0]])
def test_fractional_coordinates_require_exactly_three_components(fractional):
    with pytest.raises(ValueError, match="three components"):
        Crystal.from_fractional([2], [fractional], np.eye(3))


def test_feature_discovery_rejects_unknown_spaces():
    with pytest.raises(ValueError, match="space"):
        dft_rs.features("molecul")
