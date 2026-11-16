# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Units of the ASE calculator: controls are given in eV and Angstrom and
reports come back in eV and Angstrom, both from the unit labels of the common
driver."""

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree

from dft_rs import DftRs
from dft_rs import ase_calculator as adapter

EDGE = 6.0
HALF = 0.5 * EDGE
CELL = [[0.0, HALF, HALF], [HALF, 0.0, HALF], [HALF, HALF, 0.0]]
SETTINGS = dict(xc="svwn5", basis="sto-3g", kpts=(2, 2, 2), grid_level=2, grid_shell=1, guess="core")


def hydrogen():
    return Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


def helium():
    return Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=np.asarray(CELL) * Bohr, pbc=True)


def assert_labelled(report):
    """Every real number of a report is covered by a unit label."""
    units = report["units"]

    def visit(key, path, value):
        if path in units or key in units:
            return
        if isinstance(value, dict):
            for child, item in value.items():
                visit(child, f"{path}.{child}", item)
        elif isinstance(value, list):
            for item in value:
                visit(key, path, item)
        elif isinstance(value, float):
            raise AssertionError(f"{path} has no unit label")

    for key, value in report.items():
        if key not in ("units", "unit_system"):
            visit(key, key, value)


def assert_ase_units(report):
    """Labelled throughout, and no label is left in Hartree or Bohr except
    for entries whose name carries that unit."""
    assert_labelled(report)
    for key, label in report["units"].items():
        if isinstance(label, str) and not key.endswith(("_hartree", "_bohr")):
            assert adapter._ase_unit(label) is None, (key, label)


def assert_close(left, right, path=""):
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            if key not in ("units", "unit_system"):
                assert_close(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_close(a, b, f"{path}[{index}]")
    elif isinstance(left, float):
        assert left == pytest.approx(right, rel=1e-6, abs=1e-7), path
    else:
        assert left == right, path


def test_labels_convert_from_their_leading_expression():
    factors, renamed = adapter._ase_unit("hartree/bohr^2 per cell")
    assert adapter._factor(factors, {}) == pytest.approx(Hartree / Bohr**2)
    assert renamed == "eV/angstrom^2 per cell"
    factors, renamed = adapter._ase_unit("hartree*bohr/hbar (atomic units of velocity)")
    assert adapter._factor(factors, {}) == pytest.approx(Hartree * Bohr)
    assert renamed == "eV*angstrom/hbar (atomic units of velocity)"
    factors, renamed = adapter._ase_unit("1/bohr, Cartesian")
    assert adapter._factor(factors, {}) == pytest.approx(1.0 / Bohr)
    assert renamed == "1/angstrom, Cartesian"
    factors, _ = adapter._ase_unit("hartree/e/bohr")
    assert adapter._factor(factors, {}) == pytest.approx(Hartree / Bohr)
    factors, _ = adapter._ase_unit("electrons*hartree/hbar/bohr^2")
    assert adapter._factor(factors, {}) == pytest.approx(Hartree / Bohr**2)
    # A power can name an entry of the same object.
    factors, renamed = adapter._ase_unit("e*bohr^(rank-1); rows monomials")
    assert adapter._factor(factors, {"rank": 3}) == pytest.approx(Bohr**2)
    assert adapter._factor(factors, {}) is None
    assert renamed == "e*angstrom^(rank-1); rows monomials"
    # Nothing to convert, or no unit expression at all.
    for label in (
        "cm^-1 (negative = imaginary)",
        "K",
        "dimensionless",
        "GPa (Voigt)",
        "atomic units of magnetic field",
        "J/(mol K) per mol of primitive cells",
        "",
    ):
        assert adapter._ase_unit(label) is None, label


def test_continuous_charge_controls_preserve_chart_units_and_convert_origin():
    controls = dict(fixed_electron_count=14, physical_nuclear_charges=[6.75, 7.25],
                    charge_directions=[[-1.0, 1.0]], charge_coordinates=[0.25],
                    charge_bounds=[[-1.5, 1.5]], charge_scales=[0.5],
                    charge_origin=[0.2, -0.3, 0.4])
    converted = adapter._controls_to_atomic(controls, "continuous-species")
    for key in controls:
        if key != "charge_origin":
            assert converted[key] == controls[key]
    np.testing.assert_allclose(converted["charge_origin"],
                               np.asarray(controls["charge_origin"]) / Bohr)


def test_reports_convert_nested_entries_and_columns_and_keep_named_units():
    report = {
        "energy": -1.5,
        "iterations": 7,
        "rows": [
            {"temperature_k": 300.0, "chemical_potential": 0.25, "band": 2},
            {"temperature_k": 600.0, "chemical_potential": 0.5, "band": 3},
        ],
        "history": [[1.0, 0.5, 2.0], [2.0, 0.25, 1.0]],
        "ranks": [{"rank": 1, "cartesian": [1.0, 2.0]}, {"rank": 2, "cartesian": [3.0]}],
        "scattering_hartree": 0.001,
        "frequencies_cm1": [100.0],
        "units": {
            "energy": "hartree per cell",
            "rows.temperature_k": "K",
            "chemical_potential": "hartree",
            "history": "[dimensionless, hartree, 1/bohr] columns: order, defect, wave",
            "ranks.cartesian": "e*bohr^rank",
            "scattering_hartree": "hartree",
            "frequencies_cm1": "cm^-1",
        },
    }
    converted = adapter._to_ase_units(report)
    assert converted["energy"] == pytest.approx(-1.5 * Hartree)
    assert converted["iterations"] == 7
    rows = converted["rows"]
    assert [row["chemical_potential"] for row in rows] == pytest.approx([0.25 * Hartree, 0.5 * Hartree])
    assert [row["temperature_k"] for row in rows] == [300.0, 600.0]
    assert [row["band"] for row in rows] == [2, 3]
    np.testing.assert_allclose(
        converted["history"], [[1.0, 0.5 * Hartree, 2.0 / Bohr], [2.0, 0.25 * Hartree, 1.0 / Bohr]]
    )
    np.testing.assert_allclose(converted["ranks"][0]["cartesian"], [Bohr, 2.0 * Bohr])
    np.testing.assert_allclose(converted["ranks"][1]["cartesian"], [3.0 * Bohr**2])
    assert [entry["rank"] for entry in converted["ranks"]] == [1, 2]
    assert converted["scattering_hartree"] == 0.001
    assert converted["frequencies_cm1"] == [100.0]
    units = converted["units"]
    assert units["energy"] == "eV per cell" and units["chemical_potential"] == "eV"
    assert units["history"].startswith("[dimensionless, eV, 1/angstrom]")
    assert units["ranks.cartesian"] == "e*angstrom^rank"
    assert units["scattering_hartree"] == "hartree"
    assert_ase_units(converted)
    # The given report is left as it was.
    assert report["energy"] == -1.5 and report["units"]["energy"] == "hartree per cell"
    assert report["rows"][0]["chemical_potential"] == 0.25


def test_controls_are_given_in_ase_units():
    atomic = adapter._controls_to_atomic(
        {
            "overlap_cutoff": 12.0 * Bohr,
            "sigma": 0.02 * Hartree,
            "omega": 0.3 / Bohr,
            "gtol": 1e-4 * Hartree / Bohr,
            "stress_tol": 2e-6 * Hartree / Bohr**3,
            "field": [0.0, 0.0, 0.01 * Hartree / Bohr],
            "exchange_coefficients": [0.2, 0.1, 0.4 / Bohr],
            "origin": np.array([0.0, 0.0, 1.0]) * Bohr,
            "step_max": 1,
            "transport_frequencies_hartree": [0.1],
            "pressure_gpa": 5.0,
            "k_mesh": [2, 1, 1],
            "three_body": True,
            "xc": "pbe",
        }
    )
    assert atomic["overlap_cutoff"] == pytest.approx(12.0)
    assert atomic["sigma"] == pytest.approx(0.02)
    assert atomic["omega"] == pytest.approx(0.3)
    assert atomic["gtol"] == pytest.approx(1e-4)
    assert atomic["stress_tol"] == pytest.approx(2e-6)
    assert atomic["field"] == pytest.approx([0.0, 0.0, 0.01])
    assert atomic["exchange_coefficients"] == pytest.approx([0.2, 0.1, 0.4])
    assert atomic["origin"] == pytest.approx([0.0, 0.0, 1.0])
    # An integer is a length like any other number.
    assert atomic["step_max"] == pytest.approx(1.0 / Bohr)
    # A name that carries its unit, other units and non-numeric controls.
    assert atomic["transport_frequencies_hartree"] == [0.1]
    assert atomic["pressure_gpa"] == 5.0
    assert atomic["k_mesh"] == [2, 1, 1] and atomic["three_body"] is True and atomic["xc"] == "pbe"


def test_every_control_has_a_unit_label():
    units = adapter._control_units()
    assert len(units) > 150
    for name in ("conv_tol", "overlap_cutoff", "sigma", "k_mesh", "xc", "three_body", "pressure_gpa"):
        assert isinstance(units[name], str) and units[name], name


def test_molecular_reports_are_in_ase_units_from_ase_unit_controls():
    atoms = hydrogen()
    ase = DftRs(basis="sto-3g", xc="pbe", conv_tol=1e-10 * Hartree)
    atomic = DftRs(basis="sto-3g", xc="pbe", conv_tol=1e-10, units="atomic")
    cases = (
        ("get_gradient_report", {}, {}),
        ("get_dipole_report", {"origin": [0.0, 0.0, 0.2]}, {"origin": [0.0, 0.0, 0.2 / Bohr]}),
        ("get_frequencies_report", {}, {}),
        ("get_molecular_multipole", {"max_rank": 2}, {"max_rank": 2}),
        ("get_polarizability_report", {}, {}),
    )
    for getter, given, atomic_given in cases:
        converted = getattr(ase, getter)(atoms, **given)
        reference = getattr(atomic, getter)(atoms, **atomic_given)
        assert_labelled(reference)
        assert_ase_units(converted)
        assert_close(converted, adapter._to_ase_units(reference), getter)
    energy = ase.get_energy_report(atoms)
    assert energy["energy"] == pytest.approx(atomic.get_energy_report(atoms)["energy"] * Hartree, rel=1e-9)
    assert energy["units"]["energy"].startswith("eV")
    # The ASE properties are in eV and Angstrom in either mode.
    atoms.calc = atomic
    assert atoms.get_potential_energy() == pytest.approx(energy["energy"], rel=1e-9)


def test_periodic_reports_are_in_ase_units_from_ase_unit_controls():
    atoms = helium()
    # A cutoff away from every lattice shell (12 Bohr is one): the conversion
    # to Angstrom and back may then round it either way.
    ase = DftRs(overlap_cutoff=12.1 * Bohr, conv_tol=1e-11 * Hartree, **SETTINGS)
    atomic = DftRs(overlap_cutoff=12.1, conv_tol=1e-11, units="atomic", **SETTINGS)
    cases = (
        ("get_periodic_energy_report", dict(forces=True, stress=True), dict(forces=True, stress=True)),
        ("get_dos", dict(sigma=0.05 * Hartree, points=40), dict(sigma=0.05, points=40)),
        ("get_electronic_structure", {}, {}),
    )
    for getter, given, atomic_given in cases:
        converted = getattr(ase, getter)(atoms, **given)
        reference = getattr(atomic, getter)(atoms, **atomic_given)
        assert_labelled(reference)
        assert_ase_units(converted)
        assert_close(converted, adapter._to_ase_units(reference), getter)
    report = ase.get_periodic_energy_report(atoms, stress=True)
    assert report["units"]["stress_voigt"].startswith("eV/angstrom^3")
    np.testing.assert_allclose(report["stress_voigt"], atoms_stress(atoms, ase), rtol=1e-7, atol=1e-10)
    # One call can opt out: its controls and its report are atomic.
    plain = ase.get_dos(atoms, sigma=0.05, points=40, units="atomic")
    assert_close(plain, atomic.get_dos(atoms, sigma=0.05, points=40), "units=atomic")
    assert plain["units"]["sigma"] == "hartree"


def atoms_stress(atoms, calculator):
    atoms = atoms.copy()
    atoms.calc = calculator
    return atoms.get_stress().tolist()


def assert_native_labels(result):
    """Every real number of a dedicated result is covered by a label that
    some table knows."""
    labels = result["unit_labels"]

    def visit(key, path, value):
        if isinstance(value, dict) and "unit_labels" in value:
            assert_native_labels(value)
        elif path in labels:
            assert labels[path] != "unspecified", path
        elif isinstance(value, dict):
            for child, item in value.items():
                visit(child, f"{path}.{child}", item)
        elif isinstance(value, list):
            for item in value:
                visit(key, path, item)
        elif isinstance(value, float):
            raise AssertionError(f"{path} has no unit label")

    for key, value in result.items():
        if key not in ("units", "unit_labels", "unit_system"):
            visit(key, key, value)


def test_dedicated_results_carry_unit_labels():
    import dft_rs

    numbers = [1, 1]
    positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74 / Bohr]]
    scf = dft_rs.calculate(numbers, positions, basis="sto-3g", xc="pbe", gradient=True, wavefunction=True)
    assert_native_labels(scf)
    assert scf.unit("energy") == "hartree" and scf.unit("gradient") == "hartree/bohr"
    assert scf.unit("wavefunction.mo_energy") == "hartree"
    for result in (
        dft_rs.hessian(numbers, positions, basis="sto-3g", xc="pbe"),
        dft_rs.vibrations(numbers, positions, basis="sto-3g", xc="pbe"),
        dft_rs.polarizability(numbers, positions, basis="sto-3g", xc="pbe"),
        dft_rs.dynamic_polarizability(numbers, positions, basis="sto-3g", xc="pbe", omega=0.1),
        dft_rs.static_electric_tensor(numbers, positions, basis="sto-3g", xc="pbe", order=2),
        dft_rs.nmr(numbers, positions, basis="sto-3g", xc="pbe"),
        dft_rs.tddft(numbers, positions, basis="sto-3g", xc="pbe", nstates=1),
    ):
        assert_native_labels(result)
    beta = dft_rs.static_electric_tensor(numbers, positions, basis="sto-3g", xc="pbe", order=2)
    assert beta.unit("tensor") == "e^(order+1)*bohr^(order+1)/hartree^order"
    # A dictionary of flags has nothing to label.
    assert "unit_labels" not in dft_rs.periodic_capabilities()


def test_dedicated_getters_are_in_ase_units():
    atoms = hydrogen()
    ase = DftRs(basis="sto-3g", xc="pbe", conv_tol=1e-10 * Hartree)
    atomic = DftRs(basis="sto-3g", xc="pbe", conv_tol=1e-10, units="atomic")
    converted = ase.get_dynamic_polarizability(atoms, omega=0.1 * Hartree, damping=0.02 * Hartree)
    reference = atomic.get_dynamic_polarizability(atoms, omega=0.1, damping=0.02)
    factor = Bohr**2 / Hartree
    np.testing.assert_allclose(
        converted["alpha_real"], np.asarray(reference["alpha_real"]) * factor, rtol=1e-6, atol=1e-10
    )
    assert converted["omega"] == pytest.approx(0.1 * Hartree)
    assert converted["unit_labels"]["alpha_real"] == "e^2*angstrom^2/eV"
    assert converted["units"] == reference["units"]
    # One call can opt out.
    plain = ase.get_dynamic_polarizability(atoms, omega=0.1, damping=0.02, units="atomic")
    np.testing.assert_allclose(plain["alpha_real"], reference["alpha_real"], rtol=1e-6, atol=1e-10)
    # A tensor whose unit depends on its order.
    beta = ase.get_static_electric_tensor(atoms, order=2)
    beta_atomic = atomic.get_static_electric_tensor(atoms, order=2)
    np.testing.assert_allclose(
        beta["tensor"], np.asarray(beta_atomic["tensor"]) * Bohr**3 / Hartree**2, rtol=1e-6, atol=1e-10
    )
    # Shieldings are in ppm on every surface; the origin is a length.
    shielding = ase.get_nmr(atoms, gauge="common", origin=[0.0, 0.0, 0.37])
    shielding_atomic = atomic.get_nmr(atoms, gauge="common", origin=[0.0, 0.0, 0.37 / Bohr])
    np.testing.assert_allclose(
        shielding["shieldings"][0]["isotropic"], shielding_atomic["shieldings"][0]["isotropic"], rtol=1e-6
    )
    # The diagnostic blocks of the ASE results follow the calculator.
    atoms.calc = ase
    energy = atoms.get_potential_energy()
    scf = ase.results["scf"]
    assert scf["units"] == "ASE (eV)"
    atoms.calc = atomic
    assert atoms.get_potential_energy() == pytest.approx(energy, rel=1e-9)
    assert atomic.results["scf"]["units"] == "atomic (Hartree)"
    assert scf["e_nuc"] == pytest.approx(atomic.results["scf"]["e_nuc"] * Hartree, rel=1e-9)
