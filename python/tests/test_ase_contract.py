# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import re
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree

import dft_rs
from dft_rs import DftRs
from dft_rs import ase_calculator


@pytest.mark.parametrize("getter", ["get_dimensional_scf", "get_dimensional_field_response"])
def test_dimensional_getters_forward_all_grid_controls(monkeypatch, getter):
    seen = []

    def boundary(*args, **options):
        seen.append(options)
        return {"units": {}, "field_response": {"units": {}}}

    monkeypatch.setattr(ase_calculator.native._core, "dimensional_scf", boundary)
    atoms = Atoms("He", positions=[[0., 0., 0.]], cell=[7., 8., 9.], pbc=[1, 1, 0])
    calc = DftRs(units="atomic", grid_level=2, grid_preset="fine",
                 grid_angular_scheme="product", grid_weight_cutoff=0.)
    getattr(calc, getter)(atoms)
    assert seen[-1]["grid_level"] == 2
    assert seen[-1]["grid_preset"] == "fine"
    assert seen[-1]["grid_angular_scheme"] == "product"
    assert seen[-1]["grid_weight_cutoff"] == 0.
    getattr(calc, getter)(atoms, grid_level=1, grid_preset="ultrafine")
    assert seen[-1]["grid_level"] == 1
    assert seen[-1]["grid_preset"] == "ultrafine"


@pytest.mark.parametrize("name", ["grid_level", "vv10_grid_level"])
@pytest.mark.parametrize("value", [True, 1.5, -1, 10, "2"])
def test_dedicated_grid_levels_are_validated_without_integer_coercion(name, value):
    atoms = Atoms("H2", positions=[[0., 0., 0.], [0., 0., .74]])
    with pytest.raises(ValueError, match=name):
        calc = DftRs(**{name: value})
        calc.get_nuclear_taylor(atoms, direction=[[0., 0., 0.], [0., 0., 1.]], order=1)


@pytest.mark.parametrize("name", ["grid_level", "vv10_grid_level"])
@pytest.mark.parametrize("value", [False, 2.5, -1, 10, 2**100])
def test_native_response_grid_levels_fail_before_scf(name, value):
    with pytest.raises(ValueError, match=name):
        dft_rs.native.nuclear_taylor([1, 1], [[0., 0., 0.], [0., 0., 1.4]],
                                   [[0., 0., 0.], [0., 0., 1.]], **{name: value})


def test_grid_weight_cutoff_has_the_same_units_in_parameters_and_calls(monkeypatch):
    value = 2e-10
    expected = value / Bohr**3
    calc = DftRs(grid_weight_cutoff=value)
    assert calc._molecular_driver_controls()["grid_weight_cutoff"] == pytest.approx(expected)
    assert calc._native_options(grid_controls=True)["grid_weight_cutoff"] == pytest.approx(expected)
    seen = []
    original = ase_calculator.native._core.run_feature

    def boundary(feature, *args, **kwargs):
        if feature == "feature-registry":
            return original(feature, *args, **kwargs)
        seen.append(kwargs["grid_weight_cutoff"])
        return {"units": {}}

    monkeypatch.setattr(ase_calculator.native._core, "run_feature", boundary)
    atoms = Atoms("H2", positions=[[0., 0., 0.], [0., 0., .74]])
    calc.get_feature_report("polarizability", atoms)
    DftRs().get_feature_report("polarizability", atoms, grid_weight_cutoff=value)
    assert seen == pytest.approx([expected, expected])


@pytest.mark.parametrize("name", ["charge", "multiplicity", "max_iter", "incore_mb"])
@pytest.mark.parametrize("value", [1.9, True])
def test_integer_reference_controls_are_not_silently_rounded(name, value):
    with pytest.raises(ValueError, match=name):
        DftRs(**{name: value})._native_options()


@pytest.mark.parametrize("mesh", [(1.9, 1, 1), (True, 1, 1), (1, 1), (1, 0, 1)])
def test_periodic_mesh_controls_are_not_silently_rounded(mesh):
    with pytest.raises(ValueError, match="kpts/k_mesh"):
        DftRs(kpts=mesh)._k_mesh()


def test_numpy_integer_controls_remain_supported():
    calc = DftRs(charge=np.int64(-1), multiplicity=np.int64(2),
                 max_iter=np.int64(31), incore_mb=np.int64(128), kpts=np.array([2, 3, 1]))
    assert calc._native_options()["charge"] == -1
    assert calc._native_options()["max_iter"] == 31
    assert calc._k_mesh() == [2, 3, 1]


@pytest.mark.parametrize("name,value", [("multiplicity", True), ("charge", False), ("grid_level", 3.0)])
def test_ase_parameter_equality_does_not_hide_invalid_integer_types(name, value):
    calc = DftRs()
    with pytest.raises(ValueError, match=name):
        calc.set(**{name: value})


@pytest.mark.parametrize("name,value", [("adiis_until_error", .07), ("grid_shell", 3)])
def test_explicit_special_periodic_controls_reach_backend_validation(monkeypatch, name, value):
    seen = []

    def boundary(feature, *args, **controls):
        seen.append(controls)
        raise NotImplementedError("explicit unsupported control reached backend validation")

    monkeypatch.setattr(ase_calculator.native, "run_feature", boundary)
    atoms = Atoms("He", positions=[[0., 0., 0.]], cell=[7., 8., 9.], pbc=True)
    calc = DftRs(units="atomic", use_adiis=False, **{name: value})
    with pytest.raises(NotImplementedError, match="reached backend"):
        calc.get_periodic_spinor_scf(atoms)
    assert seen[-1][name] == value


@pytest.mark.parametrize("name", ["charge", "multiplicity", "grid_level", "grid_shell", "max_iter", "diis_size"])
@pytest.mark.parametrize("value", [1.9, True])
def test_periodic_reference_controls_are_not_silently_rounded(monkeypatch, name, value):
    def boundary(*args, **kwargs):
        raise AssertionError("invalid integer controls must fail before native execution")

    monkeypatch.setattr(ase_calculator.native, "periodic_scf", boundary)
    atoms = Atoms("He", positions=[[0., 0., 0.]], cell=[7., 8., 9.], pbc=True)
    with pytest.raises(ValueError, match=name):
        DftRs(**{name: value})._run_periodic(atoms, forces=False)


def test_molecular_scf_controls_share_one_translation_and_per_call_precedence(monkeypatch):
    calls = []

    def boundary(feature, *args, **controls):
        calls.append((feature, controls))
        return {"energy": -1., "converged": True, "units": {}}

    monkeypatch.setattr(ase_calculator.native, "run_feature", boundary)
    atoms = Atoms("H2", positions=[[0., 0., 0.], [0., 0., .74]])
    calc = DftRs(units="atomic", damping=.11, diis_size=3, level_shift=.2, use_adiis=False)
    assert calc._uses_common_molecular_reference()
    options = calc._molecular_driver_controls()
    assert options["damping"] == .11 and options["diis_size"] == 3
    assert options["level_shift"] == .2 and options["no_adiis"] is True
    calc.get_energy_report(atoms, damping=.07, no_adiis=False)
    assert calls[-1][1]["damping"] == .07
    assert calls[-1][1]["no_adiis"] is False
    calc._run(atoms, gradient=False)
    assert calls[-1][0] == "energy"
    assert calls[-1][1]["damping"] == .11
    assert calls[-1][1]["diis_size"] == 3


def test_default_translation_does_not_force_extended_controls_onto_other_models():
    controls = DftRs()._molecular_driver_controls()
    assert not {"level_shift", "damping", "diis_size", "adiis_until_error", "no_adiis"}.intersection(controls)


@pytest.mark.parametrize("name,value", [
    ("damping", .11), ("diis_size", 3), ("level_shift", .2), ("use_adiis", False),
])
def test_legacy_getters_refuse_scf_controls_they_cannot_forward(name, value):
    with pytest.raises(NotImplementedError, match=name):
        DftRs(**{name: value})._native_options()


def test_explicit_molecular_adiis_threshold_is_preserved_for_backend_validation():
    assert DftRs(units="atomic", adiis_until_error=.07)._molecular_driver_controls()[
        "adiis_until_error"] == .07


@pytest.mark.parametrize("name,value", [
    ("guess", "core"), ("electronic_temperature", 300.),
    ("field", [.001, .002, -.003]), ("solvent", "water"),
    ("misspelled_setting", 1),
])
def test_extra_molecular_settings_are_routed_instead_of_dropped(monkeypatch, name, value):
    seen = []

    def boundary(feature, *args, **controls):
        seen.append((feature, controls))
        return {"energy": -1., "converged": True, "units": {}}

    monkeypatch.setattr(ase_calculator.native, "run_feature", boundary)
    calc = DftRs(units="atomic", **{name: value})
    atoms = Atoms("H2", positions=[[0., 0., 0.], [0., 0., .74]])
    calc._run(atoms, gradient=False)
    assert seen[-1][0] == "energy" and seen[-1][1][name] == value
    with pytest.raises(NotImplementedError, match=name):
        calc._native_options()


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("value", [-1., float("nan"), float("inf")])
def test_invalid_temperature_is_preserved_for_backend_validation(monkeypatch, periodic, value):
    seen = []

    def boundary(feature, *args, **controls):
        seen.append(controls["electronic_temperature"])
        raise ValueError("temperature validation reached")

    monkeypatch.setattr(ase_calculator.native, "run_feature", boundary)
    atoms = Atoms("He", positions=[[0., 0., 0.]], cell=[7., 8., 9.], pbc=periodic)
    with pytest.raises(ValueError, match="temperature validation reached"):
        DftRs(units="atomic", electronic_temperature=value).get_energy_report(atoms)
    assert len(seen) == 1 and (seen[0] == value or np.isnan(seen[0]) and np.isnan(value))


def test_adiis_boolean_cannot_be_hidden_by_default_parameter_equality():
    with pytest.raises(ValueError, match="use_adiis"):
        DftRs(use_adiis=1)


def test_common_reference_charges_use_the_existing_validated_property_route(monkeypatch):
    calls = []

    def boundary(feature, *args, **controls):
        calls.append(feature)
        return {"energy": -1., "converged": True, "units": {},
                **({"mulliken": [.2, -.2]} if feature == "charges" else {})}

    monkeypatch.setattr(ase_calculator.native, "run_feature", boundary)
    atoms = Atoms("H2", positions=[[0., 0., 0.], [0., 0., .74]])
    atoms.calc = DftRs(units="atomic", damping=.11)
    assert atoms.get_charges() == pytest.approx([.2, -.2])
    assert calls == ["energy", "charges"]


def test_dynamic_response_separates_scf_mixing_from_optical_linewidth(monkeypatch):
    seen = []

    def boundary(feature, *args, **controls):
        seen.append((feature, controls))
        return {"units": {}}

    monkeypatch.setattr(ase_calculator.native, "run_feature", boundary)
    atoms = Atoms("H2", positions=[[0., 0., 0.], [0., 0., .74]])
    calc = DftRs(units="atomic", damping=.11)
    calc.get_dynamic_polarizability(atoms, omega=.2, damping=.003)
    assert seen[-1][1]["damping"] == .11
    assert seen[-1][1]["frequency_damping"] == .003
    calc.get_dynamic_polarizability(atoms, omega=.2, damping=.003, scf_damping=.07)
    assert seen[-1][1]["damping"] == .07
    assert seen[-1][1]["frequency_damping"] == .003


def test_dynamic_response_keeps_configured_solvent_and_per_call_precedence(monkeypatch):
    seen = []

    def boundary(feature, *args, **controls):
        seen.append(controls)
        return {"units": {}}

    monkeypatch.setattr(ase_calculator.native, "run_feature", boundary)
    atoms = Atoms("H2", positions=[[0., 0., 0.], [0., 0., .74]])
    calc = DftRs(units="atomic", solvent="water", eps=78.)
    calc.get_dynamic_polarizability(atoms)
    assert seen[-1]["solvent"] == "water" and seen[-1]["eps"] == 78.
    calc.get_dynamic_polarizability(atoms, solvent="acetone", eps=20.)
    assert seen[-1]["solvent"] == "acetone" and seen[-1]["eps"] == 20.


@pytest.mark.parametrize("surface", ["report", "ase-property"])
def test_actual_incomplete_scf_uses_requested_mixing_and_iteration_limit(tmp_path, surface):
    positions = np.array([[.13, -.21, .19], [.27, 1.21, -.78], [-.08, -1.69, -.83]])
    atoms = Atoms("OH2", positions=positions * Bohr)
    common = dict(xc="pbe", basis="sto-3g", grid_level=0, grid_weight_cutoff=0.,
        guess="core", max_iter=3, conv_tol=1e-30, conv_tol_dp=1e-30, conv_tol_grad=1e-30)
    energies = []
    for mixing in (0., .7):
        direct_log = tmp_path / f"native-{mixing}.log"
        ase_log = tmp_path / f"ase-{mixing}.log"
        with pytest.raises(RuntimeError, match="did not converge after 3 iterations"):
            dft_rs.energy([8, 1, 1], positions.tolist(), damping=mixing, no_adiis=True,
                          diis_size=2, log_file=direct_log, **common)
        calc = DftRs(units="atomic", damping=mixing, use_adiis=False, diis_size=2, **common)
        with pytest.raises(RuntimeError, match="did not converge after 3 iterations"):
            if surface == "report":
                calc.get_energy_report(atoms, log_file=ase_log)
            else:
                calc.set(log_file=ase_log)
                atoms.calc = calc
                atoms.get_potential_energy()
        histories = []
        for path in (direct_log, ase_log):
            text = path.read_text(encoding="utf-8")
            rows = re.findall(r"\[scf\] iter\s+(\d+)\s+E =\s+([+-]?\d+\.\d+)", text)
            assert [int(index) for index, _ in rows] == [1, 2, 3]
            assert "did not converge after 3 iterations" in text
            histories.append([float(energy) for _, energy in rows])
        assert histories[0] == pytest.approx(histories[1], abs=1e-12)
        energies.append(histories[1][-1])
    assert abs(energies[0] - energies[1]) > 1e-8


@pytest.mark.parametrize("feature", ["polarizability", "dynamic-polarizability", "hessian",
    "frequencies", "ir-raman", "nmr-shielding", "spin-spin-couplings", "conceptual-dft"])
def test_actual_response_routes_preserve_reference_iteration_limit(feature):
    with pytest.raises(RuntimeError, match="did not converge after 1 iterations"):
        dft_rs.run_feature(feature, [1, 1], [[.13, -.21, .17], [1.48, .19, -.11]],
            basis="sto-3g", xc="hf", dispersion="none", max_iter=1, guess="core",
            conv_tol=1e-30, conv_tol_grad=1e-30, conv_tol_dp=1e-30,
            cpu_threads=1, memory_mb=256)


@pytest.mark.parametrize("temperature", [-1., float("nan"), float("inf")])
def test_actual_molecular_backend_rejects_invalid_configured_temperature(temperature):
    atoms = Atoms("He", positions=[[0., 0., 0.]])
    with pytest.raises(ValueError, match="(?i)temperature"):
        DftRs(units="atomic", electronic_temperature=temperature, cpu_threads=1,
              memory_mb=128).get_energy_report(atoms)


def test_actual_ase_energy_does_not_ignore_unknown_calculator_parameters():
    atoms = Atoms("He", positions=[[0., 0., 0.]])
    atoms.calc = DftRs(units="atomic", misspelled_setting=1, cpu_threads=1, memory_mb=128)
    with pytest.raises(ValueError, match="unknown controls: misspelled_setting"):
        atoms.get_potential_energy()


def test_known_inapplicable_molecular_control_is_not_an_unknown_input_or_ignored():
    atoms = Atoms("H2", positions=[[0., 0., 0.], [0., 0., .74]])
    calc = DftRs(units="atomic", adiis_until_error=.07, cpu_threads=1, memory_mb=128)
    with pytest.raises(NotImplementedError, match="adiis_until_error"):
        calc.get_energy_report(atoms)


@pytest.mark.parametrize("feature,extra", [
    ("energy", {"reference": "ghf"}),
    ("spinor-polarizability", {"frequencies": [0.]}),
    ("magnetic-anisotropy", {"spin_orbit_self_consistent": True, "spin_axes": [[.3, .4, .5]]}),
])
@pytest.mark.parametrize("surface", ["native", "ase", "cli", "cli-alias"])
def test_configured_spinor_subcalculations_reach_all_public_routes(tmp_path, feature, extra, surface):
    positions = np.array([[.13, -.21, .17], [1.48, .19, -.11]])
    controls = dict(xc="hf", basis="sto-3g", dispersion="none", max_iter=40,
        conv_tol=1e-12, conv_tol_dp=1e-11, conv_tol_grad=1e-10,
        damping=.17, diis_size=3, cpu_threads=1, memory_mb=256)
    expected = dft_rs.run_feature(feature, [1, 1], positions.tolist(), **controls, **extra)
    if surface == "native":
        actual = getattr(dft_rs, feature.replace("-", "_"))([1, 1], positions.tolist(), **controls, **extra)
    elif surface == "ase":
        atoms = Atoms("H2", positions=positions * Bohr)
        actual = DftRs(units="atomic", **controls).get_feature_report(feature, atoms, **extra)
    else:
        name = "DFT_RS_CLI" if surface == "cli" else "DFT_RS_CLI_ALIAS"
        if name not in os.environ:
            pytest.skip(f"set {name} for linked-artifact acceptance")
        binary = Path(os.environ[name])
        xyz = tmp_path / "spinor-settings.xyz"
        xyz.write_text("2\nCommon spinor settings\n" + "\n".join(
            "H " + " ".join(f"{value * Bohr:.17g}" for value in row) for row in positions) + "\n",
            encoding="utf-8")
        command = [str(binary), feature, str(xyz), "--format", "json"]
        for key, value in {**controls, **extra}.items():
            command.append("--" + key.replace("_", "-") + "=" + (
                value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))))
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        actual = json.loads(result.stdout)
    assert expected["converged"] and actual["converged"]
    assert actual["energy"] == pytest.approx(expected["energy"], abs=1e-10)
    if feature == "spinor-polarizability":
        assert actual["spinor_scf_iterations"] <= controls["max_iter"]
        assert actual["polarizability_real"] == pytest.approx(np.asarray(expected["polarizability_real"]), abs=1e-10)


def test_boundary_constants_match_the_frozen_core_codata_2018_convention():
    """Pin both factors independently of imports and ASE release defaults."""
    assert ase_calculator.Bohr == 0.529177210903
    assert ase_calculator.Hartree == 27.211386245988
    factors, label = ase_calculator._ase_unit("hartree/bohr^4")
    assert label == "eV/angstrom^4"
    assert ase_calculator._factor(factors, {}) == pytest.approx(
        27.211386245988 / 0.529177210903**4, rel=1e-15)


def test_the_ase_boundary_is_the_only_place_units_change():
    """Pin every conversion in the adapter against hand-computed values.

    The native layer is atomic units throughout and ASE requires eV and
    Angstrom, so this class is the one place a factor can be wrong. A
    missing or doubled conversion still produces plausible-looking numbers —
    forces off by 1.89, a dipole off by 0.53 — which no physical assertion
    elsewhere in the suite would catch. Hence exact arithmetic here, on
    values chosen so that no two conversion factors could be confused.
    """
    energy_hartree = -1.234_5
    gradient_hartree_bohr = [[0.0, 0.0, 0.25], [0.0, 0.0, -0.25]]
    dipole_au = [0.1, -0.2, 0.3]
    hessian_hartree_bohr2 = (np.eye(6) * 0.5).tolist()
    seen = {}

    def fake_calculate(numbers, positions, **kwargs):
        seen["positions"] = np.asarray(positions, dtype=float)
        result = {
            "energy": energy_hartree,
            "dipole": dipole_au,
            "mulliken": [0.3, -0.3],
            "converged": True,
            "iterations": 1,
            "e_nuc": 0.5,
            "e_elec": -1.734_5,
            "e_xc": -0.25,
            "e_nlc": 0.0,
            "e_dispersion": 0.0,
            "s_squared": 0.0,
            "n_alpha": 1,
            "n_beta": 1,
            "unrestricted": False,
            "n_core_electrons": 0,
        }
        if kwargs["gradient"]:
            result["gradient"] = gradient_hartree_bohr
        return result

    def fake_hessian(feature, numbers, positions, **kwargs):
        assert feature == "hessian"
        return {"hessian": hessian_hartree_bohr2, "provenance": "analytic", "energy": -1.0}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ase_calculator.native, "calculate", fake_calculate)
        patch.setattr(ase_calculator.native, "run_feature", fake_hessian)
        angstrom_positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
        atoms = Atoms("H2", positions=angstrom_positions)
        atoms.calc = DftRs(units="atomic", ecp="none")

        # Geometry goes IN as Bohr.
        forces = atoms.get_forces()
        assert np.allclose(seen["positions"], np.asarray(angstrom_positions) / Bohr)
        assert seen["positions"][1][2] == pytest.approx(0.74 / 0.529_177_210_9, rel=1e-9)

        # Energy out as eV, force as eV/Angstrom and with the gradient's sign
        # flipped — a force is minus a gradient.
        assert atoms.get_potential_energy() == pytest.approx(energy_hartree * Hartree)
        assert forces == pytest.approx(
            -np.asarray(gradient_hartree_bohr) * Hartree / Bohr
        )
        assert forces[0][2] < 0.0 < forces[1][2]

        # Dipole out as e*Angstrom: e*Bohr times Angstrom-per-Bohr.
        assert atoms.calc.results["dipole"] == pytest.approx(
            np.asarray(dipole_au) * Bohr
        )
        # Populations are dimensionless and must NOT be scaled.
        assert atoms.calc.results["charges"] == pytest.approx([0.3, -0.3])

        # Hessian out as eV/Angstrom^2 — two factors of Bohr, so a single
        # missing one is a factor of 1.89 rather than 3.57.
        hessian = atoms.calc.get_hessian(atoms)
        assert hessian[0][0] == pytest.approx(0.5 * Hartree / (Bohr * Bohr))

        # The diagnostic block is a deliberate exception and stays atomic,
        # which is only safe because it says so in the payload.
        scf = atoms.calc.results["scf"]
        assert scf["e_nuc"] == pytest.approx(0.5)
        assert scf["units"] == "atomic (Hartree)"


def test_vibrations_are_lazy_cached_and_returned_as_copies():
    """`get_vibrations` must never be triggered by an ASE property request.

    A vibrational analysis costs a Hessian. If it were computed alongside
    the energy, every `get_potential_energy()` in an optimizer loop would
    pay for one.
    """
    calls = []

    def fake_vibrations(numbers, positions, **kwargs):
        calls.append(kwargs.copy())
        return {
            "frequencies": [1600.0, 3700.0, 3800.0],
            "ir_intensities": [70.0, 5.0, 40.0],
            "n_imaginary": 0,
            "n_projected": 6,
            "method": "analytic",
        }

    def fake_calculate(numbers, positions, **kwargs):
        return {
            "energy": -1.0,
            "dipole": [0.0, 0.0, 0.0],
            "mulliken": [0.0, 0.0],
            "converged": True,
            "iterations": 1,
            "gradient": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        }

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ase_calculator.native, "vibrations", fake_vibrations)
        patch.setattr(ase_calculator.native, "calculate", fake_calculate)
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        calc = DftRs(units="atomic", ecp="none")
        atoms.calc = calc

        atoms.get_potential_energy()
        atoms.get_forces()
        assert calls == [], "an ASE property request computed a vibrational analysis"

        first = calc.get_vibrations(atoms)
        assert len(calls) == 1
        first["frequencies"][0] = -1.0
        second = calc.get_vibrations(atoms)
        assert len(calls) == 1, "the cache did not serve the second request"
        assert second["frequencies"][0] == pytest.approx(1600.0)

        # Frequencies stay in cm^-1 and intensities in km/mol, as
        # `ase.vibrations` also reports them.
        assert second["frequencies"] == pytest.approx([1600.0, 3700.0, 3800.0])
        assert second["ir_intensities"] == pytest.approx([70.0, 5.0, 40.0])

        # A changed parameter is a different calculation, not a cache hit.
        calc.set(grid_level=5)
        calc.get_vibrations(atoms)
        assert len(calls) == 2


def test_energy_only_is_lazy_and_all_controls_are_forwarded(monkeypatch):
    calls = []

    def fake_calculate(numbers, positions, **kwargs):
        calls.append(kwargs.copy())
        result = {
            "energy": -1.0,
            "dipole": [0.0, 0.0, 0.0],
            "mulliken": [0.0, 0.0],
            "converged": True,
            "iterations": 1,
            "e_nuc": 0.0,
            "e_elec": -1.0,
            "e_xc": 0.0,
            "e_nlc": 0.0,
            "e_dispersion": 0.0,
            "s_squared": 0.0,
            "n_alpha": 1,
            "n_beta": 1,
            "unrestricted": False,
            "n_core_electrons": 0,
        }
        if kwargs["gradient"]:
            result["gradient"] = [[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]]
        return result

    monkeypatch.setattr(ase_calculator.native, "calculate", fake_calculate)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = DftRs(
        units="atomic",
        basis="def2-svp",
        ecp="none",
        xc="pbe0",
        dispersion="d3bj",
        density_fit="def2-universal-jfit",
        conv_tol=1e-10,
        conv_tol_grad=1e-6,
        conv_tol_dp=1e-9,
        incore_mb=512,
        vv10_grid_level=2,
    )

    assert np.isfinite(atoms.get_potential_energy())
    assert calls[-1]["gradient"] is False
    assert "forces" not in atoms.calc.results
    forces = atoms.get_forces()
    assert forces.shape == (2, 3)
    assert calls[-1]["gradient"] is True
    for key, expected in {
        "ecp": "none",
        "dispersion": "d3bj",
        "density_fit": "def2-universal-jfit",
        "conv_tol": 1e-10,
        "conv_tol_grad": 1e-6,
        "conv_tol_dp": 1e-9,
        "incore_mb": 512,
        "vv10_grid_level": 2,
    }.items():
        assert calls[-1][key] == expected


def test_hessian_cache_returns_copies(monkeypatch):
    calls = []

    def fake_hessian(feature, numbers, positions, **kwargs):
        assert feature == "hessian"
        calls.append(kwargs.copy())
        return {"hessian": np.eye(6).tolist(), "provenance": "analytic", "energy": -1.0}

    monkeypatch.setattr(ase_calculator.native, "run_feature", fake_hessian)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    calc = DftRs(units="atomic", ecp="none")
    atoms.calc = calc
    first = calc.get_hessian(atoms)
    first[0, 0] = 99.0
    second = calc.get_hessian(atoms)
    assert len(calls) == 1
    assert second[0, 0] != 99.0
    assert calls[0]["ecp"] == "none"


def test_full_pbc_uses_the_driver_for_energy_forces_and_stress(monkeypatch):
    calls = []

    def fake_run_feature(feature, numbers, positions, cell=None, pbc=None, **controls):
        calls.append((feature, positions, cell, pbc, controls.copy()))
        report = dft_rs.Report(energy=-2.0, free_energy=-2.1, converged=True, units={})
        if controls.get("forces"):
            report["forces"] = [[0.0, 0.0, 0.1]]
        if controls.get("stress"):
            report["stress_voigt"] = [0.01, 0.02, 0.03, 0.0, 0.0, 0.0]
        return report

    monkeypatch.setattr(ase_calculator.native, "run_feature", fake_run_feature)
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=[3.0, 4.0, 5.0], pbc=True)
    atoms.calc = DftRs(
        units="atomic",
        xc="r2scan",
        k_mesh=(3, 1, 1),
        overlap_cutoff=8.0,
        exchange_cutoff=7.0,
        grid_shell=1,
        electronic_temperature=1200.0,
        level_shift=0.4,
        damping=0.1,
        diis_size=5,
        use_adiis=False,
        adiis_until_error=0.07,
        guess="gwh",
        dispersion="d3bj",
        three_body=True,
    )
    assert atoms.get_potential_energy() == pytest.approx(-2.0 * Hartree)
    feature, _, cell, pbc, controls = calls[-1]
    assert feature == "periodic-energy"
    assert controls["forces"] is False and controls["stress"] is False
    assert np.allclose(np.diag(cell), np.array([3.0, 4.0, 5.0]) / Bohr)
    assert pbc == [True, True, True]
    assert controls["electronic_temperature"] == 1200.0
    assert controls["k_mesh"] == [3, 1, 1]
    assert controls["exchange_cutoff"] == 7.0
    assert controls["level_shift"] == 0.4
    assert controls["damping"] == 0.1
    assert controls["diis_size"] == 5
    assert controls["no_adiis"] is True
    assert controls["adiis_until_error"] == 0.07
    assert controls["guess"] == "gwh"
    # The dispersion model and extra calculator parameters reach the driver.
    assert controls["dispersion"] == "d3bj" and controls["three_body"] is True
    assert "units" not in controls
    assert atoms.calc.results["free_energy"] == pytest.approx(-2.1 * Hartree)
    forces = atoms.get_forces()
    assert forces.shape == (1, 3)
    assert calls[-1][4]["forces"] is True
    assert forces[0, 2] == pytest.approx(0.1 * Hartree / Bohr)
    stress = atoms.get_stress()
    assert calls[-1][4]["stress"] is True
    assert stress[0] == pytest.approx(0.01 * Hartree / Bohr**3)


@pytest.mark.parametrize("pbc", [[True, False, False], [True, True, False]])
def test_partial_pbc_reaches_the_typed_native_boundary(monkeypatch, pbc):
    calls = []

    def fake_report(feature, numbers, positions, *, cell, pbc, **kwargs):
        calls.append((feature, pbc, cell))
        return {"energy": -2.5, "converged": True}

    monkeypatch.setattr(ase_calculator.native, "run_feature", fake_report)
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=[3.0, 4.0, 5.0], pbc=pbc)
    atoms.calc = DftRs(units="atomic")
    assert atoms.get_potential_energy() == pytest.approx(-2.5 * Hartree)
    [(feature, native_pbc, cell)] = calls
    assert feature == "periodic-energy"
    assert native_pbc == pbc
    np.testing.assert_allclose(cell, atoms.cell.array / Bohr, atol=1e-14)
