# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""The common driver gives identical reports on native Python, ASE and the CLI,
and every registered calculation is reachable on all surfaces."""
import json
import inspect
import importlib.util
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr

import dft_rs
from dft_rs import DftRs

WATER_NUMBERS = [8, 1, 1]
WATER = [[0.0, 0.0, 0.1173], [0.0, 1.4332, -0.9314], [0.0, -1.4332, -0.9314]]
TIGHT = dict(basis="sto-3g", conv_tol=1e-13, conv_tol_dp=1e-12)

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("api_reference", _ROOT / "tools/api_reference.py")
_REFERENCE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REFERENCE)
_DRIVERS = [entry for entry in _REFERENCE.registry_specs(_ROOT) if entry["route"] == "Driver"]


@pytest.mark.parametrize("entry", _DRIVERS, ids=lambda entry: entry["id"])
def test_every_driver_wrapper_forwards_to_its_registered_calculation(monkeypatch, entry):
    calls = []
    resource_calls = []

    def execute(function, args, kwargs, **limits):
        resource_calls.append(limits)
        return function(*args, **kwargs)

    def record(feature, *args, **controls):
        calls.append((feature, controls))
        return dft_rs.Report(feature=feature, units={}, response_units="atomic units", converged=True)

    monkeypatch.setattr(dft_rs.native, "run_feature", record)
    monkeypatch.setattr(dft_rs.native._core, "_with_resources", execute)
    wrapper = getattr(dft_rs.native, entry["python"])
    values = {"numbers": [2], "positions": [[0.0, 0.0, 0.0]], "cell": np.eye(3).tolist(),
              "direction": [0.1, 0.2, 0.3], "directions": [[0.1, 0.2, 0.3]],
              "nuclear_direction": [[0.1, 0.2, 0.3]], "phases": [[1.0, 5.0, 3.0]]}
    if entry["id"] == "continuous-species":
        values.update(numbers=[7, 7], positions=[[0.0, 0.0, -1.05], [0.0, 0.0, 1.05]],
                      fixed_electron_count=14, charge_directions=[[-1.0, 1.0]],
                      charge_coordinates=[0.25], charge_bounds=[[-1.5, 1.5]])
    arguments = {}
    probe = object()
    forwards_controls = False
    for name, parameter in inspect.signature(wrapper).parameters.items():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            forwards_controls = True
        elif parameter.kind != inspect.Parameter.VAR_POSITIONAL and parameter.default is inspect.Parameter.empty:
            arguments[name] = values.get(name, [1, 1, 1])
    if forwards_controls:
        arguments["api_contract_probe"] = probe
    arguments.update(cpu_threads=2, memory_mb=8)
    report = wrapper(**arguments)
    assert resource_calls == [{"cpu_threads": 2, "memory_mb": 8}], entry["python"]
    assert len(calls) == 1 and calls[0][0] == entry["id"], entry["python"]
    assert "cpu_threads" not in calls[0][1] and "memory_mb" not in calls[0][1]
    assert report["feature"] == entry["id"]
    if forwards_controls:
        assert calls[0][1]["api_contract_probe"] is probe


@pytest.mark.parametrize("entry", [row for row in _DRIVERS if row["ase"] != "-"], ids=lambda entry: entry["id"])
def test_every_ase_driver_getter_forwards_to_its_registered_calculation(monkeypatch, entry):
    calls = []
    resource_calls = []
    active = []

    def execute(function, args, kwargs, **limits):
        resource_calls.append(limits)
        active.append(True)
        try:
            return function(*args, **kwargs)
        finally:
            active.pop()

    def record(feature, *args, **controls):
        calls.append((feature, controls))
        return dft_rs.Report(feature=feature, units={}, response_units="atomic units", converged=True,
                             energy=-1.0, stress_voigt=[0.1] * 6)

    monkeypatch.setattr(dft_rs.native, "run_feature", record)
    monkeypatch.setattr(dft_rs.native._core, "_with_resources", execute)
    monkeypatch.setattr(dft_rs.native._core, "_resource_usage", lambda: {} if active else None)
    monkeypatch.setattr(dft_rs.native._core, "_resource_scope_configured", lambda: bool(active))
    calculator = DftRs(units="atomic", cpu_threads=2, memory_mb=8)
    getter = getattr(calculator, entry["ase"])
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 3,
                  pbc=entry["space"] == "Periodic")
    values = {"direction": [0.1, 0.2, 0.3], "directions": [[0.1, 0.2, 0.3]],
              "nuclear_direction": [[0.1, 0.2, 0.3]], "phases": [[1.0, 5.0, 3.0]],
              "path": [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]}
    if entry["id"] == "continuous-species":
        atoms = Atoms("N2", positions=[[0.0, 0.0, -1.05], [0.0, 0.0, 1.05]])
        values.update(fixed_electron_count=14, charge_directions=[[-1.0, 1.0]],
                      charge_coordinates=[0.25], charge_bounds=[[-1.5, 1.5]])
    signature = inspect.signature(getter)
    arguments = {"atoms": atoms} if "atoms" in signature.parameters else {}
    if entry["id"] in ("superconducting-state", "no-pair-projector", "scdft-pairing-kernel"):
        arguments["units"] = "atomic"
    if "cpu_threads" in signature.parameters:
        arguments.update(cpu_threads=2, memory_mb=8)
    probe = object()
    forwards_controls = False
    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            forwards_controls = True
        elif parameter.kind != inspect.Parameter.VAR_POSITIONAL and parameter.default is inspect.Parameter.empty:
            arguments[name] = values.get(name, [1, 1, 1])
    if forwards_controls:
        arguments["api_contract_probe"] = probe
    result = getter(**arguments)
    assert resource_calls == [{"cpu_threads": 2, "memory_mb": 8}], entry["ase"]
    # ASE's standard stress getter requests energy and stress together from
    # the common periodic-energy route; the named native stress route is separate.
    expected = "periodic-energy" if entry["id"] == "stress" else entry["id"]
    assert len(calls) == 1 and calls[0][0] == expected, entry["ase"]
    assert "cpu_threads" not in calls[0][1] and "memory_mb" not in calls[0][1]
    if entry["id"] == "stress":
        assert calls[0][1]["stress"] is True and np.shape(result) == (6,)
    else:
        assert result["feature"] == entry["id"]
    if forwards_controls:
        assert calls[0][1]["api_contract_probe"] is probe


def water_atoms():
    return Atoms("OH2", positions=np.asarray(WATER) * Bohr)


def cli(*arguments):
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for route acceptance"
    result = subprocess.run([executable, *arguments], capture_output=True, text=True)
    return result


def xyz(tmp_path):
    path = tmp_path / "water.xyz"
    rows = "".join(
        f"{symbol} {x * 0.529177210903:.16f} {y * 0.529177210903:.16f} {z * 0.529177210903:.16f}\n"
        for symbol, (x, y, z) in zip(["O", "H", "H"], WATER)
    )
    path.write_text("3\nwater\n" + rows)
    return path


def test_registry_names_exist_on_every_surface():
    registry = dft_rs.feature_registry()
    assert registry["feature"] == "feature-registry"
    spaces = {"Molecule": "molecule", "Periodic": "periodic", "Both": "molecule_and_periodic", "Data": "data"}
    source = [dict(row, route=row["route"].lower(), space=spaces[row["space"]])
              for row in _REFERENCE.registry_specs(_ROOT)]
    # Applicability comments are mandatory source-audit data, not fields of
    # the public Rust FeatureSpec or its serialized feature inventory.
    for row in source:
        assert row.pop("settings_contract") in _REFERENCE.SETTINGS_CONTRACTS
    fields = tuple(source[0])
    actual = [{key: row[key] for key in fields} for row in registry["features"]]
    assert actual == source, "the compiled feature inventory differs from the source being verified"
    ids = [row["id"] for row in registry["features"]]
    assert len(ids) == len(set(ids))
    for row in registry["features"]:
        function = getattr(dft_rs.native, row["python"], None)
        assert callable(function), row["python"]
        assert getattr(dft_rs, row["python"], None) is function, row["python"]
        assert row["python"] in dft_rs.__all__, row["python"]
        if row["ase"] != "-":
            assert callable(getattr(DftRs, row["ase"], None)), row["ase"]
        for target in row["targets"]:
            assert target.startswith("T") and len(target) == 4


@pytest.mark.parametrize("metadata", [
    {}, {"provenance": "numerical"},
    {"provenance": "analytic", "method": "numerical"},
])
def test_common_hessian_never_relabels_missing_or_numerical_provenance(monkeypatch, metadata):
    report = dict(metadata, hessian=[[1.0]], energy=-1.0)
    original = report.copy()
    monkeypatch.setattr(dft_rs.native, "run_feature", lambda *args, **kwargs: report)
    with pytest.raises(RuntimeError, match="analytic.*provenance"):
        dft_rs.hessian([2], [[0.0, 0.0, 0.0]], reference="rks")
    assert report == original


def test_legacy_hessian_keeps_its_explicit_numerical_method(monkeypatch):
    report = dict(method="numerical", hessian=[[1.0]], energy=-1.0)
    monkeypatch.setattr(dft_rs.native._core, "hessian", lambda *args, **kwargs: report)
    assert dft_rs.hessian([2], [[0.0, 0.0, 0.0]])["method"] == "numerical"


@pytest.mark.parametrize("reference", [None, "rks", "uks", "roks", "gks"])
def test_ase_array_hessian_always_selects_the_analytic_driver(monkeypatch, reference):
    calls = []

    def report(self, feature, atoms=None, **controls):
        calls.append((feature, controls))
        return dict(hessian=np.eye(3).tolist(), provenance="analytic")

    def forbidden(*args, **kwargs):
        pytest.fail("the ASE analytic Hessian getter reached the legacy numerical-fallback API")

    monkeypatch.setattr(DftRs, "get_feature_report", report)
    monkeypatch.setattr(dft_rs.native, "hessian", forbidden)
    options = {} if reference is None else dict(reference=reference)
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]])
    calculator = DftRs(**options)
    actual = calculator.get_hessian(atoms)
    np.testing.assert_allclose(actual, np.eye(3) * 27.211386245988 / Bohr**2, rtol=1e-15)
    assert calls == [("hessian", {"units": "atomic"})]
    actual[0, 0] = -1.0
    assert calculator.get_hessian(atoms)[0, 0] > 0.0
    assert len(calls) == 1


def test_ase_array_hessian_preserves_an_unsupported_analytic_error(monkeypatch):
    original = NotImplementedError("analytic Hessian provider unavailable for the selected model")

    def unsupported(*args, **kwargs):
        raise original

    monkeypatch.setattr(DftRs, "get_feature_report", unsupported)
    with pytest.raises(NotImplementedError) as caught:
        DftRs().get_hessian(Atoms("He", positions=[[0.0, 0.0, 0.0]]))
    assert caught.value is original


@pytest.mark.parametrize("reference", ["rks", "uks", "roks", "gks"])
def test_ase_hessian_forwards_the_selected_reference_to_the_backend(monkeypatch, reference):
    calls = []

    def backend(feature, numbers, positions, **controls):
        calls.append((feature, controls))
        return dft_rs.Report(hessian=np.eye(3).tolist(), provenance="analytic")

    monkeypatch.setattr(dft_rs.native, "run_feature", backend)
    DftRs(units="atomic", reference=reference).get_hessian(Atoms("He", positions=[[0.0, 0.0, 0.0]]))
    assert len(calls) == 1 and calls[0][0] == "hessian"
    assert calls[0][1]["reference"] == reference
    assert "nbo" not in calls[0][1]


def test_actual_analytic_hessian_native_and_default_ase_routes_agree():
    positions = [[0.1, 0.2, 0.3], [0.5, 0.7, 1.5]]
    native = dft_rs.hessian([1, 1], positions, reference="rks", **TIGHT)
    common = dft_rs.hessian_report([1, 1], positions, reference="rks", **TIGHT)
    atoms = Atoms("H2", positions=np.asarray(positions) * Bohr)
    adapted = DftRs(units="atomic", **TIGHT).get_hessian(atoms)
    assert native["method"] == native["provenance"] == common["provenance"] == "analytic"
    assert np.shape(native["hessian"]) == (6, 6)
    np.testing.assert_allclose(native["hessian"], common["hessian"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(adapted, np.asarray(common["hessian"]) * 27.211386245988 / Bohr**2,
                               rtol=0, atol=1e-10)


@pytest.mark.parametrize("metadata", [{}, {"provenance": "numerical"},
                                      {"provenance": "analytic", "method": "numerical"}])
@pytest.mark.parametrize("periodic", [False, True])
def test_ase_array_hessian_rejects_unverified_provenance(monkeypatch, metadata, periodic):
    monkeypatch.setattr(DftRs, "get_feature_report", lambda *args, **kwargs: dict(metadata, hessian=[[1.0]]))
    monkeypatch.setattr(DftRs, "_periodic_report", lambda *args, **kwargs: dict(metadata, force_constants=[[1.0]]))
    with pytest.raises(RuntimeError, match="analytic.*provenance"):
        DftRs().get_hessian(Atoms("He", positions=[[0.0, 0.0, 0.0]], pbc=periodic, cell=np.eye(3) * 3))


@pytest.mark.parametrize("number", [1.9, "1", 0, 119])
def test_direct_driver_rejects_invalid_atomic_numbers_before_dispatch(monkeypatch, number):
    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("invalid atomic numbers reached the compiled driver")

    monkeypatch.setattr(dft_rs.native._core, "run_feature", unexpected_dispatch)
    with pytest.raises(ValueError, match="atomic numbers"):
        dft_rs.energy([number], [[0.0, 0.0, 0.0]])


@pytest.mark.parametrize("pbc", [["False", False, False], [2, 0, 0], [None, False, False]])
def test_direct_driver_rejects_ambiguous_periodicity_before_dispatch(monkeypatch, pbc):
    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("ambiguous periodicity reached the compiled driver")

    monkeypatch.setattr(dft_rs.native._core, "run_feature", unexpected_dispatch)
    with pytest.raises(ValueError, match="periodicity"):
        dft_rs.run_feature("periodic-energy", [2], [[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=pbc)


@pytest.mark.parametrize("pbc", [np.array([True, False, False]), np.array([1, 0, 0]), (True, False, False)])
def test_direct_driver_preserves_valid_numpy_and_boolean_inputs(monkeypatch, pbc):
    calls = []

    def record(feature, numbers, positions, **controls):
        calls.append((numbers, controls["pbc"]))
        return {"feature": feature, "units": {}, "converged": True}

    monkeypatch.setattr(dft_rs.native._core, "run_feature", record)
    dft_rs.run_feature("periodic-energy", np.array([2]), np.zeros((1, 3)), cell=np.eye(3), pbc=pbc)
    assert calls == [([2], [True, False, False])]


@pytest.mark.parametrize("options,omega,damping,imaginary", [
    ({"frequency": 0.05}, 0.05, 0.0, False),
    ({"response_omega": 0.05}, 0.05, 0.0, False),
    ({"frequency_damping": 0.01}, 0.0, 0.01, False),
    ({"response_damping": 0.01}, 0.0, 0.01, False),
    ({"omega": 0.05, "imaginary_frequency": True}, 0.05, 0.0, True),
])
def test_dynamic_frequency_aliases_keep_metadata_and_tensor_consistent(options, omega, damping, imaginary):
    geometry = ([1, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    report = dft_rs.dynamic_polarizability(*geometry, **options)
    canonical = dft_rs.run_feature("dynamic-polarizability", *geometry,
                                   frequency=omega, frequency_damping=damping, imaginary=imaginary)
    assert report["omega"] == omega and report["damping"] == damping
    assert report["imaginary"] is imaginary
    np.testing.assert_allclose(report["polarizability_real"], canonical["polarizability_real"], atol=1e-12)
    np.testing.assert_allclose(report["polarizability_imag"], canonical["polarizability_imag"], atol=1e-12)
    assert report["frequency_real"] == canonical["frequency_real"]
    assert report["frequency_imag"] == canonical["frequency_imag"]


@pytest.mark.parametrize("options", [
    {"omega": 0.03, "frequency": 0.05},
    {"frequency": 0.05, "response_omega": 0.05},
    {"damping": 0.01, "response_damping": 0.02},
    {"imaginary": True, "imaginary_frequency": True},
])
def test_conflicting_frequency_aliases_are_rejected(options):
    with pytest.raises(ValueError, match="use either"):
        dft_rs.dynamic_polarizability([1, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], **options)


@pytest.mark.parametrize("feature,wrapper,getter,key", [
    ("energy", dft_rs.energy, "get_energy_report", "energy"),
    ("dipole", dft_rs.dipole, "get_dipole_report", "dipole"),
    ("gradient", dft_rs.gradient, "get_gradient_report", "gradient"),
    ("polarizability", dft_rs.polarizability_report, "get_polarizability_report", "polarizability"),
])
def test_native_ase_and_cli_reports_agree(tmp_path, feature, wrapper, getter, key):
    controls = dict(TIGHT, origin="zero") if feature == "dipole" else dict(TIGHT)
    native = wrapper(WATER_NUMBERS, WATER, **controls)
    via_run = dft_rs.run_feature(feature, WATER_NUMBERS, WATER, **controls)
    calc = DftRs(units="atomic", basis="sto-3g", conv_tol=1e-13, conv_tol_dp=1e-12)
    extra = {"origin": "zero"} if feature == "dipole" else {}
    adapted = getattr(calc, getter)(water_atoms(), units="atomic", **extra)
    for report in (native, via_run, adapted):
        assert report["feature"] == feature and report["converged"]
        assert "units" in report and isinstance(report["warnings"], list)
        assert report["units"]["scf_orbital_residual"] == "hartree"
        assert 0 <= report["scf_orbital_residual"] < 1e-8
    np.testing.assert_allclose(native[key], via_run[key], atol=1e-12)
    np.testing.assert_allclose(native[key], adapted[key], atol=1e-9)
    arguments = [feature, str(xyz(tmp_path)), "--basis", "sto-3g", "--conv-tol", "1e-13",
                 "--conv-tol-dp", "1e-12", "--format", "json"]
    if feature == "dipole":
        arguments += ["--origin", "zero"]
    result = cli(*arguments)
    assert result.returncode == 0, result.stderr
    standalone = json.loads(result.stdout)
    np.testing.assert_allclose(standalone[key], native[key], atol=1e-9)
    assert standalone["units"]["scf_orbital_residual"] == "hartree"
    assert 0 <= standalone["scf_orbital_residual"] < 1e-8


def test_save_writes_the_printed_report_and_unknown_controls_fail(tmp_path):
    saved = tmp_path / "report.json"
    result = cli("orbitals", str(xyz(tmp_path)), "--basis", "sto-3g", "--format", "json", "--save", str(saved))
    assert result.returncode == 0, result.stderr
    assert json.loads(saved.read_text()) == json.loads(result.stdout)
    with pytest.raises(ValueError, match="grid_levle"):
        dft_rs.orbitals(WATER_NUMBERS, WATER, basis="sto-3g", grid_levle=2)
    failed = cli("orbitals", str(xyz(tmp_path)), "--grid-levle", "2")
    assert failed.returncode != 0 and "grid_levle" in failed.stderr


def test_finite_temperature_and_conceptual_reports_on_every_surface(tmp_path):
    lih = ([3, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 3.2]])
    controls = dict(basis="6-31g", xc="svwn5", grid_level=4, conv_tol=1e-12, conv_tol_dp=1e-10,
                    max_iter=500, electronic_temperature=8000.0)
    native = dft_rs.conceptual_dft(*lih, **controls)
    assert native["route"] == "grand_canonical"
    assert abs(sum(native["fukui"]) - 1.0) < 1e-9
    atoms = Atoms("LiH", positions=np.asarray(lih[1]) * Bohr)
    calc = DftRs(units="atomic", basis="6-31g", xc="svwn5", grid_level=4, conv_tol=1e-12, conv_tol_dp=1e-10,
                 max_iter=500, electronic_temperature=8000.0)
    adapted = calc.get_conceptual_dft(atoms, units="atomic")
    np.testing.assert_allclose(adapted["global"]["hardness"], native["global"]["hardness"], rtol=1e-8)
    free = dft_rs.free_energy(*lih, **controls)
    assert free["entropy"] > 0 and free["free_energy"] < free["internal_energy"]
    path = tmp_path / "lih.xyz"
    path.write_text(f"2\nLiH\nLi 0 0 0\nH 0 0 {3.2 * 0.529177210903:.16f}\n")
    result = cli("conceptual-dft", str(path), "--basis", "6-31g", "--xc", "svwn5", "--grid-level", "4",
                 "--conv-tol", "1e-12", "--conv-tol-dp", "1e-10", "--max-iter", "500",
                 "--electronic-temperature", "8000", "--format", "json")
    assert result.returncode == 0, result.stderr
    np.testing.assert_allclose(json.loads(result.stdout)["global"]["hardness"],
                               native["global"]["hardness"], rtol=1e-8)


def test_cli_main_shares_the_executable_entry_point(tmp_path, capfd):
    assert dft_rs.cli_main(["--version"]) == 0
    assert "dft-rs" in capfd.readouterr().out
    assert dft_rs.cli_main(["orbitals", str(xyz(tmp_path)), "--grid-levle", "2"]) == 1
    assert "grid_levle" in capfd.readouterr().err


def test_nonconverged_reference_is_rejected_on_every_surface(tmp_path):
    for function in (dft_rs.calculate, dft_rs.energy):
        with pytest.raises(RuntimeError, match="SCF did not converge"):
            function(WATER_NUMBERS, WATER, max_iter=1)
    atoms = water_atoms()
    atoms.calc = DftRs(units="atomic", max_iter=1)
    with pytest.raises(RuntimeError, match="SCF did not converge"):
        atoms.get_potential_energy()
    assert "energy" not in atoms.calc.results
    result = cli("energy", str(xyz(tmp_path)), "--max-iter", "1", "--format", "json")
    assert result.returncode != 0
    assert "SCF did not converge" in result.stderr
    assert not result.stdout.strip(), "a failed SCF emitted a successful calculation report"

