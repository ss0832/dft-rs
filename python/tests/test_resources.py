# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Resource forwarding contracts and real scoped execution acceptance."""
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs import native

GEOMETRY = ([1, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])


def test_legacy_resource_keywords_are_visible_to_signature_discovery():
    functions = [function for name, function in vars(native).items()
                 if not name.startswith("_") and inspect.isfunction(function)
                 and function.__module__ == native.__name__]
    functions += [method for name, method in vars(dft_rs.DftRs).items()
                  if callable(method) and (name.startswith("get_") or name == "calculate")]
    for function in functions:
        signature = inspect.signature(function)
        for name in ("cpu_threads", "memory_mb"):
            assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY, function.__name__
            assert signature.parameters[name].default is None
    assert dft_rs.DftRs.default_parameters["cpu_threads"] is None
    assert dft_rs.DftRs.default_parameters["memory_mb"] is None


@pytest.mark.parametrize("name,value", [("cpu_threads", 0), ("cpu_threads", -1),
    ("cpu_threads", 1.5), ("cpu_threads", True), ("cpu_threads", "2"),
    ("memory_mb", 0), ("memory_mb", 1 << 100)])
def test_invalid_limits_fail_before_any_backend_call(name, value):
    with pytest.raises(ValueError, match=name):
        native.calculate(*GEOMETRY, **{name: value})


def test_legacy_native_and_ase_forward_one_envelope_and_per_call_overrides(monkeypatch):
    active = []
    calls = []

    def execute(function, args, kwargs, **limits):
        calls.append(dict(limits))
        active.append(limits)
        try:
            return function(*args, **kwargs)
        finally:
            active.pop()

    def calculate(numbers, positions, **controls):
        assert active and "cpu_threads" not in controls and "memory_mb" not in controls
        return {"energy": -1.0, "dipole": [0.0] * 3, "mulliken": [0.0] * len(numbers),
                "converged": True, "iterations": 1}

    def driver(feature, numbers, positions, **controls):
        assert active and "cpu_threads" not in controls and "memory_mb" not in controls
        return {"feature": feature, "energy": -1.0, "units": {}, "converged": True}

    monkeypatch.setattr(native._core, "_with_resources", execute, raising=False)
    monkeypatch.setattr(native._core, "_resource_usage", lambda: active[-1] if active else None, raising=False)
    monkeypatch.setattr(native._core, "_resource_scope_configured", lambda: bool(active), raising=False)
    monkeypatch.setattr(native._core, "calculate", calculate)
    monkeypatch.setattr(native._core, "run_feature", driver)
    assert native.calculate(*GEOMETRY, cpu_threads=2, memory_mb=8)["energy"] == -1.0
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    calc = dft_rs.DftRs(units="atomic", cpu_threads=2, memory_mb=8)
    atoms.calc = calc
    atoms.get_potential_energy()
    assert calc.get_energy_report(atoms, cpu_threads=1, memory_mb=4)["energy"] == -1.0
    assert calls == [{"cpu_threads": 2, "memory_mb": 8}, {"cpu_threads": 2, "memory_mb": 8},
                     {"cpu_threads": 1, "memory_mb": 4}]
    assert calc.parameters["cpu_threads"] == 2 and calc.parameters["memory_mb"] == 8


def test_old_extensions_cannot_silently_ignore_requested_limits(monkeypatch):
    monkeypatch.setattr(native._core, "_with_resources", None, raising=False)
    with pytest.raises(NotImplementedError, match="rebuild"):
        native.energy(*GEOMETRY, cpu_threads=1)


def test_saved_model_getter_inherits_per_call_limits_without_reapplying_constructor_defaults(monkeypatch):
    usages = []

    def driver(feature, numbers, positions, **controls):
        assert feature == "continuous-species" and controls["fragment_mode"] == "evaluate"
        assert "cpu_threads" not in controls and "memory_mb" not in controls
        usages.append(native._core._resource_usage())
        return {"feature": feature, "energy": -1.0, "converged": True, "units": {}}

    monkeypatch.setattr(native._core, "run_feature", driver)
    calc = dft_rs.DftRs(units="atomic", cpu_threads=2, memory_mb=8)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    overridden = calc.get_reduced_fragment(atoms, model="opaque forwarding probe", cpu_threads=1, memory_mb=4)
    default = calc.get_reduced_fragment(atoms, model="opaque forwarding probe")
    assert usages == [{"cpu_threads": 1, "memory_mb": 4}, {"cpu_threads": 2, "memory_mb": 8}]
    assert overridden["resources"]["cpu_threads"] == 1 and default["resources"]["cpu_threads"] == 2
    assert calc.parameters["cpu_threads"] == 2 and calc.parameters["memory_mb"] == 8


def test_real_native_ase_and_cli_limits_are_discoverable_and_numerically_equivalent(tmp_path):
    registry = native.feature_registry()
    assert registry["control_units"]["cpu_threads"] == "integer"
    assert registry["control_units"]["memory_mb"] == "MiB"
    baseline = native.energy(*GEOMETRY)
    for threads in (1, 3):
        report = native.energy(*GEOMETRY, cpu_threads=threads, memory_mb=32)
        assert report["resources"]["cpu_threads"] == threads
        assert 0 < report["resources"]["managed_peak_bytes"] <= 32 * 1024**2
        assert report["energy"] == pytest.approx(baseline["energy"], abs=1e-11)
        legacy = native.calculate(*GEOMETRY, cpu_threads=threads, memory_mb=32)
        assert legacy["energy"] == pytest.approx(baseline["energy"], abs=1e-11)
    atoms = Atoms("H2", positions=np.asarray(GEOMETRY[1]) * dft_rs.units.BOHR_IN_ANGSTROM)
    calc = dft_rs.DftRs(units="atomic", cpu_threads=1, memory_mb=32)
    assert calc.get_energy_report(atoms)["energy"] == pytest.approx(baseline["energy"], abs=1e-11)
    path = tmp_path / "h2.xyz"
    path.write_text("2\nresources\nH 0 0 0\nH 0 0 0.7408480952642\n", encoding="utf-8")
    cli = os.environ["DFT_RS_CLI"]
    for command in ("energy", "nmr"):
        result = subprocess.run([cli, command, str(path), "--cpu-threads", "1", "--memory-mb", "32"],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    alias = Path(cli).with_name("dft_rs_cli.exe" if os.name == "nt" else "dft_rs_cli")
    for executable in (cli, str(alias)):
        result = subprocess.run([executable, "energy", str(path), "--cpu-threads", "1",
                                 "--memory-mb", "32", "--format", "json"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["resources"]["cpu_threads"] == 1 and report["resources"]["memory_mb"] == 32
        assert 0 < report["resources"]["managed_peak_bytes"] <= 32 * 1024**2
        assert report["energy"] == pytest.approx(baseline["energy"], abs=1e-11)
    fitter = Path(cli).with_name("fit_substituent.exe" if os.name == "nt" else "fit_substituent")
    result = subprocess.run([str(fitter), "--synthetic-check", "--cpu-threads", "1", "--memory-mb", "32"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(fitter), "--synthetic-check", "--cpu-threads", "0"],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "cpu_threads" in result.stderr


def test_real_independent_concurrent_calculations_keep_their_limits():
    def calculate(threads):
        return native.energy(*GEOMETRY, cpu_threads=threads, memory_mb=32)
    with ThreadPoolExecutor(max_workers=2) as executor:
        reports = list(executor.map(calculate, (1, 3)))
    assert [report["resources"]["cpu_threads"] for report in reports] == [1, 3]
    assert reports[0]["energy"] == pytest.approx(reports[1]["energy"], abs=1e-11)


def test_real_tiny_memory_budget_and_overflow_fail_clearly():
    with pytest.raises((RuntimeError, ValueError), match="(memory|MiB|allocation)"):
        native.energy([8, 1, 1], [[0, 0, 0], [0, 1.4, 1.1], [0, -1.4, 1.1]],
                      basis="def2-tzvp", cpu_threads=1, memory_mb=1)
    with pytest.raises(ValueError, match="memory_mb"):
        native.energy(*GEOMETRY, memory_mb=(1 << 63) - 1)


@pytest.mark.parametrize("reference", ["rks", "rhf", "uks", "uhf", "auto"])
def test_real_core_driver_resource_globals_survive_reference_normalization(reference):
    report = native._core.run_feature("energy", *GEOMETRY, reference=reference,
                                      cpu_threads=2, memory_mb=32)
    assert report["resources"]["cpu_threads"] == 2
    assert report["resources"]["memory_mb"] == 32
    baseline = native.energy(*GEOMETRY, reference=reference)
    assert report["energy"] == pytest.approx(baseline["energy"], abs=1e-11)


def test_real_explicit_null_resource_controls_do_not_leak_into_feature_adapters():
    report = native._core.run_feature("energy", *GEOMETRY, reference="rks",
                                      cpu_threads=None, memory_mb=None)
    assert "resources" not in report
    with pytest.raises(ValueError, match="unknown_contract_control"):
        native._core.run_feature("energy", *GEOMETRY, reference="rks", cpu_threads=2,
                                 memory_mb=32, unknown_contract_control=True)


@pytest.mark.parametrize("name,value", [("cpu_threads", 1.0), ("memory_mb", 32.0)])
def test_real_core_resource_integer_contract_matches_native_and_cli(name, value):
    with pytest.raises(ValueError, match=name):
        native._core.run_feature("energy", *GEOMETRY, **{name: value})


@pytest.mark.parametrize("surface", ["native", "ase"])
@pytest.mark.parametrize("name,value", [
    ("cpu_threads", 0), ("cpu_threads", True), ("cpu_threads", 1.0),
    ("cpu_threads", 1 << 100), ("memory_mb", False), ("memory_mb", -1),
    ("memory_mb", 4.0), ("memory_mb", 1 << 100),
])
def test_few_mode_invalid_resources_never_reach_dispatch(monkeypatch, surface, name, value):
    calls = []
    monkeypatch.setattr(native._core, "run_feature", lambda *a, **kw: calls.append(kw))
    controls = {"model": "opaque", "fragment_mode": "few-evaluate", name: value}
    with pytest.raises(ValueError, match=name):
        if surface == "native":
            native.evaluate_reduced_fragment([1], [[0., 0., 0.]], **controls)
        else:
            dft_rs.DftRs().get_reduced_fragment(Atoms("H"), **controls)
    assert calls == []


@pytest.mark.parametrize("surface", ["native", "ase"])
def test_few_mode_backend_nonconvergence_propagates_and_releases_scope(monkeypatch, surface):
    def fail(*args, **kwargs):
        assert native._core._resource_usage() == {"cpu_threads": 1, "memory_mb": 4}
        raise RuntimeError("SCF did not converge: retained sentinel")

    monkeypatch.setattr(native._core, "run_feature", fail)
    controls = dict(model="opaque", fragment_mode="few-evaluate", cpu_threads=1, memory_mb=4)
    calc = dft_rs.DftRs()
    with pytest.raises(RuntimeError, match="retained sentinel"):
        if surface == "native":
            native.evaluate_reduced_fragment([1], [[0., 0., 0.]], **controls)
        else:
            calc.get_reduced_fragment(Atoms("H"), **controls)
    assert "energy" not in calc.results
    assert native._core._resource_usage() is None
