# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Run-local logging must reach the real core and leave structured output intact."""
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from dft_rs import DftRs, native
from dft_rs.units import BOHR_IN_ANGSTROM, HARTREE_IN_EV

NUMBERS = [1, 1]
POSITIONS = [[0.1, 0.2, 0.3], [0.5, 0.7, 1.5]]


def require_logging_core():
    assert hasattr(native._core, "_logging_active"), "scoped-logging acceptance requires the newly built core"


def test_frontend_forwards_path_and_level_without_consuming_calculation_arguments(tmp_path, monkeypatch):
    calls = []

    def execute(function, args, kwargs, **options):
        calls.append(options)
        return function(*args, **kwargs)

    monkeypatch.setattr(native, "_core", SimpleNamespace(_with_resources=execute, _logging_active=lambda: False))
    function = native._reporting(lambda value, *, scale=1: value * scale)
    path = tmp_path / "calculation.log"
    assert function(3, scale=2, log_file=path, log_level="DEBUG", cpu_threads=1) == 6
    assert calls == [dict(cpu_threads=1, log_file=str(path), log_level="debug")]
    assert not path.exists(), "this plumbing-only probe must not pretend to run a native logger"


@pytest.mark.parametrize("options", [dict(log_file=True), dict(log_file=b"bytes.log"),
                                     dict(log_file=""), dict(log_mode="typo"), dict(log_level=2)])
def test_frontend_rejects_invalid_logging_types_before_dispatch(options, monkeypatch):
    monkeypatch.setattr(native, "_core", SimpleNamespace())
    with pytest.raises(ValueError, match="log_"):
        native._reporting(lambda: pytest.fail("invalid configuration reached calculation"))(**options)


def test_frontend_refuses_an_old_core_instead_of_silently_ignoring_a_log_file(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "_core", SimpleNamespace(_with_resources=lambda *a, **k: None))
    with pytest.raises(NotImplementedError, match="scoped logging"):
        native._reporting(lambda: None)(log_file=tmp_path / "requested.log")


def test_frontend_ase_constructor_limits_survive_an_outer_logging_only_scope(monkeypatch):
    from ase import Atoms

    calls = []

    def execute(function, args, kwargs, **options):
        calls.append(options)
        return function(*args, **kwargs)

    monkeypatch.setattr(native._core, "_with_resources", execute)
    monkeypatch.setattr(native._core, "_resource_usage", lambda: {"cpu_threads": 8, "memory_mb": None})
    monkeypatch.setattr(native._core, "_resource_scope_configured", lambda: False, raising=False)
    monkeypatch.setattr(native._core, "_logging_active", lambda: True, raising=False)
    monkeypatch.setattr(native._core, "run_feature", lambda *args, **kwargs: {
        "feature": "continuous-species", "energy": -1.0, "converged": True, "units": {}})
    calc = DftRs(units="atomic", cpu_threads=2, memory_mb=8, log_file="outer.log")
    result = calc.get_reduced_fragment(Atoms("H"), model="opaque forwarding probe")
    assert result["energy"] == -1.0
    assert calls == [dict(cpu_threads=2, memory_mb=8)]
    assert calc.parameters["cpu_threads"] == 2


def test_public_callable_discovery_includes_logging_and_resource_options():
    for name in native.__all__:
        function = getattr(native, name)
        if inspect.isfunction(function):
            assert {"log_file", "log_mode", "log_level", "cpu_threads", "memory_mb"} <= set(inspect.signature(function).parameters), name
    for name in ("calculate", "get_hessian", "get_reduced_fragment"):
        assert {"log_file", "log_mode", "log_level"} <= set(inspect.signature(getattr(DftRs, name)).parameters)


def test_native_real_scf_captures_progress_units_and_equivalent_concurrent_results(tmp_path):
    require_logging_core()

    def calculate(threads):
        path = tmp_path / f"native-{threads}.log"
        result = native.energy(NUMBERS, POSITIONS, xc="hf", basis="sto-3g",
                               log_file=path, cpu_threads=threads, memory_mb=128)
        text = path.read_text()
        for expected in ("[scf] iter", "result energy=", "hartree", "bohr", "status=completed"):
            assert expected in text
        return result.energy, {line.split()[1] for line in text.splitlines()}

    with ThreadPoolExecutor(max_workers=2) as pool:
        left, right = list(pool.map(calculate, [1, 2]))
    assert left[0] == pytest.approx(right[0], abs=1e-12)
    assert len(left[1]) == len(right[1]) == 1 and left[1].isdisjoint(right[1])
    assert not native._core._logging_active()


def test_actual_ase_scf_uses_the_same_log_scope_and_converts_only_physical_results(tmp_path):
    require_logging_core()
    from ase import Atoms

    atoms = Atoms(numbers=NUMBERS, positions=np.asarray(POSITIONS) * BOHR_IN_ANGSTROM)
    path = tmp_path / "ase.log"
    atoms.calc = DftRs(xc="hf", basis="sto-3g", cpu_threads=1, log_file=path)
    energy = atoms.get_potential_energy()
    reference = native.energy(NUMBERS, POSITIONS, xc="hf", basis="sto-3g", cpu_threads=1)
    assert energy == pytest.approx(reference.energy * HARTREE_IN_EV, abs=1e-10)
    text = path.read_text()
    assert text.count("run started") == 1 and "[scf] iter" in text
    assert "hartree" in text and "status=completed" in text


def test_actual_logging_only_envelope_preserves_nested_resource_settings(tmp_path):
    require_logging_core()
    path = tmp_path / "nested.log"

    def calculate():
        return native.energy(NUMBERS, POSITIONS, xc="hf", cpu_threads=1, memory_mb=128)

    result = native._reporting(calculate)(log_file=path)
    assert result.resources["cpu_threads"] == 1
    assert result.resources["memory_mb"] == 128
    assert path.read_text().count("run started") == 1


def test_actual_cli_library_entry_saves_independent_repeated_runs(tmp_path, capfd):
    require_logging_core()
    xyz = tmp_path / "molecule.xyz"
    xyz.write_text("2\nlogging fixture\n" + "".join("H " + " ".join(map(str, row)) + "\n"
                   for row in np.asarray(POSITIONS) * BOHR_IN_ANGSTROM))
    identities = []
    for index in range(2):
        path = tmp_path / f"library-cli-{index}.log"
        status = native.cli_main(["energy", str(xyz), "--xc", "hf", "--format", "json",
                                  "--cpu-threads", "1"], log_file=path)
        stdout, stderr = capfd.readouterr()
        assert status == 0, stderr
        assert json.loads(stdout)["converged"] is True
        text = path.read_text()
        assert "[scf] iter" in text and "status=completed" in text
        identities.append(text.splitlines()[0].split()[1])
    assert identities[0] != identities[1]
    assert not xyz.with_suffix(".dft.log").exists()
    failed = tmp_path / "library-cli-invalid.log"
    status = native.cli_main(["energy", str(xyz), "--log-level", "invalid"], log_file=failed)
    stdout, stderr = capfd.readouterr()
    assert status == 1 and not stdout and "log_level" in stderr
    assert "log_level" in failed.read_text() and "status=failed" in failed.read_text()


def test_actual_nonconvergence_and_cancellation_preserve_partial_records(tmp_path):
    require_logging_core()
    failed = tmp_path / "unconverged.log"
    with pytest.raises(RuntimeError, match="converge"):
        native.energy([2, 1], POSITIONS, charge=1, xc="hf", basis="sto-3g", max_iter=1,
                      cpu_threads=1, log_file=failed)
    text = failed.read_text()
    assert "[scf] iter" in text and "status=failed" in text
    cancelled = tmp_path / "cancelled.log"

    def interrupt():
        raise KeyboardInterrupt("fixture cancellation")

    with pytest.raises(KeyboardInterrupt, match="fixture cancellation"):
        native._reporting(interrupt)(log_file=cancelled, cpu_threads=1)
    assert "fixture cancellation" in cancelled.read_text() and "status=failed" in cancelled.read_text()


def test_actual_open_failure_is_explicit_and_default_does_not_create_files(tmp_path, monkeypatch):
    require_logging_core()
    monkeypatch.chdir(tmp_path)
    with pytest.raises((OSError, RuntimeError), match="log_file"):
        native.energy(NUMBERS, POSITIONS, xc="hf", log_file=Path("missing-parent/run.log"), cpu_threads=1)
    assert not list(tmp_path.iterdir())
    native.energy(NUMBERS, POSITIONS, xc="hf", cpu_threads=1)
    assert not list(tmp_path.iterdir())


def test_actual_cli_keeps_json_stdout_and_scf_stderr_while_saving_the_run(tmp_path):
    require_logging_core()
    cli = os.environ.get("DFT_RS_CLI") or shutil.which("dft-rs")
    assert cli, "actual logging acceptance requires one supported CLI binary"
    xyz = tmp_path / "molecule.xyz"
    xyz.write_text("2\nlogging fixture\n" + "".join("H " + " ".join(map(str, row)) + "\n"
                   for row in np.asarray(POSITIONS) * BOHR_IN_ANGSTROM))
    path = tmp_path / "cli.log"
    result = subprocess.run([cli, "energy", str(xyz), "--xc", "hf", "--cpu-threads", "1",
                             "--format", "json", "--log-file", str(path), "--log-mode", "both"],
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["converged"] is True
    assert "[scf] iter" in result.stderr and "[scf] iter" in path.read_text()
    assert "status=completed" in path.read_text()


@pytest.mark.parametrize("reference", ["uks", "roks", "gks"])
def test_actual_ammonia_cation_logs_total_spin(reference, tmp_path):
    from ase import Atoms

    require_logging_core()
    cli = os.environ.get("DFT_RS_CLI") or shutil.which("dft-rs")
    assert cli, "spin diagnostics acceptance requires the canonical CLI"
    numbers = [7, 1, 1, 1]
    positions = [[0.04, -0.06, 0.18], [1.59, 0.09, -0.60],
                 [-0.74, 1.53, -0.55], [-0.85, -1.40, -0.65]]
    atoms = Atoms(numbers, positions=np.asarray(positions) * BOHR_IN_ANGSTROM)
    controls = dict(xc="wb97x-d", basis="sto-3g", charge=1, multiplicity=2,
                    reference=reference, cpu_threads=1)
    native_path = tmp_path / "native.log"
    native_report = native.run_feature("energy", numbers, positions, log_file=native_path, **controls)
    ase_path = tmp_path / "ase.log"
    ase_report = DftRs(units="atomic").get_energy_report(atoms, log_file=ase_path, **controls)
    cli_path = tmp_path / "cli.log"
    xyz = tmp_path / "ammonia_cation.xyz"
    xyz.write_text("4\nDistorted ammonia cation\n" + "\n".join(
        f"{symbol} {x:.17g} {y:.17g} {z:.17g}"
        for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions)
    ) + "\n", encoding="ascii")
    result = subprocess.run([
        cli, "energy", str(xyz), "--xc", "wb97x-d", "--basis", "sto-3g",
        "--charge", "1", "--multiplicity", "2", "--reference", reference,
        "--cpu-threads", "1", "--format", "json", "--log-mode", "both", "--log-file", str(cli_path),
    ], capture_output=True, text=True, timeout=300, check=True)
    cli_report = json.loads(result.stdout)
    for text in (result.stderr, *(p.read_text() for p in (native_path, ase_path, cli_path))):
        assert f"[{reference}] <S^2>=" in text
        if reference == "gks":
            assert "[gks] total-spin sectors" in text
    for report in (native_report, ase_report, cli_report):
        assert report["reference_type"] == reference and report["converged"]
        assert abs(report["s_squared"] - native_report["s_squared"]) < 1e-10
        if reference == "gks":
            sectors = report["spin_sector_weights"]
            assert abs(sum(p["weight"] for p in sectors) - 1.0) < 1e-10
            assert abs(sum(p["weight"] * p["spin"] * (p["spin"] + 1.0)
                           for p in sectors) - report["s_squared"]) < 1e-10


@pytest.mark.parametrize("mode", ["default", "file", "none"])
def test_actual_legacy_text_cli_log_defaults_and_overrides(tmp_path, mode):
    require_logging_core()
    cli = os.environ.get("DFT_RS_CLI") or shutil.which("dft-rs")
    assert cli, "legacy logging acceptance requires a supported CLI binary"
    xyz = tmp_path / "legacy.xyz"
    xyz.write_text("2\nlegacy logging fixture\n" + "".join(
        "H " + " ".join(map(str, row)) + "\n" for row in np.asarray(POSITIONS) * BOHR_IN_ANGSTROM))
    default_log = xyz.with_suffix(".dft.log")
    selected_log = tmp_path / "selected.log"
    command = [cli, "energy", str(xyz), "--xc", "hf", "--basis", "sto-3g", "--cpu-threads", "1"]
    if mode == "file":
        command += ["--log-file", str(selected_log)]
    elif mode == "none":
        command += ["--log-mode", "none"]
    result = subprocess.run(command, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip(), "the legacy text calculation must retain its stdout report"
    if mode == "none":
        assert not default_log.exists() and not selected_log.exists()
        assert "[scf] iter" not in result.stderr
    else:
        path = default_log if mode == "default" else selected_log
        text = path.read_text()
        assert "[scf] iter" in text and "status=completed" in text
        assert text.count("run started") == 1
        assert ("[scf] iter" in result.stderr) == (mode == "default")
        assert not (selected_log if mode == "default" else default_log).exists()
