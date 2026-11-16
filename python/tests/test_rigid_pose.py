# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Actual explicit-primitive routes; identity fixtures do not validate chemistry."""
import copy
import json
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree


POSITIONS = [[-0.7, 0.23, -0.15], [1.1, -0.31, 0.42]]


@pytest.fixture(scope="module")
def model():
    # Repository input, never a developer-machine path or generated private fit.
    path = Path(__file__).resolve().parents[2] / "tests/data/rigid_pose_identity.json"
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(model, **controls):
    options = dict(cpu_threads=1, memory_mb=384)
    options.update(controls)
    return dft_rs.evaluate_reduced_fragment(
        [1, 1], POSITIONS, model=model, fragment_mode="rigid-evaluate",
        **options)


@pytest.fixture(scope="module")
def analytical(model):
    return evaluate(model, derivative_order=2)


def test_native_relaxed_blocks_and_e0_accounting(model, analytical):
    assert analytical["experimental"] is True
    assert analytical["chemical_accuracy"] == "not_established"
    assert analytical["converged"] is True
    assert analytical["response_paths"] == 78
    assert analytical["ao_count"] == 2
    assert analytical["generation"] == model["generation"]
    assert analytical["internal_energy_reference"] == pytest.approx(0.2, abs=1e-15)
    assert analytical["energy"] == pytest.approx(
        analytical["electronic_energy"] + analytical["classical_energy"], abs=1e-12)
    for key, shape in [("center_gradient", (2, 3)), ("euler_gradient", (2, 3)),
                       ("center_hessian", (6, 6)), ("center_angle_hessian", (6, 6)),
                       ("angle_hessian", (6, 6))]:
        assert np.shape(analytical[key]) == shape
        assert np.isfinite(analytical[key]).all()
    np.testing.assert_allclose(analytical["forces"], -np.asarray(analytical["center_gradient"]), atol=1e-14)
    for key in ["euler_gradient", "center_angle_hessian", "angle_hessian"]:
        np.testing.assert_allclose(analytical[key], 0, atol=1e-11)


def test_ase_converts_each_center_angle_unit_and_preserves_opaque_recipe(model, analytical):
    before = copy.deepcopy(model)
    atoms = Atoms(numbers=[1, 1], positions=np.asarray(POSITIONS) * Bohr)
    report = dft_rs.DftRs().get_reduced_fragment(
        atoms, model=model, fragment_mode="rigid-evaluate", derivative_order=2, cpu_threads=1)
    for key, factor in [("energy", Hartree), ("center_gradient", Hartree/Bohr),
                        ("forces", Hartree/Bohr), ("euler_gradient", Hartree),
                        ("center_hessian", Hartree/Bohr**2),
                        ("center_angle_hessian", Hartree/Bohr), ("angle_hessian", Hartree)]:
        np.testing.assert_allclose(report[key], np.asarray(analytical[key])*factor, atol=1e-8, rtol=1e-9)
    assert report["units"]["center_hessian"] == "eV/Angstrom^2"
    assert report["units"]["center_angle_hessian"] == "eV/Angstrom/radian"
    assert report["units"]["angle_hessian"] == "eV/radian^2"
    assert model == before


def test_nonzero_mixed_unit_blocks_use_one_length_factor_per_center_index():
    from dft_rs.ase_calculator import _to_ase_units

    # A pure conversion oracle: these nonzero arrays are not claimed to be a
    # physical Hessian. Zero angle blocks in the identity fixture cannot test
    # a missing or duplicated length factor by themselves.
    values = np.array([[0.13, -0.27], [0.41, -0.53]])
    atomic = dict(center_hessian=values.tolist(), center_angle_hessian=values.tolist(),
                  angle_hessian=values.tolist(), units=dict(center_hessian="hartree/bohr^2",
                  center_angle_hessian="hartree/bohr/radian", angle_hessian="hartree/radian^2"))
    before = copy.deepcopy(atomic)
    external = _to_ase_units(atomic)
    for key, factor in [("center_hessian", Hartree/Bohr**2),
                        ("center_angle_hessian", Hartree/Bohr), ("angle_hessian", Hartree)]:
        np.testing.assert_allclose(external[key], values*factor, rtol=1e-14, atol=1e-14)
    assert atomic == before


@pytest.mark.parametrize("reference", ["uks", "uhf", "roks", "rohf", "gks", "ghf"])
def test_known_unsupported_references_do_not_fallback(model, reference):
    with pytest.raises(NotImplementedError):
        evaluate(model, reference=reference, derivative_order=0)


@pytest.mark.parametrize("version", ["static-single-port-v2", "static-monovalent-region-v3", "few-center-model-v1"])
def test_distinct_schemas_are_not_relabelled_or_implicitly_loaded(model, version):
    record = copy.deepcopy(model)
    record["model_version"] = version
    with pytest.raises(NotImplementedError):
        evaluate(record, derivative_order=0)


@pytest.mark.parametrize("value", [True, 1.5, -1])
def test_derivative_order_requires_a_nonnegative_integer(model, value):
    with pytest.raises((ValueError, TypeError)):
        evaluate(model, derivative_order=value)


def test_unsupported_order_and_periodic_ase_are_not_implemented(model):
    with pytest.raises(NotImplementedError):
        evaluate(model, derivative_order=3)
    atoms = Atoms("HH", positions=np.asarray(POSITIONS)*Bohr, cell=[8, 8, 8], pbc=True)
    with pytest.raises(NotImplementedError):
        dft_rs.DftRs().get_reduced_fragment(atoms, model=model, fragment_mode="rigid-evaluate")


def test_nonconvergence_and_io_errors_retain_partial_logs(model, tmp_path):
    path = tmp_path / "failed-scf.log"
    with pytest.raises(RuntimeError):
        evaluate(model, derivative_order=1, max_iter=1, log_file=str(path), log_mode="file")
    text = path.read_text(encoding="utf-8")
    assert "[scf] iter" in text and "status=failed" in text
    with pytest.raises(OSError):
        evaluate(model, derivative_order=0, log_file=str(tmp_path / "missing" / "run.log"))


def test_independent_calls_keep_settings_and_logs_separate(model, tmp_path):
    def task(threads):
        path = tmp_path / f"threads-{threads}.log"
        controls = dict(derivative_order=1, log_file=str(path), log_mode="file", cpu_threads=threads)
        report = dft_rs.evaluate_reduced_fragment([1, 1], POSITIONS, model=model,
                    fragment_mode="rigid-evaluate", **controls)
        return report, path.read_text(encoding="utf-8")
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(task, [1, 2]))
    np.testing.assert_allclose(results[0][0]["center_gradient"], results[1][0]["center_gradient"], atol=1e-10)
    for report, text in results:
        assert report["converged"] is True
        assert "status=completed" in text and "[scf] iter" in text


def test_both_cli_entries_preserve_json_stdout_and_capability_failures(model, tmp_path):
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for actual explicit-primitive acceptance"
    xyz = tmp_path / "retained-centers.xyz"
    xyz.write_text("2\nSynthetic identity\n" + "".join(
        f"H {x*Bohr:.17g} {y*Bohr:.17g} {z*Bohr:.17g}\n" for x, y, z in POSITIONS), encoding="utf-8")
    # The declared Python CLI is the installed console script. The package
    # does not advertise a python -m dft_rs entry point.
    console = Path(os.environ.get("DFT_RS_PYTHON", os.sys.executable)).with_name("dft-rs")
    commands = [[executable], [str(console)]]
    for command in commands:
        args = command + ["continuous-species", str(xyz), "--format", "json",
            "--fragment-mode", "rigid-evaluate", "--fragment-model-json", json.dumps(model),
            "--derivative-order", "1", "--cpu-threads", "1"]
        completed = subprocess.run(args, capture_output=True, text=True, check=False)
        assert completed.returncode == 0, completed.stderr
        report = json.loads(completed.stdout)
        assert report["converged"] is True and "center_gradient" in report
        assert "[scf]" in completed.stderr
        failed = subprocess.run(args + ["--reference", "gks"], capture_output=True, text=True, check=False)
        assert failed.returncode != 0
        assert "[scf] iter" not in failed.stderr
