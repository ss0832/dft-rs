# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Timing acceptance must detect changed physical observables."""

import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).resolve().parents[2] / "tools" / "bench.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize("component", ["quadrupole_polarizability_real",
                                      "quadrupole_polarizability_imag"])
def test_benchmark_checks_off_diagonal_complex_components(component):
    reference = {"energy": -75.0, component: [[1.0, 0.02], [0.02, 2.0]]}
    changed = {"energy": -75.0, component: [[1.0, 0.03], [0.02, 2.0]]}
    mismatch = benchmark._observable_mismatch(changed, reference, 1e-8, 1e-8)
    assert component + "[0][1]" in mismatch
    assert benchmark._observable_mismatch(reference, reference, 1e-8, 1e-8) is None


@pytest.mark.parametrize("actual", [float("nan"), float("inf"), True, None])
def test_benchmark_rejects_nonfinite_or_missing_values(actual):
    assert benchmark._observable_mismatch({"energy": actual}, {"energy": -1.0}, 1e-8, 1e-8)


def test_benchmark_rejects_missing_components_and_shape_changes():
    reference = {"tensor": [[1.0, 2.0], [2.0, 3.0]]}
    assert benchmark._observable_mismatch({}, reference, 1e-8, 1e-8)
    assert benchmark._observable_mismatch({"tensor": [[1.0, 2.0]]}, reference, 1e-8, 1e-8)
    assert benchmark._observable_mismatch({"tensor": [[1.0, 2.0], [2.0, 3.0 + 1e-10]]},
                                           reference, 1e-8, 1e-8) is None


@pytest.mark.parametrize("key", ["forces", "force_constants", "raw_force_constants",
                                 "supercell_force_constants", "force_constant_tensor",
                                 "contracted_force_constants", "dispersion_force_constants",
                                 "dispersion_force_constant_tensor", "elastic_constants_clamped",
                                 "elastic_constants_relaxed", "high_order_clamped",
                                 "high_order_relaxed", "internal_relaxation",
                                 "second_internal_relaxation"])
def test_periodic_timing_rejects_changed_response_at_identical_energy(key):
    import json
    import sys

    report = {"feature": "phonons", "converged": True, "energy": -2.0,
              key: [[0.1, 0.02], [0.02, 0.3]], "path_evaluations": 4,
              "resources": {"managed_peak_bytes": 0}}
    trial = benchmark._measure([sys.executable, "-c", "import sys; print(sys.argv[1])",
                                json.dumps(report)])
    assert "error" not in trial
    assert trial["path_evaluations"] == 4
    changed = dict(trial["observables"], **{key: [[0.1, 0.03], [0.02, 0.3]]})
    assert key + "[0][1]" in benchmark._observable_mismatch(
        changed, trial["observables"], 1e-8, 1e-8)


def test_periodic_timing_rejects_unconverged_response():
    import json
    import sys

    report = {"feature": "periodic-quartic-fc-along", "converged": True,
              "response_converged": False, "energy": -2.0,
              "contracted_force_constants": [[0.1]],
              "resources": {"managed_peak_bytes": 0}}
    trial = benchmark._measure([sys.executable, "-c", "import sys; print(sys.argv[1])",
                                json.dumps(report)])
    assert trial["error"] == "response did not converge"


def test_elastic_timing_checks_ordered_curvatures_and_retains_response_solves():
    import json
    import sys

    report = {"feature": "elastic", "converged": True, "energy": -7.04,
              "strain_hessian": [[0.0] * 9 for _ in range(9)],
              "strain_solves": [{"iterations": 4, "relative_residual": 2e-12}],
              "displacement_solves": [], "resources": {"managed_peak_bytes": 0}}
    for key in ("stress", "strain_derivative", "internal_strain", "internal_relaxation",
                "force_constants", "clamped_ion", "relaxed_ion", "clamped_ion_birch_gpa",
                "relaxed_ion_birch_gpa"):
        report[key] = [[0.0]]
    trial = benchmark._measure([sys.executable, "-c", "import sys; print(sys.argv[1])",
                                json.dumps(report)])
    assert "error" not in trial
    assert trial["residuals"]["strain_solves"] == report["strain_solves"]
    changed_hessian = [row[:] for row in trial["observables"]["strain_hessian"]]
    changed_hessian[2][4] = 0.1
    changed = dict(trial["observables"], strain_hessian=changed_hessian)
    assert "strain_hessian[2][4]" in benchmark._observable_mismatch(
        changed, trial["observables"], 1e-8, 1e-8)


def test_generalized_gradient_timing_compares_each_energy_term():
    import json
    import sys

    report = {"feature": "gradient", "converged": True, "energy": -7.93,
              "response_residuals": [1e-14], "resources": {"managed_peak_bytes": 0}}
    for key in ("gradient", "forces", "parent_gradient", "generalized_pt2_gradient",
                "pt2_gradient", "target_gradient", "dispersion_gradient"):
        report[key] = [[0.01, 0.02, 0.03]]
    trial = benchmark._measure([sys.executable, "-c", "import sys; print(sys.argv[1])",
                                json.dumps(report)])
    assert "error" not in trial
    assert trial["residuals"]["response_residuals"] == [1e-14]
    changed = dict(trial["observables"], generalized_pt2_gradient=[[0.01, 0.09, 0.03]])
    assert "generalized_pt2_gradient[0][1]" in benchmark._observable_mismatch(
        changed, trial["observables"], 1e-8, 1e-8)


def test_nmr_timing_compares_couplings_even_when_energy_is_unchanged():
    import json
    import sys

    report = {"feature": "nmr-spectrum", "converged": True, "energy": -75.0,
              "shielding_tensors": [[[1.0, 0.2], [0.2, 2.0]]],
              "diamagnetic_tensors": [[[1.1, 0.1], [0.1, 2.1]]],
              "paramagnetic_tensors": [[[-0.1, 0.1], [0.1, -0.1]]],
              "chemical_shifts": [1.3, 2.4], "coupling_pairs": [[0, 1]],
              "coupling_constants_hz": [7.2], "spectrum_axis": [[1.0, 2.0]],
              "spectrum_intensity": [[0.5, 0.7]], "spin_hilbert_dimension": 4,
              "resources": {"managed_peak_bytes": 0}}
    trial = benchmark._measure([sys.executable, "-c", "import sys; print(sys.argv[1])",
                                json.dumps(report)])
    assert "error" not in trial
    changed = dict(trial["observables"], coupling_constants_hz=[7.3])
    mismatch = benchmark._observable_mismatch(changed, trial["observables"], 1e-8, 1e-8)
    assert "coupling_constants_hz[0]" in mismatch


def test_hessian_timing_checks_every_tensor_and_fresh_residual():
    import json
    import sys

    report = {"feature": "hessian", "converged": True, "energy": -2.0,
              "hessian": [[0.1, 0.02], [0.02, 0.3]],
              "dipole_derivatives": [[0.4, 0.01]],
              "orbital_response_residuals": [1e-12, 2e-12],
              "scf_orbital_residual": 3e-13, "resources": {"managed_peak_bytes": 0}}
    trial = benchmark._measure([sys.executable, "-c", "import sys; print(sys.argv[1])",
                                json.dumps(report)])
    assert "error" not in trial
    for key, changed_value, location in [
        ("hessian", [[0.1, 0.03], [0.02, 0.3]], "hessian[0][1]"),
        ("dipole_derivatives", [[0.4, 0.02]], "dipole_derivatives[0][1]"),
        ("orbital_response_residuals", [1e-12, 2e-5], "orbital_response_residuals[1]"),
    ]:
        changed = dict(trial["observables"], **{key: changed_value})
        assert location in benchmark._observable_mismatch(changed, trial["observables"], 1e-8, 1e-8)


def test_benchmark_main_rejects_changed_observable_before_speed_report(monkeypatch, tmp_path, capsys):
    import json
    import sys

    monkeypatch.setattr(benchmark, "CASES", [("audit", "energy", "water", [])])
    calls = []

    def measure(argv):
        calls.append(argv)
        return {"seconds": 0.1, "peak_rss_bytes": 1024, "managed_peak_bytes": 512,
                "exit_code": 0, "observables": {"energy": -1.0 + 0.1 * len(calls)}}

    monkeypatch.setattr(benchmark, "_measure", measure)
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.write_bytes(b"baseline")
    candidate.write_bytes(b"candidate")
    destination = tmp_path / "timings.json"
    monkeypatch.setattr(sys, "argv", ["bench.py", "--exe", str(candidate),
                                    "--baseline-exe", str(baseline), "--trials", "2",
                                    "--save", str(destination)])
    assert benchmark.main() == 1
    result = json.loads(destination.read_text(encoding="utf-8"))["cases"]["audit"]
    assert not result["numerical_acceptance"]["passed"]
    assert [Path(argv[0]).name for argv in calls] == ["candidate", "baseline"] * 2
    assert all(argv[1:] == calls[0][1:] for argv in calls)
    output = capsys.readouterr()
    assert "FAILED" in output.err
    assert "MiB RSS" not in output.out


def test_interrupted_benchmark_preserves_trials_without_accepting_a_partial_case(monkeypatch, tmp_path):
    import json
    import sys

    monkeypatch.setattr(benchmark, "CASES", [("audit", "energy", "water", [])])
    calls = 0

    def measure(argv):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return {"seconds": 0.1, "peak_rss_bytes": 1024, "managed_peak_bytes": 512,
                "exit_code": 0, "observables": {"energy": -1.0}}

    monkeypatch.setattr(benchmark, "_measure", measure)
    destination = tmp_path / "timings.json"
    monkeypatch.setattr(sys, "argv", ["bench.py", "--exe", sys.executable,
                                    "--baseline-exe", sys.executable, "--trials", "2",
                                    "--save", str(destination)])
    with pytest.raises(KeyboardInterrupt):
        benchmark.main()
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["complete"] is False
    case = report["cases"]["audit"]
    assert case["status"] == "measuring" and "numerical_acceptance" not in case
    assert len(case["candidate"]["trials"]) == 1 and case["baseline"]["trials"] == []
    assert not list(tmp_path.glob("*.tmp"))


def test_rebuilt_executable_invalidates_timing_and_preserves_identity(monkeypatch, tmp_path):
    import hashlib
    import json
    import sys

    executable = tmp_path / "candidate"
    executable.write_bytes(b"original payload")
    monkeypatch.setattr(benchmark, "CASES", [("audit", "energy", "water", [])])
    calls = []

    def measure(argv):
        calls.append(argv)
        executable.write_bytes(b"replaced payload")
        return {"seconds": 0.1, "peak_rss_bytes": 1024, "managed_peak_bytes": 512,
                "exit_code": 0, "observables": {"energy": -1.0}}

    monkeypatch.setattr(benchmark, "_measure", measure)
    destination = tmp_path / "timings.json"
    monkeypatch.setattr(sys, "argv", ["bench.py", "--exe", str(executable),
                                    "--trials", "2", "--save", str(destination)])
    assert benchmark.main() == 1
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["executables"]["candidate"] == {
        "name": "candidate", "bytes": len(b"original payload"),
        "sha256": hashlib.sha256(b"original payload").hexdigest()}
    assert len(calls) == 1
    trials = report["cases"]["audit"]["candidate"]["trials"]
    assert "during trial" in trials[0]["error"]
    assert "before trial" in trials[1]["error"]
