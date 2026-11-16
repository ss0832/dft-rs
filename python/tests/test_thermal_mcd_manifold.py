# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Canonical paramagnetic population C is a named supplied-manifold model."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
import dft_rs
from dft_rs.units import HARTREE_IN_EV


def settings():
    zero = [[0.0, 0.0], [0.0, 0.0]]
    z = [[1.0, 0.0], [0.0, -1.0]]
    return dict(
        mcd_model="thermal-manifold", temperature=4.2,
        bfield=[0.0, 0.0, 1e-5], mcd_thermal_hamiltonian=zero,
        mcd_thermal_magnetic_x=zero, mcd_thermal_magnetic_y=zero,
        mcd_thermal_magnetic_z=z, mcd_thermal_helicity_x=zero,
        mcd_thermal_helicity_y=zero, mcd_thermal_helicity_z=z,
    )


def assert_population_report(report):
    assert report["converged"] and report["geometry_used"] is False
    assert report["mcd_model"] == "thermal-manifold"
    assert "supplied" in report["model"]
    kt = report["thermal_energy"]
    np.testing.assert_allclose(report["mcd_thermal_signal"], [0, 0, np.tanh(1e-5 / kt)], atol=1e-13, rtol=0)
    np.testing.assert_allclose(report["mcd_population_c_tensor"][2][2], 1 / kt, atol=1e-10, rtol=1e-13)
    np.testing.assert_allclose(report["mcd_state_mixing_tensor"], np.zeros((3, 3)), atol=1e-13, rtol=0)
    np.testing.assert_allclose(sum(report["mcd_thermal_populations"]), 1, atol=1e-14, rtol=0)
    assert report["thermal_eigen_residual"] < 1e-12
    for key in ("mcd_thermal_signal", "mcd_population_c_tensor", "mcd_state_mixing_tensor"):
        assert report["units"][key] != "unspecified"


def test_native_ase_and_generic_thermal_population_routes():
    options = settings()
    numbers, positions = [8, 1], [[0, 0, 0], [0, 0, 1.83]]
    named = dft_rs.mcd(numbers, positions, **options)
    common = dft_rs.run_feature("mcd", numbers, positions, **options)
    atoms = Atoms("OH", positions=np.asarray(positions) * 0.529177210903)
    adapted = dft_rs.DftRs(units="atomic").get_mcd(atoms, **options)
    for report in (named, common, adapted):
        assert_population_report(report)
        for key in ("mcd_thermal_signal", "mcd_population_c_tensor", "mcd_thermal_populations"):
            np.testing.assert_allclose(report[key], named[key], atol=1e-12, rtol=0)
    # Explicit per-call SCF controls must not be silently interpreted as
    # thermal multiplet or spin-population parameters.
    with pytest.raises(NotImplementedError, match="support for controls in this context: xc"):
        dft_rs.mcd(numbers, positions, xc="pbe", **options)


def test_thermal_population_cli(tmp_path):
    cli = os.environ.get("DFT_RS_CLI")
    if cli is None:
        pytest.skip("set DFT_RS_CLI for artifact-specific CLI acceptance")
    xyz = tmp_path / "radical.xyz"
    xyz.write_text("2\nSupplied multiplet: geometry is not used\nO 0 0 0\nH 0 0 0.97\n", encoding="utf-8")
    args = [cli, "mcd", str(xyz), "--format", "json"]
    for key, value in settings().items():
        args.extend(["--" + key.replace("_", "-"), json.dumps(value) if not isinstance(value, str) else value])
    done = subprocess.run(args, capture_output=True, text=True, check=True)
    assert_population_report(json.loads(done.stdout))


@pytest.mark.parametrize("bad", [
    {"temperature": 0.0}, {"temperature": -1.0},
    {"mcd_thermal_hamiltonian": [[0.0, 0.2], [0.1, 0.0]]},
    {"mcd_thermal_magnetic_x": [[0.0]]},
    {"mcd_model": "unknown-thermal-model"},
    {"reference": "gks"},
    {"temperature": float("nan")}, {"temperature": float("inf")},
    {"mcd_thermal_populations": [-0.1, 1.1]},
    {"degeneracy_tolerance": -1.0},
    {"mcd_thermal_hamiltonian": []},
])
def test_invalid_thermal_manifolds_are_named_errors(bad):
    with pytest.raises(ValueError):
        dft_rs.run_feature("mcd", [8, 1], [[0, 0, 0], [0, 0, 1.83]], **(settings() | bad))


def test_complex_thermal_matrices_are_rejected_clearly():
    with pytest.raises((TypeError, ValueError), match="complex|real|unsupported|finite|number"):
        dft_rs.run_feature("mcd", [8, 1], [[0, 0, 0], [0, 0, 1.83]], **(settings() | {
            "mcd_thermal_hamiltonian": [[0.0, 1j], [-1j, 0.0]],
        }))


def test_shared_resource_controls_reach_the_thermal_model():
    atoms = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.97]])
    calc = dft_rs.DftRs(units="atomic", cpu_threads=1, memory_mb=64)
    report = calc.get_mcd(atoms, **settings())
    assert_population_report(report)
    assert report["resources"]["cpu_threads"] == 1
    assert report["resources"]["memory_mb"] == 64


def test_finite_field_populations_are_energy_origin_invariant():
    base = dft_rs.run_feature("mcd", **settings())
    for origin in (-1000.0, -100.0, 1000.0):
        options = settings() | {"mcd_thermal_hamiltonian": [[origin, 0.0], [0.0, origin]]}
        shifted = dft_rs.run_feature("mcd", **options)
        for key in ("mcd_thermal_signal", "mcd_thermal_populations", "mcd_thermal_relative_energies"):
            np.testing.assert_array_equal(shifted[key], base[key])
        assert shifted["mcd_energy_origin"] == origin


def test_ase_energy_units_and_normalized_thermal_observables():
    options = settings()
    base = dft_rs.run_feature("mcd", **options)
    atoms = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.97]])
    origin = 0.001
    physical = options | {"mcd_thermal_hamiltonian": [
        [origin * HARTREE_IN_EV, 0.0], [0.0, origin * HARTREE_IN_EV],
    ]}
    report = dft_rs.DftRs(units="ase").get_mcd(atoms, **physical)
    np.testing.assert_allclose(report["mcd_thermal_signal"], base["mcd_thermal_signal"], atol=1e-13, rtol=0)
    np.testing.assert_allclose(report["mcd_thermal_populations"], base["mcd_thermal_populations"], atol=1e-13, rtol=0)
    np.testing.assert_allclose(report["mcd_thermal_relative_energies"], np.asarray(base["mcd_thermal_relative_energies"]) * HARTREE_IN_EV, atol=1e-13, rtol=0)
    assert abs(report["mcd_energy_origin"] - origin * HARTREE_IN_EV) < 1e-13
    assert report["units"]["mcd_energy_origin"].startswith("eV")


def test_ase_thermal_routing_does_not_inject_default_scf(monkeypatch):
    from dft_rs import ase_calculator

    captured = {}

    def backend(feature, numbers, positions, **controls):
        captured.update(controls)
        assert feature == "mcd"
        return {"routing_only": True}

    monkeypatch.setattr(ase_calculator.native, "run_feature", backend)
    calc = ase_calculator.DftRs(units="atomic")
    report = calc.get_feature_report("mcd", Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.97]]), **settings())
    assert report == {"routing_only": True}
    assert "xc" not in captured and "basis" not in captured
    assert "electronic_temperature" not in captured
    assert captured["mcd_model"] == "thermal-manifold"
