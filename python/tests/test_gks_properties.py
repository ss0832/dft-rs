# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Independent full complex-density checks of GKS/GHF public property paths."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree

import dft_rs
from dft_rs import DftRs
from dft_rs.units import BOHR_IN_ANGSTROM

POSITIONS = np.array([[0.13, -0.21, 0.19], [1.57, 0.16, -0.08],
                      [-0.19, 1.93, 0.37], [0.42, 0.29, 2.28]])
CONTROLS = dict(xc="hf", basis="sto-3g", reference="ghf", grid_level=2,
                grid_weight_cutoff=0.0, conv_tol=1e-12, conv_tol_grad=1e-9,
                conv_tol_dp=1e-11, max_iter=300, guess="core")


@pytest.mark.parametrize("xc", ["blyp", "m06l", "wb97x-d", "wb97x-v"])
def test_generated_spin_orbit_uses_the_covariant_gks_provider(xc, tmp_path):
    numbers = [8, 1, 1]
    positions = np.array([[0.14, -0.19, 0.08], [1.88, 0.17, -0.36],
                          [-0.77, 1.45, -0.51]])
    settings = dict(reference="gks", xc=xc, basis="sto-3g", grid_level=0,
                    vv10_grid_level=0, grid_weight_cutoff=0.0,
                    conv_tol=1e-12, conv_tol_grad=1e-9, conv_tol_dp=1e-10)
    atoms = Atoms(numbers, positions=positions * Bohr)
    native = dft_rs.spin_orbit(numbers, positions, **settings)
    ase = DftRs(units="atomic").get_spin_orbit(atoms, **settings)
    xyz = tmp_path / "water.xyz"
    xyz.write_text("3\nDistorted water with spin-orbit coupling\n" + "".join(
        f"{symbol} {x:.17g} {y:.17g} {z:.17g}\n"
        for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions)
    ), encoding="ascii")
    command = [os.environ["DFT_RS_CLI"], "spin-orbit", str(xyz), "--format", "json"]
    for key, value in settings.items():
        command.extend(("--" + key.replace("_", "-"), str(value)))
    proc = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "iter" in proc.stderr and "residual" in proc.stderr
    for report in (native, ase, json.loads(proc.stdout)):
        assert report["reference_type"] == "gks"
        assert report["self_consistent_xc_model"] == "spin-covariant"
        assert max(report["self_consistent_orbital_residuals"]) < 1e-9
        assert report["units"]["self_consistent_orbital_residuals"] == "hartree"
        np.testing.assert_allclose(report["self_consistent_energies"],
                                   native["self_consistent_energies"], atol=1e-10, rtol=0)
    zero = dft_rs.spin_orbit(numbers, positions, soc_charges=[0, 0, 0], **settings)
    baseline = dft_rs.energy(numbers, positions, **settings)
    # The raw spinor model excludes the geometry-only empirical dispersion term.
    parent = baseline["internal_energy"]
    components = baseline["energy_components"]
    parent -= components.get("dispersion", 0.0)
    assert abs(zero["self_consistent_energies"][0] - parent) < 1e-9
    shift = native["self_consistent_energies"][0] - zero["self_consistent_energies"][0]
    assert shift < -1e-10
    print(f"{xc} GKS SOC energy={native['self_consistent_energies'][0]:.16f} Eh, "
          f"shift={shift:.6e} Eh, residual={native['self_consistent_orbital_residuals'][0]:.6e} Eh")


def spin_source():
    re, im = np.zeros((8, 8)), np.zeros((8, 8))
    for i in range(4):
        re[i, i] = -0.013 * (i + 1)
        re[4 + i, 4 + i] = -re[i, i]
        re[i, 4 + i] = re[4 + i, i] = 0.007 * (i + 1)
        im[i, 4 + i] = 0.009 * (4 - i)
        im[4 + i, i] = -im[i, 4 + i]
    return dict(gks_operator_real=re.tolist(), gks_operator_imaginary=im.tolist())


def validate(report):
    assert report["reference_type"] == "gks" and report["converged"]
    channel = report["channels"][0]
    c = np.asarray(channel["coefficients_real"]) + 1j * np.asarray(channel["coefficients_imaginary"])
    p = np.asarray(channel["density_real"]) + 1j * np.asarray(channel["density_imaginary"])
    f = np.asarray(channel["fock_real"]) + 1j * np.asarray(channel["fock_imaginary"])
    s = np.asarray(report["overlap_real"]) + 1j * np.asarray(report["overlap_imaginary"])
    e, occ = np.asarray(channel["energies"]), np.asarray(channel["occupations"])
    np.testing.assert_allclose(c.conj().T @ s @ c, np.eye(len(e)), rtol=0, atol=1e-10)
    np.testing.assert_allclose((c * occ) @ c.conj().T, p, rtol=0, atol=1e-10)
    np.testing.assert_allclose(f @ c, (s @ c) * e, rtol=0, atol=1e-8)
    values, u = np.linalg.eigh(s)
    x = u / np.sqrt(values)
    dense_e, vectors = np.linalg.eigh(x.conj().T @ f @ x)
    np.testing.assert_allclose(e, dense_e, rtol=0, atol=1e-8)
    dense_c = x @ vectors
    np.testing.assert_allclose(p, (dense_c * occ) @ dense_c.conj().T, rtol=0, atol=1e-8)
    spatial_s = np.asarray(report["spatial_overlap"])
    half_values, half_vectors = np.linalg.eigh(spatial_s)
    half = (half_vectors * np.sqrt(half_values)) @ half_vectors.T
    n = len(spatial_s)
    charge_density = (p[:n, :n] + p[n:, n:]).real
    magnetization = [p[:n, n:] + p[n:, :n],
                     1j * (p[:n, n:] - p[n:, :n]), p[:n, :n] - p[n:, n:]]
    np.testing.assert_allclose(report["mulliken"], 1 - np.diag(charge_density @ spatial_s), rtol=0, atol=1e-10)
    np.testing.assert_allclose(report["lowdin"], 1 - np.diag(half @ charge_density @ half), rtol=0, atol=1e-10)
    spin = np.array([np.diag(m @ spatial_s).real for m in magnetization]).T
    np.testing.assert_allclose(report["mulliken_spin_vector"], spin, rtol=0, atol=1e-10)
    np.testing.assert_allclose(report["spin_expectation"], 0.5 * spin.sum(axis=0), rtol=0, atol=1e-10)
    assert abs(np.trace(p @ s).real - report["electron_count"]) < 1e-10
    assert np.max(np.abs(p[:n, n:].imag)) > 1e-3
    print("GKS raw orbital/metric/number residuals:", report["orbital_residual"],
          report["metric_defect"], report["particle_number_error"])


def test_gks_native_and_ase_share_the_full_complex_reference():
    controls = dict(CONTROLS, **spin_source())
    report = dft_rs.orbitals([1] * 4, POSITIONS.tolist(), coefficients=True, density=True, **controls)
    validate(report)
    charges = dft_rs.charges([1] * 4, POSITIONS.tolist(), **controls)
    energy = dft_rs.energy([1] * 4, POSITIONS.tolist(), **controls)
    assert abs(energy["energy"] - report["energy"]) < 1e-10
    for key in ["mulliken", "lowdin", "mulliken_spin_vector"]:
        np.testing.assert_allclose(charges[key], report[key], rtol=0, atol=1e-10)
    atoms = Atoms("H4", positions=POSITIONS * Bohr)
    calc = DftRs(xc="hf", basis="sto-3g")
    adapted = calc.get_orbitals_report(atoms, units="atomic", coefficients=True, density=True, **controls)
    validate(adapted)
    for key in ["mulliken", "lowdin", "mulliken_spin_vector"]:
        np.testing.assert_allclose(report[key], adapted[key], rtol=0, atol=1e-10)


def test_gks_cli_exposes_the_same_complex_spinor_state(tmp_path):
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for route acceptance"
    xyz = tmp_path / "asymmetric_hydrogen.xyz"
    xyz.write_text("4\ncomplex spinor reference\n" + "".join(
        "H " + " ".join(f"{v:.17g}" for v in row) + "\n" for row in POSITIONS * BOHR_IN_ANGSTROM),
        encoding="utf-8")
    controls = dict(CONTROLS, coefficients=True, density=True, **spin_source())
    args = [executable, "orbitals", str(xyz), "--format", "json"]
    for name, value in controls.items():
        args += ["--" + name.replace("_", "-"), json.dumps(value) if isinstance(value, list) else str(value).lower()]
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    validate(json.loads(result.stdout))
    assert "[spinor-scf]" in result.stderr


def test_complex_ghf_gradient_hessian_on_native_ase_and_cli(tmp_path):
    controls = dict(CONTROLS, conv_tol=1e-13, conv_tol_grad=1e-11,
                    response_tolerance=1e-12, **spin_source())
    gradient = dft_rs.gradient([1] * 4, POSITIONS.tolist(), **controls)
    hessian = dft_rs.hessian([1] * 4, POSITIONS.tolist(), **controls)
    np.testing.assert_allclose(gradient["gradient"], hessian["gradient"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(gradient["forces"], -np.asarray(gradient["gradient"]), rtol=0, atol=0)
    assert max(hessian["reference_fock_defects"]) < 1e-10
    assert max(hessian["response_residuals"]) < 1e-10
    atoms = Atoms("H4", positions=POSITIONS * Bohr)
    calc = DftRs(xc="hf", basis="sto-3g")
    adapted = calc.get_feature_report("hessian", atoms, units="atomic", **controls)
    np.testing.assert_allclose(hessian["hessian"], adapted["hessian"], rtol=0, atol=1e-8)
    atoms.calc = DftRs(units="atomic", **controls)
    np.testing.assert_allclose(atoms.calc.get_hessian(atoms), np.asarray(hessian["hessian"]) * Hartree / Bohr**2,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(atoms.get_forces(), -np.asarray(gradient["gradient"]) * Hartree / Bohr, rtol=0, atol=1e-8)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for route acceptance"
    xyz = tmp_path / "spinor_nuclear.xyz"
    xyz.write_text("4\ncomplex spinor nuclear response\n" + "".join(
        "H " + " ".join(f"{v:.17g}" for v in row) + "\n" for row in POSITIONS * BOHR_IN_ANGSTROM),
        encoding="utf-8")
    args = [executable, "hessian", str(xyz), "--format", "json"]
    for name, value in controls.items():
        args += ["--" + name.replace("_", "-"), json.dumps(value) if isinstance(value, list) else str(value).lower()]
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    np.testing.assert_allclose(hessian["hessian"], json.loads(result.stdout)["hessian"], rtol=0, atol=1e-8)
