# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Periodic calculations on the common driver from Python, ASE and the CLI:
energy, analytic stress (ASE ``get_stress``) and variable-cell relaxation."""

import json
import os
import subprocess

import numpy as np
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree

import dft_rs
from dft_rs import DftRs

EDGE = 6.0
HALF = 0.5 * EDGE
CELL = [[0.0, HALF, HALF], [HALF, 0.0, HALF], [HALF, HALF, 0.0]]
CONTROLS = dict(xc="svwn5", basis="sto-3g", k_mesh=[2, 2, 2], overlap_cutoff=12.0,
                grid_level=2, grid_shell=1, conv_tol=1e-11, guess="core")


def helium_atoms():
    return Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=np.asarray(CELL) * Bohr, pbc=True)


def calculator():
    return DftRs(units="atomic", xc="svwn5", basis="sto-3g", k_mesh=(2, 2, 2), overlap_cutoff=12.0,
                 grid_level=2, grid_shell=1, conv_tol=1e-11, guess="core")


def cli(*arguments):
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for route acceptance"
    return subprocess.run([executable, *arguments], capture_output=True, text=True)


def test_stress_agrees_on_native_ase_and_cli(tmp_path):
    native = dft_rs.stress([2], [[0.0, 0.0, 0.0]], CELL, **CONTROLS)
    assert native["feature"] == "stress" and native["converged"]
    stress = np.asarray(native["stress"])
    np.testing.assert_allclose(stress, stress.T, atol=1e-14)
    voigt = np.asarray(native["stress_voigt"])
    np.testing.assert_allclose(voigt, [stress[0, 0], stress[1, 1], stress[2, 2],
                                       stress[1, 2], stress[0, 2], stress[0, 1]])
    assert abs(native["pressure"] + np.trace(stress) / 3.0) < 1e-15
    terms = native["strain_derivative_terms"]
    np.testing.assert_allclose(sum(np.asarray(t) for t in terms.values()),
                               native["strain_derivative"], atol=1e-12)

    atoms = helium_atoms()
    atoms.calc = calculator()
    np.testing.assert_allclose(atoms.get_stress(), voigt * Hartree / Bohr**3, rtol=1e-7, atol=1e-12)
    assert abs(atoms.get_potential_energy() - native["energy"] * Hartree) < 1e-6

    path = tmp_path / "he.xyz"
    path.write_text("1\nhelium\nHe 0 0 0\n")
    rows = ";".join(",".join(f"{v * Bohr:.16f}" for v in row) for row in CELL)
    result = cli("stress", str(path), "--cell", rows, "--xc", "svwn5", "--basis", "sto-3g",
                 "--k-mesh", "2,2,2", "--overlap-cutoff", "12", "--grid-level", "2",
                 "--grid-shell", "1", "--conv-tol", "1e-11", "--guess", "core", "--format", "json")
    assert result.returncode == 0, result.stderr
    np.testing.assert_allclose(json.loads(result.stdout)["stress"], native["stress"], atol=1e-10)


def test_cell_relaxation_reaches_the_external_pressure():
    report = dft_rs.cell_opt([2], [[0.0, 0.0, 0.0]], CELL, pressure_gpa=5.0, stress_tol=2e-6,
                             gtol=1e-4, max_steps=30, **CONTROLS)
    assert report["optimization_converged"]
    enthalpies = [frame["enthalpy"] for frame in report["trajectory"]]
    assert all(b <= a for a, b in zip(enthalpies, enthalpies[1:]))
    stress = np.asarray(report["stress"])
    target = -report["external_pressure"] * np.eye(3)
    assert np.max(np.abs(stress - target)) < 2e-6
    assert abs(report["external_pressure_gpa"] - 5.0) < 1e-12
    adapted = calculator().get_cell_opt(helium_atoms(), pressure_gpa=5.0, stress_tol=2e-6,
                                        gtol=1e-4, max_steps=30, units="atomic")
    np.testing.assert_allclose(adapted["cell"], report["cell"], atol=1e-8)


def test_gamma_phonons_of_a_one_atom_cell_are_acoustic():
    native = dft_rs.phonons([2], [[0.0, 0.0, 0.0]], CELL, **CONTROLS)
    assert native["feature"] == "phonons" and native["converged"]
    fc = np.asarray(native["force_constants"])
    assert fc.shape == (3, 3)
    # One atom per cell: the Gamma force constants are a rigid translation.
    assert np.max(np.abs(fc)) < 1e-8
    assert native["acoustic_residual"] < 1e-8
    assert all(abs(f) < 1.0 for f in native["frequencies_cm1"])
    adapted = calculator().get_phonons(helium_atoms(), units="atomic")
    np.testing.assert_allclose(adapted["force_constants"], native["force_constants"], atol=1e-10)


def test_elastic_constants_of_a_one_atom_cell_agree_on_native_and_ase():
    native = dft_rs.elastic([2], [[0.0, 0.0, 0.0]], CELL, **CONTROLS)
    assert native["feature"] == "elastic" and native["converged"]
    clamped = np.asarray(native["clamped_ion_gpa"])
    relaxed = np.asarray(native["relaxed_ion_gpa"])
    assert clamped.shape == (6, 6)
    # One atom per cell has no internal relaxation.
    np.testing.assert_allclose(relaxed, clamped, atol=1e-10)
    assert native["smallest_optical_eigenvalue"] is None
    # The fcc cell: cubic pattern up to the lab-frame quadrature.
    diagonal = clamped[[0, 1, 2], [0, 1, 2]]
    assert np.ptp(diagonal) < 1e-2 * np.max(np.abs(diagonal))
    assert native["raw_asymmetry"] < 1e-8
    adapted = calculator().get_elastic(helium_atoms(), units="atomic")
    np.testing.assert_allclose(adapted["clamped_ion_gpa"], native["clamped_ion_gpa"], atol=1e-9)


def test_elastic_moduli_and_christoffel_follow_the_returned_tensor():
    moduli = dft_rs.elastic_moduli([2], [[0.0, 0.0, 0.0]], CELL, **CONTROLS)
    assert moduli["feature"] == "elastic-moduli" and moduli["converged"]
    c = np.asarray(moduli["elastic_tensor_gpa"])
    # Birch coefficients are symmetric only under hydrostatic stress; the
    # lab-frame quadrature leaves a stress anisotropy of 0.02 GPa here, so
    # the Voigt average is taken over the full upper-left block.
    k_voigt = c[:3, :3].sum() / 9.0
    assert abs(moduli["voigt_bulk_gpa"] - k_voigt) < 1e-9 * abs(k_voigt)
    assert moduli["reuss_bulk_gpa"] <= moduli["voigt_bulk_gpa"] + 1e-9
    rho = moduli["density_kg_m3"]
    waves = dft_rs.christoffel([2], [[0.0, 0.0, 0.0]], CELL, directions=[[1.0, 0.0, 0.0]], **CONTROLS)
    np.testing.assert_allclose(waves["elastic_tensor_gpa"], moduli["elastic_tensor_gpa"], atol=1e-9)
    longitudinal = max(waves["velocities_m_s"][0])
    # Along a cubic axis the longitudinal wave speed is sqrt(C11 / rho)
    # up to the lab-frame quadrature.
    assert abs(longitudinal - np.sqrt(c[0, 0] * 1e9 / rho)) < 1e-2 * longitudinal
    adapted = calculator().get_elastic_moduli(helium_atoms(), units="atomic")
    assert abs(adapted["hill_bulk_gpa"] - moduli["hill_bulk_gpa"]) < 1e-9 * abs(moduli["hill_bulk_gpa"])


def test_effective_mass_and_transport_agree_on_native_and_ase():
    mass = dft_rs.effective_mass([2], [[0.0, 0.0, 0.0]], CELL, k_point=[0.1, 0.05, 0.0], band=0, **CONTROLS)
    assert mass["feature"] == "effective-mass" and mass["converged"]
    tensor = np.asarray(mass["inverse_mass_tensor"])
    assert tensor.shape == (3, 3)
    np.testing.assert_allclose(tensor, tensor.T, atol=1e-12)
    assert mass["fock_image_residual"] < 1e-10
    adapted = calculator().get_effective_mass(helium_atoms(), k_point=[0.1, 0.05, 0.0], band=0, units="atomic")
    np.testing.assert_allclose(adapted["inverse_mass_tensor"], mass["inverse_mass_tensor"], atol=1e-10)
    transport = dft_rs.boltzmann_transport([2], [[0.0, 0.0, 0.0]], CELL, transport_k_mesh=[4, 4, 4],
                                           transport_temperatures_k=[2000.0], transport_mu_shifts=[0.0], **CONTROLS)
    row = transport["results"][0]
    sigma = np.asarray(row["conductivity_over_tau"])
    np.testing.assert_allclose(sigma, sigma.T, atol=1e-12 * max(1.0, np.max(np.abs(sigma))))
    assert np.all(np.diag(sigma) >= 0.0)
