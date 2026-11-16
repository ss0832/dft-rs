# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""The periodic structure object and the calculator object of the common
driver, against the function interface and the ASE adapter."""

import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree

import dft_rs
from dft_rs import Crystal, DftRs, PeriodicCalculator

EDGE = 6.0
HALF = 0.5 * EDGE
CELL = [[0.0, HALF, HALF], [HALF, 0.0, HALF], [HALF, HALF, 0.0]]
CONTROLS = dict(xc="svwn5", basis="sto-3g", k_mesh=[2, 2, 2], overlap_cutoff=12.0,
                grid_level=2, grid_shell=1, conv_tol=1e-11, guess="core")


def helium():
    return Crystal([2], [[0.0, 0.0, 0.0]], CELL)


def test_crystal_geometry_and_conversions():
    crystal = Crystal([1, 1], [[0.0, 0.0, 0.0], [1.5, 0.3, 0.6]], CELL)
    assert crystal.natoms == 2
    assert abs(crystal.volume - 0.25 * EDGE**3) < 1e-12
    fractional = np.asarray(crystal.fractional_positions())
    np.testing.assert_allclose(fractional @ np.asarray(CELL), crystal.positions, atol=1e-13)
    again = Crystal.from_fractional([1, 1], fractional, CELL)
    np.testing.assert_allclose(again.positions, crystal.positions, atol=1e-13)
    scaled = crystal.scaled(1.02)
    assert abs(scaled.volume - 1.02**3 * crystal.volume) < 1e-10
    np.testing.assert_allclose(scaled.fractional_positions(), fractional, atol=1e-13)
    strained = crystal.strained([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(np.asarray(strained.cell)[:, 0], 1.01 * np.asarray(CELL)[:, 0])
    doubled = crystal.supercell((2, 1, 1))
    assert doubled.numbers == (1, 1, 1, 1)
    np.testing.assert_allclose(doubled.positions[2], np.asarray(crystal.positions[0]) + CELL[0])
    np.testing.assert_allclose(doubled.cell[0], 2.0 * np.asarray(CELL[0]))
    atoms = crystal.to_ase()
    np.testing.assert_allclose(atoms.cell.array, np.asarray(CELL) * Bohr, atol=1e-12)
    np.testing.assert_allclose(Crystal.from_ase(atoms).positions, crystal.positions, atol=1e-12)
    np.testing.assert_allclose(Crystal.from_angstrom([1, 1], atoms.get_positions(), atoms.cell.array).cell,
                               CELL, atol=1e-12)
    with pytest.raises(ValueError):
        Crystal([1], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], CELL)
    with pytest.raises(Exception):
        crystal.numbers = (2, 2)


def test_calculator_object_matches_the_function_interface():
    calc = PeriodicCalculator(helium(), **CONTROLS)
    assert "phonons" in PeriodicCalculator.features()
    assert PeriodicCalculator.features()["periodic_energy"] == "periodic-energy"
    assert "stress" in dir(calc)
    report = calc.periodic_energy(forces=True)
    native = dft_rs.periodic_energy([2], [[0.0, 0.0, 0.0]], CELL, forces=True, **CONTROLS)
    assert report["feature"] == "periodic-energy" and report["converged"]
    assert abs(report["energy"] - native["energy"]) < 1e-12
    np.testing.assert_allclose(report["forces"], native["forces"], atol=1e-12)
    assert abs(calc.run("stress")["pressure"] - dft_rs.stress([2], [[0.0, 0.0, 0.0]], CELL, **CONTROLS)["pressure"]) < 1e-12
    coarse = calc.with_controls(k_mesh=[1, 1, 1])
    assert coarse.controls["k_mesh"] == [1, 1, 1] and calc.controls["k_mesh"] == [2, 2, 2]
    with pytest.raises(AttributeError):
        calc.not_a_calculation
    with pytest.raises(TypeError):
        PeriodicCalculator(([2], [[0.0, 0.0, 0.0]], CELL))


def test_volume_scan_and_relaxation_return_structures():
    calc = PeriodicCalculator(helium(), **CONTROLS)
    scan = calc.scan_volumes([0.98, 1.0, 1.02])
    volumes = [volume for volume, _ in scan]
    np.testing.assert_allclose(volumes, [s**3 * helium().volume for s in (0.98, 1.0, 1.02)])
    assert all(report["converged"] for _, report in scan)
    relaxed, report = calc.relax(cell=True, pressure_gpa=5.0, stress_tol=2e-6, gtol=1e-4, max_steps=30)
    assert report["optimization_converged"]
    assert isinstance(relaxed, Crystal)
    np.testing.assert_allclose(relaxed.cell, report["cell"], atol=1e-14)
    after = calc.with_crystal(relaxed).run("stress")
    assert abs(after["pressure_gpa"] - 5.0) < 0.1


def test_ase_adapter_uses_the_driver_for_energy_forces_stress_and_dispersion():
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=np.asarray(CELL) * Bohr, pbc=True)
    settings = dict(xc="pbe", basis="sto-3g", overlap_cutoff=12.0, grid_level=2, grid_shell=1,
                    conv_tol=1e-11, guess="core")
    plain = DftRs(units="atomic", kpts=(2, 2, 2), **settings)
    with_d3 = DftRs(units="atomic", kpts=(2, 2, 2), dispersion="d3bj", **settings)
    reference = dft_rs.periodic_energy([2], [[0.0, 0.0, 0.0]], CELL, k_mesh=[2, 2, 2],
                                       dispersion="d3bj", stress=True, forces=True, **settings)
    assert reference["dispersion_energy"] < 0.0
    atoms.calc = with_d3
    assert abs(atoms.get_potential_energy() - reference["energy"] * Hartree) < 1e-8
    np.testing.assert_allclose(atoms.get_stress(), np.asarray(reference["stress_voigt"]) * Hartree / Bohr**3,
                               rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(atoms.get_forces(), np.asarray(reference["forces"]) * Hartree / Bohr, atol=1e-10)
    atoms.calc = plain
    assert abs(atoms.get_potential_energy() - (reference["energy"] - reference["dispersion_energy"]) * Hartree) < 1e-7
    # A calculation without the dispersion contribution refuses the model by name.
    with pytest.raises(Exception, match="dispersion"):
        with_d3.get_elastic(atoms)
    # Extra calculator parameters are driver controls.
    three_body = DftRs(units="atomic", kpts=(2, 2, 2), dispersion="d3bj", three_body=True, **settings)
    report = three_body.get_periodic_energy_report(atoms)
    assert report["three_body"] is True


def test_periodic_d2_energy_and_stress_agree_across_public_routes(tmp_path):
    settings = dict(
        xc="pbe", basis="sto-3g", k_mesh=[1, 1, 1], overlap_cutoff=7.1,
        grid_level=1, grid_shell=1, conv_tol=1e-11, guess="core",
        dispersion="d2", disp_cutoff=8.1,
    )
    positions = [[0.0, 0.0, 0.0]]
    native = dft_rs.periodic_energy([2], positions, CELL, forces=True, stress=True, **settings)
    base = dft_rs.periodic_energy([2], positions, CELL, **dict(settings, dispersion="none"))
    atoms = Atoms("He", positions=positions, cell=np.asarray(CELL) * Bohr, pbc=True)
    adapted = DftRs(units="atomic", **settings).get_periodic_energy_report(
        atoms, forces=True, stress=True, units="atomic"
    )
    xyz = tmp_path / "helium.xyz"
    xyz.write_text("1\nhelium crystal\nHe 0 0 0\n", encoding="utf-8")
    cell_arg = ";".join(",".join(f"{x * Bohr:.16f}" for x in row) for row in CELL)
    cli = subprocess.run(
        [os.environ["DFT_RS_CLI"], "periodic-energy", str(xyz), "--cell", cell_arg,
         "--xc", "pbe", "--basis", "sto-3g", "--k-mesh", "1,1,1", "--overlap-cutoff", "7.1",
         "--grid-level", "1", "--grid-shell", "1", "--conv-tol", "1e-11", "--guess", "core",
         "--dispersion", "d2", "--disp-cutoff", "8.1", "--forces", "--stress", "--format", "json"],
        capture_output=True, text=True, check=True,
    )
    assert any(
        "[pbc-scf] iter" in line and "E=" in line and "residual=" in line
        for line in cli.stderr.splitlines()
    )
    for route, report in (("ASE", adapted), ("CLI", json.loads(cli.stdout))):
        assert report["converged"]
        assert report["dispersion_model"] == "D2"
        assert report["units"] == native["units"]
        differences = {}
        for key in ("energy", "dispersion_energy", "forces", "stress", "strain_derivative"):
            np.testing.assert_allclose(report[key], native[key], rtol=0, atol=1e-11)
            differences[key] = float(np.max(np.abs(np.asarray(report[key]) - native[key])))
        print(f"D2 {route} maximum differences from native: {differences}")
    assert native["converged"] and native["dispersion_model"] == "D2"
    assert native["dispersion_energy"] < 0.0
    assert native["energy"] - base["energy"] == pytest.approx(native["dispersion_energy"], abs=1e-13)
    print(f"D2 helium PBE: E={native['energy']:.12e} Eh, dispersion={native['dispersion_energy']:.12e} Eh,")
    print(f"stress={native['stress']} Eh/Bohr^3, orbital residual={native['orbital_residual']:.3e} Eh")


def test_ase_periodic_hessian_relaxation_and_band_path():
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=np.asarray(CELL) * Bohr, pbc=True)
    calc = DftRs(units="atomic", xc="svwn5", basis="sto-3g", kpts=(2, 2, 2), overlap_cutoff=12.0, grid_level=2,
                 grid_shell=1, conv_tol=1e-11, guess="core")
    hessian = calc.get_hessian(atoms)
    phonons = calc.get_phonons(atoms, units="atomic")
    np.testing.assert_allclose(hessian, np.asarray(phonons["force_constants"]) * Hartree / Bohr**2, atol=1e-10)
    relaxed = calc.get_relaxed_atoms(atoms, cell=True, units="atomic", pressure_gpa=5.0, stress_tol=2e-6, gtol=1e-4,
                                     max_steps=30)
    report = relaxed.info["dft_rs_report"]
    assert report["optimization_converged"]
    np.testing.assert_allclose(relaxed.cell.array, np.asarray(report["cell"]) * Bohr, atol=1e-12)
    assert relaxed.calc is None
    path = atoms.cell.bandpath("GX", npoints=4)
    vertices = DftRs._path_vertices(path)
    np.testing.assert_allclose(vertices[0], [0.0, 0.0, 0.0], atol=1e-14)
    assert len(vertices) == 2
    bands = calc.get_bands(atoms, path=path, points=3)
    direct = calc.get_bands(atoms, path=vertices, points=3)
    np.testing.assert_allclose(bands["channels"][0]["band_energies"],
                               direct["channels"][0]["band_energies"], atol=1e-12)
    with pytest.raises(ValueError):
        DftRs._path_vertices(atoms.cell.bandpath("GX,LG", npoints=6))


def test_ase_reports_are_in_ev_and_angstrom():
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=np.asarray(CELL) * Bohr, pbc=True)
    # A cutoff away from every lattice shell (12 Bohr is one): the conversion
    # to Angstrom and back may then round it either way.
    calc = DftRs(xc="svwn5", basis="sto-3g", kpts=(2, 2, 2), overlap_cutoff=12.1 * Bohr, grid_level=2,
                 grid_shell=1, conv_tol=1e-11 * Hartree, guess="core")
    native = dft_rs.periodic_energy([2], [[0.0, 0.0, 0.0]], CELL, forces=True, stress=True,
                                    **dict(CONTROLS, overlap_cutoff=12.1))
    report = calc.get_periodic_energy_report(atoms, forces=True, stress=True)
    assert report["units"]["energy"].startswith("eV")
    assert report["units"]["stress"].startswith("eV/angstrom^3")
    assert report["units"]["forces"].startswith("eV/angstrom")
    assert report["units"]["volume"].startswith("angstrom^3")
    assert "unit_system" in report
    assert abs(report["energy"] - native["energy"] * Hartree) < 1e-8
    assert abs(report["volume"] - native["volume"] * Bohr**3) < 1e-9
    np.testing.assert_allclose(report["stress"], np.asarray(native["stress"]) * Hartree / Bohr**3,
                               rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(report["stress_voigt"], np.asarray(native["stress_voigt"]) * Hartree / Bohr**3,
                               rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(report["forces"], np.asarray(native["forces"]) * Hartree / Bohr, atol=1e-10)
    # Entries labelled in other units are untouched.
    np.testing.assert_allclose(report["stress_gpa"], native["stress_gpa"], rtol=1e-9, atol=1e-12)
    raw = calc.get_periodic_energy_report(atoms, forces=True, stress=True, units="atomic")
    assert "unit_system" not in raw and abs(raw["energy"] - native["energy"]) < 1e-10
    atomic = DftRs(xc="svwn5", basis="sto-3g", kpts=(2, 2, 2), overlap_cutoff=12.1, grid_level=2,
                   grid_shell=1, conv_tol=1e-11, guess="core", units="atomic")
    assert abs(atomic.get_periodic_energy_report(atoms)["energy"] - native["energy"]) < 1e-10
    with pytest.raises(ValueError):
        calc.get_periodic_energy_report(atoms, units="si")
