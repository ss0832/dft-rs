# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Scalar ECP CPKS force constants share the native, ASE and CLI driver."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
from test_periodic_gks_stress import cli_command


@pytest.mark.parametrize("xc", ["hf", "svwn5", "pbe", "tpss", "pbe0"])
@pytest.mark.parametrize("reference", ["rks", "uks"])
def test_scalar_ecp_phonon_routes(xc, reference, tmp_path):
    numbers = [1, 11, 1]
    positions = np.array([[0.13, -0.21, 0.07], [2.0, 0.45, -0.31], [-0.46, 1.29, 1.62]])
    cell = np.array([[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]])
    options = dict(
        xc=xc, reference=reference, charge=1 if reference == "rks" else 0,
        multiplicity=1 if reference == "rks" else 2,
        basis=str(Path(__file__).resolve().parents[2] / "tests/data/periodic_scalar_ecp.g94"),
        dispersion="none", k_mesh=[1, 1, 1], overlap_cutoff=5.5, xc_ao_cutoff=16.0,
        grid_level=1, grid_shell=1, grid_weight_cutoff=0.0, conv_tol=1e-12,
        conv_tol_dp=1e-11, conv_tol_grad=1e-11, response_tolerance=1e-11, max_iter=500,
    )
    native = dft_rs.phonons(numbers, positions.tolist(), cell.tolist(), **options)
    atoms = Atoms(numbers, positions=positions*Bohr, cell=cell*Bohr, pbc=True)
    calc = dft_rs.DftRs()
    atomic = calc.get_phonons(atoms, units="atomic", **options)
    physical_options = dict(options, overlap_cutoff=options["overlap_cutoff"]*Bohr,
                            xc_ao_cutoff=options["xc_ao_cutoff"]*Bohr,
                            conv_tol=options["conv_tol"]*Hartree,
                            conv_tol_grad=options["conv_tol_grad"]*Hartree)
    physical = calc.get_phonons(atoms, units="ase", **physical_options)
    command = cli_command(os.environ["DFT_RS_CLI"], tmp_path, numbers, positions, cell,
                          options, feature="phonons", pbc=[True]*3)
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "energy" in process.stderr.lower() and "residual" in process.stderr.lower()
    for report in (native, atomic, json.loads(process.stdout)):
        assert report["converged"]
        assert report["reference_type"] == reference
        assert report["raw_asymmetry"] < 1e-8 and report["acoustic_residual"] < 1e-8
        assert report["units"]["force_constants"] == "hartree/bohr^2"
        assert np.asarray(report["raw_force_constants"]).shape == (9, 9)
        for key in ("force_constants", "raw_force_constants", "energy"):
            np.testing.assert_allclose(report[key], native[key], atol=1e-9, rtol=1e-9)
    for key in ("force_constants", "raw_force_constants"):
        np.testing.assert_allclose(physical[key], np.asarray(native[key])*Hartree/Bohr**2,
                                   atol=1e-8, rtol=1e-9)
    np.testing.assert_allclose(physical["energy"], native["energy"]*Hartree, atol=1e-8, rtol=0)
    print(f"Installed scalar ECP CPKS {xc} {reference}", json.dumps({
        key: native[key] for key in ("energy", "raw_force_constants", "frequencies_cm1",
                                    "raw_asymmetry", "acoustic_residual", "response_solves")
    }, allow_nan=False, sort_keys=True))


def test_scalar_ecp_nah_elastic_routes(tmp_path):
    numbers = [11, 1]
    positions = np.array([[0.13, -0.21, 0.07], [2.885, -1.879, 2.282]])
    cell = np.array([[16.0, 0.9, 0.5], [0.6, 17.1, 0.8], [0.4, -0.7, 17.8]])
    options = dict(
        xc="pbe", reference="rks", basis="lanl2dz", k_mesh=[1, 1, 1],
        overlap_cutoff=18.0, xc_ao_cutoff=20.0, grid_level=2, grid_shell=2,
        grid_weight_cutoff=0.0, conv_tol=1e-13, conv_tol_dp=1e-12,
        conv_tol_grad=1e-11, response_tolerance=1e-11, max_iter=500,
    )
    native = dft_rs.elastic(numbers, positions.tolist(), cell.tolist(), **options)
    atoms = Atoms(numbers, positions=positions * Bohr, cell=cell * Bohr, pbc=True)
    calc = dft_rs.DftRs()
    atomic = calc.get_elastic(atoms, units="atomic", **options)
    ase_options = dict(options, overlap_cutoff=options["overlap_cutoff"] * Bohr,
                       xc_ao_cutoff=options["xc_ao_cutoff"] * Bohr,
                       conv_tol=options["conv_tol"] * Hartree,
                       conv_tol_grad=options["conv_tol_grad"] * Hartree)
    physical = calc.get_elastic(atoms, units="ase", **ase_options)
    command = cli_command(os.environ["DFT_RS_CLI"], tmp_path, numbers, positions, cell,
                          options, feature="elastic")
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "energy" in process.stderr.lower() and "residual" in process.stderr.lower()
    for report in (native, atomic, json.loads(process.stdout)):
        assert report["converged"] and report["reference_type"] == "rks"
        assert report["smallest_optical_eigenvalue"] > 0.0
        assert np.asarray(report["strain_hessian"]).shape == (9, 9)
        assert np.asarray(report["internal_strain"]).shape == (9, 6)
        for key in ("energy", "strain_hessian", "internal_strain", "clamped_ion_gpa",
                    "relaxed_ion_gpa", "raw_asymmetry", "internal_strain_asymmetry",
                    "acoustic_residual"):
            np.testing.assert_allclose(report[key], native[key], rtol=1e-9, atol=1e-9)
        for key in ("strain_solves", "displacement_solves"):
            residuals = [item["relative_residual"] for item in report[key]]
            assert residuals and max(residuals) < 1e-10
    for key, scale in (("energy", Hartree), ("strain_hessian", Hartree),
                       ("internal_strain", Hartree / Bohr), ("clamped_ion_gpa", 1.0),
                       ("relaxed_ion_gpa", 1.0)):
        np.testing.assert_allclose(physical[key], np.asarray(native[key]) * scale,
                                   rtol=1e-9, atol=1e-8)
