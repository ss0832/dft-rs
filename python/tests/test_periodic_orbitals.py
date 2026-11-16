# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Bloch coefficients on every public path and an independent dense oracle."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
from dft_rs.units import BOHR_IN_ANGSTROM

import dft_rs
from dft_rs import DftRs

POSITIONS = np.array([[0.13, -0.21, 0.19], [1.57, 0.16, -0.08]])
CELL = np.array([[4.3, 0.27, -0.11], [0.19, 6.7, 0.31], [-0.23, 0.37, 7.1]])
CONTROLS = dict(xc="pbe", basis="sto-3g", k_mesh=[3, 1, 1], overlap_cutoff=12.0,
                exchange_cutoff=12.0, grid_level=2, grid_shell=1, grid_weight_cutoff=0.0,
                conv_tol=1e-12, conv_tol_dp=1e-11, max_iter=300, guess="core")


def complex_matrix(orbitals, name):
    return np.asarray(orbitals[name + "_real"]) + 1j * np.asarray(orbitals[name + "_imaginary"])


def validate(report):
    assert report["converged"]
    population = 0.0
    max_defects = np.zeros(4)
    for channel in report["channels"]:
        orbitals = channel["orbitals"]
        c, s, f, p = [complex_matrix(orbitals, name) for name in
                      ["coefficients", "overlap", "fock", "density"]]
        for k, (ck, sk, fk, pk) in enumerate(zip(c, s, f, p)):
            e = np.asarray(channel["band_energies"][k])
            occ = np.asarray(channel["occupations"][k])
            # Independent NumPy Hermitian diagonalization in the full AO
            # metric, without using the production doubled-real solver.
            values, vectors = np.linalg.eigh(sk)
            assert np.min(values) > 1e-7
            x = vectors / np.sqrt(values)
            expected_e, u = np.linalg.eigh(x.conj().T @ fk @ x)
            expected_c = x @ u
            np.testing.assert_allclose(e, expected_e, rtol=0, atol=1e-10)
            metric = ck.conj().T @ sk @ ck - np.eye(len(e))
            eigen = fk @ ck - (sk @ ck) * e
            density = (ck * occ) @ ck.conj().T - pk
            projectors = [np.outer(ck[:, a], ck[:, a].conj()) -
                          np.outer(expected_c[:, a], expected_c[:, a].conj()) for a in range(len(e))]
            defects = [np.max(np.abs(a)) for a in [metric, eigen, density, projectors]]
            max_defects = np.maximum(max_defects, defects)
            assert max(defects) < 1e-9
            population += channel["spin_weight"] * np.trace(pk @ sk).real / len(c)
        for key in ["metric_defect", "eigen_residual", "band_energy_residual",
                    "density_reconstruction_residual"]:
            assert np.max(orbitals[key]) < 1e-9
        assert np.max(np.abs(c.imag)) > 1e-3
    assert abs(population - report["n_alpha"] - report["n_beta"]) < 1e-10
    print("all-entry metric/eigen/density/projector defects:", max_defects)


def compare_exports(native, other):
    for a, b in zip(native["channels"], other["channels"]):
        ca = complex_matrix(a["orbitals"], "coefficients")
        cb = complex_matrix(b["orbitals"], "coefficients")
        metric = complex_matrix(a["orbitals"], "overlap")
        # Each eigenvector has an arbitrary U(1) phase. Align that phase
        # once per band, then compare every independently computed entry.
        for k in range(len(ca)):
            phase = np.diag(ca[k].conj().T @ metric[k] @ cb[k])
            np.testing.assert_allclose(np.abs(phase), 1, rtol=0, atol=1e-10)
            np.testing.assert_allclose(ca[k], cb[k] * phase.conj() / np.abs(phase), rtol=0, atol=1e-10)
        for key in a["orbitals"]:
            if key not in ["convention", "coefficients_real", "coefficients_imaginary"]:
                np.testing.assert_allclose(a["orbitals"][key], b["orbitals"][key], rtol=0, atol=1e-10)


@pytest.mark.parametrize("open_shell, temperature", [(False, 0.0), (True, 50000.0)])
def test_numpy_oracle_and_native_ase_routes(open_shell, temperature):
    controls = dict(CONTROLS, charge=int(open_shell), multiplicity=2 if open_shell else 1,
                    unrestricted=open_shell, electronic_temperature=temperature, coefficients=True)
    native = dft_rs.electronic_structure([1, 1], POSITIONS.tolist(), CELL.tolist(), **controls)
    validate(native)
    atoms = Atoms("H2", positions=POSITIONS * Bohr, cell=CELL * Bohr, pbc=True)
    calculator = DftRs(**CONTROLS)
    # Pass property controls through the same driver. Energy unit conversion
    # must not alter dimensionless coefficients, overlaps or densities.
    adapted = calculator.get_electronic_structure(atoms, units="atomic", **controls)
    validate(adapted)
    compare_exports(native, adapted)
    ase_controls = dict(controls, overlap_cutoff=controls["overlap_cutoff"] * Bohr,
                        exchange_cutoff=controls["exchange_cutoff"] * Bohr,
                        conv_tol=controls["conv_tol"] * Hartree)
    converted = calculator.get_electronic_structure(atoms, **ase_controls)
    for a, b in zip(native["channels"], converted["channels"]):
        np.testing.assert_allclose(
            complex_matrix(a["orbitals"], "coefficients") @
            np.swapaxes(complex_matrix(a["orbitals"], "coefficients").conj(), -1, -2),
            complex_matrix(b["orbitals"], "coefficients") @
            np.swapaxes(complex_matrix(b["orbitals"], "coefficients").conj(), -1, -2),
            rtol=0, atol=1e-10)
        for key in ["overlap_real", "overlap_imaginary", "density_real", "density_imaginary"]:
            np.testing.assert_allclose(a["orbitals"][key], b["orbitals"][key], rtol=0, atol=1e-10)


def test_cli_bloch_coefficients_match_native_and_report_scf_iterations(tmp_path):
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for route acceptance"
    xyz = tmp_path / "cell.xyz"
    xyz.write_text("2\nasymmetric hydrogen cell\n" + "".join(
        "H " + " ".join(f"{v:.17g}" for v in row) + "\n" for row in POSITIONS * BOHR_IN_ANGSTROM),
        encoding="utf-8")
    cell = ";".join(",".join(f"{v:.17g}" for v in row) for row in CELL * BOHR_IN_ANGSTROM)
    args = [executable, "electronic-structure", str(xyz), "--cell", cell, "--format", "json"]
    for name, value in dict(CONTROLS, coefficients=True).items():
        text = ",".join(map(str, value)) if isinstance(value, list) else str(value).lower()
        args.extend(["--" + name.replace("_", "-"), text])
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "[pbc-scf] iter" in result.stderr and "residual=" in result.stderr
    report = json.loads(result.stdout)
    validate(report)
    native = dft_rs.electronic_structure([1, 1], POSITIONS.tolist(), CELL.tolist(),
                                        coefficients=True, **CONTROLS)
    compare_exports(native, report)
