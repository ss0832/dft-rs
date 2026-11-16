# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Phase 9 gate: the ASE calculator's analytic forces against ASE-side
finite differences of the potential energy (SVWN5 exercises the XC
gradient path; HF the exact-exchange path)."""

import numpy as np
from ase import Atoms

from dft_rs import DftRs


def water():
    return Atoms(
        "OH2",
        positions=[
            (0.0, 0.0, 0.1173),
            (0.0, 0.7572, -0.4692),
            (0.0, -0.7572, -0.4692),
        ],
    )


def check(xc, tol):
    atoms = water()
    atoms.calc = DftRs(units="atomic", basis="sto-3g", xc=xc, grid_level=4)
    f_an = atoms.get_forces()
    # Central differences of get_potential_energy (eV, Angstrom).
    h = 1e-3
    f_fd = np.zeros_like(f_an)
    for a in range(len(atoms)):
        for ax in range(3):
            plus = atoms.copy()
            plus.positions[a, ax] += h
            plus.calc = DftRs(units="atomic", basis="sto-3g", xc=xc, grid_level=4)
            minus = atoms.copy()
            minus.positions[a, ax] -= h
            minus.calc = DftRs(units="atomic", basis="sto-3g", xc=xc, grid_level=4)
            f_fd[a, ax] = -(plus.get_potential_energy() - minus.get_potential_energy()) / (
                2 * h
            )
    diff = np.abs(f_an - f_fd).max()
    assert diff < tol, f"{xc}: analytic vs FD force mismatch {diff:.3e} eV/A"
    print(f"{xc}: max |analytic - FD| = {diff:.3e} eV/A  OK")


def test_hf_forces():
    check("hf", 5e-4)


def test_svwn_forces():
    check("svwn5", 5e-4)


def test_wavefunction_dump():
    atoms = water()
    atoms.calc = DftRs(units="atomic", basis="sto-3g", xc="hf", wavefunction=True)
    atoms.get_potential_energy()
    wf = atoms.calc.results["wavefunction"]
    nao = wf["nao"]
    c = np.array(wf["mo_coeff"][0])
    s = np.array(wf["overlap"])
    assert c.shape == (nao, nao)
    # MO orthonormality C^T S C = I.
    ortho = c.T @ s @ c
    assert np.abs(ortho - np.eye(nao)).max() < 1e-8
    assert abs(atoms.get_charges().sum()) < 1e-9
    print("wavefunction dump: C^T S C = I OK")


if __name__ == "__main__":
    test_hf_forces()
    test_svwn_forces()
    test_wavefunction_dump()
    print("all ASE gates passed")
