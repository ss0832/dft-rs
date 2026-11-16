# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Projected density of states and Fermi-level properties agree on native
Python and ASE."""
import numpy as np
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr

import dft_rs
from dft_rs import DftRs

CELL = [[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]]
POSITIONS = [[0.0, 0.0, 0.0], [1.75, 0.1, 0.05]]
CONTROLS = dict(xc="svwn5", basis="sto-3g", k_mesh=[6, 1, 1], overlap_cutoff=14.0,
                grid_level=3, grid_shell=3, conv_tol=1e-12, conv_tol_dp=1e-10,
                max_iter=300, electronic_temperature=3000.0)


def chain_atoms():
    return Atoms("H2", positions=np.asarray(POSITIONS) * Bohr, cell=np.asarray(CELL) * Bohr,
                 pbc=True)


def calculator():
    return DftRs(units="atomic", xc="svwn5", basis="sto-3g", k_mesh=(6, 1, 1), overlap_cutoff=14.0,
                 grid_level=3, grid_shell=3, conv_tol=1e-12, conv_tol_dp=1e-10,
                 max_iter=300, electronic_temperature=3000.0)


def test_pauli_susceptibility_agrees_on_native_and_ase():
    native = dft_rs.pauli_spin_susceptibility([1, 1], POSITIONS, CELL, **CONTROLS)
    adapted = calculator().get_pauli_spin_susceptibility(chain_atoms(), units="atomic")
    assert native["feature"] == "pauli-spin-susceptibility" and native["converged"]
    row, other = native["rows"][0], adapted["rows"][0]
    assert row["temperature_k"] == 3000.0
    assert row["dos_at_fermi_level"] > 0.0 and row["heat_capacity"] > 0.0
    assert abs(row["electrons"] - 2.0) < 1e-10
    for key in ("dos_at_fermi_level", "pauli_susceptibility_molar_cgs", "heat_capacity"):
        assert abs(row[key] - other[key]) < 1e-6 * abs(row[key])


def test_projected_dos_sums_to_the_total():
    report = dft_rs.dos([1, 1], POSITIONS, CELL, sigma=0.01, points=400, projected=True,
                        **CONTROLS)
    channel = report["channels"][0]
    total = np.asarray(channel["dos"])
    atoms = np.asarray(channel["projected_atoms"])
    assert atoms.shape == (2, 400)
    assert len(report["projection_groups"]) == 2
    np.testing.assert_allclose(atoms.sum(axis=0), total, atol=1e-8 * total.max())
    assert channel["projection_sum_residual"] < 1e-8 * total.max()
