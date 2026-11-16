# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

import ast
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, orbital_energy_gradients, frontier_gap_gradient

POSITIONS = [[.1, -.2, 0.], [.1, -.2, 1.5]]
OPTIONS = dict(charge=1, conv_tol=1e-13, conv_tol_grad=1e-12,
               conv_tol_dp=1e-12, max_iter=500, grid_level=2)


@pytest.mark.parametrize("xc", ["hf", "r2scan"])
def test_orbital_and_gap_derivatives_share_native_ase_cli(xc, tmp_path):
    options = dict(OPTIONS, xc=xc)
    native = orbital_energy_gradients([2, 1], POSITIONS, coordinates=[5, 2], **options)
    gap = frontier_gap_gradient([2, 1], POSITIONS, coordinates=[5, 2], **options)
    atoms = Atoms("HeH", positions=np.asarray(POSITIONS)*Bohr)
    calc = DftRs(units="atomic", **options)
    adapted = calc.get_orbital_energy_gradients(atoms, coordinates=[5, 2])
    adapted_gap = calc.get_frontier_gap_gradient(atoms, coordinates=[5, 2])
    assert native["converged"] and gap["converged"] and adapted["converged"]
    assert native["coordinates"] == [5, 2] and native["shape"] == [2, 2, 2]
    assert "Hartree/Bohr" in native["units"] and native["units"] == adapted["units"]
    for key in ("gradients", "orbital_energies", "metric_defects", "stationarity_defects"):
        np.testing.assert_allclose(native[key], adapted[key], atol=1e-10, rtol=0)
        np.testing.assert_allclose(native[key], gap[key], atol=1e-12, rtol=0)
    for spin in range(2):
        g = gap["gaps"][spin]
        np.testing.assert_allclose(g["gradient"], np.subtract(native["gradients"][spin][1], native["gradients"][spin][0]), atol=1e-12, rtol=0)
        np.testing.assert_allclose(g["gradient"], adapted_gap["gaps"][spin]["gradient"], atol=1e-10, rtol=0)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for all-surface orbital validation"
    path = tmp_path / "orbital.xyz"
    rows = ["2", "orbital derivative gate"]
    for symbol, r in zip(["He", "H"], POSITIONS):
        rows.append(symbol+" "+" ".join(f"{v*.529177210903:.17g}" for v in r))
    path.write_text("\n".join(rows)+"\n")
    controls = [str(path), "--charge", "1", "--xc", xc, "--grid-level", "2",
                "--orbital-coordinates", "5,2", "--conv-tol", "1e-13",
                "--conv-tol-grad", "1e-12", "--conv-tol-dp", "1e-12", "--log-mode", "none"]
    for command, expected in (("orbital-energy-gradients", native), ("frontier-gap-gradient", gap)):
        result = subprocess.run([executable, command, *controls], capture_output=True, text=True, check=True)
        found = [line.split(maxsplit=3) for line in result.stdout.splitlines() if line.startswith("orbital_gradient ")]
        for row in found:
            np.testing.assert_allclose(ast.literal_eval(row[3]), expected["gradients"][int(row[1])][int(row[2])], atol=1e-10, rtol=0)
        assert len(found) == 4 and "Converged: true" in result.stdout


def test_degenerate_blocks_and_absent_frontier_gaps_are_explicit():
    grouped = frontier_gap_gradient([2, 1], POSITIONS, coordinates=[5],
                                    degeneracy_tolerance=2, xc="hf", **OPTIONS)
    assert all(g is None for spin in grouped["gradients"] for g in spin)
    assert len(grouped["clusters"]) == 2
    assert all(g["gradient"] is None for g in grouped["gaps"])
    cluster = grouped["clusters"][0]
    matrix = np.asarray(cluster["derivative_matrices"])[0]
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12, rtol=0)
    assert cluster["energy_spread"] > 0
    np.testing.assert_allclose(np.linalg.eigvalsh(np.diag(cluster["energies"])), cluster["energies"], atol=1e-12, rtol=0)
    atom = frontier_gap_gradient([1], [[0, 0, 0]], multiplicity=2, xc="hf",
                                coordinates=[0], conv_tol_grad=1e-12)
    assert atom["gaps"] == [None, None]


def test_orbital_input_contracts_reject_ambiguous_requests():
    for coordinates in ([], [6], [5, 5]):
        with pytest.raises(ValueError):
            orbital_energy_gradients([2, 1], POSITIONS, coordinates=coordinates, **OPTIONS)
    for tolerance in (-1, float("nan")):
        with pytest.raises(ValueError):
            frontier_gap_gradient([2, 1], POSITIONS, degeneracy_tolerance=tolerance, **OPTIONS)
    with pytest.raises(TypeError):
        orbital_energy_gradients([2, 1], POSITIONS, order=1, **OPTIONS)
    with pytest.raises(ValueError):
        orbital_energy_gradients([2, 1], POSITIONS, density_fit="def2-universal-jfit", **OPTIONS)
