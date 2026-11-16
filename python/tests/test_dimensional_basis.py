# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, periodic_basis, periodic_capabilities

CELL = np.array([[5.7, .4, .1], [.6, 6.1, .3], [.2, .4, 6.5]])
POSITIONS = np.array([[.2, -.1, .3], [1.6, .5, -.4]])


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_dimensional_basis_native_ase_and_actual_cli_share_all_operators(dimension, tmp_path):
    axes = [i < dimension for i in range(3)]
    mesh = [3, 1, 1]
    arguments = dict(pbc=axes, k_mesh=mesh, basis_cutoff=8.5)
    report = periodic_basis([1, 1], POSITIONS, CELL, **arguments)
    atoms = Atoms("HH", positions=POSITIONS*Bohr, cell=CELL*Bohr, pbc=axes)
    atoms.set_initial_charges([.5, .5])
    adapted = DftRs(units="atomic", basis="sto-3g").get_periodic_basis(atoms, k_mesh=mesh, basis_cutoff=8.5)
    assert report["shape"] == [3, 2, 2] and report["dimension"] == dimension
    assert report["minimum_overlap_eigenvalue"] > .1
    assert report["translation_defect"] < 1e-12 and report["hermiticity_defect"] < 1e-12
    for name in ("overlap", "kinetic"):
        for suffix in ("real", "imag", "derivatives_real", "derivatives_imag"):
            np.testing.assert_allclose(report[f"{name}_{suffix}"], adapted[f"{name}_{suffix}"], atol=1e-12, rtol=0)
        value = np.asarray(report[f"{name}_real"]) + 1j*np.asarray(report[f"{name}_imag"])
        np.testing.assert_allclose(value, value.conj().transpose(0, 2, 1), atol=1e-12, rtol=0)
    assert np.max(np.abs(report["overlap_imag"])) > 1e-4
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for actual CLI acceptance"
    conversion = .529177210903
    path = tmp_path / "basis.xyz"
    path.write_text("2\ndimensional electronic basis\n" + "".join(
        "H " + " ".join(f"{v*conversion:.17g}" for v in row) + "\n" for row in POSITIONS))
    cell = ";".join(",".join(f"{v*conversion:.17g}" for v in row) for row in CELL)
    result = subprocess.run([executable, "periodic-basis", str(path), "--cell", cell,
                             "--pbc", ",".join(str(int(v)) for v in axes), "--k-mesh", "3,1,1",
                             "--overlap-cutoff", "8.5", "--basis", "sto-3g", "--log-mode", "none"],
                            capture_output=True, text=True, check=True)
    for row in (line.split() for line in result.stdout.splitlines()):
        if not row:
            continue
        if row[0] in ("overlap", "kinetic"):
            name = row[0]
            index = tuple(map(int, row[1:4]))
            assert float(row[4]) == pytest.approx(np.asarray(report[f"{name}_real"])[index], abs=1e-11)
            assert float(row[5]) == pytest.approx(np.asarray(report[f"{name}_imag"])[index], abs=1e-11)
        if row[0] in ("overlap_derivative", "kinetic_derivative"):
            name = row[0].removesuffix("_derivative")
            index = tuple(map(int, row[1:6]))
            assert float(row[6]) == pytest.approx(np.asarray(report[f"{name}_derivatives_real"])[index], abs=1e-11)
            assert float(row[7]) == pytest.approx(np.asarray(report[f"{name}_derivatives_imag"])[index], abs=1e-11)
    assert report["units"] in result.stdout and "minimum_overlap_eigenvalue" in result.stdout


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_public_gaussian_matrix_derivatives_require_resolved_four_step_fits(dimension):
    arguments = dict(pbc=[i < dimension for i in range(3)], k_mesh=[3, 1, 1], basis_cutoff=8.5)
    exact = periodic_basis([1, 1], POSITIONS, CELL, **arguments)
    steps = np.array([.04, .02, .01, .005])
    estimates = {name: [] for name in ("overlap", "kinetic")}
    for h in steps:
        plus = POSITIONS.copy()
        minus = POSITIONS.copy()
        plus[1, 0] += h
        minus[1, 0] -= h
        p = periodic_basis([1, 1], plus, CELL, **arguments)
        m = periodic_basis([1, 1], minus, CELL, **arguments)
        for name in estimates:
            pv = np.asarray(p[f"{name}_real"]) + 1j*np.asarray(p[f"{name}_imag"])
            mv = np.asarray(m[f"{name}_real"]) + 1j*np.asarray(m[f"{name}_imag"])
            estimates[name].append((pv-mv)/(2*h))
    for name, values in estimates.items():
        actual = (np.asarray(exact[f"{name}_derivatives_real"]) +
                  1j*np.asarray(exact[f"{name}_derivatives_imag"]))[:, 1, 0]
        errors = np.max(np.abs(np.asarray(values)-actual), axis=(1, 2, 3))
        assert np.all(errors > 0) and np.all(np.isfinite(errors))
        p, intercept = np.polyfit(np.log(steps), np.log(errors), 1)
        r2 = 1 - np.sum((np.log(errors)-intercept-p*np.log(steps))**2)/np.sum((np.log(errors)-np.mean(np.log(errors)))**2)
        print(f"{dimension}D public {name}: errors={errors.tolist()} h=.04..005 p={p:.12f} R2={r2:.12f}")
        assert abs(p-2) < .02 and r2 > .99999
        np.testing.assert_allclose(errors[:-1]/errors[1:], 4, rtol=.03, atol=0)
        np.testing.assert_allclose((4*values[-1]-values[-2])/3, actual, atol=2e-9, rtol=0)


def test_dimensional_grid_preserves_open_measure_moving_weights_and_ase_units(tmp_path):
    cell = CELL.copy()
    cell[1:] = 0
    report = periodic_basis([1, 1], POSITIONS, cell, pbc=[True, False, False],
                            grid_level=0, grid_derivatives=True, basis_cutoff=8.5, grid_cutoff=8.3)
    atoms = Atoms("HH", positions=POSITIONS*Bohr, cell=cell*Bohr, pbc=[True, False, False])
    adapted = DftRs(units="atomic").get_periodic_basis(atoms, grid_level=0, grid_derivatives=True,
                                        basis_cutoff=8.5, grid_cutoff=8.3)
    assert report["grid_overlap_defect"] is not None
    for name in ("points", "weights", "raw_weights", "owners", "moving_weight_derivatives"):
        np.testing.assert_allclose(report["grid"][name], adapted["grid"][name], atol=1e-10, rtol=0)
    gradients = np.asarray(report["grid"]["moving_weight_derivatives"])
    np.testing.assert_allclose(np.sum(gradients, axis=1), 0, atol=1e-10, rtol=0)
    assert "Bohr^3" in report["units"]
    wider = cell.copy()
    wider[1:] = CELL[1:]*23
    other = periodic_basis([1, 1], POSITIONS, wider, pbc=[True, False, False], grid_level=0,
                           grid_derivatives=True, basis_cutoff=8.5, grid_cutoff=8.3)
    for key in ("points", "weights", "raw_weights", "moving_weight_derivatives"):
        assert report["grid"][key] == other["grid"][key]
    # The grid is exposed by the actual CLI, not only a point count.
    conversion = .529177210903
    path = tmp_path / "grid.xyz"
    path.write_text("2\nmoving quadrature\n" + "".join("H " + " ".join(f"{v*conversion:.17g}" for v in row) + "\n" for row in POSITIONS))
    specification = ";".join(",".join(f"{v*conversion:.17g}" for v in row) for row in cell)
    result = subprocess.run([os.environ["DFT_RS_CLI"], "periodic-basis", str(path), "--cell", specification,
                             "--pbc", "1,0,0", "--overlap-cutoff", "8.5", "--basis-grid", "--grid-level", "0",
                             "--basis-grid-cutoff", "8.3", "--basis-grid-derivatives", "--log-mode", "none"],
                            capture_output=True, text=True, check=True)
    for row in (line.split() for line in result.stdout.splitlines()):
        if row and row[0] == "grid":
            index = int(row[1])
            assert int(row[2]) == report["grid"]["owners"][index]
            np.testing.assert_allclose(list(map(float, row[3:6])), report["grid"]["points"][index], atol=1e-10, rtol=0)
            assert float(row[6]) == pytest.approx(report["grid"]["weights"][index], abs=1e-10)
        if row and row[0] == "moving_weight_derivative":
            np.testing.assert_allclose(list(map(float, row[3:])), gradients[int(row[1]), int(row[2])], atol=1e-10, rtol=0)


def test_basis_and_grid_controls_reject_irrelevant_or_invalid_inputs_early(tmp_path):
    capabilities = periodic_capabilities()
    assert capabilities["dimensional_basis_operators"] and capabilities["dimensional_xc_grid"]
    assert not capabilities["partial_boundaries"]
    for kwargs in (dict(basis_cutoff=0), dict(grid_level=10), dict(grid_derivatives=True),
                   dict(pbc=[True, False, False], k_mesh=[1, 2, 1])):
        with pytest.raises(ValueError):
            periodic_basis([1, 1], POSITIONS, CELL, **kwargs)
    with pytest.raises(TypeError):
        periodic_basis([1, 1], POSITIONS, CELL, xc="pbe")
    executable = os.environ["DFT_RS_CLI"]
    for command, flags in (("energy", ["--basis-grid"]), ("periodic-basis", ["--xc", "pbe"]),
                           ("periodic-basis", ["--basis-grid-derivatives"]),
                           ("periodic-basis", ["--basis-grid-cutoff", "8"])):
        result = subprocess.run([executable, command, str(tmp_path / "absent.xyz"),
                                 "--cell", "3,3,3", *flags, "--log-mode", "none"], capture_output=True, text=True)
        assert result.returncode != 0 and "No such file" not in result.stderr
