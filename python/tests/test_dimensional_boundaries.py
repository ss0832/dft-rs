# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, periodic_electrostatics, periodic_capabilities, monkhorst_pack

CELL = np.array([[5., 0., 0.], [1.1, 5.4, 0.], [.3, .4, 6.1]])
POSITIONS = np.array([[.3, -.4, .6]])
OPTIONS = dict(compensation="gaussian", countercharge_center=[-.4, .8, -.1],
               countercharge_width=1.3)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_native_ase_cli_share_charged_dimensional_kernel(dimension, tmp_path):
    axes = [i < dimension for i in range(3)]
    native = periodic_electrostatics(POSITIONS, [1.2], CELL, pbc=axes, **OPTIONS)
    atoms = Atoms("He", positions=POSITIONS*Bohr, cell=CELL*Bohr, pbc=axes)
    adapted = DftRs(units="atomic").get_periodic_electrostatics(atoms, charges=[1.2], **OPTIONS)
    assert native["converged"] and adapted["converged"]
    assert native["dimension"] == dimension and native["shape"] == [1, 3]
    assert native["units"] == adapted["units"]
    for key in ("energy", "forces", "site_potentials", "compensation_force",
                "compensation_potential", "reciprocal_vectors", "k_points"):
        np.testing.assert_allclose(native[key], adapted[key], atol=1e-12, rtol=0)
    assert sum(native["components"].values()) == pytest.approx(native["energy"], abs=1e-13)
    np.testing.assert_allclose(np.sum(native["forces"], axis=0) + native["compensation_force"],
                               0, atol=1e-13, rtol=0)
    assert native["supplied_charge"] + native["compensating_charge"] == 0
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required to verify all surfaces"
    # CLI geometry uses the package CODATA constant, independently of ASE's
    # conversion; both represent the identical Bohr coordinates here.
    angstrom_per_bohr = .529177210903
    path = tmp_path / "charged.xyz"
    path.write_text("1\ncharged boundary gate\nHe " + " ".join(f"{v*angstrom_per_bohr:.17g}" for v in POSITIONS[0]) + "\n")
    specification = ";".join(",".join(f"{v*angstrom_per_bohr:.17g}" for v in row) for row in CELL)
    result = subprocess.run([executable, "periodic-electrostatics", str(path),
                             "--cell", specification, "--pbc", ",".join(str(int(v)) for v in axes),
                             "--electrostatic-charges", "1.2", "--charge-compensation", "gaussian",
                             "--countercharge-center", "-0.4,0.8,-0.1", "--countercharge-width", "1.3",
                             "--log-mode", "none"], check=True, capture_output=True, text=True)
    rows = [line.split() for line in result.stdout.splitlines()]
    energy = next(float(row[1]) for row in rows if row and row[0] == "energy")
    forces = [list(map(float, row[2:])) for row in rows if row and row[0] == "force"]
    assert energy == pytest.approx(native["energy"], abs=1e-11)
    np.testing.assert_allclose(forces, native["forces"], atol=1e-11, rtol=0)
    assert "converged true" in result.stdout


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_four_step_fit_on_all_public_dimensional_force_models(dimension):
    axes = [i < dimension for i in range(3)]
    result = periodic_electrostatics(POSITIONS, [1.2], CELL, pbc=axes, **OPTIONS)
    exact = -result["forces"][0][2]
    steps = np.array([.02, .01, .005, .0025])
    estimates = []
    for step in steps:
        plus, minus = POSITIONS.copy(), POSITIONS.copy()
        plus[0, 2] += step
        minus[0, 2] -= step
        ep = periodic_electrostatics(plus, [1.2], CELL, pbc=axes, **OPTIONS)["energy"]
        em = periodic_electrostatics(minus, [1.2], CELL, pbc=axes, **OPTIONS)["energy"]
        estimates.append((ep-em)/(2*step))
    errors = np.abs(np.asarray(estimates)-exact)
    assert np.all(errors > 0) and np.all(np.isfinite(errors))
    p, intercept = np.polyfit(np.log(steps), np.log(errors), 1)
    prediction = intercept + p*np.log(steps)
    r2 = 1-np.sum((np.log(errors)-prediction)**2)/np.sum((np.log(errors)-np.mean(np.log(errors)))**2)
    print(f"{dimension}D public force errors={errors.tolist()} h=.02..0025 p={p:.12f} R2={r2:.12f}")
    assert abs(p-2) < .02 and r2 > .99999
    np.testing.assert_allclose(errors[:-1]/errors[1:], 4, rtol=.03, atol=0)
    assert (4*estimates[-1]-estimates[-2])/3 == pytest.approx(exact, abs=1e-9)


def test_masks_open_cell_vectors_and_charge_model_validation():
    capabilities = periodic_capabilities()
    assert capabilities["partial_geometry"] and capabilities["dimensional_electrostatics"]
    assert capabilities["charged_lower_dimensional_electrostatics"]
    assert not capabilities["partial_boundaries"]
    for dimension in (1, 2):
        axes = [i < dimension for i in range(3)]
        sparse = CELL.copy()
        sparse[dimension:] = 0
        native = periodic_electrostatics(POSITIONS, [-1.2], CELL, pbc=axes, **OPTIONS)
        changed = periodic_electrostatics(POSITIONS, [-1.2], sparse, pbc=axes, **OPTIONS)
        assert native["energy"] == changed["energy"]
        points = np.asarray(monkhorst_pack(sparse, [3 if a else 1 for a in axes], pbc=axes))
        assert points.shape == (3**dimension, 3)
        for compensation in ("neutral", "uniform"):
            with pytest.raises(ValueError):
                periodic_electrostatics(POSITIONS, [1], sparse, pbc=axes, compensation=compensation)
        with pytest.raises(ValueError):
            periodic_electrostatics(POSITIONS, [1], sparse, pbc=axes,
                                    compensation="gaussian", countercharge_width=1)
        with pytest.raises(ValueError):
            monkhorst_pack(sparse, [2, 2, 2], pbc=axes)
    with pytest.raises(ValueError):
        periodic_electrostatics(POSITIONS, [1, 2], CELL, **OPTIONS)
    with pytest.raises(ValueError):
        periodic_electrostatics(POSITIONS, [1], CELL, compensation="neutral", countercharge_width=1)


def test_cli_mask_and_electrostatic_controls_are_not_silently_ignored(tmp_path):
    executable = os.environ.get("DFT_RS_CLI")
    assert executable
    missing = str(tmp_path / "absent.xyz")
    for command, flags in (("energy", ["--pbc", "1,1,0"]),
                           ("energy", ["--electrostatic-charges", "1"]),
                           ("periodic-electrostatics", ["--xc", "pbe"]),
                           ("periodic-electrostatics", ["--pbc", "1,1"])):
        result = subprocess.run([executable, command, missing, *flags], capture_output=True, text=True)
        assert result.returncode != 0
        assert "No such file" not in result.stderr
