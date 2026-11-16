# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Ordinary Hall transport on the native, ASE and both CLI routes."""

import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr

import dft_rs
from dft_rs import DftRs

# A triclinic cell resolves every tensor component, rather than comparing
# relative errors of symmetry-forbidden cubic components at roundoff.
CELL = [[0.3, 3.1, 2.8], [2.9, 0.2, 3.2], [3.0, 2.7, 0.1]]
# CLI XYZ/cell conversion follows the package's CODATA 2018 constant. ASE
# uses its own Bohr for both construction and the adapter's inverse mapping.
CLI_BOHR_ANGSTROM = 0.529177210903
BASE = dict(basis="sto-3g", k_mesh=[2, 2, 2], overlap_cutoff=12.0,
            grid_level=2, grid_shell=1, conv_tol=1e-11, guess="core")
TRANSPORT = dict(transport_k_mesh=[6, 6, 6], transport_temperatures_k=[2000.0],
                 carrier_concentrations_cm3=[-1e22])


@pytest.mark.parametrize("xc", ["svwn5", "pbe", "tpss"])
def test_hall_tensors_units_and_residuals_agree_on_every_route(tmp_path, xc):
    reference = dict(BASE, xc=xc)
    native = dft_rs.hall_transport([2], [[0.0, 0.0, 0.0]], CELL, **reference, **TRANSPORT)
    assert native["feature"] == "hall-transport" and native["converged"]
    assert native["fock_image_residual"] < 1e-10
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=np.asarray(CELL) * Bohr, pbc=True)
    adapted = DftRs(units="atomic", **reference).get_hall_transport(atoms, **TRANSPORT, units="atomic")
    generic = dft_rs.run_feature("hall-transport", [2], [[0.0, 0.0, 0.0]], cell=CELL, **reference, **TRANSPORT)
    path = tmp_path / "helium.xyz"
    path.write_text("1\nhelium\nHe 0 0 0\n")
    cell = ";".join(",".join(f"{v * CLI_BOHR_ANGSTROM:.16f}" for v in row) for row in CELL)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for route acceptance"
    arguments = [executable, "hall-transport", str(path), "--cell", cell, "--xc", xc,
                 "--basis", "sto-3g", "--k-mesh", "2,2,2", "--overlap-cutoff", "12",
                 "--grid-level", "2", "--grid-shell", "1", "--conv-tol", "1e-11", "--guess", "core",
                 "--transport-k-mesh", "6,6,6", "--transport-temperatures-k", "2000",
                 "--carrier-concentrations-cm3", "-1e22", "--format", "json"]
    result = subprocess.run(arguments, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cli = json.loads(result.stdout)
    for report in [adapted, generic, cli]:
        assert report["feature"] == native["feature"]
        assert report["units"] == native["units"]
        assert report["hall_axes"] == native["hall_axes"]
        assert report["hall_convention"] == native["hall_convention"]
        assert report["hall_onsager_tolerance"] == native["hall_onsager_tolerance"]
        assert len(report["warnings"]) == len(native["warnings"])
        row = report["results"][0]
        assert row["hall_onsager_satisfied"] == native["results"][0]["hall_onsager_satisfied"]
        for key in ["hall_conductivity_derivative_over_tau_squared", "hall_coefficient", "conductivity_over_tau",
                    "raw_hall_onsager_residual", "relative_hall_onsager_residual"]:
            np.testing.assert_allclose(row[key], native["results"][0][key], rtol=1e-8, atol=1e-20)
    row = native["results"][0]
    assert row["hall_onsager_satisfied"] is False
    assert native["hall_onsager_tolerance"] == 1e-10
    assert any("Onsager defect" in warning for warning in native["warnings"])
    h = np.asarray(row["hall_conductivity_derivative_over_tau_squared"])
    r = np.asarray(row["hall_coefficient"])
    assert h.shape == r.shape == (3, 3, 3)
    assert np.isfinite(h).all() and np.isfinite(r).all()
    # Check inversion in SI, independent of the driver's atomic-unit inverse.
    inverse = np.linalg.inv(np.asarray(row["conductivity_over_tau"]))
    expected = np.stack([-inverse @ h[:, :, c] @ inverse for c in range(3)], axis=-1)
    np.testing.assert_allclose(r, expected, rtol=1e-10, atol=1e-20)
    print(xc, "Hall raw Onsager", row["raw_hall_onsager_residual"],
          "relative", row["relative_hall_onsager_residual"], "Fock residual", native["fock_image_residual"])
    zero = dft_rs.boltzmann_transport([2], [[0.0, 0.0, 0.0]], CELL, **reference, **TRANSPORT)
    np.testing.assert_array_equal(row["conductivity_over_tau"], zero["results"][0]["conductivity_over_tau"])
    assert native["units"]["results.hall_coefficient"].startswith("m^3/C")


def test_hall_controls_fail_before_reference_calculation():
    with pytest.raises(ValueError, match="must not be empty"):
        dft_rs.hall_transport([2], [[0.0, 0.0, 0.0]], CELL, transport_k_mesh=[1, 1, 1],
                             transport_temperatures_k=[300.0], transport_mu_shifts=[])
    with pytest.raises(ValueError, match="exactly one"):
        dft_rs.hall_transport([2], [[0.0, 0.0, 0.0]], CELL, **BASE,
                             **TRANSPORT, transport_mu_shifts=[0.0])
    with pytest.raises(ValueError, match="hall_onsager_tolerance must be positive"):
        dft_rs.hall_transport([2], [[0.0, 0.0, 0.0]], CELL, **BASE,
                             **TRANSPORT, hall_onsager_tolerance=0.0)
