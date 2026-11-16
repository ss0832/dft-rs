# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Charged high-order Gamma lattice response through all public routes."""

import json
import os
import subprocess

from ase import Atoms
import numpy as np
import pytest

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM as BOHR
from test_periodic_gks_stress import cli_command


@pytest.mark.parametrize("feature,order,dispersion", [
    ("periodic-cubic-fc", 3, None), ("periodic-quartic-fc", 4, None),
    ("toec", 3, None), ("foec", 4, None),
    ("periodic-cubic-fc", 3, "chg"), ("periodic-quartic-fc", 4, "chg"),
    ("toec", 3, "d3bj"), ("foec", 4, "d3bj"),
])
def test_charged_high_order_periodic_routes(tmp_path, feature, order, dispersion):
    # A single closed-shell He-like site isolates the homogeneous-strain
    # contribution. A real HeH+ trial structure exercises nonzero FC tensors.
    elastic = feature in ("toec", "foec")
    numbers = [3] if elastic else [2, 1]
    positions = [[0.17, 0.41, -0.19]]
    if not elastic:
        positions.append([0.49, -0.12, 1.19])
    cell = [[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]]
    controls = dict(xc="svwn5", charge=1, basis="sto-3g", k_mesh=[1, 1, 1],
                    overlap_cutoff=7.0, grid_level=0, grid_shell=1,
                    grid_weight_cutoff=0.0, conv_tol=1e-12, conv_tol_dp=1e-10,
                    conv_tol_grad=1e-10, response_tolerance=1e-10, cpu_threads=1)
    if dispersion == "chg":
        # The functional selects its CHG term without a dispersion override.
        controls.update(xc="wb97x-d", exchange_cutoff=6.0, disp_cutoff=12.3)
    elif dispersion == "d3bj":
        controls.update(xc="pbe", dispersion="d3bj", disp_cutoff=12.3, cn_cutoff=13.7)
    name = feature.replace("-", "_")
    native = getattr(dft_rs, name)(numbers, positions, cell, **controls)
    atoms = Atoms(numbers, positions=np.array(positions) * BOHR,
                  cell=np.array(cell) * BOHR, pbc=True)
    adapted = getattr(dft_rs.DftRs(), f"get_{name}")(atoms, units="atomic", **controls)
    cli = subprocess.run(cli_command(os.environ["DFT_RS_CLI"], tmp_path, numbers,
                                    positions, cell, controls, feature=feature),
                         capture_output=True, text=True, check=True)
    assert "residual" in cli.stderr.lower()
    keys = ("high_order_clamped", "high_order_relaxed") if elastic else ("force_constant_tensor",)
    if dispersion:
        assert native["dispersion_energy"] < 0.0
        assert native["units"]["dispersion_energy_per_primitive_cell"] == "hartree per primitive cell"
        assert native["units"]["dispersion_energy"].startswith("hartree per supercell")
    for report in (native, adapted, json.loads(cli.stdout)):
        assert report["converged"] and report["provenance"] == "analytic"
        assert report["units"] == native["units"]
        for key in keys:
            assert np.asarray(report[key]).shape == (6,) * order
            assert np.isfinite(report[key]).all()
            np.testing.assert_allclose(report[key], native[key], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(report["energy"], native["energy"], atol=1e-9, rtol=0)
        if dispersion:
            assert report["dispersion_energy"] == report["dispersion_energy_per_primitive_cell"]
            np.testing.assert_allclose(report["dispersion_energy"], native["dispersion_energy"],
                                       atol=1e-12, rtol=0)
        assert max(report["response_residuals"], default=0.0) < 1e-8
    if elastic:
        np.testing.assert_allclose(native[keys[0]], native[keys[1]], atol=1e-10, rtol=0)
        np.testing.assert_array_equal(native["elastic_constants_clamped"], native["elastic_constants_relaxed"])
        np.testing.assert_array_equal(native["internal_relaxation"], np.zeros((3, 6)))
        if order == 4:
            np.testing.assert_array_equal(native["second_internal_relaxation"], np.zeros((36, 3)))
        assert native["smallest_optical_eigenvalue"] is None
        np.testing.assert_allclose(native["volume"], np.linalg.det(cell), atol=1e-10, rtol=0)
    else:
        tensor = np.asarray(native[keys[0]]).reshape((2, 3) + (6,) * (order - 1))
        assert np.max(np.abs(tensor.sum(axis=0))) < 1e-8
    print(f"{feature} E={native['energy']:.17e} tensor_max="
          f"{np.max(np.abs(native[keys[0]])):.17e} response_max="
          f"{max(native['response_residuals'], default=0.0):.17e}")


@pytest.mark.parametrize("feature", ["toec", "foec"])
def test_biased_elastic_routes(tmp_path, feature):
    numbers = [3, 1]
    positions = [[0.17, -0.25, 0.08], [3.30, 0.44, -0.57]]
    cell = [[9.6, 0.7, 0.3], [0.4, 10.4, 0.6], [0.2, -0.5, 11.8]]
    field = [0.001, -0.0015, 0.002]
    controls = dict(xc="wb97x-d", basis="sto-3g", k_mesh=[1, 1, 1], field=field,
                    overlap_cutoff=4.2, exchange_cutoff=4.2, disp_cutoff=4.2,
                    grid_weight_cutoff=0.0, grid_level=0, grid_shell=1, cpu_threads=1,
                    conv_tol=1e-13, conv_tol_dp=1e-11, conv_tol_grad=1e-11)
    native = getattr(dft_rs, feature)(numbers, positions, cell, **controls)
    atoms = Atoms(numbers, positions=np.array(positions) * BOHR,
                  cell=np.array(cell) * BOHR, pbc=True)
    adapted = getattr(dft_rs.DftRs(), f"get_{feature}")(atoms, units="atomic", **controls)
    process = subprocess.run(cli_command(os.environ["DFT_RS_CLI"], tmp_path, numbers,
                                        positions, cell, controls, feature=feature),
                             capture_output=True, text=True, check=True)
    assert "residual" in process.stderr.lower()
    reference = dft_rs.periodic_efield(numbers, positions, cell, **controls)
    assert abs(native["field_energy"] - native["energy"]) > 1e-6
    np.testing.assert_allclose(native["field_energy"], reference["field_energy"], atol=1e-12, rtol=0.)
    for report in (native, adapted, json.loads(process.stdout)):
        assert report["converged"] and report["provenance"] == "analytic"
        assert report["units"] == native["units"]
        assert report["units"]["reference_field"] == "hartree/e/bohr"
        assert "stationary Berry electric enthalpy" in report["units"]["field_energy"]
        np.testing.assert_array_equal(report["reference_field"], field)
        for key in ("field_energy", "high_order_clamped", "high_order_relaxed",
                    "series_reference_residual", "response_residuals"):
            np.testing.assert_allclose(native[key], report[key], atol=1e-8, rtol=1e-9)
    print(f"{feature} biased enthalpy={native['field_energy']:.17e} "
          f"response_max={max(native['response_residuals'], default=0.0):.17e}")
