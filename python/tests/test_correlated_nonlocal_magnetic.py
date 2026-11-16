# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Explicit parent/target nonlocal magnetic models on all public routes.

These tests check routing, units and decompositions. Independent complete
derivative fits are in p255, p259 and p260; route equality is not such a fit.
"""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr

import dft_rs
from dft_rs import DftRs

ALPHA4 = (1 / 137.035999084) ** 4


def request(model, unrestricted=False):
    positions = [[.13, -.21, .19], [1.5, -.04, .06],
                 [.44, 3.02, .66], [1.94, 3.24, 1.15]]
    if unrestricted:
        positions.append([1.41, 1.83, -1.59])
    controls = dict(xc=model, basis="sto-3g", reference="uks" if unrestricted else "rks",
                    multiplicity=2 if unrestricted else 1, dispersion="none",
                    grid_level=0, vv10_grid_level=0, grid_weight_cutoff=0.,
                    conv_tol=1e-13, conv_tol_dp=1e-12, conv_tol_grad=1e-12,
                    response_tolerance=1e-11)
    if model == "custom":
        controls.update(xc="hf", dispersion="vv10", double_hybrid_target_xc="wb97x-v",
                        pt2_same_spin=.31, pt2_opposite_spin=.47)
    return [1] * len(positions), np.array(positions), controls


def all_routes(feature, numbers, positions, controls, tmp_path):
    def save(name, report):
        (tmp_path / (name + ".json")).write_text(
            json.dumps(report, default=lambda value: value.tolist()), encoding="utf-8")
    native = getattr(dft_rs, feature.replace("-", "_"))(
        numbers, positions.tolist(), **controls)
    save("native", native)
    common = dft_rs.run_feature(feature, numbers, positions.tolist(), **controls)
    save("common", common)
    atoms = Atoms(numbers=numbers, positions=positions * Bohr)
    ase = getattr(DftRs(units="atomic"), "get_" + feature.replace("-", "_"))(
        atoms, **controls)
    save("ase", ase)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for canonical public acceptance"
    xyz = tmp_path / "nonlocal_magnetic_cluster.xyz"
    xyz.write_text(str(len(numbers)) + "\nnonlocal magnetic response\n" + "".join(
        "H " + " ".join(f"{v:.17g}" for v in r) + "\n" for r in positions * Bohr),
        encoding="utf-8")
    args = [executable, feature, str(xyz), "--format", "json"]
    for key, value in controls.items():
        args.extend(["--" + key.replace("_", "-"),
                     value if isinstance(value, str) else json.dumps(value)])
    process = subprocess.run(args, capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    assert "iter" in process.stderr and "residual" in process.stderr
    cli = json.loads(process.stdout)
    save("cli", cli)
    return [native, common, ase, cli]


@pytest.mark.parametrize("model", ["custom", "wb97m2"])
@pytest.mark.parametrize("unrestricted", [False, True])
def test_nonlocal_nmr_complete_public_routes(model, unrestricted, tmp_path):
    numbers, positions, controls = request(model, unrestricted)
    controls.update(gauge="giao", nmr_reference_elements=[1],
                    nmr_reference_shieldings=[30.], nmr_points=101)
    reports = all_routes("nmr-spectrum", numbers, positions, controls, tmp_path)
    keys = ["shielding_tensors", "parent_shielding_tensors", "target_shielding_tensors",
            "pt2_shielding_tensors", "pt2_same_spin_alpha_shielding_tensors",
            "pt2_opposite_spin_shielding_tensors", "pt2_same_spin_beta_shielding_tensors",
            "parent_nonlocal_shielding_tensors", "target_nonlocal_shielding_tensors"]
    for report in reports:
        assert report["converged"] and report["response_converged"]
        for key in keys:
            assert np.asarray(report[key]).shape == (len(numbers), 3, 3)
            assert np.isfinite(report[key]).all()
            expected_unit = ("ppm, [nucleus][i][j] with sigma_ij = d2E/(dm_i dB_j)"
                             if key == "shielding_tensors" else "ppm")
            assert report["units"][key] == expected_unit
            np.testing.assert_allclose(report[key], reports[0][key], rtol=0, atol=1e-8)
        for key in keys[-2:]:
            assert np.linalg.norm(report[key]) > 0
        np.testing.assert_allclose(report["shielding_tensors"],
                                   np.asarray(report["target_shielding_tensors"])
                                   + report["pt2_shielding_tensors"], rtol=0, atol=1e-9)
        for key in ["amplitude_response_residuals", "orbital_frame_residuals",
                    "projector_residuals", "imaginary_energy_residuals"]:
            assert max(report[key]) < 1e-9
        assert report["units"]["amplitude_response_residuals"] == "hartree"
        assert report["units"]["orbital_frame_residuals"] == "dimensionless"


@pytest.mark.parametrize("model", ["custom", "wb97m2"])
def test_nonlocal_j_complete_public_routes(model, tmp_path):
    numbers, positions, controls = request(model)
    controls.update(nuclei=[0, 2], nuclear_g=[5.5856946893, 0., 5.5856946893, 0.])
    reports = all_routes("spin-spin-couplings", numbers, positions, controls, tmp_path)
    keys = ["reduced_couplings", "fermi_contact_tensors", "diamagnetic_tensors",
            "paramagnetic_tensors", "spin_dipole_tensors", "coupling_parent_tensors",
            "coupling_target_tensors", "coupling_correlation_tensors",
            "coupling_same_spin_alpha", "coupling_opposite_spin", "coupling_same_spin_beta",
            "coupling_parent_nonlocal_tensors", "coupling_target_nonlocal_tensors"]
    for report in reports:
        assert report["converged"] and report["response_converged"]
        assert report["coupling_response_converged"]
        for key in keys:
            assert np.asarray(report[key]).shape == (1, 3, 3)
            np.testing.assert_allclose(np.asarray(report[key]) / ALPHA4,
                                       np.asarray(reports[0][key]) / ALPHA4, rtol=0, atol=1e-8)
        for key in keys[-2:]:
            assert np.linalg.norm(report[key]) > 0
        for key in ["coupling_response_residuals", "coupling_path_residuals",
                    "coupling_amplitude_residuals", "coupling_stationarity_defects"]:
            assert max(report[key]) < 1e-9
        assert report["units"]["coupling_amplitude_residuals"] == "hartree"
        assert report["units"]["coupling_path_residuals"] == "dimensionless"
        np.testing.assert_allclose(np.asarray(report["reduced_couplings"]) / ALPHA4,
                                   (np.asarray(report["coupling_target_tensors"])
                                    + report["coupling_correlation_tensors"]) / ALPHA4,
                                   rtol=0, atol=1e-9)


def test_nonlocal_nmr_unit_convention_and_one_electron_limit(tmp_path):
    """Metadata regression, additional to the full C1 molecular fits."""
    _, _, controls = request("custom", True)
    controls.update(gauge="giao", nmr_reference_elements=[1],
                    nmr_reference_shieldings=[30.], nmr_points=21)
    reports = all_routes("nmr-spectrum", [1], np.array([[.13, -.21, .19]]), controls, tmp_path)
    for report in reports:
        assert report["converged"] and report["response_converged"]
        assert report["units"]["shielding_tensors"] == (
            "ppm, [nucleus][i][j] with sigma_ij = d2E/(dm_i dB_j)")
        for key in ["parent_nonlocal_shielding_tensors", "target_nonlocal_shielding_tensors"]:
            assert report["units"][key] == "ppm"
            assert np.asarray(report[key]).shape == (1, 3, 3)
        for key in ["pt2_shielding_tensors", "pt2_same_spin_alpha_shielding_tensors",
                    "pt2_opposite_spin_shielding_tensors", "pt2_same_spin_beta_shielding_tensors"]:
            np.testing.assert_array_equal(report[key], np.zeros((1, 3, 3)))
