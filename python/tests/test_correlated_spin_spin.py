# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Full PT2 J tensors on the existing native, ASE and canonical CLI routes."""
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
TENSORS = ["reduced_couplings", "fermi_contact_tensors", "diamagnetic_tensors",
           "paramagnetic_tensors", "spin_dipole_tensors", "coupling_parent_tensors",
           "coupling_target_tensors", "coupling_correlation_tensors",
           "coupling_same_spin_alpha", "coupling_opposite_spin", "coupling_same_spin_beta"]


def request(xc):
    positions = np.array([[.13, -.21, .19], [1.5, -.04, .06],
                          [.44, 3.02, .66], [1.94, 3.24, 1.15]])
    controls = dict(xc=xc, basis="sto-3g", reference="rks", dispersion="none",
                    grid_level=0, grid_weight_cutoff=0., conv_tol=1e-13,
                    conv_tol_dp=1e-12, conv_tol_grad=1e-12,
                    response_tolerance=1e-11, nuclei=[0, 2],
                    nuclear_g=[5.5856946893, 0., 5.5856946893, 0.])
    if xc == "hf":
        controls.update(pt2_same_spin=.31, pt2_opposite_spin=.47)
    return [1] * 4, positions, controls


@pytest.mark.parametrize("xc", ["hf", "b2plyp"])
def test_correlated_j_complete_native_common_and_ase(xc):
    numbers, positions, controls = request(xc)
    native = dft_rs.spin_spin_couplings(numbers, positions.tolist(), **controls)
    common = dft_rs.run_feature("spin-spin-couplings", numbers, positions.tolist(), **controls)
    atoms = Atoms(numbers=numbers, positions=positions * Bohr)
    ase = DftRs(units="atomic").get_spin_spin_couplings(atoms, **controls)
    for report in (native, common, ase):
        assert report["converged"] and report["response_converged"]
        assert report["coupling_response_converged"]
        assert max(report["coupling_response_residuals"]) < 1e-9
        assert max(report["coupling_path_residuals"]) < 1e-9
        assert max(report["coupling_amplitude_residuals"]) < 1e-9
        assert report["units"]["coupling_amplitude_residuals"] == "hartree"
        assert report["units"]["coupling_path_residuals"] == "dimensionless"
        assert max(report["coupling_stationarity_defects"]) < 1e-9
        for key in TENSORS:
            assert np.asarray(report[key]).shape == (1, 3, 3)
            np.testing.assert_allclose(np.asarray(report[key]) / ALPHA4,
                                       np.asarray(native[key]) / ALPHA4, rtol=0, atol=1e-8)
        np.testing.assert_allclose(np.asarray(report["reduced_couplings"]) / ALPHA4,
                                   (np.asarray(report["coupling_target_tensors"])
                                    + report["coupling_correlation_tensors"]) / ALPHA4,
                                   rtol=0, atol=1e-10)
        weighted = (report["pt2_same_spin_coefficient"]
                    * (np.asarray(report["coupling_same_spin_alpha"])
                       + report["coupling_same_spin_beta"])
                    + report["pt2_opposite_spin_coefficient"]
                    * np.asarray(report["coupling_opposite_spin"]))
        np.testing.assert_allclose(np.asarray(report["coupling_correlation_tensors"]) / ALPHA4,
                                   weighted / ALPHA4, rtol=0, atol=1e-10)
    contact = dft_rs.fermi_contact_couplings(numbers, positions.tolist(), **controls)
    np.testing.assert_allclose(np.asarray(contact["reduced_couplings"]) / ALPHA4,
                               np.asarray(native["fermi_contact_tensors"]) / ALPHA4,
                               rtol=0, atol=1e-9)


@pytest.mark.parametrize("feature", ["spin-spin-couplings", "nmr-spectrum"])
def test_correlated_j_canonical_cli_and_nmr_multiplets(feature, tmp_path):
    numbers, positions, controls = request("b2plyp")
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for public-route acceptance"
    keys = TENSORS
    if feature == "nmr-spectrum":
        controls.pop("nuclei")
        controls.update(gauge="giao", nmr_couplings=True, nuclear_spins=[1, 0, 1, 0],
                        nmr_reference_elements=[1], nmr_reference_shieldings=[30.], nmr_points=101)
        keys = ["coupling_parent_tensors", "coupling_target_tensors",
                "coupling_correlation_tensors", "coupling_same_spin_alpha",
                "coupling_opposite_spin", "coupling_same_spin_beta", "coupling_constants_hz"]
    xyz = tmp_path / "skew_hydrogen_pairs.xyz"
    xyz.write_text("4\npost-SCF J coupling\n" + "".join(
        "H " + " ".join(f"{v:.17g}" for v in point) + "\n"
        for point in positions * Bohr), encoding="utf-8")
    args = [executable, feature, str(xyz), "--format", "json"]
    for key, value in controls.items():
        args.extend(["--" + key.replace("_", "-"),
                     value if isinstance(value, str) else json.dumps(value)])
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "iter" in result.stderr and "residual" in result.stderr
    report = json.loads(result.stdout)
    native = dft_rs.run_feature(feature, numbers, positions.tolist(), **controls)
    assert report["coupling_response_converged"]
    for key in keys:
        np.testing.assert_allclose(report[key], native[key], rtol=1e-8, atol=1e-16)
