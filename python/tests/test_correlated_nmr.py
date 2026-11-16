# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Correlated orbital shieldings and spectra on the existing public APIs."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr

import dft_rs
from dft_rs import DftRs
from dft_rs.units import BOHR_IN_ANGSTROM


def request(unrestricted, xc):
    positions = [[0.13, -0.21, 0.19], [1.57, 0.16, -0.08],
                 [-0.19, 1.93, 0.37], [0.42, 0.29, 2.28]]
    if unrestricted:
        positions.append([1.41, 1.83, 1.59])
    controls = dict(xc=xc, basis="sto-3g", reference="uks" if unrestricted else "rks",
                    multiplicity=2 if unrestricted else 1, dispersion="none",
                    grid_level=0, grid_weight_cutoff=0., conv_tol=1e-13,
                    conv_tol_dp=1e-12, conv_tol_grad=1e-12,
                    response_tolerance=1e-11, gauge="common-origin",
                    nmr_reference_elements=[1], nmr_reference_shieldings=[30.],
                    nmr_points=101)
    if xc == "skala":
        model = os.environ.get("DFT_RS_SKALA_MODEL")
        assert model, "DFT_RS_SKALA_MODEL is required for learned public acceptance"
        controls["skala_model"] = model
    if xc in ("hf", "pbe0", "skala"):
        controls.update(pt2_same_spin=0.31, pt2_opposite_spin=0.47)
    return [1] * len(positions), positions, controls


TENSORS = ["shielding_tensors", "parent_shielding_tensors", "target_shielding_tensors",
           "pt2_shielding_tensors", "pt2_same_spin_alpha_shielding_tensors",
           "pt2_opposite_spin_shielding_tensors", "pt2_same_spin_beta_shielding_tensors"]


@pytest.mark.parametrize("xc", ["hf", "pbe0", "b2plyp", "skala"])
@pytest.mark.parametrize("unrestricted", [False, True])
def test_complete_correlated_nmr_native_ase_and_common_driver(xc, unrestricted):
    numbers, positions, controls = request(unrestricted, xc)
    native = dft_rs.nmr_spectrum(numbers, positions, **controls)
    common = dft_rs.run_feature("nmr-spectrum", numbers, positions, **controls)
    atoms = Atoms(numbers=numbers, positions=np.asarray(positions) * Bohr)
    ase = DftRs().get_nmr_spectrum(atoms, **controls)
    for report in (native, common, ase):
        assert report["converged"] and report["response_converged"]
        for key in TENSORS:
            assert np.asarray(report[key]).shape == (len(numbers), 3, 3)
            assert np.isfinite(report[key]).all()
            np.testing.assert_allclose(report[key], native[key], rtol=0, atol=1e-8)
        np.testing.assert_allclose(report["shielding_tensors"],
                                   np.asarray(report["target_shielding_tensors"])
                                   + report["pt2_shielding_tensors"], rtol=0, atol=1e-9)
        weighted = (report["pt2_same_spin_coefficient"]
                    * (np.asarray(report["pt2_same_spin_alpha_shielding_tensors"])
                       + report["pt2_same_spin_beta_shielding_tensors"])
                    + report["pt2_opposite_spin_coefficient"]
                    * np.asarray(report["pt2_opposite_spin_shielding_tensors"]))
        np.testing.assert_allclose(report["pt2_shielding_tensors"], weighted, rtol=0, atol=1e-9)
        np.testing.assert_allclose(report["chemical_shifts"],
                                   30. - np.asarray(report["isotropic_shieldings"]), rtol=0, atol=1e-10)
        assert max(report["amplitude_response_residuals"]) < 1e-9
        assert max(report["imaginary_energy_residuals"]) < 1e-9
    if xc != "b2plyp":
        zero = dict(controls, pt2_same_spin=0., pt2_opposite_spin=0.)
        limit = dft_rs.nmr_spectrum(numbers, positions, **zero)
        plain = dict(controls)
        plain.pop("pt2_same_spin")
        plain.pop("pt2_opposite_spin")
        parent = dft_rs.nmr_spectrum(numbers, positions, **plain)
        np.testing.assert_allclose(limit["shielding_tensors"], parent["shielding_tensors"],
                                   rtol=0, atol=1e-7)


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("xc", ["b2plyp", "skala"])
def test_correlated_nmr_canonical_cli(unrestricted, xc, tmp_path):
    numbers, positions, controls = request(unrestricted, xc)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for public-route acceptance"
    xyz = tmp_path / "skew_hydrogen_cluster.xyz"
    xyz.write_text(str(len(numbers)) + "\ncorrelated orbital NMR\n" + "".join(
        "H " + " ".join(f"{x:.17g}" for x in point) + "\n"
        for point in np.asarray(positions) * BOHR_IN_ANGSTROM), encoding="utf-8")
    args = [executable, "nmr-spectrum", str(xyz), "--format", "json"]
    for key, value in controls.items():
        args.extend(["--" + key.replace("_", "-"),
                     value if isinstance(value, str) else json.dumps(value)])
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "iter" in result.stderr and "residual" in result.stderr
    report = json.loads(result.stdout)
    native = dft_rs.nmr_spectrum(numbers, positions, **controls)
    for key in TENSORS + ["chemical_shifts", "spectrum_intensity"]:
        np.testing.assert_allclose(report[key], native[key], rtol=0, atol=1e-8)
