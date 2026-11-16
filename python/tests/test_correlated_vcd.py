# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Complete correlated VCD on the shared native, CLI and ASE entry points."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree

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
                    response_tolerance=1e-11, gauge="common-origin", gauge_origin=[0., 0., 0.])
    if xc in ("hf", "pbe0"):
        controls.update(pt2_same_spin=0.31, pt2_opposite_spin=0.47)
    return [1] * len(positions), positions, controls


FIELDS = ["atomic_axial_tensors", "atomic_axial_tensors_electronic",
          "atomic_axial_tensors_nuclear", "parent_atomic_axial_tensors_electronic",
          "pt2_axial_channel_bilinears", "pt2_wavefunction_norm_derivatives",
          "pt2_magnetic_connections", "hessian", "dipole_derivatives",
          "frequencies_cm1", "rotational_strengths"]


def reconstructed_axial(report):
    weighted = (np.asarray(report["parent_atomic_axial_tensors_electronic"])
                + report["pt2_same_spin_coefficient"]
                * (np.asarray(report["pt2_axial_channel_bilinears"])[0]
                   + np.asarray(report["pt2_axial_channel_bilinears"])[2])
                + report["pt2_opposite_spin_coefficient"]
                * np.asarray(report["pt2_axial_channel_bilinears"])[1])
    norm = report["pt2_wavefunction_norm"]
    reconstructed = (weighted / norm - np.asarray(report["pt2_wavefunction_norm_derivatives"])[:, None]
                     * report["pt2_magnetic_connections"] / (2. * norm**2))
    return reconstructed


@pytest.mark.parametrize("xc", ["hf", "pbe0", "b2plyp"])
@pytest.mark.parametrize("unrestricted", [False, True])
def test_correlated_vcd_native_common_driver_and_ase(xc, unrestricted):
    numbers, positions, controls = request(unrestricted, xc)
    native = dft_rs.vcd(numbers, positions, **controls)
    common = dft_rs.run_feature("vcd", numbers, positions, **controls)
    atoms = Atoms(numbers=numbers, positions=np.asarray(positions) * Bohr)
    ase = DftRs(units="atomic").get_vcd(atoms, **controls)
    for report in (native, common, ase):
        assert report["converged"] and report["response_converged"]
        assert "normalized first-order pair-wavefunction" in report["model"]
        assert report["pt2_wavefunction_norm"] > 1.
        for field in FIELDS:
            assert np.isfinite(report[field]).all(), field
            np.testing.assert_allclose(report[field], native[field], rtol=0., atol=1e-8)
        assert np.asarray(report["atomic_axial_tensors"]).shape == (3 * len(numbers), 3)
        assert np.asarray(report["hessian"]).shape == (3 * len(numbers), 3 * len(numbers))
        assert np.asarray(report["dipole_derivatives"]).shape == (3, 3 * len(numbers))
        assert max(report["amplitude_response_residuals"]) < 1e-9
        assert max(report["axial_connection_defects"]) < 1e-9
        assert max(report["axial_cancellation_defects"]) < 1e-9
        np.testing.assert_allclose(report["atomic_axial_tensors"],
                                   np.asarray(report["atomic_axial_tensors_electronic"])
                                   + report["atomic_axial_tensors_nuclear"], rtol=0., atol=1e-12)
        np.testing.assert_allclose(report["atomic_axial_tensors_electronic"], reconstructed_axial(report),
                                   rtol=0., atol=1e-10)
    physical = DftRs(units="ase").get_vcd(atoms, **controls)
    np.testing.assert_allclose(physical["atomic_axial_tensors_electronic"], reconstructed_axial(physical),
                               rtol=0., atol=1e-10)
    np.testing.assert_allclose(physical["hessian"], np.asarray(native["hessian"]) * Hartree / Bohr**2,
                               rtol=0., atol=1e-7)
    np.testing.assert_allclose(physical["atomic_axial_tensors"],
                               np.asarray(native["atomic_axial_tensors"]) * Bohr, rtol=0., atol=1e-8)
    np.testing.assert_allclose(physical["pt2_magnetic_connections"],
                               np.asarray(native["pt2_magnetic_connections"]) * Bohr**2,
                               rtol=0., atol=1e-8)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_correlated_vcd_canonical_cli(unrestricted, tmp_path):
    numbers, positions, controls = request(unrestricted, "b2plyp")
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for public acceptance"
    xyz = tmp_path / "pair_vcd_cluster.xyz"
    xyz.write_text(str(len(numbers)) + "\ncorrelated VCD\n" + "".join(
        "H " + " ".join(f"{x:.17g}" for x in point) + "\n"
        for point in np.asarray(positions) * BOHR_IN_ANGSTROM), encoding="utf-8")
    args = [executable, "vcd", str(xyz), "--format", "json"]
    for key, value in controls.items():
        args.extend(["--" + key.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)])
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "iter" in result.stderr and "residual" in result.stderr
    report = json.loads(result.stdout)
    native = dft_rs.vcd(numbers, positions, **controls)
    for field in FIELDS:
        np.testing.assert_allclose(report[field], native[field], rtol=0., atol=1e-8)
