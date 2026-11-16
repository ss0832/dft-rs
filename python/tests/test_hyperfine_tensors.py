# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Complete hyperfine tensors, raw residuals and all common-driver surfaces."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
import dft_rs

POSITIONS = np.array([[0., 0., 0.], [2.04, 0., 0.],
                      [-1.02, 1.76669, 0.], [-1.02, -1.76669, 0.]])
NUMBERS = [6, 1, 1, 1]
FACTORS = [1.4048236, 5.5856946893, 5.5856946893, 5.5856946893]


def check_tensor(report):
    contact = np.asarray(report["isotropic_coupling_mhz"])
    dipolar = np.asarray(report["dipolar_coupling_mhz"])
    total = np.asarray(report["total_coupling_mhz"])
    principal = np.asarray(report["principal_couplings_mhz"])
    axes = np.asarray(report["principal_axes"])
    np.testing.assert_allclose(total, dipolar + contact[:, None, None] * np.eye(3), atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(np.trace(total, axis1=1, axis2=2) / 3, contact, atol=1e-9, rtol=0)
    np.testing.assert_allclose(np.trace(dipolar, axis1=1, axis2=2), 0., atol=1e-9, rtol=0)
    np.testing.assert_allclose(total, np.swapaxes(total, 1, 2), atol=1e-9, rtol=0)
    assert np.all(np.diff(principal, axis=1) >= 0)
    reconstructed = np.einsum("aik,ak,ajk->aij", axes, principal, axes)
    np.testing.assert_allclose(reconstructed, total, atol=1e-9, rtol=1e-12)
    np.testing.assert_allclose(np.einsum("aki,akj->aij", axes, axes),
                               np.broadcast_to(np.eye(3), (len(NUMBERS), 3, 3)), atol=1e-12, rtol=0)
    assert all(report["tensor_converged"])
    assert np.max(np.abs(report["dipolar_trace_per_g_mhz"])) < 1e-9
    assert np.max(report["tensor_eigen_residual_per_g_mhz"]) < 1e-9
    assert np.max(report["tensor_axes_orthogonality_residual"]) < 1e-12


@pytest.mark.parametrize("xc", ["hf", "pbe0", "tpss"])
def test_complete_hyperfine_tensors_match_native_generic_ase_and_both_cli_names(xc, tmp_path):
    settings = dict(xc=xc, basis="sto-3g", multiplicity=2, unrestricted=True,
                    conv_tol=1e-12, conv_tol_dp=1e-10, max_iter=300, nuclear_g=FACTORS)
    native = dft_rs.hyperfine(NUMBERS, POSITIONS, **settings)
    generic = dft_rs.run_feature("hyperfine", NUMBERS, POSITIONS, **settings)
    atoms = Atoms("CH3", positions=POSITIONS * Bohr)
    adapted = dft_rs.DftRs(units="atomic").get_hyperfine(atoms, **settings)
    xyz = tmp_path / "hyperfine-tensor.xyz"
    xyz.write_text("4\nhyperfine tensor\n" + "\n".join(
        f"{symbol} " + " ".join(f"{v * .529177210903:.17g}" for v in row)
        for symbol, row in zip(["C", "H", "H", "H"], POSITIONS)) + "\n", encoding="utf-8")
    args = [os.environ["DFT_RS_CLI"], "hyperfine", str(xyz), "--format", "json"]
    for key, value in settings.items():
        if isinstance(value, bool):
            if value:
                args.append("--" + key.replace("_", "-"))
        else:
            if isinstance(value, list):
                value = ",".join(map(str, value))
            args.extend(["--" + key.replace("_", "-"), str(value)])
    cli = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
    for report in (native, generic, adapted, cli):
        assert report["converged"]
        assert report["units"] == native["units"] and report["model"] == native["model"]
        assert report["tensor_axis_convention"] == native["tensor_axis_convention"]
        check_tensor(report)
        for key in ("total_coupling_per_g_mhz", "total_coupling_mhz", "principal_couplings_mhz",
                    "dipolar_coupling_mhz", "dipolar_trace_per_g_mhz", "tensor_eigen_residual_per_g_mhz"):
            np.testing.assert_allclose(report[key], native[key], atol=1e-8, rtol=1e-8)
    print(f"complete hyperfine {xc}: trace={native['dipolar_trace_per_g_mhz']}, "
          f"eigen={native['tensor_eigen_residual_per_g_mhz']}, "
          f"axes={native['tensor_axes_orthogonality_residual']}")


def test_negative_nuclear_factors_keep_sorted_principal_values_and_matching_axes():
    factors = [-abs(g) for g in FACTORS]
    report = dft_rs.hyperfine(NUMBERS, POSITIONS, xc="hf", basis="sto-3g", multiplicity=2,
                             unrestricted=True, nuclear_g=factors, conv_tol=1e-12, conv_tol_dp=1e-10)
    check_tensor(report)
    per_g = np.asarray(report["principal_couplings_per_g_mhz"])
    np.testing.assert_allclose(report["principal_couplings_mhz"], per_g[:, ::-1] * np.asarray(factors)[:, None],
                               atol=1e-9, rtol=1e-12)


def test_ecp_sites_keep_all_electron_tensor_outputs_null_on_python_and_ase():
    positions = np.array([[0., 0., 0.], [.2, .1, 3.27]])
    settings = dict(xc="hf", basis="lanl2dz", multiplicity=2, unrestricted=True,
                    nuclear_g=[-.34218, 5.5856946893], conv_tol=1e-12,
                    conv_tol_dp=1e-10, max_iter=300)
    native = dft_rs.hyperfine([12, 1], positions, **settings)
    adapted = dft_rs.DftRs(units="atomic").get_hyperfine(
        Atoms("MgH", positions=positions * Bohr), **settings)
    for report in (native, adapted):
        assert report["converged"] and report["tensor_converged"] == [None, True]
        for key in ("total_coupling_per_g_mhz", "total_coupling_mhz",
                    "principal_couplings_per_g_mhz", "principal_couplings_mhz",
                    "principal_axes_per_g", "principal_axes", "dipolar_trace_per_g_mhz",
                    "tensor_eigen_residual_per_g_mhz"):
            assert report[key][0] is None
            assert report[key][1] is not None
        assert report["pseudo_valence_spin_density_at_nuclei"][0] is not None
        assert report["spin_density_at_nuclei"][0] is None
        np.testing.assert_allclose(report["total_coupling_mhz"][1], native["total_coupling_mhz"][1],
                                   atol=1e-8, rtol=1e-8)
