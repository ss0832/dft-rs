# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""High-order correlated nuclear/electric outputs on the four public routes.

Independent complete-component derivative fits are in p261; equality of
the public routes is additional acceptance, not a derivative error fit.
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


def request(model, unrestricted):
    positions = [[.13, -.21, .19], [1.57, .16, -.08],
                 [-.19, 1.93, .37], [.42, .29, 2.28]]
    if unrestricted:
        positions.append([1.41, 1.83, 1.59])
    controls = dict(xc=model, basis="sto-3g", reference="uks" if unrestricted else "rks",
                    multiplicity=2 if unrestricted else 1, dispersion="none", grid_level=0,
                    grid_weight_cutoff=0., conv_tol=1e-13, conv_tol_dp=1e-12,
                    conv_tol_grad=1e-12, response_tolerance=1e-11,
                    reference_field=[.003, -.002, .004])
    if model == "custom":
        controls.update(xc="hf", double_hybrid_target_xc="pbe",
                        pt2_same_spin=.31, pt2_opposite_spin=.47)
    return [1] * len(positions), np.asarray(positions), controls


def cli(feature, numbers, positions, controls, tmp_path):
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for canonical high-order acceptance"
    path = tmp_path / "biased_cluster.xyz"
    path.write_text(str(len(numbers)) + "\nbiased high-order response\n" + "".join(
        "H " + " ".join(f"{v:.17g}" for v in r) + "\n" for r in positions * Bohr),
        encoding="utf-8")
    args = [executable, feature, str(path), "--format", "json"]
    for key, value in controls.items():
        args.extend(["--" + key.replace("_", "-"),
                     value if isinstance(value, str) else json.dumps(value)])
    process = subprocess.run(args, capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    assert "iter" in process.stderr and "residual" in process.stderr
    return json.loads(process.stdout)


@pytest.mark.parametrize("feature", ["beta", "gamma", "cubic-fc-along", "quartic-fc-along"])
@pytest.mark.parametrize("model", ["custom", "b2plyp"])
@pytest.mark.parametrize("unrestricted", [False, True])
def test_complete_high_order_native_common_ase_and_cli(feature, model, unrestricted, tmp_path):
    numbers, positions, controls = request(model, unrestricted)
    nuclear = feature.endswith("fc-along")
    if nuclear:
        controls["probe"] = [.17 + .09 * np.sin(i * 1.7) for i in range(3 * len(numbers))]
        if feature == "quartic-fc-along":
            controls["second_probe"] = [-.13 + .08 * np.cos(i * .9) for i in range(3 * len(numbers))]
    name = feature.replace("-", "_")
    native = getattr(dft_rs, name)(numbers, positions.tolist(), **controls)
    common = dft_rs.run_feature(feature, numbers, positions.tolist(), **controls)
    atoms = Atoms(numbers=numbers, positions=positions * Bohr)
    ase = getattr(DftRs(units="atomic"), "get_" + name)(atoms, **controls)
    reports = [native, common, ase, cli(feature, numbers, positions, controls, tmp_path)]
    routes = ["native", "common", "ase", "cli"]
    for route, report in zip(routes, reports):
        (tmp_path / f"{route}-report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    suffix = "contracted_force_constants" if nuclear else "tensor"
    prefixes = ["", "parent_", "target_", "pt2_", "pt2_same_spin_alpha_",
                "pt2_opposite_spin_", "pt2_same_spin_beta_"]
    if nuclear:
        prefixes.append("dispersion_")
        shape = [3 * len(numbers)] * 2
    else:
        prefixes += ["parent_nonlocal_", "target_nonlocal_"]
        shape = [3] * (3 if feature == "beta" else 4)
    for route, report in zip(routes, reports):
        assert report["converged"] and report["response_converged"]
        assert report["shape"] == shape
        labels = report.get("unit_labels", report["units"])
        for prefix in prefixes:
            key = prefix + suffix
            assert np.asarray(report[key]).size == np.prod(shape)
            np.testing.assert_allclose(report[key], native[key], rtol=1e-10, atol=1e-8,
                                       err_msg=f"{route}: {feature} {model} U={unrestricted} {key}")
            assert key in labels
        total = np.asarray(report["target_" + suffix]) + report["pt2_" + suffix]
        if nuclear:
            total += report["dispersion_" + suffix]
        np.testing.assert_allclose(report[suffix], total, rtol=1e-10, atol=1e-9)
        weighted = (report["pt2_same_spin_coefficient"]
                    * (np.asarray(report["pt2_same_spin_alpha_" + suffix])
                       + report["pt2_same_spin_beta_" + suffix])
                    + report["pt2_opposite_spin_coefficient"]
                    * np.asarray(report["pt2_opposite_spin_" + suffix]))
        np.testing.assert_allclose(report["pt2_" + suffix], weighted, rtol=1e-10, atol=1e-9)
        for key in ["response_residuals", "amplitude_response_residuals", "amplitude_relative_residuals",
                    "projector_defects",
                    "response_number_defects", "orbital_frame_defects", "reference_defects"]:
            assert max(map(abs, report[key]), default=0.) < 1e-9
        if not nuclear:
            assert report["assembly"] == "energy-polarization"
            assert report["energy_permutation_defect"] is None
            assert report["intrinsic_permutation_defect"] is None
            assert report["energy_permutation_comparisons"] == 0


@pytest.mark.parametrize("feature", ["beta", "gamma"])
def test_biased_water_correlated_tensor(feature):
    positions = [[.13, -.21, .19], [1.56, .07, 1.18], [-1.12, .37, .88]]
    report = dft_rs.run_feature(feature, [8, 1, 1], positions, xc="b2plyp", basis="sto-3g",
                               dispersion="none", grid_level=0, grid_weight_cutoff=0.,
                               conv_tol=1e-13, conv_tol_dp=1e-12, conv_tol_grad=1e-12,
                               response_tolerance=1e-11, reference_field=[.003, -.002, .004])
    assert report["converged"] and report["response_converged"]
    assert np.asarray(report["tensor"]).shape == ((27,) if feature == "beta" else (81,))
    assert np.isfinite(report["tensor"]).all()
    assert np.linalg.norm(report["pt2_tensor"]) > 0


@pytest.mark.parametrize("feature, rank", [("beta", 3), ("gamma", 4)])
def test_generalized_common_pt2_hyperpolarizability_routes(feature, rank, tmp_path):
    numbers = [1] * 4
    positions = np.asarray([[.17, -.11, .23], [1.77, .18, -.53],
                            [-.31, 1.74, -.45], [-1.2, -1.31, -.61]])
    basis = tmp_path / "independent_s.g94"
    basis.write_text("H 0\nS 1 1.0\n0.82 1.0\n****\n", encoding="ascii")
    real = np.zeros((8, 8))
    imaginary = np.zeros((8, 8))
    for i in range(4):
        real[i, i] = -.013 * (i + 1)
        real[4 + i, 4 + i] = -real[i, i]
        real[i, 4 + i] = real[4 + i, i] = .007 * (i + 1)
        imaginary[i, 4 + i] = .009 * (4 - i)
        imaginary[4 + i, i] = -imaginary[i, 4 + i]
    controls = dict(reference="gks", xc="hf", basis=str(basis), charge=1, multiplicity=2,
                    double_hybrid_target_xc="svwn5", pt2_same_spin=.27, pt2_opposite_spin=.27,
                    field=[.003, -.002, .004], guess="core", dispersion="none",
                    grid_level=0, grid_weight_cutoff=0., conv_tol=1e-14, conv_tol_dp=1e-14,
                    conv_tol_grad=1e-14, max_iter=500, response_tolerance=1e-12,
                    spinor_xc_tolerance=1e-12, gks_operator_real=real.tolist(),
                    gks_operator_imaginary=imaginary.tolist())
    native = getattr(dft_rs, feature)(numbers, positions.tolist(), **controls)
    common = dft_rs.run_feature(feature, numbers, positions.tolist(), **controls)
    atoms = Atoms(numbers=numbers, positions=positions * Bohr)
    ase = getattr(DftRs(units="atomic"), "get_" + feature)(atoms, **controls)
    canonical = cli(feature, numbers, positions, controls, tmp_path)
    tensor_keys = ("tensor", "parent_tensor", "target_tensor", "target_tensor_correction",
                   "generalized_pt2_tensor", "pt2_tensor")
    residual_keys = ("response_residuals", "amplitude_relative_residuals",
                     "nuclear_stationarity_defects", "response_number_defects")
    for route, report in zip(("native", "common", "ASE", "CLI"),
                             (native, common, ase, canonical), strict=True):
        assert report["reference_type"] == "gks"
        assert report["converged"] and report["response_converged"], (
            route, report.get("warnings"), report.get("response_reference_fock_defect")
        )
        assert report["shape"] == [3] * rank
        assert report["energy_permutation_defect"] is None
        assert report["energy_permutation_comparisons"] == 0
        labels = report.get("unit_labels", report["units"])
        for key in tensor_keys:
            assert np.asarray(report[key]).size == 3**rank
            assert labels[key] == f"e^{rank}*bohr^{rank}/hartree^{rank-1}"
            np.testing.assert_allclose(report[key], native[key], rtol=0, atol=1e-10)
        for key in residual_keys:
            assert max(map(abs, report[key]), default=0.) <= 1e-12
            np.testing.assert_allclose(report[key], native[key], rtol=0, atol=1e-12)
        assert report["energy"] == pytest.approx(native["energy"], rel=0, abs=2e-12)
    np.testing.assert_allclose(native["tensor"],
                               np.asarray(native["target_tensor"]) + native["pt2_tensor"],
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(native["pt2_tensor"], .27 * np.asarray(native["generalized_pt2_tensor"]),
                               rtol=0, atol=1e-10)
    print(f"Installed GKS common PT2 {feature} energy={native['energy']:.16e} "
          f"tensor[0]={native['tensor'][0]:.16e}")
