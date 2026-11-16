# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Shared-driver parity and input guards for weighted molecular XC."""

import json
import os
import subprocess

from ase import Atoms
import numpy as np
import pytest

from dft_rs import (
    DftRs,
    dipole,
    gradient,
    molecular_multipole,
    polarizability_report,
    quadrupole_polarizability,
    run_feature,
)
from dft_rs.units import BOHR_IN_ANGSTROM, HARTREE_IN_EV


def _cli_report(feature, atoms, controls, tmp_path):
    path = tmp_path / "asymmetric.xyz"
    rows = ["H " + " ".join(format(value, ".17g") for value in row) for row in atoms.positions]
    path.write_text("\n".join([str(len(atoms)), "XC report parity", *rows, ""]), encoding="ascii")
    cli = os.environ.get("DFT_RS_CLI")
    assert cli, "DFT_RS_CLI is required for public route acceptance"
    args = [cli, feature, str(path), "--format", "json"]
    for name, value in controls.items():
        args.extend(["--" + name.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)])
    process = subprocess.run(args, check=True, capture_output=True, encoding="utf-8")
    return json.loads(process.stdout), process.stderr


@pytest.mark.parametrize("feature", ["energy", "gradient", "hessian"])
@pytest.mark.parametrize("reference", ["rks", "uks", "roks", "gks"])
def test_weighted_xc_reports_match_on_public_routes(feature, reference, tmp_path):
    if reference in ("roks", "gks"):
        positions = [[0.13, -0.21, 0.19], [1.57, 0.16, -0.08],
                     [-0.19, 1.93, 0.37], [0.42, 0.29, 2.28]]
    else:
        positions = [[0.12, -0.18, 0.24], [1.61, 0.12, -0.25],
                     [-0.34, 2.82, 0.53], [0.83, 3.36, 1.14]]
    terms = [{"x": "slater", "c": "vwn5", "weight": 0.2},
             {"xc": "pbe0", "weight": 0.65}, {"xc": "hf", "weight": 0.15}]
    if feature == "hessian":
        terms = [{"xc": "hf", "weight": 0.65}, {"xc": "svwn5", "weight": 0.35}]
    controls = dict(xc_terms=terms, reference=reference, grid_level=1, grid_weight_cutoff=0,
                    conv_tol=1e-13, conv_tol_dp=1e-11, conv_tol_grad=1e-11, max_iter=300)
    if reference != "rks":
        controls.update(charge=1, multiplicity=2)
    if reference in ("roks", "gks"):
        controls["guess"] = "core"
    native = run_feature(feature, [1] * 4, positions, **controls)
    atoms = Atoms("H4", positions=np.asarray(positions) * BOHR_IN_ANGSTROM)
    ase = DftRs().get_feature_report(feature, atoms, units="atomic", **controls)
    report, stderr = _cli_report(feature, atoms, controls, tmp_path)
    log_prefix = {"roks": "[roks-scf] iter", "gks": "[spinor-scf] iter"}.get(reference, "[scf] iter")
    assert any(log_prefix in line and "residual" in line for line in stderr.splitlines())
    for other in (ase, report):
        assert other["converged"]
        assert other["units"][feature] == native["units"][feature]
        np.testing.assert_allclose(other[feature], native[feature], rtol=0, atol=1e-11)
        assert other["units"]["scf_orbital_residual"] == "hartree"
        assert abs(other["scf_orbital_residual"] - native["scf_orbital_residual"]) < 1e-11
        if feature == "hessian" and "dipole_derivatives" in native:
            assert other["units"]["dipole_derivatives"] == native["units"]["dipole_derivatives"]
            np.testing.assert_allclose(other["dipole_derivatives"], native["dipole_derivatives"], rtol=0, atol=1e-10)
    assert native["converged"]
    assert native["scf_orbital_residual"] < 1e-7


@pytest.mark.parametrize("reference", ["rks", "uks", "roks", "gks"])
@pytest.mark.parametrize("model", ["weighted", "pbe0"])
def test_electric_moments_agree_on_public_routes(reference, model, tmp_path):
    positions = [[0.13, -0.21, 0.19], [1.57, 0.16, -0.08],
                 [-0.19, 1.93, 0.37], [0.42, 0.29, 2.28]]
    atoms = Atoms("H4", positions=np.asarray(positions) * BOHR_IN_ANGSTROM)
    basis = tmp_path / "synthetic.g94"
    basis.write_text("H 0\nS 1 1.0\n0.8 1.0\n****\n", encoding="ascii")
    controls = dict(basis=str(basis), reference=reference, origin=[0.23, -0.5, 0.42],
                    field=[0.002, -0.004, 0.003], grid_level=1, grid_weight_cutoff=0.0,
                    guess="core", conv_tol=1e-13, conv_tol_dp=1e-11, conv_tol_grad=1e-11,
                    max_iter=300, cpu_threads=2)
    if reference != "rks":
        controls.update(charge=1, multiplicity=2)
    if model == "weighted":
        controls["xc_terms"] = [{"xc": "hf", "weight": 0.65}, {"x": "slater", "c": "vwn5", "weight": 0.35}]
    else:
        controls["xc"] = model
    calc = DftRs()
    for feature, native_getter, ase_getter in (
        ("dipole", dipole, calc.get_dipole_report),
        ("molecular-multipole", molecular_multipole, calc.get_molecular_multipole),
    ):
        request = controls if feature == "dipole" else dict(controls, max_rank=2)
        native = native_getter([1] * 4, positions, **request)
        ase = ase_getter(atoms, units="atomic", **request)
        cli, stderr = _cli_report(feature, atoms, request, tmp_path)
        assert "residual" in stderr and "energy" in stderr.lower()
        for other in (ase, cli):
            assert other["reference_type"] == native["reference_type"] == reference
            assert other["converged"] and native["converged"]
            np.testing.assert_allclose(other["energy"], native["energy"], rtol=0, atol=1e-11)
            np.testing.assert_allclose(other["scf_orbital_residual"], native["scf_orbital_residual"],
                                       rtol=0, atol=1e-11)
            assert other["units"]["scf_orbital_residual"] == native["units"]["scf_orbital_residual"] == "hartree"
            assert other["origin"] == native["origin"] == controls["origin"]
            if feature == "dipole":
                assert other["units"]["dipole"] == native["units"]["dipole"] == "e*bohr"
                np.testing.assert_allclose(other["dipole"], native["dipole"], rtol=0, atol=1e-11)
            else:
                for convention in ("cartesian", "traceless", "spherical"):
                    key = f"ranks.{convention}"
                    assert other["units"][key] == native["units"][key] == "e*bohr^rank"
                    for actual, expected in zip(other["ranks"], native["ranks"], strict=True):
                        np.testing.assert_allclose(actual[convention], expected[convention], rtol=0, atol=1e-11)


@pytest.mark.parametrize("terms", [
    [],
    [{"xc": "pbe", "weight": 0}],
    [{"xc": "pbe", "weight": 1, "unexpected": True}],
    [{"x": "pbe", "weight": 1}],
])
def test_invalid_weighted_xc_is_refused(terms):
    with pytest.raises(ValueError):
        run_feature("energy", [2], [[0, 0, 0]], xc_terms=terms)


def test_unconnected_weighted_xc_is_refused():
    with pytest.raises(NotImplementedError, match="weighted XC"):
        run_feature("force-constants", [2], [[0, 0, 0]], xc_terms=[{"xc": "pbe", "weight": 1}])


@pytest.mark.parametrize(("reference", "xc"), [
    ("rks", None), ("rks", "svwn5"), ("uks", "pbe"), ("roks", "tpss"), ("gks", "pbe0"),
], ids=["weighted", "rks-lda", "uks-gga", "roks-meta-gga", "gks-global-hybrid"])
def test_electric_quadrupole_polarizability_public_units_and_frequency(reference, xc, tmp_path):
    positions = [[0.13, -0.21, 0.19], [1.57, 0.16, -0.08],
                 [-0.19, 1.93, 0.37], [0.42, 0.29, 2.28]]
    if reference != "rks":
        positions.append([-1.25, 0.39, 1.03])
    n = len(positions)
    atoms = Atoms("H" * n, positions=np.asarray(positions) * BOHR_IN_ANGSTROM)
    basis = tmp_path / "synthetic.g94"
    basis.write_text("H 0\nS 1 1.0\n0.8 1.0\n****\n", encoding="ascii")
    controls = dict(
        basis=str(basis), reference=reference, grid_level=1 if xc is None else 0,
        grid_weight_cutoff=0, multiplicity=1 if reference == "rks" else 2,
        frequency=0.02, frequency_damping=0.005, origin=[0.23, -0.5, 0.42],
    )
    if xc is None:
        controls["xc_terms"] = [{"xc": "hf", "weight": 0.65}, {"xc": "svwn5", "weight": 0.35}]
    else:
        controls["xc"] = xc
    if reference == "gks":
        real = np.zeros((2 * n, 2 * n))
        imaginary = np.zeros_like(real)
        for i in range(n):
            real[i, i], real[n + i, n + i] = -0.013 * (i + 1), 0.013 * (i + 1)
            real[i, n + i] = real[n + i, i] = 0.007 * (i + 1)
            imaginary[i, n + i], imaginary[n + i, i] = 0.009 * (n - i), -0.009 * (n - i)
        controls.update(gks_operator_real=real.tolist(), gks_operator_imaginary=imaginary.tolist())
    native = quadrupole_polarizability([1] * n, positions, **controls)
    ase = DftRs().get_quadrupole_polarizability(atoms, units="atomic", **controls)
    cli, stderr = _cli_report("quadrupole-polarizability", atoms, controls, tmp_path)
    assert "residual" in stderr
    keys = ["quadrupole_polarizability_real", "quadrupole_polarizability_imag"]
    for report in (ase, cli):
        assert report["converged"] and native["converged"]
        for key in keys:
            assert report["units"][key] == native["units"][key] == "e^2*bohr^4/hartree"
            np.testing.assert_allclose(report[key], native[key], rtol=0, atol=1e-9)
        assert report["shape"] == native["shape"] == [5, 5]
        assert report["frequency_real"] == native["frequency_real"] == 0.02
        assert report["frequency_imag"] == native["frequency_imag"] == 0.005
        np.testing.assert_allclose(report["response_residuals"], native["response_residuals"],
                                   rtol=0, atol=1e-9)
    physical = dict(controls, frequency=0.02 * HARTREE_IN_EV, frequency_damping=0.005 * HARTREE_IN_EV,
                    origin=(np.asarray(controls["origin"]) * BOHR_IN_ANGSTROM).tolist())
    _convert_spinor_operator_to_ev(physical)
    ase = DftRs().get_quadrupole_polarizability(atoms, **physical)
    for key in keys:
        expected = np.asarray(native[key]) * BOHR_IN_ANGSTROM**4 / HARTREE_IN_EV
        np.testing.assert_allclose(ase[key], expected, rtol=0, atol=1e-9)
    if xc is None:
        shift = np.array([0.34, -0.27, 0.16])
        displaced_origin = dict(controls, origin=(np.asarray(controls["origin"]) + shift).tolist())
        translated = quadrupole_polarizability([1] * n, np.asarray(positions) + shift, **displaced_origin)
        origin_only = quadrupole_polarizability([1] * n, positions, **displaced_origin)
        assert translated["converged"] and origin_only["converged"]
        for key in keys:
            np.testing.assert_allclose(translated[key], native[key], rtol=0, atol=1e-9)
            assert np.max(np.abs(np.asarray(origin_only[key]) - native[key])) > 1e-3
    print(f"Installed {reference} {xc or 'weighted'} retarded electric quadrupole polarizability",
          json.dumps(native, sort_keys=True, allow_nan=False))


def _convert_spinor_operator_to_ev(controls):
    for key in ("gks_operator_real", "gks_operator_imaginary"):
        if key in controls:
            controls[key] = (np.asarray(controls[key]) * HARTREE_IN_EV).tolist()


@pytest.mark.parametrize("reference", ["rks", "uks"])
def test_double_hybrid_quadrupole_reports_spin_channels_on_public_routes(reference, tmp_path):
    positions = [[0.13, -0.21, 0.19], [1.57, 0.16, -0.08], [-0.19, 1.93, 0.37],
                 [0.42, 0.29, 2.28], [-1.25, 0.39, 1.03], [1.36, 2.45, 1.38]]
    if reference == "uks":
        positions.pop()
    atoms = Atoms(numbers=[1] * len(positions), positions=np.asarray(positions) * BOHR_IN_ANGSTROM)
    basis = tmp_path / "synthetic.g94"
    basis.write_text("H 0\nS 1 1.0\n0.8 1.0\n****\n", encoding="ascii")
    controls = dict(basis=str(basis), xc="b2plyp", reference=reference, grid_level=1,
                    grid_weight_cutoff=0, max_rank=2, origin=[0.23, -0.5, 0.42],
                    multiplicity=2 if reference == "uks" else 1,
                    conv_tol=1e-13, conv_tol_grad=1e-11, conv_tol_dp=1e-12,
                    guess="core", max_iter=300)
    native = molecular_multipole([1] * len(positions), positions, **controls)
    ase = DftRs().get_molecular_multipole(atoms, units="atomic", **controls)
    cli, stderr = _cli_report("molecular-multipole", atoms, controls, tmp_path)
    assert "residual" in stderr
    keys = ["cartesian", "parent_cartesian", "pt2_cartesian_correction", "pt2_same_spin_alpha",
            "pt2_same_spin_beta", "pt2_opposite_spin", "target_cartesian_correction"]
    for report in (ase, cli):
        assert report["converged"] and native["converged"]
        assert abs(report["energy"] - native["energy"]) < 1e-10
        for key in keys:
            unit = f"ranks.{key}"
            assert report["units"][unit] == native["units"][unit] == "e*bohr^rank"
            np.testing.assert_allclose(report["ranks"][2][key], native["ranks"][2][key], rtol=0, atol=1e-10)


    controls.pop("max_rank")
    native = quadrupole_polarizability([1] * len(positions), positions, **controls)
    ase = DftRs().get_quadrupole_polarizability(atoms, units="atomic", **controls)
    cli, stderr = _cli_report("quadrupole-polarizability", atoms, controls, tmp_path)
    assert "residual" in stderr
    keys = ["quadrupole_polarizability", "parent_quadrupole_polarizability",
            "pt2_quadrupole_polarizability_correction", "target_quadrupole_polarizability_correction",
            "pt2_same_spin_alpha_quadrupole_polarizability",
            "pt2_same_spin_beta_quadrupole_polarizability",
            "pt2_opposite_spin_quadrupole_polarizability"]
    for report in (ase, cli):
        assert report["converged"] and native["converged"]
        assert report["shape"] == native["shape"] == [5, 5]
        assert report["independent_components"] == native["independent_components"] == 15
        assert abs(report["energy"] - native["energy"]) < 1e-10
        for key in keys:
            assert report["units"][key] == native["units"][key] == "e^2*bohr^4/hartree"
            np.testing.assert_allclose(report[key], native[key], rtol=0, atol=1e-9)
        for key in ("response_residuals", "amplitude_response_residuals"):
            np.testing.assert_allclose(report[key], native[key], rtol=0, atol=1e-9)

    physical = dict(controls, origin=(np.asarray(controls["origin"]) * BOHR_IN_ANGSTROM).tolist())
    ase = DftRs().get_quadrupole_polarizability(atoms, **physical)
    for key in keys:
        expected = np.asarray(native[key]) * BOHR_IN_ANGSTROM**4 / HARTREE_IN_EV
        np.testing.assert_allclose(ase[key], expected, rtol=0, atol=1e-9)

def _generalized_electric_fixture(xc, target, tmp_path):
    positions = [[0.13, -0.21, 0.19], [1.57, 0.16, -0.08], [-0.19, 1.93, 0.37]]
    atoms = Atoms("H3", positions=np.asarray(positions) * BOHR_IN_ANGSTROM)
    basis = tmp_path / "synthetic.g94"
    basis.write_text("H 0\nS 1 1.0\n0.8 1.0\n****\n", encoding="ascii")
    fixed_real = np.zeros((6, 6))
    fixed_imaginary = np.zeros_like(fixed_real)
    fixed_real[0, 0], fixed_real[3, 3] = -0.05, 0.05
    fixed_real[1, 4] = fixed_real[4, 1] = 0.027
    fixed_imaginary[2, 5], fixed_imaginary[5, 2] = 0.04, -0.04
    controls = dict(
        basis=str(basis), xc=xc, reference="gks", charge=1, grid_level=0,
        grid_weight_cutoff=0,
        conv_tol=1e-12, conv_tol_grad=1e-11, conv_tol_dp=1e-11, guess="core",
        gks_operator_real=fixed_real.tolist(),
        gks_operator_imaginary=fixed_imaginary.tolist(),
    )
    if xc in ("hf", "weighted"):
        controls.update(pt2_same_spin=0.27, pt2_opposite_spin=0.27)
    if xc == "weighted":
        controls.pop("xc")
        controls["xc_terms"] = [{"xc": "hf", "weight": 0.80}, {"xc": "svwn5", "weight": 0.20}]
    if target is not None:
        controls["double_hybrid_target_xc"] = target
    return positions, atoms, controls


@pytest.mark.parametrize(("xc", "target"), [
    ("hf", None), ("hf", "svwn5"), ("weighted", None), ("b2plyp", None),
], ids=["hf", "hf-svwn5", "weighted", "b2plyp"])
def test_generalized_double_hybrid_quadrupole_public_routes(xc, target, tmp_path):
    positions, atoms, controls = _generalized_electric_fixture(xc, target, tmp_path)
    controls["origin"] = [0.23, -0.5, 0.42]
    native = quadrupole_polarizability([1] * 3, positions, **controls)
    ase = DftRs().get_quadrupole_polarizability(atoms, units="atomic", **controls)
    cli, stderr = _cli_report("quadrupole-polarizability", atoms, controls, tmp_path)
    assert "residual" in stderr
    keys = [
        "quadrupole_polarizability", "parent_quadrupole_polarizability",
        "generalized_pt2_quadrupole_polarizability", "pt2_quadrupole_polarizability_correction",
    ]
    if target is not None:
        keys.append("target_quadrupole_polarizability_correction")
    for report in (native, ase, cli):
        assert report["converged"] and report["reference_type"] == "gks"
        assert report["shape"] == [5, 5] and report["independent_components"] == 15
        assert report["generalized_pt2_coefficient"] == 0.27
        assert abs(report["energy"] - native["energy"]) < 1e-10
        assert len(report["spinor_quadratic_diagnostics"]) == 15
        for key in keys:
            assert report["units"][key] == "e^2*bohr^4/hartree"
            np.testing.assert_allclose(report[key], native[key], rtol=0, atol=1e-8)
        np.testing.assert_allclose(report["response_residuals"], native["response_residuals"],
                                   rtol=0, atol=1e-9)
        if target is not None:
            assert abs(report["energy_components"]["target_correction"]
                       - native["energy_components"]["target_correction"]) < 1e-10
    np.testing.assert_allclose(
        native["pt2_quadrupole_polarizability_correction"],
        0.27 * np.asarray(native["generalized_pt2_quadrupole_polarizability"]),
        rtol=0, atol=1e-12,
    )
    if target is not None:
        expected = (np.asarray(native["parent_quadrupole_polarizability"])
                    + np.asarray(native["pt2_quadrupole_polarizability_correction"])
                    + np.asarray(native["target_quadrupole_polarizability_correction"]))
        np.testing.assert_allclose(native["quadrupole_polarizability"], expected, rtol=0, atol=1e-12)

    energy_controls = dict(controls)
    energy_controls.pop("origin")
    energy = run_feature("energy", [1] * 3, positions, **energy_controls)
    cli_energy, stderr = _cli_report("energy", atoms, energy_controls, tmp_path)
    assert "residual" in stderr
    ase_energy = DftRs().get_energy_report(atoms, units="atomic", **energy_controls)
    for result in (energy, cli_energy, ase_energy):
        assert abs(result["energy"] - native["energy"]) < 1e-10
        assert abs(result["generalized_pt2_correlation"] - native["generalized_pt2_correlation"]) < 1e-10
        assert result["units"]["generalized_pt2_amplitude_residual"] == "hartree"
        assert result["generalized_pt2_amplitude_residual"] < 1e-10


def test_generalized_double_hybrid_gradient_public_routes(tmp_path):
    positions, atoms, controls = _generalized_electric_fixture("b2plyp", None, tmp_path)
    native = gradient([1] * 3, positions, **controls)
    calculator = DftRs()
    atomic = calculator.get_gradient_report(atoms, units="atomic", **controls)
    cli, stderr = _cli_report("gradient", atoms, controls, tmp_path)
    assert "residual" in stderr
    keys = ("gradient", "forces", "parent_gradient", "generalized_pt2_gradient",
            "pt2_gradient", "target_gradient", "dispersion_gradient")
    for report in (native, atomic, cli):
        assert report["converged"] and report["response_converged"]
        assert report["provenance"] == "analytic" and report["reference_type"] == "gks"
        assert len(report["spinor_nuclear_pair_diagnostics"]) == 9
        assert abs(report["energy"] - native["energy"]) < 1e-10
        for key in keys:
            assert report["units"][key] == "hartree/bohr"
            np.testing.assert_allclose(report[key], native[key], rtol=0, atol=1e-9)
        np.testing.assert_allclose(report["response_residuals"], native["response_residuals"],
                                   rtol=0, atol=1e-9)
    np.testing.assert_allclose(native["forces"], -np.asarray(native["gradient"]), rtol=0, atol=0)
    np.testing.assert_allclose(native["pt2_gradient"],
                               0.27 * np.asarray(native["generalized_pt2_gradient"]),
                               rtol=0, atol=1e-13)
    physical_controls = dict(controls)
    _convert_spinor_operator_to_ev(physical_controls)
    for key in ("conv_tol", "conv_tol_grad"):
        physical_controls[key] *= HARTREE_IN_EV
    physical = calculator.get_gradient_report(atoms, units="ase", **physical_controls)
    for key in keys:
        expected = np.asarray(native[key]) * HARTREE_IN_EV / BOHR_IN_ANGSTROM
        np.testing.assert_allclose(physical[key], expected, rtol=0, atol=1e-8)


def test_generalized_distinct_target_dipole_polarizability_public_routes(tmp_path):
    positions, atoms, controls = _generalized_electric_fixture("hf", "svwn5", tmp_path)
    native = polarizability_report([1] * 3, positions, **controls)
    calc = DftRs()
    ase = calc.get_polarizability_report(atoms, units="atomic", **controls)
    physical_controls = dict(controls)
    _convert_spinor_operator_to_ev(physical_controls)
    for key in ("conv_tol", "conv_tol_grad"):
        physical_controls[key] *= HARTREE_IN_EV
    physical = calc.get_polarizability_report(atoms, units="ase", **physical_controls)
    zero = run_feature("dynamic-polarizability", [1] * 3, positions, frequency=0.0, **controls)
    np.testing.assert_allclose(zero["alpha_real"], native["polarizability"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(zero["alpha_imag"], np.zeros((3, 3)), rtol=0, atol=0)
    cli, stderr = _cli_report("polarizability", atoms, controls, tmp_path)
    assert "residual" in stderr

    keys = (
        "polarizability",
        "parent_polarizability",
        "generalized_pt2_polarizability",
        "pt2_polarizability_correction",
        "target_polarizability_correction",
    )
    for report in (native, ase, cli):
        assert report["converged"] and report["reference_type"] == "gks"
        assert report["shape"] == [3, 3] and report["independent_components"] == 6
        assert len(report["spinor_quadratic_diagnostics"]) == 6
        assert abs(report["energy"] - native["energy"]) < 1e-10
        for key in keys:
            assert report["units"][key] == "e^2*bohr^2/hartree"
            np.testing.assert_allclose(report[key], native[key], rtol=0, atol=1e-8)
        np.testing.assert_allclose(report["response_residuals"], native["response_residuals"],
                                   rtol=0, atol=1e-9)
    for key in keys:
        expected = np.asarray(native[key]) * BOHR_IN_ANGSTROM**2 / HARTREE_IN_EV
        np.testing.assert_allclose(physical[key], expected, rtol=0, atol=1e-9)
    print("Installed GKS HF-parent/SVWN5-target dipole polarizability",
          json.dumps(native, sort_keys=True, allow_nan=False))
