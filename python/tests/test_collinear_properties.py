# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""RKS/UKS/ROKS density properties, independently recomputed on all atoms."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
from dft_rs.units import BOHR_IN_ANGSTROM

import dft_rs
from dft_rs import DftRs

POSITIONS = np.array([[0.13, -0.21, 0.19], [0.27, 1.21, -0.78], [-0.08, -1.69, -0.83]])
CONTROLS = dict(xc="pbe", basis="sto-3g", grid_level=2, grid_weight_cutoff=0.0,
                conv_tol=1e-12, conv_tol_grad=1e-9, max_iter=300, guess="core")


def validate(charges, orbitals, open_shell):
    s = np.asarray(orbitals["overlap"])
    densities = []
    channels = orbitals["channels"]
    for ch in channels:
        c = np.asarray(ch["coefficients"])
        p = np.asarray(ch["density"])
        weight = 2 if len(channels) == 1 else 1
        np.testing.assert_allclose(c.T @ s @ c, np.eye(len(c)), rtol=0, atol=1e-10)
        np.testing.assert_allclose((c * (np.asarray(ch["occupations"]) / weight)) @ c.T,
                                   p, rtol=0, atol=1e-10)
        densities.append(p)
    if len(densities) == 1:
        densities *= 2
    p = sum(densities)
    eig, vectors = np.linalg.eigh(s)
    half = (vectors * np.sqrt(eig)) @ vectors.T
    owner = np.array([0, 0, 0, 0, 0, 1, 2])
    population = lambda m: np.bincount(owner, weights=np.diag(m), minlength=3)
    for name, expected in [
        ("mulliken", np.array([8, 1, 1]) - population(p @ s)),
        ("lowdin", np.array([8, 1, 1]) - population(half @ p @ half)),
        ("mulliken_spin", population((densities[0] - densities[1]) @ s)),
    ]:
        np.testing.assert_allclose(charges[name], expected, rtol=0, atol=1e-10)
        assert abs(sum(charges[name]) - int(open_shell)) < 1e-10
    if charges["reference_type"] == "roks":
        np.testing.assert_allclose(densities[0] @ s @ densities[1], densities[1], rtol=0, atol=1e-10)
    assert charges["converged"] and orbitals["converged"]


@pytest.mark.parametrize("kind, open_shell", [("rks", False), ("uks", False), ("roks", True)])
def test_density_properties_on_native_and_ase(kind, open_shell):
    controls = dict(CONTROLS, reference=kind, charge=int(open_shell), multiplicity=2 if open_shell else 1)
    q = dft_rs.charges([8, 1, 1], POSITIONS.tolist(), **controls)
    orbitals = dft_rs.orbitals([8, 1, 1], POSITIONS.tolist(), coefficients=True, density=True, **controls)
    validate(q, orbitals, open_shell)
    atoms = Atoms("OH2", positions=POSITIONS * Bohr)
    calc = DftRs(**CONTROLS)
    adapted_q = calc.get_charges_report(atoms, units="atomic", **controls)
    adapted_c = calc.get_orbitals_report(atoms, units="atomic", coefficients=True, density=True, **controls)
    validate(adapted_q, adapted_c, open_shell)
    for name in ["mulliken", "lowdin", "mulliken_spin"]:
        np.testing.assert_allclose(q[name], adapted_q[name], rtol=0, atol=1e-10)
    if open_shell:
        direct = dft_rs.roks([8, 1, 1], POSITIONS.tolist(), gradient=False,
                             **{k: v for k, v in controls.items() if k != "reference"})
        assert abs(q["energy"] - direct["energy"]) < 1e-10


def test_roks_population_and_orbital_cli_routes(tmp_path):
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for route acceptance"
    xyz = tmp_path / "water_cation.xyz"
    xyz.write_text("3\nasymmetric water cation\n" + "".join(
        name + " " + " ".join(f"{v:.17g}" for v in row) + "\n"
        for name, row in zip(["O", "H", "H"], POSITIONS * BOHR_IN_ANGSTROM)), encoding="utf-8")
    results = []
    for feature in ["charges", "orbitals"]:
        controls = dict(CONTROLS, reference="roks", charge=1, multiplicity=2)
        if feature == "orbitals":
            controls.update(coefficients=True, density=True)
        args = [executable, feature, str(xyz), "--format", "json"]
        for name, value in controls.items():
            args.extend(["--" + name.replace("_", "-"), str(value).lower()])
        result = subprocess.run(args, text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
        results.append(json.loads(result.stdout))
    validate(*results, True)


def test_rohf_canonical_nuclear_routes_on_native_ase_and_cli(tmp_path):
    positions = np.array([[0.13, -0.21, 0.19], [1.57, 0.16, -0.08],
                          [-0.19, 1.93, 0.37], [0.42, 0.29, 2.28]])
    controls = dict(xc="hf", basis="sto-3g", reference="rohf", charge=1,
                    multiplicity=2, grid_weight_cutoff=0.0, conv_tol=1e-13,
                    conv_tol_grad=1e-12, conv_tol_dp=1e-12, max_iter=300,
                    guess="core", response_tolerance=1e-12)
    gradient = dft_rs.gradient([1] * 4, positions.tolist(), **controls)
    hessian = dft_rs.hessian([1] * 4, positions.tolist(), **controls)
    energy = dft_rs.energy([1] * 4, positions.tolist(), **controls)
    assert gradient["converged"] and hessian["converged"] and energy["converged"]
    assert hessian["response_converged"]
    assert abs(energy["energy"] - gradient["energy"]) < 1e-11
    np.testing.assert_allclose(gradient["gradient"], hessian["gradient"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(gradient["forces"], -np.asarray(gradient["gradient"]), rtol=0, atol=0)
    atoms = Atoms("H4", positions=positions * Bohr)
    calc = DftRs(xc="hf", basis="sto-3g")
    adapted = calc.get_feature_report("hessian", atoms, units="atomic", **controls)
    np.testing.assert_allclose(hessian["hessian"], adapted["hessian"], rtol=0, atol=1e-9)
    atoms.calc = DftRs(units="atomic", **controls)
    np.testing.assert_allclose(atoms.calc.get_hessian(atoms), np.asarray(hessian["hessian"]) * Hartree / Bohr**2,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(atoms.get_forces(), -np.asarray(gradient["gradient"]) * Hartree / Bohr, rtol=0, atol=1e-8)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for route acceptance"
    xyz = tmp_path / "restricted_open_shell_nuclear.xyz"
    xyz.write_text("4\nrestricted open shell nuclear response\n" + "".join(
        "H " + " ".join(f"{v:.17g}" for v in row) + "\n"
        for row in positions * BOHR_IN_ANGSTROM), encoding="utf-8")
    args = [executable, "hessian", str(xyz), "--format", "json"]
    for name, value in controls.items():
        args += ["--" + name.replace("_", "-"), str(value).lower()]
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    np.testing.assert_allclose(hessian["hessian"], json.loads(result.stdout)["hessian"], rtol=0, atol=1e-9)


@pytest.mark.parametrize("reference", ["rks", "uks"])
def test_tpss_complete_nuclear_and_electric_tensors_share_public_routes(tmp_path, reference):
    """All tensor entries agree; independent derivative fits are in p283."""
    from test_fixed_populations import cli_report

    positions = np.array([[0., 0., .2217], [0., 1.4309, -.8867], [.13, -1.4309, -.8867]])
    atoms = Atoms("OH2", positions=positions * Bohr)
    controls = dict(xc="tpss", reference=reference, basis="sto-3g", dispersion="none",
                    grid_level=3, grid_weight_cutoff=0., max_iter=500,
                    conv_tol=1e-14, conv_tol_grad=1e-12, conv_tol_dp=1e-12)
    physical_controls = dict(controls, conv_tol=controls["conv_tol"] * Hartree,
                             conv_tol_grad=controls["conv_tol_grad"] * Hartree)
    for feature, function, key, count, factor in [
        ("gradient", dft_rs.gradient, "gradient", 9, Hartree / Bohr),
        ("hessian", dft_rs.hessian_report, "hessian", 81, Hartree / Bohr**2),
        ("polarizability", dft_rs.polarizability_report, "polarizability", 9, Bohr**2 / Hartree),
        ("beta", dft_rs.beta, "tensor", 27, Bohr**3 / Hartree**2),
    ]:
        report = function([8, 1, 1], positions.tolist(), **controls)
        assert report["converged"]
        values = np.asarray(report[key]).reshape(-1)
        assert values.size == count
        atomic = DftRs(units="atomic").get_feature_report(feature, atoms, **controls)
        physical = DftRs().get_feature_report(feature, atoms, **physical_controls)
        cli = cli_report(tmp_path, feature, atoms, controls)
        defects = []
        for other, scale in [(atomic, 1.), (physical, factor), (cli, 1.)]:
            assert other["converged"]
            candidate = np.asarray(other[key]).reshape(-1) / scale
            assert candidate.size == count
            np.testing.assert_allclose(candidate, values, rtol=0., atol=1e-8)
            defects.append(float(np.max(np.abs(candidate - values))))
            if feature == "polarizability":
                np.testing.assert_allclose(np.asarray(other["polarizability_angstrom3"]).reshape(-1),
                                           values * Bohr**3, rtol=0., atol=1e-9)
        print("TPSS complete public tensor", reference, feature, count, "defects", defects, flush=True)


@pytest.mark.parametrize("reference", ["rks", "uks"])
def test_meta_gga_response_cache_budget_preserves_every_public_alpha_entry(tmp_path, reference):
    """The bounded cache and its on-demand fallback use the same physical kernel."""
    from test_fixed_populations import cli_report

    positions = np.array([[0., 0., .2217], [0., 1.4309, -.8867], [.13, -1.4309, -.8867]])
    atoms = Atoms("OH2", positions=positions * Bohr)
    controls = dict(xc="tpss", reference=reference, basis="sto-3g", dispersion="none",
                    grid_level=3, grid_weight_cutoff=0., max_iter=500,
                    conv_tol=1e-14, conv_tol_grad=1e-12, conv_tol_dp=1e-12)
    baseline = dft_rs.polarizability_report([8, 1, 1], positions.tolist(), incore_mb=2048, **controls)
    assert baseline["converged"]
    exact = np.asarray(baseline["polarizability"])
    assert exact.shape == (3, 3)
    for budget in [1, 2048]:
        options = dict(controls, incore_mb=budget)
        native = dft_rs.polarizability_report([8, 1, 1], positions.tolist(), **options)
        ase = DftRs(units="atomic").get_feature_report("polarizability", atoms, **options)
        cli = cli_report(tmp_path, "polarizability", atoms, options)
        for report in [native, ase, cli]:
            assert report["converged"]
            np.testing.assert_array_equal(np.asarray(report["polarizability"]), exact)
            assert report["units"]["response_absolute_residuals"] == "e*bohr"
            residuals = report["response_absolute_residuals"]
            assert len(residuals) == 3 and max(residuals) <= 1e-10
            np.testing.assert_allclose(residuals, baseline["response_absolute_residuals"],
                                       rtol=0, atol=1e-12)
        print("meta-GGA public cache budget", reference, budget, "all 9 entries bitwise equal", flush=True)


@pytest.mark.parametrize("reference", ["rks", "uks"])
@pytest.mark.parametrize("feature", ["hessian", "frequencies"])
def test_nuclear_cpks_fresh_residuals_on_all_public_routes(tmp_path, reference, feature):
    """The fresh nuclear CPKS defect retains its units through every adapter."""
    from test_fixed_populations import cli_report

    positions = np.array([[0.13, -0.21, 0.19], [1.57, 0.16, -0.08],
                          [-0.19, 1.93, 0.37], [0.42, 0.29, 2.28]])
    open_shell = reference == "uks"
    controls = dict(xc="pbe", basis="sto-3g", reference=reference,
                    charge=int(open_shell), multiplicity=2 if open_shell else 1,
                    grid_level=0, grid_weight_cutoff=0.0, dispersion="none",
                    conv_tol=1e-13, conv_tol_grad=1e-12, conv_tol_dp=1e-12,
                    max_iter=400, guess="core")
    native = dft_rs.run_feature(feature, [1] * 4, positions.tolist(), **controls)
    atoms = Atoms("H4", positions=positions * Bohr)
    atomic = DftRs(units="atomic").get_feature_report(feature, atoms, **controls)
    physical_controls = dict(controls, conv_tol=controls["conv_tol"] * Hartree,
                             conv_tol_grad=controls["conv_tol_grad"] * Hartree)
    physical = DftRs().get_feature_report(feature, atoms, **physical_controls)
    cli = cli_report(tmp_path, feature, atoms, controls)
    for report, factor in [(native, 1.), (atomic, 1.), (cli, 1.), (physical, Hartree / Bohr)]:
        assert report["converged"] and report["provenance"] == "analytic"
        residuals = np.asarray(report["orbital_response_residuals"]) / factor
        assert residuals.shape == (12,)
        assert np.isfinite(residuals).all() and np.max(residuals) <= 1e-12
        scale = Hartree if report is physical else 1.
        assert 0. <= report["scf_orbital_residual"] / scale <= 1e-10
        key = "hessian" if feature == "hessian" else "frequencies_cm1"
        value_scale = Hartree / Bohr**2 if feature == "hessian" and report is physical else 1.
        np.testing.assert_allclose(np.asarray(report[key]) / value_scale, native[key], rtol=0, atol=1e-8)
