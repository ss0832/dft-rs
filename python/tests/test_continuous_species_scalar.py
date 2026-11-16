# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Real saved-scalar routes; a small algebra fixture is not chemical evidence."""
import copy
import json
import math
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs import native
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree

POSITIONS = [[-1.31, .14, .03], [-.22, 1.18, -.12], [.83, -.27, .34], [1.56, .85, -.43]]
WEIGHTS = [.46, .23, .31]


@pytest.fixture(scope="module")
def model():
    reference = native.run_feature(
        "orbitals", [1] * 4, POSITIONS, basis="sto-3g",
        xc_terms=[{"xc": "pbe", "weight": 1.}], coefficients=True,
        grid_level=1, grid_weight_cutoff=0., conv_tol=1e-14,
        conv_tol_dp=1e-13, conv_tol_grad=1e-12, max_iter=300, cpu_threads=1)
    orbitals = np.asarray(reference["channels"][0]["coefficients"])
    # Bundled H STO-3G radial contraction, normalized by the documented
    # molecular AO convention. The public parser independently checks it.
    exponents = [3.425250914, .6239137298, .1688554040]
    coefficients = [.1543289673, .5353281423, .4446345422]
    normalized = [c * (2 * a / math.pi) ** .75 for c, a in zip(coefficients, exponents)]
    norm = math.sqrt(sum(c * d * (math.pi / (a + b)) ** 1.5
                         for c, a in zip(normalized, exponents)
                         for d, b in zip(normalized, exponents)))
    shells = [{"atom": i, "l": 0, "exponents": exponents,
               "normalized_coefficients": [c / norm for c in normalized]} for i in range(4)]
    full = dict(numbers=[1] * 4, positions=POSITIONS, charge=0, multiplicity=1)
    raw = dict(numbers=[1] * 3, positions=POSITIONS[:3], charge=1, multiplicity=1)
    endpoints = []
    references = []
    for i, angle in enumerate([0., .23, -.19]):
        frozen = math.cos(angle) * orbitals[:, 0] + math.sin(angle) * orbitals[:, 1]
        endpoints.append(dict(molecule=full, basis=shells, frozen_orbitals=frozen[:, None].tolist(),
                              geometry_transport="fixed-internal-affine-v1",
                              geometry_atom_owners=[0, 1, 2, 2]))
        references.append(dict(id=f"test-{i}", site_choices=[i],
                               construction_identity="synthetic occupied-rotation API fixture; not a substituent",
                               construction_state="reactant-only", energy=reference["energy"],
                               orbital_residual=reference["scf_orbital_residual"]))
    return dict(model_version="frozen-scalar-action-v1", provider="builtin-functional-registry-v1",
                functional={"xc": "pbe"}, reference="rks",
                shell_convention="normalized-molecular-spherical-ao-v1", grid_level=1,
                sites=[dict(id="test-site", candidates=["a", "b", "c"])], references=references,
                states=[dict(id="test-state", coefficient=1., active_electrons=2,
                             raw_molecule=raw, raw_basis=shells[:3], endpoints=endpoints)])


def evaluate(model, **controls):
    return dft_rs.evaluate_reduced_fragment([1] * 3, POSITIONS[:3], model=model,
                                            fragment_mode="scalar-evaluate",
                                            fragment_weights=WEIGHTS, cpu_threads=1, **controls)


def test_scalar_native_and_ase_saved_record_units(model):
    before = copy.deepcopy(model)
    report = evaluate(model)
    text_report = evaluate(json.dumps(model))
    assert model == before
    assert report["energy"] == pytest.approx(text_report["energy"], abs=1e-12)
    assert report["units"]["reference_residuals"] == "hartree"
    atoms = Atoms("H3", positions=np.asarray(POSITIONS[:3]) * Bohr)
    external = dft_rs.DftRs().get_reduced_fragment(atoms, model=model,
               fragment_mode="scalar-evaluate", fragment_weights=WEIGHTS, cpu_threads=1)
    np.testing.assert_allclose(external["energy"], report["energy"] * Hartree, atol=1e-10)
    np.testing.assert_allclose(external["weight_gradient"], np.asarray(report["weight_gradient"]) * Hartree, atol=1e-9)
    np.testing.assert_allclose(external["reference_residuals"], np.asarray(report["reference_residuals"]) * Hartree, atol=1e-20)
    assert external["units"]["energy"] == "eV"
    assert len(report["weight_gradient"]) == 2
    assert report["states"][0]["scf_orbital_residual"] < 1e-11


def test_scalar_canonical_cli_and_default_scf_log(tmp_path, model):
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for actual scalar route acceptance"
    xyz = tmp_path / "scalar.xyz"
    xyz.write_text("3\nFixed raw AO roles\n" + "".join(
        f"H {x * Bohr:.17g} {y * Bohr:.17g} {z * Bohr:.17g}\n" for x, y, z in POSITIONS[:3]))
    result = subprocess.run([executable, "continuous-species", str(xyz), "--format", "json",
                             "--fragment-mode", "scalar-evaluate", "--fragment-model-json", json.dumps(model),
                             "--fragment-weights", json.dumps(WEIGHTS), "--cpu-threads", "1"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["energy"] == pytest.approx(evaluate(model)["energy"], abs=1e-10)
    assert "[active-scalar]" in result.stderr and "residual" in result.stderr


@pytest.mark.parametrize("strategy", ["joint", "one-pass", "cyclic"])
def test_scalar_bounded_optimizer_reports_its_actual_status(model, strategy):
    report = dft_rs.evaluate_reduced_fragment([1] * 3, POSITIONS[:3], model=model,
        fragment_mode="scalar-optimize", fragment_weights=WEIGHTS, fragment_weight_strategy=strategy,
        fragment_weight_max_iter=2, fragment_weight_tolerance=1e-4, cpu_threads=1)
    status = report["weight_optimization"]
    assert isinstance(status["converged"], bool)
    assert status["projected_gradient"] >= 0
    assert report["energy"] <= evaluate(model)["energy"] + 1e-10


def test_scalar_refuses_changed_geometry_and_incomplete_cartesian_product(model):
    changed = copy.deepcopy(model)
    changed["sites"].append(dict(id="missing-site", candidates=["a", "b"]))
    with pytest.raises((ValueError, RuntimeError)):
        evaluate(changed)
    positions = copy.deepcopy(POSITIONS[:3])
    positions[0][0] += .001
    with pytest.raises((ValueError, RuntimeError)):
        dft_rs.evaluate_reduced_fragment([1] * 3, positions, model=model,
            fragment_mode="scalar-evaluate", fragment_weights=WEIGHTS)


def test_scalar_provider_capability_refusal(model):
    hybrid = copy.deepcopy(model)
    hybrid["functional"] = {"xc": "pbe0"}
    with pytest.raises((ValueError, RuntimeError, NotImplementedError)):
        evaluate(hybrid)


def test_scalar_affine_geometry_native_ase_cli_and_bounded_optimization(model, tmp_path):
    positions = np.asarray(POSITIONS[:3]) + [[.021, -.013, .007], [-.009, .017, -.011], [0., 0., 0.]]
    controls = dict(fragment_state="test-state", fragment_endpoint="test-0", cpu_threads=1)
    report = dft_rs.evaluate_reduced_fragment([1] * 3, positions, model=model,
        fragment_mode="scalar-gradient", **controls)
    assert report["geometry_transport"] == "fixed-internal-affine-v1"
    assert report["energy"] == pytest.approx(report["electronic_energy"] + report["nuclear_energy"], abs=1e-12)
    np.testing.assert_allclose(report["forces"], -np.asarray(report["gradient"]), atol=1e-14)
    atoms = Atoms("H3", positions=positions * Bohr)
    external = dft_rs.DftRs().get_reduced_fragment(atoms, model=model,
        fragment_mode="scalar-gradient", **controls)
    np.testing.assert_allclose(external["gradient"], np.asarray(report["gradient"]) * Hartree / Bohr, atol=1e-9)
    np.testing.assert_allclose(external["full_positions"], np.asarray(report["full_positions"]) * Bohr, atol=1e-12)
    optimized = dft_rs.evaluate_reduced_fragment([1] * 3, positions, model=model,
        fragment_mode="scalar-geometry-optimize", max_steps=2, gtol=1e-5, **controls)
    assert optimized["energy"] <= report["energy"] + 1e-12
    assert isinstance(optimized["geometry_optimization"]["converged"], bool)
    assert optimized["geometry_optimization"]["gradient_tolerance"] == 1e-5
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for saved affine geometry acceptance"
    xyz = tmp_path / "affine.xyz"
    xyz.write_text("3\nMoving raw centers\n" + "".join(
        f"H {x * Bohr:.17g} {y * Bohr:.17g} {z * Bohr:.17g}\n" for x, y, z in positions))
    result = subprocess.run([executable, "continuous-species", str(xyz), "--format", "json",
        "--fragment-mode", "scalar-gradient", "--fragment-model-json", json.dumps(model),
        "--fragment-state", "test-state", "--fragment-endpoint", "test-0", "--cpu-threads", "1"],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    np.testing.assert_allclose(json.loads(result.stdout)["gradient"], report["gradient"], atol=1e-10)
    assert "[active-scalar]" in result.stderr and "residual" in result.stderr


def test_scalar_affine_geometry_refuses_missing_owners_and_weight_mixtures(model):
    controls = dict(fragment_mode="scalar-gradient", fragment_state="test-state", fragment_endpoint="test-0")
    missing = copy.deepcopy(model)
    for endpoint in missing["states"][0]["endpoints"]:
        endpoint.pop("geometry_transport")
        endpoint.pop("geometry_atom_owners")
    with pytest.raises((ValueError, RuntimeError, NotImplementedError)):
        dft_rs.evaluate_reduced_fragment([1] * 3, POSITIONS[:3], model=missing, **controls)
    with pytest.raises((ValueError, RuntimeError, NotImplementedError)):
        dft_rs.evaluate_reduced_fragment([1] * 3, POSITIONS[:3], model=model,
            fragment_weights=WEIGHTS, **controls)


def collinear_model(model):
    """Explicit spin ledger for an algebra fixture, not a fitted chemical model."""
    result = copy.deepcopy(model)
    result.update(model_version="frozen-scalar-collinear-action-v1", reference="uks")
    for state in result["states"]:
        state["active_spin_electrons"] = [1, 1]
        for endpoint in state["endpoints"]:
            endpoint["molecule"].update(charge=1, multiplicity=2)
            endpoint["frozen_orbitals_alpha"] = endpoint.pop("frozen_orbitals")
            endpoint["frozen_orbitals_beta"] = [[] for _ in range(4)]
    return result


@pytest.mark.parametrize("reference", ["rks", "uks"])
def test_scalar_hessian_native_ase_cli(model, tmp_path, reference):
    saved = collinear_model(model) if reference == "uks" else model
    controls = dict(fragment_mode="scalar-hessian", fragment_state="test-state",
                    fragment_endpoint="test-0", response_tolerance=1e-12, cpu_threads=1)
    native_report = dft_rs.evaluate_reduced_fragment([1] * 3, POSITIONS[:3], model=saved, **controls)
    assert native_report["reference"] == reference
    assert np.asarray(native_report["hessian"]).shape == (9, 9)
    assert np.asarray(native_report["raw_hessian"]).shape == (9, 9)
    assert native_report["raw_asymmetry"] < 1e-9
    assert native_report["reference_orbital_residual"] < 1e-11
    assert len(native_report["directional_response"]) == 45
    external = dft_rs.DftRs().get_reduced_fragment(
        Atoms("H3", positions=np.asarray(POSITIONS[:3]) * Bohr), model=saved, **controls)
    for name in ["hessian", "raw_hessian", "raw_asymmetry"]:
        np.testing.assert_allclose(external[name], np.asarray(native_report[name]) * Hartree / Bohr**2,
                                   rtol=1e-9, atol=1e-9)
    for original, converted in zip(native_report["directional_response"], external["directional_response"]):
        assert converted["relative_response_residual"] == pytest.approx(original["relative_response_residual"], abs=1e-11)
        assert converted["orbital_response_residual"] == pytest.approx(original["orbital_response_residual"] * Hartree / Bohr, abs=1e-10)
    assert external["units"]["hessian"] == "eV/angstrom^2"
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for scalar Hessian acceptance"
    xyz = tmp_path / "hessian.xyz"
    xyz.write_text("3\nFixed affine coordinate chart\n" + "".join(
        f"H {x * Bohr:.17g} {y * Bohr:.17g} {z * Bohr:.17g}\n" for x, y, z in POSITIONS[:3]))
    result = subprocess.run([executable, "continuous-species", str(xyz), "--format", "json",
        "--fragment-mode", "scalar-hessian", "--fragment-model-json", json.dumps(saved),
        "--fragment-state", "test-state", "--fragment-endpoint", "test-0",
        "--response-tolerance", "1e-12", "--cpu-threads", "1"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    np.testing.assert_allclose(json.loads(result.stdout)["hessian"], native_report["hessian"], rtol=1e-9, atol=1e-10)
    assert "[active-scalar]" in result.stderr and "residual" in result.stderr


def test_scalar_spin_schema_rejects_missing_blocks_and_implicit_reinterpretation(model):
    invalid = copy.deepcopy(model)
    invalid["reference"] = "uks"
    with pytest.raises((ValueError, RuntimeError, NotImplementedError)):
        evaluate(invalid)
    invalid = collinear_model(model)
    invalid["states"][0]["endpoints"][0].pop("frozen_orbitals_beta")
    with pytest.raises((ValueError, RuntimeError, NotImplementedError)):
        evaluate(invalid)
    with pytest.raises((ValueError, RuntimeError, NotImplementedError)):
        evaluate(collinear_model(model))
