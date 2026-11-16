# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Actual saved-candidate routes and accounting, without accuracy promotion."""
import copy
import json
import os
import re
import subprocess

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree


NUMBERS = [6, 1, 1, 1, 6, 1, 1, 1]
POSITIONS = [
    [-1.46, -0.14, 0.23], [-2.13, 1.69, 0.68],
    [-2.28, -1.57, 1.31], [-2.19, -0.43, -1.65],
    [1.35, 0.17, -0.12], [2.05, 2.08, 0.16],
    [2.10, -0.92, 1.45], [2.01, -0.84, -1.67],
]
SPEC = dict(explicit_atoms=[4, 5, 6, 7], retained_atoms=[0, 1, 2, 3],
            port=[4, 0, 1], electronic_centers=1,
            single_bonds=[[0, 1], [0, 2], [0, 3], [0, 4], [4, 5], [4, 6], [4, 7]])
REDUCED_NUMBERS = [6, 1, 1, 1, 1]
REDUCED_POSITIONS = [POSITIONS[i] for i in [0, 1, 2, 3, 4]]
FIELD = np.array([0.0013, -0.0017, 0.0009])
ORIGIN = np.array([0.11, -0.07, 0.19])
H1S = "one-center-h1s-local3-projector8-charge3-v1"
VALENCE_SP = "one-center-valence-sp-local3-projector8-charge3-v1"
CORE_VALENCE_SP = "one-center-core-valence-sp-local3-projector8-charge3-v1"
ERRORS = (ValueError, RuntimeError, NotImplementedError)


@pytest.fixture(scope="module")
def generated():
    return dft_rs.generate_methyl_fragment(
        NUMBERS, POSITIONS, fragment_spec=copy.deepcopy(SPEC), cpu_threads=1)


@pytest.fixture(scope="module")
def evaluated(generated):
    return dft_rs.evaluate_reduced_fragment(
        REDUCED_NUMBERS, REDUCED_POSITIONS, model=generated["model_json"],
        field=FIELD.tolist(), charge_origin=ORIGIN.tolist(), cpu_threads=1)


def check_status(report):
    assert report["candidate_status"] == "candidate-not-chemically-validated"
    assert report["shape_status"] == "not-generated"
    assert report["response_status"] == "active-only-no-internal-polarization"
    assert "polarizability" not in report
    assert "dynamic_polarizability_real" not in report
    assert report["counts"]["full_electrons"] == 18
    assert report["counts"]["frozen_electrons"] == 8
    assert report["counts"]["active_electrons"] == 10


def test_generated_native_model_is_opaque_and_keeps_frozen_electron_ledger(generated):
    check_status(generated)
    assert isinstance(generated["model_json"], str)
    model = json.loads(generated["model_json"])
    assert model["ledger"] == [18, 8, 10]
    assert model["baseline_ionic_charge"] == 1
    assert model["source_atoms"][4]["z"] == 6
    np.testing.assert_array_equal([atom["position"] for atom in model["source_atoms"]], POSITIONS)
    assert model["template_id"] == generated["template_id"] == H1S
    assert model["fingerprint"] == generated["fingerprint"]
    assert generated["counts"]["full_ao_count"] == 16
    assert generated["counts"]["reduced_ao_count"] == 9
    assert generated["units"]["model_json"].startswith("text")
    assert generated["units"]["reference_energy"] == "hartree"
    assert "unspecified" not in generated["units"].values()
    assert generated["generation_status"] == "generated-candidate-not-chemical-acceptance"


def test_native_dict_ase_default_and_atomic_getter_parity(generated, evaluated):
    saved_text = generated["model_json"]
    check_status(evaluated)
    assert evaluated["active_density_trace"] == pytest.approx(10, abs=1e-9)
    assert evaluated["physical_total_charge"] == pytest.approx(0, abs=1e-10)
    assert np.isfinite(evaluated["scf_orbital_residual"])
    assert evaluated["electronic_energy"] + evaluated["classical_energy"] == pytest.approx(
        evaluated["energy"], abs=1e-11)
    assert evaluated["energy_without_internal_reference"] + evaluated["fragment_internal_energy"] == pytest.approx(
        evaluated["energy"], abs=1e-11)
    assert evaluated["classical_energy_without_internal_reference"] + evaluated["fragment_internal_energy"] == pytest.approx(
        evaluated["classical_energy"], abs=1e-11)
    assert "excluded from trial energies" in evaluated["scf_energy_convention"]
    np.testing.assert_allclose(evaluated["forces"], -np.asarray(evaluated["gradient"]), atol=1e-12)
    model = json.loads(generated["model_json"])
    before = copy.deepcopy(model)
    atoms = Atoms(numbers=REDUCED_NUMBERS, positions=np.array(REDUCED_POSITIONS) * Bohr)
    # Ordinary calculator basis/functional defaults must not reach this mode.
    calc = dft_rs.DftRs()
    atomic = calc.get_reduced_fragment(atoms, model=model, units="atomic",
                                      field=FIELD.tolist(), charge_origin=ORIGIN.tolist(), cpu_threads=1)
    external = calc.get_reduced_fragment(atoms, model=generated["model_json"],
                                        field=(FIELD * Hartree / Bohr).tolist(),
                                        charge_origin=(ORIGIN * Bohr).tolist(), cpu_threads=1)
    assert model == before
    assert generated["model_json"] == saved_text
    for report in [atomic, external]:
        check_status(report)
        assert report["template_id"] == evaluated["template_id"]
        assert report["fingerprint"] == evaluated["fingerprint"]
    for key, factor in [("energy", Hartree), ("electronic_energy", Hartree),
                        ("classical_energy", Hartree), ("fragment_internal_energy", Hartree),
                        ("energy_without_internal_reference", Hartree),
                        ("classical_energy_without_internal_reference", Hartree),
                        ("forces", Hartree / Bohr),
                        ("gradient", Hartree / Bohr), ("dipole", Bohr),
                        ("permanent_ionic_dipole", Bohr), ("active_electronic_dipole", Bohr)]:
        np.testing.assert_allclose(atomic[key], evaluated[key], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(external[key], np.asarray(evaluated[key]) * factor,
                                   rtol=1e-9, atol=1e-8)
    assert atomic["units"]["energy"] == "hartree"
    assert external["units"]["energy"] == "eV"


def test_cli_saved_model_matches_native_and_logged_total(tmp_path, generated, evaluated):
    _assert_saved_model_cli_parity(tmp_path, generated, evaluated)


def _assert_saved_model_cli_parity(tmp_path, generated, evaluated):
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for actual reduced-fragment route acceptance"
    path = tmp_path / "reduced_methyl.xyz"
    path.write_text("5\nRetained atoms and one ionic-role center\n" + "".join(
        f"{symbol} {x * Bohr:.17g} {y * Bohr:.17g} {z * Bohr:.17g}\n"
        for symbol, (x, y, z) in zip(["C", "H", "H", "H", "H"], REDUCED_POSITIONS)), encoding="utf-8")
    # JSON is a single argv item; no shell expansion or shell-built command.
    args = [executable, "continuous-species", str(path), "--format", "json",
            "--fragment-mode", "evaluate", "--fragment-model-json", generated["model_json"],
            "--field", json.dumps(FIELD.tolist()), "--charge-origin", json.dumps(ORIGIN.tolist()),
            "--cpu-threads", "1"]
    process = subprocess.run(args, capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    check_status(report)
    energies = re.findall(r"\[scf\] converged.*\(E = ([+\-\d.eE]+)\)", process.stderr)
    assert energies, process.stderr
    assert float(energies[-1]) == pytest.approx(
        report["energy_without_internal_reference"], abs=1e-10)
    assert report["scf_energy_convention"] == evaluated["scf_energy_convention"]
    assert "[scf]" in process.stderr
    for key in ["energy", "electronic_energy", "classical_energy", "forces", "gradient", "dipole",
                "energy_without_internal_reference", "classical_energy_without_internal_reference",
                "fragment_internal_energy"]:
        np.testing.assert_allclose(report[key], evaluated[key], rtol=1e-9, atol=1e-8)


def test_legacy_missing_projection_rule_remains_a_valid_saved_model(generated, evaluated):
    model = json.loads(generated["model_json"])
    assert model["options"].pop("projection_rule") == "full-ao-metric-v1"
    report = dft_rs.evaluate_reduced_fragment(
        REDUCED_NUMBERS, REDUCED_POSITIONS, model=model,
        field=FIELD.tolist(), charge_origin=ORIGIN.tolist(), cpu_threads=1)
    check_status(report)
    assert report["fingerprint"] == evaluated["fingerprint"]
    np.testing.assert_allclose(report["forces"], evaluated["forces"], atol=1e-9, rtol=1e-9)
    assert report["energy"] == pytest.approx(evaluated["energy"], abs=1e-10)


@pytest.mark.parametrize("override", [
    {"xc": "lda"}, {"basis": "sto-3g"}, {"grid_level": 2},
    {"multiplicity": 3}, {"charge": 1}, {"frequency": 0.1},
    {"ecp": True}, {"unknown_option": True},
])
def test_saved_model_rejects_incompatible_or_unknown_controls(generated, override):
    with pytest.raises(ERRORS):
        dft_rs.evaluate_reduced_fragment(REDUCED_NUMBERS, REDUCED_POSITIONS,
                                        model=generated["model_json"], **override)


@pytest.mark.parametrize("mutation", ["version", "extra", "nested_extra", "rule"])
def test_unknown_saved_model_content_is_rejected(generated, mutation):
    model = json.loads(generated["model_json"])
    if mutation == "version":
        model["model_version"] = "future-invalid"
    elif mutation == "extra":
        model["surprise"] = 1
    elif mutation == "nested_extra":
        model["options"]["surprise"] = 1
    else:
        model["options"]["projection_rule"] = "unknown-rule"
    with pytest.raises(ERRORS):
        dft_rs.evaluate_reduced_fragment(REDUCED_NUMBERS, REDUCED_POSITIONS, model=model)


def test_scope_labels_and_fractional_element_validation(generated):
    for numbers in [[6, 1, 1, 1, 6], [6, 1, 1, 1, 1.2]]:
        with pytest.raises(ERRORS):
            dft_rs.evaluate_reduced_fragment(numbers, REDUCED_POSITIONS, model=generated["model_json"])
    with pytest.raises(ERRORS):
        dft_rs.evaluate_reduced_fragment(REDUCED_NUMBERS, REDUCED_POSITIONS,
                                        model=generated["model_json"], cell=np.eye(3).tolist(), pbc=[True] * 3)
    atoms = Atoms(numbers=REDUCED_NUMBERS, positions=np.array(REDUCED_POSITIONS) * Bohr,
                  cell=np.eye(3) * 10, pbc=True)
    with pytest.raises(PropertyNotImplementedError, match="periodic boundary conditions"):
        dft_rs.DftRs().get_reduced_fragment(atoms, model=generated["model_json"])
    invalid = copy.deepcopy(SPEC)
    invalid["unknown_spec_option"] = True
    with pytest.raises(ERRORS):
        dft_rs.generate_methyl_fragment(NUMBERS, POSITIONS, fragment_spec=invalid)
    with pytest.raises(ValueError, match="atomic numbers"):
        dft_rs.generate_methyl_fragment([6.25, *NUMBERS[1:]], POSITIONS, fragment_spec=SPEC)


@pytest.mark.skipif(os.environ.get("DFT_RS_TEST_REDUCED_SP") != "1",
                    reason="optional additional generated AO-template acceptance")
def test_valence_sp_template_generation_keeps_ledger_and_changes_only_declared_template(generated):
    report = dft_rs.generate_methyl_fragment(
        NUMBERS, POSITIONS, fragment_spec=copy.deepcopy(SPEC),
        fragment_basis_template=VALENCE_SP, cpu_threads=1)
    check_status(report)
    assert report["template_id"] == VALENCE_SP
    assert report["counts"]["reduced_ao_count"] == 12
    assert report["counts"]["full_ao_count"] == 16
    assert report["fingerprint"] != generated["fingerprint"]
    model = json.loads(report["model_json"])
    assert model["template_id"] == VALENCE_SP
    assert json.loads(json.dumps(model)) == model
    assert len(model["nonlocal_coefficients"]) == 36
    restored = dft_rs.evaluate_reduced_fragment(
        REDUCED_NUMBERS, REDUCED_POSITIONS, model=json.dumps(model), cpu_threads=1)
    check_status(restored)
    assert restored["template_id"] == VALENCE_SP
    assert restored["fingerprint"] == report["fingerprint"]
    assert restored["counts"]["reduced_ao_count"] == 12
    assert restored["active_density_trace"] == pytest.approx(10, abs=1e-9)


@pytest.mark.skipif(os.environ.get("DFT_RS_TEST_REDUCED_CORE") != "1",
                    reason="additional core-basis template requires its compiled artifact")
def test_core_basis_template_preserves_frozen_ledger_and_public_routes(tmp_path, generated):
    report = dft_rs.generate_methyl_fragment(
        NUMBERS, POSITIONS, fragment_spec=copy.deepcopy(SPEC),
        fragment_basis_template=CORE_VALENCE_SP, cpu_threads=1)
    check_status(report)
    assert report["template_id"] == CORE_VALENCE_SP
    assert report["counts"]["reduced_ao_count"] == 13
    model = json.loads(report["model_json"])
    original = json.loads(generated["model_json"])
    assert model["ledger"] == [18, 8, 10]
    assert model["baseline_ionic_charge"] == 1
    for key in ["gaussian_charges", "gaussian_body_dipoles", "local_amplitudes",
                "nonlocal_coefficients", "internal_energy_reference"]:
        assert model[key] == original[key]
    assert report["fingerprint"] != generated["fingerprint"]
    evaluated = dft_rs.evaluate_reduced_fragment(
        REDUCED_NUMBERS, REDUCED_POSITIONS, model=model,
        field=FIELD.tolist(), charge_origin=ORIGIN.tolist(), cpu_threads=1)
    check_status(evaluated)
    assert evaluated["active_density_trace"] == pytest.approx(10, abs=1e-9)
    assert evaluated["counts"]["reduced_ao_count"] == 13
    atoms = Atoms(numbers=REDUCED_NUMBERS, positions=np.array(REDUCED_POSITIONS) * Bohr)
    converted = dft_rs.DftRs().get_reduced_fragment(
        atoms, model=model, field=(FIELD * Hartree / Bohr).tolist(),
        charge_origin=(ORIGIN * Bohr).tolist(), cpu_threads=1)
    assert converted["energy"] == pytest.approx(evaluated["energy"] * Hartree, abs=1e-8)
    np.testing.assert_allclose(converted["forces"], np.asarray(evaluated["forces"]) * Hartree / Bohr,
                               atol=1e-8, rtol=1e-9)
    _assert_saved_model_cli_parity(tmp_path, report, evaluated)
