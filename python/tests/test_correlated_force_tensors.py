# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Complete force-tensor selections on every public route.

Independent complete-component derivative fits are in p262. Public equality,
unit conversion and zero-coefficient limits are additional acceptance only.
"""
import json

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs import DftRs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
from test_correlated_high_order import request, cli

PREFIXES = ["", "parent_", "target_", "pt2_", "pt2_same_spin_alpha_",
            "pt2_opposite_spin_", "pt2_same_spin_beta_", "dispersion_"]


def test_force_tensor_response_aliases_preserve_user_controls(monkeypatch):
    calls = []

    def backend(feature, numbers, positions, **controls):
        calls.append(controls)
        return {"units": {}, "response_units": "hartree"}

    monkeypatch.setattr(dft_rs.native._core, "run_feature", backend)
    dft_rs.analytic_force_constants([1], [[0., 0., 0.]], coordinates=[0],
                                    response_tolerance=1.2e-12, response_max_iterations=321)
    assert calls[-1]["response_tolerance"] == 1.2e-12
    assert calls[-1]["response_max_iterations"] == 321
    assert "response_tol" not in calls[-1] and "response_max_iter" not in calls[-1]
    with pytest.raises(ValueError, match="use either"):
        dft_rs.analytic_force_constants([1], [[0., 0., 0.]], coordinates=[0],
                                        response_tol=1e-8, response_tolerance=1e-12)
    with pytest.raises(TypeError, match="force constants"):
        dft_rs.analytic_force_constants([1], [[0., 0., 0.]], response_order=4)


def selection(mode, atoms):
    if mode == "block":
        return {"coordinates": [0, 4, 8]}
    if mode == "full":
        return {}
    return {"patterns": [[[.51 * np.sin(.7 * (3 * a + i + 1) * (p + 1))
                           + .23 * np.cos(.31 * (3 * a + i + 2) * (p + 2))
                           for i in range(3)] for a in range(atoms)] for p in range(3)]}


def compare_reports(native, report, route):
    assert report["converged"] and report["response_converged"]
    assert report["shape"] == native["shape"]
    assert report["permutation_defect"] is None
    assert report["permutation_comparisons"] == 0
    labels = report.get("unit_labels", report["units"])
    for prefix in PREFIXES:
        for suffix in ("tensor", "accumulation_roundoff"):
            key = prefix + suffix
            np.testing.assert_allclose(report[key], native[key], rtol=1e-10, atol=1e-8,
                                       err_msg=f"{route}: {key}")
            assert key in labels
    np.testing.assert_allclose(report["tensor"], np.asarray(report["target_tensor"])
                               + report["pt2_tensor"] + report["dispersion_tensor"], atol=1e-10)
    np.testing.assert_allclose(report["pt2_tensor"], report["pt2_same_spin_coefficient"]
        * (np.asarray(report["pt2_same_spin_alpha_tensor"]) + report["pt2_same_spin_beta_tensor"])
        + report["pt2_opposite_spin_coefficient"] * np.asarray(report["pt2_opposite_spin_tensor"]), atol=1e-10)
    for key in ("response_residuals", "amplitude_relative_residuals", "projector_defects",
                "metric_defects", "response_number_defects", "orbital_frame_defects"):
        assert max(map(abs, report[key]), default=0.) < 1e-9


@pytest.mark.parametrize("order", [3, 4])
@pytest.mark.parametrize("mode", ["patterns", "block"])
@pytest.mark.parametrize("model", ["custom", "b2plyp"])
@pytest.mark.parametrize("unrestricted", [False, True])
def test_complete_correlated_force_tensor_public_selections(order, mode, model, unrestricted, tmp_path):
    numbers, positions, controls = request(model, unrestricted)
    selected = selection(mode, len(numbers))
    native = dft_rs.analytic_force_constants(numbers, positions, order=order, **selected, **controls)
    common_controls = dict(controls, response_order=order, **selected)
    common = dft_rs.run_feature("force-constants", numbers, positions, **common_controls)
    atoms = Atoms(numbers=numbers, positions=positions * Bohr)
    adapted = DftRs(units="atomic").get_analytic_force_constants(atoms, order=order, **selected, **controls)
    command = cli("force-constants", numbers, positions, common_controls, tmp_path)
    for route, report in zip(("native", "common", "ase", "cli"), (native, common, adapted, command)):
        (tmp_path / f"{route}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        compare_reports(native, report, route)
        assert report["shape"] == [3] * order
        assert report["translation_residual"] is None and report["translation_comparisons"] == 0


@pytest.mark.parametrize("order", [3, 4])
def test_complete_cartesian_tensor_and_zero_correlation_limit(order):
    numbers, positions, controls = request("custom", False)
    report = dft_rs.analytic_force_constants(numbers, positions, order=order, **controls)
    dimension = 3 * len(numbers)
    assert report["converged"] and report["response_converged"]
    assert report["shape"] == [dimension] * order
    assert len(report["tensor"]) == dimension**order
    for prefix in PREFIXES:
        assert report[prefix + "translation_residual"] < 1e-8
    assert report["translation_comparisons"] == dimension**(order - 1) * order * 3
    limited = dict(controls, pt2_same_spin=0., pt2_opposite_spin=0.)
    zero = dft_rs.analytic_force_constants(numbers, positions, order=order,
                                          **selection("patterns", len(numbers)), **limited)
    assert zero["converged"]
    assert np.count_nonzero(zero["pt2_tensor"]) == 0
    np.testing.assert_allclose(zero["tensor"], np.asarray(zero["target_tensor"])
                               + zero["dispersion_tensor"], rtol=0., atol=1e-12)


@pytest.mark.parametrize("mode", ["patterns", "block"])
def test_force_tensor_ase_energy_length_and_field_units(mode):
    numbers, positions, controls = request("custom", False)
    selected = selection(mode, len(numbers))
    atomic = dft_rs.analytic_force_constants(numbers, positions, order=4, **selected, **controls)
    physical = dict(controls, conv_tol=controls["conv_tol"] * Hartree,
                    conv_tol_grad=controls["conv_tol_grad"] * Hartree,
                    reference_field=np.asarray(controls["reference_field"]) * Hartree / Bohr)
    physical_selected = selected.copy()
    if "patterns" in selected:
        physical_selected["patterns"] = np.asarray(selected["patterns"]) * Bohr
    atoms = Atoms(numbers=numbers, positions=positions * Bohr)
    report = DftRs().get_analytic_force_constants(atoms, order=4, **physical_selected, **physical)
    factor = Hartree if mode == "patterns" else Hartree / Bohr**4
    for prefix in PREFIXES:
        np.testing.assert_allclose(report[prefix + "tensor"], np.asarray(atomic[prefix + "tensor"]) * factor,
                                   rtol=1e-10, atol=1e-7)
    assert report["unit_labels"]["tensor"] == ("eV" if mode == "patterns" else "eV/angstrom^4")


def test_named_dispersion_default_matches_the_energy_model():
    numbers, positions, controls = request("custom", False)
    for key in ("double_hybrid_target_xc", "pt2_same_spin", "pt2_opposite_spin", "dispersion"):
        controls.pop(key)
    controls["xc"] = "dsd-blyp"
    report = dft_rs.analytic_force_constants(numbers, positions, order=3,
                                            **selection("patterns", len(numbers)), **controls)
    energy_controls = {key: value for key, value in controls.items() if key != "response_tolerance"}
    energy = dft_rs.energy(numbers, positions, **energy_controls)
    assert report["converged"]
    assert report["energy"] == pytest.approx(energy["energy"], rel=0., abs=1e-12)
    assert report["energy_components"].keys() == energy["energy_components"].keys()
    for key, value in energy["energy_components"].items():
        assert report["energy_components"][key] == pytest.approx(value, rel=0., abs=1e-12), key
    assert np.linalg.norm(report["dispersion_tensor"]) > 1e-10


@pytest.mark.parametrize("unrestricted", [False, True])
def test_correlated_property_energy_components_match_the_energy_driver(unrestricted):
    numbers, positions, controls = request("custom", unrestricted)
    energy_controls = {key: value for key, value in controls.items() if key != "response_tolerance"}
    expected = dft_rs.energy(numbers, positions, **energy_controls)
    reports = [dft_rs.analytic_force_constants(numbers, positions, order=3,
        **selection("patterns", len(numbers)), **controls),
        dft_rs.beta(numbers, positions, **controls)]
    for report in reports:
        assert report["converged"]
        components = report["energy_components"]
        assert components.keys() == expected["energy_components"].keys()
        for key, value in expected["energy_components"].items():
            assert components[key] == pytest.approx(value, rel=0., abs=1e-12), key
        assert components["perturbative_correlation"] == pytest.approx(report["pt2_correlation"], rel=0., abs=1e-12)
