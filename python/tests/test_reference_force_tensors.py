# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Complete constrained/spinor tensor routes; independent full fits are in p264."""
import json

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs import DftRs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
from test_correlated_high_order import cli, request


def fixture(reference):
    numbers, positions, options = request("custom", False)
    for key in ("pt2_same_spin", "pt2_opposite_spin", "double_hybrid_target_xc"):
        options.pop(key)
    options.update(reference=reference, charge=1, multiplicity=2, max_iter=500, dispersion="d3bj")
    if reference in ("gks", "ghf"):
        re = np.zeros((8, 8))
        im = np.zeros_like(re)
        for i in range(4):
            re[i, i] = -.013 * (i + 1)
            re[4 + i, 4 + i] = -re[i, i]
            re[i, 4 + i] = re[4 + i, i] = .007 * (i + 1)
            im[i, 4 + i] = .009 * (4 - i)
            im[4 + i, i] = -im[i, 4 + i]
        options.update(gks_operator_real=re.tolist(), gks_operator_imaginary=im.tolist(),
                       spin_axis=[.3, .4, .5])
    return numbers, positions, options


def selection(mode, atoms):
    if mode == "block":
        return dict(coordinates=[8, 0, 4])
    a = np.arange(1, 3 * atoms + 1, dtype=float)
    patterns = np.asarray([.51 * np.sin(.7 * a * (p + 1))
                           + .23 * np.cos(.31 * (a + 1) * (p + 2)) for p in range(3)])
    return dict(patterns=patterns.reshape(3, atoms, 3).tolist())


@pytest.mark.parametrize("reference", ["roks", "gks"])
@pytest.mark.parametrize("order", [3, 4])
@pytest.mark.parametrize("mode", ["block", "patterns"])
def test_reference_force_tensors_on_all_four_public_routes(reference, order, mode, tmp_path):
    numbers, positions, options = fixture(reference)
    chosen = selection(mode, len(numbers))
    common = dict(options, response_order=order, **chosen)
    reports = [dft_rs.analytic_force_constants(numbers, positions, order=order, **chosen, **options),
               dft_rs.run_feature("force-constants", numbers, positions, **common),
               DftRs(units="atomic").get_analytic_force_constants(
                   Atoms(numbers=numbers, positions=positions * Bohr), order=order, **chosen, **options),
               cli("force-constants", numbers, positions, common, tmp_path)]
    for index, report in enumerate(reports):
        (tmp_path / f"route-{index}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        assert report["converged"] and report["response_converged"]
        assert report["shape"] == [3] * order
        assert report["permutation_defect"] is None and report["permutation_comparisons"] == 0
        assert report["translation_residual"] is None and report["translation_comparisons"] == 0
        assert report["assembly"] == "energy-polarization"
        for prefix in ("", "parent_", "dispersion_"):
            np.testing.assert_allclose(report[prefix + "tensor"], reports[0][prefix + "tensor"],
                                       rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(report["tensor"], np.asarray(report["parent_tensor"])
                                   + report["dispersion_tensor"], rtol=1e-11, atol=1e-9)
        labels = report.get("unit_labels", report["units"])
        assert labels["tensor"] == ("hartree" if mode == "patterns" else f"hartree/bohr^{order}")
        for key in ("response_residuals", "projector_defects", "nuclear_stationarity_defects", "reference_defects"):
            assert max(map(abs, report[key]), default=0.) < 1e-9, key


@pytest.mark.parametrize("reference,alias", [("roks", "rohf"), ("gks", "ghf")])
def test_hf_reference_aliases_select_the_same_complete_tensor_provider(reference, alias):
    numbers, positions, options = fixture(reference)
    options.update(coordinates=[0, 4])
    report = dft_rs.analytic_force_constants(numbers, positions, order=4, **options)
    options["reference"] = alias
    aliased = dft_rs.analytic_force_constants(numbers, positions, order=4, **options)
    np.testing.assert_allclose(aliased["tensor"], report["tensor"], rtol=0., atol=1e-12)


@pytest.mark.parametrize("reference", ["roks", "gks"])
def test_reference_force_tensor_ase_energy_length_and_field_units(reference):
    numbers, positions, options = fixture(reference)
    options["coordinates"] = [8, 0, 4]
    atomic = dft_rs.analytic_force_constants(numbers, positions, order=4, **options)
    physical = dict(options, conv_tol=options["conv_tol"] * Hartree,
                    conv_tol_grad=options["conv_tol_grad"] * Hartree,
                    reference_field=np.asarray(options["reference_field"]) * Hartree / Bohr)
    for key in ("gks_operator_real", "gks_operator_imaginary"):
        if key in physical:
            physical[key] = np.asarray(physical[key]) * Hartree
    report = DftRs().get_analytic_force_constants(Atoms(numbers=numbers, positions=positions * Bohr),
                                               order=4, **physical)
    for prefix in ("", "parent_", "dispersion_"):
        np.testing.assert_allclose(report[prefix + "tensor"], np.asarray(atomic[prefix + "tensor"])
                                   * Hartree / Bohr**4, rtol=1e-10, atol=1e-7)
    assert report["unit_labels"]["tensor"] == "eV/angstrom^4"


_SCF_FAILURE_DIRECTIONS = (
    (.1268335815567123, .12937025318784653, .13190692481898078,
     .13444359645011503, .1369802680812493, .13951693971238355,
     .14205361134351777, .144590282974652, .14712695460578623,
     .1496636262369205, .15220029786805472, .8861023877296237),
    (.1451122705744256, .1480145159859141, .15091676139740265,
     .15381900680889116, .15672125222037966, .15962349763186814,
     .16252574304335665, .1654279884548452, .16833023386633367,
     .1712324792778222, .4503779748115661, .7391074198393475),
)


@pytest.mark.parametrize("reference", ["roks", "gks"])
@pytest.mark.parametrize("direction_id", [0, 1])
def test_retained_displaced_scf_failure_on_all_public_gradient_routes(reference, direction_id, tmp_path):
    numbers, positions, options = fixture(reference)
    # Frozen generic direction of the retained 500-iteration complex SCF
    # failure; independent complete derivative fits remain in p264.
    direction = np.asarray(_SCF_FAILURE_DIRECTIONS[direction_id]).reshape(4, 3)
    positions = positions - .03 * direction
    options.update(conv_tol=1e-14, conv_tol_grad=1e-14, conv_tol_dp=1e-14,
                   response_tolerance=1e-12)
    reports = [
        dft_rs.gradient(numbers, positions, **options),
        dft_rs.run_feature("gradient", numbers, positions, **options),
        DftRs(units="atomic").get_gradient_report(
            Atoms(numbers=numbers, positions=positions * Bohr), **options),
        cli("gradient", numbers, positions, options, tmp_path),
    ]
    for route, report in enumerate(reports):
        (tmp_path / f"scf-route-{route}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        assert report["converged"], report
        assert report["scf_iterations"] <= 500
        np.testing.assert_allclose(report["energy"], reports[0]["energy"], rtol=0., atol=1e-11)
        np.testing.assert_allclose(report["gradient"], reports[0]["gradient"],
                                   rtol=1e-10, atol=1e-9)
        if reference == "gks":
            assert report["orbital_residual"] < 1e-14
