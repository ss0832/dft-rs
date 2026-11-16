# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Nonlinear isospectral response shares references, units and public routes."""
import math

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree

import dft_rs
from test_fixed_populations import ORACLES, common, cli_report


POSITIONS = [[0., 0., -0.8], [0.2, 0.1, 0.8]]
DIRECTION = np.asarray([0.1, -0.2, 0.3])


@pytest.mark.parametrize("xc,unrestricted", [
    ("hf", False), ("svwn5", True), ("pbe", False),
    ("tpss", True), ("pbe0", False), ("cam-b3lyp", True),
])
def test_fixed_population_taylor_and_mixed_share_all_routes(tmp_path, xc, unrestricted):
    controls = common(xc=xc, unrestricted=unrestricted, occupation_mode="fixed",
                      occupations_alpha=[0.8, 0.2],
                      occupations_beta=[0.7, 0.3] if unrestricted else [0.8, 0.2],
                      field=[0.001, -0.002, 0.003])
    atoms = Atoms("H2", positions=np.asarray(POSITIONS)*Bohr)
    native = dft_rs.electric_field_taylor([1, 1], POSITIONS, DIRECTION, order=4, **controls)
    atomic = dft_rs.DftRs(units="atomic", **controls).get_electric_field_taylor(
        atoms, direction=DIRECTION, order=4)
    physical_controls = dict(controls, field=np.asarray(controls["field"])*Hartree/Bohr,
                             conv_tol=controls["conv_tol"]*Hartree)
    physical = dft_rs.DftRs(**physical_controls).get_electric_field_taylor(
        atoms, direction=DIRECTION*Hartree/Bohr, order=4)
    cli = cli_report(tmp_path, "field-taylor", atoms, dict(
        controls, field_direction=DIRECTION.tolist(), response_order=4))
    for report in [atomic, physical, cli]:
        assert report["converged"] and not report["projector_constraint"]
        assert "isospectral" in report["model"]
        assert report["reference"]["projector_defect"] > 0.1
        assert max(report["residuals"]) < 1e-10
        assert max(report["fixed_population"]["population_defects"]) < 1e-10
    for report in [atomic, cli]:
        np.testing.assert_allclose(report["dipole_coefficients"], native["dipole_coefficients"], atol=1e-9, rtol=0.)
    np.testing.assert_allclose(np.asarray(physical["dipole_coefficients"])/Bohr,
                               native["dipole_coefficients"], atol=1e-9, rtol=0.)
    np.testing.assert_allclose(np.asarray(physical["direction"])/(Hartree/Bohr), DIRECTION)
    mixed = dft_rs.mixed_electric_field_response([1, 1], POSITIONS, [DIRECTION]*3, **controls)
    adapted = dft_rs.DftRs(units="atomic", **controls).get_mixed_electric_field_response(
        atoms, directions=[DIRECTION]*3)
    cli_mixed = cli_report(tmp_path, "field-mixed", atoms, dict(
        controls, field_directions=[DIRECTION.tolist()]*3))
    expected = 6*np.asarray(native["dipole_coefficients"][3])
    for report in [mixed, adapted, cli_mixed]:
        assert report["converged"]
        np.testing.assert_allclose(report["dipole_derivative"], expected, atol=1e-9, rtol=0.)
    print("fixed nonlinear public", xc, unrestricted, "map", native["fixed_occupation_residual"],
          "equation", max(native["residuals"]),
          "population", max(native["fixed_population"]["population_defects"]),
          "metric", max(native["fixed_population"]["orthogonality_defects"]),
          "relative_stationarity", max(native["fixed_population"]["relative_stationarity_defects"]))


@pytest.mark.parametrize("order", [2, 3])
def test_fixed_population_full_tensors_preserve_units_and_component_text(tmp_path, order):
    controls = common(xc="hf", unrestricted=True, occupation_mode="fixed",
                      occupations_alpha=[0.8, 0.2], occupations_beta=[0.7, 0.3], field=[0., 0., .02])
    atoms = Atoms("H2", positions=np.asarray(POSITIONS)*Bohr)
    feature = "beta" if order == 2 else "gamma"
    native = getattr(dft_rs, feature)([1, 1], POSITIONS, **controls)
    atomic = getattr(dft_rs.DftRs(units="atomic", **controls), "get_"+feature)(atoms)
    physical = getattr(dft_rs.DftRs(), "get_"+feature)(atoms, **dict(
        controls, field=np.asarray(controls["field"])*Hartree/Bohr, conv_tol=controls["conv_tol"]*Hartree))
    cli = cli_report(tmp_path, feature, atoms, controls)
    plain = cli_report(tmp_path, feature, atoms, controls, plain=True)
    rows = [float(row.split()[-1]) for row in plain.splitlines() if row.startswith(feature+" ")]
    factor = Bohr**(order+1)/Hartree**order
    for report in [atomic, cli]:
        np.testing.assert_allclose(report["tensor"], native["tensor"], atol=1e-9, rtol=0.)
        assert report["converged"] and not report["projector_constraint"]
        assert report["fixed_population_defect"] < 1e-10
    assert atomic["units"] == native["units"]
    np.testing.assert_allclose(rows, native["tensor"], atol=1e-9, rtol=0.)
    np.testing.assert_allclose(np.asarray(physical["tensor"])/factor, native["tensor"], atol=1e-9, rtol=0.)
    assert "raw components preserved" in plain and "converged true" in plain
    if order == 2:
        other = dft_rs.beta([1, 1], POSITIONS, assembly="variational", **controls)
        np.testing.assert_allclose(other["tensor"], native["tensor"], atol=1e-9, rtol=0.)
        assert len(other["residuals"]) == 3
    series = dft_rs.electric_field_taylor([1, 1], POSITIONS, [0., 0., 1.], order=order, **controls)
    assert native["tensor"][-1] == pytest.approx(math.factorial(order)*series["dipole_coefficients"][order][2], abs=1e-9)


def test_canonical_response_options_and_invalid_aliases_are_explicit():
    controls = common(xc="hf", occupation_mode="fixed", occupations_alpha=[.8, .2], occupations_beta=[.8, .2])
    a = dft_rs.electric_field_taylor([1, 1], POSITIONS, [0., 0., 1.], order=2,
                                    response_tolerance=1e-12, response_max_iterations=500, **controls)
    b = dft_rs.electric_field_taylor([1, 1], POSITIONS, [0., 0., 1.], order=2,
                                    response_tol=1e-12, response_max_iter=500, **controls)
    np.testing.assert_array_equal(a["dipole_coefficients"], b["dipole_coefficients"])
    atoms = Atoms("H2", positions=np.asarray(POSITIONS)*Bohr)
    calc = dft_rs.DftRs(units="atomic", **controls)
    adapted = calc.get_electric_field_taylor(atoms, direction=[0., 0., 1.], order=2,
                                            response_tolerance=1e-12)
    np.testing.assert_array_equal(adapted["dipole_coefficients"], a["dipole_coefficients"])
    with pytest.raises(ValueError, match="either"):
        calc.get_electric_field_taylor(atoms, direction=[0., 0., 1.], order=2,
                                      response_tol=1e-11, response_tolerance=1e-12)
    for extra, message in [
        (dict(response_tolerance=1e-12, response_tol=1e-11), "either"),
        (dict(field=[0., 0., .01], reference_field=[0., 0., .01]), "either"),
        (dict(electronic_temperature=1000.), "fixed"),
        (dict(unused_electric_control=True), "unused_electric_control"),
    ]:
        with pytest.raises(ValueError, match=message):
            dft_rs.electric_field_taylor([1, 1], POSITIONS, [0., 0., 1.], **dict(controls, **extra))


@pytest.mark.parametrize("case", ORACLES["records"], ids=lambda case: case["name"])
def test_five_pbe_molecules_match_independent_nonlinear_oracle(case):
    reference = case["nonlinear_response"]
    controls = common(xc="pbe", occupation_mode="fixed", occupations_alpha=case["populations"],
                      occupations_beta=case["populations"], reference_field=reference["reference_field"])
    controls.update(conv_tol_grad=1e-11, conv_tol_dp=1e-12)
    scale = .01
    report = dft_rs.electric_field_taylor(case["numbers"], case["positions"], [0., 0., scale],
                                         order=3, response_tolerance=1e-12, **controls)
    assert report["converged"] and not report["projector_constraint"]
    assert max(report["residuals"]) < 1e-12
    assert max(report["fixed_population"]["population_defects"]) < 1e-12
    for order, observable in [(2, "beta"), (3, "gamma")]:
        analytic = math.factorial(order)*report["dipole_coefficients"][order][2]/scale**order
        estimates = np.asarray(reference[observable+"_estimates"])
        errors = np.abs(estimates-analytic)
        steps = np.asarray(reference["steps"])
        assert len(set(steps)) >= 4 and np.all(np.isfinite(errors)) and np.all(errors > 0.)
        x, y = np.log(steps), np.log(errors)
        dx, dy = x-x.mean(), y-y.mean()
        exponent = np.dot(dx, dy)/np.dot(dx, dx)
        intercept = y.mean()-exponent*x.mean()
        residual = y-intercept-exponent*x
        r_squared = 1-np.dot(residual, residual)/np.dot(dy, dy)
        richardson = abs((4*estimates[-1]-estimates[-2])/3-analytic)
        print("fixed five nonlinear", case["name"], observable, "analytic", analytic,
              "steps", steps.tolist(), "errors", errors.tolist(), "p", exponent,
              "R2", r_squared, "C", math.exp(intercept),
              "log_rmse", float(np.sqrt(np.mean(residual**2))), "Richardson", richardson,
              "native_response_residuals", report["residuals"])
        assert abs(exponent-2) < .025 and r_squared > .9999
        assert richardson < 2e-7+1e-8*abs(analytic)
    for sample in reference["samples"]:
        assert max(sample["response_residuals"]) < 1e-10
        assert np.max(np.abs(sample["number_defects"])) < 1e-10
