# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""One spectral driver supplies prescribed-population second derivatives."""
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree

import dft_rs
from test_fixed_populations import ORACLES, common, cli_report


@pytest.mark.parametrize("xc,unrestricted", [
    ("hf", False), ("svwn5", True), ("pbe", False),
    ("tpss", True), ("pbe0", False), ("cam-b3lyp", True),
])
def test_spectral_second_derivatives_share_native_ase_and_cli(tmp_path, xc, unrestricted):
    positions = [[0., 0., -0.8], [0.2, 0.1, 0.8]]
    controls = common(xc=xc, unrestricted=unrestricted, occupation_mode="fixed",
                      occupations_alpha=[0.8, 0.2],
                      occupations_beta=[0.7, 0.3] if unrestricted else [0.8, 0.2],
                      field=[0.001, -0.002, 0.003])
    atoms = Atoms("H2", positions=np.asarray(positions)*Bohr)
    for feature, native_function, getter, fields in [
        ("polarizability", dft_rs.polarizability_report, "get_polarizability_report",
         ["polarizability", "response_residuals", "number_defects"]),
        ("hessian", dft_rs.hessian_report, "get_hessian_report",
         ["hessian", "dipole_derivatives", "ensemble_response_residuals", "ensemble_number_defects"]),
    ]:
        native = native_function([1, 1], positions, **controls)
        atomic = getattr(dft_rs.DftRs(units="atomic"), getter)(atoms, **controls)
        physical = getattr(dft_rs.DftRs(), getter)(atoms, **dict(
            controls, field=np.asarray(controls["field"])*Hartree/Bohr))
        cli = cli_report(tmp_path, feature, atoms, controls)
        for key in fields:
            for report in [atomic, cli]:
                np.testing.assert_allclose(report[key], native[key], atol=1e-9, rtol=0.)
        if feature == "polarizability":
            np.testing.assert_allclose(np.asarray(physical["polarizability"])/(Bohr**2/Hartree),
                                       native["polarizability"], atol=1e-8, rtol=0.)
            np.testing.assert_allclose(np.asarray(physical["number_defects"])/(Bohr/Hartree),
                                       native["number_defects"], atol=1e-14, rtol=0.)
            assert "df = 0" in native["response_population_model"]
            assert max(native["response_residuals"]) < 1e-10
            assert np.max(np.abs(native["number_defects"])) < 1e-10
            assert native["raw_asymmetry"] < 1e-8
        else:
            np.testing.assert_allclose(np.asarray(physical["hessian"])/(Hartree/Bohr**2),
                                       native["hessian"], atol=1e-8, rtol=0.)
            np.testing.assert_allclose(physical["dipole_derivatives"],
                                       native["dipole_derivatives"], atol=1e-9, rtol=0.)
            assert max(native["ensemble_response_residuals"]) < 1e-10
            assert np.max(np.abs(native["ensemble_number_defects"])) < 1e-10
            assert native["raw_asymmetry"] < 1e-8
        print("fixed response public", feature, xc, unrestricted,
              "raw_asymmetry", native["raw_asymmetry"], "map", native["fixed_occupation_residual"],
              "equation", max(native["response_residuals"] if feature == "polarizability" else native["ensemble_response_residuals"]),
              "number", float(np.max(np.abs(native["number_defects"] if feature == "polarizability" else native["ensemble_number_defects"]))))


def test_population_constraints_do_not_acquire_thermal_or_integer_response():
    positions = [[0., 0., -0.7], [0., 0., 0.7]]
    controls = common(xc="pbe0", occupation_mode="fixed", occupations_alpha=[1.], occupations_beta=[1.])
    fixed = dft_rs.polarizability_report([1, 1], positions, **controls)
    ground = dft_rs.polarizability_report([1, 1], positions, **common(xc="pbe0"))
    np.testing.assert_allclose(fixed["polarizability"], ground["polarizability"], atol=1e-9, rtol=0.)
    fraction = dict(controls, occupations_alpha=[0.8, 0.2], occupations_beta=[0.8, 0.2])
    with pytest.raises(ValueError, match="response_min_denominator"):
        dft_rs.polarizability_report([1, 1], positions, **dict(fraction, response_min_denominator=10.))
    with pytest.raises(ValueError, match="thermal equilibrium"):
        dft_rs.gibbs_free_energy([1, 1], positions, **fraction)
    frequencies = dft_rs.frequencies_report([1, 1], positions, **fraction)
    assert frequencies["thermochemistry"] is None
    assert any("fixed nonthermal" in item for item in frequencies["warnings"])


@pytest.mark.parametrize("temperature", [1., 80000.])
def test_thermal_static_electric_report_shares_canonical_response(tmp_path, temperature):
    positions = [[0., 0., -0.8], [0.2, 0.1, 0.8]]
    controls = common(xc="pbe", electronic_temperature=temperature, field=[0.001, -0.002, 0.003])
    native = dft_rs.polarizability_report([1, 1], positions, **controls)
    atoms = Atoms("H2", positions=np.asarray(positions)*Bohr)
    adapted = dft_rs.DftRs(units="atomic").get_polarizability_report(atoms, **controls)
    cli = cli_report(tmp_path, "polarizability", atoms, controls)
    for report in [adapted, cli]:
        np.testing.assert_allclose(report["polarizability"], native["polarizability"], atol=1e-9, rtol=0.)
        assert "Fermi-Dirac" in report["response_population_model"]
        assert max(report["response_residuals"]) < 1e-10
        assert np.max(np.abs(report["number_defects"])) < 1e-10
    if temperature == 1.:
        ground = dft_rs.polarizability_report([1, 1], positions, **dict(controls, electronic_temperature=0.))
        np.testing.assert_allclose(native["polarizability"], ground["polarizability"], atol=1e-9, rtol=0.)


@pytest.mark.parametrize("case", ORACLES["records"], ids=lambda case: case["name"])
def test_five_molecule_pbe_polarizability_independent_oracle(case):
    controls = common(xc="pbe", occupation_mode="fixed", occupations_alpha=case["populations"],
                      occupations_beta=case["populations"])
    report = dft_rs.polarizability_report(case["numbers"], case["positions"], **controls)
    error = float(np.max(np.abs(np.asarray(report["polarizability"])-case["static_response"]["polarizability"])))
    assert error < 2e-7
    assert report["raw_asymmetry"] < 1e-8
    assert max(report["response_residuals"]) < 1e-10
    assert np.max(np.abs(report["number_defects"])) < 1e-10
    print("fixed five-molecule response", case["name"], "alpha", report["polarizability"],
          "oracle_error", error, "response_residuals", report["response_residuals"],
          "map", report["fixed_occupation_residual"], "asymmetry", report["raw_asymmetry"],
          "number_defect", float(np.max(np.abs(report["number_defects"]))))
