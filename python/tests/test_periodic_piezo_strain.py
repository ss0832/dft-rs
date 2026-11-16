# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Shared tensor conversion routes; not method-specific physical validation."""
import json
import os
import subprocess

import numpy as np
import pytest

import dft_rs
from dft_rs.units import BOHR_IN_METER, HARTREE_IN_JOULE


def inputs():
    return dict(
        piezoelectric_stress=[[0.11 + 0.04*i + 0.017*j for j in range(6)] for i in range(3)],
        piezoelectric_compliance=[[0.6 + 0.07*i if i == j else 0.025
                                  for j in range(6)] for i in range(6)],
        ion_response="clamped",
    )


@pytest.mark.parametrize("ions", ["clamped", "relaxed"])
def test_all_tensor_entries_units_and_shared_public_routes(ions):
    values = inputs()
    values["ion_response"] = ions
    expected = np.asarray(values["piezoelectric_stress"]) @ np.asarray(values["piezoelectric_compliance"])
    native = dft_rs.piezoelectric_strain(**values)
    ase = dft_rs.DftRs.get_piezoelectric_strain(**values)
    arguments = [os.environ["DFT_RS_CLI"], "piezoelectric-strain", "--format", "json"]
    for key, value in values.items():
        arguments.extend(["--"+key.replace("_", "-"),
                          json.dumps(value) if isinstance(value, list) else value])
    process = subprocess.run(arguments, capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    cli = json.loads(process.stdout)
    factor = 1.602176634e-19 * BOHR_IN_METER / HARTREE_IN_JOULE
    for report in (native, ase, cli):
        np.testing.assert_allclose(report["strain_coefficients"], expected, rtol=2e-15, atol=0)
        np.testing.assert_allclose(report["strain_coefficients_si"], expected*factor, rtol=2e-15, atol=0)
        np.testing.assert_allclose(report["strain_coefficients_pm_v"], expected*factor*1e12, rtol=2e-15, atol=0)
        assert report["units"]["strain_coefficients"] == "e*bohr/hartree"
        assert report["engineering_shear"]
        assert not report["mechanical_stability_assessed"]
        assert report["ion_response"] == ions
        assert report["voigt_order"] == ["xx", "yy", "zz", "yz", "xz", "xy"]


@pytest.mark.parametrize("case", ["shape", "nonfinite", "asymmetric", "ions"])
def test_invalid_supplied_tensors_raise_value_error(case):
    values = inputs()
    if case == "shape":
        values["piezoelectric_stress"].pop()
    elif case == "nonfinite":
        values["piezoelectric_compliance"][2][2] = float("nan")
    elif case == "asymmetric":
        values["piezoelectric_compliance"][0][1] += 0.1
    else:
        values["ion_response"] = "mixed"
    with pytest.raises(ValueError):
        dft_rs.piezoelectric_strain(**values)


@pytest.mark.parametrize("reference,feature", [
    (reference, feature) for reference in ("roks", "gks")
    for feature in ("periodic-energy", "stress", "phonons", "elastic",
                    "born-charges", "piezoelectric-tensor")
    if (reference, feature) not in {
        ("roks", "periodic-energy"), ("roks", "stress"),
        ("gks", "periodic-energy"), ("gks", "stress"),
    }
])
def test_unimplemented_reference_property_does_not_fall_back(feature, reference):
    # The connected routes have physical acceptance in test_periodic_roks and
    # test_periodic_gks_stress.
    with pytest.raises(NotImplementedError):
        dft_rs.native.run_feature(feature, [1, 1], [[0., 0., 0.], [1.4, .1, .2]],
                                 cell=[[8., 0., 0.], [.2, 8., 0.], [.1, .3, 8.]],
                                 reference=reference, xc="hf", basis="sto-3g",
                                 dispersion="none", cpu_threads=1)


def spinor_inputs():
    return dict(xc="hf", basis="sto-3g", dispersion="none", k_mesh=[2, 1, 1],
                guess="core", field_wave_indices=[0., 0., 0.],
                spin_field_real=[.0006, -.0008, 0.], nuclear_soc_scale=0., soc_scale=0.,
                overlap_cutoff=6., grid_cutoff=8., grid_level=1, grid_weight_cutoff=0.,
                max_iter=200, conv_tol=1e-12, conv_tol_grad=1e-10,
                conv_tol_dp=1e-11, cpu_threads=1, memory_mb=512)


@pytest.mark.parametrize("reference", ["gks", "ghf"])
def test_existing_generalized_bloch_reference_exposes_mesh_bands(reference):
    # Actual 3D two-k-point spinor SCF, with imaginary off-diagonal density.
    # This is not arbitrary-path interpolation, a DOS or a nuclear response test.
    report = dft_rs.native.run_feature(
        "periodic-spinor-scf", [1], [[.17, -.11, .23]],
        cell=[[8.1, 0., 0.], [.31, 8.7, 0.], [.19, -.27, 9.2]],
        reference=reference, **spinor_inputs())
    assert report["converged"] and report["backend_converged"]
    assert report["reference_type"] == reference
    assert report["orbital_residual"] <= 1e-10
    assert report["occupation_consistency"] <= 1e-11
    assert abs(report["particle_number_error"]) <= 1e-11
    assert np.max(np.abs(report["density_imag"])) > 1e-4
    energies = np.asarray(report["band_energies"])
    assert energies.shape == (2, 2) and np.isfinite(energies).all()
    np.testing.assert_array_equal(report["occupations"], [[1., 0.], [1., 0.]])
    edges = report["band_edges"]
    assert edges["valence_maximum"] == np.max(energies[:, 0])
    assert edges["conduction_minimum"] == np.min(energies[:, 1])
    assert edges["gap"] == np.min(energies[:, 1])-np.max(energies[:, 0])
    assert edges["direct_gap"] == np.min(energies[:, 1]-energies[:, 0])
    assert report["units"]["band_energies"].startswith("hartree")


@pytest.mark.parametrize("reference", ["roks", "uks", "rks"])
def test_dedicated_spinor_entry_refuses_a_different_reference_sector(reference):
    with pytest.raises(NotImplementedError):
        dft_rs.native.run_feature("periodic-spinor-scf", [1], [[0., 0., 0.]],
                                 cell=np.eye(3).tolist(), reference=reference,
                                 **spinor_inputs())


@pytest.mark.parametrize("feature", ["periodic-spinor-scf", "periodic-field-scf"])
@pytest.mark.parametrize("recipe", [dict(xc="b2plyp"), dict(pt2_same_spin=0.37)])
def test_field_reference_does_not_silently_extract_a_correlated_parent(feature, recipe):
    parameters = spinor_inputs()
    parameters.update(recipe)
    if feature == "periodic-field-scf":
        for key in ("spin_field_real", "nuclear_soc_scale", "soc_scale"):
            parameters.pop(key)
    with pytest.raises(
        NotImplementedError, match=rf"correlated periodic {feature} is not implemented"
    ):
        dft_rs.native.run_feature(feature, [1], [[0., 0., 0.]],
                                 cell=np.eye(3).tolist(), **parameters)
