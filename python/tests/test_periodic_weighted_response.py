# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Weighted XC dispatch preserves the periodic second-order reference."""

import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs.units import (
    BOHR_IN_ANGSTROM as Bohr,
    HARTREE_IN_EV as Hartree,
    HARTREE_PER_BOHR3_IN_GPA,
)
from test_periodic_gks_stress import cli_command


@pytest.mark.parametrize("feature", ["elastic", "phonons"])
def test_weighted_pbe0_on_charged_periodic_response_routes(feature, tmp_path):
    numbers = [2, 1]
    positions = np.array(
        [
            [0.13, -0.21, 0.07],
            [1.3661339888114017, 0.26263946631024193, -0.19661713484167487],
        ]
    )
    cell = np.array([[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]])
    atoms = Atoms(numbers, positions=positions * Bohr, cell=cell * Bohr, pbc=True)
    basis = tmp_path / "compact.g94"
    basis.write_text(
        "H 0\nS 1 1.0\n0.85 1.0\n****\nHe 0\nS 1 1.0\n1.1 1.0\n****\n", encoding="ascii"
    )
    shared = dict(
        reference="rks",
        basis=str(basis),
        charge=1,
        k_mesh=[2, 1, 1],
        overlap_cutoff=4.5,
        exchange_cutoff=4.5,
        xc_ao_cutoff=12.0,
        guess="core",
        grid_level=0,
        grid_shell=1,
        grid_weight_cutoff=0.0,
        cpu_threads=1,
        max_iter=200,
        conv_tol=1e-13,
        conv_tol_dp=1e-11,
        conv_tol_grad=1e-11,
        response_tolerance=1e-11,
    )
    controls = dict(
        shared,
        xc_terms=[
            {"xc": "hf", "weight": 0.25},
            {"xc": "pbe", "weight": 1.0},
            {"x": "pbe", "c": "none", "weight": -0.25},
        ],
    )
    calculate = getattr(dft_rs, feature)
    oracle = calculate(numbers, positions, cell, xc="pbe0", **shared)
    native = calculate(numbers, positions, cell, **controls)
    calculator = dft_rs.DftRs()
    getter = getattr(calculator, "get_" + feature)
    atomic = getter(atoms, units="atomic", **controls)
    default_units = dict(controls)
    for key in ("overlap_cutoff", "exchange_cutoff", "xc_ao_cutoff"):
        default_units[key] *= Bohr
    for key in ("conv_tol", "conv_tol_grad"):
        default_units[key] *= Hartree
    physical = getter(atoms, units="ase", **default_units)
    process = subprocess.run(
        cli_command(
            os.environ["DFT_RS_CLI"],
            tmp_path,
            numbers,
            positions,
            cell,
            controls,
            feature=feature,
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "energy" in process.stderr.lower() and "residual" in process.stderr.lower()
    cli = json.loads(process.stdout)
    result_scales = {
        "energy": Hartree,
        "orbital_residual": Hartree,
        "force_constants": Hartree / Bohr**2,
    }
    solve_fields = ["response_solves"]
    if feature == "elastic":
        result_scales.update(
            strain_hessian=Hartree,
            internal_strain=Hartree / Bohr,
            clamped_ion_gpa=1.0,
            relaxed_ion_gpa=1.0,
        )
        solve_fields = ["strain_solves", "displacement_solves"]
        assert np.shape(native["strain_hessian"]) == (9, 9)
        assert np.shape(native["internal_strain"]) == (9, 6)
        assert native["smallest_optical_eigenvalue"] > 0.0
    else:
        result_scales["raw_force_constants"] = Hartree / Bohr**2
        np.testing.assert_allclose(
            native["frequencies_cm1"], oracle["frequencies_cm1"], atol=0.001, rtol=1e-10
        )
    assert np.shape(native["force_constants"]) == (6, 6)
    assert native["raw_asymmetry"] < 1e-8
    assert native["acoustic_residual"] < 1e-8

    maximum = 0.0
    maximum_gpa = 0.0
    maximum_response_residual = 0.0
    maximum_orbital_residual = 0.0
    for report in (oracle, atomic, cli, native):
        assert report["converged"] and report["reference_type"] == "rks"
        assert 0.0 <= report["orbital_residual"] < shared["conv_tol_grad"]
        maximum_orbital_residual = max(
            maximum_orbital_residual, report["orbital_residual"]
        )
        for key in solve_fields:
            response_residual = max(s["relative_residual"] for s in report[key])
            assert response_residual <= 1e-11
            maximum_response_residual = max(
                maximum_response_residual, response_residual
            )
        for key in result_scales:
            expected, actual = np.asarray(native[key]), np.asarray(report[key])
            # Apply the absolute route gate in atomic units. The driver
            # returns Voigt elastic constants directly in GPa in every API.
            if key.endswith("_gpa"):
                maximum_gpa = max(maximum_gpa, np.max(np.abs(expected - actual)))
                native_key = key.removesuffix("_gpa")
                expected, atomic_tensor = (
                    np.asarray(native[native_key]),
                    np.asarray(report[native_key]),
                )
                np.testing.assert_allclose(
                    actual,
                    atomic_tensor * HARTREE_PER_BOHR3_IN_GPA,
                    atol=1e-12,
                    rtol=0.0,
                )
                actual = atomic_tensor
            np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0.0)
            assert report["units"][key] == native["units"][key]
            maximum = max(maximum, np.max(np.abs(expected - actual)))
    for key, scale in result_scales.items():
        np.testing.assert_allclose(
            physical[key], np.asarray(native[key]) * scale, atol=1e-8, rtol=1e-10
        )
    print(
        f"weighted PBE0 {feature}: E={native['energy']:.15e}, "
        f"FCxx={native['force_constants'][0][0]:.15e}, "
        f"acoustic={native['acoustic_residual']:.3e}, max route difference={maximum:.3e} au, "
        f"max orbital residual={maximum_orbital_residual:.3e}, "
        f"max response residual={maximum_response_residual:.3e}"
    )
    if feature == "elastic":
        print(
            f"clamped C11={native['clamped_ion_gpa'][0][0]:.15e}, "
            f"relaxed C11={native['relaxed_ion_gpa'][0][0]:.15e}, "
            f"max route difference={maximum_gpa:.3e} GPa"
        )
