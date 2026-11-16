# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Common-surface and unit acceptance for the fixed continuous-charge chart."""

import json
import os
import re
import subprocess

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree


NUMBERS = [7, 7]
POSITIONS = [[0.0, 0.0, -1.05], [0.0, 0.0, 1.05]]
CONTROLS = dict(
    fixed_electron_count=14,
    physical_nuclear_charges=[7.0, 7.0],
    charge_directions=[[-1.0, 1.0]],
    charge_coordinates=[0.25],
    charge_bounds=[[-1.5, 1.5]],
    charge_origin=[0.0, 0.0, 0.0],
    conserve_total_charge=True,
)
METHOD = dict(
    xc="pbe", basis="sto-3g", dispersion="none", grid_level=1,
    grid_weight_cutoff=0, conv_tol=1e-13, conv_tol_dp=1e-12,
    conv_tol_grad=1e-11, guess="core", max_iter=300,
)


def controls(**overrides):
    return dict(CONTROLS, **overrides)


@pytest.mark.parametrize("spin_counts", [None, [7, 6]])
def test_native_ase_cli_and_units_share_the_same_charge_model(tmp_path, spin_counts):
    request = dict(CONTROLS, charge_nuclear_gradient=True)
    if spin_counts is not None:
        request.update(fixed_electron_count=sum(spin_counts), fixed_spin_electron_counts=spin_counts)
    native = dft_rs.continuous_species(NUMBERS, POSITIONS, **METHOD, **request)
    assert native["feature"] == "continuous-species"
    assert native["derivative_method"] == "analytic"
    assert native["geometry_condition"] == "clamped_geometry"
    assert native["fixed_electron_count"] == request["fixed_electron_count"]
    assert native["physical_total_charge"] == 14 - request["fixed_electron_count"]
    assert native["reference"] == ("rks" if spin_counts is None else "uks")
    assert native["fixed_spin_electron_counts"] == ([7, 7] if spin_counts is None else spin_counts)
    assert native["physical_nuclear_charges"] == [6.75, 7.25]
    assert native["polarizability_parameter_derivative_status"] == "unsupported"
    assert native["stability_status"] == "not_evaluated"
    assert native["nuclear_gradient_status"] == "analytic"
    assert max(native["response_residuals"]) <= 1e-10
    atoms = Atoms("N2", positions=np.asarray(POSITIONS) * Bohr)
    ase_method = dict(METHOD, conv_tol=METHOD["conv_tol"] * Hartree,
                      conv_tol_grad=METHOD["conv_tol_grad"] * Hartree)
    calc = dft_rs.DftRs(**ase_method)
    atomic = calc.get_continuous_species(atoms, units="atomic", **request)
    adapted = calc.get_continuous_species(atoms, **request)
    for key in ("energy", "dipole", "polarizability", "parameter_energy_gradient",
                "parameter_energy_hessian", "parameter_dipole_jacobian", "gradient", "forces"):
        np.testing.assert_allclose(atomic[key], native[key], atol=1e-9, rtol=1e-8)
    for key in ("energy", "parameter_energy_gradient", "parameter_energy_hessian"):
        np.testing.assert_allclose(adapted[key], np.asarray(native[key]) * Hartree,
                                   atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(adapted["dipole"], np.asarray(native["dipole"]) * Bohr,
                               atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(adapted["parameter_dipole_jacobian"],
                               np.asarray(native["parameter_dipole_jacobian"]) * Bohr,
                               atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(adapted["polarizability"],
                               np.asarray(native["polarizability"]) * Bohr**3,
                               atol=1e-8, rtol=1e-8)
    for key in ("gradient", "forces"):
        np.testing.assert_allclose(adapted[key], np.asarray(native[key]) * Hartree / Bohr,
                                   atol=1e-8, rtol=1e-8)
    path = tmp_path / "fixed_nitrogen_template.xyz"
    rows = "".join(f"N {x*Bohr:.17g} {y*Bohr:.17g} {z*Bohr:.17g}\n"
                   for x, y, z in POSITIONS)
    path.write_text("2\nFixed nitrogen AO templates\n" + rows, encoding="utf-8")
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for route acceptance"
    arguments = [executable, "continuous-species", str(path), "--format", "json"]
    for key, value in dict(METHOD, **request).items():
        arguments.extend(["--" + key.replace("_", "-"), json.dumps(value)
                          if isinstance(value, (list, bool)) else str(value)])
    process = subprocess.run(arguments, capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    assert "[scf]" in process.stderr
    report = json.loads(process.stdout)
    iteration_energies = re.findall(r"\[scf\] converged.*\(E = ([+\-\d.eE]+)\)", process.stderr)
    assert iteration_energies, process.stderr
    assert float(iteration_energies[-1]) == pytest.approx(report["energy"], abs=1e-10)
    assert report["classical_energy"] + report["electronic_energy"] == pytest.approx(
        report["energy"], abs=1e-12)
    for key in ("energy", "dipole", "polarizability", "parameter_energy_gradient",
                "parameter_energy_hessian", "parameter_dipole_jacobian", "gradient", "forces"):
        np.testing.assert_allclose(report[key], native[key], atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("override", [
    {"fixed_electron_count": 13},
    {"fixed_electron_count": 14.5},
    {"fixed_spin_electron_counts": [7, 6]},
    {"fixed_spin_electron_counts": [6, 8]},
    {"fixed_spin_electron_counts": [7.5, 6.5]},
    {"fixed_spin_electron_counts": [7]},
    {"charge_coordinates": [float("nan")]},
    {"charge_coordinates": [2.0]},
    {"charge_directions": [[-1.0, 2.0]]},
    {"charge_bounds": [[0.5, -0.5]]},
    {"charge_scales": [0.0]},
])
def test_invalid_chart_is_rejected(override):
    with pytest.raises(ValueError):
        dft_rs.continuous_species(NUMBERS, POSITIONS, **METHOD, **controls(**override))


@pytest.mark.parametrize("number", [6.75, "7", 0, 119])
def test_real_physical_charge_does_not_relax_atomic_number_validation(number):
    with pytest.raises(ValueError, match="atomic numbers"):
        dft_rs.continuous_species([number, 7], POSITIONS, **METHOD, **CONTROLS)


@pytest.mark.parametrize("method", [
    {"xc": "hf"}, {"xc": "pbe0"}, {"unrestricted": True},
    {"dispersion": "d3bj"}, {"solvent": "cpcm:water"},
    {"electronic_temperature": 1000.0},
])
def test_unverified_model_combinations_are_rejected(method):
    settings = dict(METHOD, **method)
    with pytest.raises((NotImplementedError, ValueError)):
        dft_rs.continuous_species(NUMBERS, POSITIONS, **settings, **CONTROLS)


def test_nonconserving_chart_reports_real_charge_and_fixed_electrons():
    native = dft_rs.continuous_species(NUMBERS, POSITIONS, **METHOD, **controls(
        charge_directions=[[0.25, 0.5]], conserve_total_charge=False,
        charge_coordinates=[0.2],
    ))
    assert native["fixed_electron_count"] == 14
    assert native["physical_total_charge"] == pytest.approx(0.15)
    assert native["physical_nuclear_charges"] == pytest.approx([7.05, 7.1])
