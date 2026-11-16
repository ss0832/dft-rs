# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Constrained static tensors on native, ASE and canonical CLI surfaces.

Independent complete component fits are in p310; route comparisons here
certify neither another functional class nor constrained PT2 derivatives.
"""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs import DftRs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree

POSITIONS = np.array([[0.13, -0.21, 0.19], [1.57, 0.16, -0.08],
                      [-0.19, 1.93, 0.37], [0.42, 0.29, 2.28]])
DIRECTION = np.array([[0.03, -0.02, 0.01], [-0.02, 0.04, -0.01],
                      [0.01, -0.03, 0.02], [0.02, 0.01, -0.04]])


def test_calculator_preserves_reference_and_response_controls(monkeypatch):
    requests = []

    def capture(feature, numbers, positions, **options):
        requests.append(options)
        raise LookupError("request captured before the numerical calculation")

    monkeypatch.setattr(dft_rs.native, "run_feature", capture)
    atoms = Atoms("H4", positions=POSITIONS * Bohr)
    calc = DftRs(units="atomic", guess="gwh", response_tolerance=1e-12,
                 response_max_iterations=731)
    for call, tolerance in [
        (lambda: calc.get_gamma(atoms), 1e-12),
        (lambda: calc.get_gamma(atoms, response_tol=3e-12), 3e-12),
        (lambda: calc.get_gamma(atoms, response_tolerance=2e-12), 2e-12),
        (lambda: calc.get_static_spectroscopic_derivative(
            atoms, nuclear_direction=DIRECTION.tolist()), 1e-12),
    ]:
        with pytest.raises(LookupError, match="request captured"):
            call()
        assert requests[-1]["guess"] == "gwh"
        assert requests[-1]["response_tolerance"] == tolerance
        assert requests[-1]["response_max_iterations"] == 731


def test_default_guess_preserves_molecular_and_periodic_driver_defaults(monkeypatch):
    requests = []

    def capture(*args, **options):
        requests.append(options)
        raise LookupError("request captured before the numerical calculation")

    monkeypatch.setattr(dft_rs.native, "run_feature", capture)
    monkeypatch.setattr(dft_rs.native, "periodic_scf", capture)
    monkeypatch.setattr(dft_rs.native, "dimensional_scf", capture)
    atoms = Atoms("He", positions=[[0.1, 0.2, 0.3]])
    calc = DftRs(units="atomic")
    with pytest.raises(LookupError):
        calc.get_feature_report("energy", atoms)
    assert "guess" not in requests[-1]
    atoms.set_cell(np.eye(3) * 6 * Bohr)
    atoms.pbc = True
    with pytest.raises(LookupError):
        calc._run_periodic(atoms, forces=False)
    assert requests[-1]["guess"] == "core"
    for method in ("get_dimensional_scf", "get_dimensional_field_response"):
        if method.endswith("field_response"):
            monkeypatch.setattr(dft_rs.native, "dimensional_field_response", capture)
        with pytest.raises(LookupError):
            getattr(calc, method)(atoms)
        assert requests[-1]["guess"] == "core"


def controls(reference):
    options = dict(reference=reference, xc="hf", basis="sto-3g", charge=1,
                   multiplicity=2, grid_level=0, grid_weight_cutoff=0.0,
                   conv_tol=1e-14, conv_tol_grad=1e-12, conv_tol_dp=1e-13,
                   response_tolerance=1e-12, max_iter=600, guess="core",
                   dispersion="none", reference_field=[0.003, -0.004, 0.005])
    if reference == "gks":
        # The same fixed Hermitian test source as support/spinor_operator.rs.
        re, im = np.zeros((8, 8)), np.zeros((8, 8))
        for i in range(4):
            re[i, i] = -0.013 * (i + 1)
            re[4+i, 4+i] = -re[i, i]
            re[i, 4+i] = re[4+i, i] = 0.007 * (i + 1)
            im[i, 4+i] = 0.009 * (4 - i)
            im[4+i, i] = -im[i, 4+i]
        options.update(gks_operator_real=re.tolist(),
                       gks_operator_imaginary=im.tolist(), spinor_xc_tolerance=1e-12)
    return options


def validate(report, rank):
    assert report["converged"] and report["provenance"] == "analytic"
    assert report["shape"] == [3] * rank
    assert len(report["tensor"]) == 3**rank
    for key in ("response_residuals", "projector_defects", "nuclear_stationarity_defects",
                "reference_defects"):
        values = report[key]
        assert all(np.isfinite(values)) and max(map(abs, values), default=0) <= 1e-12
    print("constrained electric route", report["reference_type"], rank,
          {key: max(map(abs, report[key]), default=0) for key in
           ("response_residuals", "projector_defects", "nuclear_stationarity_defects",
            "reference_defects")})


@pytest.mark.parametrize("reference", ["roks", "gks"])
@pytest.mark.parametrize("observable", ["raman", "hyper-raman", "beta", "gamma"])
def test_all_ordered_constrained_tensors_on_native_ase_and_cli(reference, observable, tmp_path):
    options = controls(reference)
    # Compare identical binary inputs after each documented unit boundary.
    # A one-ulp coordinate difference is amplified by a soft open-shell
    # response and cannot justify a weaker tensor-comparison threshold.
    angstrom_positions = POSITIONS * Bohr
    positions = angstrom_positions / Bohr
    rank = {"raman": 2, "hyper-raman": 3, "beta": 3, "gamma": 4}[observable]
    feature = observable if observable in {"beta", "gamma"} else "static-spectroscopy"
    extra = {} if feature in {"beta", "gamma"} else dict(
        nuclear_direction=DIRECTION.tolist(), electric_order=rank)
    common = dft_rs.run_feature(feature, [1]*4, positions.tolist(), **options, **extra)
    validate(common, rank)
    direct = (getattr(dft_rs, feature)([1]*4, positions.tolist(), **options) if feature in {"beta", "gamma"}
              else dft_rs.static_spectroscopic_derivative(
                  [1]*4, positions.tolist(), DIRECTION.tolist(), electric_order=rank, **options))
    atoms = Atoms("H4", positions=angstrom_positions)
    calc = DftRs(units="atomic", **options)
    atomic = (getattr(calc, "get_" + feature)(atoms) if feature in {"beta", "gamma"} else
              calc.get_static_spectroscopic_derivative(atoms, **extra))
    for adapted in (direct, atomic):
        np.testing.assert_array_equal(adapted["tensor"], common["tensor"])
        assert adapted["unit_labels"] == common["units"]
    physical_options = dict(options)
    for key in ("conv_tol", "conv_tol_grad", "spinor_xc_tolerance"):
        if key in physical_options:
            physical_options[key] *= Hartree
    physical_options["reference_field"] = np.asarray(options["reference_field"]) * Hartree / Bohr
    for key in ("gks_operator_real", "gks_operator_imaginary"):
        if key in physical_options:
            physical_options[key] = (np.asarray(options[key]) * Hartree).tolist()
    physical_extra = dict(extra)
    if "nuclear_direction" in extra:
        physical_extra["nuclear_direction"] = (DIRECTION * Bohr).tolist()
    physical = DftRs(**physical_options).get_feature_report(feature, atoms, **physical_extra)
    factor = Bohr**rank / Hartree**(rank - 1)
    np.testing.assert_allclose(physical["tensor"], np.asarray(common["tensor"]) * factor,
                               rtol=1e-8, atol=1e-9)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for canonical route acceptance"
    xyz = tmp_path / "constrained_electric.xyz"
    xyz.write_text("4\nC1 constrained electric tensor\n" + "".join(
        "H " + " ".join(f"{v:.17g}" for v in row) + "\n" for row in angstrom_positions),
        encoding="utf-8")
    args = [executable, feature, str(xyz), "--format", "json"]
    for key, value in dict(options, **extra).items():
        args += ["--" + key.replace("_", "-"),
                 json.dumps(value) if isinstance(value, list) else str(value).lower()]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    validate(json.loads(result.stdout), rank)
    assert "SCF" in result.stderr or "[spinor-scf]" in result.stderr
    # XYZ uses the package reciprocal Angstrom factor; ASE divides by Bohr.
    # Check CLI against a separate native calculation at precisely its input.
    cli_positions = angstrom_positions * (1.0 / Bohr)
    assert np.max(np.abs(cli_positions - positions)) <= np.spacing(np.max(np.abs(positions)))
    cli_native = dft_rs.run_feature(feature, [1]*4, cli_positions.tolist(), **options, **extra)
    np.testing.assert_allclose(json.loads(result.stdout)["tensor"], cli_native["tensor"],
                               rtol=0, atol=1e-9)
