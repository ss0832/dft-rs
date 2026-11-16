# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Public selection and metadata for collisionless Fermi response."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs import DftRs, dynamic_polarizability, run_feature
from dft_rs.units import BOHR_IN_ANGSTROM

POSITIONS = [[0., 0., 0.], [0., 0., 3.]]
OPTIONS = dict(xc="hf", basis="sto-3g", conv_tol=1e-12,
               conv_tol_grad=1e-10, conv_tol_dp=1e-11,
               max_iter=300, incore_mb=64)
HOT = dict(electronic_temperature=40000., thermal_response="collisionless")


def test_thermal_choice_all_tensor_components_and_metadata_on_public_routes(tmp_path):
    native = dynamic_polarizability([1, 1], POSITIONS, omega=.03, damping=.01,
                                   **OPTIONS, **HOT)
    common = run_feature("dynamic-polarizability", [1, 1], POSITIONS,
                         frequency=.03, frequency_damping=.01, **OPTIONS, **HOT)
    atoms = Atoms("H2", positions=np.asarray(POSITIONS) * BOHR_IN_ANGSTROM)
    adapted = DftRs(units="atomic", **OPTIONS).get_dynamic_polarizability(
        atoms, omega=.03, damping=.01, **HOT)
    for report in (native, common, adapted):
        for key in ("alpha_real", "alpha_imag"):
            np.testing.assert_allclose(report[key], native[key], atol=1e-10, rtol=0)
        assert report["thermal_response"] == "collisionless"
        assert report["response_converged"]
        assert report["population_defect"] < OPTIONS["conv_tol_dp"]
        assert "physical thermal" in report["reference_projector_interpretation"]
        assert np.max(report["response_particle_number_defects"]) < 1e-10
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI required for public CLI acceptance"
    xyz = tmp_path / "thermal_h2.xyz"
    xyz.write_text("2\nH2 r=3Bohr\nH 0 0 0\nH 0 0 " +
                   repr(3 * BOHR_IN_ANGSTROM) + "\n")
    command = [executable, "dynamic-polarizability", str(xyz), "--format", "json",
               "--xc", "hf", "--basis", "sto-3g", "--frequency", ".03",
               "--frequency-damping", ".01", "--electronic-temperature", "40000",
               "--thermal-response", "collisionless", "--conv-tol", "1e-12",
               "--conv-tol-grad", "1e-10", "--conv-tol-dp", "1e-11",
               "--max-iter", "300", "--incore-mb", "64"]
    proc = subprocess.run(command, capture_output=True, text=True, check=True)
    report = json.loads(proc.stdout)
    for key in ("alpha_real", "alpha_imag"):
        np.testing.assert_allclose(report[key], native[key], atol=1e-9, rtol=0)


def test_thermal_selection_preserves_default_refusal_and_checks_context():
    with pytest.raises((ValueError, RuntimeError), match="integer projector"):
        dynamic_polarizability([1, 1], POSITIONS, omega=.03, damping=.01,
                               electronic_temperature=40000., **OPTIONS)
    with pytest.raises((ValueError, RuntimeError), match="thermal_response"):
        dynamic_polarizability([1, 1], POSITIONS, thermal_response="unknown", **OPTIONS)
    with pytest.raises((ValueError, RuntimeError), match="positive-temperature"):
        dynamic_polarizability([1, 1], POSITIONS,
                               thermal_response="collisionless", **OPTIONS)
    with pytest.raises((ValueError, RuntimeError), match="causal correlation kernel"):
        dynamic_polarizability([1, 1], POSITIONS, **dict(OPTIONS, xc="b2plyp"), **HOT)
    for reference in ("ghf", "gks", "rohf", "roks"):
        with pytest.raises((ValueError, RuntimeError), match="population-weighted"):
            dynamic_polarizability([1, 1], POSITIONS, reference=reference,
                                   **OPTIONS, **HOT)
