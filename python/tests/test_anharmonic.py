# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

import os
import ast
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, anharmonic

# Independently converged analytic HF gradient at this bond is below 1e-10.
GEOMETRY = [[0, 0, 0], [0, 0, 1.7564722895565053]]
OPTIONS = dict(charge=1, xc="hf", conv_tol=1e-13, conv_tol_dp=1e-12,
               conv_tol_grad=1e-12, max_iter=500)


@pytest.fixture(scope="module")
def report():
    return anharmonic([2, 1], GEOMETRY, include_raman=True, **OPTIONS)


def test_native_and_ase_share_analytic_anharmonic_bands_and_physical_diagnostics(report):
    atoms = Atoms("HeH", positions=np.asarray(GEOMETRY)*Bohr)
    other = DftRs(units="atomic", **OPTIONS).get_anharmonic(atoms, include_raman=True, mode_ids=[0])
    assert report["converged"] and other["converged"]
    assert report["shape"] == [2] and report["selected_mode_ids"] == [0]
    assert report["maximum_force"] < 1e-9
    assert report["polarizability_symmetry_comparisons"] == 9
    assert report["polarizability_symmetry_defect"] < 1e-10
    assert report["force_permutation_comparisons"] == [0, 0]
    assert report["added_resonant_states"] == 0
    assert report["hermiticity_defect"] < 1e-12 and report["eigen_residual"] < 1e-12
    assert report["minimum_external_denominator"] > 0
    assert report["units"] == other["units"]
    assert all(unit in report["units"] for unit in ["cm^-1", "km/mol", "Angstrom^4/amu"])
    assert "Coriolis" in report["model"]
    assert report["rotation"]["class"] == "linear"
    assert report["rotation"]["watson_pseudopotential"] == 0
    assert report["rotation"]["eckart_defect"] < 1e-10
    for key in ["cubic", "quartic", "patterns", "polyad_energies", "effective_hamiltonian"]:
        np.testing.assert_allclose(report[key], other[key], atol=1e-10)
    for key in ["inverse_inertia", "principal_moments", "coriolis", "watson_pseudopotential",
                "eckart_defect", "mode_metric_defect"]:
        np.testing.assert_allclose(report["rotation"][key], other["rotation"][key], atol=1e-12)
    for key in ["dipole_derivatives", "polarizability_derivatives"]:
        for rank in ["first", "second", "third"]:
            np.testing.assert_allclose(report[key][rank], other[key][rank], atol=1e-10)
    for a, b in zip(report["bands"], other["bands"]):
        for key in ["frequency_cm", "ir_intensity", "raman_activity", "transition_dipole",
                    "mechanical_dipole", "electrical_dipole", "mixed_dipole", "transition_polarizability"]:
            np.testing.assert_allclose(a[key], b[key], atol=1e-8)
        total = sum(np.asarray(a[k]) for k in ["double_harmonic_dipole", "mechanical_dipole",
                                                "electrical_dipole", "mixed_dipole"])
        np.testing.assert_allclose(a["transition_dipole"], total, atol=1e-13)
        assert a["ir_intensity"] > 0 and a["raman_activity"] > 0
    assert report["bands"][0]["frequency_cm"] < report["bands"][0]["harmonic_frequency_cm"]
    assert report["bands"][1]["double_harmonic_ir_intensity"] == 0


def test_anharmonic_reference_and_resource_boundaries_are_checked():
    with pytest.raises(ValueError, match="unknown anharmonic method"):
        anharmonic([2, 1], GEOMETRY, method="vpt2", **OPTIONS)
    for request in [dict(mode_ids=[]), dict(mode_ids=[0, 0]), dict(mode_ids=[1]),
                    dict(resonance_tol=-1), dict(resonance_coupling_factor=float("nan")),
                    dict(max_basis_states=1), dict(max_polyad_states=1), dict(incore_mb=0)]:
        with pytest.raises(ValueError):
            anharmonic([2, 1], GEOMETRY, **request, **OPTIONS)
    with pytest.raises(ValueError, match="stationary"):
        anharmonic([2, 1], [[0,0,0],[0,0,1.5]], **OPTIONS)


def test_explicit_methods_preserve_default_and_reach_distinct_formulas(report):
    retained = anharmonic([2, 1], GEOMETRY, method="retained-polyad",
                         include_raman=True, **OPTIONS)
    strict = anharmonic([2, 1], GEOMETRY, method="strict-vpt2",
                       include_raman=True, **OPTIONS)
    assert report["anharmonic_method"] == retained["anharmonic_method"] == "retained-polyad"
    assert strict["anharmonic_method"] == "strict-vpt2"
    np.testing.assert_array_equal(report["polyad_energies"], retained["polyad_energies"])
    for key in ["cubic", "quartic", "patterns", "harmonic_frequencies_cm"]:
        np.testing.assert_array_equal(retained[key], strict[key])
    conversion = 219474.6313631965
    omega = strict["harmonic_frequencies_cm"][0] / conversion
    expected = omega + strict["quartic"][0] / (8 * omega**2) \
        - 5 * strict["cubic"][0]**2 / (24 * omega**4)
    fundamental = next(b for b in strict["bands"] if b["dominant_occupation"] == [1])
    np.testing.assert_allclose(fundamental["frequency_cm"], expected * conversion, atol=2e-6)
    assert abs(fundamental["frequency_cm"] - retained["bands"][0]["frequency_cm"]) > 1e-3
    assert strict["energy_order"] == strict["transition_moment_order"] == 2
    for key in ["energy_treatment", "resonance_treatment", "eigensystem_kind"]:
        assert strict[key] != retained[key]
    atoms = Atoms("HeH", positions=np.asarray(GEOMETRY) * Bohr)
    ase = DftRs(units="atomic", **OPTIONS).get_anharmonic(
        atoms, method="strict-vpt2", include_raman=True)
    assert ase["anharmonic_method"] == "strict-vpt2"
    np.testing.assert_allclose(ase["polyad_energies"], strict["polyad_energies"], atol=1e-12)
    with pytest.raises(ValueError, match="resonan"):
        anharmonic([2, 1], GEOMETRY, method="strict-vpt2", resonance_tol=1.0, **OPTIONS)


@pytest.mark.parametrize("variable", ["DFT_RS_CLI", "DFT_RS_LEGACY_CLI"])
def test_cli_strict_method_and_invalid_selection(variable, tmp_path):
    executable = os.environ.get(variable)
    if not executable:
        pytest.skip(f"{variable} is required for this CLI route")
    path = tmp_path / "minimum.xyz"
    path.write_text(f"2\nmethod selection\nHe 0 0 0\nH 0 0 {GEOMETRY[1][2]*Bohr:.16f}\n")
    command = [executable, "anharmonic", str(path), "--charge", "1", "--xc", "hf",
               "--conv-tol", "1e-13", "--conv-tol-dp", "1e-12", "--conv-tol-grad", "1e-12"]
    selected = subprocess.run(command + ["--anharmonic-method", "strict-vpt2"],
                              capture_output=True, text=True, check=True)
    assert "anharmonic_method strict-vpt2" in selected.stdout
    native = anharmonic([2, 1], GEOMETRY, method="strict-vpt2", **OPTIONS)
    rows = [line.split() for line in selected.stdout.splitlines() if line.startswith("anharmonic_band ")]
    np.testing.assert_allclose([float(row[3]) for row in rows],
                               [b["frequency_cm"] for b in native["bands"]], atol=1e-7)
    invalid = subprocess.run(command + ["--anharmonic-method", "vpt2"],
                             capture_output=True, text=True)
    assert invalid.returncode != 0 and "unknown anharmonic method" in invalid.stderr


def test_cli_and_python_match_anharmonic_ir_raman_and_diagnostics(report, tmp_path):
    executable = os.environ.get("DFT_RS_CLI")
    if not executable:
        pytest.skip("DFT_RS_CLI is required for the independent CLI route")
    path = tmp_path / "minimum.xyz"
    path.write_text(f"2\nstationary anharmonic route validation\nHe 0 0 0\nH 0 0 {GEOMETRY[1][2]*0.529177210903:.16f}\n")
    run = subprocess.run([executable,"anharmonic",str(path),"--charge","1","--xc","hf",
        "--anharmonic-raman","--vibrational-modes","0","--conv-tol","1e-13","--conv-tol-dp","1e-12",
        "--conv-tol-grad","1e-12","--log-mode","none"], check=True, capture_output=True, text=True)
    rows = [line.split() for line in run.stdout.splitlines() if line.startswith("anharmonic_band ")]
    assert len(rows) == 2
    for line, band in zip(rows, report["bands"]):
        np.testing.assert_allclose([float(x) for x in line[3:]],
            [band[k] for k in ["frequency_cm", "harmonic_frequency_cm", "ir_intensity",
                               "double_harmonic_ir_intensity", "dominant_weight"]], atol=1e-7)
    raman = [float(line.split()[2]) for line in run.stdout.splitlines() if line.startswith("raman_activity ")]
    np.testing.assert_allclose(raman, [b["raman_activity"] for b in report["bands"]], atol=1e-7)
    assert "polarizability_symmetry_comparisons 9" in run.stdout and "Coriolis" in run.stdout
    for key in ["cubic", "quartic", "polyad_energies", "patterns"]:
        line = next(line for line in run.stdout.splitlines() if line.startswith(key+" "))
        np.testing.assert_allclose(ast.literal_eval(line[len(key)+1:]), report[key], atol=1e-10)
    for rank in ["first", "second", "third"]:
        index = ["first", "second", "third"].index(rank)+1
        for name in ["dipole_derivatives", "polarizability_derivatives"]:
            prefix = f"{name}_{index} "
            line = next(line for line in run.stdout.splitlines() if line.startswith(prefix))
            np.testing.assert_allclose(ast.literal_eval(line[len(prefix):]), report[name][rank], atol=1e-10)
    for key in ["effective_hamiltonian", "polyad_vectors"]:
        rows = [ast.literal_eval(line.split(" ", 2)[2]) for line in run.stdout.splitlines() if line.startswith(key+" ")]
        np.testing.assert_allclose(rows, report[key], atol=1e-10)
    for key in ["inverse_inertia", "principal_moments", "coriolis", "watson_pseudopotential"]:
        line = next(line for line in run.stdout.splitlines() if line.startswith(key+" "))
        np.testing.assert_allclose(ast.literal_eval(line[len(key)+1:]), report["rotation"][key], atol=1e-12)
    for key in ["eckart_defect", "mode_metric_defect"]:
        line = next(line for line in run.stdout.splitlines() if line.startswith(key+" "))
        assert "Some(" in line
        measured = float(line[len(key)+1:].removeprefix("Some(").removesuffix(")"))
        np.testing.assert_allclose(measured, report["rotation"][key], atol=1e-12)
