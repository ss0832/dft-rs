# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
from dft_rs import DftRs, dynamic_electric_tensor, dynamic_polarizability, dynamic_beta, dynamic_gamma

GEOMETRY = [[0, 0, 0], [1.75 / 3, 3.5 / 3, 3.5 / 3]]
OPTIONS = dict(charge=2, multiplicity=2, xc="hf", max_iter=250,
               conv_tol=1e-14, conv_tol_dp=1e-13, conv_tol_grad=1e-12)


@pytest.mark.parametrize("order,entry,method,command", [
    (2, dynamic_beta, "get_dynamic_beta", "dynamic-beta"),
    (3, dynamic_gamma, "get_dynamic_gamma", "dynamic-gamma"),
])
def test_all_complex_entries_and_diagnostics_agree_on_python_ase_and_cli(tmp_path, order, entry, method, command):
    frequencies = [[0.05 + 0.02 * i, 0.006 + 0.001 * i] for i in range(order)]
    direct = entry([2, 1], GEOMETRY, input_frequencies=frequencies, **OPTIONS)
    atoms = Atoms("HeH", positions=np.asarray(GEOMETRY) * Bohr)
    adapted = getattr(DftRs(units="atomic", **OPTIONS), method)(atoms, input_frequencies=frequencies)
    assert direct["converged"] and adapted["converged"]
    assert direct["shape"] == [3] * (order + 1)
    assert direct["subset_solves"] == 3 ** order * (2 ** order - 1)
    np.testing.assert_allclose(direct["output_frequency"], np.sum(frequencies, axis=0), atol=1e-15)
    for key in ("tensor_real", "tensor_imag", "residuals", "equation_defect", "projector_defect", "particle_number_defect"):
        np.testing.assert_allclose(direct[key], adapted[key], atol=1e-9)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for CLI route acceptance"
    path = tmp_path / "dynamic_response.xyz"
    path.write_text("2\ndynamic response validation\nHe 0 0 0\nH " + " ".join(f"{v * Bohr:.16f}" for v in GEOMETRY[1]) + "\n")
    cli = subprocess.run([executable, command, str(path), "--input-frequencies",
        ";".join(f"{w},{eta}" for w, eta in frequencies), "--charge", "2", "--multiplicity", "2",
        "--xc", "hf", "--conv-tol", "1e-14", "--conv-tol-dp", "1e-13", "--conv-tol-grad", "1e-12",
        "--max-iter", "250", "--log-mode", "none"], capture_output=True, text=True, check=True)
    label = "beta " if order == 2 else "gamma "
    rows = [line.split() for line in cli.stdout.splitlines() if line.startswith(label)]
    np.testing.assert_allclose([float(row[-2]) for row in rows], direct["tensor_real"], atol=1e-8)
    np.testing.assert_allclose([float(row[-1]) for row in rows], direct["tensor_imag"], atol=1e-8)
    assert direct["units"] in cli.stdout and "converged true" in cli.stdout
    assert "no Kleinman symmetrization" in cli.stdout
    for args in (["--input-frequencies", "0.1,-0.1;0.2,0.01"], ["--response-max-iter", "0"],
                 ["--input-frequencies", "0.1,0"], ["--input-frequencies", "0.1,0;0.2,0", "--response-omega", "0.1"]):
        invalid = subprocess.run([executable, command, "missing.xyz", *args], capture_output=True, text=True)
        assert invalid.returncode != 0 and "cannot read" not in invalid.stderr


def test_generic_dynamic_alpha_matches_paired_casida_and_frequency_reversal():
    direct = dynamic_electric_tensor([2, 1], GEOMETRY, input_frequencies=[[0.07, 0.006]], **OPTIONS)
    paired = dynamic_polarizability([2, 1], GEOMETRY, omega=0.07, damping=0.006, **OPTIONS)
    np.testing.assert_allclose(np.reshape(direct["tensor_real"], (3, 3)), paired["alpha_real"], atol=1e-9)
    np.testing.assert_allclose(np.reshape(direct["tensor_imag"], (3, 3)), paired["alpha_imag"], atol=1e-9)
    positive = dynamic_gamma([2, 1], GEOMETRY, input_frequencies=[[0.03, 0.002]] * 3, **OPTIONS)
    negative = dynamic_gamma([2, 1], GEOMETRY, input_frequencies=[[-0.03, 0.002]] * 3, **OPTIONS)
    np.testing.assert_allclose(positive["tensor_real"], negative["tensor_real"], atol=1e-9)
    np.testing.assert_allclose(positive["tensor_imag"], -np.asarray(negative["tensor_imag"]), atol=1e-9)
    imaginary = dynamic_beta([2, 1], GEOMETRY, input_frequencies=[[0.04, 0], [0.06, 0]], imaginary=True, **OPTIONS)
    assert imaginary["converged"]
    np.testing.assert_allclose(imaginary["tensor_imag"], 0, atol=1e-9)


def test_invalid_frequency_shapes_and_controls_are_validated_before_calculation():
    for inputs in ([], [[0, 0]] * 4, [[float("nan"), 0]], [[0.1, -0.001]]):
        with pytest.raises(ValueError):
            dynamic_electric_tensor([2, 1], GEOMETRY, input_frequencies=inputs, **OPTIONS)
    with pytest.raises(ValueError):
        dynamic_beta([2, 1], GEOMETRY, input_frequencies=[[0, 0]], **OPTIONS)
    with pytest.raises(ValueError):
        dynamic_gamma([2, 1], GEOMETRY, input_frequencies=[[0, 0]] * 2, **OPTIONS)
    with pytest.raises(ValueError):
        dynamic_electric_tensor([2, 1], GEOMETRY, input_frequencies=[[0.1, 0.001]], imaginary=True, **OPTIONS)
    with pytest.raises(ValueError):
        dynamic_electric_tensor([2, 1], GEOMETRY, input_frequencies=[[0.1, 0]], omega=0.2, **OPTIONS)


PRESETS = [
    ("shg", [0.05], [0.05, 0.05]),
    ("sfg", [0.05, 0.07], [0.05, 0.07]),
    ("dfg", [0.07, 0.05], [0.07, -0.05]),
    ("optical-rectification", [0.05], [0.05, -0.05]),
    ("electro-optic", [0.05], [0.05, 0]),
    ("thg", [0.05], [0.05, 0.05, 0.05]),
    ("efish", [0.05], [0.05, 0.05, 0]),
    ("field-induced-rectification", [0.05], [0.05, -0.05, 0]),
    ("two-photon-absorption", [0.05], [0.05, 0.05, -0.05]),
    ("optical-kerr", [0.05], [0.05, -0.05, 0.05]),
    ("dc-kerr", [0.05], [0.05, 0, 0]),
    ("four-wave-mixing", [0.05, 0.07, 0.09], [0.05, 0.07, -0.09]),
]


@pytest.mark.parametrize("process,frequencies,expected", PRESETS)
def test_physical_presets_on_all_routes(tmp_path, process, frequencies, expected):
    from dft_rs import dynamic_electric_process
    entry = dynamic_beta if len(expected) == 2 else dynamic_gamma
    command = "dynamic-beta" if len(expected) == 2 else "dynamic-gamma"
    direct = entry([2, 1], GEOMETRY, process=process, frequencies=frequencies, damping=0.006, **OPTIONS)
    factory = dynamic_electric_process([2, 1], GEOMETRY, process=process, frequencies=frequencies, damping=0.006, **OPTIONS)
    pairs = [[w, 0 if w == 0 else 0.006] for w in expected]
    explicit = entry([2, 1], GEOMETRY, input_frequencies=pairs, **OPTIONS)
    adapted = DftRs(units="atomic", **OPTIONS).get_dynamic_electric_process(
        Atoms("HeH", positions=np.asarray(GEOMETRY) * Bohr),
        process=process, frequencies=frequencies, damping=0.006)
    atoms = Atoms("HeH", positions=np.asarray(GEOMETRY) * Bohr)
    method = "get_dynamic_beta" if len(expected) == 2 else "get_dynamic_gamma"
    dedicated = getattr(DftRs(units="atomic", **OPTIONS), method)(
        atoms, process=process, frequencies=frequencies, damping=0.006)
    assert direct["converged"] and direct["process"] == process
    np.testing.assert_allclose(direct["input_frequencies"], pairs, atol=1e-15)
    for key in ("tensor_real", "tensor_imag", "output_frequency", "residuals"):
        for result in (explicit, factory, adapted, dedicated):
            np.testing.assert_allclose(direct[key], result[key], atol=1e-9)
    # Exercise both representations at the ASE boundary; neither may be converted twice.
    ase_options = dict(OPTIONS)
    for key in ("conv_tol", "conv_tol_grad"):
        ase_options[key] *= Hartree
    calculator = DftRs(**ase_options)
    physical = getattr(calculator, method)(atoms, process=process,
        frequencies=np.asarray(frequencies)*Hartree, damping=0.006*Hartree)
    physical_explicit = getattr(calculator, method)(atoms,
        input_frequencies=np.asarray(pairs)*Hartree)
    factor = Bohr**(len(expected)+1)/Hartree**len(expected)
    for result in (physical, physical_explicit):
        assert result["process"] == process if result is physical else result["process"] is None
        np.testing.assert_allclose(result["input_frequencies"], np.asarray(pairs)*Hartree, atol=1e-14)
        for key in ("tensor_real", "tensor_imag"):
            np.testing.assert_allclose(result[key], np.asarray(direct[key])*factor, rtol=1e-9, atol=1e-10)
    path = tmp_path / "electric_process.xyz"
    path.write_text("2\nelectric process validation\nHe 0 0 0\nH " + " ".join(f"{v * Bohr:.16f}" for v in GEOMETRY[1]) + "\n")
    executable = os.environ["DFT_RS_CLI"]
    result = subprocess.run([executable, command, str(path), "--electric-process", process,
        "--process-frequencies", ",".join(map(str, frequencies)), "--response-damping", "0.006",
        "--charge", "2", "--mult", "2", "--xc", "hf", "--conv-tol", "1e-14",
        "--conv-tol-dp", "1e-13", "--conv-tol-grad", "1e-12", "--log-mode", "none"],
        check=True, capture_output=True, text=True)
    prefix = "beta " if len(expected) == 2 else "gamma "
    rows = [line.split() for line in result.stdout.splitlines() if line.startswith(prefix)]
    np.testing.assert_allclose([float(r[-2]) for r in rows], direct["tensor_real"], atol=1e-8)
    np.testing.assert_allclose([float(r[-1]) for r in rows], direct["tensor_imag"], atol=1e-8)
    assert f"process {process}" in result.stdout and "converged true" in result.stdout


def test_process_arity_and_conflicting_controls_are_rejected():
    for options in (dict(process="unknown", frequencies=[0.05]), dict(process="sfg", frequencies=[0.05]),
        dict(process="shg", frequencies=[-0.05]), dict(process="shg", frequencies=[float("nan")]),
        dict(process="shg", frequencies=[0.05], imaginary=True), dict(frequencies=[0.05]),
        dict(process="shg"), dict(process="shg", frequencies=[0.05], input_frequencies=[[0.05, 0]] * 2)):
        with pytest.raises(ValueError):
            dynamic_electric_tensor([2, 1], GEOMETRY, **options, **OPTIONS)
    with pytest.raises(ValueError):
        dynamic_beta([2, 1], GEOMETRY, process="thg", frequencies=[0.05], **OPTIONS)
    with pytest.raises(ValueError):
        dynamic_gamma([2, 1], GEOMETRY, process="shg", frequencies=[0.05], **OPTIONS)
    executable = os.environ["DFT_RS_CLI"]
    for command, args in (("dynamic-beta", ["--electric-process", "thg"]),
        ("dynamic-beta", ["--electric-process", "sfg", "--response-omega", "0.05"]),
        ("dynamic-beta", ["--electric-process", "shg", "--imaginary-frequency"]),
        ("dynamic-gamma", ["--electric-process", "thg", "--process-frequencies", "0.05", "--response-omega", "0.05"]),
        ("beta", ["--electric-process", "shg"])):
        result = subprocess.run([executable, command, "missing.xyz", *args], capture_output=True, text=True)
        assert result.returncode and "cannot read" not in result.stderr
