# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Dimensionless covariant no-pair operator on native/ASE/CLI routes."""
import json
import os
import subprocess

import numpy as np

import dft_rs
from dft_rs.units import HARTREE_IN_EV as Hartree


def test_dimensionless_no_pair_operator_closed_block_and_public_routes():
    momentum = np.array([0.4, -0.3, 0.2])
    c = 5.0
    sigma = np.array([[momentum[2], momentum[0] - 1j * momentum[1]],
                      [momentum[0] + 1j * momentum[1], -momentum[2]]])
    h = np.zeros((4, 4), dtype=complex)
    h[:2, 2:] = c * sigma
    h[2:, :2] = c * sigma
    h[2:, 2:] = -2 * c * c * np.eye(2)
    energy = np.sqrt(c**4 + c*c * np.dot(momentum, momentum))
    projector = (h + (c*c + energy) * np.eye(4)) / (2 * energy)
    raw = np.array([[0.7, 0.2+0.1j, -0.3+0.2j, 0.4-0.3j],
                    [0, -0.5, 0.6-0.2j, -0.1+0.4j],
                    [0, 0, 1.3, 0.3+0.2j], [0, 0, 0, -0.9]], dtype=complex)
    operator = raw + np.triu(raw, 1).conj().T
    scaling = np.diag([1.1, 0.9, 1.2, 0.95])
    covariant = scaling @ operator @ scaling
    controls = dict(dirac_hamiltonian_real=(scaling @ h @ scaling).real.tolist(),
                    dirac_hamiltonian_imaginary=(scaling @ h @ scaling).imag.tolist(),
                    dirac_overlap_real=(scaling @ scaling).tolist(),
                    spectrum_separator=-25.0, response_order=4,
                    dirac_operator_real=[covariant.real.tolist()],
                    dirac_operator_imaginary=[covariant.imag.tolist()])
    direct = dft_rs.no_pair_projector(**controls)
    generic = dft_rs.run_feature("no-pair-projector", **controls)
    atomic = dft_rs.DftRs.get_no_pair_projector(units="atomic", **controls)
    physical = dict(controls)
    for name in ("dirac_hamiltonian_real", "dirac_hamiltonian_imaginary", "spectrum_separator"):
        physical[name] = np.asarray(controls[name]) * Hartree
    adapted = dft_rs.DftRs.get_no_pair_projector(**physical)
    argv = [os.environ["DFT_RS_CLI"], "no-pair-projector", "--format", "json"]
    for name, value in controls.items():
        argv.extend(["--" + name.replace("_", "-"), json.dumps(value)])
    process = subprocess.run(argv, text=True, capture_output=True, check=False)
    assert process.returncode == 0, process.stderr
    cli = json.loads(process.stdout)
    expected = scaling @ projector @ operator @ projector @ scaling
    actual = (np.asarray(direct["projected_operator_coefficients_real"])
              + 1j * np.asarray(direct["projected_operator_coefficients_imaginary"]))
    np.testing.assert_allclose(actual[0], expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(actual[1:], 0.0, rtol=0, atol=1e-12)
    for report in (direct, generic, atomic, adapted, cli):
        assert report["converged"]
        assert "bare no-pair projection" in report["projected_operator_model"]
        for name in ("projected_operator_coefficients_real", "projected_operator_coefficients_imaginary"):
            assert report["units"][name] == "dimensionless"
            np.testing.assert_allclose(report[name], direct[name], rtol=1e-11, atol=1e-12)
def test_static_projector_getter_preserves_explicit_resources():
    report = dft_rs.DftRs.get_no_pair_projector(
        dirac_hamiltonian_real=np.diag([1.0, 2.0, -5.0, -4.0]).tolist(),
        dirac_overlap_real=np.eye(4).tolist(),
        spectrum_separator=-1.0, response_order=0, units="atomic",
        dirac_operator_real=[np.eye(4).tolist()],
        cpu_threads=1, memory_mb=32)
    assert report["resources"]["cpu_threads"] == 1
    assert report["resources"]["memory_mb"] == 32
    np.testing.assert_allclose(
        report["projected_operator_coefficients_real"][0],
        np.diag([1.0, 1.0, 0.0, 0.0]), rtol=0, atol=1e-12)


