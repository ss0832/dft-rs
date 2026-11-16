# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Public API parity for the retarded full particle/hole response."""

import json
import os
import subprocess

import numpy as np
import pytest

import dft_rs
from dft_rs.units import HARTREE_IN_EV


def _bdg(normal, pairing):
    return np.block([[normal, pairing], [pairing.conj().T, -normal.conj()]])


def _fixture():
    normal = np.array([[-0.114, 0.032 + 0.008j], [0.032 - 0.008j, 0.086]])
    pairing = np.array([[0, 0.027 - 0.018j], [-0.027 + 0.018j, 0]])
    overlap = np.array([[1.07, 0.013 - 0.029j], [0.013 + 0.029j, 0.961]])
    probe_normal = np.array([
        [[0.29, 0.27 + 0.13j], [0.27 - 0.13j, -0.11]],
        [[-0.26, -0.09 + 0.28j], [-0.09 - 0.28j, 0.078]],
    ])
    probe_pairing = np.array([
        [[0, 0.16 + 0.21j], [-0.16 - 0.21j, 0]],
        [[0, 0.26 - 0.31j], [-0.26 + 0.31j, 0]],
    ])
    mu, kt = -0.013, 0.038
    u = np.array([[0.031, -0.012], [-0.012, 0.019]])
    controls = dict(
        pairing_model="nambu", electronic_temperature=kt / 3.166811563455608e-6,
        chemical_potential=mu, frequencies=[0.067, -0.067, 0.0],
        frequency_damping=0.008, nambu_probe_interaction=u.tolist(),
        response_tolerance=1e-12,
    )
    for key, value in (
        ("nambu_normal", normal), ("nambu_pairing", pairing), ("nambu_overlap", overlap),
        ("nambu_probe_normal", probe_normal.reshape(2, -1)),
        ("nambu_probe_pairing", probe_pairing.reshape(2, -1)),
    ):
        controls[f"{key}_real"] = value.real.tolist()
        controls[f"{key}_imaginary"] = value.imag.tolist()
    s = _bdg(overlap, np.zeros_like(overlap))
    s[2:, 2:] *= -1
    eig_s, c = np.linalg.eigh(s)
    s_inv_root = (c / np.sqrt(eig_s)) @ c.conj().T
    h = s_inv_root @ _bdg(normal - mu * overlap, pairing) @ s_inv_root
    q = s_inv_root @ np.array([_bdg(n, p) for n, p in
                               zip(probe_normal, probe_pairing)]) @ s_inv_root
    return controls, h, q, kt, u


def _matrix_response(h, q, kt, u, z):
    """Solve the general complex density-matrix equation in orbital coordinates."""
    eig_h, c = np.linalg.eigh(h)
    r = (c / (1 + np.exp(eig_h / kt))) @ c.conj().T
    sources = q @ r - r @ q
    size = h.size
    op = np.empty((size, size), dtype=complex)
    for j in range(size):
        basis = np.zeros(size).reshape(h.shape)
        basis.flat[j] = 1
        observations = np.einsum("pij,ji->p", q, basis) / 2
        induced = np.einsum("aij,a->ij", sources, u @ observations)
        op[:, j] = (z * basis - h @ basis + basis @ h - induced).ravel()
    response = np.linalg.solve(op, sources.reshape(len(q), -1).T)
    return np.einsum("aij,jib->ab", q, response.reshape(h.shape + (len(q),))) / 2


def _report_matrix(report, key):
    return np.asarray(report[key + "_real"]) + 1j * np.asarray(report[key + "_imaginary"])


def _cli(controls):
    command = [os.environ["DFT_RS_CLI"], "superconducting-state", "--format", "json"]
    for key, value in controls.items():
        command.extend(("--" + key.replace("_", "-"),
                        value if isinstance(value, str) else json.dumps(value)))
    return subprocess.run(command, capture_output=True, text=True, check=False)


def test_causal_nambu_public_routes():
    controls, h, q, kt, u = _fixture()
    native = dft_rs.superconducting_state(**controls)
    generic = dft_rs.run_feature("superconducting-state", **controls)
    atomic = dft_rs.DftRs.get_superconducting_state(units="atomic", **controls)
    energy_inputs = (
        "nambu_normal_real", "nambu_normal_imaginary", "nambu_pairing_real",
        "nambu_pairing_imaginary", "frequencies", "frequency_damping",
        "chemical_potential", "nambu_probe_interaction",
    )
    ev_controls = {key: (np.asarray(value) * HARTREE_IN_EV).tolist()
                   if key in energy_inputs else value for key, value in controls.items()}
    adapted = dft_rs.DftRs.get_superconducting_state(**ev_controls)
    proc = _cli(controls)
    assert proc.returncode == 0, proc.stderr
    cli = json.loads(proc.stdout)
    residuals = ("bare_spectral_residuals", "coupled_response_residuals",
                 "reference_eigen_residual", "reference_metric_defect")
    for key, interaction in (("nambu_bare_susceptibility", np.zeros_like(u)),
                             ("nambu_susceptibility", u)):
        expected = np.asarray([
            _matrix_response(h, q, kt, interaction, omega + 1j * controls["frequency_damping"])
            for omega in controls["frequencies"]
        ])
        for report, energy, unit in (
            (native, 1.0, "hartree"), (generic, 1.0, "hartree"), (cli, 1.0, "hartree"),
            (atomic, 1.0, "hartree"), (adapted, HARTREE_IN_EV, "eV"),
        ):
            assert report["response_order"] == 1
            assert report["units"][key + "_real"] == f"1/{unit}"
            assert report["units"][key + "_imaginary"] == f"1/{unit}"
            assert report["units"]["electronic_temperature"] == "K"
            assert report["electronic_temperature"] == pytest.approx(controls["electronic_temperature"])
            np.testing.assert_allclose(
                _report_matrix(report, key) * energy, expected, rtol=2e-12, atol=2e-12,
            )
            np.testing.assert_allclose(
                report["frequencies"], np.asarray(controls["frequencies"]) * energy,
                rtol=1e-13, atol=0,
            )
            for residual in residuals:
                assert report["units"][residual] == "dimensionless"
                assert np.max(report[residual]) < 2e-11
        np.testing.assert_allclose(expected[0].conj(), expected[1], rtol=2e-12, atol=2e-12)
        assert expected[0, 0, 0].imag < 0
    print("Causal Nambu", _report_matrix(native, "nambu_susceptibility")[0].tolist(),
          "eigen/metric and spectral/induced defects", {key: native[key] for key in residuals},
          "isospectral stability", native["isospectral_stability_margin"])


def test_causal_nambu_rejects_unstable_curvature():
    controls, _, _, _, _ = _fixture()
    controls["nambu_probe_interaction"] = [[-100.0, 0], [0, -100.0]]
    with pytest.raises(ValueError, match="unstable collisionless Nambu reference"):
        dft_rs.superconducting_state(**controls)
    proc = _cli(controls)
    assert proc.returncode != 0
    assert "Nambu" in proc.stderr and ("stability" in proc.stderr or "unstable" in proc.stderr)
    print("Causal Nambu unstable curvature CLI exit", proc.returncode)
