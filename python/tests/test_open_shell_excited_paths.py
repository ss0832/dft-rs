# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Common-orbital reference-spin TDA through native, ASE and canonical CLI."""
import json
import os
from pathlib import Path
import subprocess

from ase import Atoms
import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM
import numpy as np
import pytest


@pytest.mark.parametrize("multiplicity,dimension", [(2, 5), (4, 3)])
@pytest.mark.parametrize("feature,native_name,ase_name", [
    ("excitations", "excitations", "get_excitations"),
    ("td-state-gradient", "td_state_gradient", "get_td_state_gradient"),
    ("td-hessian", "td_hessian", "get_td_hessian"),
])
def test_common_orbital_excited_public_routes(tmp_path, multiplicity, dimension,
                                           feature, native_name, ase_name):
    positions = np.asarray([[0.17, -0.11, 0.23], [1.77, 0.18, -0.53],
                            [-0.31, 1.74, -0.45], [-1.2, -1.31, -0.61]])
    options = dict(xc="hf", basis="sto-3g", reference="roks", charge=1,
                   multiplicity=multiplicity, guess="core", max_iter=500,
                   conv_tol=1e-14, conv_tol_dp=1e-14, conv_tol_grad=1e-14,
                   grid_weight_cutoff=0.0, td_tolerance=1e-10,
                   response_tolerance=1e-11, nstates=dimension, td_spin="reference-spin")
    if feature == "excitations":
        options.update(amplitudes=True, td_nto=True)
    native = getattr(dft_rs, native_name)([1] * 4, positions.tolist(), **options)
    atoms = Atoms(numbers=[1] * 4, positions=positions * BOHR_IN_ANGSTROM)
    adapted = getattr(dft_rs.DftRs(units="atomic", **options), ase_name)(atoms, **options)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable and Path(executable).is_file()
    xyz = tmp_path / "c1_common_orbital.xyz"
    xyz.write_text("4\nC1 common-orbital reference\n" + "".join(
        "H " + " ".join(f"{v:.17g}" for v in row) + "\n"
        for row in positions * BOHR_IN_ANGSTROM
    ), encoding="utf-8")
    args = [executable, feature, str(xyz), "--format", "json"]
    for key, value in options.items():
        args += ["--" + key.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)]
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", check=False)
    assert result.returncode == 0, result.stderr
    assert "SCF" in result.stderr or "ROKS" in result.stderr
    for route, report in [("native", native), ("ASE", adapted), ("CLI", json.loads(result.stdout))]:
        assert report["td_spin"] == "reference-spin" and report["td_method"] == "tda"
        assert len(report["states"]) == dimension
        assert report["reference_stationarity"] < 1e-10
        assert report["reference_metric_defect"] < 1e-10
        pole_error = 0.0
        for actual, expected in zip(report["states"], native["states"]):
            pole_error = max(pole_error, abs(actual["energy"] - expected["energy"]))
            assert actual["residual"] < 1e-10
            assert "multiplicity" not in actual
            assert abs(actual["oscillator_strength"] - expected["oscillator_strength"]) < 1e-9
            if feature == "excitations":
                channel, = actual["natural_transition_orbitals"]
                nopen = multiplicity - 1
                nc = (3 - nopen) // 2
                no = nc + nopen
                nv = 4 - no
                rotations = np.asarray(actual["flag_amplitudes"])
                closed_open = nc*nopen
                closed_virtual = closed_open + nc*nv
                matrix = np.zeros((4, 4))
                matrix[nc:no, :nc] = rotations[:closed_open].reshape(nc, nopen).T
                matrix[no:, :nc] = 2. * rotations[closed_open:closed_virtual].reshape(nc, nv).T
                matrix[no:, nc:no] = rotations[closed_virtual:].reshape(nopen, nv).T
                singular = np.linalg.svd(matrix, compute_uv=False)
                np.testing.assert_allclose(channel["singular_values"], singular[singular > 1e-11],
                                           rtol=0., atol=1e-11)
                assert channel["reconstruction_residual"] < 1e-11
                other, = expected["natural_transition_orbitals"]
                np.testing.assert_allclose(channel["weights"], other["weights"], rtol=0., atol=1e-9)
        assert pole_error < 1e-10
        gradient_error = hessian_error = 0.0
        if feature != "excitations":
            assert report["provenance"] == "analytic" and report["response_converged"]
            assert report["response_residual"] < 1e-10
            gradient_error = np.max(np.abs(np.asarray(report["gradient"]) - native["gradient"]))
            assert gradient_error < 1e-9
            assert abs(report["excited_state_energy"] - native["excited_state_energy"]) < 1e-10
        if feature == "td-hessian":
            hessian_error = np.max(np.abs(np.asarray(report["hessian"]) - native["hessian"]))
            assert hessian_error < 1e-8
        print(f"{feature} multiplicity={multiplicity} {route}: pole_error={pole_error:.12e} "
              f"gradient_error={gradient_error:.12e} hessian_error={hessian_error:.12e} "
              f"stationarity={report['reference_stationarity']:.12e} "
              f"response_residual={report.get('response_residual', 0.0):.12e}")
