# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Full-spin TDA public routes; no exact multiplicity label is inferred."""

import json
import os
from pathlib import Path
import subprocess

from ase import Atoms
import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM
import numpy as np
import pytest


@pytest.mark.parametrize("feature,native_name,ase_name", [
    ("excitations", "excitations", "get_excitations"),
    ("td-state-gradient", "td_state_gradient", "get_td_state_gradient"),
    ("td-hessian", "td_hessian", "get_td_hessian"),
])
def test_spinor_excited_public_routes_are_the_same_model(tmp_path, feature, native_name, ase_name):
    numbers = [1] * 4
    positions = np.asarray([[0.17, -0.11, 0.23], [1.77, 0.18, -0.53],
                            [-0.31, 1.74, -0.45], [-1.2, -1.31, -0.61]])
    real, imaginary = np.zeros((8, 8)), np.zeros((8, 8))
    for i in range(4):
        real[i, i] = -0.013*(i+1)
        real[4+i, 4+i] = -real[i, i]
        real[i, 4+i] = real[4+i, i] = 0.007*(i+1)
        imaginary[i, 4+i] = 0.009*(4-i)
        imaginary[4+i, i] = -imaginary[i, 4+i]
    options = dict(xc="hf", basis="sto-3g", reference="gks", unrestricted=True,
                   charge=1, multiplicity=2, guess="core", max_iter=500,
                   conv_tol=1e-13, conv_tol_dp=1e-13, conv_tol_grad=1e-12,
                   grid_weight_cutoff=0.0, nstates=15, td_tolerance=1e-9,
                   gks_operator_real=real.tolist(), gks_operator_imaginary=imaginary.tolist())
    if feature == "excitations":
        options.update(amplitudes=True, td_nto=True, td_difference_density=True)
    native = getattr(dft_rs, native_name)(numbers, positions.tolist(), **options)
    atoms = Atoms(numbers=numbers, positions=positions * BOHR_IN_ANGSTROM)
    adapted = getattr(dft_rs.DftRs(units="atomic", **options), ase_name)(atoms, **options)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable and Path(executable).is_file(), "DFT_RS_CLI selects the fixed current artifact"
    xyz = tmp_path / "c1_odd_electron.xyz"
    xyz.write_text("4\nC1 odd-electron fixed transverse source\n" + "".join(
        "H " + " ".join(f"{v:.17g}" for v in row) + "\n"
        for row in positions * BOHR_IN_ANGSTROM
    ), encoding="utf-8")
    args = [executable, feature, str(xyz), "--format", "json"]
    for key, value in options.items():
        args += ["--" + key.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)]
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", check=False)
    assert result.returncode == 0, result.stderr
    assert "SCF" in result.stderr
    for name, report in [("native", native), ("ASE", adapted), ("CLI", json.loads(result.stdout))]:
        assert report["td_spin"] == "spinor" and report["td_method"] == "tda"
        assert len(report["states"]) == 15
        for actual, expected in zip(report["states"], native["states"]):
            assert abs(actual["energy"] - expected["energy"]) < 1e-10
            assert actual["residual"] < 1e-9
            assert "multiplicity" not in actual
            assert abs(actual["oscillator_strength"] - expected["oscillator_strength"]) < 1e-9
            if feature == "excitations":
                channel, = actual["natural_transition_orbitals"]
                matrix = np.asarray(actual["amplitudes_real"]) + 1j*np.asarray(actual["amplitudes_imaginary"])
                singular = np.linalg.svd(matrix, compute_uv=False)
                np.testing.assert_allclose(channel["singular_values"], singular[singular > 1e-11],
                                           rtol=0., atol=1e-11)
                assert channel["reconstruction_residual"] < 1e-11
                difference, = actual["natural_difference_orbitals"]
                np.testing.assert_allclose(difference["attachment_occupations"],
                                           singular[singular > 1e-11]**2, rtol=0., atol=1e-11)
                assert abs(difference["attachment_electrons"] - 1.) < 1e-10
                other, = expected["natural_transition_orbitals"]
                np.testing.assert_allclose(channel["weights"], other["weights"], rtol=0., atol=1e-9)
        if feature != "excitations":
            assert report["provenance"] == "analytic" and report["response_converged"]
            assert report["response_residual"] < 1e-9
            assert report["td_gradient_route"] == "path"
            np.testing.assert_allclose(report["gradient"], native["gradient"], rtol=0, atol=1e-9)
            assert abs(report["excited_state_energy"]-native["excited_state_energy"]) < 1e-10
        if feature == "td-hessian":
            np.testing.assert_allclose(report["hessian"], native["hessian"], rtol=0, atol=1e-8)
            np.testing.assert_allclose(report["excitation_hessian"], native["excitation_hessian"], rtol=0, atol=1e-8)
        pole_error = max(abs(a["energy"]-b["energy"]) for a, b in zip(report["states"], native["states"]))
        gradient_error = (np.max(np.abs(np.asarray(report["gradient"])-native["gradient"]))
                          if feature != "excitations" else 0.0)
        hessian_error = (np.max(np.abs(np.asarray(report["hessian"])-native["hessian"]))
                         if feature == "td-hessian" else 0.0)
        print(f"{feature} {name}: pole_error={pole_error:.12e} gradient_error={gradient_error:.12e} "
              f"hessian_error={hessian_error:.12e} reference_defect={report['reference_fock_defect']:.12e} "
              f"response_residual={report.get('response_residual', 0.0):.12e}")
