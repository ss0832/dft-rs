# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Complete public tensor maps for scalar ECP constrained/complex references.

Independent nuclear derivative fits are in p293. These checks compare every
entry of the selected public tensor, including mixed pattern components,
at finite electric bias. The additional spin/AO operator is held fixed.
"""
import json
import os
from pathlib import Path
import subprocess

from ase import Atoms
import numpy as np
import pytest

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM


POSITIONS = np.array([[0.17, -0.11, 0.23], [2.41, 0.296, -0.834],
                      [-0.502, 2.48, -0.722], [-1.748, -1.79, -0.946]])


def settings(reference):
    basis = Path(__file__).resolve().parents[2] / "tests/data/c1_compact_valence.g94"
    controls = dict(basis=str(basis), ecp="lanl2dz", xc="hf", reference=reference,
                    charge=1, multiplicity=2, unrestricted=True, dispersion="none",
                    grid_weight_cutoff=0.0, conv_tol=1e-14, conv_tol_dp=1e-14,
                    conv_tol_grad=1e-14, response_tolerance=1e-12,
                    max_iter=500, guess="core", field=[0.013, -0.007, 0.011])
    if reference == "gks":
        real = np.zeros((10, 10))
        imaginary = np.zeros_like(real)
        for i in range(5):
            real[i, i] = -0.013 * (i + 1)
            real[5+i, 5+i] = -real[i, i]
            real[i, 5+i] = real[5+i, i] = 0.007 * (i + 1)
            imaginary[i, 5+i] = 0.009 * (5 - i)
            imaginary[5+i, i] = -imaginary[i, 5+i]
        controls.update(gks_operator_real=real.tolist(),
                        gks_operator_imaginary=imaginary.tolist())
    return controls


@pytest.mark.parametrize("reference", ["roks", "gks"])
@pytest.mark.parametrize("kind", ["gradient", "hessian", "cubic", "quartic"])
def test_all_public_scalar_ecp_reference_tensor_entries(reference, kind, tmp_path):
    controls = settings(reference)
    feature = kind
    keys = ["energy", "gradient", "forces"]
    if kind == "hessian":
        keys.append("hessian")
    if kind in ("cubic", "quartic"):
        feature = "force-constants"
        order = 3 if kind == "cubic" else 4
        patterns = np.array([np.sin(0.31+np.arange(1, 13)),
                             np.cos(0.47+0.71*np.arange(1, 13))])
        patterns /= np.linalg.norm(patterns, axis=1)[:, None]
        controls.update(response_order=order, patterns=patterns.reshape(2, 4, 3).tolist())
        keys = ["energy", "tensor", "parent_tensor", "dispersion_tensor"]
    native = dft_rs.run_feature(feature, [15, 1, 1, 1], POSITIONS.tolist(), **controls)
    assert native["converged"]
    assert native["provenance"] == "analytic"
    if kind == "hessian":
        assert np.asarray(native["hessian"]).shape == (12, 12)
    if feature == "force-constants":
        assert native["shape"] == [2]*order
        assert len(native["tensor"]) == 2**order
    atoms = Atoms(numbers=[15, 1, 1, 1], positions=POSITIONS*BOHR_IN_ANGSTROM)
    calculator = dft_rs.DftRs()
    adapted = calculator.get_feature_report(feature, atoms, units="atomic", **controls)
    xyz = tmp_path / "ecp_reference.xyz"
    xyz.write_text("4\nC1 scalar ECP reference\n" + "".join(
        f"{symbol} " + " ".join(f"{value:.17g}" for value in row) + "\n"
        for symbol, row in zip(["P", "H", "H", "H"], POSITIONS*BOHR_IN_ANGSTROM)),
        encoding="utf-8")
    command = [os.environ["DFT_RS_CLI"], feature, str(xyz), "--format", "json"]
    for name, value in controls.items():
        command += ["--"+name.replace("_", "-"),
                    json.dumps(value) if isinstance(value, list) else str(value).lower()]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    canonical = json.loads(result.stdout)
    assert "residual" in result.stderr
    maximum = 0.0
    for key in keys:
        expected = np.asarray(native[key], dtype=float)
        assert np.isfinite(expected).all()
        for report in [adapted, canonical]:
            actual = np.asarray(report[key], dtype=float)
            assert expected.shape == actual.shape
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=0, equal_nan=False)
            maximum = max(maximum, float(np.max(np.abs(actual-expected))))
    print(reference, kind, "all_public_entries_maximum_error", maximum)
