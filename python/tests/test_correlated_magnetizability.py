# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Full post-SCF orbital magnetic tensors through the common public driver."""

import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM


@pytest.mark.parametrize("reference", ["rks", "uks"])
@pytest.mark.parametrize("gauge", ["common-origin", "london"])
def test_correlated_magnetizability_public_routes(tmp_path, reference, gauge):
    positions = np.array([[0.13, -0.21, 0.19], [1.57, 0.16, -0.08],
                          [-0.19, 1.93, 0.37], [0.42, 0.29, 2.28]])
    if reference == "uks":
        positions = np.vstack([positions, [1.41, 1.83, 1.59]])
    positions -= positions.mean(axis=0)
    atoms = Atoms("H" * len(positions), positions=positions * BOHR_IN_ANGSTROM)
    controls = dict(xc="hf", reference=reference, multiplicity=2 if reference == "uks" else 1,
                    pt2_same_spin=0.31, pt2_opposite_spin=0.47, basis="sto-3g",
                    dispersion="none", grid_weight_cutoff=0.0, gauge=gauge,
                    gauge_origin=[0.12, -0.23, 0.34], guess="core", max_iter=300,
                    conv_tol=1e-14, conv_tol_dp=1e-13, conv_tol_grad=1e-13,
                    response_tolerance=1e-12)
    native = dft_rs.run_feature("magnetizability", atoms.numbers.tolist(), positions.tolist(), **controls)
    ordinary_controls = {k: v for k, v in controls.items()
                         if k not in ("pt2_same_spin", "pt2_opposite_spin")}
    ordinary = dft_rs.run_feature("magnetizability", atoms.numbers.tolist(),
                                  positions.tolist(), **ordinary_controls)
    np.testing.assert_allclose(native["parent_magnetizability"], ordinary["magnetizability"],
                               rtol=1e-8, atol=1e-9)

    adapted = dft_rs.DftRs(units="atomic").get_feature_report("magnetizability", atoms, **controls)
    xyz = tmp_path / "magnetic_cluster.xyz"
    xyz.write_text(str(len(atoms)) + "\nAsymmetric magnetic response cluster\n" +
                   "".join("H " + " ".join(f"{v:.17g}" for v in row) + "\n"
                           for row in atoms.positions), encoding="utf-8")
    command = [os.environ["DFT_RS_CLI"], "magnetizability", str(xyz), "--format", "json"]
    for key, value in controls.items():
        command += ["--" + key.replace("_", "-"), json.dumps(value) if isinstance(value, list) else str(value)]
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "energy" in process.stderr.lower() and "residual" in process.stderr.lower()
    keys = ["magnetizability", "parent_magnetizability", "target_magnetizability", "pt2_magnetizability",
            "same_spin_alpha_magnetizability_unscaled", "opposite_spin_magnetizability_unscaled",
            "same_spin_beta_magnetizability_unscaled"]
    for report in [native, adapted, json.loads(process.stdout)]:
        assert report["converged"] and report["provenance"] == "analytic"
        np.testing.assert_allclose(report["energy"], native["energy"], rtol=0, atol=1e-10)
        for key in keys:
            assert np.asarray(report[key]).shape == (3, 3)
            assert report["units"][key].startswith("e^2*bohr^2/m_e")
            np.testing.assert_allclose(report[key], native[key], rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(
            report["magnetizability"], np.asarray(report["target_magnetizability"]) +
            0.31 * (np.asarray(report["same_spin_alpha_magnetizability_unscaled"]) +
                    np.asarray(report["same_spin_beta_magnetizability_unscaled"])) +
            0.47 * np.asarray(report["opposite_spin_magnetizability_unscaled"]), rtol=0, atol=1e-10)
        for key in ["response_residuals", "amplitude_response_residuals", "orbital_frame_defects"]:
            assert np.all(np.isfinite(report[key])) and np.max(report[key]) < 1e-10
    print("Installed correlated magnetizability", reference, gauge,
          json.dumps({key: native[key] for key in ["energy", *keys, "response_residuals",
                                                  "amplitude_response_residuals"]}), flush=True)
