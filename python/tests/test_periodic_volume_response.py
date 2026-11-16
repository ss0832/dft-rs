# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Actual analytic volume curvature through native, ASE and canonical CLI."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms

from dft_rs import DftRs, mode_gruneisen
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr

POSITIONS = np.array([[.11, -.17, .23], [1.24, .19, .57], [-.37, 1.59, .42]])
CELL = np.array([[7.3, .21, .17], [.35, 7.9, .41], [.13, .29, 8.6]])


def fixture(tmp_path):
    basis = tmp_path / "volume-primitive.g94"
    basis.write_text("H 0\nS 1 1.0\n1.2 1.0\n****\nHe 0\nS 1 1.0\n1.6 1.0\n****\n")
    return dict(basis=str(basis), xc="hf", charge=0, overlap_cutoff=3.,
                exchange_cutoff=3., conv_tol=1e-14, conv_tol_dp=1e-13,
                conv_tol_grad=1e-12, max_iter=500, grid_weight_cutoff=0.,
                gruneisen_order=2, relaxed_ion=False)


def assert_same(a, b):
    assert a["gruneisen_order"] == b["gruneisen_order"] == 2
    assert not a["relaxed_ion"] and not b["relaxed_ion"]
    assert a["provenance"] == b["provenance"] == "analytic"
    for name in ("energy", "pressure", "force_constants", "dynamical_jets",
                 "reference_eigenvalues", "internal_first",
                 "internal_second", "ionic_stationarity_residual"):
        np.testing.assert_allclose(a[name], b[name], atol=1e-10, rtol=1e-9)
    np.testing.assert_array_equal(a["acoustic_modes"], b["acoustic_modes"])
    acoustic = np.asarray(a["acoustic_modes"], dtype=bool)
    wa, wb = (np.asarray(report["frequencies_cm1"]) for report in (a, b))
    np.testing.assert_allclose(wa[~acoustic], wb[~acoustic], atol=1e-10, rtol=1e-9)
    # The signed square root is ill-conditioned at translational zero modes.
    # Compare every acoustic eigenvalue in operator units at matrix roundoff,
    # stricter than the independently reported reference-eigenvalue check.
    conversion = 219474.63136320
    squared = [np.sign(w) * (w / conversion) ** 2 for w in (wa, wb)]
    scale = max(np.max(np.abs(a["reference_eigenvalues"])),
                np.max(np.abs(b["reference_eigenvalues"])))
    np.testing.assert_allclose(squared[0][acoustic], squared[1][acoustic],
                               atol=128 * np.finfo(float).eps * scale, rtol=0.)
    for name in ("mode_gruneisen", "second_log_frequency"):
        for x, y in zip(a[name], b[name]):
            if x is None or y is None:
                assert x is None and y is None
            else:
                np.testing.assert_allclose(x, y, atol=1e-8, rtol=1e-9)
    for report in (a, b):
        for acoustic, first, second in zip(report["acoustic_modes"],
                                           report["mode_gruneisen"],
                                           report["second_log_frequency"]):
            if acoustic:
                assert first is None and second is None


@pytest.mark.parametrize("with_dispersion", [False, True], ids=["hf", "pbe-d2"])
def test_second_volume_response_native_ase_cli_match(tmp_path, with_dispersion):
    controls = fixture(tmp_path)
    if with_dispersion:
        controls.update(xc="pbe", dispersion="d2", disp_cutoff=3.5)
    native = mode_gruneisen([1, 1, 2], POSITIONS, CELL, **controls)
    (tmp_path / "native.json").write_text(json.dumps(native))
    atoms = Atoms("HHHe", positions=POSITIONS * Bohr, cell=CELL * Bohr, pbc=True)
    ase = DftRs(units="atomic", **controls).get_mode_gruneisen(atoms)
    (tmp_path / "ase.json").write_text(json.dumps(ase))
    xyz = tmp_path / "volume.xyz"
    xyz.write_text("3\nC1 volume-response model\n" + "".join(
        element + " " + " ".join(f"{x * Bohr:.17g}" for x in row) + "\n"
        for element, row in zip(("H", "H", "He"), POSITIONS)))
    args = [os.environ["DFT_RS_CLI"], "mode-gruneisen", str(xyz), "--cell",
            ";".join(",".join(f"{x * Bohr:.17g}" for x in row) for row in CELL),
            "--format", "json"]
    for key, value in controls.items():
        args += ["--" + key.replace("_", "-"),
                 str(value).lower() if isinstance(value, bool) else str(value)]
    result = subprocess.run(args, capture_output=True, text=True, cwd=tmp_path, check=False)
    assert result.returncode == 0, result.stderr
    assert "pbc-scf" in result.stderr and "residual" in result.stderr
    cli = json.loads(result.stdout)
    (tmp_path / "cli.json").write_text(json.dumps(cli))
    assert_same(native, ase)
    assert_same(native, cli)
    if with_dispersion:
        for report in (native, ase, cli):
            assert report["dispersion_energy"] < 0.
            assert report["dispersion_energy_per_primitive_cell"] == report["dispersion_energy"]
            assert report["units"]["dispersion_energy_per_primitive_cell"] == "hartree per primitive cell"
        for report in (ase, cli):
            for key in ("dispersion_energy", "response_residuals", "series_reference_residual"):
                np.testing.assert_allclose(native[key], report[key], atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("controls,match", [
    ({"gruneisen_shear": True}, "isotropic"),
    ({"gruneisen_force_tolerance": 0.}, "positive"),
    ({"relaxed_ion": True, "gruneisen_force_tolerance": 1e-12}, "stationar"),
])
def test_volume_response_rejects_unsupported_or_nonstationary_model(tmp_path, controls, match):
    settings = fixture(tmp_path)
    settings.update(controls)
    with pytest.raises((ValueError, RuntimeError), match=match):
        mode_gruneisen([1, 1, 2], POSITIONS, CELL, **settings)
