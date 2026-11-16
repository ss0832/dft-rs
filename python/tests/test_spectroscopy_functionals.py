# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Collinear spectroscopy functional providers on every public surface."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr

import dft_rs
from dft_rs import DftRs
from dft_rs.units import BOHR_IN_ANGSTROM

POSITIONS = np.array([[0.13, -0.21, 0.19], [1.57, 0.16, -0.08],
                      [-0.19, 1.93, 0.37], [0.42, 0.29, 2.28]])
NUMBERS = [7, 1, 1, 1]
MODEL = os.environ.get("DFT_RS_SKALA_MODEL")


def options(xc, unrestricted):
    controls = dict(xc=xc, basis="sto-3g", reference="uks" if unrestricted else "rks",
                    charge=int(unrestricted), multiplicity=2 if unrestricted else 1,
                    conv_tol=1e-13, conv_tol_dp=1e-12, conv_tol_grad=1e-11,
                    grid_level=0, grid_weight_cutoff=0.0)
    if xc == "skala":
        if not MODEL:
            pytest.skip("DFT_RS_SKALA_MODEL selects an explicit learned acceptance model")
        controls["skala_model"] = MODEL
    return controls


def feature_options(feature):
    if feature == "ecd-spectrum":
        return dict(nstates=3, broadening_ev=0.2, points=301)
    if feature == "nmr-spectrum":
        return dict(nmr_reference_elements=[1, 7], nmr_reference_shieldings=[30., 350.],
                    nmr_points=201, response_tolerance=1e-11)
    return dict(gauge="common-origin", gauge_origin=[0.3, -0.2, 0.1],
                response_tolerance=1e-11)


FIELDS = {
    "ecd-spectrum": ["ecd_length", "ecd_velocity", "molar_absorptivity"],
    "nmr-spectrum": ["shielding_tensors", "chemical_shifts", "spectrum_intensity"],
    "vcd": ["atomic_axial_tensors", "atomic_axial_tensors_electronic",
            "rotational_strengths", "frequencies_cm1"],
}


@pytest.mark.parametrize("xc", ["hf", "svwn5", "pbe", "tpss", "pbe0", "skala"])
@pytest.mark.parametrize("unrestricted", [False, True])
def test_spectroscopy_native_ase_and_driver(xc, unrestricted):
    atoms = Atoms(numbers=NUMBERS, positions=POSITIONS * Bohr)
    calculator = DftRs(units="atomic")
    for feature, fields in FIELDS.items():
        controls = dict(options(xc, unrestricted), **feature_options(feature))
        native = getattr(dft_rs, feature.replace("-", "_"))(
            NUMBERS, POSITIONS.tolist(), **controls)
        common = dft_rs.run_feature(feature, NUMBERS, POSITIONS.tolist(), **controls)
        ase = getattr(calculator, "get_" + feature.replace("-", "_"))(atoms, **controls)
        assert native["converged"] and common["converged"] and ase["converged"]
        for field in fields:
            values = np.asarray(native[field], dtype=float)
            assert np.isfinite(values).all()
            np.testing.assert_allclose(values, common[field], rtol=0, atol=1e-11)
            np.testing.assert_allclose(values, ase[field], rtol=0, atol=1e-8)
        if feature != "ecd-spectrum":
            assert max(native["response_residuals"]) <= 1e-10


@pytest.mark.parametrize("feature", list(FIELDS))
@pytest.mark.parametrize("xc", ["pbe0", "skala"])
def test_spectroscopy_canonical_cli(feature, xc, tmp_path):
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for public route acceptance"
    xyz = tmp_path / "skew_ammonia.xyz"
    xyz.write_text("4\ncollinear spectroscopy\n" + "".join(
        symbol + " " + " ".join(f"{x:.17g}" for x in point) + "\n"
        for symbol, point in zip(["N", "H", "H", "H"], POSITIONS * BOHR_IN_ANGSTROM)),
        encoding="utf-8")
    controls = dict(options(xc, True), **feature_options(feature))
    args = [executable, feature, str(xyz), "--format", "json"]
    for key, value in controls.items():
        args += ["--" + key.replace("_", "-"), value if isinstance(value, str)
                 else json.dumps(value)]
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "iter" in result.stderr and "residual" in result.stderr
    report = json.loads(result.stdout)
    native = dft_rs.run_feature(feature, NUMBERS, POSITIONS.tolist(), **controls)
    for field in FIELDS[feature]:
        np.testing.assert_allclose(report[field], native[field], rtol=0, atol=1e-8)
