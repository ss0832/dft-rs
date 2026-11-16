# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Public nuclear-g contracts; independent physics fits live in Rust suites."""
import json
import os
from pathlib import Path
import subprocess

import dft_rs
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr


NUMBERS = [1, 1]
POSITIONS = np.array([[.13, -.21, .17], [2.05, .25, .38]])
SETTINGS = dict(xc="hf", reference="uhf", basis="sto-3g", charge=1,
                multiplicity=2, unrestricted=True, ecp="none",
                conv_tol=1e-14, conv_tol_dp=1e-13, conv_tol_grad=1e-12,
                response_tolerance=1e-12, response_max_iterations=1000,
                cpu_threads=1, memory_mb=512)


@pytest.fixture(scope="module")
def second_order():
    return dft_rs.g_tensor(NUMBERS, POSITIONS, nuclear_order=2, **SETTINGS)


def check(report, order):
    assert report["converged"] and report["response_converged"]
    assert report["reference"] == "uhf"
    assert report["two_electron_soc_included"] is False
    assert report["picture_change_included"] is False
    assert report["nuclear_order"] == order
    assert report["resources"]["cpu_threads"] == 1
    assert np.asarray(report["g_tensor"]).shape == (3, 3)
    assert np.asarray(report["g_gradient"]).shape == (6, 3, 3)
    assert report["units"]["g_gradient"].startswith("1/")
    assert report["units"]["g_hessian"].startswith("1/")
    assert np.isfinite(report["g_gradient"]).all()
    if order == 1:
        assert report["g_hessian"] is None and report["mass_hessian"] is None
    else:
        assert np.asarray(report["g_hessian"]).shape == (6, 6, 3, 3)
        assert np.asarray(report["mass_hessian"]).shape == (6, 6)
        assert np.isfinite(report["g_hessian"]).all()
    assert len(report["pure_field_paths"]) == 3
    for path in report["raw_paths"]:
        assert path["real_converged"] and path["real_reference"]["converged"]
        assert np.isfinite(path["real_residuals"]).all()
        assert path["response_degrees"] == list(range(1, len(path["real_projector_defects"]) + 1))
        for mixed in path["mixed"]:
            raw = mixed["report"]
            assert raw["response_degrees"] == list(range(1, len(raw["residuals"]) + 1))
            assert mixed["nuclear_sign"] in (-1, 1)


def test_native_and_ase_units_and_shapes(second_order):
    check(second_order, 2)
    atoms = Atoms(numbers=NUMBERS, positions=POSITIONS * Bohr)
    for mode in ("atomic", "ase"):
        report = dft_rs.DftRs(units=mode).get_g_tensor(atoms, nuclear_order=2, **SETTINGS)
        check(report, 2)
        for order, key in enumerate(("g_tensor", "g_gradient", "g_hessian")):
            factor = Bohr ** (-order) if mode == "ase" else 1.
            np.testing.assert_allclose(report[key], np.asarray(second_order[key]) * factor,
                                       rtol=0, atol=1e-12)


def test_zero_electric_field_and_unrequested_hessian():
    ordinary = dft_rs.g_tensor(NUMBERS, POSITIONS, nuclear_order=1, **SETTINGS)
    explicit = dft_rs.g_tensor(NUMBERS, POSITIONS, nuclear_order=1, field=[0., 0., 0.], **SETTINGS)
    check(ordinary, 1)
    check(explicit, 1)
    np.testing.assert_array_equal(ordinary["g_tensor"], explicit["g_tensor"])
    np.testing.assert_array_equal(ordinary["g_gradient"], explicit["g_gradient"])


def test_static_order_zero_uses_existing_path():
    default = dft_rs.g_tensor(NUMBERS, POSITIONS, **SETTINGS)
    explicit = dft_rs.g_tensor(NUMBERS, POSITIONS, nuclear_order=0, **SETTINGS)
    for key in ("g_tensor", "g_shift_ppm", "mass_correction_ppm",
                "gauge_correction_ppm", "spin_orbit_ppm"):
        np.testing.assert_array_equal(default[key], explicit[key])


@pytest.mark.parametrize("controls", [
    {"reference": name} for name in ("rhf", "rks", "rohf", "roks", "ghf", "gks")
] + [{"xc": "pbe"}, {"gauge": "common-origin"}, {"stabilize": True},
     {"field": [.00013, -.00021, .00017]}, {"electronic_temperature": 300.}])
def test_unvalidated_combinations_fail_before_basis_or_scf(controls):
    options = dict(SETTINGS, basis="missing-nuclear-g-basis", **controls)
    with pytest.raises(NotImplementedError):
        dft_rs.g_tensor(NUMBERS, POSITIONS, nuclear_order=1, **options)


@pytest.mark.parametrize("value", [True, 1., -1, "1"])
def test_nuclear_order_is_a_strict_integer(value):
    with pytest.raises((ValueError, TypeError)):
        dft_rs.g_tensor(NUMBERS, POSITIONS, nuclear_order=value, **SETTINGS)


def test_unknown_control_and_response_nonconvergence_are_errors():
    with pytest.raises(ValueError, match="unknown|unused|unrecognized"):
        dft_rs.g_tensor(NUMBERS, POSITIONS, nuclear_order=1,
                        unsupported_typo=1, **SETTINGS)
    with pytest.raises(RuntimeError):
        dft_rs.g_tensor(NUMBERS, POSITIONS, nuclear_order=1,
                        **dict(SETTINGS, response_max_iterations=1, response_tolerance=1e-30))


def test_both_cli_names_match_native(second_order, tmp_path):
    primary = Path(os.environ["DFT_RS_CLI"])
    alias = Path(os.environ.get("DFT_RS_CLI_ALIAS", str(primary.with_name("dft_rs_cli"))))
    geometry = tmp_path / "one-electron-pair.xyz"
    geometry.write_text("2\nNuclear g API fixture\n" + "\n".join(
        "H " + " ".join(f"{v*Bohr:.17g}" for v in row) for row in POSITIONS) + "\n")
    args = ["g-tensor", str(geometry), "--format", "json", "--nuclear-order", "2"]
    for key, value in SETTINGS.items():
        args += ["--" + key.replace("_", "-") + "=" +
                 (value if isinstance(value, str) else json.dumps(value))]
    for binary in (primary, alias):
        completed = subprocess.run([str(binary), *args], capture_output=True, text=True, check=True)
        report = json.loads(completed.stdout)
        check(report, 2)
        np.testing.assert_allclose(report["g_hessian"], second_order["g_hessian"], rtol=0, atol=1e-12)
