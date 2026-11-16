# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Public collinear HF nuclear-A report parity, units and failure semantics.

The independent complete derivative fits live in p_hyperfine_nuclear.rs.
These tests exercise the actual common driver and language boundaries.
"""
import json
import os
from pathlib import Path
import subprocess

import dft_rs
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree


NUMBERS = [6, 1, 1, 1]
POSITIONS = np.array([[.13, -.21, .17], [2.05, .25, .38],
                      [-.65, 1.76, -.44], [-1.19, -1.48, .85]])
SETTINGS = dict(xc="hf", basis="sto-3g", multiplicity=2, unrestricted=True,
                dispersion="none", conv_tol=1e-14, conv_tol_grad=1e-12,
                conv_tol_dp=1e-13, response_tolerance=1e-12,
                response_max_iterations=1000, cpu_threads=1, memory_mb=512)
SIGNED_G = [-2., 1., 1., 1.]
PHYSICAL = tuple(f"{part}_{derivative}{suffix}"
                 for part in ("contact", "dipolar", "total")
                 for derivative in ("gradient", "hessian")
                 for suffix in ("_per_g", ""))


@pytest.fixture(scope="module")
def native_order_two():
    return dft_rs.run_feature("hyperfine", NUMBERS, POSITIONS,
                             **SETTINGS, nuclear_order=2,
                             derivative_nuclei=[0], nuclear_g=SIGNED_G)


def check_report(report):
    assert report["converged"]
    child = report["hyperfine_nuclear"]
    assert child["converged"] and child["order"] == 2
    assert child["derivative_nuclei"] == [0]
    assert len(child["raw_paths"]) == 78
    for part in ("contact", "dipolar", "total"):
        trailing = () if part == "contact" else (3, 3)
        for derivative, coordinates in (("gradient", (12,)), ("hessian", (12, 12))):
            per_g = np.asarray(child[f"{part}_{derivative}_per_g"])
            scaled = np.asarray(child[f"{part}_{derivative}"])
            assert per_g.shape == (1,) + coordinates + trailing
            assert np.isfinite(per_g).all() and np.isfinite(scaled).all()
            np.testing.assert_allclose(scaled, -2 * per_g, rtol=0, atol=1e-10)
    for path in child["raw_paths"]:
        assert path["converged"] and np.isfinite(path["residuals"]).all()
        rows = path["coefficient_diagnostics"]
        assert [row["order"] for row in rows] == [0, 1, 2]
        for name in ("stationarity_defect", "metric_defect", "projector_defect",
                     "particle_number_coefficient"):
            assert rows[0][name] is None
            assert all(np.isfinite(row[name]).all() for row in rows[1:])
        assert rows[0]["coefficient_equation"] is None
        for order in (1, 2):
            assert rows[order]["coefficient_equation"]["order"] == order


def compare_physical(native, other, converted=False):
    left, right = native["hyperfine_nuclear"], other["hyperfine_nuclear"]
    assert right["derivative_nuclei"] == left["derivative_nuclei"]
    assert right["order"] == left["order"]
    for key in PHYSICAL:
        order = 2 if "hessian" in key else 1
        expected = np.asarray(left[key]) / Bohr**order if converted else left[key]
        np.testing.assert_allclose(right[key], expected, rtol=0, atol=1e-8)


def test_native_and_both_ase_unit_modes_preserve_all_derivatives(native_order_two):
    check_report(native_order_two)
    atoms = Atoms(numbers=NUMBERS, positions=POSITIONS * Bohr)
    atomic_report = None
    for mode in ("atomic", "ase"):
        adapted = dft_rs.DftRs(units=mode).get_hyperfine(
            atoms, **SETTINGS, nuclear_order=2, derivative_nuclei=[0], nuclear_g=SIGNED_G)
        check_report(adapted)
        compare_physical(native_order_two, adapted, converted=mode == "ase")
        if mode == "atomic":
            atomic_report = adapted
        # Compare diagnostic conversions at the identical ASE geometry. The
        # independent native-to-ASE length round trip can change tiny residuals.
        for raw, actual in zip(atomic_report["hyperfine_nuclear"]["raw_paths"],
                               adapted["hyperfine_nuclear"]["raw_paths"]):
            for source, target in zip(raw["coefficient_diagnostics"], actual["coefficient_diagnostics"]):
                order = source["order"]
                for key in ("accumulation_bounds", "operator_poisson_residuals",
                            "operator_hermiticity_residuals"):
                    factor = Bohr**(-order - 3) if mode == "ase" else 1.
                    np.testing.assert_allclose(target[key], np.asarray(source[key]) * factor,
                                               rtol=1e-12, atol=1e-20)
                if order:
                    factor = Hartree / Bohr**order if mode == "ase" else 1.
                    np.testing.assert_allclose(target["stationarity_defect"],
                                               source["stationarity_defect"] * factor,
                                               rtol=1e-12, atol=1e-25)


def test_order_one_target_order_zero_factor_and_static_invariance(native_order_two):
    selected = [2, 3, 0]
    factors = [0., 1., -2., 3.]
    report = dft_rs.hyperfine(NUMBERS, POSITIONS, **SETTINGS, nuclear_order=1,
                             derivative_nuclei=selected, nuclear_g=factors)
    child = report["hyperfine_nuclear"]
    assert child["derivative_nuclei"] == selected and child["converged"]
    for part in ("contact", "dipolar", "total"):
        assert child[f"{part}_hessian"] is None
        assert child[f"{part}_hessian_per_g"] is None
        values = np.asarray(child[f"{part}_gradient_per_g"])
        scaled = np.asarray(child[f"{part}_gradient"])
        for index, atom in enumerate(selected):
            np.testing.assert_allclose(scaled[index], values[index] * factors[atom], rtol=0, atol=1e-10)
        np.testing.assert_allclose(values[2],
            native_order_two["hyperfine_nuclear"][f"{part}_gradient_per_g"][0], rtol=0, atol=1e-8)
    static_settings = dict(SETTINGS)
    del static_settings["response_tolerance"]
    del static_settings["response_max_iterations"]
    static = dft_rs.hyperfine(NUMBERS, POSITIONS, **static_settings)
    assert "hyperfine_nuclear" not in static
    for key in ("isotropic_coupling_per_g_mhz", "dipolar_coupling_per_g_mhz",
                "total_coupling_per_g_mhz"):
        np.testing.assert_allclose(static[key], report[key], rtol=0, atol=1e-8)


def test_both_cli_names_preserve_native_report(native_order_two, tmp_path):
    primary = Path(os.environ["DFT_RS_CLI"])
    alias = Path(os.environ.get("DFT_RS_CLI_ALIAS", str(primary.with_name("dft_rs_cli"))))
    xyz = tmp_path / "nuclear-a.xyz"
    xyz.write_text("4\nC1 nuclear A\n" + "\n".join(
        symbol + " " + " ".join(f"{value * Bohr:.17g}" for value in row)
        for symbol, row in zip(("C", "H", "H", "H"), POSITIONS)) + "\n", encoding="utf-8")
    args = ["hyperfine", str(xyz), "--format", "json"]
    for key, value in dict(SETTINGS, nuclear_order=2, derivative_nuclei=[0], nuclear_g=SIGNED_G).items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            assert value
            args.append(option)
        else:
            encoded = json.dumps(value, separators=(",", ":")) if isinstance(value, list) else str(value)
            args.append(option + "=" + encoded)
    for binary in (primary, alias):
        completed = subprocess.run([str(binary), *args], capture_output=True, text=True, check=True)
        report = json.loads(completed.stdout)
        check_report(report)
        compare_physical(native_order_two, report)
        assert completed.stderr.strip(), "CLI SCF progress must remain visible"


@pytest.mark.parametrize("controls", [dict(nuclear_order=True), dict(nuclear_order=-1),
    dict(nuclear_order=1.5), dict(nuclear_order=1, derivative_nuclei=[]),
    dict(nuclear_order=1, derivative_nuclei=[0, 0]),
    dict(nuclear_order=1, derivative_nuclei=[4])])
def test_malformed_nuclear_controls_fail_before_basis_setup(controls):
    with pytest.raises(ValueError):
        dft_rs.run_feature("hyperfine", NUMBERS, POSITIONS,
                           basis="invalid-test-basis", **controls)


@pytest.mark.parametrize("reference", ["roks", "rohf", "gks", "ghf"])
def test_static_reference_support_does_not_imply_nuclear_response(reference):
    with pytest.raises(NotImplementedError):
        dft_rs.run_feature("hyperfine", NUMBERS, POSITIONS, nuclear_order=1,
                           reference=reference, basis="invalid-test-basis")


def test_unaccepted_dft_nuclear_response_and_nonconvergence_are_not_success():
    with pytest.raises(NotImplementedError):
        dft_rs.run_feature("hyperfine", NUMBERS, POSITIONS,
                           **dict(SETTINGS, xc="pbe"), nuclear_order=1)
    with pytest.raises(RuntimeError, match="(?i)response|converg"):
        dft_rs.run_feature("hyperfine", NUMBERS, POSITIONS,
                           **dict(SETTINGS, response_max_iterations=1), nuclear_order=1)
