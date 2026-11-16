# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Public direct spin-spin D contracts, distinct from SOC and total ZFS.

These are adapter/parity tests. Independent derivative fits belong to the
physics acceptance suite; no finite difference is used to produce a report.
"""
import json
import os
from pathlib import Path
import subprocess

import dft_rs
import numpy as np
import pytest
from ase import Atoms
from dft_rs import native
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree


NUMBERS = [1, 1]
POSITIONS = np.array([[.13, -.21, .17], [2.05, .25, .38]])
SETTINGS = dict(xc="hf", reference="uhf", basis="sto-3g", multiplicity=3,
                unrestricted=True, conv_tol=1e-14, conv_tol_dp=1e-13,
                conv_tol_grad=1e-12, response_tolerance=1e-12,
                cpu_threads=1, memory_mb=256)


@pytest.fixture(scope="module")
def nuclear_two():
    return dft_rs.spin_spin_zfs(NUMBERS, POSITIONS, nuclear_order=2, **SETTINGS)


def check(report, order):
    assert report["converged"] and report["reference"] == "uhf"
    assert report["soc_included"] is False
    assert report["is_total_zfs"] is False
    assert report["spin_projection_included"] is False
    assert report["nuclear_order"] == order
    assert report["spin"] == 1.
    assert report["s_squared"] == pytest.approx(2., abs=1e-12)
    assert "S.D.S" in report["hamiltonian_convention"]
    assert report["report_workspace_bytes"] > 0
    assert report["resources"]["cpu_threads"] == 1
    for key, shape in (("d_tensor", (3, 3)), ("d_gradient", (6, 3, 3)),
                       ("d_hessian", (6, 6, 3, 3))):
        requested = key == "d_tensor" or order >= (2 if key == "d_hessian" else 1)
        if requested:
            assert np.asarray(report[key]).shape == shape
            assert np.isfinite(report[key]).all()
        else:
            assert report[key] is None
    for path in report["raw_paths"]:
        assert path["converged"] and np.isfinite(path["residuals"]).all()
        for row in path["coefficient_diagnostics"]:
            if row["order"] == 0:
                assert row["stationarity_defect"] is None
                assert row["coefficient_equation"] is None
            else:
                assert np.isfinite(row["stationarity_defect"])
                assert row["coefficient_equation"] is not None


@pytest.mark.parametrize("order", [0, 1])
def test_unrequested_derivatives_are_null(order):
    report = dft_rs.spin_spin_zfs(NUMBERS, POSITIONS, nuclear_order=order, **SETTINGS)
    check(report, order)


def test_native_and_ase_units_and_layout(nuclear_two):
    check(nuclear_two, 2)
    atoms = Atoms(numbers=NUMBERS, positions=POSITIONS * Bohr)
    for mode in ("atomic", "ase"):
        report = dft_rs.DftRs(units=mode).get_spin_spin_zfs(atoms, nuclear_order=2, **SETTINGS)
        check(report, 2)
        for order, key in enumerate(("d_tensor", "d_gradient", "d_hessian")):
            factor = Hartree / Bohr**order if mode == "ase" else 1.
            np.testing.assert_allclose(report[key], np.asarray(nuclear_two[key]) * factor,
                                       rtol=0, atol=1e-12)


def test_both_cli_names_share_the_report(nuclear_two, tmp_path):
    primary = Path(os.environ["DFT_RS_CLI"])
    alias = Path(os.environ.get("DFT_RS_CLI_ALIAS", str(primary.with_name("dft_rs_cli"))))
    geometry = tmp_path / "triplet-pair.xyz"
    geometry.write_text("2\nDirect spin-spin API fixture\n" + "\n".join(
        "H " + " ".join(f"{v*Bohr:.17g}" for v in row) for row in POSITIONS) + "\n")
    args = ["spin-spin-zfs", str(geometry), "--format", "json", "--nuclear-order", "2"]
    for key, value in SETTINGS.items():
        args += ["--" + key.replace("_", "-") + "=" +
                 (value if isinstance(value, str) else json.dumps(value))]
    for binary in (primary, alias):
        run = subprocess.run([str(binary), *args], capture_output=True, text=True, check=True)
        report = json.loads(run.stdout)
        check(report, 2)
        for key in ("d_tensor", "d_gradient", "d_hessian"):
            np.testing.assert_allclose(report[key], nuclear_two[key], rtol=0, atol=1e-12)
        assert "[scf]" in run.stderr.lower()


@pytest.mark.parametrize("order", [True, 1., -1, "2"])
def test_nuclear_order_has_strict_integer_type(order):
    with pytest.raises(ValueError, match="nuclear_order"):
        dft_rs.spin_spin_zfs(NUMBERS, POSITIONS, nuclear_order=order,
                            basis="missing-spin-spin-basis", multiplicity=3)


@pytest.mark.parametrize("reference", ["auto", "rhf", "rks", "uks", "rohf", "roks", "ghf", "gks"])
def test_unsupported_reference_never_downgrades(reference):
    with pytest.raises(NotImplementedError, match="reference"):
        dft_rs.spin_spin_zfs(NUMBERS, POSITIONS, reference=reference, nuclear_order=1,
                            basis="missing-spin-spin-basis", multiplicity=3)


@pytest.mark.parametrize("controls", [dict(nuclear_order=3), dict(xc="pbe"), dict(xc="pbe0"),
    dict(xc="b2plyp"), dict(ecp="unavailable-ecp"), dict(electronic_temperature=300),
    dict(unrestricted=False)])
def test_unsupported_physical_combinations_fail_before_basis(controls):
    with pytest.raises(NotImplementedError) as raised:
        dft_rs.spin_spin_zfs(NUMBERS, POSITIONS, basis="missing-spin-spin-basis",
                            multiplicity=3, **controls)
    assert "missing-spin-spin-basis" not in str(raised.value)


def test_log_and_small_budget_failure(tmp_path):
    path = tmp_path / "spin-spin.log"
    report = dft_rs.spin_spin_zfs(NUMBERS, POSITIONS, **SETTINGS,
                                log_file=path, log_mode="file", nuclear_order=0)
    check(report, 0)
    text = path.read_text()
    assert "[scf]" in text.lower() and "run" in text.lower()
    with pytest.raises((MemoryError, RuntimeError), match="(?i)memory|budget|spin-spin report"):
        dft_rs.spin_spin_zfs([1]*4, [[0., 0., 0.]]*4, nuclear_order=2,
                            multiplicity=3, charge=2, basis="missing-budget-basis",
                            cpu_threads=1, memory_mb=1)


def test_unconverged_scf_is_an_error_with_partial_log(tmp_path):
    path = tmp_path / "failed-spin-spin.log"
    with pytest.raises(RuntimeError, match="(?i)converg"):
        dft_rs.spin_spin_zfs([1]*4, [[.13, -.21, .17], [2.05, .25, .38],
            [-.65, 1.76, -.44], [-1.19, -1.48, .85]], **SETTINGS,
            charge=2, max_iter=1, nuclear_order=1, log_file=path)
    assert path.is_file() and path.read_text().strip()


def test_frontend_native_forwarding_uses_one_common_envelope(monkeypatch):
    calls, envelopes = [], []

    def backend(feature, numbers, positions, **controls):
        calls.append((feature, numbers, positions, controls))
        return {"converged": True, "units": {}}

    def envelope(function, args, kwargs, **limits):
        envelopes.append(limits)
        return function(*args, **kwargs)

    monkeypatch.setattr(native, "run_feature", backend)
    monkeypatch.setattr(native._core, "_with_resources", envelope)
    native.spin_spin_zfs(NUMBERS, POSITIONS, nuclear_order=2, reference="uhf",
                         cpu_threads=2, memory_mb=64, log_file="selected.log")
    assert len(calls) == len(envelopes) == 1
    assert calls[0][0] == "spin-spin-zfs"
    assert calls[0][3] == {"nuclear_order": 2, "reference": "uhf"}
    assert envelopes[0] == {"cpu_threads": 2, "memory_mb": 64, "log_file": "selected.log"}
