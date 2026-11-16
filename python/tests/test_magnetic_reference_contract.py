# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Unconnected magnetic references must fail instead of changing SCF models."""
import json
import os
import re
import subprocess

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM


@pytest.mark.parametrize("reference", ["roks", "rohf", "gks", "ghf"])
@pytest.mark.parametrize("feature", ["magnetizability", "hypermagnetizability", "cotton-mouton"])
@pytest.mark.parametrize("surface", ["native", "ase", "cli"])
def test_unconnected_magnetic_reference_is_never_replaced(tmp_path, reference, feature, surface):
    # For this OH case, actual ROHF and UHF reference energies differ by
    # 1.108 mEh. Returning the UHF magnetic result is a physical model error.
    xyz = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8324]]
    controls = dict(xc="hf", basis="sto-3g", multiplicity=2,
                    unrestricted=True, reference=reference, cpu_threads=1,
                    memory_mb=512)
    message = "magnetic response with reference="
    if feature == "magnetizability" and reference in ("gks", "ghf"):
        controls["gauge"] = "london"
        message = "GKS magnetizability with gauge=london"
    if surface == "native":
        with pytest.raises(NotImplementedError, match=message):
            dft_rs.run_feature(feature, [8, 1], xyz, **controls)
    elif surface == "ase":
        atoms = Atoms("OH", positions=np.array(xyz) * 0.529177210903)
        with pytest.raises(NotImplementedError, match=message):
            dft_rs.DftRs(units="atomic").get_feature_report(feature, atoms, **controls)
    else:
        cli = os.environ.get("DFT_RS_CLI")
        if cli is None:
            pytest.skip("set DFT_RS_CLI for linked-artifact acceptance")
        path = tmp_path / "hydroxyl.xyz"
        path.write_text("2\nExplicit magnetic reference contract\nO 0 0 0\nH 0 0 0.96966\n")
        command = [cli, feature, str(path), "--format", "json"]
        for key, value in controls.items():
            command.extend(["--" + key.replace("_", "-"),
                            value if isinstance(value, str) else json.dumps(value)])
        result = subprocess.run(command, text=True, capture_output=True)
        assert result.returncode != 0
        assert message in result.stderr + result.stdout


def test_common_origin_gks_orbital_magnetizability_on_all_surfaces(tmp_path):
    numbers = [7, 1, 1, 1]
    positions = np.array([[0.13, -0.27, 0.19], [1.78, 0.34, -0.08],
                          [-0.49, 1.65, 0.41], [0.31, -0.78, 1.91]])
    controls = dict(reference="gks", xc="hsesol", basis="sto-3g", charge=1,
                    multiplicity=2, grid_level=0, grid_weight_cutoff=0.0,
                    conv_tol=1e-13, conv_tol_dp=1e-12, conv_tol_grad=1e-12,
                    max_iter=300, cpu_threads=1)
    with pytest.raises(ValueError, match="explicit gauge=common-origin"):
        dft_rs.magnetizability(numbers, positions, **controls)
    controls.update(gauge="common-origin", gauge_origin=[0.08, 0.01, 0.23])
    native = dft_rs.magnetizability(numbers, positions, **controls)
    xyz = positions * BOHR_IN_ANGSTROM
    atoms = Atoms(numbers=numbers, positions=xyz)
    adapted = dft_rs.DftRs(units="atomic").get_magnetizability(atoms, **controls)
    path = tmp_path / "ammonia-cation.xyz"
    path.write_text("4\nCommon-origin GKS orbital magnetizability\n" + "".join(
        f"{z} {x:.17g} {y:.17g} {w:.17g}\n" for z, (x, y, w) in zip(numbers, xyz)
    ), encoding="utf-8")
    cli = os.environ["DFT_RS_CLI"]
    command = [cli, "magnetizability", str(path), "--format", "json"]
    for key, value in controls.items():
        command.extend(["--" + key.replace("_", "-"),
                        value if isinstance(value, str) else json.dumps(value)])
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    assert re.search(r"\[spinor-scf\].*iter.*E=.*residual=", result.stderr)
    linked = json.loads(result.stdout)
    numeric = ("magnetizability", "diamagnetic", "paramagnetic", "isotropic",
               "isotropic_cgs_ppm", "gauge_origin", "response_residuals",
               "raw_asymmetry", "magnetizability_imaginary_defect", "reference_fock_defect")
    differences = {}
    for interface, report in (("native", native), ("ASE", adapted), ("CLI", linked)):
        assert report["gauge"] == "common-origin"
        assert report["reference_type"] == "gks"
        assert report["response_converged"]
        assert "gauge_origin" in report["warnings"][0]
        differences[interface] = {}
        for key in numeric:
            np.testing.assert_allclose(report[key], native[key], atol=2e-10, rtol=0)
            assert report["units"][key] == native["units"][key]
            differences[interface][key] = float(np.max(np.abs(
                np.asarray(report[key]) - np.asarray(native[key])
            )))
    print("GKS orbital magnetic public values", {key: native[key] for key in numeric})
    print("GKS orbital magnetic public differences", differences)


@pytest.mark.parametrize("feature", ["hyperfine", "g-tensor"])
@pytest.mark.parametrize("reference", ["roks", "rohf", "gks", "ghf", "RoKs", "GhF"])
@pytest.mark.parametrize("surface", ["generic", "native", "ase", "cli"])
def test_dedicated_static_epr_reference_reaches_basis_setup_without_fallback(tmp_path, feature, reference, surface):
    # These selectors now have dedicated static providers. A missing basis
    # must reach their normal setup, rather than the old blanket rejection.
    # This does not assert numerical or nuclear-derivative acceptance.
    controls = dict(reference=reference, basis="missing-epr-preflight-basis",
                    xc="hf", multiplicity=2, unrestricted=True, cpu_threads=1)
    if surface == "cli":
        cli = os.environ.get("DFT_RS_CLI")
        if cli is None:
            pytest.skip("set DFT_RS_CLI for linked-artifact acceptance")
        path = tmp_path / "hydrogen.xyz"
        path.write_text("1\nEPR reference preflight\nH 0 0 0\n", encoding="utf-8")
        result = subprocess.run(
            [cli, feature, str(path), "--format", "json", "--reference", reference,
             "--basis", controls["basis"], "--xc", "hf", "--multiplicity", "2",
             "--unrestricted", "--cpu-threads", "1"],
            text=True, capture_output=True, check=False,
        )
        assert result.returncode != 0
        assert controls["basis"] in result.stderr
        assert "cannot replace the requested reference" not in result.stderr
        assert "[scf] iter" not in result.stderr
        assert result.stdout == ""
        return
    with pytest.raises(ValueError, match=controls["basis"]):
        if surface == "generic":
            dft_rs.run_feature(feature, [1], [[0., 0., 0.]], **controls)
        elif surface == "native":
            getattr(dft_rs, feature.replace("-", "_"))([1], [[0., 0., 0.]], **controls)
        else:
            getter = getattr(dft_rs.DftRs(units="atomic"), "get_" + feature.replace("-", "_"))
            getter(Atoms("H", positions=[[0., 0., 0.]]), **controls)


@pytest.mark.parametrize("feature", ["hyperfine", "g-tensor"])
@pytest.mark.parametrize("reference", ["rkos", "", 42, True, []])
@pytest.mark.parametrize("surface", ["generic", "ase", "cli"])
def test_epr_invalid_reference_selectors_are_not_ignored(tmp_path, feature, reference, surface):
    controls = dict(reference=reference, basis="missing-epr-preflight-basis", cpu_threads=1)
    if surface == "cli":
        executable = os.environ.get("DFT_RS_CLI")
        assert executable, "DFT_RS_CLI is required for linked-artifact acceptance"
        path = tmp_path / "preflight.xyz"
        path.write_text("1\nReference preflight\nH 0 0 0\n", encoding="utf-8")
        result = subprocess.run([executable, feature, str(path), "--format", "json",
            "--reference", json.dumps(reference), "--basis", controls["basis"],
            "--cpu-threads", "1"], capture_output=True, text=True, check=False)
        assert result.returncode != 0
        assert "reference" in result.stderr.lower()
        assert controls["basis"] not in result.stderr
        assert "[scf] iter" not in result.stderr
        return
    with pytest.raises(ValueError, match="reference") as raised:
        if surface == "generic":
            dft_rs.run_feature(feature, [1], [[0., 0., 0.]], **controls)
        else:
            dft_rs.DftRs(units="atomic").get_feature_report(
                feature, Atoms("H", positions=[[0., 0., 0.]]), **controls)
    assert controls["basis"] not in str(raised.value)


@pytest.mark.parametrize("surface", ["generic", "ase"])
@pytest.mark.parametrize(("feature", "controls", "exception"), [
    ("unavailable-calculation", {}, NotImplementedError),
    ("dynamic-beta", {}, NotImplementedError),  # The dedicated function remains available.
    ("feature-registry", {"xc": "hf"}, NotImplementedError),
    ("feature-registry", {"grid_levl": 1}, ValueError),
    ("spin-susceptibility", {"reference": "gks"}, NotImplementedError),
    ("spin-susceptibility", {"reference": "rohf"}, NotImplementedError),
    ("spin-susceptibility", {"reference": "rkos"}, ValueError),
    ("multicomponent", {"quantum_lmax": 4}, NotImplementedError),
    ("multicomponent", {"species_derivative_order": 3}, NotImplementedError),
    ("gamma", {"xc": "hf", "tensor_assembly": "variational"}, NotImplementedError),
    ("beta", {"xc": "hf", "tensor_assembly": "energy"}, NotImplementedError),
    ("beta", {"xc": "b2plyp", "tensor_assembly": "variational"}, NotImplementedError),
    ("beta", {"xc": "hf", "tensor_assembly": "unavailable-assembly"}, ValueError),
    ("energy", {"grid_level": "fine"}, ValueError),
])
def test_common_capability_errors_preserve_their_exception_type(surface, feature, controls, exception):
    options = {"cpu_threads": 1, "basis": "missing-capability-test-basis", **controls}
    # feature-registry reads no molecular controls. Omit the sentinel so this
    # also checks the classification of the explicitly supplied unused key.
    if feature == "feature-registry" and surface == "generic":
        options.pop("basis")
    with pytest.raises(exception) as raised:
        if surface == "generic":
            dft_rs.run_feature(feature, [1, 1], [[0., 0., -.7], [0., 0., .7]], **options)
        else:
            atoms = Atoms("H2", positions=[[0., 0., -.37], [0., 0., .37]])
            dft_rs.DftRs(units="atomic").get_feature_report(feature, atoms, **options)
    assert "missing-capability-test-basis" not in str(raised.value)
    if exception is NotImplementedError:
        assert "not implemented" in str(raised.value)
        if "quantum_lmax" in controls:
            assert "quantum_lmax=4" in str(raised.value)
        if "species_derivative_order" in controls:
            assert "derivative order 3" in str(raised.value)


@pytest.mark.parametrize("feature", ["spin-susceptibility", "ecd-spectrum"])
@pytest.mark.parametrize("reference", ["roks", "rohf", "gks", "ghf"])
def test_other_unconnected_references_remain_available_for_rejection(feature, reference):
    required = {"broadening_ev": 0.1, "td_method": "tda"} if feature == "ecd-spectrum" else {}
    with pytest.raises(NotImplementedError, match="reference") as raised:
        dft_rs.run_feature(feature, [1, 1], [[0., 0., -.7], [0., 0., .7]],
                           reference=reference, basis="missing-capability-test-basis", cpu_threads=1,
                           **required)
    assert "missing-capability-test-basis" not in str(raised.value)


@pytest.mark.parametrize("reference", ["roks", "rohf", "gks", "ghf"])
def test_orbital_nmr_reference_reaches_requested_basis(reference):
    basis = "missing-nmr-preflight-basis"
    with pytest.raises(ValueError, match=basis):
        dft_rs.nmr_shielding([8, 1], [[0., 0., 0.], [0., 0., 1.83]], xc="hf",
                             basis=basis, reference=reference, multiplicity=2,
                             gauge="common-origin", cpu_threads=1)


def test_unavailable_cli_command_reports_not_implemented():
    cli = os.environ.get("DFT_RS_CLI")
    if cli is None:
        pytest.skip("set DFT_RS_CLI for linked-artifact acceptance")
    result = subprocess.run([cli, "unavailable-calculation"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "not implemented" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize(("controls", "message"), [
    ({"reference": "gks"}, "not implemented"),
    ({"reference": "rkos"}, "unknown reference"),
])
def test_cli_capability_errors_fail_before_basis_setup(tmp_path, controls, message):
    cli = os.environ.get("DFT_RS_CLI")
    if cli is None:
        pytest.skip("set DFT_RS_CLI for linked-artifact acceptance")
    xyz = tmp_path / "dihydrogen.xyz"
    xyz.write_text("2\nCapability dispatch contract\nH 0 0 -.37\nH 0 0 .37\n", encoding="utf-8")
    command = [cli, "spin-susceptibility", str(xyz), "--format", "json",
               "--basis", "missing-capability-test-basis", "--cpu-threads", "1"]
    for key, value in controls.items():
        command.extend(["--" + key.replace("_", "-"), value])
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0
    assert message in result.stderr
    assert "missing-capability-test-basis" not in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("surface", ["generic", "ase"])
@pytest.mark.parametrize(("order", "shear"), [(2, True), (3, False)])
def test_unavailable_lattice_response_orders_raise_not_implemented(surface, order, shear):
    controls = dict(gruneisen_order=order, gruneisen_shear=shear,
                    basis="missing-capability-test-basis", cpu_threads=1)
    with pytest.raises(NotImplementedError, match="not implemented") as raised:
        if surface == "generic":
            dft_rs.run_feature("mode-gruneisen", [1, 1], [[0., 0., -.7], [0., 0., .7]],
                               cell=np.eye(3).tolist(), pbc=[True, True, True], **controls)
        else:
            atoms = Atoms("H2", positions=[[0., 0., -.37], [0., 0., .37]],
                          cell=np.eye(3) * 5, pbc=True)
            dft_rs.DftRs(units="atomic").get_feature_report("mode-gruneisen", atoms, **controls)
    assert "missing-capability-test-basis" not in str(raised.value)
    assert "mode Gruneisen" in str(raised.value)


def test_cli_consumed_geometry_is_transferred_to_system_once(tmp_path):
    cli = os.environ.get("DFT_RS_CLI")
    if cli is None:
        pytest.skip("set DFT_RS_CLI for linked-artifact acceptance")
    xyz = tmp_path / "dihydrogen.xyz"
    xyz.write_text("2\nIsolated geometry transfer contract\nH 0 0 -.37\nH 0 0 .37\n", encoding="utf-8")
    command = [cli, "energy", str(xyz), "--format", "json", "--xc", "hf", "--basis", "sto-3g",
               "--cpu-threads", "1", "--memory-mb", "128"]
    ordinary = subprocess.run(command, capture_output=True, text=True, check=True)
    inactive_cell = subprocess.run(command + ["--cell", "5", "--pbc", "0,0,0"],
                                   capture_output=True, text=True, check=True)
    first, second = json.loads(ordinary.stdout), json.loads(inactive_cell.stdout)
    assert first["converged"] and second["converged"]
    assert first["energy"] == pytest.approx(second["energy"], rel=0, abs=1e-14)
