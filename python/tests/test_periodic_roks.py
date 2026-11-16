# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Actual common-orbital periodic reference, not unrestricted substitution."""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM


def fixture():
    return [1, 1, 1], [[.13, -.21, .07], [1.91, .20, -.19], [.4, 1.7, .33]], [
        [8.5, 0., 0.], [.31, 9.3, 0.], [.27, -.19, 10.1]]


def options(xc="pbe"):
    return dict(reference="roks", xc=xc, basis="sto-3g", dispersion="none", multiplicity=2,
                k_mesh=[1, 1, 1], overlap_cutoff=6., xc_ao_cutoff=16., grid_level=1, grid_shell=1,
                grid_weight_cutoff=0., guess="core", conv_tol=1e-13,
                conv_tol_grad=1e-10, conv_tol_dp=1e-11, max_iter=300,
                damping=0., no_adiis=True, cpu_threads=1, memory_mb=512, forces=True, stress=True)


def save_surface(xc, surface, report):
    directory = os.environ.get("DFT_RS_PUBLIC_RAW_DIR")
    if directory is not None:
        path = Path(directory)/f"{xc}-{surface}.json"
        path.write_text(json.dumps(report, indent=2)+"\n")


def validate(report):
    assert report["converged"] and not report["unrestricted"]
    assert report["xc_ao_cutoff"] == 16.
    assert report["reference_type"] in ("roks", "rohf")
    assert report["nuclear_response_available"]
    assert report["nuclear_response_order"] == 1
    assert not report["nuclear_hessian_available"]
    for key in ("forces", "gradient", "stress", "strain_derivative"):
        assert np.asarray(report[key]).shape == (3, 3)
        assert np.isfinite(report[key]).all()
    np.testing.assert_array_equal(report["forces"], -np.asarray(report["gradient"]))
    assert report["translation_residual"] < 1e-6
    assert report["units"]["gradient"] == "hartree/bohr"
    assert "hartree/bohr^3" in report["units"]["stress"]
    assert (report["n_alpha"], report["n_beta"]) == (2, 1)
    assert report["orbital_residual"] < 1e-10 and report["density_change"] < 1e-11
    assert report["energy_change"] < 1e-13 and report["metric_defect"] < 1e-12
    assert max(np.abs(report["particle_number_errors"])) < 1e-12
    assert max(report["idempotency_defects"]) < 1e-12
    assert report["closed_subspace_defect"] < 1e-12
    for key in ("common_orbitals", "density_alpha", "density_beta", "fock_alpha", "fock_beta"):
        assert np.asarray(report[key]).shape == (3, 3)
        assert np.isfinite(report[key]).all()


def test_disabled_adiis_omits_only_automatic_threshold(monkeypatch):
    captured = []

    def record(feature, numbers, positions, **controls):
        captured.append(controls)
        return {"energy": 0.0, "units": {"energy": "hartree"}}

    monkeypatch.setattr(dft_rs.native, "run_feature", record)
    numbers, positions, cell = fixture()
    atoms = Atoms(numbers, positions=np.asarray(positions)*BOHR_IN_ANGSTROM,
                  cell=np.asarray(cell)*BOHR_IN_ANGSTROM, pbc=True)
    calc = dft_rs.DftRs(units="atomic")
    calc._periodic_report("periodic-energy", atoms, no_adiis=True)
    assert captured[-1]["no_adiis"] is True
    assert "adiis_until_error" not in captured[-1]
    calc._periodic_report("periodic-energy", atoms, no_adiis=True, adiis_until_error=.03125)
    assert captured[-1]["adiis_until_error"] == .03125


@pytest.mark.parametrize("xc", ["hf", "pbe"])
def test_actual_common_orbital_first_derivatives_across_public_surfaces(xc, tmp_path):
    numbers, positions, cell = fixture()
    parameters = options(xc)
    native = dft_rs.native.run_feature("periodic-energy", numbers, positions, cell=cell, **parameters)
    save_surface(xc, "native", native)
    validate(native)
    atoms = Atoms(numbers, positions=np.asarray(positions)*BOHR_IN_ANGSTROM,
                  cell=np.asarray(cell)*BOHR_IN_ANGSTROM, pbc=True)
    ase = dft_rs.DftRs(units="atomic", damping=0., use_adiis=False,
                       adiis_until_error=None).get_periodic_energy_report(atoms, **parameters)
    save_surface(xc, "ase", ase)
    validate(ase)
    xyz = tmp_path / "cell.xyz"
    xyz.write_text("3\nGamma common-orbital fixture\n" + "\n".join(
        "H " + " ".join(str(x*BOHR_IN_ANGSTROM) for x in position) for position in positions) + "\n")
    rows = ";".join(",".join(str(x*BOHR_IN_ANGSTROM) for x in row) for row in cell)
    command = [os.environ["DFT_RS_CLI"], "periodic-energy", str(xyz), "--cell", rows,
               "--pbc", "1,1,1", "--format", "json"]
    for key, value in parameters.items():
        if isinstance(value, bool) and value:
            command.append("--"+key.replace("_", "-"))
            continue
        command.extend(["--"+key.replace("_", "-"),
                        json.dumps(value) if isinstance(value, (list, bool)) else str(value)])
    process = subprocess.run(command,
                             text=True, capture_output=True, check=False)
    assert process.returncode == 0, process.stderr
    assert "[pbc-roks] iter" in process.stderr
    cli = json.loads(process.stdout)
    save_surface(xc, "cli", cli)
    validate(cli)
    alias_name = "dft_rs_cli.exe" if os.name == "nt" else "dft_rs_cli"
    alias = os.environ.get("DFT_RS_CLI_ALIAS", str(Path(command[0]).with_name(alias_name)))
    alias_process = subprocess.run([alias, *command[1:]], text=True, capture_output=True, check=False)
    assert alias_process.returncode == 0, alias_process.stderr
    assert "[pbc-roks] iter" in alias_process.stderr
    alias = json.loads(alias_process.stdout)
    save_surface(xc, "alias", alias)
    validate(alias)
    for surface, report in (("native", native), ("ASE", ase), ("CLI", cli), ("alias", alias)):
        np.testing.assert_allclose(report["energy"], native["energy"], rtol=0., atol=1e-10)
        np.testing.assert_allclose(report["density_alpha"], native["density_alpha"], rtol=0., atol=1e-10)
        np.testing.assert_allclose(report["density_beta"], native["density_beta"], rtol=0., atol=1e-10)
        for key in ("gradient", "forces", "stress", "strain_derivative"):
            np.testing.assert_allclose(report[key], native[key], rtol=0., atol=1e-11)
        print(json.dumps(dict(xc=xc, surface=surface, report=report)), flush=True)


@pytest.mark.parametrize("extra", [dict(k_mesh=[2, 1, 1]), dict(electronic_temperature=300.),
                                   dict(unrestricted=True),
                                   dict(reference="rohf"), dict(xc="b2plyp"), dict(pt2_same_spin=.37)])
def test_unimplemented_constrained_response_or_parent_requests_are_not_substituted(extra):
    numbers, positions, cell = fixture()
    parameters = options()
    parameters.update(extra)
    with pytest.raises(NotImplementedError):
        dft_rs.native.run_feature("periodic-energy", numbers, positions, cell=cell, **parameters)


@pytest.mark.parametrize("value", [0., -1.])
@pytest.mark.parametrize("surface", ["native", "CLI"])
def test_invalid_ao_range_is_rejected_before_scf(value, surface, tmp_path):
    numbers, positions, cell = fixture()
    parameters = options("hf")
    parameters["xc_ao_cutoff"] = value
    parameters["basis"] = "deliberately-missing-basis"
    if surface == "native":
        with pytest.raises(ValueError, match="xc_ao_cutoff must be finite and positive"):
            dft_rs.native.run_feature("periodic-energy", numbers, positions, cell=cell, **parameters)
        return
    xyz = tmp_path / "range.xyz"
    xyz.write_text("3\nAO range preflight\n"+"\n".join("H "+" ".join(str(x*BOHR_IN_ANGSTROM) for x in row) for row in positions)+"\n")
    rows = ";".join(",".join(str(x*BOHR_IN_ANGSTROM) for x in row) for row in cell)
    command = [os.environ["DFT_RS_CLI"], "periodic-energy", str(xyz), "--cell", rows, "--pbc", "1,1,1", "--reference", "roks", "--basis", "deliberately-missing-basis", "--xc-ao-cutoff", str(value), "--format", "json"]
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    assert process.returncode != 0
    assert "xc_ao_cutoff must be finite and positive" in process.stderr
    assert "[pbc-roks] iter" not in process.stderr


@pytest.mark.parametrize("mode,value", [("atomic",16.), ("ase",16.*BOHR_IN_ANGSTROM)])
def test_calculator_parameter_range_reaches_driver_in_bohr(monkeypatch, mode, value):
    registry = dft_rs.native.feature_registry()
    monkeypatch.setattr(dft_rs.native, "feature_registry", lambda: registry)
    captured = []
    def record(feature, numbers, positions, **controls):
        captured.append(controls)
        return {"energy": 0., "units": {"energy": "hartree"}}
    monkeypatch.setattr(dft_rs.native, "run_feature", record)
    numbers, positions, cell = fixture()
    atoms = Atoms(numbers, positions=np.asarray(positions)*BOHR_IN_ANGSTROM, cell=np.asarray(cell)*BOHR_IN_ANGSTROM, pbc=True)
    calculator = dft_rs.DftRs(units=mode, xc_ao_cutoff=value)
    calculator.get_periodic_energy_report(atoms)
    assert captured[-1]["xc_ao_cutoff"] == pytest.approx(16., rel=0., abs=1e-13)
