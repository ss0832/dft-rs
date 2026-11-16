# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Complex thermal operators and automatically generated molecular states."""
import json
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
import dft_rs
from dft_rs.units import HARTREE_IN_EV


def complex_options():
    zero = [[0.0, 0.0], [0.0, 0.0]]
    x = [[0.0, 1.0], [1.0, 0.0]]
    y = [[0.0, -1.0], [1.0, 0.0]]
    z = [[1.0, 0.0], [0.0, -1.0]]
    return dict(mcd_model="thermal-complex-manifold", temperature=4.2,
                bfield=[0.0, 1e-5, 0.0], mcd_thermal_hamiltonian=zero,
                mcd_thermal_hamiltonian_imag=zero,
                mcd_thermal_magnetic_x=x, mcd_thermal_magnetic_y=zero,
                mcd_thermal_magnetic_y_imag=y, mcd_thermal_magnetic_z=z,
                mcd_thermal_helicity_x=x, mcd_thermal_helicity_y=zero,
                mcd_thermal_helicity_y_imag=y, mcd_thermal_helicity_z=z)


def molecular_options():
    return dict(mcd_model="molecular-casci-soc", xc="hf", basis="sto-3g",
                temperature=4.2, bfield=[0.0, 0.0, 0.0],
                frequencies=[0.1, 0.2, 0.3], spectrum_width=0.002,
                conv_tol=1e-11, conv_tol_dp=1e-10)


def check_complex(r):
    assert r["converged"] and r["geometry_used"] is False
    assert r["electronic_states"] == 2
    np.testing.assert_allclose(r["mcd_thermal_signal"], [0, np.tanh(1e-5 / r["thermal_energy"]), 0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(r["mcd_total_thermal_tensor"], np.eye(3) / r["thermal_energy"], atol=1e-9, rtol=1e-13)
    assert len(r["mcd_thermal_populations"]) == 2


def check_molecular(r):
    assert r["converged"] and r["geometry_used"] is True
    assert r["determinant_count"] == 6
    assert "not TDDFT" in r["model"]
    np.testing.assert_allclose(r["mcd_cross_section"], np.zeros(3), atol=1e-11, rtol=0)
    energy = r["state_energies_relative"][0] + r["energy_origin"] + 1 / 1.4
    assert abs(energy - (-1.1372759437826976)) < 1e-9
    assert r["units"]["absorption_average"] == "bohr^2 per molecule"
    assert r["eigen_residual"] < 1e-9
    spin_squared = np.array(r["state_spin_squared"])
    assert abs(spin_squared[0]) < 1e-10
    assert abs(spin_squared.sum() - 6.0) < 1e-10
    assert "need not have a pure total spin" in r["spin_state_convention"]


@pytest.mark.parametrize("model", ["complex", "molecular"])
def test_native_common_ase_numerical_routes(model):
    options = complex_options() if model == "complex" else molecular_options()
    checker = check_complex if model == "complex" else check_molecular
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.4 * 0.529177210903]])
    reports = [dft_rs.mcd([1, 1], [[0, 0, 0], [0, 0, 1.4]], **options),
               dft_rs.run_feature("mcd", [1, 1], [[0, 0, 0], [0, 0, 1.4]], **options),
               dft_rs.DftRs(units="atomic", cpu_threads=1, memory_mb=64).get_mcd(atoms, **options)]
    for r in reports:
        checker(r)
    assert reports[-1]["resources"]["cpu_threads"] == 1


def test_complex_imaginary_energy_units_are_converted():
    options = complex_options()
    options["mcd_thermal_hamiltonian"] = [[0.0, 0.0], [0.0, 0.0]]
    options["mcd_thermal_hamiltonian_imag"] = [[0.0, -0.001], [0.001, 0.0]]
    base = dft_rs.run_feature("mcd", **options)
    physical = options | {"mcd_thermal_hamiltonian_imag": (np.array(options["mcd_thermal_hamiltonian_imag"]) * HARTREE_IN_EV).tolist()}
    r = dft_rs.DftRs(units="ase").get_mcd(Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]]), **physical)
    np.testing.assert_allclose(r["mcd_thermal_signal"], base["mcd_thermal_signal"], atol=1e-12, rtol=0)


def test_molecular_nested_units_and_linewidth_are_converted():
    options = molecular_options()
    base = dft_rs.run_feature("mcd", [1, 1], [[0, 0, 0], [0, 0, 1.4]], **options)
    physical = options | {
        "frequencies": (np.array(options["frequencies"]) * HARTREE_IN_EV).tolist(),
        "spectrum_width": options["spectrum_width"] * HARTREE_IN_EV,
    }
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.4 * 0.529177210903]])
    r = dft_rs.DftRs(units="ase").get_mcd(atoms, **physical)
    np.testing.assert_allclose(r["absorption_average"], np.array(base["absorption_average"]) * 0.529177210903**2, atol=1e-12, rtol=0)
    assert abs(r["spectrum_width"] - base["spectrum_width"] * HARTREE_IN_EV) < 1e-13
    for a, b in zip(r["transitions"], base["transitions"]):
        assert abs(a["energy"] - b["energy"] * HARTREE_IN_EV) < 1e-11
        da = np.array(a["dipole_real"]) + 1j * np.array(a["dipole_imag"])
        db = np.array(b["dipole_real"]) + 1j * np.array(b["dipole_imag"])
        np.testing.assert_allclose(np.outer(da, da.conj()), np.outer(db, db.conj()) * 0.529177210903**2, atol=1e-11, rtol=0)


@pytest.mark.parametrize("model", ["complex", "molecular"])
def test_both_cli_models(tmp_path, model):
    cli = os.environ.get("DFT_RS_CLI")
    if cli is None:
        pytest.skip("set DFT_RS_CLI for artifact-specific numerical acceptance")
    xyz = tmp_path / "hydrogen.xyz"
    xyz.write_text("2\nNo fitted scale or shifted energy\nH 0 0 0\nH 0 0 0.7408480952642\n", encoding="utf-8")
    options = complex_options() if model == "complex" else molecular_options()
    cmd = [cli, "mcd", str(xyz), "--format", "json", "--cpu-threads", "1", "--memory-mb", "64"]
    for k, v in options.items():
        cmd.extend(["--" + k.replace("_", "-"), v if isinstance(v, str) else json.dumps(v)])
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    r = json.loads(result.stdout)
    if model == "complex":
        check_complex(r)
    else:
        # XYZ's explicitly rounded Angstrom coordinate is compared with
        # the corresponding atomic-coordinate public route, not a fitted
        # adjustment to the theoretical energy.
        native = dft_rs.run_feature("mcd", [1, 1], [[0, 0, 0], [0, 0, 0.7408480952642 / 0.529177210903]], **options)
        np.testing.assert_allclose(r["state_energies_relative"], native["state_energies_relative"], atol=1e-10, rtol=0)
        assert "scf" in result.stderr.lower() or "iteration" in result.stderr.lower()


@pytest.mark.parametrize("bad", [
    {"mcd_thermal_hamiltonian_imag": [[0, 0.1], [0.1, 0]]},
    {"temperature": 0}, {"temperature": -1},
    {"mcd_thermal_helicity_y_imag": [[0]]},
])
def test_invalid_complex_input_fails_clearly(bad):
    with pytest.raises(ValueError):
        dft_rs.run_feature("mcd", **(complex_options() | bad))


@pytest.mark.parametrize("bad", [
    {"mcd_active_orbitals": 0}, {"mcd_frozen_core": 2},
    {"spectrum_width": 0}, {"temperature": 0},
    {"frequencies": [0.2, 0.1]}, {"mcd_orientation": "unknown"},
    {"mcd_orientation": "isotropic", "bfield": [1e-5, 0, 0]},
])
def test_invalid_automatic_model_fails_clearly(bad):
    with pytest.raises((ValueError, RuntimeError)):
        dft_rs.run_feature("mcd", [1, 1], [[0, 0, 0], [0, 0, 1.4]], **(molecular_options() | bad))


@pytest.mark.parametrize("surface", ["native", "ase", "cli"])
def test_london_linear_spectrum_is_independent_of_gauge_origin(tmp_path, surface):
    positions = np.array([[0.0, 0.0, 0.0], [0.2, -0.3, 1.8]])
    atoms = Atoms("OH", positions=positions * 0.529177210903)
    options = molecular_options() | dict(
        mcd_magnetic_basis="london-linear-frozen-soc", multiplicity=2,
        unrestricted=True, mcd_frozen_core=1, mcd_active_orbitals=4,
        max_states=16, bfield=[7e-6, -4e-6, 1.1e-5],
        frequencies=[0.20, 0.22, 0.24, 0.25])
    cli = os.environ.get("DFT_RS_CLI")
    if surface == "cli" and cli is None:
        pytest.skip("set DFT_RS_CLI for artifact-specific numerical acceptance")
    reports = []
    for origin in [[0, 0, 0], [1.3, -0.8, 0.4], [-2.1, 0.7, 1.8]]:
        controls = options | dict(gauge_origin=origin)
        if surface == "native":
            report = dft_rs.run_feature("mcd", [8, 1], positions.tolist(), **controls)
        elif surface == "ase":
            report = dft_rs.DftRs(units="atomic", cpu_threads=1).get_mcd(atoms, **controls)
        else:
            xyz = tmp_path / "radical.xyz"
            xyz.write_text("2\nLondon first-order model\n" + "\n".join(
                symbol + " " + " ".join(map(str, row))
                for symbol, row in zip(["O", "H"], atoms.positions)) + "\n", encoding="utf-8")
            command = [cli, "mcd", str(xyz), "--format", "json", "--cpu-threads", "1"]
            for key, value in controls.items():
                command.extend(["--" + key.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)])
            report = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
        assert report["mcd_magnetic_basis"] == "london-linear-frozen-soc"
        assert "frozen" in report["soc_field_treatment"]
        assert report["magnetic_field_order"] == 1
        assert max(abs(np.array(report["mcd_cross_section"]))) > 1e-12
        reports.append(report)
    for report in reports[1:]:
        for key in ["absorption_left", "absorption_right", "mcd_cross_section",
                    "magnetic_operators_real", "magnetic_operators_imag",
                    "electric_magnetic_derivatives_real", "electric_magnetic_derivatives_imag"]:
            np.testing.assert_allclose(report[key], reports[0][key], atol=1e-10, rtol=0)
