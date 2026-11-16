# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, periodic_thermal_response
from dft_rs.native import BOHR_PER_ANGSTROM


GEOMETRY = np.array([[0, 0, -0.7], [0, 0, 0.7]])
CELL = np.eye(3) * 9.0
SCF = dict(xc="svwn5", overlap_cutoff=7.0, grid_level=0, grid_shell=1,
           conv_tol=1e-12, conv_tol_dp=1e-11, conv_tol_grad=1e-10,
           max_iter=250, electronic_temperature=40000.0)


@pytest.mark.parametrize("xc", ["wb97x-d", "wb97x-v"])
def test_molecular_gks_mermin_report_and_fermi_occupations(xc, tmp_path):
    positions = np.array([[0.13, -0.18, 0.08], [1.82, 0.29, -0.17]])
    options = dict(reference="gks", xc=xc, basis="sto-3g", electronic_temperature=120000.0,
                   grid_level=0, vv10_grid_level=0, grid_weight_cutoff=0.0,
                   conv_tol=1e-12, conv_tol_dp=1e-10, conv_tol_grad=1e-10)
    native = dft_rs.mermin_free_energy([1, 1], positions, **options)
    energy = dft_rs.energy([1, 1], positions, **options)
    alias = dft_rs.run_feature("free-energy", [1, 1], positions, **options)
    atoms = Atoms("H2", positions=positions * Bohr)
    ase = DftRs(units="atomic").get_mermin_free_energy(atoms, **options)
    xyz = tmp_path / "hydrogen.xyz"
    xyz.write_text("2\nHydrogen molecule\n" + "".join(
        f"H {x:.17g} {y:.17g} {z:.17g}\n" for x, y, z in atoms.positions
    ), encoding="ascii")
    command = [os.environ["DFT_RS_CLI"], "mermin-free-energy", str(xyz), "--format", "json"]
    for key, value in options.items():
        command.extend(("--" + key.replace("_", "-"), str(value)))
    executed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "[spinor-scf]" in executed.stderr and "residual" in executed.stderr
    cli = json.loads(executed.stdout)
    for report in (native, alias, ase, cli):
        assert report["converged"] and report["reference_type"] == "gks"
        assert report["units"]["mermin_free_energy"] == "hartree"
        assert report["electronic_temperature_kelvin"] == options["electronic_temperature"]
        assert abs(report["particle_number_error"]) < 1e-10
        assert report["orbital_residual"] < 1e-10
        (f,) = np.asarray(report["occupations"])
        (mu,) = report["chemical_potentials"]
        eps = np.asarray(report["channels"][0]["energies"])
        kt = report["kt"]
        assert np.all((0.0 < f) & (f < 1.0))
        assert abs(np.sum(f) - 2.0) < 1e-12
        np.testing.assert_allclose(f, 1.0 / (1.0 + np.exp((eps - mu) / kt)),
                                   rtol=0.0, atol=1e-10)
        entropy = -np.sum(f * np.log(f) + (1.0 - f) * np.log1p(-f))
        np.testing.assert_allclose(report["entropy"], entropy, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(report["entropy_term"], -kt * entropy, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(report["mermin_free_energy"], report["internal_energy"] - kt * entropy,
                                   rtol=0.0, atol=1e-12)
        assert abs(report["mermin_free_energy"] - energy["mermin_free_energy"]) < 1e-10
        assert abs(report["free_energy"] - report["mermin_free_energy"]) < 1e-12
    print(f"{xc} GKS molecular Mermin A={native['mermin_free_energy']:.15f} Eh, "
          f"E={native['internal_energy']:.15f} Eh, S={native['entropy']:.15f} k_B, "
          f"orbital residual={native['orbital_residual']:.6e} Eh")


@pytest.mark.parametrize("xc", ["wb97x-d", "wb97x-v"])
def test_open_shell_roks_mermin_occupations_and_public_routes(xc, tmp_path):
    # Distorted ammonia cation, with nine electrons in one shared frame.
    numbers = [7, 1, 1, 1]
    positions = np.array([[0.17, -0.11, 0.23], [1.77, 0.18, -0.53],
                          [-0.31, 1.74, -0.45], [-1.20, -1.31, -0.61]])
    options = dict(reference="roks", xc=xc, basis="sto-3g", charge=1, multiplicity=2,
                   electronic_temperature=80000.0, grid_level=0, vv10_grid_level=0,
                   grid_weight_cutoff=0.0, conv_tol=1e-12, conv_tol_dp=1e-10,
                   conv_tol_grad=1e-10, gradient=True)
    native = dft_rs.mermin_free_energy(numbers, positions, **options)
    atoms = Atoms(numbers, positions=positions * Bohr)
    ase = DftRs(units="atomic").get_mermin_free_energy(atoms, **options)
    xyz = tmp_path / "ammonia.xyz"
    xyz.write_text("4\nDistorted ammonia cation\n" + "".join(
        f"{symbol} {x:.17g} {y:.17g} {z:.17g}\n"
        for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions)
    ), encoding="ascii")
    command = [os.environ["DFT_RS_CLI"], "mermin-free-energy", str(xyz), "--format", "json"]
    for key, value in options.items():
        command.extend(("--" + key.replace("_", "-"),
                        value if isinstance(value, str) else json.dumps(value)))
    proc = subprocess.run(command, text=True, capture_output=True, check=True)
    assert "[roks-scf]" in proc.stderr and "residual" in proc.stderr
    for report in (native, ase, json.loads(proc.stdout)):
        assert report["converged"] and report["reference_type"] == "roks"
        assert report["common_orbital_metric_defect"] < 1e-11
        assert report["orbital_residual"] < 1e-10
        assert report["thermal_population_defect"] < 1e-10
        f = np.asarray(report["occupations"])
        eps = np.asarray(report["orbital_levels"])
        mu = np.asarray(report["chemical_potentials"])[:, None]
        active = (0.0 < f) & (f < 1.0)
        assert np.all(active.any(axis=1))
        np.testing.assert_allclose(f.sum(axis=1), [5.0, 4.0], rtol=0.0, atol=1e-11)
        np.testing.assert_allclose(f, 1.0 / (1.0 + np.exp((eps - mu) / report["kt"])),
                                   rtol=0.0, atol=1e-10)
        partial = f[active]
        entropy = -np.sum(partial * np.log(partial) + (1.0 - partial) * np.log1p(-partial))
        np.testing.assert_allclose(report["entropy"], entropy, rtol=0.0, atol=1e-12)
        assert abs(report["mermin_free_energy"] - report["internal_energy"]
                   + report["kt"] * entropy) < 1e-11
        for key in ("gradient", "mermin_free_energy", "occupations"):
            assert report["units"][key] == native["units"][key]
            np.testing.assert_allclose(report[key], native[key], rtol=0.0, atol=1e-9)
    print(f"{xc} ROKS ammonia cation A={native['mermin_free_energy']:.15f} Eh, "
          f"E={native['internal_energy']:.15f} Eh, S={native['entropy']:.15f} k_B, "
          f"orbital residual={native['orbital_residual']:.6e} Eh, "
          f"population defect={native['thermal_population_defect']:.6e} e")


@pytest.mark.parametrize("reference", ["roks", "gks"])
@pytest.mark.parametrize("xc", ["wb97x-d", "wb97x-v"])
@pytest.mark.parametrize("property_name", ["gradient", "hessian", "fc3", "fc4"])
def test_thermal_nuclear_derivatives_are_connected_to_all_public_routes(
    reference, xc, property_name, tmp_path
):
    positions = np.array([[0.13, -0.18, 0.08], [1.82, 0.29, -0.17]])
    atoms = Atoms("H2", positions=positions * Bohr)
    options = dict(reference=reference, xc=xc, charge=1, multiplicity=2, basis="sto-3g",
                   electronic_temperature=120000.0, grid_level=0, vv10_grid_level=0,
                   grid_weight_cutoff=0.0, conv_tol=1e-13, conv_tol_grad=1e-11,
                   conv_tol_dp=1e-11, response_tolerance=1e-9)
    pattern = [[0.0, 0.0, 0.0], [0.44, -0.19, 0.32]]
    feature, key, extra = {
        "gradient": ("gradient", "gradient", {}),
        "hessian": ("hessian", "hessian", {}),
        "fc3": ("force-constants", "tensor", dict(response_order=3, patterns=[pattern])),
        "fc4": ("force-constants", "tensor", dict(response_order=4, patterns=[pattern])),
    }[property_name]
    xyz = tmp_path / "hydrogen.xyz"
    xyz.write_text("2\nHydrogen molecular ion\n" + "".join(
        f"H {x:.17g} {y:.17g} {z:.17g}\n" for x, y, z in atoms.positions
    ), encoding="ascii")
    selected = dict(options, **extra)
    calculator = DftRs(units="atomic")
    if property_name == "gradient":
        native = dft_rs.gradient([1, 1], positions, **selected)
        ase = calculator.get_gradient_report(atoms, **selected)
    elif property_name == "hessian":
        native = dft_rs.hessian_report([1, 1], positions, **selected)
        ase = calculator.get_hessian_report(atoms, **selected)
    else:
        native = dft_rs.run_feature(feature, [1, 1], positions, **selected)
        ase = calculator.get_feature_report(feature, atoms, **selected)
    command = [os.environ["DFT_RS_CLI"], feature, str(xyz), "--format", "json"]
    for name, value in selected.items():
        command.extend(("--" + name.replace("_", "-"),
                        value if isinstance(value, str) else json.dumps(value)))
    proc = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "iter" in proc.stderr and "residual" in proc.stderr
    discrepancies = []
    for report in (native, ase, json.loads(proc.stdout)):
        assert report["converged"], report
        assert report["units"][key] == native["units"][key]
        np.testing.assert_allclose(report[key], native[key], rtol=0.0, atol=1e-8)
        discrepancies.append(float(np.max(np.abs(np.asarray(report[key]) - native[key]))))
        assert max(report.get("thermal_number_defects", [0.0])) < 1e-9
    print(f"{reference} {xc} {property_name} "
          f"max(abs({key}))={np.max(np.abs(native[key])):.12e} {native['units'][key]}, "
          f"public difference={max(discrepancies):.6e}, "
          f"response residual={max(native.get('response_residuals', [0.0]), default=0.0):.6e}, "
          f"number defect={max(native.get('thermal_number_defects', [0.0]), default=0.0):.6e}")


def test_native_and_ase_thermal_reports_preserve_path_units_and_mermin_identity():
    native = periodic_thermal_response([1, 1], GEOMETRY, CELL,
                                      reciprocal_cutoff=0.75, **SCF)
    atoms = Atoms("H2", positions=GEOMETRY * Bohr, cell=CELL * Bohr, pbc=True)
    ase = DftRs(units="atomic", **SCF).get_periodic_thermal_response(
        atoms, delta_temperature=400.0, reciprocal_cutoff=0.75)
    for key in ("delta_entropy", "delta_internal_energy", "delta_free_energy"):
        assert abs(ase[key] - 400 * native[key]) < 1e-10
    for spin in range(2):
        for part in ("real", "imag"):
            assert np.max(np.abs(np.asarray(ase["density"][spin][0][part])
                                 - 400 * np.asarray(native["density"][spin][0][part]))) < 1e-10
    for report in (native, ase):
        assert report["converged"]
        assert report["shape"] == [2, 1, 2, 2]
        assert max(report["residuals"]) <= 1e-10
        assert max(report["number_constraint_defects"]) < 1e-12
        assert report["occupation_equation_defect"] < 1e-12
        assert report["delta_entropy"] > 0
        assert "density variation dimensionless" in report["units"]
        entropy = (report["energy"] - report["free_energy"]) / report["temperature_hartree"]
        assert abs(report["delta_free_energy"]
                   + entropy * report["delta_temperature_hartree"]) < 1e-10


def test_cli_thermal_report_matches_native_values(tmp_path):
    binary = os.environ.get("DFT_RS_CLI")
    if not binary:
        pytest.skip("DFT_RS_CLI selects the compiled CLI for route acceptance")
    native = periodic_thermal_response([1, 1], GEOMETRY, CELL,
                                      delta_temperature=400.0,
                                      reciprocal_cutoff=0.75, **SCF)
    path = tmp_path / "thermal.xyz"
    positions = GEOMETRY / BOHR_PER_ANGSTROM
    path.write_text("2\nthermal response validation\n"
                    + "".join(f"H {p[0]:.16f} {p[1]:.16f} {p[2]:.16f}\n" for p in positions))
    args = [binary, "thermal-response", str(path), "--cell", f"{9 / BOHR_PER_ANGSTROM:.16f}",
            "--xc", "svwn5", "--overlap-cutoff", "7", "--grid-level", "0", "--grid-shell", "1",
            "--conv-tol", "1e-12", "--conv-tol-dp", "1e-11", "--conv-tol-grad", "1e-10",
            "--max-iter", "250", "--temperature", "40000", "--delta-temperature", "400",
            "--reciprocal-cutoff", "0.75", "--log-mode", "none"]
    completed = subprocess.run(args, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    output = completed.stdout
    for label, key in (("Delta entropy (k_B)", "delta_entropy"),
                       ("Delta internal energy (Hartree)", "delta_internal_energy"),
                       ("Delta free energy (Hartree)", "delta_free_energy")):
        line = next(line for line in output.splitlines() if line.startswith(label + ":"))
        assert abs(float(line.split(":", 1)[1]) - native[key]) < 1e-10
    assert "density variation dimensionless" in output
    assert "occupation" in output.lower()


def test_thermal_request_boundaries_are_checked_before_scf():
    with pytest.raises(ValueError):
        periodic_thermal_response([1, 1], GEOMETRY, CELL)
    with pytest.raises(ValueError):
        periodic_thermal_response([1, 1], GEOMETRY, CELL, delta_temperature=float("nan"), **SCF)
    with pytest.raises(ValueError):
        periodic_thermal_response([1, 1], GEOMETRY, CELL, delta_mu=(0.01, 0), **SCF)
    with pytest.raises(ValueError):
        periodic_thermal_response([1, 1], GEOMETRY, CELL, fixed_number=False,
                                  delta_numbers=(0.01, 0), **SCF)
    with pytest.raises(ValueError):
        periodic_thermal_response([1, 1], GEOMETRY, CELL, delta_numbers=(0.01,), **SCF)
