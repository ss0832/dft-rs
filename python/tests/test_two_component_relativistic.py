# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Common driver, CLI and ASE parity for the finite-nucleus molecular model."""

import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM as BOHR, HARTREE_IN_EV as HARTREE


@pytest.mark.parametrize("method,spin,xc,current", [
    ("x2c", "spin-orbit", "wb97x-d", None),
    ("dkh2", "scalar", "wb97x-v", None),
    ("matrix-zora", "spin-orbit", "wb97m-v", None),
    ("dirac", "spin-orbit", "wb97x-d", None),
    ("dirac", "spin-orbit", "wb97x-v", None),
    ("dirac", "spin-orbit", "wb97x-d", "gaunt"),
    ("dirac", "spin-orbit", "wb97x-v", "breit"),
])
def test_molecular_relativistic_public_routes(tmp_path, method, spin, xc, current):
    numbers, bond, radii = (
        ([1, 1], 1.4, [0.0000159, 0.0000159])
        if current else ([9, 1], 1.7325, [0.0000547, 0.0000159])
    )
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond]])
    atoms = Atoms(numbers, positions=positions * BOHR)
    controls = {
        "relativistic_spin": spin,
        "xc": xc,
        "basis": "sto-3g",
        "nuclear_rms_radii": radii,
        "grid_level": 0,
        "vv10_grid_level": 0,
        "grid_weight_cutoff": 0.0,
        "scf_optimizer": "legacy",
        "cpu_threads": 1,
        "conv_tol": 2e-12,
        "conv_tol_dp": 2e-10,
        "conv_tol_grad": 2e-10,
    }
    if method == "dirac":
        feature = "dirac-dks"
        native = dft_rs.dirac_dks
        get_report = dft_rs.DftRs().get_dirac_dks
    else:
        feature = "two-component-relativistic"
        native = dft_rs.two_component_relativistic
        get_report = dft_rs.DftRs().get_two_component_relativistic
        controls["relativistic_method"] = method
    if current:
        controls.update(dirac_interaction=current, response_order=1,
                        frequencies=[0.1, -0.1], frequency_damping=0.014)
    direct = native(numbers, positions, **controls)
    ase_atomic = get_report(atoms, units="atomic", **controls)
    physical_controls = dict(controls, nuclear_rms_radii=np.asarray(controls["nuclear_rms_radii"]) * BOHR)
    if current:
        physical_controls["frequencies"] = [HARTREE * w for w in controls["frequencies"]]
        physical_controls["frequency_damping"] *= HARTREE
    ase_physical = get_report(atoms, **physical_controls)
    xyz = tmp_path / "molecule.xyz"
    xyz.write_text(
        "2\nFixed-geometry molecule\n" + "".join(
            f"{symbol} {' '.join(f'{x:.17g}' for x in point)}\n"
            for symbol, point in zip(atoms.symbols, atoms.positions)
        ),
        encoding="utf-8",
    )
    command = [os.environ["DFT_RS_CLI"], feature, str(xyz), "--format", "json"]
    for key, value in controls.items():
        command.extend(["--" + key.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)])
    process = subprocess.run(command, check=True, encoding="utf-8", capture_output=True)
    cli = json.loads(process.stdout)
    assert any("[spinor-scf] iter" in line and "residual" in line for line in process.stderr.splitlines())
    for report in [direct, ase_atomic, cli]:
        assert report["converged"]
        assert report["units"]["energy"] == "hartree"
        assert report["units"]["dipole"] == "e*bohr"
        assert report["units"]["decoupling_metric_defect"] == "dimensionless"
        assert report["relativistic_method"] == method
        assert report["relativistic_spin"] == spin
        assert report["energy"] == pytest.approx(direct["energy"], abs=3e-10, rel=0)
        assert report["dipole"] == pytest.approx(direct["dipole"], abs=3e-10, rel=0)
        for key, value in direct["energy_components"].items():
            assert report["energy_components"][key] == pytest.approx(value, abs=3e-10, rel=0)
        assert report["scf_orbital_residual"] <= controls["conv_tol_grad"]
        assert report["decoupling_metric_defect"] < 2e-10
        if method == "dirac":
            assert report["dirac_interaction"] == (current or "coulomb")
            assert report["spectral_gap"] > 10_000.0
            assert report["units"]["spectral_gap"] == "hartree"
        if current:
            assert report["units"]["current_hermiticity_defect"] == "hartree"
            assert report["current_hermiticity_defect"] < 1e-10
            assert abs(report["energy_components"]["current_exchange"]) > 1e-10
            for key in ("polarizability_real", "polarizability_imaginary", "response_residuals"):
                np.testing.assert_allclose(report[key], direct[key], atol=3e-10, rtol=0)
    assert ase_physical["energy"] == pytest.approx(direct["energy"] * HARTREE)
    np.testing.assert_allclose(ase_physical["dipole"], np.asarray(direct["dipole"]) * BOHR, atol=3e-10)
    np.testing.assert_allclose(ase_physical["nuclear_rms_radii"], physical_controls["nuclear_rms_radii"])
    if method == "dirac":
        assert ase_physical["spectral_gap"] == pytest.approx(direct["spectral_gap"] * HARTREE)
    if current:
        real = np.asarray(direct["polarizability_real"])
        imag = np.asarray(direct["polarizability_imaginary"])
        np.testing.assert_allclose(real[0], real[1], atol=1e-9, rtol=0)
        np.testing.assert_allclose(imag[0], -imag[1], atol=1e-9, rtol=0)
        assert np.max(np.asarray(direct["response_residuals"])) < 1e-8
        factor = BOHR**2 / HARTREE
        np.testing.assert_allclose(ase_physical["polarizability_real"], real * factor, atol=3e-10)
        np.testing.assert_allclose(ase_physical["polarizability_imaginary"], imag * factor, atol=3e-10)
        for key in ("current_direct", "current_exchange"):
            assert ase_physical["energy_components"][key] == pytest.approx(
                direct["energy_components"][key] * HARTREE, abs=3e-12, rel=0,
            )


def test_lih_dirac_quadrupole_frequency_and_public_routes(tmp_path):
    numbers = [3, 1]
    positions = np.array([[0.11, -0.22, 0.13], [0.57, 0.63, 2.99]])
    atoms = Atoms(numbers, positions=positions * BOHR)
    controls = dict(
        relativistic_method="dirac", xc="wb97x-d", basis="sto-3g",
        nuclear_rms_radii=[4.62e-5, 1.59e-5], origin=[0.2, -0.1, 0.3],
        grid_level=0, grid_weight_cutoff=0.0, cpu_threads=1,
        frequency=0.10, frequency_damping=0.014,
    )
    native = dft_rs.quadrupole_polarizability(numbers, positions, **controls)
    reverse = dft_rs.quadrupole_polarizability(numbers, positions, **dict(controls, frequency=-0.10))
    ase = dft_rs.DftRs(units="atomic").get_quadrupole_polarizability(atoms, **controls)
    path = tmp_path / "lithium_hydride.xyz"
    path.write_text("2\nLithium hydride\n" + "\n".join(
        " ".join([symbol, *(f"{x:.17g}" for x in point)])
        for symbol, point in zip(atoms.get_chemical_symbols(), atoms.positions)
    ) + "\n", encoding="utf-8")
    command = [os.environ["DFT_RS_CLI"], "quadrupole-polarizability", str(path), "--format", "json"]
    for key, value in controls.items():
        command.extend(["--" + key.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)])
    process = subprocess.run(command, check=True, encoding="utf-8", capture_output=True)
    assert "[spinor-scf] iter" in process.stderr and "residual=" in process.stderr
    cli = json.loads(process.stdout)
    assert native["converged"] and native["relativistic_method"] == "dirac"
    assert native["shape"] == [5, 5]
    assert max(native["response_residuals"] + native["dipole_source_residuals"]) < 1e-8
    for name, unit in (
        ("quadrupole_polarizability", "e^2*bohr^4/hartree"),
        ("dipole_from_field_gradient", "e^2*bohr^3/hartree"),
        ("field_conjugate_moment_from_field", "e^2*bohr^3/hartree"),
    ):
        for suffix, sign in (("_real", 1.0), ("_imag", -1.0)):
            key = name + suffix
            assert native["units"][key] == ase["units"][key] == cli["units"][key] == unit
            np.testing.assert_allclose(ase[key], native[key], rtol=0, atol=3e-9)
            np.testing.assert_allclose(cli[key], native[key], rtol=0, atol=3e-9)
            np.testing.assert_allclose(reverse[key], sign * np.asarray(native[key]), rtol=0, atol=3e-9)
    print(f"LiH DKS electric quadrupole: {json.dumps(native)}")
