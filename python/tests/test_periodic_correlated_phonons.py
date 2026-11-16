# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Public double-hybrid phonon spectra and thermodynamics share one driver."""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree

POSITIONS = np.array([[0.17, 0.41, -0.19], [0.49, -0.12, 1.19]])
CELL = np.array([[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]])
FREQUENCY_KEYS = {"frequencies_cm1", "mesh_frequencies_cm1"}


def assert_single_cell_modes(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    # This fixture has one force-constant cell. Its interpolation
    # preserves the three rigid translations at every sampled wave vector.
    # The signed square root amplifies tiny dynamical eigenvalue differences.
    acoustic = np.abs(expected) < 1e-3
    np.testing.assert_array_equal(acoustic.sum(axis=-1), np.full(expected.shape[0], 3))
    assert np.all(np.abs(actual[acoustic]) < 1e-3)
    np.testing.assert_allclose(actual[~acoustic], expected[~acoustic], atol=2e-7, rtol=1e-9)


@pytest.mark.parametrize("feature", ["phonon-bands", "phonon-thermo"])
def test_correlated_phonon_routes(feature, tmp_path):
    check_correlated_phonon_routes(feature, tmp_path, charged=False)


@pytest.mark.parametrize("feature", ["phonon-bands", "phonon-thermo"])
def test_charged_correlated_phonon_routes(feature, tmp_path):
    check_correlated_phonon_routes(feature, tmp_path, charged=True)


@pytest.mark.parametrize("feature", ["phonon-bands", "phonon-thermo"])
def test_scalar_ecp_correlated_phonon_routes(feature, tmp_path):
    check_correlated_phonon_routes(feature, tmp_path, charged=True, scalar_ecp=True)


def check_correlated_phonon_routes(feature, tmp_path, *, charged, scalar_ecp=False):
    positions, cell = POSITIONS, CELL
    controls = dict(
        xc="b2plyp", basis="sto-3g", dispersion="none",
        k_mesh=[1, 1, 1], supercell=[1, 1, 1], overlap_cutoff=5.5,
        grid_level=1, grid_shell=1, grid_weight_cutoff=0.0,
        conv_tol=1e-12, conv_tol_dp=1e-11, conv_tol_grad=1e-11,
        response_tolerance=1e-11,
    )
    numbers = [2, 1] if charged else [1, 1]
    symbols = ("He", "H") if charged else ("H", "H")
    if charged:
        controls["charge"] = 1
    if scalar_ecp:
        numbers, symbols = [1, 11, 1], ("H", "Na", "H")
        positions = np.array([[0.13, -0.21, 0.07], [2.0, 0.45, -0.31], [-0.46, 1.29, 1.62]])
        controls.update(basis=str(Path(__file__).resolve().parents[2] / "tests/data/periodic_scalar_ecp.g94"),
                        xc_ao_cutoff=16.0)
    if feature == "phonon-bands":
        controls.update(path=[[0, 0, 0], [0.5, 0, 0]], points=2, dos_mesh=[2, 1, 1])
        quantities = ("frequencies_cm1", "dos_frequencies_cm1", "dos")
    else:
        controls.update(q_mesh=[2, 1, 1], temperatures=[0.0, 300.0])
        quantities = ("mesh_frequencies_cm1", "free_energy_kj_mol", "heat_capacity_j_mol_k")
    getter = feature.replace("-", "_")
    native = getattr(dft_rs, getter)(numbers, positions.tolist(), cell.tolist(), **controls)
    atoms = Atoms(numbers, positions=positions * Bohr, cell=cell * Bohr, pbc=True)
    calc = dft_rs.DftRs()
    ase_getter = getattr(calc, f"get_{getter}")
    atomic = ase_getter(atoms, units="atomic", **controls)
    physical_controls = dict(
        controls, overlap_cutoff=controls["overlap_cutoff"] * Bohr,
        conv_tol=controls["conv_tol"] * Hartree, conv_tol_grad=controls["conv_tol_grad"] * Hartree,
    )
    if "xc_ao_cutoff" in controls:
        physical_controls["xc_ao_cutoff"] = controls["xc_ao_cutoff"] * Bohr
    physical = ase_getter(atoms, units="ase", **physical_controls)
    cli_bohr = 1.0 / dft_rs.native.BOHR_PER_ANGSTROM
    xyz = tmp_path / "correlated_cell.xyz"
    xyz.write_text(
        f"{len(numbers)}\ncorrelated phonon cell\n" + "\n".join(
            f"{atom} {x:.17g} {y:.17g} {z:.17g}"
            for atom, (x, y, z) in zip(symbols, positions * cli_bohr, strict=True)
        ) + "\n",
        encoding="utf-8",
    )
    command = [
        os.environ["DFT_RS_CLI"], feature, str(xyz),
        "--cell", json.dumps((cell * cli_bohr).tolist()), "--format", "json",
    ]
    for name, value in controls.items():
        command.extend(["--" + name.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)])
    process = subprocess.run(command, text=True, capture_output=True, check=True)
    assert "energy" in process.stderr.lower() and "residual" in process.stderr.lower()
    cli = json.loads(process.stdout)
    # All adapters report the actual shared homogeneous-path solves. A
    # Cartesian Hessian needs n diagonal and n*(n-1) mixed paths after
    # exact opposite-direction reuse; unit conversion must not change it.
    for report in (native, atomic, physical, cli):
        assert report["path_evaluations"] == (3 * len(numbers)) ** 2
    assert native["converged"]
    assert native["double_hybrid_model"] == "b2plyp"
    assert native["force_constant_asymmetry"] is None
    assert np.asarray(native["supercell_force_constants"]).shape == (3 * len(numbers), 3 * len(numbers))
    assert native["units"]["supercell_force_constants"] == "hartree/bohr^2"
    for report in (atomic, cli):
        if charged:
            assert report["charge_compensation"] == "uniform 3D Ewald background"
            for key in ("primitive_cell_charge", "force_constant_supercell_charge", "born_von_karman_charge"):
                assert report[key] == 1 and report["units"][key] == "e"
        for key in (*quantities, "supercell_force_constants", "energy", "acoustic_residual",
                    "series_reference_residual", "amplitude_response_residuals"):
            if key in FREQUENCY_KEYS:
                assert_single_cell_modes(report[key], native[key])
            else:
                np.testing.assert_allclose(report[key], native[key], atol=2e-7, rtol=1e-9)
    for key in quantities:
        if key in FREQUENCY_KEYS:
            assert_single_cell_modes(physical[key], native[key])
        else:
            np.testing.assert_allclose(physical[key], native[key], atol=2e-7, rtol=1e-9)
    np.testing.assert_allclose(
        physical["supercell_force_constants"], np.asarray(native["supercell_force_constants"]) * Hartree / Bohr**2,
        atol=1e-8, rtol=1e-9,
    )
    assert abs(physical["energy"] - native["energy"] * Hartree) < 1e-8
    assert native["acoustic_residual"] < 1e-8
    if feature == "phonon-bands":
        invalid = subprocess.run(
            command + ["--dos-sigma", "0"], text=True, capture_output=True,
        )
        assert invalid.returncode != 0 and "positive Gaussian width" in invalid.stderr
    keys = (*FREQUENCY_KEYS.intersection(quantities), "energy", "acoustic_residual",
            "series_reference_residual", "amplitude_response_residuals",
            "pt2_same_spin_alpha", "pt2_same_spin_beta", "pt2_opposite_spin",
            "supercell_force_constants")
    if feature == "phonon-thermo":
        keys += ("free_energy_kj_mol", "heat_capacity_j_mol_k")
    print(f"Installed {feature} charged={charged} scalar_ecp={scalar_ecp}", json.dumps({
        key: native[key] for key in keys
    }, allow_nan=False, sort_keys=True))


def test_charged_hf_phonon_bands(tmp_path):
    from test_periodic_gks_stress import cli_command

    numbers = [2, 1]
    controls = dict(xc="hf", charge=1, basis="sto-3g", dispersion="none",
                    k_mesh=[1, 1, 1], overlap_cutoff=5.5, grid_weight_cutoff=0.0,
                    supercell=[1, 1, 1], path=[[0, 0, 0], [0.5, 0, 0]], points=2)
    direct = dft_rs.phonon_bands(numbers, POSITIONS.tolist(), CELL.tolist(), **controls)
    atoms = Atoms(numbers, positions=POSITIONS * Bohr, cell=CELL * Bohr, pbc=True)
    adapted = dft_rs.DftRs().get_phonon_bands(atoms, units="atomic", **controls)
    command = cli_command(os.environ["DFT_RS_CLI"], tmp_path, numbers, POSITIONS, CELL,
                          controls, feature="phonon-bands", pbc=[True] * 3)
    process = subprocess.run(command, text=True, capture_output=True, check=True)
    assert "energy" in process.stderr.lower() and "residual" in process.stderr.lower()
    for report in (direct, adapted, json.loads(process.stdout)):
        assert report["converged"] and report["acoustic_residual"] < 1e-8
        np.testing.assert_allclose(report["energy"], direct["energy"], atol=1e-11, rtol=0)
        np.testing.assert_allclose(report["supercell_force_constants"], direct["supercell_force_constants"],
                                   atol=1e-10, rtol=1e-9)
        assert_single_cell_modes(report["frequencies_cm1"], direct["frequencies_cm1"])
    print("Installed charged HF phonon bands", json.dumps({
        key: direct[key] for key in ("energy", "frequencies_cm1", "acoustic_residual")
    }, allow_nan=False, sort_keys=True))
