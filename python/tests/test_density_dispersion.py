# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832

"""Molecular dispersion energy controls use the same shared driver."""

import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM, HARTREE_IN_EV

POSITIONS = np.array([
    [0.13, -0.27, 0.19],
    [1.78, 0.34, -0.08],
    [-0.49, 1.65, 0.41],
    [0.31, -0.78, 1.91],
])
NUMBERS = [7, 1, 1, 1]
BOHR = BOHR_IN_ANGSTROM
EH = HARTREE_IN_EV
PARAMETERS = dict(
    dispersion_free_polarizabilities=[7.4, 4.5, 4.5, 4.5],
    dispersion_free_c6=[24.2, 6.5, 6.5, 6.5],
    dispersion_free_radii=[1.77 / BOHR, 1.64 / BOHR, 1.64 / BOHR, 1.64 / BOHR],
)

def public_energy_reports(tmp_path, settings):
    native = dft_rs.energy(NUMBERS, POSITIONS, **settings)
    atoms = Atoms(numbers=NUMBERS, positions=POSITIONS * BOHR)
    adapted = dft_rs.DftRs(units="atomic", **settings).get_energy_report(atoms)
    xyz = tmp_path / "ammonia.xyz"
    xyz.write_text("4\nC1 ammonia\n" + "\n".join(
        " ".join([symbol, *(f"{v:.16g}" for v in point)])
        for symbol, point in zip(atoms.get_chemical_symbols(), atoms.positions)
    ) + "\n", encoding="utf-8")
    command = [os.environ["DFT_RS_CLI"], "energy", str(xyz), "--format", "json"]
    for key, value in settings.items():
        command.extend(["--" + key.replace("_", "-"), json.dumps(value) if isinstance(value, list) else str(value)])
    cli = subprocess.run(command, capture_output=True, text=True, check=True)
    iterations = [line for line in cli.stderr.splitlines() if "[scf] iter" in line]
    assert iterations and all("E =" in line and "|SCF residual| =" in line for line in iterations)
    print(iterations[-1])
    return native, (("ASE", adapted), ("CLI", json.loads(cli.stdout))), atoms


@pytest.mark.parametrize("model,xc,scale", [
    ("ts", "pbe", 0.94), ("mbd-ts", "pbe", 0.81), ("vv09", "spw92", None),
    ("d1", "blyp", None), ("mbd", "pbe", 0.81),
])
def test_molecular_density_dispersion_energy_routes(tmp_path, model, xc, scale):
    settings = dict(
        xc=xc, basis="sto-3g", reference="rks", grid_level=int(scale is not None),
        dispersion=model, cpu_threads=1,
    )
    if scale is not None:
        settings.update(dispersion_radius_scale=scale, **PARAMETERS)
    elif model == "d1":
        settings.update(
            dispersion_energy_scale=0.83,
            dispersion_free_c6=[18.2, 3.7, 4.1, 4.4],
            dispersion_free_radii=[2.6, 1.9, 1.7, 2.1],
        )
    native, others, atoms = public_energy_reports(tmp_path, settings)
    keys = ["energy", "dispersion_energy", "scf_orbital_residual"]
    if scale is not None:
        keys += ["dispersion_volume_ratios", "dispersion_effective_polarizabilities",
                 "dispersion_effective_c6", "dispersion_effective_radii"]
    if model in ("mbd-ts", "mbd"):
        keys += ["mbd_frequencies", "mbd_coupled_frequencies", "mbd_minimum_squared_frequency"]
    if model == "mbd":
        keys += ["mbd_minimum_screening_eigenvalue", "mbd_screening_residual", "mbd_frequency_scale",
                 "mbd_unscreened_polarizabilities", "mbd_unscreened_c6", "mbd_unscreened_radii"]
    elif model == "d1":
        keys += ["dispersion_energy_scale"]
    assert native["converged"] and abs(native["dispersion_energy"]) > 1e-8
    for route, report in others:
        assert report["converged"] and report["dispersion_model"] == native["dispersion_model"]
        differences = {}
        for key in keys:
            assert report["units"][key] == native["units"][key]
            np.testing.assert_allclose(report[key], native[key], atol=2e-12, rtol=0)
            differences[key] = float(np.max(np.abs(np.asarray(report[key]) - native[key])))
        print(f"{model} {route} defects from native: {differences}")
    print(f"{model} ammonia: correction={native['dispersion_energy']:.12e} Hartree total={native['energy']:.12e} Hartree")

    if scale is None:
        return
    converted = dict(settings)
    for key, multiplier in {
        "dispersion_free_polarizabilities": BOHR**3,
        "dispersion_free_c6": EH * BOHR**6,
        "dispersion_free_radii": BOHR,
    }.items():
        converted[key] = (np.asarray(settings[key]) * multiplier).tolist()
    physical = dft_rs.DftRs(**converted).get_energy_report(atoms)
    np.testing.assert_allclose(physical["dispersion_energy"], native["dispersion_energy"] * EH, atol=2e-11, rtol=0)
    np.testing.assert_allclose(physical["dispersion_effective_c6"], np.asarray(native["dispersion_effective_c6"]) * EH * BOHR**6, atol=2e-11, rtol=0)


@pytest.mark.parametrize("order", [3, 4])
def test_molecular_d1_force_constant_block_uses_public_routes(tmp_path, order):
    # A molecular Cartesian block exercises the electronic response and the
    # separate analytic pair correction through the same public controls.
    numbers = [3, 1]
    positions = np.array([[0.13, -0.23, 0.17], [0.66, 0.41, 3.08]])
    atoms = Atoms(numbers=numbers, positions=positions * BOHR)
    controls = dict(
        xc="pbe", basis="sto-3g", reference="rks", grid_level=0,
        grid_weight_cutoff=0.0, cpu_threads=1, conv_tol=1e-12,
    )
    with_d1 = dict(controls, dispersion="d1", dispersion_energy_scale=0.81,
                   dispersion_free_c6=[35.0, 6.5], dispersion_free_radii=[3.1, 2.4])
    coordinates = [5]
    parent = dft_rs.analytic_force_constants(numbers, positions, order=order,
                                             coordinates=coordinates, **controls)
    native = dft_rs.analytic_force_constants(numbers, positions, order=order,
                                             coordinates=coordinates, **with_d1)
    ase = dft_rs.DftRs(units="atomic").get_analytic_force_constants(
        atoms, order=order, coordinates=coordinates, **with_d1)
    path = tmp_path / "lithium_hydride.xyz"
    path.write_text("2\nLithium hydride\n" + "\n".join(
        " ".join([element, *(f"{v:.17g}" for v in point)])
        for element, point in zip(atoms.get_chemical_symbols(), atoms.positions)
    ) + "\n", encoding="utf-8")
    command = [os.environ["DFT_RS_CLI"], "force-constants", str(path), "--format", "json"]
    for key, value in dict(with_d1, response_order=order, coordinates=coordinates).items():
        command.extend(["--" + key.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)])
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "[scf] iter" in process.stderr and "SCF residual" in process.stderr
    cli = json.loads(process.stdout)
    c6 = 2.0 * 35.0 * 6.5 / (35.0 + 6.5)
    radius = np.linalg.norm(positions[1] - positions[0])
    energy = -0.81 * c6 / (radius**6 * (1.0 + np.exp(23.0 * (1.0 - radius / 5.5))))
    assert abs(native["tensor"][0] - parent["tensor"][0]) > 1e-10
    assert native["energy"] - parent["energy"] == pytest.approx(energy, abs=3e-12, rel=0.0)
    for report in (native, ase, cli):
        labels = report.get("unit_labels", report["units"])
        assert report["converged"] and labels["tensor"] == f"hartree/bohr^{order}"
        assert report["dispersion_model"] == "D1"
        assert report["dispersion_energy"] == pytest.approx(energy, abs=3e-13, rel=0.0)
        np.testing.assert_allclose(report["tensor"], native["tensor"], atol=2e-9, rtol=0)
    print(f"D1 LiH order {order}: total={native['tensor'][0]:.17e}, "
          f"correction={native['tensor'][0] - parent['tensor'][0]:.17e}, "
          f"orbital_residual={native['scf_orbital_residual']:.3e}")


@pytest.mark.parametrize("model", [
    "vdw-df", "vdw-df2", "optpbe-vdw", "optb88-vdw", "optb86b-vdw", "vdw-df-cx", "rev-vdw-df2",
])
def test_fixed_density_vdw_df_common_energy_routes(tmp_path, model):
    settings = dict(
        xc="spw92", basis="sto-3g", reference="rks", grid_level=0, grid_weight_cutoff=0.0,
        dispersion=model, vdw_df_radial_order=14, vdw_df_angular_degree=9,
        vdw_df_table_points=72, vdw_df_momentum_cutoff=48.0, cpu_threads=1,
    )
    native, others, _ = public_energy_reports(tmp_path, settings)
    keys = ["energy", "scf_orbital_residual", "vdw_df_parent_energy", "vdw_df_parent_xc_energy",
            "vdw_df_exchange_energy", "vdw_df_local_correlation_energy",
            "vdw_df_nonlocal_correlation_energy", "vdw_df_xc_energy"]
    for route, report in others:
        assert report["functional"] == native["functional"] == model
        assert report["vdw_df_parent_functional"] == native["vdw_df_parent_functional"]
        defects = {}
        for key in keys:
            assert report["units"][key] == native["units"][key] == "hartree"
            np.testing.assert_allclose(report[key], native[key], atol=3e-12, rtol=0)
            defects[key] = abs(report[key] - native[key])
        assert report["converged"]
        print(f"{model} {route} defects from native: {defects}")
    np.testing.assert_allclose(
        native["energy"] - native["vdw_df_parent_energy"],
        native["vdw_df_xc_energy"] - native["vdw_df_parent_xc_energy"],
        atol=5e-14, rtol=0,
    )
