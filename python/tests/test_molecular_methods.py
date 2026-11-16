# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832

import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Hartree

import dft_rs


WATER = [8, 1, 1], [[0.12, -0.17, 0.08], [1.68, 0.61, -0.29], [-0.66, 1.37, 0.67]]


def test_shared_weighted_xc_in_constrained_dft_and_car_parrinello_md():
    numbers, coordinates = WATER
    controls = {
        "reference": "rks", "grid_level": 0, "grid_weight_cutoff": 0.0,
        "xc_terms": [{"xc": "pbe", "weight": 0.45}, {"xc": "tpssh", "weight": 0.55}],
    }
    energy = dft_rs.run_feature("energy", numbers, coordinates, **controls)["energy"]
    constrained = dft_rs.constrained_dft(
        numbers, coordinates, density_constraints=[{"atoms": [1], "target": 1.04}], **controls,
    )
    dynamics = dft_rs.car_parrinello_md(
        numbers, coordinates, md_steps=2, md_timestep=1.0, **controls,
    )
    assert constrained["converged"] and dynamics["converged"]
    assert constrained["functional"] == dynamics["functional"]
    assert max(abs(value) for value in constrained["constraint_residuals"]) < 1e-9
    assert dynamics["frames"][0]["potential_energy"] == pytest.approx(energy, abs=2e-9, rel=0)
    assert dynamics["max_extended_energy_drift"] < 2e-5


@pytest.mark.parametrize("xc", ["wb97x-d", "wb97x-v"])
def test_fragment_electron_constraint_gradient_public_parity(tmp_path, xc):
    from dft_rs.units import BOHR_IN_ANGSTROM, HARTREE_IN_EV

    numbers, coordinates = WATER
    atoms = Atoms(numbers=numbers, positions=np.asarray(coordinates) * BOHR_IN_ANGSTROM)
    controls = {
        "xc": xc, "reference": "rks", "grid_level": 0, "vv10_grid_level": 0,
        "grid_weight_cutoff": 0.0, "gradient": True, "constraint_electric_response": True,
        "density_constraints": [{"atoms": [0, 1], "electrons": 8.96}],
    }
    direct = dft_rs.constrained_dft(numbers, coordinates, **controls)
    calc = dft_rs.DftRs()
    atomic = calc.get_constrained_dft(atoms, units="atomic", **controls)
    physical = calc.get_constrained_dft(atoms, **controls)
    geometry = tmp_path / "water.xyz"
    atoms.write(geometry, format="xyz")
    argv = [os.environ["DFT_RS_CLI"], "constrained-dft", str(geometry), "--format", "json"]
    for key, value in controls.items():
        argv.extend(["--" + key.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)])
    process = subprocess.run(argv, check=True, capture_output=True, text=True)
    cli = json.loads(process.stdout)
    for report in (direct, atomic, physical, cli):
        assert abs(report["constraint_expectations"][0] - 8.96) < 1e-9
        np.testing.assert_allclose(report["forces"], -np.asarray(report["gradient"]))
    for report in (atomic, cli):
        for key in ("gradient", "polarizability", "electric_constraint_defect"):
            np.testing.assert_allclose(report[key], direct[key], rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(physical["forces"], np.asarray(direct["forces"]) * HARTREE_IN_EV / BOHR_IN_ANGSTROM)
    assert "iter" in process.stderr and "residual" in process.stderr


@pytest.mark.parametrize(
    ("feature", "name", "key", "scale", "extra"),
    [
        (
            "constrained-dft", "constrained_dft", "constraint_multipliers", Hartree,
            {"density_constraints": [{"atoms": [1], "target": 1.04}]},
        ),
        (
            "car-parrinello-md", "car_parrinello_md", "max_extended_energy_drift", Hartree,
            {"md_steps": 3, "md_timestep": 1.0, "fictitious_mass": 200.0,
             "nuclear_velocities": [[0.0, 0.0, 0.0], [0.0015, -0.0003, 0.0007], [-0.0004, 0.0011, -0.0003]]},
        ),
        (
            "nexafs", "nexafs", "spectrum_energies", Hartree,
            {"absorber_atoms": [0], "nexafs_broadening": 0.025, "polarization": [0.37, -0.81, 0.45], "points": 51},
        ),
        (
            "magnetic-quadrupole", "magnetic_quadrupole", "magnetic_quadrupole_components", 1.0,
            {"reference": "gks", "magnetic_field": [0.014, -0.019, 0.023], "origin": [0.23, -0.31, 0.39]},
        ),
    ],
)
@pytest.mark.parametrize("xc", ["wb97x-d", "wb97x-v"])
def test_molecular_methods_public_parity(tmp_path, feature, name, key, scale, extra, xc):
    numbers, coordinates = WATER
    atoms = Atoms(numbers=numbers, positions=np.asarray(coordinates) * Bohr)
    controls = {
        "xc": xc, "grid_level": 0, "vv10_grid_level": 0, "grid_weight_cutoff": 0.0, **extra,
    }
    direct = getattr(dft_rs, name)(numbers, coordinates, **controls)
    calc = dft_rs.DftRs()
    getter = getattr(calc, f"get_{name}")
    atomic = getter(atoms, units="atomic", **controls)
    physical = dict(controls)
    if feature == "nexafs":
        physical["nexafs_broadening"] *= Hartree
    if feature == "magnetic-quadrupole":
        physical["origin"] = (np.asarray(physical["origin"]) * Bohr).tolist()
    ase = getter(atoms, **physical)
    np.testing.assert_allclose(atomic[key], direct[key], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(ase[key], np.asarray(direct[key]) * scale, rtol=1e-7, atol=1e-9)

    geometry = tmp_path / "water.xyz"
    atoms.write(geometry)
    argv = [os.environ["DFT_RS_CLI"], feature, str(geometry), "--format", "json"]
    for control, value in controls.items():
        argv.extend(["--" + control.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)])
    process = subprocess.run(argv, check=True, capture_output=True, text=True)
    report = json.loads(process.stdout)
    np.testing.assert_allclose(report[key], direct[key], rtol=2e-6, atol=1e-8)
    assert "iter" in process.stderr and "E" in process.stderr
