# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Common-driver equality for coherent, quantum and finite-frequency Hall."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
import dft_rs
from dft_rs.ase_calculator import DftRs

POSITIONS = np.array([[0.0, 0.0, 0.0], [1.6, 0.1, 0.05]])
CELL = np.array([[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]])
METHOD = dict(xc="pbe0", basis="sto-3g", k_mesh=[4, 1, 1], overlap_cutoff=12.0,
              exchange_cutoff=5.5, grid_level=2, grid_shell=1, conv_tol=1e-11,
              conv_tol_dp=1e-9, max_iter=300)


def cli_report(tmp_path, feature, controls, numbers=(1, 1), positions=POSITIONS, cell=CELL):
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "The acceptance runner must supply DFT_RS_CLI"
    path = tmp_path / "cell.xyz"
    symbols = {1: "H", 2: "He"}
    path.write_text(str(len(numbers)) + "\ncell\n" + "\n".join(
        symbols[z] + " " + " ".join(f"{v*0.529177210903:.17g}" for v in r)
        for z, r in zip(numbers, positions)) + "\n", encoding="utf-8")
    command = [executable, feature, str(path), "--cell",
               ",".join(f"{v*0.529177210903:.17g}" for v in np.asarray(cell).flatten()),
               "--format", "json"]
    for key, value in controls.items():
        if isinstance(value, (list, tuple)):
            value = ",".join(str(x) for x in value)
        command.extend(["--" + key.replace("_", "-"), str(value)])
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def compare_numeric(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare_numeric(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare_numeric(x, y)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10)
    else:
        assert a == b


@pytest.mark.parametrize("feature, controls", [
    ("coherent-transport", dict(transport_k_mesh=[4, 2, 2],
                               transport_frequencies_hartree=[0.0, 0.13, -0.13])),
    ("quantum-hall", dict(topology_mesh=[4, 4])),
])
def test_coherent_and_quantum_reports_agree_on_every_surface(tmp_path, feature, controls):
    settings = METHOD | controls
    native = getattr(dft_rs, feature.replace("-", "_"))([1, 1], POSITIONS, CELL, **settings)
    generic = dft_rs.run_feature(feature, [1, 1], POSITIONS, cell=CELL, **settings)
    atoms = Atoms("H2", positions=POSITIONS * Bohr, cell=CELL * Bohr, pbc=True)
    calc = DftRs(units="atomic", **METHOD)
    ase = getattr(calc, "get_" + feature.replace("-", "_"))(atoms, **controls)
    cli = cli_report(tmp_path, feature, settings)
    for report in [generic, ase, cli]:
        compare_numeric(native["results"], report["results"])
        assert native["units"] == report["units"]
        assert native["model"] == report["model"]
    assert native["fock_image_residual"] < 1e-10
    if feature == "coherent-transport":
        positive = np.asarray(native["results"][1]["conductivity"])
        negative = np.asarray(native["results"][2]["conductivity"])
        np.testing.assert_allclose(positive[..., 0], negative[..., 0], rtol=1e-12, atol=1e-8)
        np.testing.assert_allclose(positive[..., 1], -negative[..., 1], rtol=1e-12, atol=1e-8)
        for row in native["results"]:
            assert row["raw_current_hermitian_residual"] < 1e-10
            assert row["raw_equation_residual"] < 1e-13
        print("coherent raw residuals", [(r["raw_current_hermitian_residual"], r["raw_equation_residual"]) for r in native["results"]])
    else:
        assert abs(native["chern_number"]) < 1e-12
        assert abs(native["slice_hall_conductance"]) < 1e-16
        assert native["results"][0]["sampled_min_gap"] > 0.0
        print("quantum raw diagnostics", native["results"])


def test_finite_frequency_hall_has_common_units_and_drude_factors(tmp_path):
    cell = np.array([[0.3, 3.1, 2.8], [2.9, 0.2, 3.2], [3.0, 2.7, 0.1]])
    settings = METHOD | dict(k_mesh=[2, 2, 2], transport_k_mesh=[6, 6, 6],
                             transport_temperatures_k=[2000.0], carrier_concentrations_cm3=[-1e22],
                             transport_frequencies_hartree=[0.0, 0.02, -0.02], relaxation_time_fs=10.0)
    native = dft_rs.hall_transport([2], [[0.0]*3], cell, **settings)
    generic = dft_rs.run_feature("hall-transport", [2], [[0.0]*3], cell=cell, **settings)
    atoms = Atoms("He", positions=[[0.0]*3], cell=cell*Bohr, pbc=True)
    calc = DftRs(units="atomic", **(METHOD | dict(k_mesh=[2, 2, 2])))
    extra = {k: v for k, v in settings.items() if k not in METHOD}
    ase = calc.get_hall_transport(atoms, **extra, units="atomic")
    cli = cli_report(tmp_path, "hall-transport", settings, numbers=[2], positions=[[0.0]*3], cell=cell)
    for report in [generic, ase, cli]:
        compare_numeric(native["results"], report["results"])
        assert native["units"] == report["units"]
    row = native["results"][0]
    sigma = np.asarray(row["conductivity_over_tau"])
    hall = np.asarray(row["hall_conductivity_derivative_over_tau_squared"])
    atomic_time = 1.054571817e-34 / 4.3597447222071e-18
    for ac in row["frequency_results"]:
        z = 1e-14 / (1.0 - 1j * ac["angular_frequency_hartree"] * 1e-14 / atomic_time)
        actual = np.asarray(ac["conductivity"])
        actual_hall = np.asarray(ac["hall_conductivity_derivative"])
        np.testing.assert_allclose(actual[..., 0]+1j*actual[..., 1], sigma*z, rtol=1e-12, atol=1e-16)
        np.testing.assert_allclose(actual_hall[..., 0]+1j*actual_hall[..., 1], hall*z*z, rtol=1e-12, atol=1e-16)
