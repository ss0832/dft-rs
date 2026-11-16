# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Energy-dependent transport through the common native, ASE and CLI driver."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
import dft_rs

CELL = [[0.3, 3.1, 2.8], [2.9, 0.2, 3.2], [3.0, 2.7, 0.1]]
BASE = dict(basis="sto-3g", k_mesh=[2, 2, 2], overlap_cutoff=12.0,
            grid_level=2, grid_shell=1, conv_tol=1e-11, guess="core")
TRANSPORT = dict(transport_k_mesh=[4, 4, 4], transport_temperatures_k=[2000.0],
                 carrier_concentrations_cm3=[-1e22])
TABLE = [[-4.0, 3.0], [1.0, 23.0]]
FREQUENCIES = [0.0, 0.02, -0.02]


@pytest.mark.parametrize("xc", ["svwn5", "pbe0", "tpss"])
def test_energy_dependent_hall_and_thermal_tensors_agree_on_every_route(tmp_path, xc):
    controls = dict(BASE, **TRANSPORT, xc=xc, scattering_table=TABLE,
                    transport_frequencies_hartree=FREQUENCIES)
    native = dft_rs.hall_transport([2], [[0.0, 0.0, 0.0]], CELL, **controls)
    assert native["converged"] and native["scattering_table"] == TABLE
    generic = dft_rs.run_feature("hall-transport", [2], [[0.0, 0.0, 0.0]], cell=CELL, **controls)
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=np.asarray(CELL) * Bohr, pbc=True)
    adapted = dft_rs.DftRs(units="atomic", xc=xc, **BASE).get_hall_transport(
        atoms, **TRANSPORT, scattering_table=TABLE, transport_frequencies_hartree=FREQUENCIES, units="atomic")
    xyz = tmp_path / "helium.xyz"
    xyz.write_text("1\ntransport\nHe 0 0 0\n", encoding="utf-8")
    cell = ";".join(",".join(f"{v * 0.529177210903:.16f}" for v in row) for row in CELL)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for route acceptance"
    arguments = [executable, "hall-transport", str(xyz), "--cell", cell, "--xc", xc,
                 "--basis", "sto-3g", "--k-mesh", "2,2,2", "--overlap-cutoff", "12",
                 "--grid-level", "2", "--grid-shell", "1", "--conv-tol", "1e-11", "--guess", "core",
                 "--transport-k-mesh", "4,4,4", "--transport-temperatures-k", "2000",
                 "--carrier-concentrations-cm3", "-1e22", "--scattering-table", "-4,3;1,23",
                 "--transport-frequencies-hartree", "0,0.02,-0.02", "--format", "json"]
    result = subprocess.run(arguments, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cli = json.loads(result.stdout)
    row = native["results"][0]
    keys = ["conductivity", "seebeck", "electronic_thermal_conductivity",
            "hall_conductivity_derivative", "hall_coefficient", "raw_hall_onsager_residual",
            "relative_hall_onsager_residual"]
    for other in [generic, adapted, cli]:
        assert other["units"] == native["units"]
        assert other["scattering_convention"] == native["scattering_convention"]
        assert other["hall_axes"] == native["hall_axes"]
        for key in keys:
            np.testing.assert_allclose(other["results"][0][key], row[key], rtol=1e-8, atol=1e-20)
        for a, b in zip(other["results"][0]["frequency_results"], row["frequency_results"]):
            for key in ["conductivity", "hall_conductivity_derivative", "seebeck", "peltier", "electronic_thermal_conductivity"]:
                np.testing.assert_allclose(a[key], b[key], rtol=1e-8, atol=1e-20)
            assert a["raw_thermoelectric_inverse_residual"] < 1e-10
            assert a["raw_kelvin_residual"] < 1e-10
    assert "conductivity_over_tau" not in row and "hall_conductivity_derivative_over_tau_squared" not in row
    assert native["units"]["results.conductivity"] == "S/m"
    assert native["units"]["results.hall_conductivity_derivative"] == "S/(m T)"
    zero, plus, minus = row["frequency_results"]
    np.testing.assert_allclose(np.asarray(zero["conductivity"])[..., 0], row["conductivity"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(zero["hall_conductivity_derivative"])[..., 0], row["hall_conductivity_derivative"], rtol=1e-12)
    for key in ["seebeck", "electronic_thermal_conductivity"]:
        np.testing.assert_allclose(np.asarray(zero[key])[..., 0], row[key], rtol=1e-10, atol=1e-20)
    for ac in row["frequency_results"]:
        np.testing.assert_allclose(ac["peltier"], np.swapaxes(ac["seebeck"], 0, 1) * 2000.0, rtol=1e-10, atol=1e-20)
    for key in ["conductivity", "hall_conductivity_derivative", "seebeck", "peltier", "electronic_thermal_conductivity"]:
        np.testing.assert_allclose(plus[key], np.asarray(minus[key]) * [1, -1], rtol=1e-12)
    inverse = np.linalg.inv(np.asarray(row["conductivity"]))
    expected = np.stack([-inverse @ np.asarray(row["hall_conductivity_derivative"])[:, :, c] @ inverse for c in range(3)], axis=-1)
    np.testing.assert_allclose(expected, row["hall_coefficient"], rtol=1e-10, atol=1e-20)
    boltzmann = dft_rs.boltzmann_transport([2], [[0.0, 0.0, 0.0]], CELL, **controls)
    np.testing.assert_array_equal(boltzmann["results"][0]["conductivity"], row["conductivity"])
    print(xc, "weighted Hall raw Onsager", row["raw_hall_onsager_residual"],
          "relative", row["relative_hall_onsager_residual"], "Fock residual", native["fock_image_residual"])
    if xc == "svwn5":
        converted = dft_rs.DftRs(units="atomic", xc=xc, **BASE).get_hall_transport(
            atoms, **TRANSPORT, scattering_table=[[e * Hartree, time] for e, time in TABLE],
            transport_frequencies_hartree=FREQUENCIES, units="ase")
        np.testing.assert_allclose(converted["scattering_table"], [[e * Hartree, time] for e, time in TABLE], rtol=1e-14)
        assert converted["units"]["scattering_table"].startswith("[eV, fs]")
        for key in ["conductivity", "seebeck", "electronic_thermal_conductivity", "hall_coefficient"]:
            np.testing.assert_allclose(converted["results"][0][key], row[key], rtol=1e-10, atol=1e-20)


def test_constant_table_recovers_legacy_dc_and_dynamic_units():
    common = dict(BASE, **TRANSPORT, xc="svwn5", transport_frequencies_hartree=FREQUENCIES)
    legacy = dft_rs.hall_transport([2], [[0.0, 0.0, 0.0]], CELL, **common, relaxation_time_fs=7.0)["results"][0]
    weighted = dft_rs.hall_transport([2], [[0.0, 0.0, 0.0]], CELL, **common, scattering_table=[[0.0, 7.0]])["results"][0]
    for key, old_key, scale in [("conductivity", "conductivity_over_tau", 7e-15),
                                ("electronic_thermal_conductivity", "electronic_thermal_conductivity_over_tau", 7e-15),
                                ("hall_conductivity_derivative", "hall_conductivity_derivative_over_tau_squared", (7e-15)**2),
                                ("seebeck", "seebeck", 1.0), ("hall_coefficient", "hall_coefficient", 1.0)]:
        np.testing.assert_allclose(weighted[key], np.asarray(legacy[old_key]) * scale, rtol=1e-10, atol=1e-20)
    for a, b in zip(weighted["frequency_results"], legacy["frequency_results"]):
        for key in ["conductivity", "hall_conductivity_derivative", "seebeck", "peltier", "electronic_thermal_conductivity"]:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-10, atol=1e-20)


def test_invalid_scattering_controls_fail_before_reference_calculation():
    for table in [[], [[0, 0]], [[1, 1], [0, 2]], [[0, 1, 2]]]:
        with pytest.raises(ValueError, match="scattering"):
            dft_rs.boltzmann_transport([2], [[0.0, 0.0, 0.0]], CELL, **TRANSPORT, scattering_table=table)
    with pytest.raises(ValueError, match="without relaxation_time_fs"):
        dft_rs.boltzmann_transport([2], [[0.0, 0.0, 0.0]], CELL, **TRANSPORT,
                                 scattering_table=TABLE, relaxation_time_fs=1.0)
