# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Unequal-channel fixed-number spin response on all shared public routes."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
import dft_rs
from dft_rs.ase_calculator import _to_ase_units
from dft_rs.units import BOHR_IN_ANGSTROM


POSITIONS = [[0.0, 0.0, -0.9], [0.0, 0.0, 0.9], [1.7, 0.3, 0.2]]
CELL = np.eye(3) * 9.5
BASE = dict(basis="sto-3g", multiplicity=2, unrestricted=True, k_mesh=[1, 1, 1],
            grid_level=1, grid_shell=1, overlap_cutoff=7.0, conv_tol=1e-12,
            guess="core", max_iter=250)
SPIN = dict(exchange_enhanced=True, temperatures_k=[150000.0],
            spin_g_factor=2.0, response_reciprocal_cutoff=16.0)


def _cli_report(tmp_path, controls):
    xyz = tmp_path / "hydrogen.xyz"
    xyz.write_text("3\nspin-biased response\n" + "".join(
        f"H {x*BOHR_IN_ANGSTROM:.16f} {y*BOHR_IN_ANGSTROM:.16f} {z*BOHR_IN_ANGSTROM:.16f}\n" for x, y, z in POSITIONS), encoding="utf-8")
    argv = [os.environ["DFT_RS_CLI"], "pauli-spin-susceptibility", str(xyz),
            "--cell", ";".join(",".join(f"{v*BOHR_IN_ANGSTROM:.16f}" for v in row) for row in CELL),
            "--format", "json"]
    for key, value in controls.items():
        argv.append("--" + key.replace("_", "-"))
        if isinstance(value, list):
            argv.append(",".join(str(item) for item in value))
        elif value is not True:
            argv.append(str(value).lower() if isinstance(value, bool) else str(value))
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("xc", ["svwn5", "pbe", "tpss"])
def test_polarized_periodic_spin_native_generic_ase_cli(tmp_path, xc):
    controls = dict(BASE, **SPIN, xc=xc)
    native = dft_rs.pauli_spin_susceptibility([1, 1, 1], POSITIONS, CELL.tolist(), **controls)
    generic = dft_rs.run_feature("pauli-spin-susceptibility", [1, 1, 1], POSITIONS,
                                 cell=CELL.tolist(), **controls)
    atoms = Atoms("H3", positions=np.asarray(POSITIONS) * Bohr, cell=CELL * Bohr, pbc=True)
    adapted = dft_rs.DftRs(units="atomic", **BASE, xc=xc).get_pauli_spin_susceptibility(
        atoms, **SPIN, units="atomic")
    cli = _cli_report(tmp_path, controls)
    row = native["rows"][0]
    assert native["converged"] and row["response_converged"]
    assert abs(row["electrons"] - 3.0) < 1e-10
    assert abs(row["reference_spin_bias"]) > 0.01
    assert abs(sum(row["unconstrained_delta_spin_numbers"])) > 1e-5
    assert abs(row["chemical_potential_response"]) > 1e-5
    assert row["raw_spin_density_cancellation_residual"] > 1e-6
    assert row["raw_charge_constraint_residual"] < 1e-10
    assert row["bare_charge_constraint_residual"] < 1e-10
    assert row["constrained_equation_residual"] <= 1e-10
    assert max(row["response_residuals"]) <= 1e-10
    assert row["bare_dos_consistency_residual"] < 1e-10
    dos = row["spin_dos_at_fermi_level"]
    expected = dos[0] * dos[1] / sum(dos)
    np.testing.assert_allclose(row["pauli_susceptibility_per_cell"], expected, rtol=1e-10, atol=1e-12)
    converted = _to_ase_units(native)
    converted_row = converted["rows"][0]
    for key in ["chemical_potential_response", "bare_chemical_potential_response", "reference_spin_bias"]:
        np.testing.assert_allclose(converted_row[key], Hartree * row[key], rtol=1e-13, atol=1e-14)
    for key in ["charge_compressibility", "bare_charge_compressibility", "spin_dos_at_fermi_level"]:
        np.testing.assert_allclose(converted_row[key], np.asarray(row[key]) / Hartree, rtol=1e-13, atol=1e-14)
    for key in ["electrons", "delta_spin_numbers", "unconstrained_delta_spin_numbers",
                "raw_charge_constraint_residual", "bare_charge_constraint_residual",
                "constrained_equation_residual", "response_residuals"]:
        np.testing.assert_allclose(converted_row[key], row[key], rtol=0, atol=0)
    assert converted["units"]["rows.chemical_potential_response"].startswith("eV")
    assert native["units"]["rows.chemical_potential_response"].startswith("hartree")
    for other in [generic, adapted, cli]:
        assert other["units"] == native["units"]
        assert other["model"] == native["model"]
        for key in ["enhanced_spin_susceptibility_per_cell", "pauli_susceptibility_per_cell",
                    "chemical_potential_response", "bare_chemical_potential_response",
                    "charge_compressibility", "bare_charge_compressibility",
                    "spin_chemical_potentials", "spin_dos_at_fermi_level", "reference_spin_bias",
                    "delta_spin_numbers", "unconstrained_delta_spin_numbers"]:
            np.testing.assert_allclose(other["rows"][0][key], row[key], rtol=1e-9, atol=1e-12)
    print(xc, "unequal-channel periodic spin", row)


@pytest.mark.parametrize("xc", ["svwn5", "pbe", "tpss", "pbe0"])
def test_unequal_rigid_spin_thermodynamics_on_all_routes(tmp_path, xc):
    controls = dict(BASE, xc=xc, electronic_temperature=150000.0,
                    temperatures_k=[150000.0], exchange_enhanced=False, spin_g_factor=2.0)
    native = dft_rs.pauli_spin_susceptibility([1, 1, 1], POSITIONS, CELL.tolist(), **controls)
    generic = dft_rs.run_feature("pauli-spin-susceptibility", [1, 1, 1], POSITIONS,
                                 cell=CELL.tolist(), **controls)
    atoms = Atoms("H3", positions=np.asarray(POSITIONS) * Bohr, cell=CELL * Bohr, pbc=True)
    adapted = dft_rs.DftRs(units="atomic", **BASE, xc=xc).get_pauli_spin_susceptibility(
        atoms, electronic_temperature=150000.0, temperatures_k=[150000.0],
        exchange_enhanced=False, spin_g_factor=2.0, units="atomic")
    cli = _cli_report(tmp_path, controls)
    assert native["converged"] and native["exchange_enhanced"] is False
    assert native["thermodynamic_spin_bias"] == 0.0
    assert "zero spin reservoir bias" in native["model"]
    row = native["rows"][0]
    assert row["thermal_number_residual"] < 1e-12
    assert row["raw_charge_constraint_residual"] < 1e-12
    assert abs(sum(row["spin_electrons"]) - 3.0) < 1e-12
    assert abs(row["spin_electrons"][0] - 2.0) > 1e-3
    assert row["spin_chemical_potentials"] == [row["chemical_potential"]] * 2
    da, db = row["spin_dos_at_fermi_level"]
    np.testing.assert_allclose(row["pauli_susceptibility_per_cell"], da * db / (da + db),
                               rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(row["chemical_potential_response"], 0.5 * (da - db) / (da + db),
                               rtol=1e-12, atol=1e-14)
    for other in [generic, adapted, cli]:
        assert other["model"] == native["model"]
        assert other["units"] == native["units"]
        for key in ["chemical_potential", "chemical_potential_response", "spin_electrons",
                    "spin_dos_at_fermi_level", "pauli_susceptibility_per_cell", "heat_capacity",
                    "entropy", "band_energy", "delta_spin_numbers", "fixed_number_spin_dos"]:
            np.testing.assert_allclose(other["rows"][0][key], row[key], rtol=1e-9, atol=1e-12)
    converted = _to_ase_units(native)
    np.testing.assert_allclose(converted["rows"][0]["chemical_potential_response"],
                               Hartree * row["chemical_potential_response"], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(converted["rows"][0]["fixed_number_spin_dos"],
                               row["fixed_number_spin_dos"] / Hartree, rtol=1e-13, atol=1e-14)
    scaled = dft_rs.pauli_spin_susceptibility([1, 1, 1], POSITIONS, CELL.tolist(),
                                            **dict(controls, spin_g_factor=1.4))["rows"][0]
    np.testing.assert_allclose(scaled["pauli_susceptibility_per_cell"],
                               row["pauli_susceptibility_per_cell"] * 0.7**2, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(scaled["chemical_potential_response"],
                               row["chemical_potential_response"] * 0.7, rtol=1e-11, atol=1e-13)
    print(xc, "unequal rigid spin thermodynamics", row)
