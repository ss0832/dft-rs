# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Self-consistent thermal spin susceptibility on every public route."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
import dft_rs

POSITIONS = [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]
CELL = np.eye(3) * 9.0
BASE = dict(basis="sto-3g", k_mesh=[1, 1, 1], grid_level=1,
            grid_shell=1, overlap_cutoff=7.0, conv_tol=1e-12, guess="core")
SPIN = dict(exchange_enhanced=True, temperatures_k=[150000.0],
            spin_g_factor=2.0, response_reciprocal_cutoff=0.75)


@pytest.mark.parametrize("xc", ["svwn5", "pbe", "tpss"])
def test_screened_spin_matches_native_generic_ase_and_cli(tmp_path, xc):
    controls = dict(BASE, **SPIN, xc=xc)
    native = dft_rs.pauli_spin_susceptibility([1, 1], POSITIONS, CELL.tolist(), **controls)
    generic = dft_rs.run_feature("pauli-spin-susceptibility", [1, 1], POSITIONS,
                                 cell=CELL.tolist(), **controls)
    atoms = Atoms("H2", positions=np.asarray(POSITIONS) * Bohr, cell=CELL * Bohr, pbc=True)
    adapted = dft_rs.DftRs(units="atomic", **BASE, xc=xc).get_pauli_spin_susceptibility(
        atoms, **SPIN, units="atomic")
    xyz = tmp_path / "hydrogen.xyz"
    xyz.write_text("2\nspin response\n" + "".join(
        f"H {x*Bohr:.16f} {y*Bohr:.16f} {z*Bohr:.16f}\n" for x,y,z in POSITIONS), encoding="utf-8")
    result = subprocess.run([os.environ["DFT_RS_CLI"], "pauli-spin-susceptibility", str(xyz),
        "--cell", ";".join(",".join(f"{v*Bohr:.16f}" for v in row) for row in CELL),
        "--basis", "sto-3g", "--xc", xc, "--k-mesh", "1,1,1", "--grid-level", "1",
        "--grid-shell", "1", "--overlap-cutoff", "7", "--conv-tol", "1e-12", "--guess", "core",
        "--exchange-enhanced", "true", "--temperatures-k", "150000", "--spin-g-factor", "2",
        "--response-reciprocal-cutoff", "0.75", "--format", "json"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cli = json.loads(result.stdout)
    row = native["rows"][0]
    assert native["converged"] and row["response_converged"]
    assert row["enhanced_spin_susceptibility_per_cell"] > row["pauli_susceptibility_per_cell"] > 0
    assert row["bare_dos_consistency_residual"] < 1e-10
    for key in ["raw_charge_constraint_residual", "raw_spin_density_cancellation_residual",
                "raw_density_hermitian_residual"]:
        assert row[key] < 1e-10
    assert max(row["response_residuals"]) <= 1e-10
    assert abs(row["electrons"] - 2.0) < 1e-10
    for other in [generic, adapted, cli]:
        assert other["units"] == native["units"]
        assert other["model"] == native["model"]
        for key in ["enhanced_spin_susceptibility_per_cell", "enhanced_spin_susceptibility_volume_si",
                    "pauli_susceptibility_per_cell", "enhancement_factor", "delta_spin_numbers",
                    "reference_energy", "reference_free_energy"]:
            np.testing.assert_allclose(other["rows"][0][key], row[key], rtol=1e-9, atol=1e-12)
    scaled = dft_rs.pauli_spin_susceptibility([1,1], POSITIONS, CELL.tolist(),
        **dict(controls, spin_g_factor=1.3))["rows"][0]
    np.testing.assert_allclose(scaled["enhanced_spin_susceptibility_per_cell"],
        row["enhanced_spin_susceptibility_per_cell"] * (1.3/2)**2, rtol=1e-10, atol=1e-12)
    print(xc, "enhanced spin", row)


def test_screened_spin_control_validation():
    for extra in [dict(temperatures_k=[0]), dict(spin_g_factor=0),
                  dict(response_reciprocal_cutoff=0), dict(fermi_k_mesh=[1,1])]:
        with pytest.raises(ValueError):
            dft_rs.pauli_spin_susceptibility([1,1], POSITIONS, CELL.tolist(),
                **(dict(BASE, **SPIN) | extra))
