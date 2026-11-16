# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Public routes for the charged 3D spinor orbital current calculation."""

import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms

from dft_rs import DftRs, orbital_susceptibility_q
from dft_rs.units import BOHR_IN_ANGSTROM


POSITIONS = np.array([[0.17, -0.21, 0.34], [1.63, 0.08, 0.16]])
CELL = np.array([[4.4, 0.17, -0.31], [0.31, 6.0, 0.26], [-0.19, 0.38, 6.5]])
Q = [0.5, 0.0, 0.0]


@pytest.mark.parametrize(("xc", "reference", "spin_bias"), [
    ("svwn5", "gks", True),
    ("pbe", "gks", False),
    ("tpss", "gks", False),
    ("m06l", "gks", False),
    ("wb97x-d", "gks", False),
    ("hf", "ghf", True),
])
def test_charged_spinor_orbital_current_agrees_across_public_routes(tmp_path, xc, reference, spin_bias):
    basis = tmp_path / "heh.g94"
    basis.write_text("He 0\nS 1 1.0\n0.9 1.0\n****\nH 0\nS 1 1.0\n0.72 1.0\n****\n",
                     encoding="ascii")
    controls = dict(
        basis=str(basis), xc=xc, reference=reference, charge=1, charge_compensation="uniform",
        k_mesh=[2, 1, 1], q_points=Q + [-q for q in Q], overlap_cutoff=6.0,
        grid_cutoff=8.0, xc_ao_cutoff=10.0,
        grid_level=2, grid_shell=2, grid_prune=False, grid_weight_cutoff=0.0,
        conv_tol=2e-14, conv_tol_dp=2e-13, conv_tol_grad=1e-12, response_tolerance=1e-12,
        max_iter=200, damping=0.25, diis_size=12, cpu_threads=1,
        field_wave_indices=[1, 0, 0], soc_scale=0.0, nuclear_soc_scale=0.0,
    )
    if spin_bias:
        controls.update(spin_field_real=[0.05, -0.01, 0.04], spin_field_imag=[0.02, 0.045, -0.03])
    if xc in {"hf", "wb97x-d"}:
        controls["exchange_cutoff"] = 6.0
    atoms = Atoms("HeH", positions=POSITIONS * BOHR_IN_ANGSTROM,
                  cell=CELL * BOHR_IN_ANGSTROM, pbc=True)
    geometry = tmp_path / "heh.xyz"
    geometry.write_text("2\nSkew helium hydride ion cell\n" + "".join(
        symbol + " " + " ".join(f"{x:.17g}" for x in row) + "\n"
        for symbol, row in zip(atoms.get_chemical_symbols(), atoms.positions, strict=True)
    ), encoding="ascii")
    cell = ";".join(",".join(f"{x:.17g}" for x in row) for row in atoms.cell)
    command = [os.environ["DFT_RS_CLI"], "orbital-susceptibility-q", str(geometry),
               "--cell", cell, "--format", "json"]
    for key, value in controls.items():
        if isinstance(value, list):
            value = ",".join(map(str, value))
        elif isinstance(value, bool):
            value = str(value).lower()
        command.extend(("--" + key.replace("_", "-"), str(value)))
    cli = subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    assert any("[spinor-scf]" in line and "residual=" in line for line in cli.stderr.splitlines())
    assert any("[spinor-response]" in line and "residual=" in line for line in cli.stderr.splitlines())
    native = orbital_susceptibility_q([2, 1], POSITIONS, CELL, **controls)
    ase_controls = dict(controls)
    calculator = DftRs(units="atomic", reference=ase_controls.pop("reference"))
    ase = calculator.get_orbital_susceptibility_q(atoms, **ase_controls)
    with pytest.raises((ValueError, RuntimeError), match="use the core guess and CDIIS"):
        calculator.get_orbital_susceptibility_q(atoms, adiis_until_error=0.1, **ase_controls)
    point = native["points"][0]
    for report in (native, ase, json.loads(cli.stdout)):
        assert report["converged"] and report["reference_type"] == reference
        assert report["units"] == native["units"]
        assert report["orbital_residual"] < 1e-12
        np.testing.assert_allclose(report["energy"], native["energy"], atol=2e-11, rtol=0)
        current, reverse = report["points"]
        axis_sign = np.diag([1.0, -1.0])
        for key in ("chi_transverse", "paramagnetic_transverse", "uniform_transverse"):
            np.testing.assert_allclose(current[key], point[key], atol=1e-11, rtol=0)
            np.testing.assert_allclose(reverse[key], axis_sign @ current[key] @ axis_sign,
                                       atol=1e-11, rtol=0)
        assert np.max(current["response_residuals"]) <= 1e-12
        assert current["reciprocity_defect"] < 1e-11
        assert current["imaginary_defect"] < 1e-11
        np.testing.assert_allclose(current["chi"], np.diag(current["chi_transverse"]), atol=0, rtol=0)
    assert abs(point["chi_transverse"][0][1]) > 1e-9
    print("HeH+ periodic orbital current", json.dumps(native))


def test_spinor_current_rejects_unsupported_reference_conditions_before_scf():
    method = dict(xc="svwn5", reference="gks", charge=1, k_mesh=[2, 1, 1], q_points=Q)
    with pytest.raises((ValueError, RuntimeError), match="finite-temperature transverse current"):
        orbital_susceptibility_q([2, 1], POSITIONS, CELL, electronic_temperature=300.0, **method)
    with pytest.raises((ValueError, RuntimeError), match="nonlocal spin-orbit operator"):
        orbital_susceptibility_q([2, 1], POSITIONS, CELL, nuclear_soc_scale=1.0, **method)
