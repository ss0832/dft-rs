# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""C1 g/h/i HF energies on native, ASE and canonical CLI surfaces.

Every case also compares an independent ground-state energy. Complete
nuclear derivative acceptance is separate in p299; energy route agreement
does not certify an unmeasured derivative/reference combination.
"""
import json
import os
from pathlib import Path
import subprocess

from ase import Atoms
import numpy as np
import pytest

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM, HARTREE_IN_EV


ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads((ROOT / "tests/oracle_data/high_angular_hf_terms.json")
                  .read_text(encoding="utf-8"))["cases"]


def _case_inputs(case):
    ecp, unrestricted = case["scalar_ecp"], case["unrestricted"]
    basis = ROOT / f'tests/data/c1_{case["shell"]}_model.g94'
    controls = dict(xc="hf", basis=str(basis),
                    reference="uks" if unrestricted else "rks", unrestricted=unrestricted,
                    charge=(9 if ecp else 11)-case["electrons"],
                    multiplicity=2 if unrestricted else 1, dispersion="none",
                    conv_tol=1e-14, conv_tol_grad=1e-14, conv_tol_dp=1e-14,
                    max_iter=500, grid_weight_cutoff=0.0, guess="core")
    numbers = [15 if ecp else 7, 2, 1, 1]
    positions = np.array(case["positions"])
    return numbers, positions, controls


def _public_reports(case, tmp_path):
    numbers, positions, controls = _case_inputs(case)
    ecp = case["scalar_ecp"]
    native = dft_rs.run_feature("energy", numbers, positions.tolist(), **controls)
    atoms = Atoms(numbers=numbers, positions=positions*BOHR_IN_ANGSTROM)
    calculator = dft_rs.DftRs()
    adapted = calculator.get_feature_report("energy", atoms, units="atomic", **controls)
    physical_controls = dict(controls,
                             conv_tol=controls["conv_tol"]*HARTREE_IN_EV,
                             conv_tol_grad=controls["conv_tol_grad"]*HARTREE_IN_EV)
    physical = calculator.get_feature_report("energy", atoms, **physical_controls)
    xyz = tmp_path / "high_angular.xyz"
    xyz.write_text("4\nC1 high-angular numerical model\n" + "".join(
        symbol + " " + " ".join(f"{value:.17g}" for value in point) + "\n"
        for symbol, point in zip(["P" if ecp else "N", "He", "H", "H"],
                                 positions*BOHR_IN_ANGSTROM)), encoding="utf-8")
    command = [os.environ["DFT_RS_CLI"], "energy", str(xyz), "--format", "json"]
    for key, value in controls.items():
        command.extend(["--"+key.replace("_", "-"), str(value).lower()
                        if isinstance(value, bool) else str(value)])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "residual" in result.stderr
    canonical = json.loads(result.stdout)
    return native, adapted, physical, canonical


@pytest.mark.parametrize("case", CASES, ids=[case["case"] for case in CASES])
def test_high_angular_energy_public_routes(case, tmp_path):
    native, adapted, physical, canonical = _public_reports(case, tmp_path)
    maximum = 0.0
    for report in [native, adapted, canonical]:
        assert report["converged"]
        assert report["units"]["energy"] == "hartree"
        defect = abs(report["energy"]-case["energy"])
        assert defect < 2e-10
        maximum = max(maximum, defect)
    assert physical["units"]["energy"] == "eV"
    np.testing.assert_allclose(physical["energy"], native["energy"]*HARTREE_IN_EV,
                               atol=2e-9, rtol=0)
    print(case["case"], "all_public_energy_maximum_defect_hartree", maximum)


def test_i_shell_uhf_newton_stall_recovery_public_routes(tmp_path):
    """Recover the retained C1 failure without changing any SCF gate."""
    case = dict(next(case for case in CASES if case["case"] == "i_ae_uhf"))
    # Frozen generic direction 285 of the shared twelve-dimensional cubic
    # interpolation set; this is fixture data, not a symmetry reconstruction.
    direction = np.array([
        0.12347746788641105, 0.8885748435478132, 0.1284165666018675,
        0.1308861159595957, 0.13335566531732396, 0.13582521467505218,
        0.13829476403278038, 0.14076431339050863, 0.1432338627482368,
        0.14570341210596505, 0.14817296146369324, 0.15064251082142147,
    ]).reshape(4, 3)
    case["positions"] = (np.array(case["positions"]) + 0.04*direction).tolist()
    numbers, positions, controls = _case_inputs(case)
    assert controls["conv_tol"] == controls["conv_tol_grad"] == controls["conv_tol_dp"] == 1e-14
    reference = dft_rs.run_feature("energy", numbers, positions.tolist(),
                                   **dict(controls, no_soscf=True))
    assert reference["converged"]
    native, adapted, physical, canonical = _public_reports(case, tmp_path)
    for report in [native, adapted, canonical]:
        assert report["converged"]
        assert report["units"]["energy"] == "hartree"
        assert abs(report["energy"]-reference["energy"]) < 2e-12
    assert physical["converged"]
    assert physical["units"]["energy"] == "eV"
    np.testing.assert_allclose(physical["energy"], reference["energy"]*HARTREE_IN_EV,
                               atol=2e-9, rtol=0)
    print("i AE UHF displaced SCF recovery: native, ASE atomic/physical, CLI and pure-CDIIS reference pass")
