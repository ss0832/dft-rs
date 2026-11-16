# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Named functional recipes reach native Python, ASE and the canonical CLI."""

import json
import os
from pathlib import Path
import subprocess

from ase import Atoms
import dft_rs
import numpy as np
import pytest

from dft_rs.units import BOHR_IN_ANGSTROM


@pytest.mark.parametrize("multiplicity,charge", [(1, 0), (2, 1)])
@pytest.mark.parametrize("xc", ["blyp35", "wc04", "wp04", "pkzb", "tpssloc", "task", "mtask", "edf2", "dsd-pbeb95", "wb2plyp", "wb2gp-plyp", "rsx-qidh", "rsx-0dh", "n12-sx", "dldf", "b98", "thcth", "thcthhyb", "bmk", "pbe1kcis", "mpw1kcis", "tm", "revtm", "wb97x-d3"])
def test_named_functionals_common_driver_native_ase_and_cli(xc, multiplicity, charge, tmp_path):
    _exercise_named_routes(xc, multiplicity, charge, tmp_path, "gradient")


@pytest.mark.parametrize("multiplicity,charge", [(1, 0), (2, 1)])
@pytest.mark.parametrize("xc", ["n12-sx"])
def test_n12_sx_hessian_common_driver_routes(xc, multiplicity, charge, tmp_path):
    _exercise_named_routes(xc, multiplicity, charge, tmp_path, "hessian")


@pytest.mark.parametrize("multiplicity,charge", [(1, 0), (2, 1)])
def test_bmk_hessian_common_driver_routes(multiplicity, charge, tmp_path):
    _exercise_named_routes("bmk", multiplicity, charge, tmp_path, "hessian")


def _exercise_named_routes(xc, multiplicity, charge, tmp_path, calculation):
    positions = np.array([[0.17, -0.11, 0.23], [1.43, 0.81, -0.62],
                          [-0.63, 1.59, -0.43], [-1.12, -1.36, -0.71]])
    numbers = [7, 1, 1, 1]
    if xc in ["dsd-pbeb95", "wb2plyp", "wb2gp-plyp", "rsx-qidh", "rsx-0dh"]:
        # Route parity needs a small noncoplanar system. DSD ammonia force
        # fits are in p304; the range B2 parents have full H4 fits in p305.
        numbers = [1, 1, 1, 1]
    controls = dict(xc=xc, basis="sto-3g", multiplicity=multiplicity,
                    charge=charge, unrestricted=multiplicity == 2, grid_level=3,
                    grid_weight_cutoff=0.0, conv_tol=1e-12, conv_tol_dp=1e-10,
                    conv_tol_grad=1e-10, dispersion="none")
    if xc in ("dsd-pbeb95", "wb97x-d3"):
        controls.pop("dispersion")  # Exercise the defining pair correction.
    native = getattr(dft_rs, calculation)(numbers, positions.tolist(), **controls)
    common = dft_rs.run_feature(calculation, numbers, positions.tolist(), **controls)
    assert native["converged"] and common["converged"]
    np.testing.assert_allclose(native[calculation], common[calculation], rtol=0, atol=1e-11)
    atoms = Atoms(numbers=numbers, positions=positions * BOHR_IN_ANGSTROM)
    calculator = dft_rs.DftRs(units="atomic", **controls)
    # Explicit common-driver reports use the declared atomic derivative units.
    adapted = getattr(calculator, f"get_{calculation}_report")(atoms, **controls)
    np.testing.assert_allclose(native[calculation], adapted[calculation], rtol=0, atol=1e-10)
    assert abs(native["energy"] - adapted["energy"]) < 1e-10
    executable = os.environ.get("DFT_RS_CLI")
    assert executable and Path(executable).is_file(), "DFT_RS_CLI selects the freshly built canonical CLI"
    xyz = tmp_path / ("c1_tetrahydrogen.xyz" if xc in ["dsd-pbeb95", "wb2plyp", "wb2gp-plyp", "rsx-qidh", "rsx-0dh"] else "c1_ammonia.xyz")
    xyz.write_text(f"4\n{xc} C1 route geometry\n" + "".join(
        symbol + " " + " ".join(f"{x:.17g}" for x in point) + "\n"
        for symbol, point in zip(atoms.get_chemical_symbols(), positions * BOHR_IN_ANGSTROM)
    ), encoding="utf-8")
    args = [executable, calculation, str(xyz), "--format", "json"]
    for key, value in controls.items():
        args += ["--" + key.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)]
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    cli = json.loads(result.stdout)
    np.testing.assert_allclose(native[calculation], cli[calculation], rtol=0, atol=1e-10)
    assert abs(native["energy"] - cli["energy"]) < 1e-10
    assert "SCF" in result.stderr
    if xc == "wb97x-d3":
        d3 = common["energy_components"]["dispersion"]
        assert d3 < 0.0
        for report in (adapted, cli):
            assert abs(report["energy_components"]["dispersion"] - d3) < 1e-13
        print(f"{xc} charge={charge} energy={common['energy']:.16e} D3={d3:.16e} Hartree")
    for label, report in [("common", common), ("ASE", adapted), ("CLI", cli)]:
        defect = np.max(np.abs(np.asarray(native[calculation]) - report[calculation]))
        energy_defect = abs(native["energy"] - report["energy"])
        unit = "Hartree/Bohr" if calculation == "gradient" else "Hartree/Bohr^2"
        print(f"{xc} multiplicity={multiplicity} {label}: {calculation} defect={defect:.12e} "
              f"{unit}; energy defect={energy_defect:.12e} Hartree")
    print("\n".join(result.stderr.splitlines()[-3:]))
