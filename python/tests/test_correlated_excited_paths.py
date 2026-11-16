# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""The defined correlated TDA nuclear surface through all public routes."""
import json
import os
from pathlib import Path
import subprocess

from ase import Atoms
import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM
import numpy as np
import pytest


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("feature,native_name,ase_name", [
    ("td-state-gradient", "td_state_gradient", "get_td_state_gradient"),
    ("td-hessian", "td_hessian", "get_td_hessian"),
])
def test_correlated_excited_nuclear_public_routes(tmp_path, unrestricted, feature,
                                                native_name, ase_name):
    positions = np.array([[0.17, -0.11, 0.23], [1.77, 0.18, -0.53],
                          [-0.31, 1.74, -0.45], [-1.2, -1.31, -0.61]])
    options = dict(xc="hf", basis="sto-3g", guess="core", dispersion="none",
                   unrestricted=unrestricted, reference="uks" if unrestricted else "rks",
                   charge=1 if unrestricted else 2, multiplicity=2 if unrestricted else 1,
                   double_hybrid_model="custom", pt2_same_spin=0.17 if unrestricted else 0.2,
                   pt2_opposite_spin=0.31 if unrestricted else 0.2,
                   conv_tol=1e-13, conv_tol_dp=1e-13, conv_tol_grad=1e-12, max_iter=400,
                   grid_weight_cutoff=0.0, td_tolerance=1e-10, response_tolerance=1e-11,
                   td_memory_mb=128, pt2_memory_mb=128, root=1)
    if feature == "td-state-gradient":
        options["td_gradient_route"] = "path"
    native = getattr(dft_rs, native_name)([1]*4, positions.tolist(), **options)
    atoms = Atoms(numbers=[1]*4, positions=positions*BOHR_IN_ANGSTROM)
    adapted = getattr(dft_rs.DftRs(units="atomic", **options), ase_name)(atoms, **options)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable and Path(executable).is_file()
    xyz = tmp_path / "correlated_excited.xyz"
    xyz.write_text("4\nC1 correlated excited derivative model\n" + "".join(
        "H " + " ".join(f"{v:.17g}" for v in point) + "\n"
        for point in positions*BOHR_IN_ANGSTROM), encoding="utf-8")
    command = [executable, feature, str(xyz), "--format", "json"]
    for key, value in options.items():
        command += ["--"+key.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)
    assert completed.returncode == 0, completed.stderr
    assert "SCF" in completed.stderr
    for route, report in [("native", native), ("ASE", adapted), ("CLI", json.loads(completed.stdout))]:
        assert report["provenance"] == "analytic" and report["response_converged"]
        assert "second-order normal-ordered full-Coulomb TDA" in report["correlated_excitation_model"]
        assert report["td_spin"] == ("spin-resolved" if unrestricted else "singlet")
        assert report["td_method"] == "tda" and report["td_gradient_route"] == "path"
        assert report["response_residual"] < 1e-10
        assert report["propagator_equation_residual"] < 1e-10
        assert report["parent_pole_defect"] < 1e-10
        assert abs(report["excited_state_energy"] - report["energy"] - report["excitation_energy"]) < 1e-10
        assert abs(report["excitation_energy"] - report["parent_excitation_energy"]
                   - report["excitation_correlation_shift"]) < 1e-10
        assert abs(report["excitation_energy"]-native["excitation_energy"]) < 1e-10
        np.testing.assert_allclose(report["forces"], -np.asarray(report["gradient"]), atol=1e-14, rtol=0)
        maximum = 0.0
        for key in ["gradient", "excitation_gradient", "excitation_correlation_gradient"]:
            defect = np.max(np.abs(np.asarray(report[key])-native[key]))
            assert defect < 1e-9
            maximum = max(maximum, defect)
        if feature == "td-hessian":
            for key in ["hessian", "excitation_hessian", "excitation_correlation_hessian"]:
                defect = np.max(np.abs(np.asarray(report[key])-native[key]))
                assert defect < 1e-8
                maximum = max(maximum, defect)
        print(feature, unrestricted, route, "maximum_public_tensor_defect", maximum,
              "physical_equation_residual", report["propagator_equation_residual"])
