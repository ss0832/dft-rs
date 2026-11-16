# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Public full-RPA derivative selection and total/excitation Hessian semantics."""
import json
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM, HARTREE_IN_EV


def controls():
    return dict(xc="hf", basis="sto-3g", td_method="rpa", nstates=1,
                root=1, conv_tol=1e-12, conv_tol_dp=1e-11,
                td_tolerance=1e-12, response_tolerance=1e-12,
                grid_weight_cutoff=0.0)


def check(report):
    assert report["td_method"] == "rpa"
    assert report["provenance"] == "analytic"
    assert report["response_converged"]
    assert "total energy" in report["hessian_quantity"]
    assert "excitation energy only" in report["excitation_hessian_quantity"]
    np.testing.assert_allclose(
        report["hessian"], np.array(report["ground_hessian"]) + report["excitation_hessian"],
        atol=1e-12, rtol=0)


@pytest.mark.parametrize("unit_mode", ["atomic", "ase"])
def test_excited_hessian_adapter_preserves_total_and_excitation_quantities(monkeypatch, unit_mode):
    calls = []
    registry = dft_rs.feature_registry()
    ground = np.eye(3) * 0.75
    excitation = np.eye(3) * -0.25

    def backend(feature, *args, **kwargs):
        calls.append(feature)
        return dft_rs.Report(feature=feature, td_method="rpa", provenance="analytic", response_converged=True,
            hessian=(ground + excitation).tolist(), ground_hessian=ground.tolist(),
            excitation_hessian=excitation.tolist(), hessian_quantity="excited-state total energy",
            excitation_hessian_quantity="excitation energy only",
            units={"hessian": "hartree/bohr^2, of excited_state_energy",
                   "ground_hessian": "hartree/bohr^2, of ground-state energy",
                   "excitation_hessian": "hartree/bohr^2"})

    monkeypatch.setattr(dft_rs.native, "run_feature", backend)
    monkeypatch.setattr(dft_rs.native, "feature_registry", lambda: registry)
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]])
    report = dft_rs.DftRs(units=unit_mode).get_td_hessian(atoms, **controls())
    check(report)
    factor = 1.0 if unit_mode == "atomic" else HARTREE_IN_EV / BOHR_IN_ANGSTROM**2
    for name, tensor in (("hessian", ground + excitation), ("ground_hessian", ground),
                         ("excitation_hessian", excitation)):
        np.testing.assert_allclose(report[name], tensor * factor, atol=0, rtol=1e-15)
    assert report["hessian_quantity"] == "excited-state total energy"
    assert report["excitation_hessian_quantity"] == "excitation energy only"
    assert calls == ["td-hessian"]


@pytest.mark.parametrize("surface", ["native", "ase", "cli"])
def test_rpa_hessian_quantities_are_explicit(tmp_path, surface):
    xyz = [[0.0, 0.0, 0.0], [0.2, -0.1, 1.4]]
    if surface == "native":
        report = dft_rs.run_feature("td-hessian", [1, 1], xyz, **controls())
    elif surface == "ase":
        atoms = Atoms("H2", positions=np.array(xyz) * 0.529177210903)
        report = dft_rs.DftRs(units="atomic").get_feature_report("td-hessian", atoms, **controls())
    else:
        cli = os.environ.get("DFT_RS_CLI")
        if cli is None:
            pytest.skip("set DFT_RS_CLI for linked-artifact acceptance")
        path = tmp_path / "hydrogen.xyz"
        path.write_text("2\nExplicit total and excitation curvatures\n" + "\n".join(
            "H " + " ".join(map(str, np.array(row) * 0.529177210903)) for row in xyz) + "\n")
        command = [cli, "td-hessian", str(path), "--format", "json", "--cpu-threads", "1"]
        for key, value in controls().items():
            command.extend(["--" + key.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)])
        report = json.loads(subprocess.run(command, text=True, capture_output=True, check=True).stdout)
    check(report)


def test_rpa_default_gradient_uses_paired_path_and_explicit_tda_formula_is_refused():
    xyz = [[0, 0, 0], [0.2, -0.1, 1.4]]
    report = dft_rs.run_feature("td-state-gradient", [1, 1], xyz, **controls())
    assert report["td_gradient_route"] == "path"
    assert report["td_method"] == "rpa"
    with pytest.raises(NotImplementedError, match="RPA"):
        dft_rs.run_feature("td-state-gradient", [1, 1], xyz,
                          **(controls() | dict(td_gradient_route="z-vector")))
