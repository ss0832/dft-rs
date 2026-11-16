# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""One optimizer control reaches the native, ASE and installed CLI drivers."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs import DftRs
from dft_rs.units import BOHR_IN_ANGSTROM, HARTREE_IN_EV

POSITIONS = np.array([[.17, -.11, .23], [1.57, .18, .33],
                      [.31, 4.74, -.45], [1.60, 5.01, -.61]])
OPTIONS = dict(xc="pbe", basis="sto-3g", dispersion="none", grid_level=1,
               grid_weight_cutoff=0.0, conv_tol=1e-10, conv_tol_dp=1e-10,
               conv_tol_grad=1e-9, max_iter=300, scf_optimizer="trah")


def compare_public_routes(numbers, positions, options, tmp_path):
    native = dft_rs.run_feature("energy", numbers, positions.tolist(), **options)
    atoms = Atoms(numbers=numbers, positions=positions*BOHR_IN_ANGSTROM)
    adapted = DftRs(units="atomic").get_energy_report(atoms, **options)
    assert native["converged"] and adapted["converged"]
    assert abs(native["energy"]-adapted["energy"]) < 1e-10
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for installed CLI acceptance"
    xyz = tmp_path/"trah.xyz"
    xyz.write_text(f"{len(numbers)}\nTRAH control parity\n"+"".join(
        symbol+" "+" ".join(f"{x:.17g}" for x in point)+"\n"
        for symbol, point in zip(atoms.get_chemical_symbols(), atoms.positions)),
        encoding="utf-8")
    command = [executable, "energy", str(xyz), "--format", "json"]
    for key, value in options.items():
        command.extend(["--"+key.replace("_", "-"), str(value)])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert abs(report["energy"]-native["energy"]) < 1e-10
    assert "iter" in result.stderr and "residual" in result.stderr
    assert report["reference_type"] == options["reference"]
    print(f"TRAH public {options['reference']} energy={native['energy']:.15e} "
          f"ASE_defect={abs(native['energy']-adapted['energy']):.12e} "
          f"CLI_defect={abs(native['energy']-report['energy']):.12e}")
    return native, adapted, report


@pytest.mark.parametrize("reference", ["rks", "uks", "roks", "gks"])
def test_trah_native_ase_and_installed_cli(reference, tmp_path):
    compare_public_routes([1]*4, POSITIONS, dict(OPTIONS, reference=reference), tmp_path)


def test_stationary_non_aufbau_determinant_reports_occupation_order(tmp_path):
    positions = np.array([[0., 0., 0.], [0., 0., .957], [.927, 0., -.240],
                          [0., 0., 2.980], [0., .757, 3.549],
                          [0., -.757, 3.549]]) / BOHR_IN_ANGSTROM
    options = dict(OPTIONS, reference="uks", charge=1, multiplicity=2,
                   conv_tol=1e-12, conv_tol_grad=1e-10, conv_tol_dp=1e-10,
                   max_iter=120, scf_optimizer="auto")
    reports = compare_public_routes([8, 1, 1, 8, 1, 1], positions, options, tmp_path)
    for report in reports:
        gaps = report["occupied_virtual_gaps"]
        assert gaps[0] > 0 and gaps[1] < -1e-7
        assert report["units"]["occupied_virtual_gaps"] == "hartree"
        assert any("non-Aufbau" in warning for warning in report["warnings"])
        np.testing.assert_allclose(gaps, reports[0]["occupied_virtual_gaps"],
                                   rtol=1e-9, atol=1e-10)
    print("Stationary determinant occupation gaps", reports[0]["occupied_virtual_gaps"])


def test_roks_double_hybrid_spin_components_public_routes(tmp_path):
    positions = np.array([[.17,-.11,.23],[1.77,.18,-.53],
                          [-.31,1.74,-.45],[-1.2,-1.31,-.61]])
    options = dict(OPTIONS, xc="b2plyp", reference="roks", charge=1,
                   multiplicity=2, conv_tol=1e-12, conv_tol_dp=1e-11,
                   conv_tol_grad=1e-10, max_iter=500)
    reports = compare_public_routes([7,1,1,1], positions, options, tmp_path)
    for report in reports:
        for key in ["pt2_same_spin_alpha", "pt2_same_spin_beta", "pt2_opposite_spin"]:
            assert abs(report[key]) > 1e-7
            assert abs(report[key]-reports[0][key]) < 1e-10
            assert report["units"][key] == "hartree"
        for key in ["reference_common_orbital_residual",
                    "reference_semicanonical_residual", "pt2_amplitude_residual"]:
            assert report[key] < 1e-9
            assert report["units"][key] == "hartree"
        assert report["pt2_excitation_model"] == "semicanonical doubles only"
        assert abs(report["energy"]-report["reference_energy"]
                   -report["pt2_correlation"]) < 1e-12
    atoms = Atoms(numbers=[7,1,1,1], positions=positions*BOHR_IN_ANGSTROM)
    # Keep the physical SCF gates identical when changing the API unit system.
    ase_options = dict(options, conv_tol=options["conv_tol"]*HARTREE_IN_EV,
                       conv_tol_grad=options["conv_tol_grad"]*HARTREE_IN_EV)
    ase_units = DftRs().get_energy_report(atoms, **ase_options)
    assert abs(ase_units["energy"]-reports[0]["energy"]*HARTREE_IN_EV) < 1e-8
    assert ase_units["units"]["reference_eigen_residual"] == "dimensionless"
    assert abs(ase_units["reference_eigen_residual"]
               -reports[0]["reference_eigen_residual"]) < 1e-12
    print("ROKS doubles-only public report", reports[0])


def test_trah_incompatible_controls_fail_explicitly():
    with pytest.raises(ValueError, match="scf_optimizer"):
        dft_rs.run_feature("energy", [1]*4, POSITIONS.tolist(),
                          **dict(OPTIONS, scf_optimizer="unknown"))
    with pytest.raises((ValueError, RuntimeError), match="forced trust-region augmented-Hessian SCF"):
        dft_rs.run_feature("energy", [1]*4, POSITIONS.tolist(),
                          **dict(OPTIONS, no_soscf=True))
