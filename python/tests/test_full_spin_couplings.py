# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""C1 UKS/GKS electronic Ramsey tensors on every public calculation surface."""

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs import DftRs
from dft_rs.units import BOHR_IN_ANGSTROM, HARTREE_IN_EV


@pytest.mark.parametrize("surface", ["native", "ase", "cli", "cli-alias"])
def test_hf_ramsey_grid_resolution_reaches_each_public_surface(surface, tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from test_fixed_populations import cli_report

    positions = np.array([[.13, -.21, .17], [1.48, .19, -.11]])
    atoms = Atoms("H2", positions=positions * BOHR_IN_ANGSTROM)
    options = dict(xc="hf", basis="sto-3g", dispersion="none", grid_weight_cutoff=0.,
        conv_tol=1e-13, conv_tol_dp=1e-12, cpu_threads=1, memory_mb=256)
    if surface == "cli-alias":
        primary = Path(os.environ["DFT_RS_CLI"])
        monkeypatch.setenv("DFT_RS_CLI", os.environ.get("DFT_RS_CLI_ALIAS",
            str(primary.with_name("dft_rs_cli"))))
    reports = []
    for level in (0, 1):
        controls = dict(options, grid_level=level)
        if surface == "native":
            report = dft_rs.spin_spin_couplings([1, 1], positions.tolist(), **controls)
        elif surface == "ase":
            report = DftRs(units="atomic").get_spin_spin_couplings(atoms, **controls)
        else:
            report = cli_report(tmp_path, "spin-spin-couplings", atoms, controls)
        assert report["converged"]
        reports.append(report)
    np.testing.assert_array_equal(reports[0]["energy"], reports[1]["energy"])
    difference = np.max(np.abs(np.asarray(reports[0]["diamagnetic_tensors"])
                               - np.asarray(reports[1]["diamagnetic_tensors"])))
    assert difference > 1e-18, "HF property integration must not silently use the default grid"


@pytest.mark.parametrize("feature", ["periodic-field-scf", "periodic-spinor-scf"])
@pytest.mark.parametrize("name", ["operator_tolerance", "grid_tolerance"])
@pytest.mark.parametrize("value", [0., -1., float("inf")])
def test_dimensional_grid_tolerances_are_validated_before_basis(feature, name, value):
    with pytest.raises(ValueError, match=name):
        dft_rs.run_feature(feature, [2], [[0., 0., 0.]],
            cell=np.diag([8., 9., 10.]).tolist(), basis="missing-grid-contract-basis",
            xc="hf", **{name: value})


@pytest.mark.parametrize("name,value", [
    ("grid_preset", "fine"), ("grid_angular_scheme", "product"),
    ("grid_weight_cutoff", 0.),
])
def test_magnetic_tensor_grid_refuses_unmapped_atomic_grid_controls(name, value):
    with pytest.raises(NotImplementedError, match="atomic-grid"):
        dft_rs.run_feature("periodic-magnetic-scf", [2], [[0., 0., 0.]],
            cell=np.diag([8., 9., 10.]).tolist(), basis="missing-grid-contract-basis",
            xc="hf", **{name: value})


@pytest.mark.parametrize("reference", ["uks", "ghf"])
def test_c1_full_spin_couplings_share_native_ase_and_cli(reference, tmp_path):
    from test_fixed_populations import cli_report

    positions = np.array([[.13, -.21, .19], [1.57, .16, -.08],
                          [-.19, 1.93, .37], [.42, .29, 2.28]])
    controls = dict(reference=reference, xc="hf", basis="sto-3g", charge=1,
                    multiplicity=2, dispersion="none", grid_weight_cutoff=0.,
                    conv_tol=1e-13, conv_tol_grad=1e-13, conv_tol_dp=1e-13,
                    max_iter=500, response_tolerance=1e-11, nuclei=[0, 1],
                    nuclear_g=[5.5856946893] * 4,
                    spin_response_constraint="fixed-spin-direction")
    if reference == "ghf":
        real = np.zeros((8, 8))
        imag = np.zeros_like(real)
        for i in range(4):
            real[i, i] = -.013 * (i + 1)
            real[4+i, 4+i] = -real[i, i]
            real[i, 4+i] = real[4+i, i] = .007 * (i + 1)
            imag[i, 4+i] = .009 * (4 - i)
            imag[4+i, i] = -imag[i, 4+i]
        controls.update(gks_operator_real=real.tolist(),
                        gks_operator_imaginary=imag.tolist())
    native = dft_rs.spin_spin_couplings([1] * 4, positions.tolist(), **controls)
    atoms = Atoms("H4", positions=positions * BOHR_IN_ANGSTROM)
    adapted = DftRs(units="atomic").get_spin_spin_couplings(atoms, **controls)
    physical_controls = dict(controls)
    for key in ["conv_tol", "conv_tol_grad", "gks_operator_real", "gks_operator_imaginary"]:
        if key in physical_controls:
            physical_controls[key] = (np.asarray(physical_controls[key]) * HARTREE_IN_EV).tolist()
    physical = DftRs().get_spin_spin_couplings(atoms, **physical_controls)
    command = cli_report(tmp_path, "spin-spin-couplings", atoms, controls)
    fields = ["reduced_couplings", "fermi_contact_tensors", "spin_dipole_tensors",
              "paramagnetic_tensors", "diamagnetic_tensors", "spin_orbit_mixed_tensors"]
    for report in [native, adapted, physical, command]:
        assert report["spin_response_constraint"] == "fixed-spin-direction"
        assert report["spin_response_constraint_rank"] >= 2
        assert report["response_residual"] < 1e-10
        assert np.max(report["spin_response_constraint_residuals"]) < 1e-10
        np.testing.assert_allclose(report["reduced_couplings"],
                                   np.sum([report[key] for key in fields[1:]], axis=0),
                                   rtol=0., atol=1e-18)
    for report in [adapted, physical, command]:
        for field in fields:
            np.testing.assert_allclose(report[field], native[field], rtol=1e-7, atol=1e-16)
    print("full-spin J public", reference, "rank", native["spin_response_constraint_rank"],
          "relative_residual", native["response_residual"], "constraint_residual",
          float(np.max(native["spin_response_constraint_residuals"])), "reference_fock",
          native["response_reference_fock_defect"], flush=True)
