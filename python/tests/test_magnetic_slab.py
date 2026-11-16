# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Normal slab flux, physical open Coulomb boundary and aligned public APIs."""
import json
import math
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
import dft_rs


@pytest.mark.parametrize("model", ["hf", "charged_range", "svwn5", "blyp", "m06l", "weighted"])
@pytest.mark.parametrize("magnetic_model", ["finite_gamma", "crystal_bz"])
def test_normal_flux_slab_public_surfaces_and_physical_terms(model, magnetic_model, tmp_path):
    basis = tmp_path / "compact.gbs"
    basis.write_text("H 0\nS 1 1.0\n0.7 1.0\n****\n", encoding="utf-8")
    cell = np.diag([9., 9., 0.])
    positions = np.zeros((1, 3))
    field = [0., 0., 2 * np.pi / 81]
    settings = dict(xc="hf" if model == "charged_range" else model, basis=str(basis),
                    k_mesh=[2, 1, 1] if magnetic_model == "crystal_bz" else [1, 1, 1],
                    magnetic_model=magnetic_model, overlap_cutoff=3., magnetic_field=field,
                    magnetic_reciprocal_cutoff=14., magnetic_open_order=64,
                    nuclear_widths=[.3], magnetic_grid_order=48, grid_cutoff=8.,
                    ewald_alpha=.55, ewald_real_cutoff=14., ewald_reciprocal_cutoff=8.,
                    conv_tol=1e-13, conv_tol_dp=1e-12, conv_tol_grad=1e-11)
    if model != "charged_range":
        settings["reference"] = "ghf" if model == "hf" else "gks"
    if model == "charged_range":
        settings.update(charge=-1, exchange_coefficients=[.4, .3, .5],
                        charge_compensation="gaussian", countercharge_center=[0., 0., 0.],
                        countercharge_width=.6)
    if model == "weighted":
        settings.pop("xc")
        settings["xc_terms"] = [{"xc": "svwn5", "weight": .6}, {"xc": "hf", "weight": .25}]
    native = dft_rs.periodic_magnetic_scf([1], positions, cell, pbc=[True, True, False], **settings)
    assert native["converged"], (
        native["orbital_residual"], native["coulomb_operator_defect"],
        native["xc_grid_energy_defect"], native["xc_grid_operator_defect"],
        native["xc_grid_number_defect"], native["xc_diagnostics"],
    )
    reports = [native]
    if model in {"hf", "charged_range", "svwn5"}:
        reports.append(dft_rs.run_feature(
            "periodic-magnetic-scf", [1], positions, cell=cell,
            pbc=[True, True, False], **settings,
        ))
    atoms = Atoms("H", positions=positions * Bohr, cell=cell * Bohr, pbc=[True, True, False])
    reports.append(dft_rs.DftRs(units="atomic").get_periodic_magnetic_scf(atoms, **settings, units="atomic"))
    xyz = tmp_path / "magnetic-slab.xyz"
    xyz.write_text("1\nmagnetic slab\nH 0 0 0\n", encoding="utf-8")
    args = [os.environ["DFT_RS_CLI"], "periodic-magnetic-scf", str(xyz), "--cell",
            ";".join(",".join(f"{v * .529177210903:.17g}" for v in row) for row in cell),
            "--pbc", "1,1,0", "--format", "json"]
    for key, value in settings.items():
        if key == "xc_terms":
            value = json.dumps(value)
        elif isinstance(value, list):
            value = ",".join(map(str, value))
        args.extend(["--" + key.replace("_", "-"), str(value)])
    completed = subprocess.run(args, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "[spinor-scf]" in completed.stderr and "residual=" in completed.stderr
    reports.append(json.loads(completed.stdout))
    for report in reports:
        assert report["converged"] and report["dimension"] == 2
        assert report["flux_quanta"] == [1, 0, 0]
        assert report["pbc"] == [True, True, False]
        assert report["reference_type"] == settings.get("reference", "gks")
        assert report["particle_number_error"] < 1e-12
        assert report["orbital_residual"] < 1e-11
        assert report["coulomb_operator_defect"] < 1e-10
        assert report["image_operator_defect"] < 1e-10
        assert "continuous open-normal" in report["coulomb_boundary_convention"]
        assert report["units"] == native["units"] and report["model"] == native["model"]
        if model in {"blyp", "m06l"}:
            angular = report["xc_diagnostics"]
            assert angular["converged"]
            assert angular["angular_operator_defect"] < 1e-10
            assert report["xc_grid_operator_defect"] < 1e-10
            assert report["xc_grid_number_defect"] < 1e-10
            assert report["energy_components"]["semilocal"] < 0
            np.testing.assert_allclose(sum(report["xc_component_energies"]),
                                       report["energy_components"]["semilocal"],
                                       atol=1e-12, rtol=0)
        for key in ("energy", "free_energy", "nuclear_soc_energy", "spinor_energy_derivative",
                    "density_real", "density_imag", "fock_real", "fock_imag", "magnetization"):
            np.testing.assert_allclose(report[key], native[key], atol=1e-11, rtol=1e-10)
        np.testing.assert_allclose(report["energy"], sum(report["energy_components"].values()), atol=1e-13, rtol=0)
    electrons = 2 if model == "charged_range" else 1
    kinetic = electrons * (1.5 * .7 + field[2] ** 2 / (16 * .7))
    np.testing.assert_allclose(native["energy_components"]["kinetic"], kinetic, atol=1e-12, rtol=0)
    if model == "hf":
        if magnetic_model == "finite_gamma":
            np.testing.assert_allclose(native["energy_components"]["hartree"], -native["energy_components"]["full_exchange"], atol=1e-13, rtol=0)
        else:
            # Independent erfc normal integral on the enlarged transfer plane.
            # Its area is multiplied by Nk, including the finite normal zero mode.
            lengths = np.array([18., 9.])
            waves = np.array([[2*np.pi*i/lengths[0], 2*np.pi*j/lengths[1]]
                              for i in range(-100, 101) for j in range(-100, 101)
                              if i or j])
            norm = np.linalg.norm(waves, axis=1)
            area = np.prod(lengths)
            eta = .5
            exact = (2*np.pi/area*sum(math.erfc(q/(2*math.sqrt(.7)))/q for q in norm)
                     - 2*math.sqrt(np.pi)/(area*math.sqrt(.7))
                     + math.sqrt(np.pi)*eta
                     - 2*np.pi/area*np.sum(np.exp(-norm**2/(4*eta**2))/norm))
            np.testing.assert_allclose(-2*native["energy_components"]["full_exchange"], exact, atol=1e-10, rtol=0)
            assert native["exchange_transfer_modes"] > 0
    if model == "weighted":
        controls = {key: value for key, value in settings.items() if key != "xc_terms"}
        pure = {
            xc: dft_rs.periodic_magnetic_scf(
                [1], positions, cell, pbc=[True, True, False], xc=xc, **controls
            )
            for xc in ("svwn5", "hf")
        }
        # One occupied spinor in one spatial AO gives the same stationary
        # density for both endpoints and their weighted combination.
        for endpoint in pure.values():
            assert endpoint["converged"]
            np.testing.assert_allclose(endpoint["density_real"], native["density_real"],
                                       atol=1e-11, rtol=0)
            np.testing.assert_allclose(endpoint["density_imag"], native["density_imag"],
                                       atol=1e-11, rtol=0)
        np.testing.assert_allclose(native["xc_component_energies"],
                                   .6 * np.asarray(pure["svwn5"]["xc_component_energies"]),
                                   atol=1e-11, rtol=0)
        np.testing.assert_allclose(native["energy_components"]["full_exchange"],
                                   .25 * pure["hf"]["energy_components"]["full_exchange"],
                                   atol=1e-11, rtol=0)
    parity = max(abs(report["energy"] - native["energy"]) for report in reports)
    print(f"magnetic slab public {magnetic_model}/{model}: energy={native['energy']:.15e}, parity={parity:.3e}, "
          f"Coulomb={native['coulomb_operator_defect']:.3e}, "
          f"grid={native['xc_grid_operator_defect']:.3e}/{native['xc_grid_number_defect']:.3e}, "
          f"orbital={native['orbital_residual']:.3e} number={native['particle_number_error']:.3e}")


@pytest.mark.parametrize(
    "reference, xc, message",
    [("rks", "svwn5", "periodic-magnetic-scf with reference=rks"),
     ("uks", "svwn5", "periodic-magnetic-scf with reference=uks"),
     ("roks", "svwn5", "periodic-magnetic-scf with reference=roks"),
     ("ghf", "svwn5", "periodic GHF with a non-HF energy provider")],
)
def test_uniform_flux_reference_fails_by_name(reference, xc, message, tmp_path):
    cell = np.diag([9., 9., 0.])
    pbc = [True, True, False]
    field = [0., 0., 2 * np.pi / 81]
    settings = dict(reference=reference, xc=xc, basis="sto-3g", magnetic_field=field)
    with pytest.raises(NotImplementedError, match=message):
        dft_rs.periodic_magnetic_scf([1], [[0., 0., 0.]], cell, pbc=pbc, **settings)
    atoms = Atoms("H", positions=[[0., 0., 0.]], cell=cell * Bohr, pbc=pbc)
    with pytest.raises(NotImplementedError, match=message):
        dft_rs.DftRs(units="atomic", **settings).get_periodic_magnetic_scf(atoms)
    xyz = tmp_path / "magnetic-slab.xyz"
    xyz.write_text("1\nmagnetic slab\nH 0 0 0\n", encoding="utf-8")
    length = 9 * Bohr
    result = subprocess.run(
        [os.environ["DFT_RS_CLI"], "periodic-magnetic-scf", str(xyz),
         "--cell", f"{length},0,0;0,{length},0;0,0,0", "--pbc", "1,1,0",
         "--reference", reference, "--xc", xc, "--basis", "sto-3g",
         "--magnetic-field", ",".join(map(str, field)), "--format", "json"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0 and message in result.stderr
