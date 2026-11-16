# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832

import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM, HARTREE_IN_EV


def cli_report(feature, atoms, controls, tmp_path):
    geometry = tmp_path / "geometry.xyz"
    rows = [
        f"{symbol} " + " ".join(f"{coordinate:.17g}" for coordinate in row)
        for symbol, row in zip(atoms.get_chemical_symbols(), atoms.positions)
    ]
    geometry.write_text(f"{len(atoms)}\nMolecular geometry\n" + "\n".join(rows) + "\n", encoding="utf-8")
    argv = [os.environ["DFT_RS_CLI"], feature, str(geometry), "--format", "json"]
    for key, value in controls.items():
        argv.extend(["--" + key.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)])
    process = subprocess.run(argv, check=True, capture_output=True, text=True)
    assert "iter" in process.stderr and "residual" in process.stderr
    return json.loads(process.stdout)


@pytest.mark.parametrize(
    ("xc", "reference", "charge", "multiplicity"),
    [
        ("wb97x-d", "rks", 0, 1),
        ("wb97x-v", "uks", 1, 2),
        ("wb97x-v", "roks", 1, 2),
        ("wb97x-d", "gks", 0, 1),
        ("wb97x-v", "gks", 1, 2),
    ],
)
def test_hubbard_public_api_parity(tmp_path, xc, reference, charge, multiplicity):
    numbers = [7, 1, 1, 1]
    positions = [[0.13, -0.27, 0.19], [1.78, 0.34, -0.08], [-0.49, 1.65, 0.41], [0.31, -0.78, 1.91]]
    atoms = Atoms(numbers=numbers, positions=np.asarray(positions) * BOHR_IN_ANGSTROM)
    coulomb = [[0.34, 0.044, 0.21], [0.044, 0.052, 0.037], [0.21, 0.037, 0.29]]
    pairs = [0, 1, 1, 2]
    controls = {
        "xc": xc, "reference": reference, "charge": charge, "multiplicity": multiplicity,
        "basis": "sto-3g", "grid_level": 0, "vv10_grid_level": 0, "grid_weight_cutoff": 0.0,
        "hubbard_sites": [
            {"orbitals": [5], "u_eff": 0.143},
            {
                "orbitals": [6, 7], "model": "liechtenstein", "u": 0.2625, "j": 0.052,
                "integrals": [coulomb[ab][cd] for ab in pairs for cd in pairs],
            },
        ],
    }
    native = dft_rs.dft_plus_u(numbers, positions, **controls)
    calc = dft_rs.DftRs()
    atomic = calc.get_dft_plus_u(atoms, units="atomic", **controls)
    physical = calc.get_dft_plus_u(atoms, **controls)

    cli = cli_report("dft-plus-u", atoms, controls, tmp_path)
    keys = ("energy", "hubbard_energy", "scf_orbital_residual")
    for report in (atomic, cli, physical):
        assert report["reference_type"] == reference
        assert report["converged"]
    for key in keys:
        assert native["units"][key] == "hartree"
        for report in (atomic, cli):
            assert report[key] == pytest.approx(native[key], rel=0.0, abs=1e-10)
        assert physical[key] == pytest.approx(native[key] * HARTREE_IN_EV, rel=0.0, abs=2e-9)
    occupations = (
        ("occupations_real", "occupations_imaginary") if reference == "gks"
        else ("occupations_alpha", "occupations_beta")
    )
    for a, b, c, d in zip(native["hubbard_sites"], atomic["hubbard_sites"], cli["hubbard_sites"],
                          physical["hubbard_sites"]):
        for key in occupations:
            assert native["units"][f"hubbard_sites.{key}"] == "electrons"
            for report in (b, c, d):
                np.testing.assert_allclose(report[key], a[key], atol=1e-10, rtol=0.0)
        for key in ("interaction", "double_counting"):
            assert d[key] == pytest.approx(a[key] * HARTREE_IN_EV, rel=0.0, abs=2e-9)
    print(f"{xc} {reference} mixed sites {[(key, native[key]) for key in keys]}")


@pytest.mark.parametrize("xc", ["wb97x-d", "wb97x-v"])
def test_pz_sic_public_api_parity(tmp_path, xc):
    positions = [[0.13, -0.27, 0.19], [1.77, 0.36, -0.08]]
    atoms = Atoms(numbers=[1, 1], positions=np.asarray(positions) * BOHR_IN_ANGSTROM)
    controls = {
        "xc": xc, "reference": "uks", "charge": 1, "multiplicity": 2,
        "basis": "sto-3g", "dispersion": "auto" if xc == "wb97x-v" else "none", "grid_level": 0,
        "vv10_grid_level": 0, "grid_weight_cutoff": 0.0, "conv_tol_grad": 1e-9,
    }
    native = dft_rs.pz_sic([1, 1], positions, **controls)
    calc = dft_rs.DftRs()
    atomic = calc.get_pz_sic(atoms, units="atomic", **controls)
    physical = calc.get_pz_sic(atoms, **controls)
    cli = cli_report("pz-sic", atoms, controls, tmp_path)
    assert native["converged"] and atomic["converged"] and cli["converged"] and physical["converged"]
    for key in ("energy", "self_interaction_correction", "scf_orbital_residual"):
        assert native["units"][key] == "hartree"
        assert atomic[key] == pytest.approx(native[key], abs=1e-10, rel=0.0)
        assert cli[key] == pytest.approx(native[key], abs=1e-10, rel=0.0)
        assert physical[key] == pytest.approx(native[key] * HARTREE_IN_EV, abs=2e-9, rel=0.0)


@pytest.mark.parametrize("xc", ["wb97x-d", "wb97x-v"])
def test_fixed_flo_sic_public_api_parity(tmp_path, xc):
    positions = [[0.13, -0.21, 0.19], [1.72, 0.37, -0.16]]
    atoms = Atoms(numbers=[1, 1], positions=np.asarray(positions) * BOHR_IN_ANGSTROM)
    controls = dict(xc=xc, reference="uks", charge=1, multiplicity=2, basis="sto-3g",
                    grid_level=0, vv10_grid_level=0, grid_weight_cutoff=0.0, cpu_threads=1,
                    fod_alpha=[[0.73, 0.04, 0.13]], fod_beta=[])
    native = dft_rs.flo_sic_fixed([1, 1], positions, **controls)
    atomic = dft_rs.DftRs(units="atomic").get_flo_sic_fixed(atoms, **controls)
    cli = cli_report("flo-sic-fixed", atoms, controls, tmp_path)
    for report in (native, atomic, cli):
        assert report["reference_type"] == "uks" and report["converged"]
        assert report["units"]["fermi_descriptor_densities"] == "electrons/bohr^3"
        assert len(report["fermi_descriptor_densities"][0]) == 1
        assert report["fermi_descriptor_densities"][1] == []
        assert report["fermi_descriptor_densities"][0][0] > 0.0
        assert report["localized_density_residual"] < 1e-12
        assert max(report["localized_orthogonality_residuals"]) < 1e-12
        assert report["parent_orbital_residual"] < 1e-7
        assert report["energy"] - report["parent_energy"] == pytest.approx(
            report["self_interaction_correction"], rel=0.0, abs=1e-12)
        assert report["self_interaction_correction"] == pytest.approx(
            -report["orbital_self_energies"][0][0]["total"], rel=0.0, abs=1e-12)
        for key in ("energy", "parent_energy", "self_interaction_correction", "parent_orbital_residual"):
            assert report["units"][key] == "hartree"
            assert report[key] == pytest.approx(native[key], rel=0.0, abs=1e-10)
    print(f"{xc} frozen FLO-SIC H2+ total={native['energy']:.17e} "
          f"correction={native['self_interaction_correction']:.17e} "
          f"parent residual={native['parent_orbital_residual']:.17e}")


def test_complex_flo_sic_nitrogen_real_limit_and_public_routes(tmp_path):
    positions = [[-1.036, 0, 0], [1.036, 0, 0]]
    real = [[-1.036, 0, 0], [1.036, 0, 0], [-2.18, 0, 0], [2.18, 0, 0],
            [0.02, 0.45, 0.03], [-0.04, -0.22, 0.40], [0.04, -0.21, -0.39]]
    imaginary = [[0, 0, 0]] * 4 + [[0.03, -0.12, 0.13], [0.06, 0.17, -0.05],
                                  [-0.04, -0.10, -0.16]]
    controls = dict(xc="pbe", reference="uks", basis="sto-3g", dispersion="none",
                    grid_level=1, grid_weight_cutoff=0.0, cpu_threads=1,
                    fod_alpha=real, fod_beta=real, fod_gradient=True)
    original = dft_rs.flo_sic_fixed([7, 7], positions, **controls)
    real_limit = dft_rs.flo_sic_fixed([7, 7], positions,
                                     **(controls | {"fod_alpha_imaginary": [[0, 0, 0]] * 7}))
    assert real_limit["energy"] == pytest.approx(original["energy"], abs=1e-10)
    np.testing.assert_allclose(np.asarray(real_limit["fod_gradient_alpha"])[:, :3],
                               original["fod_gradient_alpha"], atol=1e-10, rtol=0)
    assert np.max(np.abs(np.asarray(real_limit["fod_gradient_alpha"])[:, 3:])) < 1e-12
    complex_controls = controls | {"fod_alpha_imaginary": imaginary, "fod_beta_imaginary": imaginary}
    native = dft_rs.flo_sic_fixed([7, 7], positions, **complex_controls)
    atoms = Atoms(numbers=[7, 7], positions=np.asarray(positions) * BOHR_IN_ANGSTROM)
    cli = cli_report("flo-sic-fixed", atoms, complex_controls, tmp_path)
    ase = dft_rs.DftRs(units="atomic").get_flo_sic_fixed(atoms, **complex_controls)
    physical = dft_rs.DftRs().get_flo_sic_fixed(atoms, **(complex_controls | {
        key: (np.asarray(complex_controls[key]) * BOHR_IN_ANGSTROM).tolist()
        for key in ("fod_alpha", "fod_beta", "fod_alpha_imaginary", "fod_beta_imaginary")
    }))
    for report in (native, cli, ase, physical):
        assert report["converged"]
        assert report["localized_density_residual"] < 1e-11
        assert max(report["localized_orthogonality_residuals"]) < 1e-11
    for report in (cli, ase):
        assert report["energy"] == pytest.approx(native["energy"], rel=0, abs=1e-10)
        np.testing.assert_allclose(report["fod_gradient_alpha"], native["fod_gradient_alpha"],
                                   atol=1e-10, rtol=0)
    np.testing.assert_allclose(physical["fod_gradient_alpha"],
                               np.asarray(native["fod_gradient_alpha"]) * HARTREE_IN_EV / BOHR_IN_ANGSTROM,
                               atol=1e-9, rtol=0)
    assert np.max(np.abs(native["flo_coefficients_imaginary"])) > 1e-4
    print(f"N2 real FLO-SIC E={original['energy']:.14e} complex E={native['energy']:.14e} "
          f"complex correction={native['self_interaction_correction']:.14e} "
          f"density residual={native['localized_density_residual']:.4e}")
