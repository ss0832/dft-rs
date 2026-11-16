# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""IGMH scalar fields and units for actual molecular references."""

import os
import subprocess

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
from test_fixed_populations import cli_report


WATER_DIMER = [[.01, -.07, -.1], [.16, .08, 1.70], [-1.63, .53, -.49],
               [.27, .18, 5.08], [1.63, -.42, 6.10], [-1.04, -.82, 5.78]]


@pytest.mark.parametrize("reference,xc", [("rks", "wb97x-d"), ("uks", "wb97x-d"),
                                         ("roks", "wb97x-v"), ("gks", "pbe")])
def test_molecular_igmh_public_routes(tmp_path, reference, xc):
    # Use H2+ for the genuinely spin-polarized ROKS and GKS paths.
    is_dimer = reference in ("rks", "uks")
    positions = np.array(WATER_DIMER if is_dimer else [[.12, -.08, -.64], [.37, .24, .74]])
    atoms = Atoms("OHHOHH" if is_dimer else "H2", positions=Bohr*positions)
    origin = [.31, -.19, 3.13] if is_dimer else [.22, .07, .17]
    grid = [origin, [.19, .04, .01], [.02, .22, .03], [-.03, .01, .28]]
    shape = [2, 1, 2]
    controls = dict(
        xc=xc, reference=reference, basis="sto-3g", grid_level=0, vv10_grid_level=0,
        grid_weight_cutoff=0., cpu_threads=1, charge=0 if is_dimer else 1,
        multiplicity=1 if is_dimer else 2, fragments=[0, 0, 0, 2, 2, 2] if is_dimer else [4, 10],
        igmh_grid=grid, igmh_grid_shape=shape, igmh_atom_details=True,
    )
    native = dft_rs.igmh(atoms.numbers, positions, **controls)
    atomic = dft_rs.DftRs(units="atomic").get_igmh(atoms, **controls)
    physical = dft_rs.DftRs().get_igmh(atoms, **dict(controls, igmh_grid=np.asarray(grid)*Bohr))
    cli = cli_report(tmp_path, "igmh", atoms, controls)
    assert native["reference_type"] == reference
    assert native["igmh_gradient"] == "published"
    assert native["igmh_grid_shape"] == shape
    assert native["igmh_vacuum_points"] == 0
    np.testing.assert_allclose(native["igmh_points"],
                               [np.asarray(origin)+i*np.asarray(grid[1])+k*np.asarray(grid[3])
                                for i in range(2) for k in range(2)], atol=1e-14)
    assert native["igmh_density_residual"] < 1e-13
    assert native["igmh_gradient_residual"] < 1e-13
    powers = {"igmh_points": 1, "igmh_density": -3, "igmh_signed_density": -3,
              "igmh_density_gradient": -4, "igmh_density_hessian": -5,
              "igmh_delta_g": -4, "igmh_delta_g_inter": -4, "igmh_delta_g_intra": -4,
              "igmh_atom_densities": -3, "igmh_atom_vectors": -4}
    defects = []
    for key, power in powers.items():
        for other in (atomic, cli):
            defects.append(np.max(np.abs(np.asarray(other[key]) - native[key])))
            np.testing.assert_allclose(other[key], native[key], rtol=0., atol=1e-10)
            assert other["units"][key] == native["units"][key]
        np.testing.assert_allclose(np.asarray(physical[key])/Bohr**power, native[key],
                                   rtol=0., atol=1e-10)
        assert "angstrom" in physical["units"][key]
    assert abs(native["energy"] - physical["energy"]/Hartree) < 1e-10
    np.testing.assert_allclose([cli["energy"], atomic["energy"]], native["energy"], atol=1e-10, rtol=0.)
    hess = np.asarray(native["igmh_density_hessian"])[:, [[0, 1, 2], [1, 3, 4], [2, 4, 5]]]
    expected_colour = np.sign(np.linalg.eigvalsh(hess)[:, 1])*native["igmh_density"]
    np.testing.assert_allclose(native["igmh_signed_density"], expected_colour, atol=1e-15)
    assert native["igmh_delta_g_inter"][0] > 0.0
    if is_dimer:
        assert native["igmh_signed_density"][0] < 0.0
    else:
        np.testing.assert_array_equal(native["igmh_delta_g_intra"], np.zeros(4))
    print(f"{reference}/{xc}: E={native['energy']:.15e}, "
          f"inter={native['igmh_delta_g_inter'][0]:.15e}, "
          f"signed density={native['igmh_signed_density'][0]:.15e}, "
          f"SCF residual={native['scf_orbital_residual']:.3e}, max route difference={max(defects):.3e}")


def test_igmh_point_conventions_and_refusal(tmp_path):
    atoms = Atoms("OHHOHH", positions=Bohr*np.asarray(WATER_DIMER))
    controls = dict(xc="pbe", basis="sto-3g", grid_level=0, grid_weight_cutoff=0.,
                    cpu_threads=1, fragments=[0, 0, 0, 1, 1, 1], igmh_atom_details=True,
                    igmh_points=[[.31, -.19, 3.13], [900., 900., 900.]])
    published = dft_rs.igmh(atoms.numbers, WATER_DIMER, **controls)
    strict = dft_rs.igmh(atoms.numbers, WATER_DIMER, **dict(controls, igmh_gradient="hirshfeld"))
    assert abs(published["igmh_delta_g_inter"][0]-strict["igmh_delta_g_inter"][0]) > 1e-5
    assert published["igmh_vacuum_points"] == 1
    np.testing.assert_array_equal(published["igmh_atom_densities"], strict["igmh_atom_densities"])
    assert published["igmh_delta_g_inter"][1] == strict["igmh_signed_density"][1] == 0.0
    xyz = tmp_path / "dimer.xyz"
    xyz.write_text("6\nwater dimer\n" + "".join(
        f"{symbol} {x:.17g} {y:.17g} {z:.17g}\n"
        for symbol, (x, y, z) in zip(atoms.symbols, atoms.positions)), encoding="utf-8")
    failed = subprocess.run(
        [os.environ["DFT_RS_CLI"], "igmh", str(xyz), "--igmh-points", "[[0,0,0]]",
         "--fragments", "0,1", "--format", "json"], capture_output=True, text=True)
    assert failed.returncode == 1
    assert "one nonnegative label per atom" in failed.stderr
    assert not failed.stdout


def test_igmh_weighted_functional_uses_the_same_scf_density(tmp_path):
    positions = np.asarray(WATER_DIMER)
    atoms = Atoms("OHHOHH", positions=Bohr * positions)
    options = dict(xc_terms=[{"weight": .75, "xc": "pbe"}, {"weight": .25, "xc": "blyp"}],
                   basis="sto-3g", grid_level=0, grid_weight_cutoff=0., cpu_threads=1)
    energy = dft_rs.run_feature("energy", atoms.numbers, positions, **options)
    controls = dict(options, igmh_points=[[.31, -.19, 3.13]], fragments=[0, 0, 0, 1, 1, 1])
    native = dft_rs.igmh(atoms.numbers, positions, **controls)
    atomic = dft_rs.DftRs(units="atomic").get_igmh(atoms, **controls)
    cli = cli_report(tmp_path, "igmh", atoms, controls)
    assert abs(native["energy"] - energy["energy"]) < 1e-12
    for other in (atomic, cli):
        assert abs(native["energy"] - other["energy"]) < 1e-10
        np.testing.assert_allclose(native["igmh_delta_g_inter"], other["igmh_delta_g_inter"],
                                   rtol=0., atol=1e-11)
    print(f"weighted PBE/BLYP RKS: E={native['energy']:.15e}; "
          f"inter={native['igmh_delta_g_inter'][0]:.15e}; "
          f"SCF residual={native['scf_orbital_residual']:.3e}")

