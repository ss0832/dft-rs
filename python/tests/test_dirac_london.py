# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Four-component DKS in a finite uniform magnetic field."""

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
import numpy as np
import pytest
from ase import Atoms


def lithium_hydride_basis(tmp_path):
    basis = tmp_path / "lih-three-s.g94"
    basis.write_text(
        "Li 0\nS 1 1.0\n5.25 1.0\nS 1 1.0\n0.1562 1.0\n****\n"
        "H 0\nS 1 1.0\n0.61 1.0\n****\n",
        encoding="ascii",
    )
    return basis


@pytest.mark.parametrize("xc", ["wb97x-d", "wb97x-v"])
def test_lithium_hydride_london_rmb_and_public_units(tmp_path, xc):
    numbers = [3, 1]
    positions = np.array([[0.13, -0.22, 0.14], [2.91, 0.48, -0.82]])
    basis = lithium_hydride_basis(tmp_path)
    field = [0.0054, -0.0031, 0.0078]
    options = dict(xc=xc, basis=str(basis), grid_level=0, grid_weight_cutoff=0.,
                   nuclear_rms_radii=[4.62e-5, 1.59e-5], cpu_threads=1,
                   conv_tol=1e-13, conv_tol_grad=1e-10, conv_tol_dp=1e-10)
    magnetic = dict(options, dirac_integrals="london-rmb", magnetic_field=field,
                    magnetic_phase_order=3, gauge_origin=[0.21, -0.16, 0.89])
    native = dft_rs.dirac_dks(numbers, positions, **magnetic)
    refined = dft_rs.dirac_dks(numbers, positions, **dict(magnetic, magnetic_phase_order=4))
    baseline = dft_rs.dirac_dks(numbers, positions, **options)
    london_zero = dft_rs.dirac_dks(numbers, positions, dirac_integrals="london-rmb", **options)
    reversed_field = dft_rs.dirac_dks(
        numbers, positions, **dict(magnetic, magnetic_field=[-x for x in field]),
    )
    atoms = Atoms(numbers, positions=positions * Bohr)
    atomic = dft_rs.DftRs(units="atomic").get_dirac_dks(
        atoms, **dict(magnetic, gauge_origin=[-1.81, 0.69, -0.32]),
    )
    physical_options = dict(magnetic, **{
        key: np.asarray(magnetic[key])*Bohr for key in ("nuclear_rms_radii", "gauge_origin")
    })
    physical_options.update(conv_tol=options["conv_tol"]*Hartree,
                            conv_tol_grad=options["conv_tol_grad"]*Hartree)
    physical = dft_rs.DftRs().get_dirac_dks(atoms, **physical_options)
    xyz = tmp_path / "lih.xyz"
    xyz.write_text(
        "2\nLithium hydride\n" + "\n".join(
            f"{n} {x:.17g} {y:.17g} {z:.17g}"
            for n, (x, y, z) in zip(numbers, atoms.positions, strict=True)
        ) + "\n",
        encoding="ascii",
    )
    command = [os.environ["DFT_RS_CLI"], "dirac-dks", str(xyz), "--format", "json"]
    for key, value in magnetic.items():
        command.extend(["--" + key.replace("_", "-"),
                        value if isinstance(value, str) else json.dumps(value)])
    executed = subprocess.run(command, text=True, capture_output=True, check=True)
    assert "[spinor-scf]" in executed.stderr and "residual" in executed.stderr
    cli = json.loads(executed.stdout)
    for result in (native, refined, baseline, london_zero, reversed_field, atomic, physical, cli):
        assert result["converged"], result
        assert result["coulomb_hermiticity_defect"] < 2e-11
        assert abs(result["particle_number_error"]) < 2e-10
    for result in (atomic, cli, reversed_field):
        assert result["dirac_integrals"] == "london-rmb"
        np.testing.assert_allclose(result["energy"], native["energy"], rtol=0., atol=3e-10)
        np.testing.assert_allclose(result["dipole"], native["dipole"], rtol=0., atol=3e-10)
        for key, value in native["energy_components"].items():
            np.testing.assert_allclose(result["energy_components"][key], value, rtol=0., atol=3e-10)
    np.testing.assert_allclose(london_zero["energy"], baseline["energy"], rtol=0., atol=2e-10)
    np.testing.assert_allclose(refined["energy"], native["energy"], rtol=0., atol=2e-8)
    np.testing.assert_allclose(physical["energy"], native["energy"]*Hartree, rtol=0., atol=2e-9)
    np.testing.assert_allclose(physical["dipole"], np.asarray(native["dipole"])*Bohr,
                               rtol=0., atol=3e-10)
    assert abs(native["energy"] - baseline["energy"]) > 1e-7
    print(f"LiH {xc} London E {native['energy']:.16e}; zero-field energy "
          f"{baseline['energy']:.16e}; dipole {native['dipole']}; phase-order energy "
          f"change {refined['energy']-native['energy']:.6e}; orbital residual "
          f"{native['scf_orbital_residual']:.6e}; pair Hermiticity defect "
          f"{native['coulomb_hermiticity_defect']:.6e}")


@pytest.mark.parametrize("xc", ["pbe", "hf"])
def test_lithium_hydride_dirac_indirect_nuclear_coupling_public_routes(tmp_path, xc):
    numbers = [3, 1]
    positions = np.array([[0.13, -0.22, 0.14], [2.91, 0.48, -0.82]])
    controls = dict(relativistic_method="dirac", xc=xc, basis=str(lithium_hydride_basis(tmp_path)),
                    grid_level=0, grid_weight_cutoff=0., cpu_threads=1,
                    nuclear_rms_radii=[4.62e-5, 1.59e-5],
                    nuclear_magnetic_rms_radii=[4.62e-5, 1.59e-5], nuclear_g=[2.0, 5.5],
                    response_tolerance=1e-10, conv_tol=1e-13, conv_tol_grad=1e-10)
    atoms = Atoms(numbers, positions=positions*Bohr)
    converted = dict(controls, **{
        key: np.asarray(controls[key])*Bohr
        for key in ("nuclear_rms_radii", "nuclear_magnetic_rms_radii")
    })
    converted.update(conv_tol=controls["conv_tol"]*Hartree,
                     conv_tol_grad=controls["conv_tol_grad"]*Hartree)
    xyz = tmp_path / "lih.xyz"
    xyz.write_text(
        "2\nLithium hydride\n" + "\n".join(
            f"{n} {x:.17g} {y:.17g} {z:.17g}"
            for n, (x, y, z) in zip(numbers, atoms.positions, strict=True)
        ) + "\n", encoding="ascii",
    )
    command = [os.environ["DFT_RS_CLI"], "spin-spin-couplings", str(xyz), "--format", "json"]
    for key, value in controls.items():
        command.extend(["--" + key.replace("_", "-"),
                        value if isinstance(value, str) else json.dumps(value)])
    # The four references are independent. Each calculation still uses
    # one numerical-library thread and the same fixed integration controls.
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [
            pool.submit(dft_rs.spin_spin_couplings, numbers, positions, **controls),
            pool.submit(dft_rs.DftRs(units="atomic").get_spin_spin_couplings, atoms, **controls),
            pool.submit(dft_rs.DftRs().get_spin_spin_couplings, atoms, **converted),
            pool.submit(subprocess.run, command, text=True, capture_output=True, check=True),
        ]
        native, atomic, physical, executed = [job.result() for job in jobs]
    assert "[spinor-scf]" in executed.stderr and "residual" in executed.stderr
    cli = json.loads(executed.stdout)
    assert native["coupling_pairs"] == [[0, 1]]
    assert np.asarray(native["reduced_couplings"]).shape == (1, 3, 3)
    for result in (native, atomic, physical, cli):
        assert result["converged"] and result["provenance"] == "analytic"
        assert result["response_residual"] < 2e-10
        assert result["coupling_source_reciprocity_defect"] < 2e-10
        for key in ("reduced_couplings", "coupling_explicit_sector_tensors",
                    "coupling_orbital_relaxation_tensors", "coupling_constants_hz"):
            np.testing.assert_allclose(result[key], native[key], rtol=0., atol=2e-9)
    np.testing.assert_allclose(np.asarray(native["coupling_explicit_sector_tensors"])
                               + np.asarray(native["coupling_orbital_relaxation_tensors"]),
                               native["reduced_couplings"], rtol=0., atol=1e-13)
    np.testing.assert_allclose(physical["nuclear_magnetic_rms_radii"],
                               converted["nuclear_magnetic_rms_radii"], rtol=0., atol=1e-13)
    print(f"LiH 4c {xc} coupling {native['reduced_couplings']}, "
          f"J {native['coupling_constants_hz']} Hz for supplied g, "
          f"coupled residual {native['response_residual']:.6e}")

@pytest.mark.parametrize("feature,report_key", [
    ("magnetizability", "magnetizability"),
    ("nmr_shielding", "shielding_tensors"),
])
@pytest.mark.parametrize("interaction", ["coulomb", "gaunt", "breit"])
def test_lithium_hydride_four_component_magnetic_derivatives(tmp_path, feature, report_key, interaction):
    numbers = [3, 1]
    positions = np.array([[0.13, -0.22, 0.14], [2.91, 0.48, -0.82]])
    controls = dict(relativistic_method="dirac", dirac_interaction=interaction, xc="hf", basis=str(lithium_hydride_basis(tmp_path)),
                    grid_level=0, grid_weight_cutoff=0., cpu_threads=1,
                    nuclear_rms_radii=[4.62e-5, 1.59e-5],
                    magnetic_phase_order=2, gauge_origin=[-0.21, 0.78, -0.43],
                    response_tolerance=1e-10, conv_tol=1e-13, conv_tol_grad=1e-10)
    if feature == "nmr_shielding":
        controls.update(nuclei=[1], nuclear_magnetic_rms_radii=[4.62e-5, 1.59e-5])
    atoms = Atoms(numbers, positions=positions*Bohr)
    xyz = tmp_path / "lih.xyz"
    xyz.write_text(
        "2\nLithium hydride\n" + "\n".join(
            f"{n} {x:.17g} {y:.17g} {z:.17g}"
            for n, (x, y, z) in zip(numbers, atoms.positions, strict=True)
        ) + "\n", encoding="ascii",
    )
    command = [os.environ["DFT_RS_CLI"], feature.replace("_", "-"), str(xyz), "--format", "json"]
    for key, value in controls.items():
        command.extend(["--" + key.replace("_", "-"),
                        value if isinstance(value, str) else json.dumps(value)])
    translated_gauge = dict(controls, gauge_origin=[1.11, -1.32, 0.94])
    native_function = getattr(dft_rs, feature)
    ase_function = getattr(dft_rs.DftRs(units="atomic"), "get_" + feature)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(native_function, numbers, positions, **controls),
                pool.submit(ase_function, atoms, **translated_gauge),
                pool.submit(subprocess.run, command, text=True, capture_output=True, check=True)]
        native, atomic, executed = [job.result() for job in jobs]
    assert "[spinor-scf]" in executed.stderr and "residual" in executed.stderr
    cli = json.loads(executed.stdout)
    expected = np.asarray(native[report_key])
    assert expected.shape == ((1, 3, 3) if feature == "nmr_shielding" else (3, 3))
    for result in (native, atomic, cli):
        assert result["converged"] and result["provenance"] == "analytic"
        assert result["magnetic_response_reciprocity_defect"] < 2e-10
        assert result["response_residual"] < 2e-10
        assert result["units"][report_key] == ("ppm" if feature == "nmr_shielding" else "e^2*bohr^2/m_e")
        np.testing.assert_allclose(result[report_key], expected, rtol=0., atol=5e-9)
        np.testing.assert_allclose(result["magnetic_dipole"], [0.0]*3, rtol=0., atol=1e-11)
    print(f"LiH 4c {interaction} {feature} {expected.tolist()}, largest CPKS residual "
          f"{native['response_residual']:.6e}, sector residual "
          f"{native['magnetic_response_sector_operator_residual']:.6e} Hartree")
