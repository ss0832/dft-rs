# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Four-component sector projection: dense oracle and common surfaces."""
import json
import math
import os
import subprocess

import numpy as np
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
import dft_rs


def inputs():
    momentum=np.array([0.4, -0.3, 0.2])
    c=5.0
    sigma=np.array([[momentum[2],momentum[0]-1j*momentum[1]],
                    [momentum[0]+1j*momentum[1],-momentum[2]]])
    h=np.zeros((4,4),dtype=complex)
    h[:2,2:]=c*sigma;h[2:,:2]=c*sigma
    h[2:,2:]=-2*c*c*np.eye(2)
    scaling=np.diag([1.1,0.9,1.2,0.95])
    h=scaling@h@scaling
    s=scaling@scaling
    dh=np.diag([0.01,-0.02,0.03,-0.01])
    dh[0,3]=dh[3,0]=0.04
    ds=np.diag([0.003,-0.002,0.001,-0.004])
    return dict(dirac_hamiltonian_real=h.real.tolist(),
                dirac_hamiltonian_imaginary=h.imag.tolist(),
                dirac_overlap_real=s.tolist(),
                dirac_hamiltonian_direction_real=dh.tolist(),
                dirac_overlap_direction_real=ds.tolist(),
                spectrum_separator=-25.0,response_order=4)


def test_generated_london_balance_ammonia_public_routes(tmp_path):
    numbers = [7, 1, 1, 1]
    positions = np.array([[0.12, -0.21, 0.15], [1.74, 0.18, 0.78],
                          [-0.76, 1.27, -0.57], [-0.61, -1.52, -0.73]])
    controls = dict(dirac_integrals="london-rmb", basis="sto-3g", cpu_threads=1,
                    nuclear_rms_radii=[4.84e-5, 1.59e-5, 1.59e-5, 1.59e-5],
                    magnetic_direction=[0.11, -0.27, 0.32], gauge_origin=[0.14, -0.11, 0.82],
                    response_order=1)
    direct = dft_rs.no_pair_projector(numbers, positions, **controls)
    other_origin = dft_rs.no_pair_projector(
        numbers, positions, **dict(controls, gauge_origin=[-2.18, 0.72, -1.04]),
    )
    atoms = Atoms(numbers, positions=positions*Bohr)
    atomic = dft_rs.DftRs.get_no_pair_projector(atoms, units="atomic", **controls)
    physical_controls = dict(controls, **{
        k: np.asarray(controls[k])*Bohr for k in ("nuclear_rms_radii", "gauge_origin")
    })
    physical = dft_rs.DftRs.get_no_pair_projector(atoms, **physical_controls)
    xyz = tmp_path / "ammonia.xyz"
    xyz.write_text(
        "4\nC1 ammonia\n" + "\n".join(
            f"{number} {x:.17g} {y:.17g} {z:.17g}"
            for number, (x, y, z) in zip(numbers, atoms.positions, strict=True)
        ) + "\n",
        encoding="utf-8",
    )
    command = [os.environ["DFT_RS_CLI"], "no-pair-projector", str(xyz), "--format", "json"]
    for key, value in controls.items():
        command.extend(["--" + key.replace("_", "-"),
                        value if isinstance(value, str) else json.dumps(value)])
    proc = subprocess.run(command, check=True, text=True, capture_output=True)
    cli = json.loads(proc.stdout)
    keys = ("dirac_hamiltonian_coefficients", "dirac_overlap_coefficients",
            "projector_coefficients", "projected_hamiltonian_coefficients")
    atomic_defects = {}
    for index, report in enumerate((atomic, cli, other_origin)):
        assert report["converged"]
        assert report["magnetic_phase_order"] == 1
        assert report["units"] == direct["units"]
        for prefix in keys:
            for component in ("real", "imaginary"):
                key = f"{prefix}_{component}"
                np.testing.assert_allclose(report[key], direct[key], rtol=0., atol=2e-10)
                error = np.max(np.abs(np.asarray(report[key])-np.asarray(direct[key])))
                atomic_defects[index] = max(atomic_defects.get(index, 0.), error)
                scale = Hartree if "hamiltonian" in key else 1.
                np.testing.assert_allclose(physical[key], np.asarray(direct[key])*scale,
                                           rtol=0., atol=2e-9)
    assert direct["electronic_dimension"] == 16
    assert np.max(np.abs(direct["large_potential_coefficients_imaginary"][1])) > 1e-7
    assert max(direct["relative_commutator_defects"]) < 1e-10
    assert max(direct["overlap_hermiticity_defects"]) < 1e-12
    np.testing.assert_allclose(physical["spectral_gap"], direct["spectral_gap"]*Hartree,
                               rtol=0., atol=1e-8)
    print(f"C1 NH3 London atomic route matrix discrepancies {atomic_defects}; "
          f"sector gap {direct['spectral_gap']:.16e}; "
          f"metric {direct['metric_defects']}; commutator {direct['relative_commutator_defects']}")


def test_dirac_coulomb_scf_public_surfaces(tmp_path):
    check_dirac_coulomb_scf_public_surfaces(tmp_path)


def check_dirac_coulomb_scf_public_surfaces(tmp_path, interaction=None):
    numbers = [2, 1]
    positions = np.array([[0.12, -0.27, 0.09], [1.41, 0.31, -0.46]])
    atoms = Atoms(numbers, positions=positions * Bohr)
    controls = dict(
        basis="sto-3g", charge=1, speed_of_light=7.4,
        nuclear_rms_radii=[0.12, 0.081], cpu_threads=1,
        conv_tol=2e-13, conv_tol_grad=1e-11, conv_tol_dp=1e-11,
    )
    if interaction is not None:
        controls["dirac_interaction"] = interaction
    direct = dft_rs.dirac_coulomb_scf(numbers, positions, **controls)
    atomic = dft_rs.DftRs().get_dirac_coulomb_scf(atoms, units="atomic", **controls)
    physical = dict(controls, nuclear_rms_radii=np.asarray(controls["nuclear_rms_radii"]) * Bohr)
    adapted = dft_rs.DftRs().get_dirac_coulomb_scf(atoms, **physical)
    xyz = tmp_path / "helium_hydride.xyz"
    xyz.write_text(
        "2\nSynthetic finite-nucleus Dirac–Coulomb test\n"
        + "\n".join(f"{z} {x:.17g} {y:.17g} {zpos:.17g}"
                    for z, (x, y, zpos) in zip(numbers, atoms.positions, strict=True)) + "\n",
        encoding="utf-8",
    )
    command = [os.environ["DFT_RS_CLI"], "dirac-coulomb-scf", str(xyz), "--format", "json"]
    for name, value in controls.items():
        command.extend(["--" + name.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)])
    process = subprocess.run(command, check=True, encoding="utf-8", capture_output=True)
    cli = json.loads(process.stdout)
    assert any("[spinor-scf] iter" in line and "residual" in line for line in process.stderr.splitlines())
    rejected = subprocess.run(
        [*command, "--magnetic-field", "[0.01, 0, 0]"], encoding="utf-8", capture_output=True,
    )
    assert rejected.returncode != 0 and "magnetic_field" in rejected.stderr
    assert direct["converged"] and direct["electronic_dimension"] == 4
    for report in (atomic, cli):
        assert report["converged"] and report["units"] == direct["units"]
        keys = ["energy", "orbital_energies", "occupations", "scf_orbital_residual", "spectral_gap"]
        if interaction is None:
            keys.append("coulomb_hermiticity_defect")
        else:
            assert report["dirac_interaction"] == interaction
            keys.extend(("current_hermiticity_defect", "two_electron_hermiticity_defect"))
        for key in keys:
            np.testing.assert_allclose(report[key], direct[key], rtol=0, atol=1e-11)
        for key, value in direct["energy_components"].items():
            np.testing.assert_allclose(report["energy_components"][key], value, rtol=0, atol=1e-11)
    np.testing.assert_allclose(adapted["energy"], direct["energy"] * Hartree, rtol=0, atol=1e-10)
    np.testing.assert_allclose(adapted["nuclear_rms_radii"], physical["nuclear_rms_radii"], rtol=0, atol=1e-12)
    distance = np.linalg.norm(positions[0] - positions[1])
    omega = math.sqrt(1.5 / sum(radius**2 for radius in controls["nuclear_rms_radii"]))
    np.testing.assert_allclose(direct["energy_components"]["classical_nuclear_repulsion"],
                               2.0 * math.erf(omega * distance) / distance, rtol=0, atol=1e-13)
    if interaction is not None:
        terms = direct["energy_components"]
        np.testing.assert_allclose(sum(terms.values()), direct["energy"], rtol=0, atol=1e-12)
        assert abs(terms["current_exchange"]) > 1e-8
        assert abs(terms["current_direct"]) < 1e-10
        assert direct["current_hermiticity_defect"] < 1e-11
        assert direct["two_electron_hermiticity_defect"] < 1e-11
        np.testing.assert_allclose(adapted["energy_components"]["current_exchange"],
                                   terms["current_exchange"] * Hartree, rtol=0, atol=1e-10)
    print("Installed Dirac Coulomb SCF", json.dumps(direct, allow_nan=False, sort_keys=True))


def test_no_pair_projection_dense_oracle_and_all_public_surfaces():
    controls=inputs()
    direct=dft_rs.no_pair_projector(**controls)
    generic=dft_rs.run_feature("no-pair-projector",**controls)
    atomic=dft_rs.DftRs.get_no_pair_projector(units="atomic",**controls)
    physical=dict(controls)
    for name in ("dirac_hamiltonian_real","dirac_hamiltonian_imaginary",
                 "dirac_hamiltonian_direction_real","spectrum_separator"):
        physical[name]=np.asarray(controls[name])*Hartree
    adapted=dft_rs.DftRs.get_no_pair_projector(**physical)
    argv=[os.environ["DFT_RS_CLI"],"no-pair-projector","--format","json"]
    for name,value in controls.items():argv +=["--"+name.replace("_","-"),json.dumps(value)]
    process=subprocess.run(argv,text=True,capture_output=True)
    assert process.returncode==0,process.stderr
    cli=json.loads(process.stdout)
    assert direct["converged"] and direct["electronic_dimension"]==2
    s=np.asarray(controls["dirac_overlap_real"])
    h=np.asarray(controls["dirac_hamiltonian_real"])+1j*np.asarray(controls["dirac_hamiltonian_imaginary"])
    eigenvalues,u=np.linalg.eigh(s)
    x=(u/eigenvalues**0.5)@u.T
    eigenvalues,u=np.linalg.eigh(x@h@x)
    c=x@u[:,eigenvalues>controls["spectrum_separator"]]
    expected=c@c.conj().T
    actual=np.asarray(direct["projector_coefficients_real"])[0]+1j*np.asarray(direct["projector_coefficients_imaginary"])[0]
    np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-12)
    for report in (generic,atomic,cli):
        assert report["units"]==direct["units"] and report["model"]==direct["model"]
        for name in ("projector_coefficients_real","projector_coefficients_imaginary",
                     "projected_hamiltonian_coefficients_real","projected_hamiltonian_coefficients_imaginary"):
            np.testing.assert_allclose(report[name],direct[name],rtol=1e-10,atol=1e-11)
    for name in ("projector_coefficients_real","projector_coefficients_imaginary"):
        np.testing.assert_allclose(adapted[name],direct[name],rtol=1e-10,atol=1e-11)
    for name in ("projected_hamiltonian_coefficients_real","projected_hamiltonian_coefficients_imaginary","spectral_gap"):
        np.testing.assert_allclose(adapted[name],np.asarray(direct[name])*Hartree,rtol=1e-10,atol=1e-11)
    assert not any(unit=="unspecified" for unit in direct["units"].values())
    print("no-pair raw",{key:direct[key] for key in ("projector_defects","metric_defects","commutator_defects",
          "relative_commutator_defects","number_defects","complex_embedding_defects")})


def test_generated_finite_nucleus_balanced_operator_and_all_public_surfaces(tmp_path):
    numbers=[8,1,1]
    positions=np.array([[0.0,0.0,0.12],[0.0,1.43,-0.93],[0.0,-1.43,-0.93]])
    controls=dict(dirac_integrals="rkb",basis="sto-3g",speed_of_light=20.0,
                  nuclear_rms_radii=[0.004,0.003,0.003],
                  nuclear_radius_direction=[0.0002,0.0001,-0.0001],
                  nuclear_displacements=[[0.01,-0.02,0.03],[0.02,0.01,0.0],[-0.01,0.02,-0.03]],
                  speed_of_light_direction=0.3,response_order=4)
    direct=dft_rs.no_pair_projector(numbers,positions,**controls)
    generic=dft_rs.run_feature("no-pair-projector",numbers,positions,**controls)
    atoms=Atoms(numbers,positions=positions*Bohr)
    atomic=dft_rs.DftRs.get_no_pair_projector(atoms,units="atomic",**controls)
    physical=dict(controls)
    for name in ("nuclear_rms_radii","nuclear_radius_direction","nuclear_displacements"):
        physical[name]=np.asarray(controls[name])*Bohr
    adapted=dft_rs.DftRs.get_no_pair_projector(atoms,**physical)
    xyz=tmp_path/"water.xyz"
    geometry=positions/dft_rs.native.BOHR_PER_ANGSTROM
    xyz.write_text("3\nSynthetic finite-nucleus acceptance geometry\n"+"\n".join(
        f"{z} {x:.16g} {y:.16g} {zz:.16g}" for z,(x,y,zz) in zip(numbers,geometry))+"\n")
    argv=[os.environ["DFT_RS_CLI"],"no-pair-projector",str(xyz),"--format","json"]
    for name,value in controls.items():
        argv +=["--"+name.replace("_","-"),json.dumps(value) if not isinstance(value,str) else value]
    process=subprocess.run(argv,text=True,capture_output=True)
    assert process.returncode==0,process.stderr
    cli=json.loads(process.stdout)
    assert direct["converged"] and direct["electronic_dimension"]==14
    h=np.asarray(direct["dirac_hamiltonian_coefficients_real"])[0]+1j*np.asarray(direct["dirac_hamiltonian_coefficients_imaginary"])[0]
    s=np.asarray(direct["dirac_overlap_coefficients_real"])[0]
    eigenvalues,u=np.linalg.eigh(s)
    x=(u/eigenvalues**0.5)@u.T
    eigenvalues,u=np.linalg.eigh(x@h@x)
    c=x@u[:,eigenvalues>direct["spectrum_separator"]]
    expected=c@c.conj().T
    actual=np.asarray(direct["projector_coefficients_real"])[0]+1j*np.asarray(direct["projector_coefficients_imaginary"])[0]
    np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-10)
    assert np.max(np.abs(h.imag))>1e-6
    assert np.max(np.abs(direct["dirac_hamiltonian_coefficients_real"][4]))>1e-12
    nuclear=0.0
    for i in range(3):
        for j in range(i):
            distance=np.linalg.norm(positions[i]-positions[j])
            omega=(1.5/(controls["nuclear_rms_radii"][i]**2+controls["nuclear_rms_radii"][j]**2))**0.5
            nuclear+=numbers[i]*numbers[j]*math.erf(omega*distance)/distance
    np.testing.assert_allclose(direct["nuclear_repulsion_coefficients"][0],nuclear,rtol=1e-13)
    matrix_keys=("dirac_hamiltonian_coefficients_real","dirac_hamiltonian_coefficients_imaginary",
                 "dirac_overlap_coefficients_real","projector_coefficients_real","projector_coefficients_imaginary",
                 "projected_hamiltonian_coefficients_real","projected_hamiltonian_coefficients_imaginary",
                 "small_potential_coefficients_real","small_potential_coefficients_imaginary")
    for report in (generic,atomic,cli):
        assert report["model"]==direct["model"] and report["converged"]
        assert not any(unit=="unspecified" for unit in report["units"].values())
        for name in matrix_keys:
            np.testing.assert_allclose(report[name],direct[name],rtol=1e-9,atol=1e-10)
    for name in matrix_keys:
        scale=1.0 if name.startswith(("dirac_overlap","projector_")) else Hartree
        np.testing.assert_allclose(adapted[name],np.asarray(direct[name])*scale,rtol=1e-9,atol=1e-10)
    print("generated Dirac raw",{key:direct[key] for key in ("projector_defects","metric_defects",
          "relative_commutator_defects","number_defects","hamiltonian_hermiticity_defects","overlap_hermiticity_defects")})
