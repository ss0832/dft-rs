# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Connected long-wave, joint-field, magnetic-balance and pairing reports."""
import json
import os
import subprocess
import re

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
import dft_rs


def test_default_text_magnetic_hessian_uses_analytic_driver_and_scf_stderr(tmp_path):
    """The default CLI route must preserve the requested field and diagnostics."""
    xyz=tmp_path/"magnetic-hessian.xyz"
    xyz.write_text("2\nMagnetic Hessian CLI acceptance\nH 0 0 -0.3704240476321\nH 0 0 0.3704240476321\n",encoding="utf-8")
    for field in (["--bfield","0,0,0.006"],["--bfield=0,0,0.006"]):
        process=subprocess.run([os.environ["DFT_RS_CLI"],"hessian",str(xyz),"--xc","hf",
                                "--bfield-initial-guess","core",*field],capture_output=True,text=True)
        assert process.returncode==0,process.stderr
        assert "analytic nuclear response" in process.stdout
        assert re.search(r"\[magnetic-scf\].*iter.*E=.*residual=",process.stderr),process.stderr


def command(tmp_path, feature, numbers, positions, controls, cell=None):
    native_bohr = 1/dft_rs.native.BOHR_PER_ANGSTROM
    xyz = tmp_path/(feature+".xyz")
    xyz.write_text(str(len(numbers))+"\nConnected response acceptance\n"+"".join(
        f"{z} {x*native_bohr:.16g} {y*native_bohr:.16g} {zz*native_bohr:.16g}\n"
        for z, (x, y, zz) in zip(numbers, positions)))
    argv = [os.environ["DFT_RS_CLI"], feature, str(xyz), "--format", "json"]
    if cell is not None:
        argv += ["--cell", json.dumps((np.asarray(cell)*native_bohr).tolist())]
    for key, value in controls.items():
        argv += ["--"+key.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)]
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    if feature == "periodic-magnetic-scf":
        assert re.search(r"\[spinor-scf\].*iter.*E=.*residual=",result.stderr),result.stderr
    if feature == "periodic-efield":
        assert re.search(r"\[periodic-field-scf\].*iter.*E=.*residual=",result.stderr),result.stderr
    return json.loads(result.stdout)


def test_magnetically_balanced_complex_metric_on_all_surfaces(tmp_path):
    numbers, positions = [1, 1], np.array([[0, 0, -0.7], [0, 0, 0.7]])
    controls = dict(dirac_integrals="rmb", basis="sto-3g", speed_of_light=5.0,
                    nuclear_rms_radii=[0.25, 0.3], nuclear_radius_direction=[0.02, -0.01],
                    magnetic_field=[0.3, -0.2, 0.4], magnetic_direction=[0.04, 0.03, -0.02],
                    gauge_origin=[0.1, 0.2, -0.1], response_order=4)
    direct = dft_rs.no_pair_projector(numbers, positions, **controls)
    atoms = Atoms(numbers, positions=positions*Bohr)
    atomic = dft_rs.DftRs.get_no_pair_projector(atoms, units="atomic", **controls)
    physical = dict(controls)
    for key in ("nuclear_rms_radii", "nuclear_radius_direction", "gauge_origin"):
        physical[key] = np.asarray(controls[key])*Bohr
    adapted = dft_rs.DftRs.get_no_pair_projector(atoms, **physical)
    cli = command(tmp_path, "no-pair-projector", numbers, positions, controls)
    assert direct["converged"] and "magnetic balance" in direct["balance"]
    h = np.asarray(direct["dirac_hamiltonian_coefficients_real"])[0]+1j*np.asarray(direct["dirac_hamiltonian_coefficients_imaginary"])[0]
    s = np.asarray(direct["dirac_overlap_coefficients_real"])[0]+1j*np.asarray(direct["dirac_overlap_coefficients_imaginary"])[0]
    assert np.max(abs(s.imag)) > 1e-8
    values, u = np.linalg.eigh(s)
    x = (u/np.sqrt(values))@u.conj().T
    values, u = np.linalg.eigh(x@h@x)
    c = x@u[:, values > -25]
    expected = c@c.conj().T
    actual = np.asarray(direct["projector_coefficients_real"])[0]+1j*np.asarray(direct["projector_coefficients_imaginary"])[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
    for key in ("projector_coefficients_real", "projector_coefficients_imaginary",
                "dirac_overlap_coefficients_real", "dirac_overlap_coefficients_imaginary",
                "balanced_kinetic_coefficients_real", "balanced_kinetic_coefficients_imaginary"):
        for report in (atomic, cli):
            np.testing.assert_allclose(report[key], direct[key], rtol=1e-9, atol=1e-10)
        scale = Hartree if key.startswith("balanced_kinetic") else 1.0
        np.testing.assert_allclose(adapted[key], np.asarray(direct[key])*scale, rtol=1e-9, atol=1e-10)
    for key in ("projector_defects", "metric_defects", "relative_commutator_defects", "number_defects"):
        assert np.max(direct[key]) < 1e-10
    print("RMB public raw", {k: direct[k] for k in ("projector_defects", "metric_defects", "relative_commutator_defects")})


def test_joint_mixed_energy_and_finite_field_gradient_public_routes(tmp_path):
    numbers, positions = [1, 1], np.array([[0, 0, 0], [0, 0, 1.4]])
    controls = dict(xc="hf", basis="sto-3g", gauge="london", conv_tol=1e-13, conv_tol_dp=1e-12,
                    magnetic_directions=[[0, 0, 0], [0, 0, 0], [0.6, 0, 0], [0.6, 0, 0]],
                    electric_directions=[[0, 0, 0.2], [0, 0, 0.2], [0, 0, 0], [0, 0, 0]])
    direct = dft_rs.magnetizability(numbers, positions, **controls)
    atoms = Atoms(numbers, positions=positions*Bohr)
    atomic = dft_rs.DftRs(units="atomic").get_magnetizability(atoms, **controls)
    physical = dict(controls, electric_directions=np.asarray(controls["electric_directions"])*Hartree/Bohr,
                    conv_tol=controls["conv_tol"]*Hartree)
    adapted = dft_rs.DftRs().get_magnetizability(atoms, **physical)
    cli = command(tmp_path, "magnetizability", numbers, positions, controls)
    assert abs(direct["magnetic_mixed_derivative"]) > 1e-8
    for report in (atomic, cli):
        np.testing.assert_allclose(report["magnetic_mixed_derivative"], direct["magnetic_mixed_derivative"], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(adapted["magnetic_mixed_derivative"], direct["magnetic_mixed_derivative"]*Hartree, rtol=1e-9, atol=1e-9)
    finite = dict(xc="hf", basis="sto-3g", charge=1, multiplicity=2,
                  bfield=[0.1, 0.05, 0.08], field=[0.01, -0.015, 0.02],
                  spin_zeeman=False, conv_tol=1e-13, conv_tol_dp=1e-12)
    gradient = dft_rs.magnetic_field_gradient(numbers, positions, **finite)
    gcli = command(tmp_path, "bfield-gradient", numbers, positions, finite)
    np.testing.assert_allclose(gcli["gradient"], gradient["gradient"], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(np.sum(gradient["gradient"], axis=0), -np.asarray(finite["field"]), atol=1e-10)
    assert gradient["translation_residual"] < 1e-10
    print("joint-field public raw", {k: direct[k] for k in (
        "magnetic_mixed_response_residual", "magnetic_mixed_projector_defect",
        "magnetic_mixed_stationarity_defect", "magnetic_mixed_reference_defect")},
        "translation", gradient["translation_residual"])


PERIODIC = dict(xc="hf", basis="sto-3g", k_mesh=[1, 1, 1], overlap_cutoff=4.0,
                grid_level=0, supercell=[1, 1, 1], conv_tol=1e-12, conv_tol_grad=1e-12,
                conv_tol_dp=1e-12, response_tolerance=1e-12)


def test_born_lattice_long_wave_component_and_order_dependent_units(tmp_path):
    numbers, positions, cell = [1, 1], np.array([[0, 0, -0.7], [0, 0, 0.7]]), np.eye(3)*9
    born = np.concatenate([np.eye(3), -np.eye(3)]).tolist()
    controls = dict(PERIODIC, path=[[0, 0, 0], [0.1, 0, 0]], points=1, response_order=4,
                    wavevector_direction=[0.2, -0.1, 0.3], lattice_flexoelectric=True, born_charge_rows=born)
    direct = dft_rs.phonon_bands(numbers, positions, cell, **controls)
    atoms = Atoms(numbers, positions=positions*Bohr, cell=cell*Bohr, pbc=True)
    atomic = dft_rs.DftRs(units="atomic").get_phonon_bands(atoms, **controls)
    physical = dict(controls, overlap_cutoff=4*Bohr, wavevector_direction=np.asarray(controls["wavevector_direction"])/Bohr,
                    conv_tol=1e-12*Hartree, conv_tol_grad=1e-12*Hartree)
    adapted = dft_rs.DftRs().get_phonon_bands(atoms, **physical)
    cli = command(tmp_path, "phonon-bands", numbers, positions, controls, cell)
    assert direct["long_wave_converged"]
    for n, row in enumerate(direct["long_wave_relaxation"]):
        assert row["order"] == n and max(row["relative_residuals"]) < 1e-10
        for report in (atomic, cli):
            np.testing.assert_allclose(report["long_wave_relaxation"][n]["displacement_imaginary"], row["displacement_imaginary"], rtol=1e-8, atol=1e-10)
        for key, scale in (("displacement_real", Bohr**n), ("displacement_imaginary", Bohr**n),
                           ("polarization_real", Bohr**(n-3)), ("polarization_imaginary", Bohr**(n-3))):
            np.testing.assert_allclose(adapted["long_wave_relaxation"][n][key], np.asarray(row[key])*scale, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(adapted["born_lattice_flexoelectric"], np.asarray(direct["born_lattice_flexoelectric"])/Bohr, atol=1e-10)
    print("long-wave public raw", direct["long_wave_uniform_force_defect"], [r["relative_residuals"] for r in direct["long_wave_relaxation"]])

    # Independently generated Born charges also use the same relaxation driver.
    lih_positions = [[0, 0, 0], [0, 0, 3.0]]
    generated_controls = {k: v for k, v in controls.items() if k != "born_charge_rows"}
    generated = dft_rs.phonon_bands([3, 1], lih_positions, cell, **generated_controls)
    supplied_controls = dict(generated_controls, born_charge_rows=generated["long_wave_born_charges"])
    supplied = dft_rs.phonon_bands([3, 1], lih_positions, cell, **supplied_controls)
    assert generated["long_wave_converged"] and supplied["long_wave_converged"]
    np.testing.assert_allclose(generated["born_lattice_flexoelectric"], supplied["born_lattice_flexoelectric"], rtol=1e-8, atol=1e-10)


def test_dfpt_gamma_vertices_feed_the_common_pairing_and_gap_drivers(tmp_path):
    numbers, positions, cell = [1, 1], np.array([[0, 0, -0.7], [0, 0, 0.7]]), np.eye(3)*9
    controls = dict(PERIODIC, electronic_temperature=3000.0, scdft_pairing=True,
                    screened_pair_coulomb=[[0.0, 0.0], [0.0, 0.0]], eliashberg_t_min_k=10.0)
    direct = dft_rs.electron_phonon(numbers, positions, cell, **controls)
    atoms = Atoms(numbers, positions=positions*Bohr, cell=cell*Bohr, pbc=True)
    atomic = dft_rs.DftRs(units="atomic").get_electron_phonon(atoms, **controls)
    physical = dict(controls, overlap_cutoff=4*Bohr, conv_tol=1e-12*Hartree, conv_tol_grad=1e-12*Hartree)
    adapted = dft_rs.DftRs().get_electron_phonon(atoms, **physical)
    cli = command(tmp_path, "electron-phonon", numbers, positions, controls, cell)
    for report in (atomic, cli):
        np.testing.assert_allclose(report["scdft_pairing_kernel"], direct["scdft_pairing_kernel"], rtol=1e-8, atol=1e-11)
        np.testing.assert_allclose(report["scdft_mass_renormalization"], direct["scdft_mass_renormalization"], rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(adapted["scdft_pairing_kernel"], np.asarray(direct["scdft_pairing_kernel"])*Hartree, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(adapted["scdft_inputs"]["pair_couplings_squared"], np.asarray(direct["scdft_inputs"]["pair_couplings_squared"])*Hartree**2, rtol=1e-8, atol=1e-10)
    inputs = direct["scdft_inputs"]
    component = dft_rs.scdft_pairing_kernel(**inputs)
    np.testing.assert_allclose(component["pairing_kernel"], direct["scdft_pairing_kernel"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(component["mass_renormalization"], direct["scdft_mass_renormalization"], rtol=1e-12, atol=1e-13)
    normal = dft_rs.superconducting_state(**inputs, pairing_model="lm2005", electron_number=direct["scdft_electron_number"],
                                         initial_gap_real=[0.0]*len(inputs["band_energies"]), response_order=4)
    assert normal["converged"] and normal["warnings"]
    assert np.max(normal["number_residuals"]) < 1e-10
    print("DFPT pairing public raw", direct["scdft_selected_modes"], direct["scdft_spin_defect"], normal["number_residuals"])


def test_finite_electric_bias_nuclear_derivatives_share_all_public_routes(tmp_path):
    numbers = [1, 1]
    positions = np.array([[0, 0, -0.7], [0, 0, 0.7]])
    cell = np.eye(3)*10
    atoms = Atoms(numbers, positions=positions*Bohr, cell=cell*Bohr, pbc=True)
    base = dict(PERIODIC, overlap_cutoff=5.0, field=[0.001, -0.0015, 0.002])
    probe = [-0.2, 0.1, -0.3, 0.2, -0.1, 0.3]
    for feature, native, adapted, tensor_key, order in (
        ("phonons", dft_rs.phonons, "get_phonons", "force_constants", 2),
        ("periodic-cubic-fc-along", dft_rs.periodic_cubic_fc_along,
         "get_periodic_cubic_fc_along", "contracted_force_constants", 3),
        ("periodic-quartic-fc-along", dft_rs.periodic_quartic_fc_along,
         "get_periodic_quartic_fc_along", "contracted_force_constants", 4),
    ):
        controls = {k: v for k, v in base.items() if k != "supercell"}
        if order > 2:
            controls["probe"] = probe
        direct = native(numbers, positions, cell, **controls)
        atomic = getattr(dft_rs.DftRs(units="atomic"), adapted)(atoms, **controls)
        physical = dict(controls, overlap_cutoff=5*Bohr,
                        field=np.asarray(controls["field"])*Hartree/Bohr,
                        conv_tol=1e-12*Hartree, conv_tol_grad=1e-12*Hartree)
        converted = getattr(dft_rs.DftRs(), adapted)(atoms, **physical)
        cli = command(tmp_path, feature, numbers, positions, controls, cell)
        for report in (atomic, cli):
            np.testing.assert_allclose(report[tensor_key], direct[tensor_key], rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(report["reference_field"], direct["reference_field"], atol=1e-14)
        np.testing.assert_allclose(converted[tensor_key], np.asarray(direct[tensor_key])*Hartree/Bohr**order,
                                   rtol=1e-8, atol=1e-8)
        assert direct["series_reference_residual"] < 1e-10
        assert "response_residuals" in direct
        assert max(direct["response_residuals"], default=0) <= 1e-12
        print("finite-bias nuclear public", feature, "reference", direct["series_reference_residual"],
              "response", direct.get("response_residuals"), flush=True)


@pytest.mark.parametrize("charge,multiplicity,spin_zeeman",[(0,1,False),(1,2,True)])
def test_molecular_joint_field_nuclear_tensors_share_all_public_routes(tmp_path,charge,multiplicity,spin_zeeman):
    """Public tensor/units checks complement the independent Rust force fits."""
    numbers = [1, 1]
    positions = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    atoms = Atoms(numbers, positions=positions*Bohr)
    base = dict(xc="hf", basis="sto-3g", bfield=[0.0, 0.0, 0.006],
                field=[0.0007, 0.0005, -0.001], spin_zeeman=spin_zeeman,
                charge=charge,multiplicity=multiplicity,
                bfield_initial_guess="core", conv_tol=1e-13,
                conv_tol_dp=1e-12, conv_tol_grad=1e-12,
                response_tolerance=1e-12)
    probe = [0.02, 0.0, -0.3, 0.02, 0.0, 0.3]
    for feature, native, adapted, tensor_key, order in (
        ("hessian", dft_rs.hessian_report, "get_hessian_report", "hessian", 2),
        ("cubic-fc-along", dft_rs.cubic_fc_along,
         "get_cubic_fc_along", "contracted_force_constants", 3),
        ("quartic-fc-along", dft_rs.quartic_fc_along,
         "get_quartic_fc_along", "contracted_force_constants", 4),
    ):
        controls = dict(base)
        if order > 2:
            controls["probe"] = probe
        direct = native(numbers, positions, **controls)
        atomic = getattr(dft_rs.DftRs(units="atomic"), adapted)(atoms, **controls)
        physical = dict(controls, field=np.asarray(base["field"])*Hartree/Bohr,
                        conv_tol=1e-13*Hartree, conv_tol_grad=1e-12*Hartree)
        converted = getattr(dft_rs.DftRs(), adapted)(atoms, **physical)
        cli = command(tmp_path, feature, numbers, positions, controls)
        for report in (direct, atomic, converted, cli):
            assert report["converged"] and report["response_converged"]
            assert report["charge"]==charge and report["multiplicity"]==multiplicity
            assert report["unrestricted"]==(multiplicity==2)
            assert (report["n_alpha"],report["n_beta"])==((1,0) if charge else (1,1))
            assert max(report["response_residuals"], default=0.0) <= 1e-12
            assert max(report["projector_defects"], default=0.0) < 1e-10
            assert max(report["stationarity_defects"], default=0.0) < 1e-10
            assert report["series_order"] >= order+2
            assert "raw_asymmetry" not in report
        for report in (atomic, cli):
            np.testing.assert_allclose(report[tensor_key], direct[tensor_key], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(converted[tensor_key], np.asarray(direct[tensor_key])*Hartree/Bohr**order,
                                   rtol=1e-8, atol=1e-8)
        translation=float(np.max(abs(np.asarray(direct[tensor_key]).reshape(6,2,3).sum(axis=1))))
        assert translation<1e-8
        print("molecular joint-field nuclear public", "charge",charge,"multiplicity",multiplicity,"Zeeman",spin_zeeman,feature, "degree", direct["series_order"],
              "reference", direct["reference_defect"], "response_max", max(direct["response_residuals"],default=None),"response_count",len(direct["response_residuals"]),
              "projector_max",max(direct["projector_defects"],default=None), "stationarity_max",max(direct["stationarity_defects"],default=None),"translation",translation,flush=True)


def _periodic_magnetic_nuclear_case(tmp_path):
    basis = tmp_path/"magnetic-nuclear.g94"
    basis.write_text("H 0\nS 1 1.0\n0.7 1.0\n****\n", encoding="utf-8")
    numbers = [1, 1]
    positions = np.array([[0.1, 0.05, -0.7], [-0.1, -0.05, 0.7]])
    cell = np.eye(3)*5.0
    atoms = Atoms(numbers, positions=positions*Bohr, cell=cell*Bohr, pbc=True)
    probe = np.array([0.07, -0.05, -0.3, -0.11, 0.09, 0.3])
    base = dict(xc="hf", basis=str(basis), guess="core", k_mesh=[1, 1, 1],
                magnetic_model="finite_gamma", magnetic_field=[0, 0, 2*np.pi/25],
                overlap_cutoff=12.0, magnetic_reciprocal_cutoff=16.0,
                magnetic_operator_tolerance=1e-10, nuclear_soc_scale=0.0,
                ewald_alpha=0.55, ewald_real_cutoff=16.0, ewald_reciprocal_cutoff=8.0,
                conv_tol=1e-14, conv_tol_dp=1e-13, conv_tol_grad=1e-13,
                response_tolerance=1e-12, max_iter=500)
    return numbers,positions,cell,atoms,probe,base


def test_periodic_joint_field_scf_iteration_diagnostics(tmp_path):
    """The canonical/alias CLI must log the actual joint-field SCF by default."""
    numbers,positions,cell,_,_,controls=_periodic_magnetic_nuclear_case(tmp_path)
    controls.pop("response_tolerance")
    controls["field"]=[0.0007,0.0005,-0.001]
    report=command(tmp_path,"periodic-magnetic-scf",numbers,positions,controls,cell)
    direct=dft_rs.periodic_magnetic_scf(numbers,positions,cell,**controls)
    atoms=Atoms(numbers,positions=positions*Bohr,cell=cell*Bohr,pbc=True)
    atomic=dft_rs.DftRs(units="atomic").get_periodic_magnetic_scf(atoms,**controls)
    for result in (report,direct,atomic):
        assert "static single-point Berry" in result["electric_field_model"]
        assert "integer full-spinor occupied subspace" in result["electric_field_model"]
    assert report["converged"] and report["scf_iterations"]>0
    np.testing.assert_allclose(report["reference_field"],controls["field"],atol=1e-14)
    np.testing.assert_allclose(report["energy_components"]["orbital_field"],
                               -np.asarray(controls["field"])@report["electric_dipole"],atol=1e-12)


def test_joint_field_branch_translation_and_nuclear_path(tmp_path):
    """A neutral translation parallel to B must not change the Berry branch."""
    numbers,positions,cell,_,probe,controls=_periodic_magnetic_nuclear_case(tmp_path)
    direction=probe.reshape(2,3).copy()
    direction[:,2]+=0.07
    controls.update(field=[0.0007,0.0005,-0.001],nuclear_order=4,nuclear_direction=direction.ravel().tolist())
    reference=None
    for shift in (0.0,1.26,2.6):
        shifted=positions+np.array([0,0,shift])
        atoms=Atoms(numbers,positions=shifted*Bohr,cell=cell*Bohr,pbc=True)
        direct=dft_rs.periodic_magnetic_scf(numbers,shifted,cell,**controls)
        atomic=dft_rs.DftRs(units="atomic").get_periodic_magnetic_scf(atoms,**controls)
        physical=dict(controls,overlap_cutoff=12*Bohr,magnetic_reciprocal_cutoff=16/Bohr,
                      ewald_alpha=0.55/Bohr,ewald_real_cutoff=16*Bohr,ewald_reciprocal_cutoff=8/Bohr,
                      conv_tol=1e-14*Hartree,conv_tol_grad=1e-13*Hartree,
                      field=np.asarray(controls["field"])*Hartree/Bohr)
        converted=dft_rs.DftRs().get_periodic_magnetic_scf(atoms,**physical)
        cli=command(tmp_path,"periodic-magnetic-scf",numbers,shifted,controls,cell)
        values=np.array([row["derivative"] for row in direct["nuclear_response"]["energy_path"]])
        if reference is None:reference=(values,np.asarray(direct["electric_dipole"]))
        np.testing.assert_allclose(values,reference[0],rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose(direct["electric_dipole"],reference[1],atol=1e-10)
        for report in (direct,atomic,converted,cli):
            assert report["converged"] and report["nuclear_response"]["converged"]
            assert "electron-number-weighted" in report["electric_field_model"]
            assert max(report["nuclear_response"]["response_residuals"])<=1e-12
        for report in (atomic,cli):
            np.testing.assert_allclose([row["derivative"] for row in report["nuclear_response"]["energy_path"]],values,rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose([row["derivative"] for row in converted["nuclear_response"]["energy_path"]],values*Hartree/Bohr**np.arange(5),rtol=1e-8,atol=1e-8)
        print("continued Berry branch public shift",shift,"energy",direct["energy"],"dipole",direct["electric_dipole"],
              "nuclear",values.tolist(),"response",direct["nuclear_response"]["response_residuals"],flush=True)


def test_real_periodic_berry_metric_translation_on_all_public_routes(tmp_path):
    """The Berry identity must use S from the SCF, even at a short image cutoff."""
    numbers,positions,cell,_,_,magnetic=_periodic_magnetic_nuclear_case(tmp_path)
    controls=dict(xc="hf",basis=magnetic["basis"],guess="core",k_mesh=[1,1,1],overlap_cutoff=4.0,
                  grid_level=0,field=[0,0,-0.001],conv_tol=1e-14,conv_tol_dp=1e-13,conv_tol_grad=1e-13)
    baseline=None
    for shift in (0.0,1.26,2.6):
        shifted=positions+np.array([0,0,shift])
        atoms=Atoms(numbers,positions=shifted*Bohr,cell=cell*Bohr,pbc=True)
        direct=dft_rs.periodic_efield(numbers,shifted,cell,**controls)
        atomic=dft_rs.DftRs(units="atomic").get_periodic_efield(atoms,**controls)
        physical=dict(controls,overlap_cutoff=4*Bohr,field=np.asarray(controls["field"])*Hartree/Bohr,
                      conv_tol=1e-14*Hartree,conv_tol_grad=1e-13*Hartree)
        converted=dft_rs.DftRs().get_periodic_efield(atoms,**physical)
        cli=command(tmp_path,"periodic-efield",numbers,shifted,controls,cell)
        if baseline is None:baseline=(direct["field_energy"],np.asarray(direct["dipole"]))
        for report in (direct,atomic,cli):
            assert report["converged"] and report["series_reference_residual"]<1e-12
            np.testing.assert_allclose(report["field_energy"],baseline[0],atol=2e-11,rtol=0)
            np.testing.assert_allclose(report["dipole"],baseline[1],atol=2e-10,rtol=0)
        np.testing.assert_allclose(converted["field_energy"],direct["field_energy"]*Hartree,atol=2e-10,rtol=0)
        np.testing.assert_allclose(converted["dipole"],np.asarray(direct["dipole"])*Bohr,atol=2e-10,rtol=0)
        print("real Berry metric public shift",shift,"energy",direct["field_energy"],"dipole",direct["dipole"],
              "reference",direct["series_reference_residual"],flush=True)


def test_periodic_electric_contracted_residuals_all_public_routes(tmp_path):
    """Reduced real nuclear paths must preserve actual solver diagnostics."""
    numbers,positions,_,_,probe,magnetic=_periodic_magnetic_nuclear_case(tmp_path)
    cell=np.eye(3)*9.0
    atoms=Atoms(numbers,positions=positions*Bohr,cell=cell*Bohr,pbc=True)
    base=dict(xc="hf",basis=magnetic["basis"],k_mesh=[1,1,1],overlap_cutoff=4.0,
              grid_level=0,conv_tol=1e-12,conv_tol_grad=1e-12,conv_tol_dp=1e-12,
              response_tolerance=1e-12,field=[0.001,-0.0015,0.002],probe=probe.tolist())
    for order,native,adapted,feature in (
        (3,dft_rs.periodic_cubic_fc_along,"get_periodic_cubic_fc_along","periodic-cubic-fc-along"),
        (4,dft_rs.periodic_quartic_fc_along,"get_periodic_quartic_fc_along","periodic-quartic-fc-along")):
        direct=native(numbers,positions,cell,**base)
        atomic=getattr(dft_rs.DftRs(units="atomic"),adapted)(atoms,**base)
        physical=dict(base,overlap_cutoff=4*Bohr,conv_tol=1e-12*Hartree,conv_tol_grad=1e-12*Hartree,
                      field=np.asarray(base["field"])*Hartree/Bohr)
        converted=getattr(dft_rs.DftRs(),adapted)(atoms,**physical)
        cli=command(tmp_path,feature,numbers,positions,base,cell)
        for report in (direct,atomic,converted,cli):
            assert report["response_converged"]
            assert len(report["response_residuals"])>=report["path_evaluations"]
            assert max(report["response_residuals"])<=1e-12
        for report in (atomic,cli):
            np.testing.assert_allclose(report["contracted_force_constants"],direct["contracted_force_constants"],rtol=1e-8,atol=1e-10)
            np.testing.assert_allclose(report["response_residuals"],direct["response_residuals"],atol=1e-12)
        np.testing.assert_allclose(converted["contracted_force_constants"],np.asarray(direct["contracted_force_constants"])*Hartree/Bohr**order,rtol=1e-8,atol=1e-8)
        print("electric nuclear residual public",feature,"paths",direct["path_evaluations"],
              "response_max",max(direct["response_residuals"]),"count",len(direct["response_residuals"]),flush=True)


@pytest.mark.parametrize("electric",[False,True],ids=["magnetic","joint"])
def test_charged_coherent_spinor_nuclear_path_and_units(tmp_path,electric):
    """A physical three-face field produces laboratory spin coherence."""
    numbers,positions,cell,atoms,probe,base=_periodic_magnetic_nuclear_case(tmp_path)
    q=2*np.pi/25
    controls=dict(base,charge=1,charge_compensation="uniform",magnetic_field=[q,q,q],
                  nuclear_soc_scale=1.0,nuclear_order=4,nuclear_direction=probe.tolist())
    if electric:
        controls["field"]=[0.0007,0.0005,-0.001]
    direct=dft_rs.periodic_magnetic_scf(numbers,positions,cell,**controls)
    atomic=dft_rs.DftRs(units="atomic").get_periodic_magnetic_scf(atoms,**controls)
    physical=dict(controls,overlap_cutoff=12*Bohr,magnetic_reciprocal_cutoff=16/Bohr,
                  ewald_alpha=0.55/Bohr,ewald_real_cutoff=16*Bohr,ewald_reciprocal_cutoff=8/Bohr,
                  conv_tol=1e-14*Hartree,conv_tol_grad=1e-13*Hartree)
    if electric:
        physical["field"]=np.asarray(controls["field"])*Hartree/Bohr
    converted=dft_rs.DftRs().get_periodic_magnetic_scf(atoms,**physical)
    cli=command(tmp_path,"periodic-magnetic-scf",numbers,positions,controls,cell)
    density=np.asarray(direct["density_real"])[0]+1j*np.asarray(direct["density_imag"])[0]
    size=density.shape[0]//2
    coherence=float(np.max(abs(density[:size,size:])))
    assert coherence>1e-3
    interaction=direct["energy_components"]["hartree"]+direct["energy_components"]["full_exchange"]
    assert abs(interaction)<2e-12
    for report in (direct,atomic,converted,cli):
        assert report["converged"] and report["nuclear_response"]["converged"]
        assert report["flux_quanta"]==[1,1,1]
        assert report["particle_number_error"]<1e-10
        assert max(report["nuclear_response"]["response_residuals"],default=0.0)<=1e-12
    if electric:
        for report in (direct,atomic,cli):
            np.testing.assert_allclose(report["reference_field"],controls["field"],atol=1e-14)
            np.testing.assert_allclose(report["electric_dipole"],direct["electric_dipole"],atol=1e-10)
            np.testing.assert_allclose(report["energy_components"]["orbital_field"],-np.asarray(controls["field"])@report["electric_dipole"],atol=1e-12)
        np.testing.assert_allclose(converted["reference_field"],physical["field"],atol=1e-12)
        np.testing.assert_allclose(converted["electric_dipole"],np.asarray(direct["electric_dipole"])*Bohr,atol=1e-10)
    for n,row in enumerate(direct["nuclear_response"]["energy_path"]):
        for report in (atomic,cli):
            np.testing.assert_allclose(report["nuclear_response"]["energy_path"][n]["derivative"],row["derivative"],rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose(converted["nuclear_response"]["energy_path"][n]["derivative"],row["derivative"]*Hartree/Bohr**n,rtol=1e-8,atol=1e-8)
    print("charged spinor nuclear public electric",electric,"coherence",coherence,"self_interaction",interaction,
          "response",direct["nuclear_response"]["response_residuals"],
          "reference",direct["nuclear_response"]["reference_defect"],flush=True)


@pytest.mark.parametrize("electric",[False,True],ids=["magnetic","joint"])
def test_periodic_magnetic_nuclear_response_common_driver_and_units(tmp_path,electric):
    """Finite-torus response; the independent SCF derivative fits are in Rust."""
    numbers,positions,cell,atoms,probe,base=_periodic_magnetic_nuclear_case(tmp_path)
    if electric:
        base["field"]=[0.0007,0.0005,-0.001]
    derivative_path = None
    for mode, order in [("path", 4), ("gradient", 1), ("hessian", 2),
                        ("cubic", 3), ("quartic", 4)]:
        controls = dict(base, nuclear_order=order)
        if mode == "path":
            controls["nuclear_direction"] = probe.tolist()
        elif order >= 3:
            controls["probe"] = probe.tolist()
        direct = dft_rs.periodic_magnetic_scf(numbers, positions, cell, **controls)["nuclear_response"]
        atomic = dft_rs.DftRs(units="atomic").get_periodic_magnetic_scf(atoms, **controls)["nuclear_response"]
        physical = dict(controls, overlap_cutoff=12*Bohr,
                        magnetic_reciprocal_cutoff=16/Bohr, ewald_alpha=0.55/Bohr,
                        ewald_real_cutoff=16*Bohr, ewald_reciprocal_cutoff=8/Bohr,
                        conv_tol=1e-14*Hartree, conv_tol_grad=1e-13*Hartree)
        if electric:
            physical["field"]=np.asarray(controls["field"])*Hartree/Bohr
        converted = dft_rs.DftRs().get_periodic_magnetic_scf(atoms, **physical)["nuclear_response"]
        cli = command(tmp_path, "periodic-magnetic-scf", numbers, positions, controls, cell)["nuclear_response"]
        for report in (direct, atomic, converted, cli):
            assert report["converged"] and max(report["response_residuals"], default=0.0) <= 1e-12
            assert max(report["projector_defects"], default=0.0) < 1e-10
            assert "raw_asymmetry" not in report
        if mode == "path":
            derivative_path = [row["derivative"] for row in direct["energy_path"]]
            for n, row in enumerate(direct["energy_path"]):
                for report in (atomic, cli):
                    np.testing.assert_allclose(report["energy_path"][n]["derivative"], row["derivative"], rtol=1e-8, atol=1e-10)
                np.testing.assert_allclose(converted["energy_path"][n]["derivative"], row["derivative"]*Hartree/Bohr**n, rtol=1e-8, atol=1e-8)
        else:
            key = mode if order <= 2 else "contracted_force_constants"
            for report in (atomic, cli):
                np.testing.assert_allclose(report[key], direct[key], rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(converted[key], np.asarray(direct[key])*Hartree/Bohr**order, rtol=1e-8, atol=1e-8)
            tensor = np.asarray(direct[key])
            contracted = tensor.ravel()@probe if order == 1 else probe@tensor@probe
            np.testing.assert_allclose(contracted, derivative_path[order], rtol=1e-7, atol=1e-9)
        print("periodic magnetic nuclear public", "electric",electric, mode, "reference", direct["reference_defect"],
              "energy_defect", direct["reference_energy_defect"], "response_max", max(direct["response_residuals"], default=None),
              "projector_max", max(direct["projector_defects"], default=None),
              "stationarity_max", max(direct["stationarity_defects"], default=None),
              "paths", direct["path_evaluations"], flush=True)
