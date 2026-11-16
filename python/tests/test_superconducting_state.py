# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Canonical gap/number/phase reports and independent dense BdG oracles."""
import json
import math
import os
import subprocess

import numpy as np
import pytest
from dft_rs.units import HARTREE_IN_EV as Hartree
import dft_rs

KB = 3.166811563455608e-6


@pytest.mark.parametrize("kt,phase", [(0.004, "superconducting"), (0.009, "normal")])
def test_thermodynamic_phase_selection_and_public_routes(kt, phase):
    w, z, target_gap = 0.73, 0.2, 0.01
    coupling = 2 * target_gap / (w * math.tanh(target_gap / (2 * 0.004)))
    controls = dict(
        band_energies=[0.031], band_weights=[w], mass_renormalization=[z],
        pairing_kernel=[[-coupling * (1 + z)]], electronic_temperature=kt / KB,
        electron_number=w, response_tolerance=1e-13, scdft_phase_selection=True,
    )
    native = dft_rs.superconducting_state(**controls)
    atomic = dft_rs.DftRs.get_superconducting_state(units="atomic", **controls)
    adapted = dft_rs.DftRs.get_superconducting_state(**(controls | {
        "band_energies": [0.031 * Hartree],
        "pairing_kernel": [[-coupling * (1 + z) * Hartree]],
    }))
    command = [os.environ["DFT_RS_CLI"], "superconducting-state", "--format", "json"]
    for key, value in controls.items():
        command.extend(("--" + key.replace("_", "-"), json.dumps(value)))
    process = subprocess.run(command, check=True, capture_output=True, text=True)
    cli = json.loads(process.stdout)
    assert "[scdft] iter" in process.stderr
    for report, scale, unit in (
        (native, 1.0, "hartree"), (atomic, 1.0, "hartree"),
        (cli, 1.0, "hartree"), (adapted, Hartree, "eV"),
    ):
        assert report["selected_phase"] == phase
        assert report["units"]["condensation_free_energy"] == unit
        assert report["units"]["pairing_stability_margin"] == "dimensionless"
        for key in ("gap_real", "condensation_free_energy", "normal_chemical_potential",
                    "pairing_energy", "gap_residuals"):
            np.testing.assert_allclose(report[key], np.asarray(native[key]) * scale, atol=5e-13)
        for key in ("pairing_stability_margin", "normal_pairing_eigenvalue", "number_residuals",
                    "entropy_over_boltzmann", "normal_entropy_over_boltzmann"):
            np.testing.assert_allclose(report[key], native[key], atol=5e-13)
    expected = 0.0
    if phase == "superconducting":
        expected = target_gap**2 / coupling - 2 * w * kt * math.log(math.cosh(target_gap / (2 * kt)))
        assert abs(native["gap_real"][0]) == pytest.approx(target_gap, abs=1e-14)
    assert native["condensation_free_energy"] == pytest.approx(expected, abs=1e-14)
    print("SCDFT phase", phase, "gap", native["gap_real"], "dF", native["condensation_free_energy"],
          "margin", native["pairing_stability_margin"], "residuals",
          native["gap_residuals"], native["number_residuals"])


def _critical_temperature_reports(controls):
    native = dft_rs.superconducting_state(**controls)
    generic = dft_rs.run_feature("superconducting-state", **controls)
    atomic = dft_rs.DftRs.get_superconducting_state(units="atomic", **controls)
    adapted_inputs = {
        key: (np.asarray(value) * Hartree).tolist()
        if key in ("band_energies", "pairing_kernel", "phonon_energies", "screened_pair_coulomb")
        else (np.asarray(value) * Hartree**2).tolist()
        if key == "pair_couplings_squared"
        else value
        for key, value in controls.items()
    }
    adapted = dft_rs.DftRs.get_superconducting_state(**adapted_inputs)
    command = [os.environ["DFT_RS_CLI"], "superconducting-state", "--format", "json"]
    for key, value in controls.items():
        command.extend(("--" + key.replace("_", "-"),
                        value if isinstance(value, str) else json.dumps(value)))
    process = subprocess.run(command, check=True, capture_output=True, text=True)
    cli = json.loads(process.stdout)
    numeric = (
        "critical_temperature_k", "critical_temperature_bracket_k",
        "leading_pairing_eigenvalue", "chemical_potential", "linearized_gap_mode",
        "normal_occupations", "mass_renormalization", "linearized_gap_residual",
        "pairing_eigenvector_residual", "number_residual", "kernel_asymmetry",
    )
    for report in (generic, atomic, cli, adapted):
        assert report["converged"]
        assert report["units"]["critical_temperature_k"] == "K"
        assert report["units"]["critical_temperature_bracket_k"] == "K"
        scale = Hartree if report is adapted else 1.0
        for key in numeric:
            factor = scale if key in ("chemical_potential", "kernel_asymmetry") else 1.0
            np.testing.assert_allclose(report[key], np.asarray(native[key]) * factor,
                                       rtol=5e-11, atol=1e-11)
    assert native["units"]["chemical_potential"] == "hartree"
    assert adapted["units"]["chemical_potential"] == "eV"
    assert native["units"]["linearized_gap_mode"] == "dimensionless"
    print("SCDFT onset", native["pairing_model"], {key: native[key] for key in numeric})
    return native


def test_critical_temperature_weighted_bands_and_public_routes():
    energies = np.array([-0.009, 0.0027, 0.012])
    weights = np.array([0.2, 0.47, 0.91])
    mass = np.array([0.24, 0.45, 0.08])
    kernel = np.array([
        [-0.024, -0.017, -0.026], [-0.017, 0.004, -0.031], [-0.026, -0.031, -0.004],
    ])
    target = 1.46
    kt_ref = 0.006

    def dense_untransformed(kt, candidate_kernel):
        low, high = energies.min() - 35 * kt, energies.max() + 35 * kt
        for _ in range(90):
            middle = (low + high) / 2
            occupations = 2 / (1 + np.exp((energies - middle) / kt))
            if occupations @ weights > target:
                high = middle
            else:
                low = middle
        mu = (low + high) / 2
        xi = energies - mu
        assert np.all(np.abs(xi) > 1e-7)
        q = np.tanh(xi / (2 * kt)) / (2 * xi)
        # A is the original gap map and is generally nonsymmetric.
        operator = -candidate_kernel * (weights * q)[None, :] / (1 + mass)[:, None]
        values, vectors = np.linalg.eig(operator)
        assert np.max(np.abs(values.imag)) < 1e-13
        leader = np.argmax(values.real)
        return values[leader].real, vectors[:, leader].real, mu

    kernel /= dense_untransformed(kt_ref, kernel)[0]
    eigen, mode, mu = dense_untransformed(kt_ref, kernel)
    assert abs(eigen - 1) < 1e-13
    mode *= np.sign(mode[np.argmax(np.abs(mode))])
    controls = dict(
        pairing_model="supplied", band_energies=energies.tolist(), band_weights=weights.tolist(),
        mass_renormalization=mass.tolist(), pairing_kernel=kernel.tolist(), electron_number=target,
        scdft_temperature_bracket=[0.004 / KB, 0.011 / KB], response_tolerance=1e-12,
    )
    report = _critical_temperature_reports(controls)
    assert abs(report["critical_temperature_k"] * KB - kt_ref) < 1e-13
    assert abs(report["chemical_potential"] - mu) < 5e-14
    np.testing.assert_allclose(report["linearized_gap_mode"], mode, rtol=2e-11, atol=1e-12)
    assert abs(weights @ report["normal_occupations"] - target) < 1e-13
    for name in ("linearized_gap_residual", "pairing_eigenvector_residual", "number_residual"):
        assert report[name] < 1e-12
    with pytest.raises(ValueError, match="SCDFT critical temperature is not bracketed"):
        dft_rs.superconducting_state(**(controls | {"scdft_temperature_bracket": [5000.0, 6000.0]}))


@pytest.mark.parametrize("model", ["lm2005-off-diagonal", "lm2005"])
def test_critical_temperature_regenerates_phonon_kernel_at_each_trial(model):
    energy, weight, xi, kt, phonon, coulomb = 0.03, 0.72, 0.007, 0.004, 0.025, 0.001
    q = math.tanh(xi / (2 * kt)) / (2 * xi)
    off_diagonal = 2 * (contour(xi, xi, phonon, kt) - contour(xi, -xi, phonon, kt))
    off_diagonal /= math.tanh(xi / (2 * kt))**2
    explicit_mass = {"mass_renormalization": [0.2]} if model == "lm2005-off-diagonal" else {}
    if explicit_mass:
        coupling = (-(1 + explicit_mass["mass_renormalization"][0]) / (weight * q)
                    - coulomb) / off_diagonal
    else:
        generated_mass_factor = mass_factor(xi, xi, phonon, kt)
        coupling = -(1 + weight * q * coulomb) / (weight * (generated_mass_factor + q * off_diagonal))
    assert coupling > 0
    controls = dict(
        pairing_model=model, band_energies=[energy], band_weights=[weight],
        electron_number=weight * (1 - 2 * xi * q), phonon_energies=[phonon],
        pair_couplings_squared=[[coupling]], screened_pair_coulomb=[[coulomb]],
        scdft_temperature_bracket=[0.003 / KB, 0.006 / KB], response_tolerance=1e-12,
        **explicit_mass,
    )
    result = _critical_temperature_reports(controls)
    assert abs(result["critical_temperature_k"] * KB - kt) < 1e-13
    assert abs(result["chemical_potential"] - (energy - xi)) < 1e-13
    z = explicit_mass["mass_renormalization"][0] if explicit_mass else weight * coupling * generated_mass_factor
    np.testing.assert_allclose(result["mass_renormalization"], [z], rtol=5e-11, atol=1e-12)


def test_full_nambu_interband_density_dense_oracle_and_all_public_routes():
    rng=np.random.default_rng(417)
    def random_matrix():
        return rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
    a=random_matrix()*0.035+np.eye(4)
    s=a.conj().T@a
    h=random_matrix()*0.08
    h=(h+h.conj().T)/2
    delta=random_matrix()*0.025
    delta=delta-delta.T
    dh=random_matrix()*0.02
    dh=(dh+dh.conj().T)/2
    ds=random_matrix()*0.008
    ds=(ds+ds.conj().T)/2
    dd=random_matrix()*0.01
    dd=dd-dd.T
    controls=dict(pairing_model="nambu",chemical_potential=0.04,chemical_potential_direction=0.008,
        electronic_temperature=50000.0,temperature_direction=700.0,response_order=4,response_tolerance=1e-12)
    for prefix,base,direction in (("nambu_normal",h,dh),("nambu_pairing",delta,dd),("nambu_overlap",s,ds)):
        for part in ("real","imaginary"):
            controls[prefix+"_"+part]=(base.real if part=="real" else base.imag).tolist()
            controls[prefix+"_direction_"+part]=(direction.real if part=="real" else direction.imag).tolist()
    direct=dft_rs.superconducting_state(**controls)
    generic=dft_rs.run_feature("superconducting-state",**controls)
    atomic=dft_rs.DftRs.get_superconducting_state(units="atomic",**controls)
    converted=dict(controls)
    for name in converted:
        if name.startswith(("nambu_normal_","nambu_pairing_")) or name in ("chemical_potential","chemical_potential_direction"):
            converted[name]=(np.asarray(converted[name])*Hartree).tolist()
    adapted=dft_rs.DftRs.get_superconducting_state(**converted)
    argv=[os.environ["DFT_RS_CLI"],"superconducting-state","--format","json"]
    for key,value in controls.items():
        argv += ["--"+key.replace("_","-"),value if isinstance(value,str) else json.dumps(value)]
    process=subprocess.run(argv,capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    cli=json.loads(process.stdout)
    assert direct["converged"] and direct["ensemble"]=="grand_canonical"
    r=np.asarray(direct["nambu_density_coefficients_real"])+1j*np.asarray(direct["nambu_density_coefficients_imaginary"])
    for key in ("nambu_density_coefficients_real","nambu_density_coefficients_imaginary",
                "normal_density_coefficients_real","normal_density_coefficients_imaginary",
                "anomalous_density_coefficients_real","anomalous_density_coefficients_imaginary","electron_number_coefficients"):
        for report in (generic,atomic,adapted,cli):
            np.testing.assert_allclose(report[key],direct[key],rtol=1e-9,atol=1e-11)
    np.testing.assert_allclose(adapted["nambu_hamiltonian_coefficients_real"],np.asarray(direct["nambu_hamiltonian_coefficients_real"])*Hartree,rtol=1e-10,atol=1e-11)
    def dense(t):
        overlap=s+t*ds
        normal=h+t*dh-(controls["chemical_potential"]+t*controls["chemical_potential_direction"])*overlap
        pair=delta+t*dd
        operator=np.block([[normal,pair],[pair.conj().T,-normal.conj()]])
        metric=np.block([[overlap,np.zeros((4,4))],[np.zeros((4,4)),overlap.conj()]])
        values,u=np.linalg.eigh(metric)
        x=(u/np.sqrt(values))@u.conj().T
        values,u=np.linalg.eigh(x@operator@x)
        coefficients=x@u
        occupation=1/(1+np.exp(values/((controls["electronic_temperature"]+t*controls["temperature_direction"])*KB)))
        return (coefficients*occupation)@coefficients.conj().T
    np.testing.assert_allclose(r[0],dense(0),rtol=1e-10,atol=1e-12)
    steps=np.array([0.24,0.18,0.12,0.06])
    estimates=[[],[],[],[]]
    for step in steps:
        minus2,minus,zero,plus,plus2=[dense(t*step) for t in (-2,-1,0,1,2)]
        estimates[0].append((plus-minus)/(2*step))
        estimates[1].append((plus-2*zero+minus)/step**2)
        estimates[2].append((plus2-2*plus+2*minus-minus2)/(2*step**3))
        estimates[3].append((plus2-4*plus+6*zero-4*minus+minus2)/step**4)
    for order,values in enumerate(estimates,1):
        target=math.factorial(order)*r[order]
        errors=np.array([np.linalg.norm(value-target) for value in values])
        assert np.all(np.isfinite(errors)) and np.all(errors>0)
        x,y=np.log(steps),np.log(errors)
        p,intercept=np.linalg.lstsq(np.column_stack([x,np.ones(4)]),y,rcond=None)[0]
        quality=1-np.sum((y-p*x-intercept)**2)/np.sum((y-y.mean())**2)
        rich=np.linalg.norm((4*values[3]-values[2])/3-target)
        print("full Nambu dense order",order,"steps",steps.tolist(),"errors",errors.tolist(),"p",p,"R2",quality,"Richardson",rich)
        assert abs(p-2)<0.05 and quality>0.999 and rich<1e-7
    for key in ("particle_hole_defects","pairing_antisymmetry_defects","number_imaginary_defects",
                "metric_defects","density_hermiticity_defects","relative_commutator_defects","complex_embedding_defects"):
        assert np.max(direct[key])<1e-10
    print("full Nambu public raw",{key:direct[key] for key in ("particle_hole_defects","metric_defects","relative_commutator_defects","maximum_inverse_residual","exponential_tail_estimate")})


def contour(x,y,w,kt):
    ex,ey,ew=math.exp(x/kt),math.exp(y/kt),math.exp(w/kt)
    return ((ex-ey*ew)/(x-y-w)-(ey-ex*ew)/(x-y+w))/((1+ex)*(1+ey)*(ew-1))


def mass_factor(x, y, w, kt):
    """Independent nonresonant exponential mass expression for this fixture."""
    def f(x):
        return 1/(1+math.exp(x/kt))
    def divided(b, y):
        return (-f(b)*(1-f(b))/kt-(f(b)-f(y))/(b-y))/(b-y)
    n = 1/(math.exp(w/kt)-1)
    def term(y):
        return (n+f(x))*divided(x-w, y)+(n+f(-x))*divided(x+w, y)
    return (term(y)+term(-y))/math.tanh(x/(2*kt))


def inputs(model):
    gap,xi,kt,mass=0.02,0.007,0.004,0.2
    energy=math.hypot(gap,xi)
    q=math.tanh(energy/(2*kt))/(2*energy)
    kernel=-(1+mass)/q
    controls=dict(band_energies=[0.03],band_weights=[1.0],mass_renormalization=[mass],
                  electronic_temperature=kt/KB,electron_number=1-2*xi*q,
                  chemical_potential=0.025,initial_gap_real=[0.0144],initial_gap_imaginary=[0.0192],
                  response_order=4,band_energy_direction=[0.004],mass_renormalization_direction=[0.02],
                  temperature_direction=80.0,pairing_model=model,response_tolerance=1e-12)
    if model=="supplied":
        controls.update(pairing_kernel=[[kernel]],pairing_kernel_direction=[[-0.003]])
    else:
        w=0.025
        factor=2*(contour(xi,xi,w,kt)-contour(xi,-xi,w,kt))/math.tanh(xi/(2*kt))**2
        coupling=(kernel-0.005)/factor
        if model == "lm2005":
            zfactor = mass_factor(xi, xi, w, kt)
            coupling = -(1+0.005*q)/(zfactor+factor*q)
            assert coupling > 0
            controls.pop("mass_renormalization")
            controls.pop("mass_renormalization_direction")
        controls.update(phonon_energies=[w],pair_couplings_squared=[[coupling]],
                        screened_pair_coulomb=[[0.005]],pair_couplings_squared_direction=[[coupling*0.06]],
                        phonon_energy_direction=[0.001])
    return controls


@pytest.mark.parametrize("model",["supplied","lm2005-off-diagonal","lm2005"])
def test_superconducting_gap_native_ase_cli_and_independent_bdg(model):
    controls=inputs(model)
    direct=dft_rs.superconducting_state(**controls)
    generic=dft_rs.run_feature("superconducting-state",**controls)
    atomic=dft_rs.DftRs.get_superconducting_state(units="atomic",**controls)
    converted=dict(controls)
    energy_names=("band_energies","chemical_potential","initial_gap_real","initial_gap_imaginary",
                  "pairing_kernel","pairing_kernel_direction","band_energy_direction","phonon_energies",
                  "phonon_energy_direction","screened_pair_coulomb")
    for name in energy_names:
        if name in converted:converted[name]=np.asarray(converted[name])*Hartree
    for name in ("pair_couplings_squared","pair_couplings_squared_direction"):
        if name in converted:converted[name]=np.asarray(converted[name])*Hartree**2
    adapted=dft_rs.DftRs.get_superconducting_state(**converted)
    argv=[os.environ["DFT_RS_CLI"],"superconducting-state","--format","json"]
    for name,value in controls.items():
        argv.extend(["--"+name.replace("_","-"),value if isinstance(value,str) else json.dumps(value)])
    process=subprocess.run(argv,capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    assert "[scdft] iter" in process.stderr
    cli=json.loads(process.stdout)
    assert direct["converged"] and direct["response_order"]==4
    np.testing.assert_allclose(direct["gap_real"],[0.012],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(direct["gap_imaginary"],[0.016],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(direct["chemical_potential"],0.023,rtol=1e-10,atol=1e-12)
    gap=complex(direct["gap_real"][0],direct["gap_imaginary"][0])
    xi=0.03-direct["chemical_potential"]
    matrix=np.array([[xi,gap],[gap.conjugate(),-xi]])
    eigenvalues,vectors=np.linalg.eigh(matrix)
    density=(vectors/(1+np.exp(eigenvalues/(controls["electronic_temperature"]*KB))))@vectors.conj().T
    chi=-density[0,1]
    np.testing.assert_allclose(direct["occupation_coefficients"][0],[2*density[0,0].real],rtol=1e-12,atol=1e-13)
    np.testing.assert_allclose(direct["anomalous_density_coefficients_real"][0],[chi.real],rtol=1e-12,atol=1e-13)
    np.testing.assert_allclose(direct["anomalous_density_coefficients_imaginary"][0],[chi.imag],rtol=1e-12,atol=1e-13)
    keys=("gap_coefficients_real","gap_coefficients_imaginary","chemical_potential_coefficients",
          "anomalous_density_coefficients_real","anomalous_density_coefficients_imaginary","occupation_coefficients",
          "quasiparticle_energy_squared_coefficients","pairing_kernel_coefficients", "mass_renormalization_coefficients")
    for report in (generic,atomic,cli):
        assert report["converged"] and report["units"]==direct["units"]
        for key in keys:np.testing.assert_allclose(report[key],direct[key],rtol=1e-10,atol=1e-12)
    for key in keys:
        scale=Hartree**2 if key=="quasiparticle_energy_squared_coefficients" else (
            1.0 if key.startswith(("anomalous_","occupation_","mass_")) else Hartree)
        np.testing.assert_allclose(adapted[key],np.asarray(direct[key])*scale,rtol=1e-9,atol=1e-11)
    for key in ("gap_residuals","number_residuals","phase_residuals","phase_multipliers","goldstone_defects","linear_residuals","kernel_asymmetry"):
        assert max(abs(x) for x in direct[key])<1e-10
    assert not any(unit=="unspecified" for unit in direct["units"].values())
    if model == "lm2005":
        expected_mass = controls["pair_couplings_squared"][0][0]*mass_factor(xi, xi, 0.025, 0.004)
        np.testing.assert_allclose(direct["mass_renormalization"], [expected_mass], rtol=1e-12)
        assert expected_mass > 0 and "particle-hole-symmetric" in direct["mass_model"]
        kernel_inputs = {k: controls[k] for k in ("band_energies", "band_weights", "electronic_temperature",
                         "phonon_energies", "pair_couplings_squared", "screened_pair_coulomb")}
        component = dft_rs.scdft_pairing_kernel(**kernel_inputs, chemical_potential=direct["chemical_potential"])
        np.testing.assert_allclose(component["mass_renormalization"], direct["mass_renormalization"], rtol=1e-12)
        np.testing.assert_allclose(component["pairing_kernel"], direct["pairing_kernel_coefficients"][0], rtol=1e-12)
    print("SCDFT public raw",model,{key:direct[key] for key in ("gap_residuals","number_residuals","phase_residuals","phase_multipliers","goldstone_defects","linear_residuals")})


def test_superconducting_explicit_mass_reciprocity_and_normal_branch():
    controls=inputs("supplied")
    missing=dict(controls);missing.pop("mass_renormalization")
    with pytest.raises(Exception,match="mass_renormalization"):
        dft_rs.superconducting_state(**missing)
    normal=dft_rs.superconducting_state(**dict(controls,initial_gap_real=[0.0],initial_gap_imaginary=[0.0]))
    assert normal["converged"] and normal["warnings"]
    np.testing.assert_array_equal(normal["gap_coefficients_real"],np.zeros((5,1)))
    bad=dict(controls,band_energies=[0.01,0.02],band_weights=[0.5,0.5],mass_renormalization=[0.2,0.2],
             initial_gap_real=[0.02,0.02],initial_gap_imaginary=[0.0,0.0],
             pairing_kernel=[[-0.1,-0.02],[-0.03,-0.1]],band_energy_direction=[0.0,0.0],
             mass_renormalization_direction=[0.0,0.0],pairing_kernel_direction=[[0.0,0.0],[0.0,0.0]])
    with pytest.raises(Exception,match="reciprocal"):
        dft_rs.superconducting_state(**bad)
    with pytest.raises(Exception, match="generates its own"):
        dft_rs.superconducting_state(**dict(inputs("lm2005"), mass_renormalization=[0.0]))
