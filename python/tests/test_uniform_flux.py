# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Uniform-flux continuum references on every common-driver surface."""
import json
from itertools import product
import math
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
import dft_rs


def single_gaussian_torus_oracle(cell, field, exponent):
    """Closed Gaussian moments, independent of production AO integration.

    At the origin each translated pair has phase wave -B cross T/2,
    mean T/2+i wave/(4a) and covariance I/(4a). The finite-cell kinetic
    expectation is the translated quadratic-form sum divided by overlap.
    """
    flux = [int(round(np.dot(field, np.cross(cell[i], cell[j]))/(2*np.pi)))
            for i,j in ((0,1),(1,2),(2,0))]
    overlap, kinetic = 0.0, 0.0
    for n in product(range(-2,3), repeat=3):
        translation = np.asarray(n)@cell
        wave = -.5*np.cross(field, translation)
        parity = flux[0]*n[0]*n[1]+flux[1]*n[1]*n[2]-flux[2]*n[0]*n[2]
        pair = (-1)**parity*np.exp(-.5*exponent*np.dot(translation,translation)
                                  -np.dot(wave,wave)/(8*exponent))
        moment = (1.5*exponent-.5*exponent**2*np.dot(translation,translation)
                  +np.dot(wave,wave)/4+np.dot(field,field)/(16*exponent)
                  -np.dot(field,field)*np.dot(wave,wave)/(128*exponent**2))
        overlap += pair
        kinetic += pair*moment
    return overlap, kinetic/overlap


def reciprocal_modes(cell, cutoff):
    bounds=np.ceil(cutoff*np.linalg.norm(cell,axis=1)/(2*np.pi)).astype(int)
    indices=np.asarray(list(product(*(range(-n,n+1) for n in bounds))))
    waves=indices@(2*np.pi*np.linalg.inv(cell).T)
    norm2=np.einsum("ij,ij->i",waves,waves)
    mask=(norm2>0)&(norm2<=cutoff**2)
    return indices[mask],waves[mask],norm2[mask]


def gaussian_interaction_oracle(cell, field, exponent, width, omega):
    """Closed translated Gaussian density products and discrete Coulomb sums."""
    indices,waves,norm2=reciprocal_modes(cell,16.0)
    flux=[int(round(np.dot(field,np.cross(cell[i],cell[j]))/(2*np.pi)))
          for i,j in ((0,1),(1,2),(2,0))]
    fourier=np.zeros(len(waves))
    for n in product(range(-2,3),repeat=3):
        translation=np.asarray(n)@cell
        wave=waves-.5*np.cross(field,translation)
        cocycle=(-1)**(flux[0]*n[0]*n[1]+flux[1]*n[1]*n[2]-flux[2]*n[0]*n[2])
        sign=np.where((indices@np.asarray(n))%2==0,1.,-1.)
        fourier+=cocycle*sign*np.exp(-.5*exponent*np.dot(translation,translation)
                                    -np.einsum("ij,ij->i",wave,wave)/(8*exponent))
    overlap,_=single_gaussian_torus_oracle(cell,field,exponent)
    fourier/=overlap
    coulomb=4*np.pi/(abs(np.linalg.det(cell))*norm2)
    full=np.sum(coulomb*fourier**2)
    long_range=0.0 if omega==0 else np.sum(coulomb*fourier**2*np.exp(-norm2/(4*omega**2)))
    nuclear=-np.sum(coulomb*fourier*np.exp(-norm2*width**2/4))
    return full,long_range,nuclear


def source_green_oracle(cell, displacement, width):
    """Independent scalar Ewald sum; no analytic gradient code is reused."""
    alpha=.55
    beta=1/np.sqrt(width**2+1/alpha**2)
    real=0.0
    for n in product(range(-3,4),repeat=3):
        distance=np.linalg.norm(displacement+np.asarray(n)@cell)
        if distance==0:
            assert width>0
            real+=2*(1/width-beta)/np.sqrt(np.pi)
        else:
            first=1.0 if width==0 else math.erf(distance/width)
            real+=(first-math.erf(beta*distance))/distance
    _,waves,norm2=reciprocal_modes(cell,8.0)
    reciprocal=np.sum(4*np.pi/(abs(np.linalg.det(cell))*norm2)
                      *np.exp(-norm2*(width**2+1/alpha**2)/4)*np.cos(waves@displacement))
    return real+reciprocal-np.pi/(alpha**2*abs(np.linalg.det(cell)))


def source_self_oracle(cell,width):
    if width:
        return .5*source_green_oracle(cell,np.zeros(3),np.sqrt(2)*width)
    # Bare point self excluded. Separate real/reciprocal sums and Ewald
    # self/background are evaluated directly, rather than calling the core.
    alpha=.55
    real=0.0
    for n in product(range(-3,4),repeat=3):
        if n==(0,0,0): continue
        distance=np.linalg.norm(np.asarray(n)@cell)
        real+=math.erfc(alpha*distance)/distance
    _,_,norm2=reciprocal_modes(cell,8.0)
    volume=abs(np.linalg.det(cell))
    reciprocal=np.sum(4*np.pi/(volume*norm2)*np.exp(-norm2/(4*alpha**2)))
    return .5*(real+reciprocal-2*alpha/np.sqrt(np.pi)-np.pi/(alpha**2*volume))


def source_gradient_acceptance(cell,points,width,analytic):
    steps=np.asarray([.004,.002,.001,.0005])
    errors=[]
    estimates=[]
    for h in steps:
        estimate=np.zeros_like(analytic)
        for g,r in enumerate(points):
            for axis in range(3):
                displacement=np.eye(3)[axis]*h
                estimate[g,axis]=(-source_green_oracle(cell,r+displacement,width)
                                  +source_green_oracle(cell,r-displacement,width))/(2*h)
        estimates.append(estimate)
        errors.append(np.max(np.abs(estimate-analytic)))
    assert np.all(np.isfinite(errors)) and np.all(np.asarray(errors)>0)
    fit=np.polyfit(np.log(steps),np.log(errors),1)
    residual=np.log(errors)-np.polyval(fit,np.log(steps))
    quality=1-np.dot(residual,residual)/np.sum((np.log(errors)-np.mean(np.log(errors)))**2)
    rich=np.max(np.abs((4*estimates[3]-estimates[2])/3-analytic))
    print("independent magnetic source gradient fit",width,"steps",steps.tolist(),
          "errors",errors,"exponent",fit[0],"R2",quality,"Richardson",rich)
    assert abs(fit[0]-2)<.02 and quality>.99999
    assert rich<1e-8


def magnetic_bloch_oracle(cell,field,mesh,omega,eta,width):
    """Closed translated Gaussian transfers and independent BZ singular limit."""
    reciprocal=2*np.pi*np.linalg.inv(cell).T
    ks=np.asarray(list(product(*(np.arange(n)/n for n in mesh))))@reciprocal
    flux=[int(round(np.dot(field,np.cross(cell[i],cell[j]))/(2*np.pi)))
          for i,j in ((0,1),(1,2),(2,0))]
    def vertex(right,waves):
        total=np.zeros(len(waves),dtype=complex)
        for n in product(range(-2,3),repeat=3):
            t=np.asarray(n)@cell
            w=waves-.5*np.cross(field,t)
            parity=flux[0]*n[0]*n[1]+flux[1]*n[1]*n[2]-flux[2]*n[0]*n[2]
            total+=(-1)**parity*np.exp(1j*(np.dot(right,t)+waves@t/2)
                     -.35*np.dot(t,t)-np.einsum("ij,ij->i",w,w)/5.6)
        return total
    overlap=np.asarray([vertex(k,np.zeros((1,3)))[0].real for k in ks])
    _,waves,norm2=reciprocal_modes(cell,16.)
    volume=abs(np.linalg.det(cell))
    kernel=4*np.pi/(volume*norm2)
    rho=np.mean([vertex(k,waves)/s for k,s in zip(ks,overlap)],axis=0)
    hartree=.5*np.sum(kernel*np.abs(rho)**2)
    nuclear=-np.sum(kernel*rho.real*np.exp(-norm2*width**2/4))
    exchange=0.0
    long_range=0.0
    for i,left in enumerate(ks):
        for j,right in enumerate(ks):
            shift=left-right
            _,g,_=reciprocal_modes(cell,16.+np.linalg.norm(shift))
            q=np.vstack((g,np.zeros((1,3))))+shift
            q2=np.einsum("ij,ij->i",q,q)
            mask=(q2>0)&(q2<=16.**2)
            q,q2=q[mask],q2[mask]
            term=4*np.pi/(volume*q2)*np.abs(vertex(right,q))**2/(overlap[i]*overlap[j]*len(ks)**2)
            exchange-=.5*np.sum(term)
            if omega:
                long_range-=.5*np.sum(term*np.exp(-q2/(4*omega**2)))
    enlarged=cell*np.asarray(mesh)[:,None]
    _,_,q2=reciprocal_modes(enlarged,16.)
    def correction(width):
        if width==0: return 0.0
        return 2*width/np.sqrt(np.pi)-np.sum(4*np.pi/(abs(np.linalg.det(enlarged))*q2)
                                            *np.exp(-q2/(4*width**2)))
    effective=0.0 if eta==0 or omega==0 else eta*omega/np.sqrt(eta**2+omega**2)
    corrections=np.asarray([correction(eta),correction(effective)])
    return hartree,exchange-.5*corrections[0],long_range-.5*corrections[1],nuclear,corrections


@pytest.mark.parametrize("xc,charge,exchange",[("hf",0,None),("hf",-1,[.4,.3,.5]),("svwn5",0,None)])
def test_magnetic_crystal_bz_on_all_surfaces(tmp_path,xc,charge,exchange):
    basis=tmp_path/"magnetic-bloch.g94"
    basis.write_text("****\nH 0\nS 1 1.00\n0.7 1.0\n****\n",encoding="utf-8")
    cell=np.eye(3)*8
    field=np.asarray([1.,1.,-1.])*(2*np.pi/64)
    settings=dict(xc=xc,basis=str(basis),charge=charge,magnetic_model="crystal_bz",
                  k_mesh=[2,1,1],magnetic_field=field.tolist(),nuclear_widths=[.3],
                  magnetic_reciprocal_cutoff=16.,magnetic_grid_order=40,
                  magnetic_exchange_auxiliary_width=.5,overlap_cutoff=12.,
                  current_points=[.2,.3,-.1],ewald_alpha=.55,ewald_real_cutoff=16.,
                  ewald_reciprocal_cutoff=8.,conv_tol=1e-13,conv_tol_dp=1e-12,
                  conv_tol_grad=1e-11)
    if charge: settings["charge_compensation"]="uniform"
    if exchange is not None: settings["exchange_coefficients"]=exchange
    native=dft_rs.periodic_magnetic_scf([1],[[0,0,0]],cell,**settings)
    generic=dft_rs.run_feature("periodic-magnetic-scf",[1],[[0,0,0]],cell=cell,**settings)
    atoms=Atoms("H",positions=[[0,0,0]],cell=cell*Bohr,pbc=True)
    ase=dft_rs.DftRs(units="atomic").get_periodic_magnetic_scf(atoms,**settings, units="atomic")
    xyz=tmp_path/"magnetic-bloch.xyz"
    xyz.write_text("1\nmagnetic crystal\nH 0 0 0\n",encoding="utf-8")
    args=[os.environ["DFT_RS_CLI"],"periodic-magnetic-scf",str(xyz),"--cell",
          ",".join(f"{x*.529177210903:.17g}" for x in cell.flatten()),"--format","json"]
    for key,value in settings.items():
        if isinstance(value,list): value=",".join(map(str,value))
        args.extend(["--"+key.replace("_","-"),str(value)])
    process=subprocess.run(args,capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    cli=json.loads(process.stdout)
    for report in (native,generic,ase,cli):
        assert report["converged"] and report["backend_converged"]
        assert report["magnetic_model"]=="crystal_bz" and report["k_mesh"]==[2,1,1]
        assert len(report["density_real"])==2
        assert report["units"]==native["units"] and report["model"]==native["model"]
        assert report["orbital_residual"]<1e-11 and report["particle_number_error"]<1e-12
        for key in ("energy","free_energy","density_real","density_imag","fock_real","fock_imag",
                    "k_points","exchange_singularity_correction","magnetization",
                    "number_density","spin_density","pauli_particle_current_density"):
            np.testing.assert_allclose(report[key],native[key],rtol=1e-9,atol=1e-11)
        for term in native["energy_components"]:
            np.testing.assert_allclose(report["energy_components"][term],native["energy_components"][term],rtol=1e-9,atol=1e-11)
    cx,clr,omega=native["exchange_coefficients"]
    j,k,lr,v,correction=magnetic_bloch_oracle(cell,field,[2,1,1],omega,.5,.3)
    electrons=1-charge
    expected=dict(hartree=electrons**2*j,full_exchange=cx*electrons*k,
                  long_range_exchange=clr*electrons*lr,electron_nuclear=electrons*v)
    for term,oracle in expected.items():
        print("magnetic crystal independent term",xc,charge,term,native["energy_components"][term]-oracle)
        np.testing.assert_allclose(native["energy_components"][term],oracle,rtol=0,atol=1e-11)
    if cx or clr:
        np.testing.assert_allclose(native["exchange_singularity_correction"],correction,rtol=0,atol=1e-12)
        assert native["exchange_transfer_modes"]>0
    assert native["xc_grid_number_defect"]<=1e-10
    print("magnetic crystal raw residuals",xc,charge,native["orbital_residual"],native["particle_number_error"],
          native["coulomb_operator_defect"],native["exchange_singularity_correction"])


@pytest.mark.parametrize("width,charge,exchange", [(0.0,0,None),(.3,0,None),(0.0,-1,None),
                                                   (0.0,-1,[.4,.3,.5]),(0.0,-1,[.4,.3,0.0])])
def test_uniform_flux_native_ase_cli_and_term_oracles(tmp_path, width, charge, exchange):
    # An independently specified single normalized Gaussian makes kinetic,
    # spin, self-interaction and nuclear-contact checks transparent.
    basis = tmp_path / "magnetic-oracle.g94"
    basis.write_text("****\nH 0\nS 1 1.00\n0.7 1.0\n****\n", encoding="utf-8")
    cell = np.eye(3)*8.0
    field = np.array([1.0, 1.0, -1.0])*(2*np.pi/64.0)
    settings = dict(xc="hf", basis=str(basis), charge=charge,
                    k_mesh=[1,1,1], overlap_cutoff=12.0,
                    magnetic_field=field.tolist(), magnetic_origin=[1.0,-2.0,.5],
                    current_points=[.2,.3,-.1,-.25,.1,.2],
                    magnetic_reciprocal_cutoff=16.0, nuclear_widths=[width],
                    ewald_alpha=.55, ewald_real_cutoff=16.0,
                    ewald_reciprocal_cutoff=8.0,
                    conv_tol=1e-13, conv_tol_dp=1e-12,
                    conv_tol_grad=1e-11, max_iter=150)
    if charge:
        settings["charge_compensation"]="uniform"
    if exchange is not None:
        settings["exchange_coefficients"]=exchange
    native = dft_rs.periodic_magnetic_scf([1], [[0,0,0]], cell, **settings)
    generic = dft_rs.run_feature("periodic-magnetic-scf", [1], [[0,0,0]], cell=cell, **settings)
    atoms = Atoms("H", positions=[[0,0,0]], cell=cell*Bohr, pbc=True)
    ase = dft_rs.DftRs(units="atomic").get_periodic_magnetic_scf(atoms, **settings, units="atomic")
    xyz = tmp_path / "magnetic.xyz"
    xyz.write_text("1\nmagnetic reference\nH 0 0 0\n", encoding="utf-8")
    # Flat nine-component cells must agree with native row matrices.
    args = [os.environ["DFT_RS_CLI"], "periodic-magnetic-scf", str(xyz),
            "--cell", ",".join(f"{v*.529177210903:.17g}" for v in cell.flatten()),
            "--format", "json"]
    for key,value in settings.items():
        if isinstance(value,list): value=",".join(map(str,value))
        args.extend(["--"+key.replace("_","-"),str(value)])
    process = subprocess.run(args, capture_output=True, text=True)
    assert process.returncode==0, process.stderr
    cli = json.loads(process.stdout)
    for report in (native,generic,ase,cli):
        assert report["converged"] and report["backend_converged"]
        expected_flux = [int(round(np.dot(field, np.cross(cell[i], cell[j]))/(2*np.pi)))
                         for i,j in ((0,1),(1,2),(2,0))]
        assert report["flux_quanta"]==expected_flux
        assert report["flux_defect"]<1e-12
        assert report["orbital_residual"]<1e-11
        assert report["particle_number_error"]<1e-12
        assert report["occupation_consistency"]<1e-12
        assert report["coulomb_operator_defect"]<=1e-10
        assert report["image_operator_defect"]<=1e-10
        assert report["units"]==native["units"] and report["model"]==native["model"]
        for key in ("energy", "free_energy", "nuclear_soc_energy", "energy_components",
                    "density_real", "density_imag", "fock_real", "fock_imag", "magnetization",
                    "number_density", "spin_density", "source_potential_gradient",
                    "kinetic_particle_current_density", "spin_magnetization_particle_current_density",
                    "nuclear_soc_particle_current_density", "pauli_particle_current_density",
                    "pauli_electric_current_density", "spinor_energy_derivative"):
            if isinstance(native[key],dict):
                for term in native[key]:
                    np.testing.assert_allclose(report[key][term],native[key][term],rtol=1e-9,atol=1e-11)
            else:
                np.testing.assert_allclose(report[key],native[key],rtol=1e-9,atol=1e-11)
        np.testing.assert_allclose(report["energy"],sum(report["energy_components"].values()),rtol=0,atol=1e-13)
        assert report["external_field_breaks_time_reversal"]
        kinetic=np.asarray(report["kinetic_particle_current_density"])
        spin=np.asarray(report["spin_magnetization_particle_current_density"])
        soc=np.asarray(report["nuclear_soc_particle_current_density"])
        np.testing.assert_allclose(report["pauli_particle_current_density"],kinetic+spin+soc,rtol=0,atol=1e-13)
        np.testing.assert_allclose(report["pauli_electric_current_density"],-(kinetic+spin+soc),rtol=0,atol=1e-13)
    full,long_range,nuclear=gaussian_interaction_oracle(cell,field,.7,width,native["exchange_coefficients"][2])
    cx,clr,_=native["exchange_coefficients"]
    electrons=1-charge
    oracles=dict(source=source_self_oracle(cell,width),electron_nuclear=electrons*nuclear,
                 hartree=.5*electrons**2*full,full_exchange=-.5*cx*electrons*full,
                 long_range_exchange=-.5*clr*electrons*long_range)
    for term,expected in oracles.items():
        defect=native["energy_components"][term]-expected
        print("independent magnetic term oracle",width,charge,exchange,term,defect)
        assert abs(defect)<1e-11,(term,defect)
    overlap,kinetic=single_gaussian_torus_oracle(cell,field,.7)
    points=np.asarray(native["current_points"])
    rho=electrons*(1.4/np.pi)**1.5*np.exp(-1.4*np.einsum("ij,ij->i",points,points))/overlap
    np.testing.assert_allclose(native["number_density"],rho,rtol=0,atol=1e-12)
    expected_current=.5*np.cross(field,points)*rho[:,None]
    np.testing.assert_allclose(native["kinetic_particle_current_density"],expected_current,rtol=0,atol=1e-12)
    source_gradient_acceptance(cell,points,width,np.asarray(native["source_potential_gradient"]))
    if charge==0:
        np.testing.assert_allclose(native["magnetization"],-field/np.linalg.norm(field),rtol=0,atol=1e-11)
        np.testing.assert_allclose(native["energy_components"]["hartree"]+native["energy_components"]["full_exchange"],0,rtol=0,atol=1e-12)
        expected_overlap,expected_kinetic=single_gaussian_torus_oracle(cell,field,.7)
        np.testing.assert_allclose(native["energy_components"]["kinetic"],expected_kinetic,rtol=0,atol=1e-8)
        np.testing.assert_allclose(np.trace(np.asarray(native["density_real"][0])),
                                   1/expected_overlap,rtol=0,atol=1e-12)
        print("finite-cell kinetic versus isolated Gaussian",
              native["energy_components"]["kinetic"]-(1.5*.7+np.dot(field,field)/(16*.7)),
              "closed torus oracle defect",native["energy_components"]["kinetic"]-expected_kinetic)
        radial=(2*np.sqrt(1.4/np.pi)/(1+1.4*width*width)**1.5-np.pi/(.7*512))/(12*137.035999084**2)
        np.testing.assert_allclose(native["nuclear_soc_energy"],-np.linalg.norm(field)*radial,rtol=0,atol=1e-11)
        assert abs(native["density_imag"][0][0][1])>.1
    else:
        np.testing.assert_allclose(native["magnetization"],[0,0,0],rtol=0,atol=1e-11)
        assert native["energy_components"]["hartree"]>0
        assert native["spin_coefficient_conjugation_defect"]<1e-12
        assert np.max(np.abs(native["kinetic_particle_current_density"]))>1e-3
    print("uniform flux raw residuals", width, charge, native["orbital_residual"],
          native["particle_number_error"], native["coulomb_operator_defect"], native["image_operator_defect"])


def test_noninteger_flux_and_uncompensated_charge_are_refused(tmp_path):
    basis = tmp_path/"magnetic-oracle.g94"
    basis.write_text("****\nH 0\nS 1 1.00\n0.7 1.0\n****\n", encoding="utf-8")
    common=dict(xc="hf",basis=str(basis))
    with pytest.raises(ValueError,match="flux must obey"):
        dft_rs.periodic_magnetic_scf([1],[[0,0,0]],np.eye(3)*8,
                                   magnetic_field=[0,0,np.pi/64],**common)
    with pytest.raises(ValueError,match="explicit compensation"):
        dft_rs.periodic_magnetic_scf([1],[[0,0,0]],np.eye(3)*8,
                                   magnetic_field=[0,0,2*np.pi/64],charge=-1,**common)


@pytest.mark.parametrize("magnetic_model",["finite_gamma","crystal_bz"])
def test_fixed_flux_additional_operator_derivative_on_all_surfaces(tmp_path,magnetic_model):
    basis=tmp_path/"magnetic-oracle.g94"
    basis.write_text("****\nH 0\nS 1 1.00\n0.7 1.0\n****\n",encoding="utf-8")
    cell=np.eye(3)*8
    field=np.asarray([1.,1.,-1.])*(2*np.pi/64)
    overlap,_=single_gaussian_torus_oracle(cell,field,.7)
    v=np.asarray([.3,.1,-.2])
    generator=.5*overlap*np.asarray([[v[2],v[0]-1j*v[1]],[v[0]+1j*v[1],-v[2]]])
    settings=dict(xc="hf",basis=str(basis),magnetic_field=field.tolist(),nuclear_widths=[.3],
                  magnetic_reciprocal_cutoff=16.,overlap_cutoff=12.,ewald_alpha=.55,
                  ewald_real_cutoff=16.,ewald_reciprocal_cutoff=8.,conv_tol=1e-13,
                  conv_tol_dp=1e-12,conv_tol_grad=1e-11,spinor_amplitude=.6,
                  spinor_operator_real=generator.real.flatten().tolist(),
                  spinor_operator_imag=generator.imag.flatten().tolist())
    settings["magnetic_model"]=magnetic_model
    settings["k_mesh"]=[1,1,1] if magnetic_model=="finite_gamma" else [2,1,1]
    native=dft_rs.periodic_magnetic_scf([1],[[0,0,0]],cell,**settings)
    generic=dft_rs.run_feature("periodic-magnetic-scf",[1],[[0,0,0]],cell=cell,**settings)
    atoms=Atoms("H",positions=[[0,0,0]],cell=cell*Bohr,pbc=True)
    ase=dft_rs.DftRs(units="atomic").get_periodic_magnetic_scf(atoms,**settings, units="atomic")
    xyz=tmp_path/"magnetic.xyz"
    xyz.write_text("1\nfixed flux spin perturbation\nH 0 0 0\n",encoding="utf-8")
    args=[os.environ["DFT_RS_CLI"],"periodic-magnetic-scf",str(xyz),"--cell",
          ",".join(f"{x*.529177210903:.17g}" for x in cell.flatten()),"--format","json"]
    for key,value in settings.items():
        if isinstance(value,list): value=",".join(map(str,value))
        args.extend(["--"+key.replace("_","-"),str(value)])
    process=subprocess.run(args,capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    cli=json.loads(process.stdout)
    radial=(2*np.sqrt(1.4/np.pi)/(1+1.4*.09)**1.5-np.pi/(.7*512))/(12*137.035999084**2)
    total=field*(1+2*radial)+.6*v
    closed=-.5*np.dot(total,v)/np.linalg.norm(total)
    for report in (native,generic,ase,cli):
        assert report["converged"] and report["backend_converged"]
        if magnetic_model=="finite_gamma":
            np.testing.assert_allclose(report["spinor_energy_derivative"],closed,rtol=0,atol=1e-11)
            np.testing.assert_allclose(report["magnetization"],-total/np.linalg.norm(total),rtol=0,atol=1e-10)
        np.testing.assert_allclose(report["spinor_energy_derivative"],native["spinor_energy_derivative"],rtol=0,atol=1e-11)
        np.testing.assert_allclose(report["magnetization"],native["magnetization"],rtol=0,atol=1e-10)
        assert report["units"]==native["units"]
    steps=np.asarray([.02,.01,.005,.0025])
    errors=[]
    values=[]
    for h in steps:
        plus=dft_rs.periodic_magnetic_scf([1],[[0,0,0]],cell,**(settings|dict(spinor_amplitude=.6+h)))
        minus=dft_rs.periodic_magnetic_scf([1],[[0,0,0]],cell,**(settings|dict(spinor_amplitude=.6-h)))
        assert plus["converged"] and minus["converged"]
        derivative=(plus["energy"]-minus["energy"])/(2*h)
        values.append(derivative)
        errors.append(abs(derivative-native["spinor_energy_derivative"]))
    assert np.all(np.asarray(errors)>0) and np.all(np.isfinite(errors))
    fit=np.polyfit(np.log(steps),np.log(errors),1)
    residual=np.log(errors)-np.polyval(fit,np.log(steps))
    quality=1-np.dot(residual,residual)/np.sum((np.log(errors)-np.mean(np.log(errors)))**2)
    rich=abs((4*values[3]-values[2])/3-native["spinor_energy_derivative"])
    print("public fixed-flux amplitude fit",magnetic_model,steps.tolist(),errors,"p",fit[0],"R2",quality,"Richardson",rich)
    assert abs(fit[0]-2)<.02 and quality>.99999 and rich<1e-8
