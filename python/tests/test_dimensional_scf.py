# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Actual dimensional SCF, compensated energy and public-route consistency."""
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, dimensional_scf, periodic_coulomb_operators, periodic_xc_operators

CELL=np.diag([7.0,7.0,7.0])
POSITION=np.array([[.2,-.1,.3]])
ENERGIES=("energy","free_energy","entropy_term","source_energy","electronic_energy",
    "kinetic_energy","electron_nuclear_energy","hartree_energy","full_exchange_energy",
    "long_range_exchange_energy","semilocal_energy")
DIAGNOSTICS=("orbital_residual","eigen_residual","orthonormality_defect","projector_defect",
    "particle_number_error","occupation_consistency","component_energy_defect","final_fock_change",
    "coulomb_operator_defect","coulomb_derivative_defect","coulomb_source_defect","grid_number_defect")
def arrays(r,key):
    return np.asarray(r[key+"_real"])+1j*np.asarray(r[key+"_imag"])

@pytest.mark.parametrize("dimension",[1,2,3])
@pytest.mark.parametrize("charge",[-1,0,1])
def test_actual_native_ase_cli_signed_dimensional_scf(dimension,charge,tmp_path):
    pbc=[i<dimension for i in range(3)];multiplicity=2 if charge==0 else 1
    options=dict(pbc=pbc,k_mesh=[3 if axis else 1 for axis in pbc],xc="hf",charge=charge,
        multiplicity=multiplicity,basis_cutoff=4.3,compensation="gaussian",
        countercharge_center=[-.5,.7,.9],countercharge_width=1.1,alpha=.55,
        real_cutoff=16.,reciprocal_cutoff=8.,conv_tol_grad=1e-9)
    r=dimensional_scf([1],POSITION,CELL,**options)
    assert r["converged"] and r["scf_converged"]
    assert r["dimension"]==dimension and r["formal_charge"]==charge
    assert r["n_alpha"]+r["n_beta"]==1-charge
    assert r["compensating_charge"]==-charge
    assert r["particle_number_error"]<1e-12 and r["occupation_consistency"]<1e-12
    assert r["orbital_residual"]<1e-12 and r["component_energy_defect"]<1e-12
    np.testing.assert_allclose(sum(r[key] for key in ("source_energy","kinetic_energy",
        "electron_nuclear_energy","hartree_energy","full_exchange_energy",
        "long_range_exchange_energy","semilocal_energy")),r["energy"],atol=1e-13,rtol=0)
    atoms=Atoms("H",positions=POSITION*Bohr,cell=CELL*Bohr,pbc=pbc)
    adapted=DftRs(units="atomic").get_dimensional_scf(atoms,**{k:v for k,v in options.items() if k!="pbc"})
    for key in ENERGIES+DIAGNOSTICS:
        assert abs(r[key]-adapted[key])<1e-11
    for key in ("density","fock","energy_weighted_density","overlap","h_core"):
        np.testing.assert_allclose(arrays(r,key),arrays(adapted,key),atol=1e-12,rtol=0)
    xyz=tmp_path/"dimensional.xyz"
    xyz.write_text("1\ncompensated dimensional SCF\nH "+" ".join(f"{v*.529177210903:.17g}" for v in POSITION[0])+"\n")
    flags=["--cell",";".join(",".join(f"{v*.529177210903:.17g}" for v in row) for row in CELL),
        "--pbc",",".join(str(int(v)) for v in pbc),"--k-mesh",",".join(map(str,options["k_mesh"])),
        "--charge",str(charge),"--multiplicity",str(multiplicity),"--xc","hf","--overlap-cutoff","4.3",
        "--charge-compensation","gaussian","--countercharge-center","-.5,.7,.9","--countercharge-width","1.1",
        "--ewald-alpha",".55","--ewald-real-cutoff","16","--ewald-reciprocal-cutoff","8",
        "--conv-tol-grad","1e-9","--log-mode","none"]
    process=subprocess.run([os.environ["DFT_RS_CLI"],"dimensional-scf",str(xyz),*flags],capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    rows=[line.split() for line in process.stdout.splitlines() if line]
    for key in ENERGIES+DIAGNOSTICS:
        value=next(float(row[1]) for row in rows if row[0]==key)
        assert abs(value-r[key])<1e-10
    for key,rank in (("density",4),("fock",4),("energy_weighted_density",4),("overlap",3),("h_core",3)):
        values=[row for row in rows if row[0]==key]
        assert len(values)==arrays(r,key).size
        for row in values:
            index=tuple(map(int,row[1:rank+1]));value=complex(float(row[rank+1]),float(row[rank+2]))
            assert abs(value-arrays(r,key)[index])<1e-11
    assert r["model"] in process.stdout and r["units"] in process.stdout

@pytest.mark.parametrize("name",["hf","svwn5","pbe","pbe0","cam-b3lyp","r2scan"])
def test_self_consistent_components_independently_reassemble_fock(name,tmp_path):
    pbc=[True,False,False];pos=[[0.,0.,0.],[1.4,.3,.2]]
    r=dimensional_scf([1,1],pos,CELL,pbc=pbc,k_mesh=[3,1,1],xc=name,
        basis_cutoff=4.3,grid_level=4,conv_tol_grad=1e-9)
    assert r["scf_converged"] and r["particle_number_error"]<1e-11
    assert r["orbital_residual"]<1e-9 and r["occupation_consistency"]<1e-8
    p=arrays(r,"density");s=arrays(r,"overlap");f=arrays(r,"fock")
    assert np.max(abs(f@p@s-s@p@f))<1e-8
    j=periodic_coulomb_operators([1,1],pos,CELL,pbc=pbc,k_mesh=[3,1,1],basis_cutoff=4.3,
        density_real=p.sum(axis=0).real.tolist(),density_imag=p.sum(axis=0).imag.tolist())
    assert abs(j["source_energy"]-r["source_energy"])<1e-11
    assert abs(sum(j["electron_nuclear_components"])-r["electron_nuclear_energy"])<1e-11
    assert abs(sum(j["hartree_energy_components"])-r["hartree_energy"])<1e-11
    if name!="hf":
        x=periodic_xc_operators([1,1],pos,CELL,pbc=pbc,k_mesh=[3,1,1],xc=name,
            basis_cutoff=4.3,grid_level=4,density_real=p.real.tolist(),density_imag=p.imag.tolist())
        assert abs(x["energy"]-r["semilocal_energy"])<1e-12
        np.testing.assert_allclose(x["component_energies"],r["semilocal_component_energies"],atol=1e-12,rtol=0)
        if name in ("svwn5","pbe","r2scan"):
            np.testing.assert_allclose(f,arrays(r,"h_core")[None]+arrays(j,"hartree")[None]+arrays(x,"potential"),atol=1e-11,rtol=0)
    xyz=tmp_path/"functional-scf.xyz"
    xyz.write_text("2\nfunctional SCF route\n"+"".join("H "+" ".join(
        f"{v*.529177210903:.17g}" for v in row)+"\n" for row in pos))
    process=subprocess.run([os.environ["DFT_RS_CLI"],"dimensional-scf",str(xyz),"--cell","3.704240476321",
        "--pbc","1,0,0","--k-mesh","3,1,1","--xc",name,"--overlap-cutoff","4.3",
        "--grid-level","4","--conv-tol-grad","1e-9","--log-mode","none"],capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    rows=[line.split() for line in process.stdout.splitlines() if line]
    for key in ENERGIES+DIAGNOSTICS:
        value=next(float(row[1]) for row in rows if row[0]==key)
        assert abs(value-r[key])<1e-10

def test_dimensional_thermal_metadata_and_control_validation(tmp_path):
    pos=[[0.,0.,0.],[1.4,.3,.2]]
    r=dimensional_scf([1,1],pos,CELL,pbc=[True,False,False],xc="hf",temperature=200000,
        basis_cutoff=4.3,conv_tol_grad=1e-9)
    assert r["converged"] and r["entropy_term"]>0
    assert r["particle_number_error"]<1e-11
    assert abs(r["energy"]-r["entropy_term"]-r["free_energy"])<1e-13
    assert r["occupation_consistency"]<1e-8
    atoms=Atoms("HH",positions=np.asarray(pos)*Bohr,cell=CELL*Bohr,pbc=[True,False,False])
    adapted=DftRs(units="atomic", xc="hf",electronic_temperature=200000,k_mesh=(1,1,1)).get_dimensional_scf(
        atoms,basis_cutoff=4.3,conv_tol_grad=1e-9)
    assert adapted["temperature"]==200000
    assert abs(adapted["free_energy"]-r["free_energy"])<1e-11
    for configured in ({"ecp":"configured-ecp"},{"density_fit":"configured-auxiliary"},
                       {"stabilize":True}):
        with pytest.raises(NotImplementedError):
            DftRs(units="atomic", **configured).get_dimensional_scf(atoms)
    with pytest.raises(ValueError):
        DftRs(units="atomic", grid_shell=3).get_dimensional_scf(atoms)
    for charge in (-1,1):
        compensated=dimensional_scf([1],POSITION,CELL,xc="hf",charge=charge,
            basis_cutoff=4.3,compensation="uniform")
        assert compensated["converged"] and compensated["compensating_charge"]==-charge
        assert compensated["n_alpha"]+compensated["n_beta"]==1-charge
        assert compensated["particle_number_error"]<1e-12
    with pytest.raises((ValueError,NotImplementedError)):
        dimensional_scf([1],POSITION,CELL,pbc=[True,False,False],xc="hf",charge=1,
            compensation="uniform")
    for option in ({"temperature":-1},{"grid_level":10},{"max_iter":0},{"basis_cutoff":0},
        {"pbc":[False,False,False]},{"charge":1},{"k_mesh":[1,2,1]}):
        controls=dict(pbc=[True,False,False],**{k:v for k,v in option.items() if k!="pbc"})
        if "pbc" in option:
            controls["pbc"]=option["pbc"]
        with pytest.raises((ValueError,RuntimeError)):
            dimensional_scf([1,1],pos,CELL,**controls)
    with pytest.raises(NotImplementedError):
        dimensional_scf([1,1],pos,CELL,xc="wb97x-v")
    executable=os.environ["DFT_RS_CLI"]
    for flags in (["--forces"],["--field-wave",".1,0,0"],["--grid-shell","3"],["--alpha-density-matrices","missing.csv"]):
        p=subprocess.run([executable,"dimensional-scf",str(tmp_path/"missing.xyz"),*flags],capture_output=True,text=True)
        assert p.returncode!=0 and "No such file" not in p.stderr
