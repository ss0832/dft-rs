# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Actual dimensional electric/magnetic response on Rust-backed public routes."""
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, dimensional_field_response

POSITIONS=np.array([[0,0,-.7],[.4,.2,.7]])
CELL=np.eye(3)*9
VECTORS=("vector_potential","electric_field","magnetic_field","paramagnetic_number_current",
         "diamagnetic_number_current","number_current","charge_current")
SCALARS=("scalar_potential","number_density","charge_density")
DIAGNOSTICS=("continuity_defect","reference_electrons","occupation_consistency_defect",
             "hartree_quadrature_defect","reference_operator_defect","temperature_hartree")

def value(r,key):
    return np.asarray(r[key+"_real"])+1j*np.asarray(r[key+"_imag"])

@pytest.mark.parametrize("dimension",[1,2,3])
@pytest.mark.parametrize("xc",["svwn5","pbe"])
@pytest.mark.parametrize("kind",["electric","magnetic","mixed","uniform_electric"])
def test_native_ase_cli_dimensional_fields(dimension,xc,kind,tmp_path):
    basis=tmp_path/"primitive.g94";basis.write_text("H 0\nS 1 1.0\n0.7 1.0\n****\n")
    pbc=[i<dimension for i in range(3)]
    controls=dict(basis=str(basis),xc=xc,basis_cutoff=4.3,grid_level=0,max_iter=250,
        conv_tol=1e-12,conv_tol_dp=1e-11,conv_tol_grad=1e-10,
        alpha=.65,real_cutoff=18.,reciprocal_cutoff=3.7)
    scalar=.37+.22j if kind in ("electric","mixed") else 0j
    vector=(.1-.2j,-.3+.11j,.27+.19j) if kind in ("magnetic","mixed") else (0j,0j,0j)
    fractional=(.231,0.,0.)
    if kind=="uniform_electric":
        fractional=(0.,0.,0.);vector=tuple(-1j*e/(.13+.04j) for e in (.1,-.3,.27))
    field=dict(transfer_fractional=fractional,omega=.13,damping=.04,scalar_potential=scalar,vector_potential=vector)
    r=dimensional_field_response([1,1],POSITIONS,CELL,pbc=pbc,**controls,**field)
    atoms=Atoms("HH",positions=POSITIONS*Bohr,cell=CELL*Bohr,pbc=pbc)
    configured=DftRs(units="atomic", basis=str(basis),xc=xc,overlap_cutoff=4.3,grid_level=0,max_iter=250,
        conv_tol=1e-12,conv_tol_dp=1e-11,conv_tol_grad=1e-10)
    adapted=configured.get_dimensional_field_response(atoms,alpha=.65,real_cutoff=18.,reciprocal_cutoff=3.7,**field)
    assert max(r["residuals"])<1e-10 and r["hartree_quadrature_defect"]<1e-11
    assert r["shape"]==[2,1,2,2] and r["dimension"]==dimension
    assert r["reference_operator_defect"]<1e-11 and r["occupation_consistency_defect"]<1e-9
    assert r["converged"]==adapted["converged"]
    for key in VECTORS+SCALARS+("density",):
        np.testing.assert_allclose(value(r,key),value(adapted,key),atol=1e-11,rtol=0)
    for key in DIAGNOSTICS+("reference_energy","reference_free_energy"):
        assert abs(r[key]-adapted[key])<1e-11
    wave=np.asarray(fractional)*(2*np.pi/9)
    np.testing.assert_allclose(r["wave"],wave,atol=1e-14,rtol=0)
    np.testing.assert_allclose(value(r,"electric_field"),1j*(wave*scalar+(.13+.04j)*np.asarray(vector)),atol=1e-14,rtol=0)
    np.testing.assert_allclose(value(r,"magnetic_field"),1j*np.cross(wave,vector),atol=1e-14,rtol=0)
    np.testing.assert_allclose(value(r,"diamagnetic_number_current"),2*np.asarray(vector),atol=1e-12,rtol=0)
    np.testing.assert_allclose(value(r,"number_current"),value(r,"paramagnetic_number_current")+value(r,"diamagnetic_number_current"),atol=1e-13,rtol=0)
    raw=abs((.13+.04j)*value(r,"number_density")-np.dot(wave,value(r,"number_current")))
    assert abs(raw-r["continuity_defect"])<1e-14
    if kind=="uniform_electric":
        np.testing.assert_allclose(value(r,"electric_field"),[.1,-.3,.27],atol=1e-14,rtol=0)
        assert abs(value(r,"number_density"))<1e-11
    xyz=tmp_path/"field.xyz"
    xyz.write_text("2\ndimensional periodic field\n"+"".join("H "+" ".join(f"{v*.529177210903:.17g}" for v in row)+"\n" for row in POSITIONS))
    flags=["--cell","4.762594898127","--pbc",",".join(str(int(v)) for v in pbc),"--basis",str(basis),
        "--xc",xc,"--overlap-cutoff","4.3","--grid-level","0","--max-iter","250","--conv-tol","1e-12",
        "--conv-tol-dp","1e-11","--conv-tol-grad","1e-10","--ewald-alpha",".65","--ewald-real-cutoff","18",
        "--ewald-reciprocal-cutoff","3.7","--field-transfer",",".join(map(str,fractional)),"--response-omega",".13",
        "--response-damping",".04","--field-scalar",f"{scalar.real},{scalar.imag}","--field-vector",
        ";".join(f"{z.real},{z.imag}" for z in vector),"--log-mode","none"]
    cli=subprocess.run([os.environ["DFT_RS_CLI"],"dimensional-scf",str(xyz),*flags],text=True,capture_output=True)
    assert cli.returncode==0,cli.stderr
    rows=[row.split() for row in cli.stdout.splitlines() if row]
    for key in VECTORS:
        entries=[row for row in rows if row[0]==key];assert len(entries)==3
        np.testing.assert_allclose([complex(float(row[2]),float(row[3])) for row in entries],value(r,key),atol=1e-11,rtol=0)
    for key in SCALARS:
        row=next(row for row in rows if row[0]==key)
        assert abs(complex(float(row[1]),float(row[2]))-value(r,key))<1e-11
    for key in DIAGNOSTICS:
        assert abs(next(float(row[1]) for row in rows if row[0]==key)-r[key])<1e-11
    density=value(r,"density");entries=[row for row in rows if row[0]=="field_density"]
    assert len(entries)==density.size
    for row in entries:
        index=tuple(map(int,row[1:5]));assert abs(density[index]-complex(float(row[5]),float(row[6])))<1e-11
    assert r["model"] in cli.stdout and r["units"] in cli.stdout
    assert next(row[1] for row in rows if row[0]=="field_response_converged")==str(r["converged"]).lower()

@pytest.mark.parametrize("dimension",[1,2,3])
@pytest.mark.parametrize("charge",[-1,0,1])
def test_signed_charge_fields_and_ase_thermal_folded_number_constraint(dimension,charge,tmp_path):
    pbc=[i<dimension for i in range(3)];pos=[[.2,-.1,.3]]
    controls=dict(pbc=pbc,xc="svwn5",charge=charge,multiplicity=2 if charge==0 else 1,
        basis_cutoff=4.3,k_mesh=[3 if i==0 else 1 for i in range(3)],compensation="gaussian",
        countercharge_center=[-.5,.7,.9],countercharge_width=1.1,alpha=.65,real_cutoff=18.,reciprocal_cutoff=3.7)
    r=dimensional_field_response([1],pos,CELL,transfer_fractional=[.231,0,0],vector_potential=[0,.1,0],omega=.13,damping=.04,**controls)
    assert r["formal_charge"]==charge and abs(r["reference_electrons"]-(1-charge))<1e-12
    np.testing.assert_allclose(value(r,"diamagnetic_number_current"),[0,.1*(1-charge),0],atol=1e-12,rtol=0)
    assert max(r["residuals"])<1e-10

def test_dimensional_field_validation_and_static_thermal_folded_sector(tmp_path):
    for controls in ({"transfer_fractional":[.1,.1,0]}, {"scalar_potential":complex(float("nan"),0)},
        {"vector_potential":[0,0]}, {"imaginary":True,"damping":.1}, {"response_tol":0}):
        with pytest.raises((ValueError,RuntimeError)):
            dimensional_field_response([1,1],POSITIONS,CELL,pbc=[True,False,False],**controls)
    basis=tmp_path/"thermal.g94";basis.write_text("H 0\nS 1 1.0\n0.7 1.0\n****\n")
    atoms=Atoms("HH",positions=POSITIONS*Bohr,cell=CELL*Bohr,pbc=[True,False,False])
    calculator=DftRs(units="atomic", basis=str(basis),xc="svwn5",overlap_cutoff=4.3,grid_level=0,electronic_temperature=200000,
        conv_tol=1e-12,conv_tol_dp=1e-11,conv_tol_grad=1e-10)
    for fractional in ([0.,0.,0.],[1.,0.,0.],[-1.,0.,0.]):
        r=calculator.get_dimensional_field_response(atoms,transfer_fractional=fractional,scalar_potential=.37+.22j)
        assert r["temperature_hartree"]>0
        np.testing.assert_array_equal(r["reduced_transfer_fractional"],[0.,0.,0.])
        np.testing.assert_array_equal(r["target_k_points"],r["k_points"])
        assert r["particle_number_error"]<1e-11
        if fractional==[0.,0.,0.]:
            assert abs(value(r,"number_density"))<1e-11
            assert np.max(abs(value(r,"density")))<1e-11
    executable=os.environ["DFT_RS_CLI"]
    for flags in (["--field-transfer",".1,.1,0","--pbc","1,0,0"],["--field-transfer",".1,0,0","--response-q",".1,0,0"],
        ["--field-transfer",".1,0,0","--response-modes","0"],["--field-transfer",".1,0,0","--field-wave",".1,0,0"],
        ["--field-transfer",".1,0,0","--imaginary-frequency","--response-damping",".1"]):
        cli=subprocess.run([executable,"dimensional-scf",str(tmp_path/"missing.xyz"),"--cell","4.7",*flags],text=True,capture_output=True)
        assert cli.returncode!=0 and "No such file" not in cli.stderr
