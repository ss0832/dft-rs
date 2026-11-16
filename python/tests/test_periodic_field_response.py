# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Actual screened electric/magnetic responses through all public surfaces."""
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, periodic_field_response

POSITIONS=np.array([[0,0,-.7],[.4,.2,.7]])
CELL=np.eye(3)*9
VECTORS=("vector_potential","electric_field","magnetic_field","paramagnetic_number_current",
         "diamagnetic_number_current","number_current","charge_current")
SCALARS=("scalar_potential","number_density","charge_density")

def complex_value(report,key):
    return np.asarray(report[key+"_real"])+1j*np.asarray(report[key+"_imag"])

def densities(report):
    return np.asarray([[np.asarray(k["real"])+1j*np.asarray(k["imag"]) for k in spin]
                       for spin in report["density"]])

@pytest.mark.parametrize("xc",["svwn5","pbe"])
@pytest.mark.parametrize("kind",["electric","magnetic","mixed","uniform_electric"])
def test_native_ase_and_cli_screened_field_current_match(xc,kind,tmp_path):
    basis=tmp_path/"primitive.g94"
    basis.write_text("H 0\nS 1 1.0\n0.7 1.0\n****\n")
    controls=dict(basis=str(basis),xc=xc,overlap_cutoff=4.3,grid_level=0,grid_shell=1,
        max_iter=250,conv_tol=1e-12,conv_tol_dp=1e-11,conv_tol_grad=1e-10)
    scalar=.37+.22j if kind in ("electric","mixed") else 0j
    vector=(.1-.2j,-.3+.11j,.27+.19j) if kind in ("magnetic","mixed") else (0j,0j,0j)
    q=(.043,.017,-.021)
    if kind=="uniform_electric":
        q=(0.,0.,0.)
        vector=tuple(-1j*e/(.13+.04j) for e in (.1,-.3,.27))
    field=dict(q=q,omega=.13,damping=.04,reciprocal_cutoff=.85,
        scalar_potential=scalar,vector_potential=vector)
    r=periodic_field_response([1,1],POSITIONS,CELL,**controls,**field)
    atoms=Atoms("HH",positions=POSITIONS*Bohr,cell=CELL*Bohr,pbc=True)
    adapted=DftRs(units="atomic", **controls).get_periodic_field_response(atoms,**field)
    assert r["converged"] and max(r["residuals"])<1e-10
    assert r["shape"]==[2,1,2,2]
    assert "LDA/GGA" in r["model"] and "per-cell" in r["units"]
    for key in VECTORS+SCALARS:
        np.testing.assert_allclose(complex_value(r,key),complex_value(adapted,key),atol=1e-11,rtol=0)
    np.testing.assert_allclose(densities(r),densities(adapted),atol=1e-11,rtol=0)
    for key in ("continuity_defect","reference_electrons","occupation_consistency_defect","energy","free_energy"):
        assert abs(r[key]-adapted[key])<1e-11
    np.testing.assert_allclose(complex_value(r,"electric_field"),
        1j*(np.asarray(field["q"])*scalar+(.13+.04j)*np.asarray(vector)),atol=1e-14,rtol=0)
    np.testing.assert_allclose(complex_value(r,"magnetic_field"),
        1j*np.cross(field["q"],vector),atol=1e-14,rtol=0)
    np.testing.assert_allclose(complex_value(r,"diamagnetic_number_current"),2*np.asarray(vector),atol=1e-12,rtol=0)
    np.testing.assert_allclose(complex_value(r,"number_current"),complex_value(r,"paramagnetic_number_current")+
        complex_value(r,"diamagnetic_number_current"),atol=1e-13,rtol=0)
    np.testing.assert_allclose(complex_value(r,"charge_current"),-complex_value(r,"number_current"),atol=1e-14,rtol=0)
    ward=abs((.13+.04j)*complex_value(r,"number_density")-np.dot(field["q"],complex_value(r,"number_current")))
    assert abs(ward-r["continuity_defect"])<1e-14
    if kind=="uniform_electric":
        np.testing.assert_allclose(complex_value(r,"electric_field"),[.1,-.3,.27],atol=1e-14,rtol=0)
        assert abs(complex_value(r,"number_density"))<1e-11 and ward<1e-11
    np.testing.assert_allclose(np.asarray(r["target_k_points"])-np.asarray(r["k_points"]),[field["q"]],atol=1e-14,rtol=0)
    d=densities(r)
    assert np.max(abs(d-d.swapaxes(-1,-2).conj()))>1e-5
    xyz=tmp_path/"field.xyz"
    xyz.write_text("2\nperiodic field response\n"+"".join("H "+" ".join(
        f"{v*.529177210903:.17g}" for v in row)+"\n" for row in POSITIONS))
    args=[os.environ["DFT_RS_CLI"],"field-response",str(xyz),"--cell","4.762594898127",
        "--basis",str(basis),"--xc",xc,"--overlap-cutoff","4.3","--grid-level","0","--grid-shell","1",
        "--max-iter","250","--conv-tol","1e-12","--conv-tol-dp","1e-11","--conv-tol-grad","1e-10",
        "--response-q",",".join(map(str,q)),"--response-omega",".13","--response-damping",".04",
        "--reciprocal-cutoff",".85","--field-scalar",f"{scalar.real},{scalar.imag}",
        "--field-vector",";".join(f"{a.real},{a.imag}" for a in vector),"--log-mode","none"]
    result=subprocess.run(args,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    rows=[row.split() for row in result.stdout.splitlines() if row]
    for key in VECTORS:
        entries=[row for row in rows if row[0]==key]
        assert len(entries)==3
        values=[complex(float(row[2]),float(row[3])) for row in entries]
        np.testing.assert_allclose(values,complex_value(r,key),atol=1e-11,rtol=0)
    for key in SCALARS:
        row=next(row for row in rows if row[0]==key)
        assert abs(complex(float(row[1]),float(row[2]))-complex_value(r,key))<1e-11
    entries=[row for row in rows if row[0]=="density"]
    assert len(entries)==d.size
    for row in entries:
        index=tuple(map(int,row[1:5]))
        assert abs(d[index]-complex(float(row[5]),float(row[6])))<1e-11
    for key in ("continuity_defect","reference_electrons","occupation_consistency_defect"):
        value=next(float(row[1]) for row in rows if row[0]==key)
        assert abs(value-r[key])<1e-11
    assert r["units"] in result.stdout and r["model"] in result.stdout

def test_field_input_validation_and_uniform_scalar_gauge(tmp_path):
    for controls in ({"scalar_potential":complex(float("nan"),0)},
                     {"vector_potential":[0,0]}, {"q":[float("nan"),0,0]},
                     {"imaginary":True,"damping":.1},{"response_tol":0}):
        with pytest.raises((ValueError,RuntimeError)):
            periodic_field_response([1,1],POSITIONS,CELL,**controls)
    executable=os.environ["DFT_RS_CLI"]
    for flags in (["--field-wave",".1,0,0"],["--field-frequency",".1,.01"],
                  ["--field-k-offset",".1,0,0"],["--response-modes","0"],["--forces"]):
        result=subprocess.run([executable,"field-response",str(tmp_path/"missing.xyz"),"--cell","4.7",*flags],
            text=True,capture_output=True)
        assert result.returncode!=0 and "No such file" not in result.stderr
    basis=tmp_path/"gauge.g94";basis.write_text("H 0\nS 1 1.0\n0.7 1.0\n****\n")
    r=periodic_field_response([1,1],POSITIONS,CELL,basis=str(basis),xc="svwn5",overlap_cutoff=4.3,
        grid_level=0,grid_shell=1,reciprocal_cutoff=.85,scalar_potential=1.0,
        electronic_temperature=40000,conv_tol_grad=1e-10,conv_tol=1e-12,conv_tol_dp=1e-11)
    assert np.max(abs(densities(r)))<1e-10
    assert abs(complex_value(r,"number_density"))<1e-10
    assert r["occupation_consistency_defect"]<1e-9
    assert r["temperature_hartree"]>0
    np.testing.assert_allclose(complex_value(r,"electric_field"),0,atol=0,rtol=0)
    np.testing.assert_allclose(complex_value(r,"magnetic_field"),0,atol=0,rtol=0)
    assert any("Berry" in warning for warning in r["warnings"])
