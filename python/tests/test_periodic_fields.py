# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Bare dimensional electromagnetic vertices through native, ASE and both CLIs."""
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, periodic_field_operators
CELL=np.array([[5.2,.5,.3],[.8,5.6,.4],[.3,.2,6.]])
POSITIONS=np.array([[.2,-.1,.3],[1.6,.5,-.4]])
WAVE=np.array([.137,-.084,.067])
OFFSET=np.array([.041,-.023,.027])
SCALAR=np.array([.03,-.02])
VECTOR=np.array([[.05,.03],[-.02,.01],[.01,-.04]])
def complex_array(report,key):
    return np.asarray(report[key+"_real"])+1j*np.asarray(report[key+"_imag"])

@pytest.mark.parametrize("dimension",[1,2,3])
@pytest.mark.parametrize("charge",[-1,0,1])
def test_actual_native_ase_cli_electric_magnetic_vertices(dimension,charge,tmp_path):
    pbc=[i<dimension for i in range(3)]
    options=dict(pbc=pbc,k_mesh=[2 if i<dimension else 1 for i in range(3)],
        wave=WAVE,k_offset=OFFSET,frequency=[.23,.01],charge=charge,
        scalar_potential=SCALAR,vector_potential=VECTOR)
    report=periodic_field_operators([1,1],POSITIONS,CELL,**options)
    assert report["dimension"]==dimension and report["formal_charge"]==charge
    assert report["shape"]==[2**dimension,2,2]
    assert report["transversality_defect"]<1e-17
    assert np.max(np.abs(complex_array(report,"density_vertex")
        -complex_array(report,"density_vertex").conj().transpose(0,2,1)))>1e-3
    scalar=complex(*SCALAR);vector=VECTOR[:,0]+1j*VECTOR[:,1]
    density=complex_array(report,"density_vertex")
    current=complex_array(report,"current_vertex")
    np.testing.assert_allclose(complex_array(report,"first_order"),
        scalar*density+np.einsum("a,akij->kij",vector,current),atol=1e-14,rtol=0)
    np.testing.assert_allclose(complex_array(report,"second_order"),
        np.dot(vector,vector)*complex_array(report,"contact_vertex"),atol=1e-14,rtol=0)
    np.testing.assert_allclose(np.asarray(report["target_k_points"])-report["k_points"],
        np.tile(WAVE,(2**dimension,1)),atol=1e-15,rtol=0)
    atoms=Atoms("HH",positions=POSITIONS*Bohr,cell=CELL*Bohr,pbc=pbc)
    adapted=DftRs(units="atomic").get_periodic_field_operators(atoms,**{k:v for k,v in options.items() if k!="pbc"})
    for key in ("density_vertex","current_vertex","first_order","second_order","contact_vertex"):
        np.testing.assert_allclose(complex_array(report,key),complex_array(adapted,key),atol=1e-13,rtol=0)
    executable=os.environ.get("DFT_RS_CLI")
    assert executable,"actual CLI is required"
    xyz=tmp_path/"fields.xyz"
    xyz.write_text("2\nperiodic Fourier fields\n"+"".join(
        "H "+" ".join(f"{v*.529177210903:.17g}" for v in row)+"\n" for row in POSITIONS))
    flags=["--cell",";".join(",".join(f"{v*.529177210903:.17g}" for v in row) for row in CELL),
        "--pbc",",".join(str(int(v)) for v in pbc),"--k-mesh",",".join(map(str,options["k_mesh"])),
        "--charge",str(charge),"--field-wave",",".join(map(str,WAVE)),
        "--field-k-offset",",".join(map(str,OFFSET)),"--field-frequency",".23,.01",
        "--field-scalar",".03,-.02","--field-vector",".05,.03;-.02,.01;.01,-.04","--log-mode","none"]
    process=subprocess.run([executable,"periodic-field",str(xyz),*flags],capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    rows=[line.split() for line in process.stdout.splitlines() if line]
    for key,count in (("density_vertex",3),("current_vertex",4),("first_order",3),("second_order",3),("contact_vertex",3)):
        values=[r for r in rows if r[0]==key]
        assert len(values)==complex_array(report,key).size
        for row in values:
            index=tuple(map(int,row[1:count+1]))
            actual=complex(float(row[count+1]),float(row[count+2]))
            assert abs(actual-complex_array(report,key)[index])<1e-12
    assert report["model"] in process.stdout and report["units"] in process.stdout
    for key in ("electric_field","magnetic_field"):
        expected=np.asarray(report[key])
        values=[r for r in rows if r[0]==key]
        for row in values:
            np.testing.assert_allclose(list(map(float,row[2:])),expected[int(row[1])],atol=1e-14,rtol=0)

@pytest.mark.parametrize("dimension",[1,2,3])
def test_q_reversal_and_em_field_gauge_transformation(dimension):
    kwargs=dict(pbc=[i<dimension for i in range(3)],k_mesh=[2 if i<dimension else 1 for i in range(3)],
        k_offset=OFFSET,wave=WAVE,scalar_potential=SCALAR,vector_potential=VECTOR,frequency=[.23,.01])
    plus=periodic_field_operators([1,1],POSITIONS,CELL,**kwargs)
    reverse=periodic_field_operators([1,1],POSITIONS,CELL,**dict(kwargs,wave=-WAVE,k_offset=OFFSET+WAVE))
    for key in ("density_vertex","current_vertex"):
        values=complex_array(reverse,key)
        np.testing.assert_allclose(complex_array(plus,key),np.swapaxes(values.conj(),-1,-2),atol=1e-12,rtol=0)
    lam=.02-.03j;z=.23+.01j;av=VECTOR[:,0]+1j*VECTOR[:,1];v=complex(*SCALAR)
    ag=av+1j*WAVE*lam;vg=v-1j*z*lam
    transformed=periodic_field_operators([1,1],POSITIONS,CELL,**dict(kwargs,
        scalar_potential=[vg.real,vg.imag],vector_potential=np.array([ag.real,ag.imag]).T))
    for key in ("electric_field","magnetic_field"):
        np.testing.assert_allclose(plus[key],transformed[key],atol=1e-17,rtol=0)

def test_invalid_and_incompatible_field_controls(tmp_path):
    for options in ({"wave":[np.nan,0,0]},{"k_mesh":[0,1,1]},{"frequency":[0,-.1]},
                    {"basis_cutoff":0},{"pbc":[False,False,False]},{"vector_potential":[[0,0],[0,0]]}):
        with pytest.raises(ValueError):
            periodic_field_operators([1,1],POSITIONS,CELL,**options)
    executable=os.environ["DFT_RS_CLI"]
    for command,options in (("energy",["--field-wave","0,0,.1"]),
            ("periodic-field",["--xc","pbe"]),("periodic-field",["--field-frequency","0,-.1"]),
            ("periodic-field",["--forces"])):
        result=subprocess.run([executable,command,str(tmp_path/"missing.xyz"),*options],capture_output=True,text=True)
        assert result.returncode!=0 and "No such file" not in result.stderr

