# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Actual orbital magnetic/NMR routes and independent complex spectral fits."""
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, magnetic_response, nmr

POSITIONS=np.array([[.16,-.24,.31],[1.41,.38,-.62],[-.72,1.22,.47]])
NUMBERS=[8,1,1]
ORIGIN=[.43,-.36,.29]

def options(name,opened):
    return dict(xc=name,charge=int(opened),multiplicity=2 if opened else 1,
        grid_level=2,max_iter=300,conv_tol=1e-12,conv_tol_grad=1e-10,conv_tol_dp=1e-11,
        origin=ORIGIN,response_tol=1e-10)

@pytest.mark.parametrize("opened",[False,True])
@pytest.mark.parametrize("name",["hf","pbe","r2scan"])
def test_actual_native_ase_cli_density_and_nmr(opened,name,tmp_path):
    opts=options(name,opened)
    report=magnetic_response(NUMBERS,POSITIONS,**opts)
    shielding=nmr(NUMBERS,POSITIONS,**opts)
    assert report["shape"]==[2,3,7,7]
    assert report["occupied"]==([5,4] if opened else [5,5])
    assert report["converged"] and shielding["converged"]
    assert max(report["residuals"])<=1e-10
    assert report["solver_methods"]==["MINRES"]*6
    assert report["projector_defect"]<1e-9
    assert report["particle_number_defect"]<1e-12
    assert report["antisymmetry_defect"]<1e-12
    assert report["origin_units"]=="bohr" and shielding["units"]=="ppm"
    atoms=Atoms(numbers=NUMBERS,positions=POSITIONS*Bohr)
    controls={k:v for k,v in opts.items() if k not in ("origin","response_tol")}
    calculator=DftRs(units="atomic", **controls)
    adapted=calculator.get_magnetic_response(atoms,origin=ORIGIN)
    adapted_nmr=calculator.get_nmr(atoms,origin=ORIGIN)
    for key in ("density_hat","overlap_hat","explicit_fock_hat","residuals",
        "projector_defect","particle_number_defect","antisymmetry_defect","energy"):
        np.testing.assert_allclose(report[key],adapted[key],atol=1e-11,rtol=0)
    for a,b in zip(shielding["shieldings"],adapted_nmr["shieldings"]):
        for key in ("tensor","diamagnetic","paramagnetic"):
            np.testing.assert_allclose(a[key],b[key],atol=1e-9,rtol=0)
    executable=os.environ.get("DFT_RS_CLI")
    assert executable,"actual CLI is required for acceptance"
    xyz=tmp_path/"magnetic.xyz"
    xyz.write_text("3\norbital magnetic response\n"+"".join(
        symbol+" "+" ".join(f"{v*.529177210903:.17g}" for v in row)+"\n"
        for symbol,row in zip(("O","H","H"),POSITIONS)))
    flags=["--xc",name,"--basis","sto-3g","--charge",str(int(opened)),
        "--mult",str(2 if opened else 1),"--grid-level","2","--max-iter","300",
        "--conv-tol","1e-12","--conv-tol-grad","1e-10","--conv-tol-dp","1e-11",
        "--magnetic-origin",",".join(map(str,ORIGIN)),"--log-mode","none"]
    density=subprocess.run([executable,"magnetic-response",str(xyz),*flags],
        capture_output=True,text=True,check=True).stdout
    rows=[line.split() for line in density.splitlines() if line]
    for key,dimensions in (("density_hat",4),("explicit_fock_hat",4),("overlap_hat",3)):
        values=[row for row in rows if row[0]==key]
        assert len(values)==np.asarray(report[key]).size
        for row in values:
            index=tuple(map(int,row[1:1+dimensions]))
            assert float(row[1+dimensions])==pytest.approx(np.asarray(report[key])[index],abs=1e-10)
    assert report["units"] in density and report["model"] in density and "Converged: true" in density
    output=subprocess.run([executable,"nmr",str(xyz),*flags],capture_output=True,text=True,check=True).stdout
    rows=[line.split() for line in output.splitlines() if line.startswith("shielding ")]
    assert len(rows)==27
    for row in rows:
        atom,i,j=map(int,row[1:4])
        for part,value in zip(("tensor","diamagnetic","paramagnetic"),row[4:]):
            assert float(value)==pytest.approx(shielding["shieldings"][atom][part][i][j],abs=1e-8)
    assert shielding["units"] in output and shielding["model"] in output

def finite_spectral_projector(f,s,fh,sh,h,occupied):
    # Independent complex generalized eigendecomposition. The finite
    # stationary model F(B)=F0+i*B*Fhat and S(B)=S0+i*B*Shat has the exact
    # first derivative for pure LDA/GGA orbital magnetic response, because
    # their real-density kernels vanish for antisymmetric imaginary density.
    overlap=s+1j*h*sh
    se,sv=np.linalg.eigh(overlap)
    assert min(se)>1e-8
    x=(sv/se**.5)@sv.conj().T
    _,v=np.linalg.eigh(x.conj().T@(f+1j*h*fh)@x)
    c=x@v[:,:occupied]
    return c@c.conj().T

@pytest.mark.parametrize("opened",[False,True])
@pytest.mark.parametrize("name",["svwn5","pbe"])
@pytest.mark.parametrize("gauge",["giao","common_origin"])
def test_public_magnetic_derivative_has_four_step_spectral_fit(opened,name,gauge):
    report=magnetic_response(NUMBERS,POSITIONS,**dict(options(name,opened),gauge=gauge))
    w=np.array([.4,-.3,.2])
    exact=np.einsum("a,saij->sij",w,report["density_hat"])
    fh=np.einsum("a,saij->sij",w,report["explicit_fock_hat"])
    sh=np.einsum("a,aij->ij",w,report["overlap_hat"])
    steps=np.array([.01,.005,.0025,.00125])
    values=[]
    for h in steps:
        result=[]
        for spin in range(2):
            args=(np.asarray(report["reference_fock"])[spin],np.asarray(report["overlap"]),
                fh[spin],sh)
            a=finite_spectral_projector(*args,h,report["occupied"][spin])
            b=finite_spectral_projector(*args,-h,report["occupied"][spin])
            result.append(((a-b)/(2*h)).imag)
        values.append(result)
    values=np.array(values)
    errors=np.max(np.abs(values-exact),axis=(1,2,3))
    assert np.all(errors>0) and np.all(np.isfinite(errors))
    p,b=np.polyfit(np.log(steps),np.log(errors),1)
    r2=1-np.sum((np.log(errors)-b-p*np.log(steps))**2)/np.sum((np.log(errors)-np.mean(np.log(errors)))**2)
    print(f"public {name} open={opened} {gauge} orbital magnetic: errors={errors.tolist()} reduced field steps=.01..00125 p={p:.12f} R2={r2:.12f}")
    assert abs(p-2)<.02 and r2>.99999
    np.testing.assert_allclose(errors[:-1]/errors[1:],4,rtol=.03,atol=0)
    assert np.max(np.abs((4*values[-1]-values[-2])/3-exact))<1e-8

def test_empty_spin_closed_shell_limit_and_invalid_controls(tmp_path):
    atom=magnetic_response([1],[[0,0,0]],multiplicity=2)
    assert atom["occupied"]==[1,0] and atom["converged"]
    assert np.max(np.abs(atom["density_hat"]))==0
    a=magnetic_response([1,1],[[0,0,0],[1.4,0,0]])
    b=magnetic_response([1,1],[[0,0,0],[1.4,0,0]],unrestricted=True)
    np.testing.assert_allclose(a["density_hat"],b["density_hat"],atol=1e-11,rtol=0)
    for args in ({"gauge":"invalid"},{"origin":[np.nan,0,0]},{"response_tol":0},
                 {"response_max_iter":0},{"response_restart":0}):
        with pytest.raises(ValueError):
            magnetic_response([1,1],[[0,0,0],[1.4,0,0]],**args)
    with pytest.raises(NotImplementedError,match="nonlocal"):
        magnetic_response(NUMBERS,POSITIONS,xc="wb97m-v",grid_level=1)
    executable=os.environ["DFT_RS_CLI"]
    for command,flags in (("energy",["--magnetic-origin","0,0,0"]),
                          ("magnetic-response",["--field","0,0,.01"]),
                          ("nmr",["--forces"]),
                          ("magnetic-response",["--response-tol","0"])):
        process=subprocess.run([executable,command,str(tmp_path/"missing.xyz"),*flags],
            capture_output=True,text=True)
        assert process.returncode!=0 and "No such file" not in process.stderr

