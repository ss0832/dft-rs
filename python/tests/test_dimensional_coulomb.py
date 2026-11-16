# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Actual public contracted Coulomb operators, density constraints and fits."""
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, periodic_basis, periodic_coulomb_operators, periodic_capabilities

CELL=np.array([[5.2,.5,.3],[.8,5.6,.4],[.3,.2,6.]])
POSITIONS=np.array([[.2,-.1,.3],[1.6,.5,-.4]])
CONTROLS=dict(charge=1,k_mesh=[3,1,1],basis_cutoff=4.3,compensation="gaussian",
              countercharge_center=[-.5,.7,.9],countercharge_width=1.1,
              alpha=.55,real_cutoff=16.,reciprocal_cutoff=8.)


def inputs(dimension):
    options=dict(CONTROLS,pbc=[i<dimension for i in range(3)])
    matrices=periodic_basis([1,1],POSITIONS,CELL,k_mesh=[3,1,1],
                           basis_cutoff=4.3,pbc=options["pbc"])
    overlap=np.array(matrices["overlap_real"])+1j*np.array(matrices["overlap_imag"])
    real=np.tile([[.6,.2],[.2,.5]],(3,1,1))
    imaginary=np.tile([[0.,.07],[-.07,0.]],(3,1,1))
    count=np.einsum("kij,kji->",real+1j*imaginary,overlap).real/3
    options.update(density_real=real/count,density_imag=imaginary/count)
    return options


def raw(report):
    for key in ("quadrature_operator_defect","quadrature_derivative_defect","quadrature_source_defect",
                "hermiticity_defect","translation_defect","hartree_potential_imaginary_defect"):
        assert report[key]<1e-11, (key,report[key])


@pytest.mark.parametrize("dimension",[1,2,3])
@pytest.mark.parametrize("nuclear_widths",[None,[1.0,1.3]])
def test_all_public_paths_share_density_operators_components_and_gradients(dimension,nuclear_widths,tmp_path):
    options=inputs(dimension)
    if nuclear_widths is not None:
        options["nuclear_widths"]=nuclear_widths
    report=periodic_coulomb_operators([1,1],POSITIONS,CELL,**options)
    raw(report)
    assert report["converged"] and report["electron_count"]==pytest.approx(1.,abs=1e-12)
    assert report["formal_charge"]==1 and report["compensating_charge"]==-1
    assert report["shape"]==[3,2,2] and report["centre_count"]==3
    assert report["real_space_density_imaginary_max"]>1e-3
    assert np.max(np.abs(report["nuclear_imag"]))>1e-4
    atoms=Atoms("HH",positions=POSITIONS*Bohr,cell=CELL*Bohr,pbc=options["pbc"])
    adapted=DftRs(units="atomic").get_periodic_coulomb_operators(atoms,**{k:v for k,v in options.items() if k!="pbc"})
    for name in ("nuclear","hartree"):
        for suffix in ("real","imag","components_real","components_imag","derivatives_real","derivatives_imag"):
            np.testing.assert_allclose(report[f"{name}_{suffix}"],adapted[f"{name}_{suffix}"],atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.sum(report[f"{name}_components_real"],axis=1),report[f"{name}_real"],atol=1e-12,rtol=0)
    for key in ("source_energy","total_coulomb_energy","source_gradients","energy_gradients","electron_count",
                "expected_electrons","charge_residual","measure"):
        np.testing.assert_allclose(report[key],adapted[key],atol=1e-12,rtol=0)
    executable=os.environ.get("DFT_RS_CLI")
    assert executable,"DFT_RS_CLI is required for actual CLI acceptance"
    conversion=.529177210903
    xyz=tmp_path/"density.xyz"
    xyz.write_text("2\ncontracted Coulomb operators\n"+"".join(
        "H "+" ".join(f"{v*conversion:.17g}" for v in row)+"\n" for row in POSITIONS))
    csv=tmp_path/"density.csv"
    csv.write_text("k,row,col,real,imag\n"+"".join(
        f"{k},{i},{j},{options['density_real'][k,i,j]:.17g},{options['density_imag'][k,i,j]:.17g}\n"
        for k in range(3) for i in range(2) for j in range(2)))
    cell=";".join(",".join(f"{v*conversion:.17g}" for v in row) for row in CELL)
    command=[executable,"periodic-coulomb",str(xyz),"--cell",cell,"--pbc",",".join(str(int(v)) for v in options["pbc"]),
             "--basis","sto-3g","--charge","1","--k-mesh","3,1,1","--overlap-cutoff","4.3",
             "--charge-compensation","gaussian","--countercharge-center","-.5,.7,.9","--countercharge-width","1.1",
             "--ewald-alpha",".55","--ewald-real-cutoff","16","--ewald-reciprocal-cutoff","8",
             "--density-matrices",str(csv),"--log-mode","none"]
    if nuclear_widths is not None:
        command += ["--nuclear-widths",",".join(str(value) for value in nuclear_widths)]
    result=subprocess.run(command,capture_output=True,text=True,check=True)
    rows=[line.split() for line in result.stdout.splitlines() if line]
    for name in ("nuclear","hartree"):
        for label,suffix,indices in ((name,"",3),(name+"_component","components_",4),
                                     (name+"_derivative","derivatives_",5)):
            found=[row for row in rows if row[0]==label]
            expected=np.asarray(report[name+"_"+suffix+"real"])
            assert len(found)==expected.size
            for row in found:
                index=tuple(map(int,row[1:1+indices]))
                assert float(row[1+indices])==pytest.approx(expected[index],abs=1e-12)
                assert float(row[2+indices])==pytest.approx(np.asarray(report[name+"_"+suffix+"imag"])[index],abs=1e-12)
    scalar_names=("source_energy","total_coulomb_energy","electron_count","expected_electrons","charge_residual",
                  "quadrature_operator_defect","quadrature_derivative_defect","quadrature_source_defect",
                  "hermiticity_defect","translation_defect","real_space_density_imaginary_max",
                  "hartree_potential_imaginary_defect")
    for key in scalar_names:
        assert float(next(row[1] for row in rows if row[0]==key))==pytest.approx(report[key],abs=1e-12)
    for key in ("source_energy_components",):
        row=next(row for row in rows if row[0]==key)
        assert {name:float(value) for name,value in zip(row[1::2],row[2::2])}==pytest.approx(report[key],abs=1e-12)
    for key in ("electron_nuclear_components","hartree_energy_components"):
        np.testing.assert_allclose(list(map(float,next(row[1:] for row in rows if row[0]==key))),report[key],atol=1e-12,rtol=0)
    for label,key in (("source_gradient","source_gradients"),("energy_gradient","energy_gradients")):
        found=[row for row in rows if row[0]==label]
        assert len(found)==3
        for row in found:np.testing.assert_allclose(list(map(float,row[2:])),report[key][int(row[1])],atol=1e-12,rtol=0)
    assert report["model"] in result.stdout and report["units"] in result.stdout
    assert "converged true quadrature_order 192" in result.stdout
    valid=csv.read_text()
    lines=valid.splitlines()
    for content,reason in (("\n".join(lines[:-1])+"\n","complete matrix entries"),
                           ("\n".join(lines[:-1]+[lines[1]])+"\n","duplicate matrix entry"),
                           (valid.replace(lines[0],"row,col,real"),"requires header"),
                           (valid.replace(lines[1],"0,0,0,nan,0"),"finite matrix shape")):
        csv.write_text(content)
        invalid=subprocess.run(command,capture_output=True,text=True)
        assert invalid.returncode!=0 and reason in invalid.stderr
    csv.write_text(valid)


@pytest.mark.parametrize("dimension",[1,2,3])
def test_public_contracted_derivatives_have_four_step_matrix_and_energy_fits(dimension):
    options=inputs(dimension)
    exact=periodic_coulomb_operators([1,1],POSITIONS,CELL,**options)
    steps=np.array([.02,.01,.005,.0025])
    estimates={name:[] for name in ("nuclear","hartree","energy")}
    for h in steps:
        plus,minus=POSITIONS.copy(),POSITIONS.copy()
        plus[1,0]+=h;minus[1,0]-=h
        p=periodic_coulomb_operators([1,1],plus,CELL,**options)
        m=periodic_coulomb_operators([1,1],minus,CELL,**options)
        raw(p);raw(m)
        for name in ("nuclear","hartree"):
            a=np.array(p[f"{name}_real"])+1j*np.array(p[f"{name}_imag"])
            b=np.array(m[f"{name}_real"])+1j*np.array(m[f"{name}_imag"])
            estimates[name].append((a-b)/(2*h))
        estimates["energy"].append((p["total_coulomb_energy"]-m["total_coulomb_energy"])/(2*h))
    for name,values in estimates.items():
        actual=(exact["energy_gradients"][1][0] if name=="energy" else
            (np.array(exact[f"{name}_derivatives_real"])+1j*np.array(exact[f"{name}_derivatives_imag"]))[:,1,0])
        errors=np.array([np.max(np.abs(value-actual)) for value in values])
        assert np.all(errors>0) and np.all(np.isfinite(errors))
        p,intercept=np.polyfit(np.log(steps),np.log(errors),1)
        r2=1-np.sum((np.log(errors)-intercept-p*np.log(steps))**2)/np.sum((np.log(errors)-np.mean(np.log(errors)))**2)
        print(f"{dimension}D public contracted {name}: errors={errors.tolist()} h=.02..0025 p={p:.12f} R2={r2:.12f}")
        assert abs(p-2)<.02 and r2>.99999
        np.testing.assert_allclose(errors[:-1]/errors[1:],4,rtol=.03,atol=0)
        np.testing.assert_allclose((4*values[-1]-values[-2])/3,actual,atol=1e-8,rtol=0)


def test_density_absence_charge_residual_and_input_contracts_are_explicit(tmp_path):
    assert periodic_capabilities()["dimensional_coulomb_operators"]
    options=inputs(1)
    empty={k:v for k,v in options.items() if not k.startswith("density_")}
    nuclear=periodic_coulomb_operators([1,1],POSITIONS,CELL,**empty)
    assert nuclear["hartree_real"] is None and nuclear["total_coulomb_energy"] is None
    assert nuclear["energy_gradients"] is None and nuclear["electron_count"] is None
    malformed=dict(options,density_real=np.asarray(options["density_real"])*1.2)
    report=periodic_coulomb_operators([1,1],POSITIONS,CELL,**malformed)
    assert not report["converged"] and abs(report["charge_residual"])>.1
    for changed in ({"density_imag":[[[0.]]]}, {"compensation":"uniform","countercharge_center":None,"countercharge_width":None},
                    {"density_real":None}, {"xc":"pbe"}, {"temperature":100.}):
        with pytest.raises((ValueError,TypeError)):
            periodic_coulomb_operators([1,1],POSITIONS,CELL,**dict(options,**changed))
    for command,flags in (("energy",["--density-matrices","missing.csv"]),
                          ("periodic-coulomb",["--xc","pbe"]),
                          ("periodic-coulomb",["--forces"])):
        result=subprocess.run([os.environ["DFT_RS_CLI"],command,str(tmp_path/"missing.xyz"),*flags],
                              capture_output=True,text=True)
        assert result.returncode!=0 and "No such file" not in result.stderr
