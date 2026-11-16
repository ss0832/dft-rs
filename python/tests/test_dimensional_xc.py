# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Actual dimensional XC routes, semilocal conventions and variational fits."""
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, periodic_basis, periodic_capabilities, periodic_xc_operators

CELL = np.array([[5.2,.5,.3],[.8,5.6,.4],[.3,.2,6.]])
POSITIONS = np.array([[.2,-.1,.3],[1.6,.5,-.4]])

def inputs(dimension, name, *, basis_cutoff=12., grid_cutoff=12., grid_level=4):
    pbc = [i<dimension for i in range(3)]
    basis = periodic_basis([1,1],POSITIONS,CELL,pbc=pbc,k_mesh=[3,1,1],basis_cutoff=basis_cutoff)
    overlap = np.array(basis["overlap_real"])+1j*np.array(basis["overlap_imag"])
    total = np.tile(np.array([[.6,.2+.07j],[.2-.07j,.5]]),(3,1,1))
    count = np.einsum("kij,kji->",total,overlap).real/3
    density = np.array([.65*total/count,.35*total/count])
    return dict(density_real=density.real,density_imag=density.imag,pbc=pbc,k_mesh=[3,1,1],
                basis_cutoff=basis_cutoff,grid_cutoff=grid_cutoff,grid_level=grid_level,charge=1,xc=name)

@pytest.mark.parametrize("dimension",[1,2,3])
@pytest.mark.parametrize("name",["svwn5","pbe","r2scan"])
def test_actual_native_ase_cli_potentials_and_component_reports(dimension,name,tmp_path):
    options = inputs(dimension,name)
    report = periodic_xc_operators([1,1],POSITIONS,CELL,**options)
    assert report["shape"]==[2,3,2,2] and report["formal_charge"]==1
    assert report["converged"] and report["grid_number_defect"]<1e-5
    assert report["electron_count"]==pytest.approx(1.,abs=1e-12)
    assert report["hermiticity_defect"]<1e-11 and report["component_energy_defect"]<1e-11
    assert sum(report["component_energies"])==pytest.approx(report["energy"],abs=1e-12)
    assert np.max(np.abs(report["potential_imag"]))>1e-4
    atoms = Atoms("HH",positions=POSITIONS*Bohr,cell=CELL*Bohr,pbc=options["pbc"])
    adapted = DftRs(units="atomic").get_periodic_xc_operators(atoms,**{k:v for k,v in options.items() if k!="pbc"})
    for key in ("potential_real","potential_imag","component_energies","energy","grid_electron_count",
                "electron_count","charge_residual","grid_number_defect","hermiticity_defect","component_energy_defect"):
        np.testing.assert_allclose(report[key],adapted[key],atol=1e-12,rtol=0)
    assert adapted["units"]==report["units"] and adapted["model"]==report["model"]
    executable = os.environ.get("DFT_RS_CLI")
    assert executable,"DFT_RS_CLI is required for actual CLI acceptance"
    conversion = .529177210903
    xyz = tmp_path/"xc.xyz"
    xyz.write_text("2\nsemilocal XC operators\n"+"".join(
        "H "+" ".join(f"{v*conversion:.17g}" for v in row)+"\n" for row in POSITIONS))
    files=[]
    for spin in range(2):
        path = tmp_path/f"spin-{spin}.csv"
        path.write_text("k,row,col,real,imag\n"+"".join(
            f"{k},{i},{j},{options['density_real'][spin,k,i,j]:.17g},{options['density_imag'][spin,k,i,j]:.17g}\n"
            for k in range(3) for i in range(2) for j in range(2)))
        files.append(path)
    command=[executable,"periodic-xc",str(xyz),"--cell",
        ";".join(",".join(f"{v*conversion:.17g}" for v in row) for row in CELL),
        "--pbc",",".join(str(int(v)) for v in options["pbc"]),"--charge","1","--xc",name,
        "--basis","sto-3g","--k-mesh","3,1,1","--overlap-cutoff",str(options["basis_cutoff"]),
        "--basis-grid-cutoff",str(options["grid_cutoff"]),"--grid-level",str(options["grid_level"]),
        "--alpha-density-matrices",str(files[0]),"--beta-density-matrices",str(files[1]),
        "--log-mode","none"]
    result=subprocess.run(command,capture_output=True,text=True,check=True)
    rows=[line.split() for line in result.stdout.splitlines() if line]
    potentials=[row for row in rows if row[0]=="potential"]
    assert len(potentials)==24
    for row in potentials:
        index=tuple(map(int,row[1:5]))
        assert float(row[5])==pytest.approx(np.asarray(report["potential_real"])[index],abs=1e-12)
        assert float(row[6])==pytest.approx(np.asarray(report["potential_imag"])[index],abs=1e-12)
    for key in ("energy","electron_count","grid_electron_count","expected_electrons","charge_residual",
                "grid_number_defect","hermiticity_defect","component_energy_defect"):
        assert float(next(row[1] for row in rows if row[0]==key))==pytest.approx(report[key],abs=1e-12)
    for row in rows:
        if row[0]=="component_energy":
            assert float(row[2])==pytest.approx(report["component_energies"][int(row[1])],abs=1e-12)
    assert report["model"] in result.stdout and report["units"] in result.stdout
    assert "formal_charge 1 converged true" in result.stdout

@pytest.mark.parametrize("dimension",[1,2,3])
@pytest.mark.parametrize("name",["svwn5","pbe","r2scan"])
def test_public_density_direction_has_four_step_variational_fit(dimension,name):
    options=inputs(dimension,name)
    exact=periodic_xc_operators([1,1],POSITIONS,CELL,**options)
    density=options["density_real"]+1j*options["density_imag"]
    delta=.2*density
    potential=np.asarray(exact["potential_real"])+1j*np.asarray(exact["potential_imag"])
    analytic=np.einsum("skij,skji->",delta,potential).real/3
    steps=np.array([.04,.02,.01,.005])
    values=[]
    for h in steps:
        plus,minus=density+h*delta,density-h*delta
        a=periodic_xc_operators([1,1],POSITIONS,CELL,**dict(options,density_real=plus.real,density_imag=plus.imag))
        b=periodic_xc_operators([1,1],POSITIONS,CELL,**dict(options,density_real=minus.real,density_imag=minus.imag))
        values.append((a["energy"]-b["energy"])/(2*h))
    errors=np.abs(np.array(values)-analytic)
    assert np.all(errors>0) and np.all(np.isfinite(errors))
    p,intercept=np.polyfit(np.log(steps),np.log(errors),1)
    r2=1-np.sum((np.log(errors)-intercept-p*np.log(steps))**2)/np.sum((np.log(errors)-np.mean(np.log(errors)))**2)
    print(f"{dimension}D public {name} XC variational: errors={errors.tolist()} density steps=.04..005 p={p:.12f} R2={r2:.12f}")
    assert abs(p-2)<.02 and r2>.99999
    np.testing.assert_allclose(errors[:-1]/errors[1:],4,rtol=.03,atol=0)
    assert abs((4*values[-1]-values[-2])/3-analytic)<1e-8

def test_semilocal_scope_hybrid_nonlocal_and_particle_diagnostics_are_explicit():
    assert periodic_capabilities()["dimensional_xc_operators"]
    options=inputs(2,"pbe0")
    hybrid=periodic_xc_operators([1,1],POSITIONS,CELL,**options)
    assert hybrid["exact_exchange_coefficients"]==pytest.approx([.25,0.,0.])
    tail=periodic_xc_operators([1,1],POSITIONS,CELL,**dict(options,xc="wb97mv"))
    assert tail["required_nonlocal"]=="VV10" and "semilocal" in tail["model"]
    hf=periodic_xc_operators([1,1],POSITIONS,CELL,**dict(options,xc="hf"))
    assert hf["energy"]==0. and not hf["component_energies"]
    assert np.max(np.abs(hf["potential_real"]))==0.
    wrong=periodic_xc_operators([1,1],POSITIONS,CELL,**dict(options,density_real=options["density_real"]*1.2,
                                                               density_imag=options["density_imag"]*1.2))
    assert not wrong["converged"] and abs(wrong["charge_residual"])>.1

def test_input_contracts_reject_invalid_spin_density_and_unused_controls(tmp_path):
    options=inputs(1,"pbe")
    for changes in ({"density_real":[[[[0.]]]]}, {"density_imag":[[[[0.]]]]},
                    {"basis_cutoff":-1.}, {"pbc":[True,False,False],"k_mesh":[3,2,1]},
                    {"forces":True}, {"temperature":100.}, {"operator_tolerance":float("nan")}):
        with pytest.raises((ValueError,TypeError)):
            periodic_xc_operators([1,1],POSITIONS,CELL,**dict(options,**changes))
    for command,flags in (("periodic-xc",["--forces"]), ("periodic-xc",["--density-matrices","missing.csv"]),
                          ("energy",["--alpha-density-matrices","missing.csv"]),
                          ("periodic-xc",["--cell","4,4,4"])):
        result=subprocess.run([os.environ["DFT_RS_CLI"],command,str(tmp_path/"missing.xyz"),*flags],capture_output=True,text=True)
        assert result.returncode!=0 and "No such file" not in result.stderr

def test_underresolved_grid_reports_its_error_without_changing_the_metric_or_threshold():
    options=inputs(3,"pbe",basis_cutoff=8.,grid_cutoff=10.,grid_level=3)
    report=periodic_xc_operators([1,1],POSITIONS,CELL,**options)
    # Value of the grid with the cell-volume-scaled Becke neighbourhood.
    assert report["grid_number_defect"]==pytest.approx(.000713514437052809,abs=1e-12)
    assert report["grid_tolerance"]==1e-5 and not report["converged"]
    assert abs(report["charge_residual"])<1e-11 and report["electron_count"]==pytest.approx(1.,abs=1e-12)
    assert any("refine" in warning for warning in report["warnings"])
