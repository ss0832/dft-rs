# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Shared public nonlocal/magnetic paths and SCF stderr progress."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
from ase.units import create_units
import dft_rs


def cli_report(feature, atoms, controls, tmp_path):
    # The adapter and XYZ reader share the package CODATA 2018 factors.
    xyz=tmp_path / "atoms.xyz"
    xyz.write_text(str(len(atoms))+"\npublic response acceptance\n"+"".join(
        f"{symbol} {p[0]:.16f} {p[1]:.16f} {p[2]:.16f}\n"
        for symbol,p in zip(atoms.get_chemical_symbols(),atoms.positions)),encoding="utf-8")
    args=[os.environ["DFT_RS_CLI"],feature,str(xyz),"--format","json"]
    if atoms.pbc.any():
        args.extend(["--cell",";".join(",".join(f"{v:.16f}" for v in row) for row in atoms.cell)])
    for key,value in controls.items():
        args.extend(["--"+key.replace("_","-"),
                     ",".join(map(str,value)) if isinstance(value,list) else str(value).lower()])
    result=subprocess.run(args,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert "[pbc-scf] iter" in result.stderr or "[scf] iter" in result.stderr
    assert "residual" in result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("model",["hf","wb97m-v"])
def test_periodic_nonlocal_energy_shared_surfaces_and_term_labels(model,tmp_path):
    controls=dict(xc=model,basis="sto-3g",k_mesh=[1,1,1],grid_level=0,
                  overlap_cutoff=7.0,grid_shell=1,vv10=True,vv10_cutoff=11.3,
                  conv_tol=1e-12,conv_tol_grad=1e-12,max_iter=250)
    cell=(np.eye(3)*9.0).tolist()
    native=dft_rs.periodic_energy([2],[[0.0,0.0,0.0]],cell,**controls)
    atoms=Atoms("He",positions=[[0.0,0.0,0.0]],cell=np.asarray(cell)*Bohr,pbc=True)
    adapted=dft_rs.DftRs(units="atomic").get_periodic_energy_report(atoms,units="atomic",**controls)
    cli=cli_report("periodic-energy",atoms,controls,tmp_path)
    assert native["converged"]
    assert native["vv10_b"]==pytest.approx(5.9 if model=="hf" else 6.0)
    assert native["vv10_c"]==pytest.approx(0.0093 if model=="hf" else 0.01)
    components=native["energy_components"]
    assert abs(components["nonlocal_correlation"])>1e-10
    assert components["exchange_correlation"]==pytest.approx(
        components["semilocal_exchange_correlation"]+components["nonlocal_correlation"],abs=1e-13)
    for report in (adapted,cli):
        assert report["energy"]==pytest.approx(native["energy"],abs=1e-10)
        assert report["energy_components"]==pytest.approx(components,abs=1e-10)
        assert report["vv10_cutoff"]==11.3
        assert report["units"]["vv10_cutoff"]=="bohr"
        assert report["nonlocal_correlation_model"]==native["nonlocal_correlation_model"]
    converted_controls=dict(controls,overlap_cutoff=7.0*Bohr,vv10_cutoff=11.3*Bohr,
                            conv_tol=controls["conv_tol"]*Hartree,
                            conv_tol_grad=controls["conv_tol_grad"]*Hartree)
    converted=dft_rs.DftRs().get_periodic_energy_report(atoms,**converted_controls)
    assert converted["energy_components"]["nonlocal_correlation"]==pytest.approx(
        components["nonlocal_correlation"]*Hartree,abs=1e-9)
    assert converted["vv10_cutoff"]==pytest.approx(11.3*Bohr)


def test_higher_magnetic_energy_shared_surfaces_and_unit_conventions(tmp_path):
    controls=dict(xc="hf",basis="sto-3g",gauge="london",response_order=4,
                  magnetic_direction=[0.6,-0.2,0.1],response_tolerance=1e-12)
    positions=[[0.0,0.0,-0.7],[0.0,0.0,0.7]]
    native=dft_rs.magnetizability([1,1],positions,**controls)
    atoms=Atoms("H2",positions=np.asarray(positions)*Bohr)
    adapted=dft_rs.DftRs(units="atomic").get_magnetizability(atoms,units="atomic",**controls)
    cli=cli_report("magnetizability",atoms,controls,tmp_path)
    derivatives=native["magnetic_energy_derivatives"]
    assert len(derivatives)==5 and abs(derivatives[4])>1e-8
    assert native["magnetic_response_order"]==4
    assert np.max(np.abs(native["magnetic_path_residuals"]))<1e-10
    for report in (adapted,cli):
        np.testing.assert_allclose(report["magnetic_energy_derivatives"],derivatives,rtol=1e-9,atol=1e-11)
        assert report["units"]==native["units"]
        assert report["magnetic_path_model"]==native["magnetic_path_model"]


def test_cotton_mouton_static_energy_maxwell_contraction_and_dynamic_limit(tmp_path):
    positions=np.array([[0.0,0.0,-0.7],[0.0,0.0,0.7]])
    atoms=Atoms("H2",positions=positions*Bohr)
    controls=dict(xc="hf",basis="sto-3g",gauge="london",temperature=298.15,
                  conv_tol=1e-13,conv_tol_dp=1e-12,response_tolerance=1e-12)
    direct=dft_rs.cotton_mouton([1,1],positions,**controls)
    adapted=dft_rs.DftRs(units="atomic").get_cotton_mouton(atoms,**controls)
    cli=cli_report("cotton-mouton",atoms,controls,tmp_path)
    for key in ("cotton_mouton_constant","orientational_term","hypermagnetizability_anisotropy",
                "polarizability","magnetizability","hypermagnetizability"):
        for report in (adapted,cli):
            np.testing.assert_allclose(report[key],direct[key],rtol=1e-9,atol=1e-10)
    eta=np.asarray(direct["hypermagnetizability"])
    alpha=np.asarray(direct["polarizability"])
    xi=np.asarray(direct["magnetizability"])
    # London phases also give transverse dipole terms at finite magnetic
    # field, even in a two-s-orbital basis. Evaluate every ordered component
    # through the energy response independently of the dipole response.
    independent=np.zeros((3,3,3,3))
    response_controls={k:v for k,v in controls.items() if k!="temperature"}
    axes=np.eye(3).tolist()
    for a,b,c,d in np.ndindex(3,3,3,3):
        mixed=dft_rs.magnetizability([1,1],positions,**response_controls,
            magnetic_directions=[[0,0,0],[0,0,0],axes[c],axes[d]],
            electric_directions=[axes[a],axes[b],[0,0,0],[0,0,0]])
        independent[a,b,c,d]=-mixed["magnetic_mixed_derivative"]
    np.testing.assert_allclose(eta,independent,rtol=1e-9,atol=1e-10)
    invariant=(sum(independent[a,b,a,b] for a in range(3) for b in range(3))
               -sum(independent[a,a,b,b] for a in range(3) for b in range(3))/3)/5
    constants=create_units("2018")
    kT=controls["temperature"]*constants.kB/constants.Hartree
    thermal=2*(np.sum(alpha*xi)-np.trace(alpha)*np.trace(xi)/3)/(15*kT)
    expected=2*np.pi/27*(invariant+thermal)
    np.testing.assert_allclose(direct["cotton_mouton_constant"],expected,rtol=1e-10,atol=1e-10)
    steps=np.array([0.04,0.02,0.01,0.005])
    errors=np.array([abs(dft_rs.cotton_mouton([1,1],positions,**controls,frequency=float(w))[
        "cotton_mouton_constant"]-direct["cotton_mouton_constant"]) for w in steps])
    assert np.all(np.isfinite(errors)) and np.all(errors>0)
    x,y=np.log(steps),np.log(errors)
    p,intercept=np.linalg.lstsq(np.column_stack([x,np.ones(4)]),y,rcond=None)[0]
    quality=1-np.sum((y-p*x-intercept)**2)/np.sum((y-y.mean())**2)
    print("Cotton-Mouton independent energy contraction",expected,"frequency steps",steps.tolist(),
          "errors",errors.tolist(),"p",p,"R2",quality)
    assert abs(p-2)<0.05 and quality>0.999
