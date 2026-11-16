# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Screened and double-hybrid energies on native, generic, ASE and CLI routes."""
import json
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
import dft_rs

POSITIONS=[[0.0,0.0,-0.7],[0.0,0.0,0.7]]
BASE=dict(xc="b2plyp",basis="sto-3g",grid_level=5,conv_tol=1e-13,conv_tol_grad=1e-10,conv_tol_dp=1e-12)


def test_hsesol_energy_agrees_on_public_routes(tmp_path):
    controls = dict(BASE, xc="hsesol", grid_level=9, grid_weight_cutoff=0.0)
    native = dft_rs.energy([1, 1], POSITIONS, **controls)
    generic = dft_rs.run_feature("energy", [1, 1], POSITIONS, **controls)
    atoms = Atoms("H2", positions=np.asarray(POSITIONS) * Bohr)
    calculator = dft_rs.DftRs(**controls)
    adapted = calculator.get_energy_report(atoms, units="atomic")
    xyz = tmp_path / "hydrogen.xyz"
    xyz.write_text(f"2\nHSEsol\nH 0 0 {-0.7 * Bohr:.16f}\nH 0 0 {0.7 * Bohr:.16f}\n", encoding="utf-8")
    result = subprocess.run(
        [os.environ["DFT_RS_CLI"], "energy", str(xyz), "--xc", "hsesol", "--basis", "sto-3g",
         "--grid-level", "9", "--grid-weight-cutoff", "0", "--conv-tol", "1e-13",
         "--conv-tol-grad", "1e-10", "--conv-tol-dp", "1e-12", "--format", "json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "[scf] iter" in result.stderr and "SCF residual" in result.stderr
    cli = json.loads(result.stdout)
    assert native["converged"]
    np.testing.assert_allclose(native["energy"], -1.140838811287166, rtol=0, atol=5e-7)
    for other in (generic, adapted, cli):
        assert other["units"] == native["units"]
        np.testing.assert_allclose(other["energy"], native["energy"], rtol=0, atol=1e-12)
        assert other["scf_iterations"] == native["scf_iterations"]
    atoms.calc = calculator
    np.testing.assert_allclose(atoms.get_potential_energy(), native["energy"] * Hartree, rtol=0, atol=1e-10)


def test_double_hybrid_energy_and_terms_agree_on_all_routes(tmp_path):
    native=dft_rs.energy([1,1],POSITIONS,**BASE)
    generic=dft_rs.run_feature("energy",[1,1],POSITIONS,**BASE)
    atoms=Atoms("H2",positions=np.asarray(POSITIONS)*Bohr)
    adapted=dft_rs.DftRs(units="atomic",**BASE).get_energy_report(atoms,units="atomic")
    xyz=tmp_path/"hydrogen.xyz"
    xyz.write_text("2\ndouble hybrid\n"+"".join(f"H 0 0 {p[2]/dft_rs.native.BOHR_PER_ANGSTROM:.16f}\n" for p in POSITIONS),encoding="utf-8")
    result=subprocess.run([os.environ["DFT_RS_CLI"],"energy",str(xyz),"--xc","b2plyp",
        "--basis","sto-3g","--grid-level","5","--conv-tol","1e-13","--conv-tol-grad","1e-10",
        "--conv-tol-dp","1e-12","--format","json"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    cli=json.loads(result.stdout)
    keys=["energy","reference_energy","pt2_same_spin_alpha","pt2_same_spin_beta","pt2_opposite_spin",
          "pt2_correlation","one_electron_energy","hartree_energy","exact_exchange_energy",
          "nonsemilocal_exchange_correlation_energy"]
    assert native["converged"]
    for other in [generic,adapted,cli]:
        assert other["units"]==native["units"]
        for key in keys:
            np.testing.assert_allclose(other[key],native[key],rtol=1e-10,atol=1e-12)
    calculated=dft_rs.calculate([1,1],POSITIONS,**BASE)
    np.testing.assert_allclose(calculated["energy"],native["energy"],atol=1e-12,rtol=0)
    atoms.calc=dft_rs.DftRs(**BASE)
    np.testing.assert_allclose(atoms.get_potential_energy(),native["energy"]*Hartree,atol=1e-10,rtol=1e-10)
    plain=subprocess.run([os.environ["DFT_RS_CLI"],"energy",str(xyz),"--xc=b2plyp","--basis","sto-3g"],capture_output=True,text=True)
    assert plain.returncode==0,plain.stderr
    assert "pt2_opposite_spin" in plain.stdout
    np.testing.assert_allclose(native["energy"],-1.1490167435786924,atol=1e-6,rtol=0)
    np.testing.assert_allclose(native["pt2_opposite_spin"],-0.016279084164712576,atol=1e-6,rtol=0)
    assert native["pt2_same_spin_alpha"]==native["pt2_same_spin_beta"]==0.0
    assert abs(native["nonsemilocal_exchange_correlation_energy"]) < 1e-12
    for key in ["pt2_amplitude_residual","pt2_lagrangian_residual","reference_energy_sum_residual","semilocal_component_sum_residual"]:
        assert native[key]<1e-9
    converted=dft_rs.DftRs(**BASE).get_energy_report(atoms)
    for key in keys:
        np.testing.assert_allclose(converted[key],native[key]*Hartree,rtol=1e-10,atol=1e-10)
    print("double-hybrid route energy",native["energy"],"SS",native["pt2_same_spin_alpha"]+native["pt2_same_spin_beta"],
          "OS",native["pt2_opposite_spin"],"amplitude residual",native["pt2_amplitude_residual"])


@pytest.mark.parametrize("xc",["pbe","tpss","pbe0","m06"])
def test_custom_parent_zero_pt2_limit_uses_the_actual_functional(xc):
    report=dft_rs.energy([1,1],POSITIONS,xc=xc,
        pt2_same_spin=0.0,pt2_opposite_spin=0.0,basis="sto-3g",conv_tol=1e-11)
    parent=dft_rs.energy([1,1],POSITIONS,xc=xc,basis="sto-3g",conv_tol=1e-11)
    assert report["energy"]==report["reference_energy"]
    np.testing.assert_allclose(report["energy"],parent["energy"],atol=1e-11,rtol=0)
    assert report["reference_energy_sum_residual"]<1e-9
    correlated=dft_rs.energy([1,1],POSITIONS,xc=xc,pt2_same_spin=.2,pt2_opposite_spin=.3,
        basis="sto-3g",conv_tol=1e-11)
    assert correlated["pt2_correlation"]<0
    np.testing.assert_allclose(correlated["reference_energy"],parent["energy"],atol=1e-11,rtol=0)
    np.testing.assert_allclose(correlated["energy"]-correlated["reference_energy"],
        .2*(correlated["pt2_same_spin_alpha"]+correlated["pt2_same_spin_beta"])+.3*correlated["pt2_opposite_spin"],atol=1e-14,rtol=0)


def test_double_hybrid_invalid_controls_fail_explicitly():
    for controls in [dict(double_hybrid_model="custom"),
                     dict(electronic_temperature=300),dict(double_hybrid_model="b2plyp",xc="pbe"),dict(double_hybrid_model="invalid")]:
        with pytest.raises(ValueError):
            dft_rs.energy([1,1],POSITIONS,**(dict(xc="b2plyp") | controls))


def test_pt2_memory_limit_uses_the_existing_runtime_error_contract():
    with pytest.raises(RuntimeError,match="configured 0 MiB"):
        dft_rs.energy([1,1],POSITIONS,xc="b2plyp",pt2_memory_mb=0)


@pytest.mark.parametrize("xc,expected",[("dsd-pbep86",-1.1426902547663869),("dsd-blyp",-1.1444485389908157),
    ("pwpb95",-1.1484603976848788),("wb97m2",-1.1485633685002774),
    ("dsd-pbeb95",-1.1409486729303697),
    ("wb2plyp",-1.1374739594617378),("wb2gp-plyp",-1.1373291663276985),
    ("rsx-qidh",-1.1425991960589896),("rsx-0dh",-1.1460099688706236),
    ("wb97x-2-lp",-1.147648039793244),("wb97x-2-tqz",-1.143857540697865),
    ("xdh-pbe0",-1.148926066046099)])
def test_named_recipes_share_the_existing_energy_api(tmp_path,xc,expected):
    controls=dict(BASE,xc=xc,vv10_grid_level=1)
    report=dft_rs.energy([1,1],POSITIONS,**controls)
    calculated=dft_rs.calculate([1,1],POSITIONS,**controls)
    atoms=Atoms("H2",positions=np.asarray(POSITIONS)*Bohr)
    atoms.calc=dft_rs.DftRs(**controls)
    adapted=atoms.calc.get_energy_report(atoms,units="atomic")
    np.testing.assert_allclose(report["energy"],expected,atol=1e-6,rtol=0)
    for other in [calculated,adapted]:
        for key in ["energy","pt2_opposite_spin","model_dispersion_energy"]:
            np.testing.assert_allclose(other[key],report[key],atol=1e-12,rtol=0)
    np.testing.assert_allclose(atoms.get_potential_energy(),report["energy"]*Hartree,atol=1e-10,rtol=0)
    xyz=tmp_path/"hydrogen.xyz"
    xyz.write_text("2\nDSD\n"+"".join(f"H 0 0 {p[2]/dft_rs.native.BOHR_PER_ANGSTROM:.16f}\n" for p in POSITIONS),encoding="utf-8")
    result=subprocess.run([os.environ["DFT_RS_CLI"],"energy",str(xyz),"--xc",xc,"--grid-level","5",
        "--conv-tol","1e-13","--conv-tol-grad","1e-10","--conv-tol-dp","1e-12",
        "--vv10-grid-level","1","--format","json"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    cli=json.loads(result.stdout)
    for key in ["energy","pt2_opposite_spin","model_dispersion_energy"]:
        np.testing.assert_allclose(cli[key],report[key],atol=1e-12,rtol=0)
    if xc=="wb97m2":
        assert report["distinct_target_energy"]
        assert report["target_vv10_b"]==10.0
        assert report["target_vv10_scale"]==.65904
        assert report["functional"]!=report["orbital_functional"]
        assert report["model_dispersion_energy"]==0.0
        assert report["energy_components"]["nonlocal_correlation"]==report["target_nonlocal_correlation"]
    elif xc in ["wb2plyp", "wb2gp-plyp", "rsx-qidh", "rsx-0dh", "wb97x-2-lp", "wb97x-2-tqz", "xdh-pbe0"]:
        assert report["model_dispersion"] is None
        assert report["model_dispersion_energy"]==0.0
    else:
        assert report["model_dispersion"]["model"]=="D3(BJ), pairwise"
        assert report["model_dispersion_energy"]<0
    assert report["pt2_amplitude_residual"]<1e-12
