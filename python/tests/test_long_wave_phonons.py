# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Analytic Fourier force-constant moments on the existing phonon API."""
import json
import os
import subprocess

import numpy as np
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
import dft_rs


def test_long_wave_moments_native_ase_cli_and_phase_convention(tmp_path):
    positions=np.array([[0.0,0.0,-0.7],[0.0,0.0,0.7]])
    cell=np.eye(3)*9.0
    controls=dict(xc="hf",basis="sto-3g",k_mesh=[1,1,1],overlap_cutoff=4.0,
                  grid_level=0,supercell=[1,1,1],path=[[0,0,0],[0.1,0.0,0.0]],points=2,
                  phonon_phase="atomic",wavevector_direction=[0.2,-0.1,0.3],response_order=4,
                  conv_tol=1e-12,conv_tol_grad=1e-12,conv_tol_dp=1e-12,response_tolerance=1e-12)
    native=dft_rs.phonon_bands([1,1],positions.tolist(),cell.tolist(),**controls)
    atoms=Atoms("H2",positions=positions*Bohr,cell=cell*Bohr,pbc=True)
    atomic=dft_rs.DftRs().get_phonon_bands(atoms,units="atomic",**controls)
    physical=dict(controls,overlap_cutoff=controls["overlap_cutoff"]*Bohr,
                  wavevector_direction=np.asarray(controls["wavevector_direction"])/Bohr,
                  conv_tol=controls["conv_tol"]*Hartree,conv_tol_grad=controls["conv_tol_grad"]*Hartree)
    adapted=dft_rs.DftRs().get_phonon_bands(atoms,units="ase",**physical)
    # CLI geometry uses the core's CODATA convention; ASE geometry uses ASE's.
    native_bohr=1.0/dft_rs.native.BOHR_PER_ANGSTROM
    xyz=tmp_path/"hydrogen.xyz"
    xyz.write_text("2\nlong-wave force constants\n"+"".join(
        f"H {x*native_bohr:.16f} {y*native_bohr:.16f} {z*native_bohr:.16f}\n" for x,y,z in positions),encoding="utf-8")
    argv=[os.environ["DFT_RS_CLI"],"phonon-bands",str(xyz),"--cell",json.dumps((cell*native_bohr).tolist()),"--format","json"]
    for name,value in controls.items():argv +=["--"+name.replace("_","-"),value if isinstance(value,str) else json.dumps(value)]
    process=subprocess.run(argv,text=True,capture_output=True)
    assert process.returncode==0,process.stderr
    cli=json.loads(process.stdout)
    assert native["converged"] and native["wavevector_response_order"]==4
    assert native["phonon_phase"]=="atomic"
    for name in ("dynamical_q_coefficients_real","dynamical_q_coefficients_imaginary",
                 "force_constant_q_coefficients_real","force_constant_q_coefficients_imaginary"):
        expected=np.asarray(native[name])
        assert expected.shape==(3,5,6,6)
        for report in (atomic,cli):np.testing.assert_allclose(report[name],expected,rtol=1e-8,atol=1e-11)
        np.testing.assert_allclose(adapted[name],expected*Hartree/Bohr**2,rtol=1e-8,atol=1e-10)
    assert np.max(np.abs(np.asarray(native["force_constant_q_coefficients_real"])[:,2]))>1e-7
    assert np.max(np.abs(native["dynamical_q_hermiticity_defects"]))<1e-10
    assert not any(unit=="unspecified" for unit in native["units"].values())
    print("long-wave raw Hermiticity",native["dynamical_q_hermiticity_defects"])
