# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Mixed magnetic energy derivatives use the common magnetizability report."""
import json
import os
import subprocess
import numpy as np
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
import dft_rs


def test_mixed_magnetic_energy_native_ase_and_cli(tmp_path):
    positions=[[0,0,-0.7],[0,0,0.7]]
    directions=[[0.7,0.1,0.0],[0.7,0.1,0.0],[0.1,0.4,0.2],[0.1,0.4,0.2]]
    controls=dict(xc="hf",basis="sto-3g",gauge="london",magnetic_directions=directions,
                  conv_tol=1e-12,conv_tol_grad=1e-12,response_tolerance=1e-12)
    direct=dft_rs.magnetizability([1,1],positions,**controls)
    generic=dft_rs.run_feature("magnetizability",[1,1],positions,**controls)
    atoms=Atoms("H2",positions=np.asarray(positions)*Bohr)
    atomic=dft_rs.DftRs().get_magnetizability(atoms,units="atomic",**controls)
    physical=dict(controls,conv_tol=controls["conv_tol"]*Hartree,conv_tol_grad=controls["conv_tol_grad"]*Hartree)
    adapted=dft_rs.DftRs().get_magnetizability(atoms,units="ase",**physical)
    native_bohr=1.0/dft_rs.native.BOHR_PER_ANGSTROM
    xyz=tmp_path/"hydrogen.xyz"
    xyz.write_text("2\nmixed magnetic derivative\n"+"".join(
        f"H {x*native_bohr:.16f} {y*native_bohr:.16f} {z*native_bohr:.16f}\n" for x,y,z in positions),encoding="utf-8")
    argv=[os.environ["DFT_RS_CLI"],"magnetizability",str(xyz),"--format","json"]
    for name,value in controls.items():argv +=["--"+name.replace("_","-"),value if isinstance(value,str) else json.dumps(value)]
    process=subprocess.run(argv,text=True,capture_output=True)
    assert process.returncode==0,process.stderr
    cli=json.loads(process.stdout)
    assert direct["converged"] and direct["magnetic_mixed_order"]==4
    assert abs(direct["magnetic_mixed_derivative"])>1e-7
    for report in (generic,atomic,cli):
        np.testing.assert_allclose(report["magnetic_mixed_derivative"],direct["magnetic_mixed_derivative"],rtol=1e-8,atol=1e-11)
        assert report["units"]==direct["units"]
    np.testing.assert_allclose(adapted["magnetic_mixed_derivative"],direct["magnetic_mixed_derivative"]*Hartree,rtol=1e-8,atol=1e-10)
    assert direct["magnetic_mixed_response_residual"]<1e-10
    assert direct["magnetic_mixed_projector_defect"]<1e-10
    assert not any(unit=="unspecified" for unit in direct["units"].values())
    print("mixed magnetic raw",{key:direct[key] for key in ("magnetic_mixed_derivative","magnetic_mixed_roundoff",
          "magnetic_mixed_response_residual","magnetic_mixed_projector_defect","magnetic_mixed_stationarity_defect")})
