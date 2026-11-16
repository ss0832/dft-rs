# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Finite complex orbital references use one driver on every public route."""
import json
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
import dft_rs

@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_complex_periodic_reference_native_ase_cli_and_time_reversal(dimension, tmp_path):
    cell=np.eye(3)*40.0
    pos=np.array([[0.3,-0.1,0.2]])
    pbc=[i<dimension for i in range(3)]
    settings=dict(xc="hf",basis="cc-pvdz",multiplicity=2,k_mesh=[1,1,1],
        overlap_cutoff=4.3,exchange_cutoff=4.3,grid_cutoff=10.0,
        ewald_alpha=.55,ewald_real_cutoff=16.0,ewald_reciprocal_cutoff=8.0,
        field_wave_indices=[1,0,0],field_vector_real=[0.0,.37,0.0],
        field_vector_imag=[0.0,.12,0.0],field_amplitude=.4,
        conv_tol=1e-13,conv_tol_dp=1e-12,conv_tol_grad=1e-11,max_iter=150)
    native=dft_rs.periodic_field_scf([1],pos,cell,pbc=pbc,**settings)
    generic=dft_rs.run_feature("periodic-field-scf",[1],pos,cell=cell,pbc=pbc,**settings)
    atoms=Atoms("H",positions=pos*Bohr,cell=cell*Bohr,pbc=pbc)
    ase=dft_rs.DftRs(units="atomic").get_periodic_field_scf(atoms,**settings, units="atomic")
    path=tmp_path/"field.xyz"
    path.write_text("1\nstatic periodic field\nH "+" ".join(f"{x*.529177210903:.17g}" for x in pos[0])+"\n",encoding="utf-8")
    args=[os.environ["DFT_RS_CLI"],"periodic-field-scf",str(path),"--cell",
        ";".join(",".join(f"{v*.529177210903:.17g}" for v in row) for row in cell),
        "--pbc",",".join(str(int(x)) for x in pbc),"--format","json"]
    for key,value in settings.items():
        if isinstance(value,list): value=",".join(map(str,value))
        args.extend(["--"+key.replace("_","-"),str(value)])
    process=subprocess.run(args,capture_output=True,text=True,check=True)
    cli=json.loads(process.stdout)
    keys=["energy","free_energy","external_field_energy","field_energy_derivative",
          "density_real","density_imag","fock_real","fock_imag",
          "orbital_density_residual","orbital_fock_residual","spinful_density_residual"]
    for report in [generic,ase,cli]:
        assert report["converged"] and report["model"]==native["model"]
        assert report["units"]==native["units"]
        for key in keys: np.testing.assert_allclose(native[key],report[key],rtol=1e-9,atol=1e-10)
    assert not native["orbital_conjugation_constraint"]
    assert max(native["orbital_density_residual"])>1e-4
    assert native["orbital_residual"]<1e-10
    assert native["particle_number_error"]<1e-12
    assert native["component_energy_defect"]<1e-12
    reverse=dft_rs.periodic_field_scf([1],pos,cell,pbc=pbc,**(settings|dict(field_amplitude=-.4)))
    np.testing.assert_allclose(native["energy"],reverse["energy"],rtol=0,atol=1e-11)
    np.testing.assert_allclose(native["density_real"],reverse["density_real"],rtol=0,atol=1e-10)
    np.testing.assert_allclose(native["density_imag"],-np.asarray(reverse["density_imag"]),rtol=0,atol=1e-10)
    print("complex reference",dimension,native["orbital_density_residual"],native["orbital_residual"],native["component_energy_defect"])
