# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""General spin coherence and common-driver route acceptance."""
import json
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
import dft_rs

@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_spinor_scf_native_ase_cli_and_closed_zeeman_oracle(dimension, tmp_path):
    # One spatial AO gives an independent 2x2 Zeeman oracle. Two real
    # Fourier conjugate modes at q=0 give H=amplitude*sigma.B.
    cell=np.eye(3)*40.0
    positions=np.zeros((1,3))
    pbc=[axis<dimension for axis in range(3)]
    # Unused open-axis vectors need not create an artificial 3D volume.
    cell[dimension:]=0.0
    spin=np.array([.03,.04,.05])
    settings=dict(xc="hf",basis="sto-3g",k_mesh=[1,1,1],overlap_cutoff=4.3,
        exchange_cutoff=4.3,grid_cutoff=8.0,ewald_alpha=.55,
        ewald_real_cutoff=16.0,ewald_reciprocal_cutoff=8.0,
        field_wave_indices=[0,0,0],spin_field_real=spin.tolist(),field_amplitude=.4,
        conv_tol=1e-13,conv_tol_dp=1e-12,conv_tol_grad=1e-11,max_iter=150)
    native=dft_rs.periodic_spinor_scf([1],positions,cell,pbc=pbc,**settings)
    generic=dft_rs.run_feature("periodic-spinor-scf",[1],positions,cell=cell,pbc=pbc,**settings)
    atoms=Atoms("H",positions=positions*Bohr,cell=cell*Bohr,pbc=pbc)
    ase=dft_rs.DftRs(units="atomic").get_periodic_spinor_scf(atoms,**settings, units="atomic")
    xyz=tmp_path/"spinor.xyz"
    xyz.write_text("1\nspinor reference\nH 0 0 0\n",encoding="utf-8")
    args=[os.environ["DFT_RS_CLI"],"periodic-spinor-scf",str(xyz),"--cell",
          ";".join(",".join(f"{v*.529177210903:.17g}" for v in row) for row in cell),
          "--pbc",",".join(str(int(v)) for v in pbc),"--format","json"]
    for key,value in settings.items():
        if isinstance(value,list): value=",".join(map(str,value))
        args.extend(["--"+key.replace("_","-"),str(value)])
    cli=json.loads(subprocess.run(args,capture_output=True,text=True,check=True).stdout)
    for report in (native,generic,ase,cli):
        assert report["converged"]
        assert not report["separate_spin_population_constraint"]
        assert not report["orbital_conjugation_constraint"]
        assert report["particle_number_error"]<1e-12
        assert report["occupation_consistency"]<1e-12
        assert report["orbital_residual"]<1e-11
        assert report["units"]==native["units"] and report["model"]==native["model"]
        for key in ("energy","free_energy","field_energy_derivative","magnetization",
                    "density_real","density_imag","fock_real","fock_imag"):
            np.testing.assert_allclose(report[key],native[key],rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(native["magnetization"],-spin/np.linalg.norm(spin),rtol=0,atol=1e-11)
    np.testing.assert_allclose(native["field_energy_derivative"],-np.linalg.norm(spin),rtol=0,atol=1e-12)
    np.testing.assert_allclose(native["energy_components"]["spin_soc"],-.4*np.linalg.norm(spin),rtol=0,atol=1e-12)
    np.testing.assert_allclose(native["energy"],sum(native["energy_components"].values()),rtol=0,atol=1e-13)
    assert abs(np.asarray(native["density_real"])[0,0,1])>.1
    assert abs(np.asarray(native["density_imag"])[0,0,1])>.1

def test_general_spinor_operator_shape_and_hermiticity_are_validated():
    cell=np.eye(3)*40.0
    common=dict(xc="hf",basis="sto-3g",pbc=[True,False,False],field_wave_indices=[0,0,0])
    with pytest.raises(ValueError,match="flattened"):
        dft_rs.periodic_spinor_scf([1],[[0,0,0]],cell,spinor_operator_real=[0.0],spinor_operator_imag=[0.0],**common)
    with pytest.raises(ValueError,match="Hermitian"):
        dft_rs.periodic_spinor_scf([1],[[0,0,0]],cell,spinor_operator_real=[0,1,0,0],spinor_operator_imag=[0]*4,**common)


@pytest.mark.parametrize("xc", ["blyp", "m06l"])
def test_generalized_periodic_field_scf_uses_complex_reference(xc, tmp_path):
    cell = [[14.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    positions = [[0.18, -0.11, 0.23]]
    pbc = [True, False, False]
    spin = [0.03, 0.04, 0.05]
    options = dict(xc=xc, reference="gks", basis="sto-3g", k_mesh=[1, 1, 1],
                   overlap_cutoff=7.5, exchange_cutoff=7.5, grid_cutoff=10.0, xc_ao_cutoff=8.0,
                   grid_level=3, grid_weight_cutoff=0.0, field_wave_indices=[0, 0, 0],
                   spin_field_real=spin, field_amplitude=0.4, conv_tol=1e-12,
                   conv_tol_dp=1e-10, conv_tol_grad=1e-10)
    canonical = dft_rs.periodic_spinor_scf(
        [2], positions, cell, pbc=pbc, nuclear_soc_scale=0.0, **options,
    )
    assert canonical["converged"]
    native = dft_rs.periodic_field_scf([2], positions, cell, pbc=pbc, **options)
    atoms = Atoms("He", positions=np.asarray(positions) * Bohr, cell=np.asarray(cell) * Bohr, pbc=pbc)
    ase = dft_rs.DftRs(units="atomic").get_periodic_field_scf(atoms, **options)
    geometry = tmp_path / "wire.xyz"
    geometry.write_text("1\nHelium chain\nHe " + " ".join(f"{x:.17g}" for x in atoms.positions[0])
                        + "\n", encoding="ascii")
    command = [os.environ["DFT_RS_CLI"], "periodic-field-scf", str(geometry),
               "--cell", f"{14.0 * Bohr:.17g},0,0;0,0,0;0,0,0", "--pbc", "1,0,0",
               "--format", "json"]
    for key, value in options.items():
        command.extend(("--" + key.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)))
    execution = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "[spinor-scf]" in execution.stderr and "residual" in execution.stderr
    cli = json.loads(execution.stdout)
    for result in (native, ase, cli):
        assert result["feature"] == "periodic-field-scf"
        assert result["reference_type"] == "gks"
        assert result["converged"]
        assert result["orbital_residual"] < 1e-10
        assert abs(result["particle_number_error"]) < 1e-10
        assert not result["separate_spin_population_constraint"]
        for key in ("energy", "orbital_residual", "field_energy_derivative", "magnetization",
                    "density_real", "density_imag", "fock_real", "fock_imag"):
            assert result["units"][key] == canonical["units"][key]
            np.testing.assert_allclose(result[key], canonical[key], rtol=0.0, atol=1e-10)
    # A filled pair has zero static spin moment. Its Pauli-y operator
    # still supplies a nonzero imaginary off-diagonal Fock element.
    np.testing.assert_allclose(native["magnetization"], np.zeros(3), rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(native["fock_imag"][0][0][1], -0.4 * spin[1], rtol=0.0, atol=1e-10)
    print(f"{xc} helium chain energy={native['energy']:.15f} Eh, "
          f"orbital residual={native['orbital_residual']:.6e} Eh, "
          f"field derivative={native['field_energy_derivative']:.12e}")


def test_fully_polarized_m06l_wire_retains_the_empty_spin_channel():
    cell = [[30.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    positions = [[0.18, -0.11, 0.23]]
    spin = np.array([0.03, 0.04, 0.05])
    options = dict(xc="m06l", reference="gks", basis="sto-3g", k_mesh=[1, 1, 1],
                   overlap_cutoff=7.5, exchange_cutoff=7.5, grid_cutoff=10.0, xc_ao_cutoff=8.0,
                   grid_level=3, grid_weight_cutoff=0.0, field_wave_indices=[0, 0, 0],
                   spin_field_real=spin.tolist(), conv_tol=1e-12, conv_tol_dp=1e-10,
                   conv_tol_grad=1e-10, pbc=[True, False, False])
    weak, strong = (dft_rs.periodic_field_scf([1], positions, cell, field_amplitude=amplitude, **options)
                    for amplitude in (0.3, 0.4))
    for report in (weak, strong):
        assert report["converged"]
        assert report["orbital_residual"] < 1e-10
        assert report["xc_grid_number_defect"] < 1e-5
        np.testing.assert_allclose(report["magnetization"], -spin / np.linalg.norm(spin), rtol=0.0, atol=1e-10)
    # One spatial AO leaves only the closed-form 2x2 Pauli energy shift.
    np.testing.assert_allclose(strong["energy"] - weak["energy"], -0.1 * np.linalg.norm(spin),
                               rtol=0.0, atol=1e-11)
    print(f"M06-L hydrogen wire energy={strong['energy']:.15f} Eh, "
          f"orbital residual={strong['orbital_residual']:.6e} Eh")
