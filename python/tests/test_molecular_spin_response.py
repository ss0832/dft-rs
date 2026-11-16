# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Molecular spin-only response through the shared public driver."""
import json
import os
import re
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
import dft_rs

@pytest.mark.parametrize("xc",["svwn5","pbe","tpss","pbe0"])
def test_generic_spin_response_public_routes(tmp_path,xc):
    positions=[[0.,0.,-.7],[0.,0.,.7]]
    options=dict(xc=xc,basis="sto-3g",unrestricted=True,electronic_temperature=150000,
        grid_level=4,dispersion="none",conv_tol=1e-13,conv_tol_grad=1e-11,conv_tol_dp=1e-12,
        spin_g_factor=2.0)
    result=dft_rs.spin_susceptibility([1,1],positions,**options)
    generic=dft_rs.run_feature("spin-susceptibility",[1,1],positions,**options)
    atoms=Atoms("H2",positions=np.array(positions)*Bohr)
    adapted=dft_rs.DftRs(**options).get_spin_susceptibility(atoms,units="atomic")
    xyz=tmp_path/"hydrogen.xyz"
    xyz.write_text("2\nspin response\n"+"".join(f"H 0 0 {p[2]/dft_rs.native.BOHR_PER_ANGSTROM:.16f}\n" for p in positions),encoding="utf-8")
    argv=[os.environ["DFT_RS_CLI"],"spin-susceptibility",str(xyz),"--format","json"]
    for key,value in options.items():
        argv.append("--"+key.replace("_","-"))
        if value is not True:
            argv.append(str(value))
    executed=subprocess.run(argv,capture_output=True,text=True)
    assert executed.returncode==0,executed.stderr
    cli=json.loads(executed.stdout)
    for other in [generic,adapted,cli]:
        for key in ["spin_susceptibility","bare_spin_susceptibility","spin_stiffness","spin_enhancement",
                    "spin_density_response","response_residual","reference_orbital_residual"]:
            np.testing.assert_allclose(other[key],result[key],atol=1e-11,rtol=1e-10)
        assert other["units"]==result["units"]
    assert result["spin_susceptibility"]>0
    assert result["response_residual"]<1e-10
    assert result["reference_orbital_residual"]<1e-9
    assert max(result["number_defects"])<1e-12
    np.testing.assert_allclose(result["spin_enhancement"],result["spin_susceptibility"]/result["bare_spin_susceptibility"],atol=1e-13,rtol=0)


@pytest.mark.parametrize("feature,native_name,ase_name", [
    ("spin-spin-couplings", "spin_spin_couplings", "get_spin_spin_couplings"),
    ("fermi-contact-couplings", "fermi_contact_couplings", "get_fermi_contact_couplings"),
])
def test_indirect_coupling_pair_axes_and_public_routes(tmp_path, feature, native_name, ase_name):
    """Check every returned tensor component with its labelled nucleus order.

    The production response is evaluated independently in both pair orders;
    the canonical CLI and rebuilt native/ASE routes share the same contract.
    """
    numbers = [8, 1, 1]
    positions = np.array([[0., .05, .23], [.15, 1.4309, -.8867], [.12, -1.4809, -.9067]])
    options = dict(xc="hf", basis="sto-3g", nuclei=[2, 0, 1],
                   nuclear_g=[-.7575, 5.5856946893, 5.5856946893],
                   grid_level=3, dispersion="none", conv_tol=1e-12,
                   conv_tol_grad=1e-11, conv_tol_dp=1e-11, max_iter=300,
                   response_tolerance=1e-11)
    native = getattr(dft_rs, native_name)(numbers, positions, **options)
    generic = dft_rs.run_feature(feature, numbers, positions, **options)
    atoms = Atoms(numbers=numbers, positions=positions * Bohr)
    adapted = getattr(dft_rs.DftRs(units="atomic"), ase_name)(atoms, **options)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for the canonical CLI acceptance"
    xyz = tmp_path / "nuclear-couplings.xyz"
    native_length = 1 / dft_rs.native.BOHR_PER_ANGSTROM
    xyz.write_text("3\nindirect nuclear coupling pair axes\n" + "".join(
        f"{symbol} " + " ".join(f"{v * native_length:.17g}" for v in row) + "\n"
        for symbol, row in zip(("O", "H", "H"), positions)), encoding="utf-8")
    argv = [executable, feature, str(xyz), "--format", "json"]
    for key, value in options.items():
        argv.extend(["--" + key.replace("_", "-"),
                     json.dumps(value) if isinstance(value, list) else str(value)])
    executed = subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True)
    assert executed.returncode == 0, executed.stderr
    assert re.search(r"\[scf\] iter.*E\s*=.*SCF residual", executed.stderr), executed.stderr
    cli = json.loads(executed.stdout)
    tensors = ["reduced_couplings", "fermi_contact_tensors"]
    scalars = ["reduced_isotropic", "fermi_contact_isotropic",
               "coupling_constants_hz", "fermi_contact_hz"]
    if feature == "spin-spin-couplings":
        tensors += ["diamagnetic_tensors", "paramagnetic_tensors", "spin_dipole_tensors"]
        scalars += ["diamagnetic_isotropic", "paramagnetic_isotropic", "spin_dipole_isotropic"]
    for report in (native, generic, adapted, cli):
        assert report["converged"]
        assert report["coupling_pairs"] == [[2, 0], [2, 1], [0, 1]]
        assert report["axis_convention"] == native["axis_convention"]
        assert "reversing a pair transposes" in report["axis_convention"]
        assert report["units"] == native["units"]
        assert report["response_residual"] < 1e-10
        for key in tensors + scalars:
            np.testing.assert_allclose(report[key], native[key], atol=1e-17, rtol=1e-8)
        np.testing.assert_allclose(report["reduced_isotropic"],
                                   np.trace(report["reduced_couplings"], axis1=1, axis2=2) / 3,
                                   atol=1e-20, rtol=1e-12)
        if feature == "spin-spin-couplings":
            total = sum(np.asarray(report[key]) for key in tensors[1:])
            np.testing.assert_allclose(report["reduced_couplings"], total, atol=1e-20, rtol=1e-12)

    reversed_order = dft_rs.run_feature(feature, numbers, positions,
                                      **dict(options, nuclei=[1, 0, 2]))
    for index, (first, second) in enumerate(native["coupling_pairs"]):
        other = reversed_order["coupling_pairs"].index([second, first])
        for key in tensors:
            np.testing.assert_allclose(np.asarray(native[key])[index],
                                       np.asarray(reversed_order[key])[other].T,
                                       atol=1e-17, rtol=1e-8)
    with pytest.raises(ValueError, match="distinct atom indices"):
        getattr(dft_rs, native_name)(numbers, positions, **dict(options, nuclei=[0, 0]))
    route_errors = {key: max(float(np.max(np.abs(np.asarray(report[key]) - np.asarray(native[key]))))
                            for report in [generic, adapted, cli]) for key in tensors + scalars}
    reciprocal_errors = {key: max(float(np.max(np.abs(
        np.asarray(native[key])[index] - np.asarray(reversed_order[key])[
            reversed_order["coupling_pairs"].index([second, first])].T)))
        for index, (first, second) in enumerate(native["coupling_pairs"])) for key in tensors}
    print(f"{feature}: response_residual={native['response_residual']:.12e} "
          f"route_defects={route_errors} reciprocal_defects={reciprocal_errors}; "
          "each defect uses its named entry's reported unit")
