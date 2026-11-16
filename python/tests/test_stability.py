# This file is part of the dft-rs library.
"""Stability channels, reports and internal following on all public routes."""
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, stability, stabilize, calculate

OPTIONS=dict(xc="hf",max_iter=250,conv_tol=1e-14,conv_tol_dp=1e-13,conv_tol_grad=1e-12)

def cli_geometry(tmp_path,r):
    path=tmp_path/"orbital_stability.xyz"
    # The CLI uses the package CODATA Bohr constant; ASE converts with its
    # own selected CODATA edition. Both paths must represent the same Bohr input.
    path.write_text(f"2\norbital stability validation\nH 0 0 0\nH 0 0 {r*0.529177210903:.16f}\n")
    return path

def cli(path,command,*arguments):
    executable=os.environ.get("DFT_RS_CLI")
    assert executable,"DFT_RS_CLI is required for route acceptance"
    return subprocess.run([executable,command,str(path),"--xc","hf","--max-iter","250",
        "--conv-tol","1e-14","--conv-tol-dp","1e-13","--conv-tol-grad","1e-12",
        "--log-mode","none",*arguments],capture_output=True,text=True,check=True)

@pytest.mark.parametrize("r,expected",[(1.4,"stable"),(4.5,"unstable")])
def test_channels_and_curvatures_match_python_ase_cli(tmp_path,r,expected):
    direct=stability([1,1],[[0,0,0],[0,0,r]],**OPTIONS)
    atoms=Atoms("H2",positions=[[0,0,0],[0,0,r*Bohr]])
    adapted=DftRs(units="atomic", **OPTIONS).get_stability(atoms)
    assert direct["classification"]==adapted["classification"]==expected
    assert direct["converged"] and direct["provenance"]=="analytic"
    assert [b["channel"] for b in direct["channels"]]==["internal-real","external-spin","external-complex"]
    output=cli(cli_geometry(tmp_path,r),"stability").stdout
    rows=[line.split() for line in output.splitlines() if line.startswith("stability_mode ")]
    for a,b,row in zip(direct["channels"],adapted["channels"],rows):
        np.testing.assert_allclose(a["eigenvalues"],b["eigenvalues"],atol=1e-10)
        assert abs(float(row[3])-a["lowest"])<1e-9
        assert abs(float(row[5])-a["modes"][0]["energy_curvature"])<1e-9
        assert a["dominant"][0]["from"]==0 and a["dominant"][0]["to"]==1
        assert abs(sum(x*x for x in a["eigenvector"])-1)<1e-12
        assert a["max_residual"]<1e-7 and a["converged"] and a["matvec_count"]>0
        assert a["resolution"]==1e-6
    assert "hartree" in output
    if expected=="unstable":
        warning=" ".join(direct["warnings"])
        assert "saddle" in warning and "spin polarization" in warning and "unrestricted" in warning

def test_following_returns_lower_state_on_all_routes_and_applies_before_cli_properties(tmp_path):
    geometry=[[0,0,0],[0,0,4.5]]
    direct=stabilize([1,1],geometry,unrestricted=True,tolerance=1e-10,max_iterations=500,**OPTIONS)
    atoms=Atoms("H2",positions=np.asarray(geometry)*Bohr)
    adapted=DftRs(units="atomic", **OPTIONS).get_stab_following(atoms,unrestricted=True,tolerance=1e-10,max_iterations=500)
    assert direct["stabilized"] and direct["following_converged"] and direct["unrestricted"]
    assert direct["initial"]["classification"]=="unstable" and direct["classification"]=="stable"
    assert len(direct["descents"])==len(adapted["descents"])==1
    assert direct["initial_energy"]-direct["energy"]>0.1
    np.testing.assert_allclose(direct["energy"],adapted["energy"],atol=1e-11)
    np.testing.assert_allclose(direct["density"],adapted["density"],atol=1e-9)
    path=cli_geometry(tmp_path,4.5)
    for command,extra in [("stab-following",[]),("stability",["--stab-follow"]),("energy",["--stab-follow"]),
                          ("dynamic-polarizability",["--stab-follow","--response-omega","0.03","--response-damping","0.006"])]:
        output=cli(path,command,"--unrestricted","--stab-tol","1e-10","--stab-max-iter","500",*extra).stdout
        assert "stabilized true" in output and "stability_descent unrestricted-internal" in output
        record=next(line.split() for line in output.splitlines() if line.startswith("stabilized "))
        assert abs(float(record[-1])-direct["energy"])<1e-9
    restricted=stabilize([1,1],geometry,**OPTIONS)
    assert not restricted["stabilized"] and not restricted["unrestricted"] and restricted["descents"]==[]
    assert restricted["classification"]=="unstable"

def test_unresolved_sign_and_eigensolver_nonconvergence_are_reported():
    indefinite=stability([1,1],[[0,0,0],[0,0,1.4]],resolution=1000,**OPTIONS)
    assert indefinite["classification"]=="indefinite" and not indefinite["stable"]
    assert indefinite["converged"]
    # Remove symmetry-isolated one-dimensional excitation blocks: such a
    # block can correctly converge on the first Davidson iteration.
    water=[[0,0,0],[1.4,0.2,1.1],[-1.2,-0.3,1.3]]
    report=stability([8,1,1],water,n_roots=1,max_iterations=1,tolerance=1e-14,
                     block_buffer=0,max_subspace=4,channels=["internal-real"],**OPTIONS)
    assert not report["converged"] and report["provenance"]=="unconverged"
    assert report["channels"][0]["max_residual"]>1e-14
    assert any("did NOT" in warning for warning in report["warnings"])

def test_invalid_controls_are_checked_before_geometry(tmp_path):
    for opts in [dict(n_roots=0),dict(resolution=1e-12),dict(step_angle=0),dict(max_descents=0),
                 dict(min_energy_drop=-1),dict(channels=["wrong"]),dict(max_subspace=2)]:
        with pytest.raises(ValueError):stability([],[],**opts)
    executable=os.environ["DFT_RS_CLI"]
    for command,arguments in [("stability",["--stab-roots","0"]),("stability",["--response-omega","0.1"]),
        ("stability",["--stab-resolution","1e-12"]),("energy",["--stab-roots","1"]),
        ("stability",["--stab-follow","--stab-step-angle","0"]),
        ("stability",["--response-restart","10"]),("optimize",["--stab-follow"])]:
        result=subprocess.run([executable,command,"missing.xyz",*arguments],capture_output=True,text=True)
        assert result.returncode!=0 and "cannot read" not in result.stderr

def test_stabilized_single_point_properties_share_the_final_reference(tmp_path):
    geometry=[[0,0,0],[0,0,4.5]]
    final=stabilize([1,1],geometry,unrestricted=True,**OPTIONS)
    point=calculate([1,1],geometry,unrestricted=True,stabilize=True,
                    gradient=True,wavefunction=True,**OPTIONS)
    assert point["stabilized"] and point["unrestricted"] and point["converged"]
    assert point["stability"]["stabilized"]
    np.testing.assert_allclose(point["energy"],final["energy"],atol=1e-11)
    np.testing.assert_allclose(point["wavefunction"]["density"],final["density"],atol=1e-9)
    np.testing.assert_allclose(point["wavefunction"]["mo_energy"],final["mo_energy"],atol=1e-9)
    atoms=Atoms("H2",positions=np.asarray(geometry)*Bohr)
    atoms.calc=DftRs(units="atomic",unrestricted=True,stabilize=True,wavefunction=True,**OPTIONS)
    from dft_rs.units import HARTREE_IN_EV as Hartree
    np.testing.assert_allclose(atoms.get_potential_energy(),point["energy"]*Hartree,atol=1e-10)
    np.testing.assert_allclose(atoms.get_forces(),-np.asarray(point["gradient"])*Hartree/Bohr,atol=1e-9)
    np.testing.assert_allclose(atoms.get_charges(),point["mulliken"],atol=1e-10)
    np.testing.assert_allclose(atoms.get_dipole_moment(),np.asarray(point["dipole"])*Bohr,atol=1e-10)
    assert atoms.calc.results["stability"]["stabilized"]
    path=cli_geometry(tmp_path,4.5)
    output=cli(path,"gradient","--unrestricted","--stab-follow").stdout
    assert "stabilized true" in output
    table=output.split("(Hartree/Bohr)\n",1)[1]
    rows=[line.split() for line in table.splitlines() if len(line.split())==4 and line.split()[0].isdigit()]
    np.testing.assert_allclose([[float(v) for v in row[1:]] for row in rows],point["gradient"],atol=1e-9)
    with pytest.raises(NotImplementedError):atoms.calc.get_hessian(atoms)
    with pytest.raises(NotImplementedError):
        calculate([1,1],geometry,unrestricted=True,stabilize=True,gradient=True,density_fit="def2-universal-jfit",**OPTIONS)

def test_stabilized_gradient_has_four_step_second_order_energy_error_fit():
    r=4.5
    def point(x,gradient=False):
        return calculate([1,1],[[0,0,0],[0,0,x]],unrestricted=True,stabilize=True,gradient=gradient,**OPTIONS)
    analytic=point(r,True)["gradient"][1][2]
    steps=np.asarray([0.02,0.01,0.005,0.0025])
    values=np.asarray([(point(r+h)["energy"]-point(r-h)["energy"])/(2*h) for h in steps])
    errors=np.abs(values-analytic)
    assert np.all(np.isfinite(errors)) and np.all(errors>0)
    x,y=np.log(steps),np.log(errors)
    exponent,intercept=np.polyfit(x,y,1)
    sse=np.sum((y-(intercept+exponent*x))**2)
    r_squared=1-sse/np.sum((y-y.mean())**2)
    print(f"stabilized gradient h={steps.tolist()} errors={errors.tolist()} p={exponent:.12f} R2={r_squared:.12f} C={np.exp(intercept):.12e}")
    assert abs(exponent-2)<0.02
    assert r_squared>0.99999
    assert np.max(np.abs(errors[:-1]/errors[1:]/4-1))<0.03
    assert abs((4*values[-1]-values[-2])/3-analytic)<1e-6
