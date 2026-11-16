# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Independent public Gaussian Coulomb energy, derivative and route gates."""
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, periodic_gaussian_integral, periodic_capabilities, periodic_electrostatics

CELL = np.array([[5.2, .5, .3], [.8, 5.6, .4], [.3, .2, 6.0]])
POSITIONS = np.array([[.1, -.2, .3], [.8, .4, -.1], [-.4, .9, .6], [.6, -.7, 1.1]])
EXPONENTS = [.6, .4, .5, .7]
ANGULAR = [[1, 0, 0], [0, 1, 0], [0, 0, 2], [1, 0, 0]]
OPTIONS = dict(alpha=.55, real_cutoff=16., reciprocal_cutoff=8.)


def calculation(dimension, nuclear):
    positions = (np.vstack([POSITIONS[:2], [-.5, .7, .9]]) if nuclear else POSITIONS.copy())
    exponents, angular = (EXPONENTS[:2], ANGULAR[:2]) if nuclear else (EXPONENTS, ANGULAR)
    controls = dict(OPTIONS, pbc=[i < dimension for i in range(3)])
    if nuclear:
        controls.update(source_charge=2., source_width=.3)
    return positions, exponents, angular, controls


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("nuclear", [True, False])
def test_native_ase_cli_share_complete_gaussian_integral(dimension, nuclear, tmp_path):
    positions, exponents, angular, controls = calculation(dimension, nuclear)
    report = periodic_gaussian_integral(positions, exponents, angular, CELL, **controls)
    atoms = Atoms("H"*len(positions), positions=positions*Bohr, cell=CELL*Bohr, pbc=controls["pbc"])
    adapted = DftRs(units="atomic").get_periodic_gaussian_integral(atoms, exponents=exponents, angular=angular,
                        **{k:v for k,v in controls.items() if k != "pbc"})
    assert report["dimension"] == dimension and report["shape"] == [len(positions), 3]
    assert report["converged"] and report["quadrature_order"] == 192
    assert report["translation_defect"] < 1e-11
    for key in ("value", "center_gradients", "quadrature_value_defect", "quadrature_gradient_defect",
                "translation_defect", "alpha", "real_cutoff", "reciprocal_cutoff", "measure"):
        np.testing.assert_allclose(report[key], adapted[key], rtol=0, atol=1e-12)
    assert report["components"] == pytest.approx(adapted["components"], abs=1e-12)
    assert sum(report["components"].values()) == pytest.approx(report["value"], abs=1e-13)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for actual CLI acceptance"
    conversion = .529177210903
    path = tmp_path / "primitive.xyz"
    path.write_text(f"{len(positions)}\nnormalized Cartesian Gaussian centres\n" + "".join(
        "H " + " ".join(f"{v*conversion:.17g}" for v in row) + "\n" for row in positions))
    cell = ";".join(",".join(f"{v*conversion:.17g}" for v in row) for row in CELL)
    flags = ["--cell", cell, "--pbc", ",".join(str(int(v)) for v in controls["pbc"]),
             "--gaussian-exponents", ",".join(map(str, exponents)), "--gaussian-angular",
             ";".join(",".join(map(str, v)) for v in angular), "--ewald-alpha", ".55",
             "--ewald-real-cutoff", "16", "--ewald-reciprocal-cutoff", "8", "--log-mode", "none"]
    if nuclear:
        flags += ["--gaussian-source-charge", "2", "--gaussian-source-width", ".3"]
    result = subprocess.run([executable, "periodic-gaussian-integral", str(path), *flags],
                             capture_output=True, text=True, check=True)
    rows = [line.split() for line in result.stdout.splitlines() if line]
    assert float(next(row[1] for row in rows if row[0] == "value")) == pytest.approx(report["value"], abs=1e-12)
    gradients = [row for row in rows if row[0] == "center_gradient"]
    assert len(gradients) == len(positions)
    for row in gradients:
        np.testing.assert_allclose(list(map(float, row[2:])), report["center_gradients"][int(row[1])], atol=1e-12, rtol=0)
    component = next(row for row in rows if row[0] == "components")
    for key,value in zip(component[1::2], component[2::2]):
        assert float(value) == pytest.approx(report["components"][key], abs=1e-12)
    residuals = next(row for row in rows if row[0] == "converged")
    assert residuals[:4] == ["converged", "true", "quadrature_order", "192"]
    for key,value in zip(residuals[4::2], residuals[5::2]):
        assert float(value) == pytest.approx(report[key], abs=1e-12)
    assert report["model"] in result.stdout and report["units"] in result.stdout


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("nuclear", [True, False])
def test_public_gaussian_center_derivatives_have_four_step_fits(dimension, nuclear):
    positions, exponents, angular, controls = calculation(dimension, nuclear)
    exact = periodic_gaussian_integral(positions, exponents, angular, CELL, **controls)
    steps = np.array([.02, .01, .005, .0025])
    estimates = []
    for h in steps:
        plus, minus = positions.copy(), positions.copy()
        plus[0, 0] += h
        minus[0, 0] -= h
        p = periodic_gaussian_integral(plus, exponents, angular, CELL, **controls)
        m = periodic_gaussian_integral(minus, exponents, angular, CELL, **controls)
        assert p["converged"] and m["converged"]
        estimates.append((p["value"]-m["value"])/(2*h))
    actual = exact["center_gradients"][0][0]
    errors = np.abs(np.array(estimates)-actual)
    assert np.all(errors > 0) and np.all(np.isfinite(errors))
    p, intercept = np.polyfit(np.log(steps), np.log(errors), 1)
    r2 = 1 - np.sum((np.log(errors)-intercept-p*np.log(steps))**2)/np.sum((np.log(errors)-np.mean(np.log(errors)))**2)
    print(f"{dimension}D public {'nuclear' if nuclear else 'eri'}: errors={errors.tolist()} h=.02..0025 p={p:.12f} R2={r2:.12f}")
    assert abs(p-2) < .02 and r2 > .99999
    np.testing.assert_allclose(errors[:-1]/errors[1:], 4, rtol=.03, atol=0)
    assert (4*estimates[-1]-estimates[-2])/3 == pytest.approx(actual, abs=1e-8)


def test_charged_gaussian_uniform_background_has_zero_average_fourier_energy():
    positions = np.array([[.3, -.4, .6], [-.7, .8, -.2]])
    charges, widths = np.array([1.2, -.4]), np.array([1.3, .9])
    report = periodic_electrostatics(positions, charges, CELL, widths=widths, compensation="uniform")
    integers = np.array(np.meshgrid(*[np.arange(-18, 19)]*3, indexing="ij")).reshape(3, -1).T
    waves = integers@ (2*np.pi*np.linalg.inv(CELL).T)
    squared = np.sum(waves**2, axis=1)
    waves, squared = waves[(squared>0)&(squared<=100)], squared[(squared>0)&(squared<=100)]
    density = np.sum(charges[None,:]*np.exp(-squared[:,None]*widths[None,:]**2/4)
                     *np.exp(1j*(waves@positions.T)), axis=1)
    oracle = .5*np.sum(4*np.pi/abs(np.linalg.det(CELL))/squared*np.abs(density)**2)
    assert report["energy"] == pytest.approx(oracle, abs=1e-11)
    assert report["converged"]


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_independent_nuclear_source_derivative_has_four_step_fit(dimension):
    positions, exponents, angular, controls = calculation(dimension, True)
    exact = periodic_gaussian_integral(positions, exponents, angular, CELL, **controls)
    steps = np.array([.02, .01, .005, .0025])
    estimates = []
    for h in steps:
        plus, minus = positions.copy(), positions.copy()
        plus[2, 0] += h
        minus[2, 0] -= h
        p = periodic_gaussian_integral(plus, exponents, angular, CELL, **controls)
        m = periodic_gaussian_integral(minus, exponents, angular, CELL, **controls)
        assert p["converged"] and m["converged"]
        estimates.append((p["value"]-m["value"])/(2*h))
    actual = exact["center_gradients"][2][0]
    errors = np.abs(np.array(estimates)-actual)
    assert np.all(errors > 0) and np.all(np.isfinite(errors))
    p, intercept = np.polyfit(np.log(steps), np.log(errors), 1)
    r2 = 1 - np.sum((np.log(errors)-intercept-p*np.log(steps))**2)/np.sum((np.log(errors)-np.mean(np.log(errors)))**2)
    print(f"{dimension}D public nuclear source: errors={errors.tolist()} h=.02..0025 p={p:.12f} R2={r2:.12f}")
    assert abs(p-2) < .02 and r2 > .99999
    np.testing.assert_allclose(errors[:-1]/errors[1:], 4, rtol=.03, atol=0)
    assert (4*estimates[-1]-estimates[-2])/3 == pytest.approx(actual, abs=1e-8)


def test_gaussian_input_contracts_reject_noop_controls_and_overflows(tmp_path):
    assert periodic_capabilities()["dimensional_gaussian_integrals"]
    for kwargs in ({"source_charge":1}, {"source_width":0}, {"compensation":"uniform"},
                   {"xc":"pbe"}, {"k_mesh":[1,1,1]}):
        with pytest.raises((ValueError, TypeError)):
            periodic_gaussian_integral(POSITIONS, EXPONENTS, ANGULAR, CELL, **kwargs)
    for exponents, angular in (([0.,.4,.5,.7], ANGULAR), (EXPONENTS,[[2**64-1,0,0]]*4)):
        with pytest.raises(ValueError):
            periodic_gaussian_integral(POSITIONS, exponents, angular, CELL)
    path = tmp_path/"nonexistent.xyz"
    for command, extra in (("energy",["--gaussian-tol","1e-10"]),
                            ("periodic-gaussian-integral",["--xc","pbe"]),
                            ("periodic-gaussian-integral",["--k-mesh","2,1,1"]),
                            ("periodic-gaussian-integral",["--electrostatic-tol","1e-10"])):
        result = subprocess.run([os.environ["DFT_RS_CLI"],command,str(path),*extra], capture_output=True,text=True)
        assert result.returncode != 0 and "No such file" not in result.stderr
