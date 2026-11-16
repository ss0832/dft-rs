# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Independent original TPSSloc correlation and TPSS-X composition fixtures.

Constantin, Fabiano and Della Sala, PRB 86, 035130 (2012).
Localized beta0=0.0375, gradient coefficient=0.08, C0=0.35, d=4.5.
Optional PySCF/libxc is used only when freezing numerical values.
"""

import json
from decimal import Decimal, localcontext
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np
import pyscf
from pyscf import dft, gto, lib

from gen_functional_points import POINTS, spin_inputs
from gen_integrals import DATA_DIR, OUT_DIR, dftrs_ao_permutation, parse_g94
from gen_pkzb import point_value
from gen_scf import MOLECULES, Z


def probe_zero_gradient_kernels(result):
    """Return diagnostic inward VXC estimates without changing any FXC.

    Raw values are retained. Diagnostic numbers use only independent first
    potentials, never production energies/derivatives or fitted thresholds.
    """
    lib.num_threads(2)
    index = len(result["points"]) - 1
    point = result["points"][index]
    assert point[2:5] == [0.0, 0.0, 0.0]
    records = []
    for name, code in [("exchange", "MGGA_X_TPSS,"), ("combined", "MGGA_X_TPSS,MGGA_C_TPSSLOC")]:
        entry = result["components"][name][index]
        raw = entry["kernel"]
        for variable in [2, 4]:
            def value(step):
                displaced = point.copy()
                displaced[variable] += step
                assert displaced[variable] <= 8 * displaced[variable // 2 - 1] * displaced[variable // 2 + 4]
                evaluated = dft.libxc.eval_xc(code, spin_inputs(displaced, True), spin=1, deriv=1)
                return float(evaluated[1][1][0][variable - 2])
            def estimate(h):
                values = [value(k * h) for k in range(9)]
                weights = [-sum((Fraction(1, k) for k in range(1, 9)), Fraction(0))]
                weights += [Fraction((-1)**(k + 1) * comb(8, k), k) for k in range(1, 9)]
                with localcontext() as ctx:
                    ctx.prec = 60
                    total = sum(Decimal.from_float(v) * Decimal(w.numerator) / Decimal(w.denominator)
                                for v, w in zip(values, weights))
                    return float(total / Decimal.from_float(h))
            steps = [0.016, 0.008, 0.004, 0.002, 0.001]
            estimates = [estimate(h) for h in steps]
            resolved = (256 * estimates[-1] - estimates[-2]) / 255
            uncertainty = abs(estimates[-1] - estimates[-2])
            fit_steps = [0.008, 0.004, 0.002, 0.001]
            records.append({"component": name, "point_index": index, "variable": variable,
                "raw_fxc": raw[variable][variable], "inward_reference": resolved,
                "high_order_steps": steps, "high_order_estimates": estimates,
                "estimate_spread": uncertainty,
                "diagnostic_spread_bound_met": uncertainty < 1e-10 * (1 + abs(resolved)),
                "fit_steps": fit_steps,
                "fit_potentials": [[value(0), value(h), value(2 * h)] for h in fit_steps],
                "convention": "positive-sigma first derivative of independent VXC; degree8 forward interpolation and fixed Richardson limit; raw FXC retained"})
    return records


def apply_approved_zero_gradient_references(result):
    """Reproduce the four explicitly approved physical inward references.

    The original 7.0.0 zero-sigma FXC values and failed comparisons are
    archived separately. Paper-equation/high-precision/native-operation
    audits establish cancellation in those four raw values. This narrow
    correction uses the approved independent VXC limits, never production
    outputs. Changed inputs or a changed raw reference require fresh review.
    """
    point = [0.31, 0.22, 0.0, 0.0, 0.0, 0.37, 0.21]
    assert result["points"][10] == point
    corrections = [
        ("exchange", 2, 1.4642734346580563e22, -0.028203103069504982),
        ("exchange", 4, -9.270039550903819e21, -0.10945909911553753),
        ("combined", 2, 1.4642734346580563e22, -0.02370518485567705),
        ("combined", 4, -9.270039550903819e21, -0.10496118090174651),
    ]
    for component, axis, raw, approved in corrections:
        kernel = result["components"][component][10]["kernel"]
        if kernel[axis][axis] != raw:
            raise ValueError("TPSSloc raw boundary reference changed; investigate before applying its approved correction")
        kernel[axis][axis] = approved


def main():
    lib.num_threads(2)
    points = [*POINTS, [0.31, 0.22, 0.041, -0.017, 0.029, 0.37, 0.21],
              [0.31, 0.22, 0.0, 0.0, 0.0, 0.37, 0.21]]
    result = {"provenance": f"PySCF {pyscf.__version__}, libxc {dft.libxc.__version__}",
              "publication": "doi:10.1103/PhysRevB.86.035130",
              "definition": "original TPSSloc correlation; explicit combined recipe TPSS-X + TPSSloc-C",
              "variables": "rho_a,rho_b,sigma_aa,sigma_ab,sigma_bb,tau_a,tau_b",
              "units": "atomic units; energy per volume Hartree/Bohr^3; tau includes one-half",
              "components": {}, "points": points, "scf": []}
    for name, code in [("exchange", "MGGA_X_TPSS,"), ("correlation", ",MGGA_C_TPSSLOC"),
                       ("combined", "MGGA_X_TPSS,MGGA_C_TPSSLOC")]:
        result["components"][name] = [point_value(code, p) for p in points]
    apply_approved_zero_gradient_references(result)
    for name in ["h2o", "ch3_doublet"]:
        charge, multiplicity, atoms = MOLECULES[name]
        basis = parse_g94(str(Path(DATA_DIR) / "sto-3g.g94"), {a[0] for a in atoms})
        mol = gto.M(atom=atoms, basis=basis, unit="Bohr", charge=charge,
                    spin=multiplicity - 1, cart=False, verbose=0, max_memory=256)
        mf = dft.RKS(mol) if multiplicity == 1 else dft.UKS(mol)
        mf._chkfile.close()
        mf.chkfile = None
        mf.xc = "MGGA_X_TPSS,MGGA_C_TPSSLOC"
        mf.max_memory = 256
        mf.grids.level = 9
        mf.conv_tol = 1e-13
        mf.conv_tol_grad = 1e-10
        mf.max_cycle = 200
        mf.kernel()
        assert mf.converged, name
        density = mf.make_rdm1()
        spins = np.array([density * 0.5, density * 0.5]) if density.ndim == 2 else density
        total = spins.sum(axis=0)
        terms = {"core": float(np.einsum("ij,ji", mf.get_hcore(), total)),
                 "coulomb": float(0.5 * np.einsum("ij,ji", mf.get_j(dm=total), total))}
        for part, code in [("exchange", "MGGA_X_TPSS,"), ("correlation", ",MGGA_C_TPSSLOC")]:
            terms[part] = float(mf._numint.nr_uks(mol, mf.grids, code, spins, max_memory=256)[1])
        defect = mol.energy_nuc() + sum(terms.values()) - mf.e_tot
        assert abs(defect) < 2e-11
        permutation = dftrs_ao_permutation(mol)
        result["scf"].append({"molecule": name, "numbers": [Z[a[0]] for a in atoms],
                              "positions": [list(a[1:]) for a in atoms], "charge": charge,
                              "multiplicity": multiplicity, "total": float(mf.e_tot), **terms,
                              "reconstruction_error": float(defect),
                              "density": spins[:, permutation][:, :, permutation].tolist()})
        for split in [False, True]:
            suffix = "_split" if split else ""
            path = Path(OUT_DIR) / f"{name}_sto3g_tpssloc{suffix}_dft.csv"
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(f"# PROVENANCE: {result['provenance']}; TPSS-X + TPSSloc-C247\n")
                handle.write("# spherical STO-3G, grid9; no frozen core or extra correction\n")
                handle.write("basis,sto-3g\n")
                handle.write("x,tpss\nc,tpssloc\n" if split else "functional,tpssloc\n")
                handle.write(f"charge,{charge}\nmult,{multiplicity}\n")
                for symbol, x, y, z in atoms:
                    handle.write(f"atom,{Z[symbol]},{x},{y},{z}\n")
                handle.write(f"e_tot,{mf.e_tot:.16e}\n")
        print(name, f"{mf.e_tot:.14f}", "term reconstruction", defect, flush=True)
    (Path(OUT_DIR) / "tpssloc.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
