// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Organometallic gate: Ni(CO)4 (Td, 9 atoms, LANL2DZ with the Ni ECP) -
//! energies, analytic gradients (incl. the ECP derivative terms), and
//! Hessian columns (central differences of our analytic gradient vs the
//! frozen PySCF analytic Hessian).

use dft_rs::basis::{data, MolBasis};
use dft_rs::gradient::analytic_gradient;
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::scf::{EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec, ResolvedFunctional};

struct OmCase {
    functional: String,
    molecule: Molecule,
    e_tot: f64,
    grad: Vec<Vec3>,
    hess: Option<Matrix>,
}

fn load(fname: &str) -> OmCase {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/oracle_data")
        .join(fname);
    let text = std::fs::read_to_string(path).unwrap();
    let mut functional = String::new();
    let mut atoms = Vec::new();
    let mut e_tot = f64::NAN;
    let mut grad = Vec::new();
    let mut hess_rows: Vec<(usize, Vec<f64>)> = Vec::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let p: Vec<&str> = line.split(',').collect();
        match p[0] {
            "basis" | "charge" | "mult" => {}
            "functional" => functional = p[1].into(),
            "atom" => atoms.push(Atom {
                z: p[1].parse().unwrap(),
                position: Vec3::new(
                    p[2].parse().unwrap(),
                    p[3].parse().unwrap(),
                    p[4].parse().unwrap(),
                ),
            }),
            "e_tot" => e_tot = p[1].parse().unwrap(),
            "grad" => grad.push(Vec3::new(
                p[2].parse().unwrap(),
                p[3].parse().unwrap(),
                p[4].parse().unwrap(),
            )),
            "hess" => {
                let row: Vec<f64> = p[2..].iter().map(|v| v.parse().unwrap()).collect();
                hess_rows.push((p[1].parse().unwrap(), row));
            }
            other => panic!("unknown key {other}"),
        }
    }
    let hess = if hess_rows.is_empty() {
        None
    } else {
        let dim = hess_rows[0].1.len();
        let mut m = Matrix::zeros(dim, dim);
        for (r, row) in &hess_rows {
            for (c, v) in row.iter().enumerate() {
                m[(*r, c)] = *v;
            }
        }
        Some(m)
    };
    OmCase {
        functional,
        molecule: Molecule::new(atoms),
        e_tot,
        grad,
        hess,
    }
}

fn opts() -> ScfOptions {
    ScfOptions {
        e_tol: 1e-11,
        p_tol: 1e-9,
        max_iter: 300,
        eri_mode: EriMode::InCore { limit_mb: 1024 },
        ..ScfOptions::default()
    }
}

fn setup(case: &OmCase) -> (MolBasis, ResolvedFunctional, Option<MolecularGrid>) {
    let set = data::load("lanl2dz").unwrap();
    let mb = MolBasis::build(&case.molecule, &set).unwrap();
    let func = if case.functional == "hf" {
        ResolvedFunctional::hartree_fock()
    } else {
        resolve(&FunctionalSpec::Combined(case.functional.clone())).unwrap()
    };
    let grid = if func.parts.is_empty() {
        None
    } else {
        Some(
            MolecularGrid::build(
                &case.molecule,
                &GridOptions {
                    level: 8,
                    ..GridOptions::default()
                },
            )
            .unwrap(),
        )
    };
    (mb, func, grid)
}

/// The analytic Hessian of a molecule carrying an ECP, against central
/// differences of the analytic gradient — and the same geometry in an
/// all-electron basis as a control.
///
/// The control is what cracked this. Every ECP derivative integral was
/// verified element-wise (`ecp_second_contraction` against differences of
/// `ecp_gradient`, `ecp_derivative_matrices` against differences of
/// `ecp_matrix`) and the assembled Hessian was still ~150 Hartree/bohr^2
/// wrong, while the all-electron Hessian on the identical geometry was
/// exact to 1e-8. That combination said the fault could not be in the ECP
/// integrals: it had to be somewhere the potential's PRESENCE changes an
/// input. It was `charges` — the Hessian built its nuclear terms from the
/// bare atomic number instead of Z_eff, so sulphur entered as Z = 16
/// against the Z_eff = 6 the SCF had used.
#[test]
fn ecp_analytic_hessian_matches_finite_differences() {
    use dft_rs::hessian::analytic::analytic_hessian_full;
    use dft_rs::scf::run_scf;

    let mol = Molecule::new(vec![
        Atom {
            z: 16,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 1.81, 1.45),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.3, -1.75, 1.50),
        },
    ]);
    let set = data::load("lanl2dz").unwrap();
    let func = ResolvedFunctional::hartree_fock();
    let o = opts();
    let h = 2e-4;
    let check = |set: &dft_rs::basis::BasisSet, label: &str| {
        let build = |m: &Molecule| MolBasis::build(m, set).unwrap();
        let mb = build(&mol);
        let scf = run_scf(&mol, &mb, &o, &func, None).unwrap();
        let an = analytic_hessian_full(&mol, &mb, &o, &func, &scf, None)
            .unwrap_or_else(|e| panic!("{label}: analytic Hessian refused: {e}"));
        let grad_at = |at: usize, ax: usize, s: f64| -> Vec<Vec3> {
            let mut m = mol.clone();
            match ax {
                0 => m.atoms[at].position.x += s,
                1 => m.atoms[at].position.y += s,
                _ => m.atoms[at].position.z += s,
            }
            analytic_gradient(&m, &build(&m), &o, &func, None)
                .unwrap()
                .gradient
        };
        let mut worst = 0.0_f64;
        for at in 0..3 {
            let ax = 2;
            let gp = grad_at(at, ax, h);
            let gm = grad_at(at, ax, -h);
            for bt in 0..3 {
                for by in 0..3 {
                    let fd = (gp[bt].get(by) - gm[bt].get(by)) / (2.0 * h);
                    worst = worst.max((an.hessian[(3 * at + ax, 3 * bt + by)] - fd).abs());
                }
            }
        }
        assert!(worst < 2e-5, "{label}: worst analytic-vs-FD gap {worst:e}");
    };
    check(&set, "LANL2DZ (ECP on S)");
    // The all-electron run on the identical geometry is the control: it was
    // always exact, which is what said the ECP-basis failure had to be
    // about the potential rather than anything generic.
    check(
        &data::load("6-31g").unwrap(),
        "6-31G (all-electron control)",
    );
}
/// HF: energy + full analytic gradient + three Hessian columns (central
/// differences of the analytic gradient) against the PySCF oracle.
#[test]
fn nico4_hf_energy_gradient_hessian() {
    let case = load("nico4_lanl2dz_hf_om.csv");
    let (mb, func, grid) = setup(&case);
    let o = opts();
    let res = analytic_gradient(&case.molecule, &mb, &o, &func, grid.as_ref()).unwrap();
    assert!(
        (res.scf.energy - case.e_tot).abs() < 1e-7,
        "E {} vs oracle {} (delta {:.2e})",
        res.scf.energy,
        case.e_tot,
        res.scf.energy - case.e_tot
    );
    let mut gmax = 0.0f64;
    for (g, o_g) in res.gradient.iter().zip(&case.grad) {
        gmax = gmax.max((*g - *o_g).norm());
    }
    assert!(gmax < 1e-6, "gradient max deviation {gmax:.2e}");
    // Ni sits at the Td center: its gradient must vanish by symmetry.
    assert!(
        res.gradient[0].norm() < 1e-7,
        "Ni gradient {:.2e} not symmetric",
        res.gradient[0].norm()
    );

    // Hessian columns: coordinates (Ni x), (C1 y), (O1 z).
    let hess = case.hess.as_ref().expect("HF oracle carries the Hessian");
    let h = 5e-3;
    for &col in &[0usize, 4, 17] {
        let (atom, ax) = (col / 3, col % 3);
        let displaced = |sgn: f64| -> Vec<Vec3> {
            let mut m = case.molecule.clone();
            match ax {
                0 => m.atoms[atom].position.x += sgn * h,
                1 => m.atoms[atom].position.y += sgn * h,
                _ => m.atoms[atom].position.z += sgn * h,
            }
            let set = data::load("lanl2dz").unwrap();
            let mbx = MolBasis::build(&m, &set).unwrap();
            analytic_gradient(&m, &mbx, &o, &func, None)
                .unwrap()
                .gradient
        };
        let gp = displaced(1.0);
        let gm = displaced(-1.0);
        let mut cmax = 0.0f64;
        for b in 0..case.molecule.len() {
            for by in 0..3 {
                let fd = (gp[b].get(by) - gm[b].get(by)) / (2.0 * h);
                let want = hess[(col, 3 * b + by)];
                cmax = cmax.max((fd - want).abs());
            }
        }
        assert!(
            cmax < 5e-5,
            "Hessian column {col}: max |ours(FD) - PySCF| = {cmax:.2e}"
        );
    }
}

/// B3LYP: energy + analytic gradient (grid level 8) against the oracle.
#[test]
fn nico4_b3lyp_energy_gradient() {
    let case = load("nico4_lanl2dz_b3lyp_om.csv");
    let (mb, func, grid) = setup(&case);
    let o = opts();
    let res = analytic_gradient(&case.molecule, &mb, &o, &func, grid.as_ref()).unwrap();
    assert!(
        (res.scf.energy - case.e_tot).abs() < 5e-7,
        "E {} vs oracle {} (delta {:.2e})",
        res.scf.energy,
        case.e_tot,
        res.scf.energy - case.e_tot
    );
    let mut gmax = 0.0f64;
    for (g, o_g) in res.gradient.iter().zip(&case.grad) {
        gmax = gmax.max((*g - *o_g).norm());
    }
    assert!(gmax < 1e-5, "gradient max deviation {gmax:.2e}");
}
