// SPDX-License-Identifier: Apache-2.0
//! Staged shared-context oracles. This synthetic scalar AO operator tests
//! transport/coupled response and complete Hessian extraction, NOT chemical
//! fragment accuracy or the independent Gaussian/projector pair kernel.
use super::*;
use crate::acceptance::components;
use crate::basis::{data, MolBasis};
use crate::grid::{GridOptions, MolecularGrid};
use crate::integrals::overlap_matrix;
use crate::math::Vec3;
use crate::scf::{run_scf, EriMode, Guess, Reference, ScfOptions};
use crate::system::{Atom, Molecule};
use crate::xc::{resolve, FunctionalSpec};

const W: [f64; 11] = [
    0.31, -0.23, 0.19, -0.12, 0.41, -0.27, -0.19, -0.18, 0.08, 0.71, -0.43,
];
fn molecule(open: bool, x: &[f64; 11]) -> Molecule {
    let p = [
        [0.13, -0.21, 0.17],
        [1.34, 0.82, -0.23],
        [-0.74, 1.13, 0.91],
    ];
    Molecule::new(
        (0..3)
            .map(|a| Atom {
                z: if a == 0 { 8 } else { 1 },
                position: Vec3::new(
                    p[a][0] + x[3 * a],
                    p[a][1] + x[3 * a + 1],
                    p[a][2] + x[3 * a + 2],
                ),
            })
            .collect(),
    )
    .with_charge(i32::from(open))
    .with_multiplicity(if open { 2 } else { 1 })
}
fn amplitude(x: &[f64; 11]) -> f64 {
    0.17 * (0.36 + W.iter().zip(x).map(|(a, b)| a * b).sum::<f64>()).sin()
}
fn options(open: bool) -> ScfOptions {
    ScfOptions {
        reference: if open { Reference::Uhf } else { Reference::Rhf },
        guess: Guess::Core,
        max_iter: 300,
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-11,
        eri_mode: EriMode::InCore { limit_mb: 64 },
        ..ScfOptions::default()
    }
}
fn grid(mol: &Molecule) -> MolecularGrid {
    MolecularGrid::build(
        mol,
        &GridOptions {
            level: 1,
            weight_cutoff: 0.,
            ..GridOptions::default()
        },
    )
    .unwrap()
}
fn energy(f: &str, open: bool, x: &[f64; 11]) -> f64 {
    let mol = molecule(open, x);
    let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let quadrature = grid(&mol);
    let functional = resolve(&FunctionalSpec::Combined(f.into())).unwrap();
    let mut opts = options(open);
    let mut extra = overlap_matrix(&basis);
    for v in extra.as_mut_slice() {
        *v *= amplitude(x);
    }
    opts.extra_hcore = Some(extra);
    let state = run_scf(&mol, &basis, &opts, &functional, Some(&quadrature)).unwrap();
    assert!(state.converged);
    state.energy
}
fn directional(f: &str, open: bool, d: [f64; 11]) -> f64 {
    let x = [0.; 11];
    let mol = molecule(open, &x);
    let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let quadrature = grid(&mol);
    let functional = resolve(&FunctionalSpec::Combined(f.into())).unwrap();
    let mut opts = options(open);
    let mut zero = overlap_matrix(&basis);
    for v in zero.as_mut_slice() {
        *v *= amplitude(&x);
    }
    opts.extra_hcore = Some(zero);
    let state = run_scf(&mol, &basis, &opts, &functional, Some(&quadrature)).unwrap();
    let direction = MolecularDirection::nuclear(
        (0..3)
            .map(|a| [d[3 * a], d[3 * a + 1], d[3 * a + 2]])
            .collect(),
    );
    let base = GeometryTaylor::new(
        &mol,
        &basis,
        Some(&quadrature),
        &functional,
        None,
        &direction.nuclear,
        [0.; 3],
        [0.; 3],
        2,
        opts.incore_mb,
    )
    .unwrap();
    let speed = W.iter().zip(d).map(|(w, v)| w * v).sum::<f64>();
    let a = [
        amplitude(&x),
        0.17 * 0.36_f64.cos() * speed,
        -0.5 * amplitude(&x) * speed * speed,
    ];
    let mut op = vec![Matrix::zeros(basis.nao_sph, basis.nao_sph); 3];
    for n in 0..3 {
        for k in 0..=n {
            super::super::add(&mut op[n], &base.overlap[n - k], a[k]);
        }
    }
    let identity = ExtraCoreIdentity {
        chart: 11,
        operator_layout: 23,
        grid_recipe: 1,
        coordinate_map: 0,
    };
    let source = ExtraCoreTaylor::new(
        &mol,
        &basis,
        &functional,
        Some(&quadrature),
        &state,
        &opts,
        &direction,
        &d[9..],
        identity.clone(),
        ExtraCoreCoordinates::IndependentRoles,
        [op.clone(), op],
        ExternalClassicalTaylor {
            interaction: vec![0.; 3],
            internal_energy_reference: 0.,
            permanent_dipole: vec![[0.; 3]; 3],
        },
        [0.; 3],
        ExtraCoreFieldConvention::ProvidedCompleteDeltaFromOrdinaryOriginZero,
        2,
    )
    .unwrap();
    let bound = source
        .geometry(
            &mol,
            &basis,
            &functional,
            Some(&quadrature),
            &state,
            &opts,
            &direction,
            &d[9..],
            &identity,
        )
        .unwrap();
    let config = ResponseSolverConfig {
        tolerance: 1e-10,
        ..ResponseSolverConfig::default()
    };
    let result = molecular_taylor_with_extra_core(
        &mol,
        &basis,
        &functional,
        Some(&quadrature),
        &state,
        &opts,
        &direction,
        &d[9..],
        &identity,
        &source,
        2,
        &config,
        bound,
    )
    .unwrap();
    assert!(
        result.response.converged,
        "source diagnostics {:?}",
        result.response.warnings
    );
    assert_eq!(result.external_classical_coefficients, vec![0.; 3]);
    2. * result.response.value.energy_coefficients[2]
}
fn component(f: &str, open: bool, i: usize, j: usize) {
    let mut di = [0.; 11];
    di[i] = 1.;
    let analytic = if i == j {
        directional(f, open, di)
    } else {
        let mut dj = [0.; 11];
        dj[j] = 1.;
        let mut sum = di;
        sum[j] = 1.;
        0.5 * (directional(f, open, sum) - directional(f, open, di) - directional(f, open, dj))
    };
    let label = format!("extra-context {f} open={open} E[{i},{j}]");
    let mut failures = Vec::new();
    components::fit_components(
        std::slice::from_ref(&label),
        &[[0.04, 0.02, 0.01, 0.005], [0.02, 0.01, 0.005, 0.0025]],
        |steps| {
            let mut errors = [0.; 4];
            for (k, &h) in steps.iter().enumerate() {
                let measured = if i == j {
                    let mut p = [0.; 11];
                    p[i] = h;
                    let mut m = [0.; 11];
                    m[i] = -h;
                    (energy(f, open, &p) - 2. * energy(f, open, &[0.; 11]) + energy(f, open, &m))
                        / (h * h)
                } else {
                    let mut values = [0.; 4];
                    for (n, (a, b)) in [(1., 1.), (1., -1.), (-1., 1.), (-1., -1.)]
                        .into_iter()
                        .enumerate()
                    {
                        let mut x = [0.; 11];
                        x[i] = a * h;
                        x[j] = b * h;
                        values[n] = energy(f, open, &x);
                    }
                    (values[0] - values[1] - values[2] + values[3]) / (4. * h * h)
                };
                errors[k] = (measured - analytic).abs();
            }
            vec![errors]
        },
        &mut failures,
    );
    components::assert_no_failures(&label, &failures);
}
#[test]
fn context_lda_rks_00_00() {
    component("lda", false, 0, 0);
}
#[test]
fn context_lda_rks_00_01() {
    component("lda", false, 0, 1);
}
#[test]
fn context_lda_rks_00_02() {
    component("lda", false, 0, 2);
}
#[test]
fn context_lda_rks_00_03() {
    component("lda", false, 0, 3);
}
#[test]
fn context_lda_rks_00_04() {
    component("lda", false, 0, 4);
}
#[test]
fn context_lda_rks_00_05() {
    component("lda", false, 0, 5);
}
#[test]
fn context_lda_rks_00_06() {
    component("lda", false, 0, 6);
}
#[test]
fn context_lda_rks_00_07() {
    component("lda", false, 0, 7);
}
#[test]
fn context_lda_rks_00_08() {
    component("lda", false, 0, 8);
}
#[test]
fn context_lda_rks_00_09() {
    component("lda", false, 0, 9);
}
#[test]
fn context_lda_rks_00_10() {
    component("lda", false, 0, 10);
}
#[test]
fn context_lda_rks_01_01() {
    component("lda", false, 1, 1);
}
#[test]
fn context_lda_rks_01_02() {
    component("lda", false, 1, 2);
}
#[test]
fn context_lda_rks_01_03() {
    component("lda", false, 1, 3);
}
#[test]
fn context_lda_rks_01_04() {
    component("lda", false, 1, 4);
}
#[test]
fn context_lda_rks_01_05() {
    component("lda", false, 1, 5);
}
#[test]
fn context_lda_rks_01_06() {
    component("lda", false, 1, 6);
}
#[test]
fn context_lda_rks_01_07() {
    component("lda", false, 1, 7);
}
#[test]
fn context_lda_rks_01_08() {
    component("lda", false, 1, 8);
}
#[test]
fn context_lda_rks_01_09() {
    component("lda", false, 1, 9);
}
#[test]
fn context_lda_rks_01_10() {
    component("lda", false, 1, 10);
}
#[test]
fn context_lda_rks_02_02() {
    component("lda", false, 2, 2);
}
#[test]
fn context_lda_rks_02_03() {
    component("lda", false, 2, 3);
}
#[test]
fn context_lda_rks_02_04() {
    component("lda", false, 2, 4);
}
#[test]
fn context_lda_rks_02_05() {
    component("lda", false, 2, 5);
}
#[test]
fn context_lda_rks_02_06() {
    component("lda", false, 2, 6);
}
#[test]
fn context_lda_rks_02_07() {
    component("lda", false, 2, 7);
}
#[test]
fn context_lda_rks_02_08() {
    component("lda", false, 2, 8);
}
#[test]
fn context_lda_rks_02_09() {
    component("lda", false, 2, 9);
}
#[test]
fn context_lda_rks_02_10() {
    component("lda", false, 2, 10);
}
#[test]
fn context_lda_rks_03_03() {
    component("lda", false, 3, 3);
}
#[test]
fn context_lda_rks_03_04() {
    component("lda", false, 3, 4);
}
#[test]
fn context_lda_rks_03_05() {
    component("lda", false, 3, 5);
}
#[test]
fn context_lda_rks_03_06() {
    component("lda", false, 3, 6);
}
#[test]
fn context_lda_rks_03_07() {
    component("lda", false, 3, 7);
}
#[test]
fn context_lda_rks_03_08() {
    component("lda", false, 3, 8);
}
#[test]
fn context_lda_rks_03_09() {
    component("lda", false, 3, 9);
}
#[test]
fn context_lda_rks_03_10() {
    component("lda", false, 3, 10);
}
#[test]
fn context_lda_rks_04_04() {
    component("lda", false, 4, 4);
}
#[test]
fn context_lda_rks_04_05() {
    component("lda", false, 4, 5);
}
#[test]
fn context_lda_rks_04_06() {
    component("lda", false, 4, 6);
}
#[test]
fn context_lda_rks_04_07() {
    component("lda", false, 4, 7);
}
#[test]
fn context_lda_rks_04_08() {
    component("lda", false, 4, 8);
}
#[test]
fn context_lda_rks_04_09() {
    component("lda", false, 4, 9);
}
#[test]
fn context_lda_rks_04_10() {
    component("lda", false, 4, 10);
}
#[test]
fn context_lda_rks_05_05() {
    component("lda", false, 5, 5);
}
#[test]
fn context_lda_rks_05_06() {
    component("lda", false, 5, 6);
}
#[test]
fn context_lda_rks_05_07() {
    component("lda", false, 5, 7);
}
#[test]
fn context_lda_rks_05_08() {
    component("lda", false, 5, 8);
}
#[test]
fn context_lda_rks_05_09() {
    component("lda", false, 5, 9);
}
#[test]
fn context_lda_rks_05_10() {
    component("lda", false, 5, 10);
}
#[test]
fn context_lda_rks_06_06() {
    component("lda", false, 6, 6);
}
#[test]
fn context_lda_rks_06_07() {
    component("lda", false, 6, 7);
}
#[test]
fn context_lda_rks_06_08() {
    component("lda", false, 6, 8);
}
#[test]
fn context_lda_rks_06_09() {
    component("lda", false, 6, 9);
}
#[test]
fn context_lda_rks_06_10() {
    component("lda", false, 6, 10);
}
#[test]
fn context_lda_rks_07_07() {
    component("lda", false, 7, 7);
}
#[test]
fn context_lda_rks_07_08() {
    component("lda", false, 7, 8);
}
#[test]
fn context_lda_rks_07_09() {
    component("lda", false, 7, 9);
}
#[test]
fn context_lda_rks_07_10() {
    component("lda", false, 7, 10);
}
#[test]
fn context_lda_rks_08_08() {
    component("lda", false, 8, 8);
}
#[test]
fn context_lda_rks_08_09() {
    component("lda", false, 8, 9);
}
#[test]
fn context_lda_rks_08_10() {
    component("lda", false, 8, 10);
}
#[test]
fn context_lda_rks_09_09() {
    component("lda", false, 9, 9);
}
#[test]
fn context_lda_rks_09_10() {
    component("lda", false, 9, 10);
}
#[test]
fn context_lda_rks_10_10() {
    component("lda", false, 10, 10);
}
#[test]
fn context_lda_uks_00_00() {
    component("lda", true, 0, 0);
}
#[test]
fn context_lda_uks_00_01() {
    component("lda", true, 0, 1);
}
#[test]
fn context_lda_uks_00_02() {
    component("lda", true, 0, 2);
}
#[test]
fn context_lda_uks_00_03() {
    component("lda", true, 0, 3);
}
#[test]
fn context_lda_uks_00_04() {
    component("lda", true, 0, 4);
}
#[test]
fn context_lda_uks_00_05() {
    component("lda", true, 0, 5);
}
#[test]
fn context_lda_uks_00_06() {
    component("lda", true, 0, 6);
}
#[test]
fn context_lda_uks_00_07() {
    component("lda", true, 0, 7);
}
#[test]
fn context_lda_uks_00_08() {
    component("lda", true, 0, 8);
}
#[test]
fn context_lda_uks_00_09() {
    component("lda", true, 0, 9);
}
#[test]
fn context_lda_uks_00_10() {
    component("lda", true, 0, 10);
}
#[test]
fn context_lda_uks_01_01() {
    component("lda", true, 1, 1);
}
#[test]
fn context_lda_uks_01_02() {
    component("lda", true, 1, 2);
}
#[test]
fn context_lda_uks_01_03() {
    component("lda", true, 1, 3);
}
#[test]
fn context_lda_uks_01_04() {
    component("lda", true, 1, 4);
}
#[test]
fn context_lda_uks_01_05() {
    component("lda", true, 1, 5);
}
#[test]
fn context_lda_uks_01_06() {
    component("lda", true, 1, 6);
}
#[test]
fn context_lda_uks_01_07() {
    component("lda", true, 1, 7);
}
#[test]
fn context_lda_uks_01_08() {
    component("lda", true, 1, 8);
}
#[test]
fn context_lda_uks_01_09() {
    component("lda", true, 1, 9);
}
#[test]
fn context_lda_uks_01_10() {
    component("lda", true, 1, 10);
}
#[test]
fn context_lda_uks_02_02() {
    component("lda", true, 2, 2);
}
#[test]
fn context_lda_uks_02_03() {
    component("lda", true, 2, 3);
}
#[test]
fn context_lda_uks_02_04() {
    component("lda", true, 2, 4);
}
#[test]
fn context_lda_uks_02_05() {
    component("lda", true, 2, 5);
}
#[test]
fn context_lda_uks_02_06() {
    component("lda", true, 2, 6);
}
#[test]
fn context_lda_uks_02_07() {
    component("lda", true, 2, 7);
}
#[test]
fn context_lda_uks_02_08() {
    component("lda", true, 2, 8);
}
#[test]
fn context_lda_uks_02_09() {
    component("lda", true, 2, 9);
}
#[test]
fn context_lda_uks_02_10() {
    component("lda", true, 2, 10);
}
#[test]
fn context_lda_uks_03_03() {
    component("lda", true, 3, 3);
}
#[test]
fn context_lda_uks_03_04() {
    component("lda", true, 3, 4);
}
#[test]
fn context_lda_uks_03_05() {
    component("lda", true, 3, 5);
}
#[test]
fn context_lda_uks_03_06() {
    component("lda", true, 3, 6);
}
#[test]
fn context_lda_uks_03_07() {
    component("lda", true, 3, 7);
}
#[test]
fn context_lda_uks_03_08() {
    component("lda", true, 3, 8);
}
#[test]
fn context_lda_uks_03_09() {
    component("lda", true, 3, 9);
}
#[test]
fn context_lda_uks_03_10() {
    component("lda", true, 3, 10);
}
#[test]
fn context_lda_uks_04_04() {
    component("lda", true, 4, 4);
}
#[test]
fn context_lda_uks_04_05() {
    component("lda", true, 4, 5);
}
#[test]
fn context_lda_uks_04_06() {
    component("lda", true, 4, 6);
}
#[test]
fn context_lda_uks_04_07() {
    component("lda", true, 4, 7);
}
#[test]
fn context_lda_uks_04_08() {
    component("lda", true, 4, 8);
}
#[test]
fn context_lda_uks_04_09() {
    component("lda", true, 4, 9);
}
#[test]
fn context_lda_uks_04_10() {
    component("lda", true, 4, 10);
}
#[test]
fn context_lda_uks_05_05() {
    component("lda", true, 5, 5);
}
#[test]
fn context_lda_uks_05_06() {
    component("lda", true, 5, 6);
}
#[test]
fn context_lda_uks_05_07() {
    component("lda", true, 5, 7);
}
#[test]
fn context_lda_uks_05_08() {
    component("lda", true, 5, 8);
}
#[test]
fn context_lda_uks_05_09() {
    component("lda", true, 5, 9);
}
#[test]
fn context_lda_uks_05_10() {
    component("lda", true, 5, 10);
}
#[test]
fn context_lda_uks_06_06() {
    component("lda", true, 6, 6);
}
#[test]
fn context_lda_uks_06_07() {
    component("lda", true, 6, 7);
}
#[test]
fn context_lda_uks_06_08() {
    component("lda", true, 6, 8);
}
#[test]
fn context_lda_uks_06_09() {
    component("lda", true, 6, 9);
}
#[test]
fn context_lda_uks_06_10() {
    component("lda", true, 6, 10);
}
#[test]
fn context_lda_uks_07_07() {
    component("lda", true, 7, 7);
}
#[test]
fn context_lda_uks_07_08() {
    component("lda", true, 7, 8);
}
#[test]
fn context_lda_uks_07_09() {
    component("lda", true, 7, 9);
}
#[test]
fn context_lda_uks_07_10() {
    component("lda", true, 7, 10);
}
#[test]
fn context_lda_uks_08_08() {
    component("lda", true, 8, 8);
}
#[test]
fn context_lda_uks_08_09() {
    component("lda", true, 8, 9);
}
#[test]
fn context_lda_uks_08_10() {
    component("lda", true, 8, 10);
}
#[test]
fn context_lda_uks_09_09() {
    component("lda", true, 9, 9);
}
#[test]
fn context_lda_uks_09_10() {
    component("lda", true, 9, 10);
}
#[test]
fn context_lda_uks_10_10() {
    component("lda", true, 10, 10);
}
#[test]
fn context_pbe_rks_00_00() {
    component("pbe", false, 0, 0);
}
#[test]
fn context_pbe_rks_00_01() {
    component("pbe", false, 0, 1);
}
#[test]
fn context_pbe_rks_00_02() {
    component("pbe", false, 0, 2);
}
#[test]
fn context_pbe_rks_00_03() {
    component("pbe", false, 0, 3);
}
#[test]
fn context_pbe_rks_00_04() {
    component("pbe", false, 0, 4);
}
#[test]
fn context_pbe_rks_00_05() {
    component("pbe", false, 0, 5);
}
#[test]
fn context_pbe_rks_00_06() {
    component("pbe", false, 0, 6);
}
#[test]
fn context_pbe_rks_00_07() {
    component("pbe", false, 0, 7);
}
#[test]
fn context_pbe_rks_00_08() {
    component("pbe", false, 0, 8);
}
#[test]
fn context_pbe_rks_00_09() {
    component("pbe", false, 0, 9);
}
#[test]
fn context_pbe_rks_00_10() {
    component("pbe", false, 0, 10);
}
#[test]
fn context_pbe_rks_01_01() {
    component("pbe", false, 1, 1);
}
#[test]
fn context_pbe_rks_01_02() {
    component("pbe", false, 1, 2);
}
#[test]
fn context_pbe_rks_01_03() {
    component("pbe", false, 1, 3);
}
#[test]
fn context_pbe_rks_01_04() {
    component("pbe", false, 1, 4);
}
#[test]
fn context_pbe_rks_01_05() {
    component("pbe", false, 1, 5);
}
#[test]
fn context_pbe_rks_01_06() {
    component("pbe", false, 1, 6);
}
#[test]
fn context_pbe_rks_01_07() {
    component("pbe", false, 1, 7);
}
#[test]
fn context_pbe_rks_01_08() {
    component("pbe", false, 1, 8);
}
#[test]
fn context_pbe_rks_01_09() {
    component("pbe", false, 1, 9);
}
#[test]
fn context_pbe_rks_01_10() {
    component("pbe", false, 1, 10);
}
#[test]
fn context_pbe_rks_02_02() {
    component("pbe", false, 2, 2);
}
#[test]
fn context_pbe_rks_02_03() {
    component("pbe", false, 2, 3);
}
#[test]
fn context_pbe_rks_02_04() {
    component("pbe", false, 2, 4);
}
#[test]
fn context_pbe_rks_02_05() {
    component("pbe", false, 2, 5);
}
#[test]
fn context_pbe_rks_02_06() {
    component("pbe", false, 2, 6);
}
#[test]
fn context_pbe_rks_02_07() {
    component("pbe", false, 2, 7);
}
#[test]
fn context_pbe_rks_02_08() {
    component("pbe", false, 2, 8);
}
#[test]
fn context_pbe_rks_02_09() {
    component("pbe", false, 2, 9);
}
#[test]
fn context_pbe_rks_02_10() {
    component("pbe", false, 2, 10);
}
#[test]
fn context_pbe_rks_03_03() {
    component("pbe", false, 3, 3);
}
#[test]
fn context_pbe_rks_03_04() {
    component("pbe", false, 3, 4);
}
#[test]
fn context_pbe_rks_03_05() {
    component("pbe", false, 3, 5);
}
#[test]
fn context_pbe_rks_03_06() {
    component("pbe", false, 3, 6);
}
#[test]
fn context_pbe_rks_03_07() {
    component("pbe", false, 3, 7);
}
#[test]
fn context_pbe_rks_03_08() {
    component("pbe", false, 3, 8);
}
#[test]
fn context_pbe_rks_03_09() {
    component("pbe", false, 3, 9);
}
#[test]
fn context_pbe_rks_03_10() {
    component("pbe", false, 3, 10);
}
#[test]
fn context_pbe_rks_04_04() {
    component("pbe", false, 4, 4);
}
#[test]
fn context_pbe_rks_04_05() {
    component("pbe", false, 4, 5);
}
#[test]
fn context_pbe_rks_04_06() {
    component("pbe", false, 4, 6);
}
#[test]
fn context_pbe_rks_04_07() {
    component("pbe", false, 4, 7);
}
#[test]
fn context_pbe_rks_04_08() {
    component("pbe", false, 4, 8);
}
#[test]
fn context_pbe_rks_04_09() {
    component("pbe", false, 4, 9);
}
#[test]
fn context_pbe_rks_04_10() {
    component("pbe", false, 4, 10);
}
#[test]
fn context_pbe_rks_05_05() {
    component("pbe", false, 5, 5);
}
#[test]
fn context_pbe_rks_05_06() {
    component("pbe", false, 5, 6);
}
#[test]
fn context_pbe_rks_05_07() {
    component("pbe", false, 5, 7);
}
#[test]
fn context_pbe_rks_05_08() {
    component("pbe", false, 5, 8);
}
#[test]
fn context_pbe_rks_05_09() {
    component("pbe", false, 5, 9);
}
#[test]
fn context_pbe_rks_05_10() {
    component("pbe", false, 5, 10);
}
#[test]
fn context_pbe_rks_06_06() {
    component("pbe", false, 6, 6);
}
#[test]
fn context_pbe_rks_06_07() {
    component("pbe", false, 6, 7);
}
#[test]
fn context_pbe_rks_06_08() {
    component("pbe", false, 6, 8);
}
#[test]
fn context_pbe_rks_06_09() {
    component("pbe", false, 6, 9);
}
#[test]
fn context_pbe_rks_06_10() {
    component("pbe", false, 6, 10);
}
#[test]
fn context_pbe_rks_07_07() {
    component("pbe", false, 7, 7);
}
#[test]
fn context_pbe_rks_07_08() {
    component("pbe", false, 7, 8);
}
#[test]
fn context_pbe_rks_07_09() {
    component("pbe", false, 7, 9);
}
#[test]
fn context_pbe_rks_07_10() {
    component("pbe", false, 7, 10);
}
#[test]
fn context_pbe_rks_08_08() {
    component("pbe", false, 8, 8);
}
#[test]
fn context_pbe_rks_08_09() {
    component("pbe", false, 8, 9);
}
#[test]
fn context_pbe_rks_08_10() {
    component("pbe", false, 8, 10);
}
#[test]
fn context_pbe_rks_09_09() {
    component("pbe", false, 9, 9);
}
#[test]
fn context_pbe_rks_09_10() {
    component("pbe", false, 9, 10);
}
#[test]
fn context_pbe_rks_10_10() {
    component("pbe", false, 10, 10);
}
#[test]
fn context_pbe_uks_00_00() {
    component("pbe", true, 0, 0);
}
#[test]
fn context_pbe_uks_00_01() {
    component("pbe", true, 0, 1);
}
#[test]
fn context_pbe_uks_00_02() {
    component("pbe", true, 0, 2);
}
#[test]
fn context_pbe_uks_00_03() {
    component("pbe", true, 0, 3);
}
#[test]
fn context_pbe_uks_00_04() {
    component("pbe", true, 0, 4);
}
#[test]
fn context_pbe_uks_00_05() {
    component("pbe", true, 0, 5);
}
#[test]
fn context_pbe_uks_00_06() {
    component("pbe", true, 0, 6);
}
#[test]
fn context_pbe_uks_00_07() {
    component("pbe", true, 0, 7);
}
#[test]
fn context_pbe_uks_00_08() {
    component("pbe", true, 0, 8);
}
#[test]
fn context_pbe_uks_00_09() {
    component("pbe", true, 0, 9);
}
#[test]
fn context_pbe_uks_00_10() {
    component("pbe", true, 0, 10);
}
#[test]
fn context_pbe_uks_01_01() {
    component("pbe", true, 1, 1);
}
#[test]
fn context_pbe_uks_01_02() {
    component("pbe", true, 1, 2);
}
#[test]
fn context_pbe_uks_01_03() {
    component("pbe", true, 1, 3);
}
#[test]
fn context_pbe_uks_01_04() {
    component("pbe", true, 1, 4);
}
#[test]
fn context_pbe_uks_01_05() {
    component("pbe", true, 1, 5);
}
#[test]
fn context_pbe_uks_01_06() {
    component("pbe", true, 1, 6);
}
#[test]
fn context_pbe_uks_01_07() {
    component("pbe", true, 1, 7);
}
#[test]
fn context_pbe_uks_01_08() {
    component("pbe", true, 1, 8);
}
#[test]
fn context_pbe_uks_01_09() {
    component("pbe", true, 1, 9);
}
#[test]
fn context_pbe_uks_01_10() {
    component("pbe", true, 1, 10);
}
#[test]
fn context_pbe_uks_02_02() {
    component("pbe", true, 2, 2);
}
#[test]
fn context_pbe_uks_02_03() {
    component("pbe", true, 2, 3);
}
#[test]
fn context_pbe_uks_02_04() {
    component("pbe", true, 2, 4);
}
#[test]
fn context_pbe_uks_02_05() {
    component("pbe", true, 2, 5);
}
#[test]
fn context_pbe_uks_02_06() {
    component("pbe", true, 2, 6);
}
#[test]
fn context_pbe_uks_02_07() {
    component("pbe", true, 2, 7);
}
#[test]
fn context_pbe_uks_02_08() {
    component("pbe", true, 2, 8);
}
#[test]
fn context_pbe_uks_02_09() {
    component("pbe", true, 2, 9);
}
#[test]
fn context_pbe_uks_02_10() {
    component("pbe", true, 2, 10);
}
#[test]
fn context_pbe_uks_03_03() {
    component("pbe", true, 3, 3);
}
#[test]
fn context_pbe_uks_03_04() {
    component("pbe", true, 3, 4);
}
#[test]
fn context_pbe_uks_03_05() {
    component("pbe", true, 3, 5);
}
#[test]
fn context_pbe_uks_03_06() {
    component("pbe", true, 3, 6);
}
#[test]
fn context_pbe_uks_03_07() {
    component("pbe", true, 3, 7);
}
#[test]
fn context_pbe_uks_03_08() {
    component("pbe", true, 3, 8);
}
#[test]
fn context_pbe_uks_03_09() {
    component("pbe", true, 3, 9);
}
#[test]
fn context_pbe_uks_03_10() {
    component("pbe", true, 3, 10);
}
#[test]
fn context_pbe_uks_04_04() {
    component("pbe", true, 4, 4);
}
#[test]
fn context_pbe_uks_04_05() {
    component("pbe", true, 4, 5);
}
#[test]
fn context_pbe_uks_04_06() {
    component("pbe", true, 4, 6);
}
#[test]
fn context_pbe_uks_04_07() {
    component("pbe", true, 4, 7);
}
#[test]
fn context_pbe_uks_04_08() {
    component("pbe", true, 4, 8);
}
#[test]
fn context_pbe_uks_04_09() {
    component("pbe", true, 4, 9);
}
#[test]
fn context_pbe_uks_04_10() {
    component("pbe", true, 4, 10);
}
#[test]
fn context_pbe_uks_05_05() {
    component("pbe", true, 5, 5);
}
#[test]
fn context_pbe_uks_05_06() {
    component("pbe", true, 5, 6);
}
#[test]
fn context_pbe_uks_05_07() {
    component("pbe", true, 5, 7);
}
#[test]
fn context_pbe_uks_05_08() {
    component("pbe", true, 5, 8);
}
#[test]
fn context_pbe_uks_05_09() {
    component("pbe", true, 5, 9);
}
#[test]
fn context_pbe_uks_05_10() {
    component("pbe", true, 5, 10);
}
#[test]
fn context_pbe_uks_06_06() {
    component("pbe", true, 6, 6);
}
#[test]
fn context_pbe_uks_06_07() {
    component("pbe", true, 6, 7);
}
#[test]
fn context_pbe_uks_06_08() {
    component("pbe", true, 6, 8);
}
#[test]
fn context_pbe_uks_06_09() {
    component("pbe", true, 6, 9);
}
#[test]
fn context_pbe_uks_06_10() {
    component("pbe", true, 6, 10);
}
#[test]
fn context_pbe_uks_07_07() {
    component("pbe", true, 7, 7);
}
#[test]
fn context_pbe_uks_07_08() {
    component("pbe", true, 7, 8);
}
#[test]
fn context_pbe_uks_07_09() {
    component("pbe", true, 7, 9);
}
#[test]
fn context_pbe_uks_07_10() {
    component("pbe", true, 7, 10);
}
#[test]
fn context_pbe_uks_08_08() {
    component("pbe", true, 8, 8);
}
#[test]
fn context_pbe_uks_08_09() {
    component("pbe", true, 8, 9);
}
#[test]
fn context_pbe_uks_08_10() {
    component("pbe", true, 8, 10);
}
#[test]
fn context_pbe_uks_09_09() {
    component("pbe", true, 9, 9);
}
#[test]
fn context_pbe_uks_09_10() {
    component("pbe", true, 9, 10);
}
#[test]
fn context_pbe_uks_10_10() {
    component("pbe", true, 10, 10);
}
