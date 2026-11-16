// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent dense all-pair Liouvillian and finite electric-kick fits.
//! The finite kicks are propagated by the linearized physical TDHF/TDKS
//! equation and Laplace transformed exactly, avoiding time-window errors.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{eval::eval_aos, GridOptions, MolecularGrid};
use dft_rs::integrals::{dipole_matrices, eri_incore, CoulombKernel};
use dft_rs::linalg::{symmetric_eigen, Matrix};
use dft_rs::response::complex::Complex;
use dft_rs::response::ensemble::EnsembleSolver;
use dft_rs::response::thermal_dynamic::{self, ThermalSolver};
use dft_rs::response::{Frequency, ResponseSolverConfig};
use dft_rs::scf::{run_scf, EriMode, ScfOptions, ScfResult};
use dft_rs::xc::{resolve, FunctionalSpec, GridVars, ResolvedFunctional, XcOut};
use dft_rs::{Atom, Molecule, Vec3};

#[derive(Clone, Copy)]
struct Pair {
    s: usize,
    p: usize,
    q: usize,
    df: f64,
    gap: f64,
}
struct Oracle {
    pairs: Vec<Pair>,
    kernel: Matrix,
    dipoles: [[Matrix; 3]; 2],
}

fn molecule(open: bool) -> Molecule {
    let mut m = Molecule::new(vec![
        Atom {
            z: 7,
            position: Vec3::new(0.13, -0.21, 0.19),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.51, 0.31, -0.37),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.29, 1.72, 0.63),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.42, 0.29, 2.28),
        },
    ]);
    if open {
        m.charge = 1;
        m.multiplicity = 2;
    }
    m
}

fn options(temperature: f64) -> ScfOptions {
    ScfOptions {
        electronic_temperature: temperature,
        max_iter: 400,
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-11,
        eri_mode: EriMode::InCore { limit_mb: 128 },
        ..Default::default()
    }
}

fn oracle(
    basis: &MolBasis,
    scf: &ScfResult,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
) -> Oracle {
    let n = basis.nao_sph;
    let mut pairs = Vec::new();
    for s in 0..2 {
        for p in 0..scf.mo_coeff[s].cols {
            for q in 0..scf.mo_coeff[s].cols {
                let df = scf.occupations[s][q] - scf.occupations[s][p];
                if p != q && df != 0.0 {
                    pairs.push(Pair {
                        s,
                        p,
                        q,
                        df,
                        gap: scf.mo_energy[s][p] - scf.mo_energy[s][q],
                    });
                }
            }
        }
    }
    let d = pairs.len();
    let eri = eri_incore(basis, CoulombKernel::Coulomb, 128).unwrap();
    let mut kernel = Matrix::zeros(d, d);
    // Independent explicit four-index AO contraction, including nonsymmetric
    // exchange. Neither production response nor its J/K builder is called.
    for (j, input) in pairs.iter().enumerate() {
        let c = &scf.mo_coeff[input.s];
        let mut density = Matrix::zeros(n, n);
        for l in 0..n {
            for k in 0..n {
                density[(l, k)] = c[(l, input.p)] * c[(k, input.q)];
            }
        }
        let mut coulomb = Matrix::zeros(n, n);
        let mut exchange = Matrix::zeros(n, n);
        for u in 0..n {
            for v in 0..n {
                for l in 0..n {
                    for k in 0..n {
                        coulomb[(u, v)] += eri.get(u, v, l, k) * density[(l, k)];
                        exchange[(u, v)] += eri.get(u, l, k, v) * density[(l, k)];
                    }
                }
            }
        }
        for (i, out) in pairs.iter().enumerate() {
            let co = &scf.mo_coeff[out.s];
            for u in 0..n {
                for v in 0..n {
                    kernel[(i, j)] += co[(u, out.p)]
                        * co[(v, out.q)]
                        * (coulomb[(u, v)]
                            - if out.s == input.s {
                                functional.hybrid.cx * exchange[(u, v)]
                            } else {
                                0.0
                            });
                }
            }
        }
    }
    if let Some(grid) = grid {
        // Independently differentiate the grid energy with respect to MO
        // density entries. Full product gradients, not a factor-two shortcut,
        // make the local/GGA action explicitly blind to antisymmetric density.
        let ao = eval_aos(basis, &grid.points, true);
        let phi: [Matrix; 2] = std::array::from_fn(|s| ao.phi.matmul(&scf.mo_coeff[s]));
        let grad: [[Matrix; 3]; 2] = std::array::from_fn(|s| {
            std::array::from_fn(|a| ao.grad.as_ref().unwrap()[a].matmul(&scf.mo_coeff[s]))
        });
        let m = grid.len();
        let mut rho = [vec![0.0; m], vec![0.0; m]];
        let mut g = vec![[[0.0; 3]; 2]; m];
        for t in 0..m {
            for s in 0..2 {
                for p in 0..phi[s].cols {
                    let f = scf.occupations[s][p];
                    rho[s][t] += f * phi[s][(t, p)].powi(2);
                    for a in 0..3 {
                        g[t][s][a] += 2.0 * f * phi[s][(t, p)] * grad[s][a][(t, p)];
                    }
                }
            }
        }
        let mut sigma = [vec![0.0; m], vec![0.0; m], vec![0.0; m]];
        for t in 0..m {
            for a in 0..3 {
                sigma[0][t] += g[t][0][a] * g[t][0][a];
                sigma[1][t] += g[t][0][a] * g[t][1][a];
                sigma[2][t] += g[t][1][a] * g[t][1][a];
            }
        }
        let vars = GridVars {
            rho: [&rho[0], &rho[1]],
            sigma: Some([&sigma[0], &sigma[1], &sigma[2]]),
            tau: None,
        };
        let mut out = XcOut::default();
        functional.eval(&vars, 2, &mut out).unwrap();
        for t in 0..m {
            let (v, h) = out.potential_and_kernel(t, functional.family).unwrap();
            let mut x = vec![[0.0; 5]; d];
            let mut dg = vec![[0.0; 3]; d];
            for (i, p) in pairs.iter().enumerate() {
                x[i][p.s] = phi[p.s][(t, p.p)] * phi[p.s][(t, p.q)];
                for a in 0..3 {
                    dg[i][a] = grad[p.s][a][(t, p.p)] * phi[p.s][(t, p.q)]
                        + phi[p.s][(t, p.p)] * grad[p.s][a][(t, p.q)];
                    x[i][if p.s == 0 { 2 } else { 4 }] += 2.0 * g[t][p.s][a] * dg[i][a];
                    x[i][3] += g[t][1 - p.s][a] * dg[i][a];
                }
            }
            for i in 0..d {
                for j in 0..d {
                    let mut value = 0.0;
                    for a in 0..5 {
                        for b in 0..5 {
                            value += x[i][a] * h[a][b] * x[j][b];
                        }
                    }
                    let dot: f64 = (0..3).map(|a| dg[i][a] * dg[j][a]).sum();
                    value += if pairs[i].s == pairs[j].s {
                        2.0 * v[if pairs[i].s == 0 { 2 } else { 4 }] * dot
                    } else {
                        v[3] * dot
                    };
                    kernel[(i, j)] += grid.weights[t] * value;
                }
            }
        }
    }
    let ao_dipoles = dipole_matrices(basis, Vec3::zero());
    let dipoles = std::array::from_fn(|s| {
        std::array::from_fn(|a| {
            scf.mo_coeff[s]
                .transpose_matmul(&ao_dipoles[a])
                .matmul(&scf.mo_coeff[s])
        })
    });
    Oracle {
        pairs,
        kernel,
        dipoles,
    }
}

fn dense_solve(mut a: Vec<Vec<Complex>>, mut b: Vec<Complex>) -> Vec<Complex> {
    let n = b.len();
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))
            .unwrap();
        assert!(a[pivot][k].abs() > 1e-13);
        a.swap(k, pivot);
        b.swap(k, pivot);
        for i in k + 1..n {
            let factor = a[i][k] / a[k][k];
            for j in k..n {
                let v = a[k][j];
                a[i][j] -= factor * v;
            }
            let v = b[k];
            b[i] -= factor * v;
        }
    }
    let mut x = vec![Complex::ZERO; n];
    for i in (0..n).rev() {
        let mut v = b[i];
        for j in i + 1..n {
            v -= a[i][j] * x[j];
        }
        x[i] = v / a[i][i];
    }
    x
}

impl Oracle {
    fn matrix(&self, z: Complex) -> Vec<Vec<Complex>> {
        self.pairs
            .iter()
            .enumerate()
            .map(|(i, p)| {
                self.pairs
                    .iter()
                    .enumerate()
                    .map(|(j, _)| {
                        Complex::from(-p.df * self.kernel[(i, j)])
                            + if i == j {
                                z - Complex::from(p.gap)
                            } else {
                                Complex::ZERO
                            }
                    })
                    .collect()
            })
            .collect()
    }
    fn alpha(&self, z: Complex, scf: &ScfResult, kick: Option<f64>) -> [[Complex; 3]; 3] {
        let mut result = [[Complex::ZERO; 3]; 3];
        for b in 0..3 {
            let mut kick_rhs: Vec<Matrix> = Vec::new();
            if let Some(step) = kick {
                for s in 0..2 {
                    let (e, u) = symmetric_eigen(&self.dipoles[s][b]).unwrap();
                    let n = e.len();
                    let mut f = Matrix::zeros(n, n);
                    for p in 0..n {
                        f[(p, p)] = scf.occupations[s][p];
                    }
                    let f = u.transpose_matmul(&f).matmul(&u);
                    let mut imaginary = Matrix::zeros(n, n);
                    for p in 0..n {
                        for q in 0..n {
                            let plus = -(step * (e[p] - e[q])).sin() * f[(p, q)];
                            imaginary[(p, q)] =
                                -stencils::Stencil::Second.first(plus, -plus, 0.0, 0.0, step);
                        }
                    }
                    kick_rhs.push(u.matmul(&imaginary).matmul(&u.transpose()));
                }
            }
            let rhs = self
                .pairs
                .iter()
                .map(|p| {
                    Complex::from(if kick.is_some() {
                        kick_rhs[p.s][(p.p, p.q)]
                    } else {
                        p.df * self.dipoles[p.s][b][(p.p, p.q)]
                    })
                })
                .collect();
            let x = dense_solve(self.matrix(z), rhs);
            for a in 0..3 {
                for (p, v) in self.pairs.iter().zip(&x) {
                    result[a][b] -= *v * self.dipoles[p.s][a][(p.q, p.p)];
                }
            }
        }
        result
    }
}

#[test]
fn all_complex_components_dense_and_four_step_electric_kicks() {
    let steps = [0.08, 0.04, 0.02, 0.01];
    let mut failures = Vec::new();
    for name in ["hf", "pbe"] {
        for open in [false, true] {
            if std::env::var("DFT_RS_TEST_MODEL")
                .ok()
                .is_some_and(|v| v != name)
            {
                continue;
            }
            let mol = molecule(open);
            let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
            let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
            let grid = if functional.parts.is_empty() {
                None
            } else {
                Some(
                    MolecularGrid::build(
                        &mol,
                        &GridOptions {
                            level: 0,
                            weight_cutoff: 0.0,
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                )
            };
            let opts = options(0.18);
            let scf = run_scf(&mol, &basis, &opts, &functional, grid.as_ref()).unwrap();
            assert!(scf.converged);
            let oracle = oracle(&basis, &scf, &functional, grid.as_ref());
            let config = ResponseSolverConfig {
                tolerance: 1e-12,
                ..Default::default()
            };
            // A nonsymmetric source makes an exchange-index transpose visible
            // independently of the symmetric electric observable contractions.
            let n = basis.nao_sph;
            let potential: [Matrix; 2] = std::array::from_fn(|s| {
                let mut v = Matrix::zeros(n, n);
                for i in 0..n {
                    for j in 0..n {
                        v[(i, j)] = 0.03 * ((i + 3 * j + 2 * s + 1) as f64).sin();
                    }
                }
                v
            });
            let probe = Frequency::Retarded {
                omega: 0.21,
                damping: 0.13,
            };
            let response =
                ThermalSolver::new(&mol, &basis, &functional, grid.as_ref(), &scf, &opts)
                    .unwrap()
                    .solve(&potential, probe, &config)
                    .unwrap();
            let vmo: [Matrix; 2] = std::array::from_fn(|s| {
                scf.mo_coeff[s]
                    .transpose_matmul(&potential[s])
                    .matmul(&scf.mo_coeff[s])
            });
            let rhs = oracle
                .pairs
                .iter()
                .map(|p| Complex::from(p.df * vmo[p.s][(p.p, p.q)]))
                .collect();
            let x = dense_solve(oracle.matrix(probe.energy().unwrap()), rhs);
            let actual_re: [Matrix; 2] = std::array::from_fn(|s| {
                let sc = scf.overlap.matmul(&scf.mo_coeff[s]);
                sc.transpose_matmul(&response.real[s]).matmul(&sc)
            });
            let actual_im: [Matrix; 2] = std::array::from_fn(|s| {
                let sc = scf.overlap.matmul(&scf.mo_coeff[s]);
                sc.transpose_matmul(&response.imaginary[s]).matmul(&sc)
            });
            for (p, expected) in oracle.pairs.iter().zip(x) {
                assert!(
                    (Complex::new(actual_re[p.s][(p.p, p.q)], actual_im[p.s][(p.p, p.q)])
                        - expected)
                        .abs()
                        < 2e-8
                );
            }
            for frequency in [
                Frequency::Retarded {
                    omega: 0.21,
                    damping: 0.13,
                },
                Frequency::Imaginary { frequency: 0.37 },
                Frequency::Static,
                // A free Liouvillian denominator is exactly zero here, while
                // the interacting dense operator remains nonsingular. The
                // identity preconditioner entry must not shift that operator.
                Frequency::Retarded {
                    omega: scf.mo_energy[0][0] - scf.mo_energy[0][1],
                    damping: 0.0,
                },
            ] {
                let native = thermal_dynamic::polarizability(
                    &mol,
                    &basis,
                    &functional,
                    grid.as_ref(),
                    &scf,
                    &opts,
                    frequency,
                    &config,
                )
                .unwrap();
                let z = frequency.energy().unwrap();
                let expected = oracle.alpha(z, &scf, None);
                let kicks: Vec<_> = steps
                    .iter()
                    .map(|&h| oracle.alpha(z, &scf, Some(h)))
                    .collect();
                for a in 0..3 {
                    for b in 0..3 {
                        let actual = native.response.value.alpha[a][b];
                        assert!((actual-expected[a][b]).abs()<2e-8,"{name} open{open} {frequency:?} ({a},{b}) actual{actual:?} expected{:?}",expected[a][b]);
                        let h = steps[3];
                        let fourth = Complex::new(
                            stencils::Stencil::Fourth.first(
                                h * kicks[3][a][b].re,
                                -h * kicks[3][a][b].re,
                                2.0 * h * kicks[2][a][b].re,
                                -2.0 * h * kicks[2][a][b].re,
                                h,
                            ),
                            stencils::Stencil::Fourth.first(
                                h * kicks[3][a][b].im,
                                -h * kicks[3][a][b].im,
                                2.0 * h * kicks[2][a][b].im,
                                -2.0 * h * kicks[2][a][b].im,
                                h,
                            ),
                        );
                        assert!(
                            (fourth - actual).abs() < (kicks[3][a][b] - actual).abs(),
                            "fourth-order kick must improve every component"
                        );
                        for imaginary in [false, true] {
                            if imaginary
                                && (!matches!(frequency, Frequency::Retarded { .. }) || z.im == 0.0)
                            {
                                continue;
                            }
                            let errors = std::array::from_fn(|k| {
                                let delta = kicks[k][a][b] - actual;
                                if imaginary {
                                    delta.im.abs()
                                } else {
                                    delta.re.abs()
                                }
                            });
                            let label = format!(
                                "{name} open{open} {frequency:?} alpha{a}{b} {} kick",
                                if imaginary { "imaginary" } else { "real" }
                            );
                            stencils::fit_components_staged(
                                &[label],
                                &[(steps, stencils::Stencil::Second)],
                                |_, _| vec![errors],
                                &mut failures,
                            );
                        }
                    }
                }
                assert!(native.number_defects.iter().flatten().all(|v| *v < 1e-11));
                assert!(native.response.value.reciprocity_defect < 2e-8);
                if z.re == 0.0 || z.im == 0.0 {
                    assert!(native
                        .response
                        .value
                        .alpha
                        .iter()
                        .flatten()
                        .all(|z| z.im.abs() < 1e-10));
                }
                if let Frequency::Retarded { omega, damping } = frequency {
                    let negative = thermal_dynamic::polarizability(
                        &mol,
                        &basis,
                        &functional,
                        grid.as_ref(),
                        &scf,
                        &opts,
                        Frequency::Retarded {
                            omega: -omega,
                            damping,
                        },
                        &config,
                    )
                    .unwrap();
                    for a in 0..3 {
                        for b in 0..3 {
                            assert!(
                                (negative.response.value.alpha[a][b]
                                    - native.response.value.alpha[a][b].conj())
                                .abs()
                                    < 2e-9
                            );
                        }
                    }
                }
            }
            let frozen = thermal_dynamic::polarizability(
                &mol,
                &basis,
                &functional,
                grid.as_ref(),
                &scf,
                &opts,
                Frequency::Static,
                &config,
            )
            .unwrap();
            let isothermal =
                EnsembleSolver::new(&mol, &basis, &functional, grid.as_ref(), &scf, &opts)
                    .unwrap()
                    .polarizability(&config)
                    .unwrap();
            let difference = (0..3)
                .flat_map(|a| (0..3).map(move |b| (a, b)))
                .map(|(a, b)| {
                    (frozen.response.value.alpha[a][b].re - isothermal.tensor[a][b]).abs()
                })
                .fold(0.0_f64, f64::max);
            println!("{name} open{open} collisionless/isothermal maximumdifference {difference:e}");
            assert!(difference > 1e-5);
        }
    }
    components::assert_no_failures("frequency-specific finite electric kicks", &failures);
}

#[test]
fn zero_temperature_limit_and_reference_rejections() {
    let mol = molecule(false);
    let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let functional = ResolvedFunctional::hartree_fock();
    let opts = options(0.0);
    let scf = run_scf(&mol, &basis, &opts, &functional, None).unwrap();
    assert!(ThermalSolver::new(&mol, &basis, &functional, None, &scf, &opts).is_err());
    let warm = options(0.001);
    let mut thermal = run_scf(&mol, &basis, &warm, &functional, None).unwrap();
    let frequency = Frequency::Retarded {
        omega: 0.21,
        damping: 0.13,
    };
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let a = thermal_dynamic::polarizability(
        &mol,
        &basis,
        &functional,
        None,
        &thermal,
        &warm,
        frequency,
        &config,
    )
    .unwrap();
    assert!(
        dft_rs::response::dynamic_polarizability(
            &mol,
            &basis,
            &functional,
            None,
            &thermal,
            &warm,
            frequency,
            &config,
        )
        .is_err(),
        "legacy integer response must still refuse a thermal reference"
    );
    // Mixed custom providers are checked by their actual family capabilities,
    // even if a caller supplies an incorrect lower-rung aggregate annotation.
    let meta = resolve(&FunctionalSpec::Combined("tpss".into())).unwrap();
    let mixed = ResolvedFunctional::custom(
        "mixed provider",
        meta.parts.clone(),
        dft_rs::xc::HybridParams::PURE,
        dft_rs::xc::Family::Gga,
    );
    assert!(ThermalSolver::new(&mol, &basis, &mixed, None, &thermal, &warm).is_err());
    let b = dft_rs::response::dynamic_polarizability(
        &mol,
        &basis,
        &functional,
        None,
        &scf,
        &opts,
        frequency,
        &config,
    )
    .unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!((a.response.value.alpha[i][j] - b.value.alpha[i][j]).abs() < 2e-8);
        }
    }
    thermal.occupations[0][0] -= 0.01;
    assert!(ThermalSolver::new(&mol, &basis, &functional, None, &thermal, &warm).is_err());
}

#[test]
fn equal_population_degenerate_subspace_needs_no_gap_regularization() {
    let mol = Molecule::new(vec![Atom {
        z: 2,
        position: Vec3::zero(),
    }]);
    let basis = MolBasis::build(&mol, &data::load("cc-pvdz").unwrap()).unwrap();
    let functional = ResolvedFunctional::hartree_fock();
    let opts = options(0.2);
    let mut scf = run_scf(&mol, &basis, &opts, &functional, None).unwrap();
    // Restore the exact atomic p degeneracy from floating-point eigensolver
    // splitting, far below the measured reference tolerance. No finite gap
    // or unequal population is replaced or regularized by production code.
    for s in 0..2 {
        let e = scf.mo_energy[s][2..5].iter().sum::<f64>() / 3.0;
        let f = scf.occupations[s][2..5].iter().sum::<f64>() / 3.0;
        assert!(scf.mo_energy[s][2..5].iter().all(|x| (x - e).abs() < 1e-12));
        for p in 2..5 {
            scf.mo_energy[s][p] = e;
            scf.occupations[s][p] = f;
        }
    }
    let result = thermal_dynamic::polarizability(
        &mol,
        &basis,
        &functional,
        None,
        &scf,
        &opts,
        Frequency::Static,
        &ResponseSolverConfig {
            tolerance: 1e-12,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(result
        .response
        .value
        .alpha
        .iter()
        .flatten()
        .all(|x| x.is_finite()));
    assert!(result.number_defects.iter().flatten().all(|x| *x < 1e-12));
}

struct Primal<'a> {
    n: usize,
    linear: Matrix,
    core: [Matrix; 2],
    phi: [Matrix; 2],
    grad: [[Matrix; 3]; 2],
    weights: Vec<f64>,
    functional: &'a ResolvedFunctional,
}

impl<'a> Primal<'a> {
    fn new(
        basis: &MolBasis,
        scf: &ScfResult,
        functional: &'a ResolvedFunctional,
        grid: &MolecularGrid,
    ) -> Self {
        let n = basis.nao_sph;
        let n2 = n * n;
        let d = 2 * n2;
        let eri = eri_incore(basis, CoulombKernel::Coulomb, 128).unwrap();
        let mut linear = Matrix::zeros(d, d);
        for si in 0..2 {
            for p in 0..n {
                for q in 0..n {
                    let c = &scf.mo_coeff[si];
                    let j = si * n2 + p * n + q;
                    let mut coulomb = Matrix::zeros(n, n);
                    let mut exchange = Matrix::zeros(n, n);
                    for u in 0..n {
                        for v in 0..n {
                            for l in 0..n {
                                for k in 0..n {
                                    let density = c[(l, p)] * c[(k, q)];
                                    coulomb[(u, v)] += eri.get(u, v, l, k) * density;
                                    exchange[(u, v)] += eri.get(u, l, k, v) * density;
                                }
                            }
                        }
                    }
                    for so in 0..2 {
                        let c = &scf.mo_coeff[so];
                        for r in 0..n {
                            for t in 0..n {
                                let i = so * n2 + r * n + t;
                                for u in 0..n {
                                    for v in 0..n {
                                        linear[(i, j)] += c[(u, r)]
                                            * c[(v, t)]
                                            * (coulomb[(u, v)]
                                                - if so == si {
                                                    functional.hybrid.cx * exchange[(u, v)]
                                                } else {
                                                    0.0
                                                });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let ao = eval_aos(basis, &grid.points, true);
        Self {
            n,
            linear,
            core: std::array::from_fn(|s| {
                scf.mo_coeff[s]
                    .transpose_matmul(&scf.h_core)
                    .matmul(&scf.mo_coeff[s])
            }),
            phi: std::array::from_fn(|s| ao.phi.matmul(&scf.mo_coeff[s])),
            grad: std::array::from_fn(|s| {
                std::array::from_fn(|a| ao.grad.as_ref().unwrap()[a].matmul(&scf.mo_coeff[s]))
            }),
            weights: grid.weights.clone(),
            functional,
        }
    }

    fn fock_energy(&self, p: &[Complex]) -> (Vec<Complex>, f64) {
        let n = self.n;
        let n2 = n * n;
        let d = 2 * n2;
        let mut f = vec![Complex::ZERO; d];
        let mut energy = 0.0;
        for i in 0..d {
            f[i] = Complex::from(self.core[i / n2][(i % n2 / n, i % n)]);
            for j in 0..d {
                f[i] += p[j] * self.linear[(i, j)];
            }
        }
        for s in 0..2 {
            for r in 0..n {
                for q in 0..n {
                    let i = s * n2 + r * n + q;
                    let trans = s * n2 + q * n + r;
                    energy += (p[trans] * (f[i] + Complex::from(self.core[s][(r, q)]))).re * 0.5;
                }
            }
        }
        if !self.functional.parts.is_empty() {
            let m = self.weights.len();
            let mut rho = [vec![0.0; m], vec![0.0; m]];
            let mut g = vec![[[0.0; 3]; 2]; m];
            for t in 0..m {
                for s in 0..2 {
                    for r in 0..n {
                        for q in 0..n {
                            let density = p[s * n2 + r * n + q].re;
                            rho[s][t] += density * self.phi[s][(t, r)] * self.phi[s][(t, q)];
                            for a in 0..3 {
                                g[t][s][a] += density
                                    * (self.grad[s][a][(t, r)] * self.phi[s][(t, q)]
                                        + self.phi[s][(t, r)] * self.grad[s][a][(t, q)]);
                            }
                        }
                    }
                }
            }
            let mut sigma = [vec![0.0; m], vec![0.0; m], vec![0.0; m]];
            for t in 0..m {
                for a in 0..3 {
                    sigma[0][t] += g[t][0][a].powi(2);
                    sigma[1][t] += g[t][0][a] * g[t][1][a];
                    sigma[2][t] += g[t][1][a].powi(2);
                }
            }
            let mut out = XcOut::default();
            self.functional
                .eval(
                    &GridVars {
                        rho: [&rho[0], &rho[1]],
                        sigma: Some([&sigma[0], &sigma[1], &sigma[2]]),
                        tau: None,
                    },
                    1,
                    &mut out,
                )
                .unwrap();
            for t in 0..m {
                energy += self.weights[t] * out.e[t];
                for s in 0..2 {
                    for r in 0..n {
                        for q in 0..n {
                            let mut value =
                                out.vrho[s][t] * self.phi[s][(t, r)] * self.phi[s][(t, q)];
                            for a in 0..3 {
                                value +=
                                    (2.0 * out.vsigma[if s == 0 { 0 } else { 2 }][t] * g[t][s][a]
                                        + out.vsigma[1][t] * g[t][1 - s][a])
                                        * (self.grad[s][a][(t, r)] * self.phi[s][(t, q)]
                                            + self.phi[s][(t, r)] * self.grad[s][a][(t, q)]);
                            }
                            f[s * n2 + r * n + q].re += self.weights[t] * value;
                        }
                    }
                }
            }
        }
        (f, energy)
    }

    fn derivative(&self, p: &[Complex]) -> Vec<Complex> {
        let (f, _) = self.fock_energy(p);
        let n = self.n;
        let n2 = n * n;
        let mut out = vec![Complex::ZERO; p.len()];
        for s in 0..2 {
            for i in 0..n {
                for j in 0..n {
                    let mut comm = Complex::ZERO;
                    for k in 0..n {
                        comm += f[s * n2 + i * n + k] * p[s * n2 + k * n + j]
                            - p[s * n2 + i * n + k] * f[s * n2 + k * n + j];
                    }
                    out[s * n2 + i * n + j] = Complex::new(comm.im, -comm.re);
                }
            }
        }
        out
    }

    fn step(&self, p: &[Complex], dt: f64) -> Vec<Complex> {
        let k1 = self.derivative(p);
        let trial = |k: &[Complex], scale: f64| {
            p.iter()
                .zip(k)
                .map(|(a, b)| *a + *b * scale)
                .collect::<Vec<_>>()
        };
        let k2 = self.derivative(&trial(&k1, dt * 0.5));
        let k3 = self.derivative(&trial(&k2, dt * 0.5));
        let k4 = self.derivative(&trial(&k3, dt));
        (0..p.len())
            .map(|i| p[i] + (k1[i] + k2[i] * 2.0 + k3[i] * 2.0 + k4[i]) * (dt / 6.0))
            .collect()
    }
}

#[derive(Default)]
struct Drift {
    number: f64,
    hermiticity: f64,
    purity: f64,
    energy: f64,
}

fn nonlinear_trajectory(
    primal: &Primal<'_>,
    scf: &ScfResult,
    dipoles: &[[Matrix; 3]; 2],
    axis: usize,
    kick: f64,
    dt: f64,
    steps: usize,
    z: Complex,
) -> ([Complex; 3], Drift) {
    let n = primal.n;
    let n2 = n * n;
    let mut p = vec![Complex::ZERO; 2 * n2];
    let mut baseline = [0.0; 3];
    let mut initial_purity = [0.0; 2];
    for s in 0..2 {
        let (e, u) = symmetric_eigen(&dipoles[s][axis]).unwrap();
        let mut pop = Matrix::zeros(n, n);
        for i in 0..n {
            pop[(i, i)] = scf.occupations[s][i];
            initial_purity[s] += scf.occupations[s][i].powi(2);
            for a in 0..3 {
                baseline[a] -= dipoles[s][a][(i, i)] * scf.occupations[s][i];
            }
        }
        let pop = u.transpose_matmul(&pop).matmul(&u);
        let mut re = Matrix::zeros(n, n);
        let mut im = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let angle = kick * (e[i] - e[j]);
                re[(i, j)] = pop[(i, j)] * angle.cos();
                im[(i, j)] = -pop[(i, j)] * angle.sin();
            }
        }
        let re = u.matmul(&re).matmul(&u.transpose());
        let im = u.matmul(&im).matmul(&u.transpose());
        for i in 0..n {
            for j in 0..n {
                p[s * n2 + i * n + j] = Complex::new(re[(i, j)], im[(i, j)]);
            }
        }
    }
    let initial_energy = primal.fock_energy(&p).1;
    let mut drift = Drift::default();
    let mut integral = [Complex::ZERO; 3];
    for t in 0..=steps {
        let time = t as f64 * dt;
        let weight = if t == 0 || t == steps {
            1.0
        } else if t % 2 == 1 {
            4.0
        } else {
            2.0
        };
        let laplace = Complex::new((z.re * time).cos(), (z.re * time).sin())
            * ((-z.im * time).exp() * weight * dt / 3.0);
        let mut dipole = [0.0; 3];
        for s in 0..2 {
            let mut number = Complex::ZERO;
            let mut purity = Complex::ZERO;
            for i in 0..n {
                number += p[s * n2 + i * n + i];
                for j in 0..n {
                    let a = p[s * n2 + i * n + j];
                    let b = p[s * n2 + j * n + i];
                    purity += a * b;
                    drift.hermiticity = drift.hermiticity.max((a - b.conj()).abs());
                    for axis in 0..3 {
                        dipole[axis] -= dipoles[s][axis][(j, i)] * a.re;
                    }
                }
            }
            let expected = scf.occupations[s].iter().sum::<f64>();
            drift.number = drift.number.max((number - Complex::from(expected)).abs());
            drift.purity = drift
                .purity
                .max((purity - Complex::from(initial_purity[s])).abs());
        }
        for a in 0..3 {
            integral[a] += laplace * (dipole[a] - baseline[a]);
        }
        if t < steps {
            p = primal.step(&p, dt);
        }
    }
    drift.energy = (primal.fock_energy(&p).1 - initial_energy).abs();
    (integral, drift)
}

#[test]
fn full_nonlinear_time_domain_all_nine_timestep_fits() {
    let mol = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.13, -0.21, 0.19),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.57, 0.16, -0.08),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.19, 1.93, 0.37),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.42, 0.29, 2.28),
        },
    ]);
    let grid = MolecularGrid {
        points: vec![
            Vec3::new(0.3, 0.2, 0.1),
            Vec3::new(1.1, -0.3, 0.4),
            Vec3::new(-0.5, 1.2, 0.7),
            Vec3::new(0.6, 0.4, 1.5),
            Vec3::new(-0.2, -0.7, 0.9),
        ],
        weights: vec![0.8, 1.1, 0.7, 0.9, 1.2],
        raw_weights: vec![0.8, 1.1, 0.7, 0.9, 1.2],
        atom_of: vec![0, 1, 2, 3, 0],
    };
    let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let dt = [0.16, 0.08, 0.04, 0.02];
    let kick = 2e-5;
    let frequency = Frequency::Retarded {
        omega: 0.31,
        damping: 1.2,
    };
    let z = frequency.energy().unwrap();
    let mut failures = Vec::new();
    for name in ["hf", "pbe"] {
        if std::env::var("DFT_RS_TEST_MODEL")
            .ok()
            .is_some_and(|v| v != name)
        {
            continue;
        }
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let grid_ref = if functional.parts.is_empty() {
            None
        } else {
            Some(&grid)
        };
        let opts = options(0.22);
        let scf = run_scf(&mol, &basis, &opts, &functional, grid_ref).unwrap();
        assert!(scf.converged);
        let config = ResponseSolverConfig {
            tolerance: 1e-12,
            ..Default::default()
        };
        let native = thermal_dynamic::polarizability(
            &mol,
            &basis,
            &functional,
            grid_ref,
            &scf,
            &opts,
            frequency,
            &config,
        )
        .unwrap();
        let primal = Primal::new(&basis, &scf, &functional, &grid);
        let ao_dipoles = dipole_matrices(&basis, Vec3::zero());
        let dipoles: [[Matrix; 3]; 2] = std::array::from_fn(|s| {
            std::array::from_fn(|a| {
                scf.mo_coeff[s]
                    .transpose_matmul(&ao_dipoles[a])
                    .matmul(&scf.mo_coeff[s])
            })
        });
        let mut measured = [[[Complex::ZERO; 3]; 3]; 4];
        for k in 0..4 {
            for b in 0..3 {
                let steps = 256usize << k;
                let (plus, dp) =
                    nonlinear_trajectory(&primal, &scf, &dipoles, b, kick, dt[k], steps, z);
                let (minus, dm) =
                    nonlinear_trajectory(&primal, &scf, &dipoles, b, -kick, dt[k], steps, z);
                for a in 0..3 {
                    measured[k][a][b] = (plus[a] - minus[a]) / (2.0 * kick);
                }
                println!("{name} nonlinear dt={} axis{b} number={:e} Hermiticity={:e} purity={:e} energy={:e}",dt[k],dp.number.max(dm.number),dp.hermiticity.max(dm.hermiticity),dp.purity.max(dm.purity),dp.energy.max(dm.energy));
                assert!(dp.number.max(dm.number) < 1e-11);
                assert!(dp.hermiticity.max(dm.hermiticity) < 1e-11);
                if k == 3 {
                    let (half_plus, _) = nonlinear_trajectory(
                        &primal,
                        &scf,
                        &dipoles,
                        b,
                        kick * 0.5,
                        dt[k],
                        steps,
                        z,
                    );
                    let (half_minus, _) = nonlinear_trajectory(
                        &primal,
                        &scf,
                        &dipoles,
                        b,
                        -kick * 0.5,
                        dt[k],
                        steps,
                        z,
                    );
                    for a in 0..3 {
                        let half = (half_plus[a] - half_minus[a]) / kick;
                        println!(
                            "{name} nonlinear alpha{a}{b} finite-kick difference {:e}",
                            (half - measured[k][a][b]).abs()
                        );
                        assert!(
                            (half - measured[k][a][b]).abs()
                                < (measured[k][a][b] - native.response.value.alpha[a][b]).abs()
                        );
                    }
                }
            }
        }
        // A conservative operator-norm bound, independent of the observed
        // oscillatory signal, controls the discarded tail for every component.
        let max_r = dipoles
            .iter()
            .flatten()
            .flat_map(|m| m.as_slice())
            .fold(0.0_f64, |a, v| a.max(v.abs()));
        let tail = 2.0 * max_r * 4.0 * 4.0 / kick * (-z.im * 40.96).exp() / z.im;
        println!("{name} nonlinear Laplace tail bound {tail:e}");
        assert!(tail < 1e-12);
        for a in 0..3 {
            for b in 0..3 {
                for imaginary in [false, true] {
                    let errors = std::array::from_fn(|k| {
                        let delta = measured[k][a][b] - native.response.value.alpha[a][b];
                        if imaginary {
                            delta.im.abs()
                        } else {
                            delta.re.abs()
                        }
                    });
                    let label = format!(
                        "{name} full nonlinear alpha{a}{b} {} timestep",
                        if imaginary { "imag" } else { "real" }
                    );
                    if !components::component_fit_of_order(&label, &dt, &errors, 4.0) {
                        failures.push(label);
                    }
                }
            }
        }
    }
    components::assert_no_failures("full nonlinear time-domain response", &failures);
}
