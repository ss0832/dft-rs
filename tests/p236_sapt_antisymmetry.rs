// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Independent four-electron permutation oracle for SAPT(0).

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{aos::sph_block, cart_components, data, MolBasis};
use dft_rs::grid::{gauss_legendre, GridOptions, MolecularGrid};
use dft_rs::integrals::eri::{eri_incore, EriTensor};
use dft_rs::integrals::one_electron::{
    dipole_matrices, kinetic_matrix, nuclear_matrix, overlap_matrix,
};
use dft_rs::integrals::CoulombKernel;
use dft_rs::linalg::{solve_linear, solve_linear_matrix, symmetric_eigen, Matrix};
use dft_rs::math::Vec3;
use dft_rs::response::tddft::{TdaSolver, TdaSpin};
use dft_rs::response::ResponseSolver;
use dft_rs::sapt::{
    sapt_interaction_diagnostics, sapt_interaction_regularized,
    sapt_interaction_regularized_variant, sapt_interaction_variant, SaptConfig,
    SaptCorrelationTerms, SaptDiagnostics, SaptResult, SaptThirdOrder, SaptVariant,
};
use dft_rs::scf::{run_scf, EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{built_in_dispersion, resolve, FunctionalSpec, ResolvedFunctional};

type Orbitals = [usize; 4];
const PERMUTATIONS: [([usize; 4], f64); 4] = [
    ([0, 1, 2, 3], 1.0),
    ([1, 0, 2, 3], -1.0),
    ([0, 1, 3, 2], -1.0),
    ([1, 0, 3, 2], 1.0),
];

struct OrbitalIntegrals {
    overlap: Matrix,
    external: [Matrix; 2],
    full_core: Matrix,
    eri: Vec<f64>,
    nuclear: [f64; 2],
}

impl OrbitalIntegrals {
    fn eri(&self, a: usize, b: usize, c: usize, d: usize) -> f64 {
        let n = self.overlap.rows;
        self.eri[((a * n + b) * n + c) * n + d]
    }

    fn permutation(
        &self,
        bra: Orbitals,
        ket: Orbitals,
        perm: [usize; 4],
        full: bool,
    ) -> (f64, f64) {
        let ket = perm.map(|i| ket[i]);
        let overlaps: [f64; 4] = std::array::from_fn(|i| self.overlap[(bra[i], ket[i])]);
        let spectator = |i, j| {
            (0..4)
                .filter(|&k| k != i && k != j)
                .map(|k| overlaps[k])
                .product::<f64>()
        };
        let overlap = overlaps.iter().product::<f64>();
        let mut numerator = overlap * self.nuclear[usize::from(full)];
        for i in 0..4 {
            let operator = if full {
                &self.full_core
            } else {
                &self.external[1 - i % 2]
            };
            numerator += operator[(bra[i], ket[i])] * spectator(i, i);
            for j in i + 1..4 {
                if full || i % 2 != j % 2 {
                    numerator += self.eri(bra[i], ket[i], bra[j], ket[j]) * spectator(i, j);
                }
            }
        }
        (numerator, overlap)
    }

    fn matrix_element(&self, bra: Orbitals, ket: Orbitals, full: bool) -> (f64, f64) {
        PERMUTATIONS
            .iter()
            .fold((0.0, 0.0), |(n, s), &(perm, sign)| {
                let (numerator, overlap) = self.permutation(bra, ket, perm, full);
                (n + sign * numerator, s + sign * overlap)
            })
    }
}

fn transform_eri(eri: &EriTensor, c: &Matrix) -> Vec<f64> {
    let n = c.rows;
    let index = |a, b, d, e| ((a * n + b) * n + d) * n + e;
    let mut previous = vec![0.0; n.pow(4)];
    let mut next = previous.clone();
    for a in 0..n {
        for b in 0..n {
            for d in 0..n {
                for e in 0..n {
                    previous[index(a, b, d, e)] = eri.get(a, b, d, e);
                }
            }
        }
    }
    for axis in 0..4 {
        for (flat, value) in next.iter_mut().enumerate() {
            let mut indices = [flat / n.pow(3), flat / n.pow(2) % n, flat / n % n, flat % n];
            let mo = indices[axis];
            *value = (0..n)
                .map(|ao| {
                    indices[axis] = ao;
                    previous[index(indices[0], indices[1], indices[2], indices[3])] * c[(ao, mo)]
                })
                .sum();
        }
        std::mem::swap(&mut previous, &mut next);
    }
    previous
}

fn direct_gaussian_moment(centers: [f64; 3], exponents: [f64; 3], powers: [u32; 2]) -> f64 {
    let p = exponents.iter().sum::<f64>();
    let center = centers
        .iter()
        .zip(exponents)
        .map(|(&x, e)| x * e)
        .sum::<f64>()
        / p;
    let exponent = centers
        .iter()
        .zip(exponents)
        .map(|(&x, e)| e * (x - center).powi(2))
        .sum::<f64>();
    let choose = |n: u32, k: u32| {
        (0..k)
            .map(|i| f64::from(n - i) / f64::from(i + 1))
            .product::<f64>()
    };
    let mut moment = 0.0;
    for i in 0..=powers[0] {
        for j in 0..=powers[1] {
            if (i + j) % 2 != 0 {
                continue;
            }
            let even_moment = (1..=(i + j) / 2)
                .map(|k| f64::from(2 * k - 1) / (2.0 * p))
                .product::<f64>();
            moment += choose(powers[0], i)
                * (center - centers[0]).powi((powers[0] - i) as i32)
                * choose(powers[1], j)
                * (center - centers[1]).powi((powers[1] - j) as i32)
                * even_moment;
        }
    }
    moment * (-exponent).exp() * (std::f64::consts::PI / p).sqrt()
}

/// Independent Laplace integral of Gaussian overlaps for e^(-eta r²)/r.
/// Cartesian powers are integrated by ordinary Gaussian moments instead
/// of the production translated-Hermite and Boys recurrence.
fn penetration_quadrature(basis: &MolBasis, partner: &Molecule, eta: f64, n: usize) -> Matrix {
    let (nodes, weights) = gauss_legendre(n);
    let prefactor = 2.0 / std::f64::consts::PI.sqrt();
    let mut result = Matrix::zeros(basis.nao_sph, basis.nao_sph);
    for a in &basis.shells {
        let a_powers = cart_components(a.l);
        for b in &basis.shells {
            let b_powers = cart_components(b.l);
            let mut block = Matrix::zeros(a_powers.len(), b_powers.len());
            for (a_id, &(ax, ay, az)) in a_powers.iter().enumerate() {
                for (b_id, &(bx, by, bz)) in b_powers.iter().enumerate() {
                    let pa = [ax, ay, az];
                    let pb = [bx, by, bz];
                    for (&ea, &ca) in a.exps.iter().zip(&a.coefs) {
                        for (&eb, &cb) in b.exps.iter().zip(&b.coefs) {
                            for atom in &partner.atoms {
                                for (&node, &weight) in nodes.iter().zip(&weights) {
                                    let x = (1.0 + node) * 0.5;
                                    let t = x / (1.0 - x);
                                    let integral = (0..3)
                                        .map(|i| {
                                            direct_gaussian_moment(
                                                [
                                                    a.center.get(i),
                                                    b.center.get(i),
                                                    atom.position.get(i),
                                                ],
                                                [ea, eb, eta + t * t],
                                                [pa[i] as u32, pb[i] as u32],
                                            )
                                        })
                                        .product::<f64>();
                                    block[(a_id, b_id)] -=
                                        f64::from(atom.z) * ca * cb * prefactor * weight * 0.5
                                            / (1.0 - x).powi(2)
                                            * integral;
                                }
                            }
                        }
                    }
                }
            }
            let sph = sph_block(a.l, b.l, &block);
            for i in 0..sph.rows {
                for j in 0..sph.cols {
                    result[(a.sph_off + i, b.sph_off + j)] = sph[(i, j)];
                }
            }
        }
    }
    result
}

fn h2(points: [[f64; 3]; 2]) -> Molecule {
    Molecule::new(
        points
            .into_iter()
            .map(|p| Atom {
                z: 1,
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    )
}

#[test]
fn kohn_sham_fdds_and_exchange_match_dense_reference() {
    let monomers = [
        h2([[0.04, 0.12, -0.06], [1.41, -0.31, 0.63]]),
        h2([[1.02, 0.77, 4.62], [1.67, 1.86, 5.15]]),
    ];
    let set = data::load("6-31g").unwrap();
    let basis = monomers
        .each_ref()
        .map(|m| MolBasis::build(m, &set).unwrap());
    let dimer = Molecule::new(monomers.iter().flat_map(|m| m.atoms.clone()).collect());
    let combined_basis = MolBasis::build(&dimer, &set).unwrap();
    let eri = eri_incore(&combined_basis, CoulombKernel::Coulomb, 256).unwrap();
    let offsets = [0, basis[0].nao_sph];
    let n = combined_basis.nao_sph;
    let dimension = basis[0].nao_sph - 1;
    let g = GridOptions {
        level: 1,
        weight_cutoff: 0.0,
        ..GridOptions::default()
    };
    let grids = monomers
        .each_ref()
        .map(|m| MolecularGrid::build(m, &g).unwrap());
    for (name, shifts) in [
        ("pbe", [Some(0.08), Some(0.08)]),
        ("wb97x-d", [None, None]),
        ("wb97x-v", [None, None]),
        ("wb97m-v", [None, None]),
    ] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let o = ScfOptions {
            vv10: built_in_dispersion(name).0,
            vv10_grid_level: 0,
            vv10_grid_weight_cutoff: 0.0,
            ..reference()
        };
        let scf: [_; 2] = std::array::from_fn(|i| {
            run_scf(
                &monomers[i],
                &basis[i],
                &ScfOptions {
                    asymptotic_correction: shifts[i],
                    ..o.clone()
                },
                &functional,
                Some(&grids[i]),
            )
            .unwrap()
        });
        let mut c = Matrix::zeros(n, n);
        for (i, b) in basis.iter().enumerate() {
            for p in 0..b.nao_sph {
                for q in 0..b.nao_sph {
                    c[(offsets[i] + p, offsets[i] + q)] = scf[i].mo_coeff[0][(p, q)];
                }
            }
        }
        let transform = |op: &Matrix| c.transpose_matmul(&op.matmul(&c));
        let integrals = OrbitalIntegrals {
            overlap: transform(&overlap_matrix(&combined_basis)),
            external: [
                transform(&nuclear_matrix(
                    &combined_basis,
                    &dimer,
                    Some(&[1.0, 1.0, 0.0, 0.0]),
                )),
                transform(&nuclear_matrix(
                    &combined_basis,
                    &dimer,
                    Some(&[0.0, 0.0, 1.0, 1.0]),
                )),
            ],
            full_core: Matrix::zeros(0, 0),
            eri: transform_eri(&eri, &c),
            nuclear: [
                dimer.nuclear_repulsion()
                    - monomers
                        .iter()
                        .map(Molecule::nuclear_repulsion)
                        .sum::<f64>(),
                0.0,
            ],
        };
        let bra = [0, offsets[1], 0, offsets[1]];
        let identity = PERMUTATIONS[0].0;
        let (elst, _) = integrals.permutation(bra, bra, identity, false);
        let (numerator, norm) = integrals.matrix_element(bra, bra, false);
        let first = numerator / norm;
        let hessians: [_; 2] = std::array::from_fn(|i| {
            TdaSolver::new(
                &monomers[i],
                &basis[i],
                &functional,
                Some(&grids[i]),
                &scf[i],
                &o,
                TdaSpin::Singlet,
            )
            .unwrap()
            .orbital_hessians()
            .unwrap()
        });
        let mut expected_induction = [0.0; 2];
        let mut expected_exchange_induction = [0.0; 2];
        let mut expected_regularized = [[0.0; 2]; 2];
        let eta = 3.0;
        for i in 0..2 {
            let offset = offsets[i];
            let j = offsets[1 - i];
            let sources = (1..=dimension)
                .map(|v| {
                    integrals.external[1 - i][(offset + v, offset)]
                        + 2.0 * integrals.eri(offset + v, offset, j, j)
                })
                .collect::<Vec<_>>();
            let amplitudes = solve_linear(&hessians[i].0, &sources).unwrap();
            let c = &scf[i].mo_coeff[0];
            let short_range = penetration_quadrature(&basis[i], &monomers[1 - i], eta, 128);
            let short_mo = c.transpose_matmul(&short_range.matmul(c));
            let regularized_sources = (0..dimension)
                .map(|a| sources[a] - short_mo[(a + 1, 0)])
                .collect::<Vec<_>>();
            let regularized_amplitudes =
                solve_linear(&hessians[i].0, &regularized_sources).unwrap();
            for (a, &u) in amplitudes.iter().enumerate() {
                let mut sum = 0.0;
                for slot in [i, i + 2] {
                    let mut ket = bra;
                    ket[slot] = offset + a + 1;
                    let (m, s) = integrals.matrix_element(bra, ket, false);
                    sum += (m - first * s) / norm;
                }
                expected_induction[i] -= 2.0 * u * sources[a];
                expected_exchange_induction[i] -= u * (sum - 2.0 * sources[a]);
                let relaxed = regularized_amplitudes[a];
                expected_regularized[i][0] -= 2.0 * relaxed * sources[a];
                expected_regularized[i][1] -= relaxed * (sum - 2.0 * sources[a]);
            }
        }
        let mut coupling = Matrix::zeros(dimension, dimension);
        let mut uncoupled = 0.0;
        let mut exchange_dispersion = 0.0;
        for a in 1..=dimension {
            for b in 1..=dimension {
                let v = integrals.eri(0, a, offsets[1], offsets[1] + b);
                coupling[(a - 1, b - 1)] = v;
                let gap = scf[0].mo_energy[0][a] - scf[0].mo_energy[0][0] + scf[1].mo_energy[0][b]
                    - scf[1].mo_energy[0][0];
                uncoupled -= 4.0 * v * v / gap;
                for slot_a in [0, 2] {
                    for slot_b in [1, 3] {
                        let mut ket = bra;
                        ket[slot_a] = a;
                        ket[slot_b] = offsets[1] + b;
                        let (m, s) = integrals.matrix_element(bra, ket, false);
                        exchange_dispersion -= v * ((m - first * s) / norm - v) / gap;
                    }
                }
            }
        }
        let integral = |points| {
            let (nodes, weights) = gauss_legendre(points);
            let response = |i: usize, frequency: f64| {
                let (plus, minus) = &hessians[i];
                let mut lhs = minus.matmul(plus);
                for p in 0..dimension {
                    lhs[(p, p)] += frequency * frequency;
                }
                solve_linear_matrix(&lhs, minus).unwrap()
            };
            nodes
                .iter()
                .zip(weights)
                .map(|(&node, weight)| {
                    let x = std::f64::consts::FRAC_PI_4 * (1.0 + node);
                    let u = x.tan();
                    let a = response(0, u);
                    let b = response(1, u);
                    -2.0 * weight
                        * (1.0 + u * u)
                        * a.matmul(&coupling).frobenius_dot(&coupling.matmul(&b))
                })
                .sum::<f64>()
        };
        let config = SaptConfig {
            asymptotic_shift: shifts,
            rpa_dispersion: true,
            casimir_polder_points: 16,
            ..SaptConfig::default()
        };
        let measured = sapt_interaction_regularized_variant(
            &monomers[0],
            &monomers[1],
            &set,
            &functional,
            &o,
            Some(&g),
            &config,
            SaptVariant::KsFdds,
            eta,
        )
        .unwrap();
        let dispersion = integral(128);
        let quadrature = (integral(64) - dispersion).abs();
        let expected = [
            elst,
            first - elst,
            expected_induction[0],
            expected_induction[1],
            expected_exchange_induction.iter().sum::<f64>(),
            dispersion,
            exchange_dispersion,
            first
                + expected_induction.iter().sum::<f64>()
                + expected_exchange_induction.iter().sum::<f64>()
                + dispersion
                + exchange_dispersion,
        ];
        let values = terms(&measured);
        let error = values
            .into_iter()
            .zip(expected)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        let calibration = (measured.e_disp_unc - uncoupled).abs();
        let spectral = (measured.rpa.as_ref().unwrap().e_disp - dispersion).abs();
        println!("{name} KS-FDDS energy terms {values:?}, dense difference {error:.3e}, quadrature change {quadrature:.3e}, uncoupled calibration {calibration:.3e}, spectral difference {spectral:.3e}");
        assert!(error < 1e-10 && quadrature < 1e-11 && calibration < 1e-11 && spectral < 1e-10);
        let regularized = measured.regularized_induction.unwrap();
        for (i, [induction, exchange]) in expected_regularized.into_iter().enumerate() {
            let transfer =
                expected_induction[i] + expected_exchange_induction[i] - induction - exchange;
            let error = (regularized.induction[i] - induction)
                .abs()
                .max((regularized.exchange_induction[i] - exchange).abs())
                .max((regularized.charge_transfer[i] - transfer).abs());
            println!("{name} KS-FDDS regularized direction {i}: induction {:+.15e} Eh, exchange-induction {:+.15e} Eh, charge transfer {:+.15e} Eh; independent maximum {error:.3e}", regularized.induction[i], regularized.exchange_induction[i], regularized.charge_transfer[i]);
            assert!(error < 1e-10);
        }
    }
}

fn water(points: [[f64; 3]; 3]) -> Molecule {
    Molecule::new(
        points
            .into_iter()
            .zip([8, 1, 1])
            .map(|(p, z)| Atom {
                z,
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    )
}

#[test]
fn regularized_water_induction_matches_independent_gaussian_quadrature() {
    let monomers = [
        water([[0.0, 0.0, 0.0], [0.04, 0.10, 1.81], [1.72, 0.10, -0.45]]),
        water([[0.12, -0.24, 5.35], [1.45, 0.66, 6.20], [-1.00, 1.04, 6.01]]),
    ];
    let set = data::load("6-31g").unwrap();
    let f = ResolvedFunctional::hartree_fock();
    let o = reference();
    let basis = monomers
        .each_ref()
        .map(|m| MolBasis::build(m, &set).unwrap());
    let scf: [_; 2] =
        std::array::from_fn(|i| run_scf(&monomers[i], &basis[i], &o, &f, None).unwrap());
    let dimer = Molecule::new(monomers.iter().flat_map(|m| m.atoms.clone()).collect());
    let dimer_basis = MolBasis::build(&dimer, &set).unwrap();
    let eri = eri_incore(&dimer_basis, CoulombKernel::Coulomb, 256).unwrap();
    let offsets = [0, basis[0].nao_sph];
    let eta = 3.0;
    let evaluate = |m: [&Molecule; 2]| {
        sapt_interaction_regularized(m[0], m[1], &set, &f, &o, None, &SaptConfig::default(), eta)
            .unwrap()
            .regularized_induction
            .unwrap()
    };
    let production = evaluate(monomers.each_ref());
    let swapped = evaluate([&monomers[1], &monomers[0]]);
    for (i, b) in basis.iter().enumerate() {
        let j = 1 - i;
        let mut potential = nuclear_matrix(b, &monomers[j], None);
        let density = scf[j].p_total();
        for a in 0..b.nao_sph {
            for c in 0..b.nao_sph {
                for d in 0..density.rows {
                    for e in 0..density.cols {
                        potential[(a, c)] += density[(d, e)]
                            * eri.get(
                                a + offsets[i],
                                c + offsets[i],
                                d + offsets[j],
                                e + offsets[j],
                            );
                    }
                }
            }
        }
        let analytic = production.induction[i];
        let numerical = |points| {
            let short_range = penetration_quadrature(b, &monomers[j], eta, points);
            let source = Matrix::from_row_major(
                potential.rows,
                potential.cols,
                potential
                    .as_slice()
                    .iter()
                    .zip(short_range.as_slice())
                    .map(|(x, y)| x - y)
                    .collect(),
            );
            let solver = ResponseSolver::new(&monomers[i], b, &f, None, &scf[i], &o).unwrap();
            0.5 * solver.solve(&[source]).unwrap()[0]
                .iter()
                .map(|d| d.frobenius_dot(&potential))
                .sum::<f64>()
        };
        let convergence_error = (numerical(96) - numerical(192)).abs();
        let comparison_error = (analytic - numerical(192)).abs();
        let [exchange, transfer] = [
            production.exchange_induction[i],
            production.charge_transfer[i],
        ];
        let swap_error = (analytic - swapped.induction[j])
            .abs()
            .max((exchange - swapped.exchange_induction[j]).abs())
            .max((transfer - swapped.charge_transfer[j]).abs());
        println!("Regularized water direction {i}: induction {analytic:+.15e} Eh, exchange-induction {exchange:+.15e} Eh, charge transfer {transfer:+.15e} Eh, independent quadrature {comparison_error:.3e}, quadrature refinement {convergence_error:.3e}, fragment swap {swap_error:.3e}");
        assert!(convergence_error < 1e-11 && comparison_error < 1e-11 && swap_error < 1e-11);
    }
}

fn reference() -> ScfOptions {
    ScfOptions {
        e_tol: 1e-13,
        p_tol: 1e-11,
        max_iter: 180,
        eri_mode: EriMode::InCore { limit_mb: 256 },
        ..ScfOptions::default()
    }
}

fn atom(z: u8, x: f64) -> Molecule {
    Molecule::new(vec![Atom {
        z,
        position: Vec3::new(x, 0.0, 0.0),
    }])
}

fn terms(r: &SaptResult) -> [f64; 8] {
    [
        r.e_elst,
        r.e_exch,
        r.e_ind_a,
        r.e_ind_b,
        r.e_exch_ind,
        r.e_disp_coupled,
        r.e_exch_disp,
        r.interaction_total(),
    ]
}

fn restricted_third_polarization(
    ints: &OrbitalIntegrals,
    energies: [&[f64]; 2],
    offsets: [usize; 2],
) -> [f64; 3] {
    // Independent spatial-orbital equations for the three bare cubic
    // terms. The monomers each have a single doubly occupied orbital.
    let virtuals = energies.map(|e| 1..e.len());
    let field = |x: usize, p, q| {
        let [a, b] = [offsets[x], offsets[1 - x]];
        ints.external[1 - x][(a + p, a + q)] + 2.0 * ints.eri(a + p, a + q, b, b)
    };
    let centered = |x, p, q| field(x, p, q) - if p == q { field(x, 0, 0) } else { 0.0 };
    let integral = |p, r, q, s| ints.eri(p, r, offsets[1] + q, offsets[1] + s);
    let gap = |x: usize, p: usize| energies[x][p] - energies[x][0];
    let single = |x, p| -field(x, p, 0) / gap(x, p);
    let double = |p, q| -integral(p, 0, q, 0) / (gap(0, p) + gap(1, q));
    let mut induction = 0.0;
    let mut dispersion = 0.0;
    let mut mixed = 0.0;
    for x in 0..2 {
        for p in virtuals[x].clone() {
            for r in virtuals[x].clone() {
                let w = centered(x, p, r);
                induction += 2.0 * single(x, p) * w * single(x, r);
                for q in virtuals[1 - x].clone() {
                    let [t_p, t_r, c] = if x == 0 {
                        [double(p, q), double(r, q), integral(p, r, 0, q)]
                    } else {
                        [double(q, p), double(q, r), integral(0, q, p, r)]
                    };
                    let diagonal = if x == 0 {
                        integral(0, 0, 0, q)
                    } else {
                        integral(0, q, 0, 0)
                    };
                    mixed += 4.0 * t_p * w * t_r
                        + 8.0 * single(x, p) * (c - if p == r { diagonal } else { 0.0 }) * t_r;
                }
            }
        }
    }
    for a in virtuals[0].clone() {
        for b in virtuals[1].clone() {
            induction += 16.0 * single(0, a) * integral(a, 0, b, 0) * single(1, b);
            for c in virtuals[0].clone() {
                for d in virtuals[1].clone() {
                    let mut v = integral(a, c, b, d);
                    if a == c {
                        v -= integral(0, 0, b, d);
                    }
                    if b == d {
                        v -= integral(a, c, 0, 0);
                    }
                    if a == c && b == d {
                        v += integral(0, 0, 0, 0);
                    }
                    dispersion += 4.0 * double(a, b) * v * double(c, d);
                }
            }
        }
    }
    [induction, dispersion, mixed]
}

fn third_order_oracle(
    integrals: &OrbitalIntegrals,
    gaps: [&[f64]; 2],
    offsets: [usize; 2],
    result: &SaptResult,
    swapped: &SaptResult,
) -> [f64; 8] {
    // Model Hamiltonian in the complete four-spin-slot product basis,
    // assembled by direct permutations independently of the Slater-Condon
    // and cofactor kernels. In this dimer each fixed-spin fragment has one
    // electron and all states are retained by the second-order path.
    let state_list = (0..gaps[0].len())
        .flat_map(|i| {
            (0..gaps[1].len()).flat_map(move |j| {
                (0..gaps[0].len()).flat_map(move |k| (0..gaps[1].len()).map(move |l| [i, j, k, l]))
            })
        })
        .collect::<Vec<_>>();
    let orbitals = state_list
        .iter()
        .map(|i| std::array::from_fn(|slot| offsets[slot % 2] + i[slot]))
        .collect::<Vec<_>>();
    let energy_gaps = state_list
        .iter()
        .map(|i| {
            i.iter()
                .enumerate()
                .map(|(slot, &orb)| gaps[slot % 2][orb] - gaps[slot % 2][0])
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let n = state_list.len();
    let identity = PERMUTATIONS[0].0;
    let mut v = Matrix::zeros(n, n);
    for (i, &bra) in orbitals.iter().enumerate() {
        for (j, &ket) in orbitals.iter().enumerate().take(i + 1) {
            let el = integrals.permutation(bra, ket, identity, false).0;
            v[(i, j)] = el;
            v[(j, i)] = el;
        }
    }
    let antisymmetrized: Vec<_> = orbitals
        .iter()
        .map(|&ket| integrals.matrix_element(orbitals[0], ket, false))
        .collect();
    let (reference, norm) = antisymmetrized[0];
    let first = reference / norm;
    let diagonal = v[(0, 0)];
    let mut c1 = vec![0.0; n];
    let mut bare_second = [0.0; 3];
    let mut sym_second = [0.0; 3];
    let mut first_overlap = 0.0;
    let class = |i: usize| match (
        state_list[i][0] + state_list[i][2] != 0,
        state_list[i][1] + state_list[i][3] != 0,
    ) {
        (true, false) => 0,
        (false, true) => 1,
        _ => 2,
    };
    for i in 1..n {
        c1[i] = -v[(i, 0)] / energy_gaps[i];
        let direction = class(i);
        let (m, s) = antisymmetrized[i];
        bare_second[direction] += c1[i] * v[(0, i)];
        sym_second[direction] += c1[i] * (m - first * s) / norm;
        first_overlap += c1[i] * s / norm;
    }
    let v_subtracted = |i, j| v[(i, j)] - if i == j { diagonal } else { 0.0 };
    let mut sym_third = -sym_second.iter().sum::<f64>() * first_overlap;
    for i in 1..n {
        let amplitude = -(1..n).map(|j| v_subtracted(i, j) * c1[j]).sum::<f64>() / energy_gaps[i];
        let (m, s) = antisymmetrized[i];
        sym_third += amplitude * (m - first * s) / norm;
    }
    let srs_vertex: Vec<_> = antisymmetrized
        .iter()
        .map(|&(m, overlap)| (m - first * overlap) / norm)
        .collect();
    let final_ranks = |i: usize| {
        [0, 1].map(|m| usize::from(state_list[i][m] != 0) + usize::from(state_list[i][m + 2] != 0))
    };
    let mut fields = Matrix::zeros(n, n);
    for (i, &x) in orbitals.iter().enumerate() {
        for (j, &y) in orbitals.iter().enumerate() {
            // Sum the two permanent monomer potentials with separate
            // spin-slot spectator overlaps, then remove their reference
            // expectation. No transition lists or cofactors are involved.
            let diagonal = i == j;
            for s in 0..4 {
                let m = s % 2;
                let other = offsets[1 - m];
                let omega = |a, b| {
                    integrals.external[1 - m][(a, b)] + 2.0 * integrals.eri(a, b, other, other)
                };
                let allowed = (0..4).filter(|&k| k != s).all(|k| x[k] == y[k]);
                if allowed {
                    fields[(i, j)] += omega(x[s], y[s]);
                }
                if diagonal {
                    fields[(i, j)] -= omega(offsets[m], offsets[m]);
                }
            }
        }
    }
    let d = n - 1;
    let mut middle: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(d, d));
    for i in 1..n {
        let [a, b] = final_ranks(i);
        let local = a == 0 || b == 0;
        let single = a + b == 1;
        for j in 1..n {
            let from_dispersion = class(j) == 2;
            let h = fields[(i, j)];
            let w = v_subtracted(i, j) - h;
            let [induction, dispersion, mixed] = &mut middle;
            let entry = (i - 1, j - 1);
            // Products of two induction amplitudes, one of each type, and
            // two dispersion amplitudes form separate excitation sectors.
            if from_dispersion {
                if single {
                    induction[entry] = h;
                    mixed[entry] = w;
                } else {
                    dispersion[entry] = w;
                    mixed[entry] = h;
                }
            } else if local {
                induction[entry] = h + w;
            } else {
                induction[entry] = h;
                mixed[entry] = w;
            }
        }
    }
    let o_i: f64 = (1..n)
        .filter(|&i| class(i) != 2)
        .map(|i| c1[i] * antisymmetrized[i].1 / norm)
        .sum();
    let o_d = first_overlap - o_i;
    let s_i = sym_second[0] + sym_second[1];
    let s_d = sym_second[2];
    let norm_correction = [s_i * o_i, s_d * o_d, s_i * o_d + s_d * o_i];
    let dense = middle.each_ref().map(|m| {
        let mut t = [0.0; 2];
        for i in 1..n {
            let ket = -(1..n).map(|j| m[(i - 1, j - 1)] * c1[j]).sum::<f64>() / energy_gaps[i];
            let bare = v[(0, i)];
            t[0] += bare * ket;
            t[1] += (srs_vertex[i] - bare) * ket;
        }
        t
    });
    let numerical = |group: usize, h: f64| {
        let mut shifted = middle[group].clone();
        for value in shifted.as_mut_slice() {
            *value *= h;
        }
        for i in 0..d {
            shifted[(i, i)] += energy_gaps[i + 1];
        }
        let source: Vec<_> = (1..n).map(|i| energy_gaps[i] * c1[i]).collect();
        let x = solve_linear(&shifted, &source).unwrap();
        let mut e = [0.0; 2];
        for (i, &ket) in x.iter().enumerate() {
            let bare = v[(0, i + 1)];
            e[0] += bare * ket;
            e[1] += (srs_vertex[i + 1] - bare) * ket;
        }
        e
    };
    let t = result.third_order.unwrap();
    let s = swapped.third_order.unwrap();
    let flatten = |x: SaptThirdOrder| {
        [
            x.induction,
            x.dispersion,
            x.induction_dispersion,
            x.exchange_induction,
            x.exchange_dispersion,
            x.exchange_induction_dispersion,
            x.exchange,
            x.total(),
        ]
    };
    let values = flatten(t);
    let swapped = flatten(s);
    let orbital = restricted_third_polarization(integrals, gaps, offsets);
    for (i, (a, b)) in orbital.iter().zip(values).enumerate() {
        let error = (a - b).abs();
        println!("SAPT(30) explicit spatial orbital term {i} discrepancy {error:.3e} Eh");
        assert!(error < 1e-19);
    }
    let [ind, disp, mixed] = dense;
    let reference = [
        ind[0],
        disp[0],
        mixed[0],
        ind[1] - norm_correction[0],
        disp[1] - norm_correction[1],
        mixed[1] - norm_correction[2],
        sym_third - ind[0] - disp[0] - mixed[0],
        sym_third,
    ];
    let labels = [
        "induction",
        "dispersion",
        "mixed",
        "exchange induction",
        "exchange dispersion",
        "exchange mixed",
    ]
    .map(|s| format!("SAPT(30) {s} coefficient"));
    let mut failures = Vec::new();
    components::fit_components(
        &labels,
        &[[16.0, 12.0, 8.0, 6.0], [32.0, 24.0, 16.0, 12.0]],
        |steps| {
            let samples = steps.map(|h| [0, 1, 2].map(|g| (numerical(g, h), numerical(g, -h))));
            (0..6)
                .map(|i| {
                    steps.map(|h| {
                        let k = steps.iter().position(|x| *x == h).unwrap();
                        let group = i % 3;
                        let kind = usize::from(i >= 3);
                        let (positive, negative) = samples[k][group];
                        let mut value = stencils::Stencil::Second.first(
                            positive[kind],
                            negative[kind],
                            0.0,
                            0.0,
                            h,
                        );
                        if kind == 1 {
                            value -= norm_correction[group];
                        }
                        (value - values[i]).abs()
                    })
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("third-order SAPT sector coefficients", &failures);
    let names = [
        "induction",
        "dispersion",
        "induction-dispersion",
        "exchange-induction",
        "exchange-dispersion",
        "exchange-induction-dispersion",
        "total exchange",
        "total",
    ];
    for i in 0..names.len() {
        let defect = (values[i] - reference[i]).abs();
        let swap_defect = (values[i] - swapped[i]).abs();
        println!("SAPT(30) {}: {:+.15e} Eh, dense permutation difference {defect:.3e}, swap difference {swap_defect:.3e}", names[i], values[i]);
        assert!(defect < 1e-11 && swap_defect < 1e-11);
    }
    let measured_second = [result.e_ind_a, result.e_ind_b, result.e_disp_coupled];
    let measured_exchange = [
        result.e_exch_ind_directions[0],
        result.e_exch_ind_directions[1],
        result.e_exch_disp,
    ];
    for i in 0..3 {
        assert!((measured_second[i] - bare_second[i]).abs() < 1e-11);
        assert!((measured_exchange[i] + bare_second[i] - sym_second[i]).abs() < 1e-11);
    }
    assert!((result.e_elst + result.e_exch - first).abs() < 1e-11);
    let expected_total = first + sym_second.iter().sum::<f64>() + sym_third;
    assert!((result.interaction_total() - expected_total).abs() < 1e-11);
    println!(
        "SAPT(30) interaction energy {:+.15e} Eh, dense difference {:.3e}",
        result.interaction_total(),
        (result.interaction_total() - expected_total).abs()
    );

    // Independent extraction of the cubic coefficient from diagonalization
    // at a signed coupling. No perturbative amplitudes enter these points.
    let exact = |h: f64| {
        let mut hamiltonian = v.clone();
        for (i, x) in hamiltonian.as_mut_slice().iter_mut().enumerate() {
            *x *= h;
            if i / n == i % n {
                *x += energy_gaps[i / n];
            }
        }
        let (_, orbitals) = symmetric_eigen(&hamiltonian).unwrap();
        let (numerator, overlap) = antisymmetrized
            .iter()
            .enumerate()
            .fold((0.0, 0.0), |(n, s), (i, &(m, o))| {
                (n + orbitals[(i, 0)] * m, s + orbitals[(i, 0)] * o)
            });
        h * numerator / overlap
    };
    let name = ["SAPT(30) total cubic coefficient".to_string()];
    let mut failures = Vec::new();
    components::fit_components(
        &name,
        &[[1.2, 0.9, 0.6, 0.45], [2.4, 1.8, 1.2, 0.9]],
        |steps| {
            vec![steps.map(|h| {
                let estimated = ((exact(h) - exact(-h)) / (2.0 * h) - first) / (h * h);
                (estimated - t.total()).abs()
            })]
        },
        &mut failures,
    );
    components::assert_no_failures("Cubic SAPT exchange and polarization", &failures);
    reference
}

fn fixed_orbital_spectral_check(
    integrals: &OrbitalIntegrals,
    mo_energies: [&[f64]; 2],
    offsets: [usize; 2],
    computed: SaptCorrelationTerms,
    swapped: SaptCorrelationTerms,
) -> [f64; 20] {
    // One alpha and one beta electron per fragment. Its exact Hamiltonian
    // is a direct unlike-spin Coulomb integral plus two one-body terms.
    let sizes = mo_energies.map(|e| e.len());
    let local = sizes.map(|n| (0..n * n).map(|k| [k / n, k % n]).collect::<Vec<_>>());
    let gap = [0, 1].map(|m| {
        local[m]
            .iter()
            .map(|[p, q]| mo_energies[m][*p] + mo_energies[m][*q] - 2.0 * mo_energies[m][0])
            .collect::<Vec<_>>()
    });
    let full_monomer = [0, 1].map(|m| {
        let n = local[m].len();
        let mut h = Matrix::zeros(n, n);
        let other = 1 - m;
        let offset = offsets[m];
        let core = |p, q| integrals.full_core[(p, q)] - integrals.external[other][(p, q)];
        for (i, &[p, r]) in local[m].iter().enumerate() {
            for (j, &[q, s]) in local[m].iter().enumerate() {
                let [p, q, r, s] = [p, q, r, s].map(|a| a + offset);
                h[(i, j)] = integrals.eri(p, q, r, s);
                if p == q {
                    h[(i, j)] += core(r, s);
                }
                if r == s {
                    h[(i, j)] += core(p, q);
                }
            }
        }
        let shift = h[(0, 0)];
        for i in 0..n {
            h[(i, i)] -= shift;
        }
        h
    });

    let product = local[0]
        .iter()
        .flat_map(|&[a, b]| {
            local[1]
                .iter()
                .map(move |&[c, d]| [a, c + offsets[1], b, d + offsets[1]])
        })
        .collect::<Vec<_>>();
    let n = product.len();
    let mut v = Matrix::zeros(n, n);
    let mut va = Matrix::zeros(n, n);
    let mut antisymmetrizer = Matrix::zeros(n, n);
    for (i, &bra) in product.iter().enumerate() {
        for (j, &ket) in product.iter().enumerate() {
            v[(i, j)] = integrals.permutation(bra, ket, PERMUTATIONS[0].0, false).0;
            (va[(i, j)], antisymmetrizer[(i, j)]) = integrals.matrix_element(bra, ket, false);
        }
    }

    let field_element = |m: usize, p: usize, q: usize, partner: &Matrix| {
        let [i, j] = [offsets[m], offsets[1 - m]];
        let mut field = integrals.external[1 - m][(i + p, i + q)];
        for k in 0..partner.rows {
            for l in 0..partner.cols {
                field += partner[(k, l)] * integrals.eri(i + p, i + q, j + k, j + l);
            }
        }
        field
    };
    let static_field = [0, 1].map(|m| {
        let mut field = Matrix::zeros(sizes[m], sizes[m]);
        let mut partner = Matrix::zeros(sizes[1 - m], sizes[1 - m]);
        partner[(0, 0)] = 2.0;
        for p in 0..field.rows {
            for q in 0..field.cols {
                field[(p, q)] = field_element(m, p, q, &partner);
            }
        }
        field
    });
    let partner_density = |m: usize, coefficient: &Matrix| {
        let n = sizes[m];
        let amplitudes =
            Matrix::from_row_major(n, n, (0..n * n).map(|i| coefficient[(i, 0)]).collect());
        let mut density = amplitudes.matmul_transpose_seq(&amplitudes);
        let beta = amplitudes.transpose_matmul(&amplitudes);
        for (x, y) in density.as_mut_slice().iter_mut().zip(beta.as_slice()) {
            *x += y;
        }
        density
    };
    let mp2_doubles_only = |m: usize, field_strength: f64| {
        let n = sizes[m];
        let start = offsets[m];
        let mut f = static_field[m].clone();
        for x in f.as_mut_slice() {
            *x *= field_strength;
        }
        for p in 0..n {
            f[(p, p)] += mo_energies[m][p];
        }
        // Eigenvectors in the original MO basis with a fixed external
        // field. No Hartree-Fock self-consistent field is included here.
        let (eps, c) = symmetric_eigen(&f).unwrap();
        let mut energy = 0.0;
        for a in 1..n {
            for b in 1..n {
                let mut amplitude = 0.0;
                for p in 0..n {
                    for q in 0..n {
                        for r in 0..n {
                            for s in 0..n {
                                amplitude += c[(p, a)]
                                    * c[(q, 0)]
                                    * c[(r, b)]
                                    * c[(s, 0)]
                                    * integrals.eri(start + p, start + q, start + r, start + s);
                            }
                        }
                    }
                }
                energy -= amplitude * amplitude / (eps[a] + eps[b] - 2.0 * eps[0]);
            }
        }
        energy
    };

    // At each signed coupling, recompute the full monomer spectra and
    // sum over exact product excitations instead of expanding a resolvent.
    let cache = std::cell::RefCell::new(std::collections::BTreeMap::new());
    let values = |coupling: f64| {
        if let Some(&stored) = cache.borrow().get(&coupling.to_bits()) {
            return stored;
        }
        let [(ea, a), (eb, b)] = [0, 1].map(|m| {
            let mut h = full_monomer[m].clone();
            for element in h.as_mut_slice() {
                *element *= coupling;
            }
            for i in 0..h.rows {
                h[(i, i)] += (1.0 - coupling) * gap[m][i];
            }
            symmetric_eigen(&h).unwrap()
        });
        let coefficient = |i, j, p| a[(p / b.rows, i)] * b[(p % b.rows, j)];
        let g = Matrix::from_row_major(n, 1, (0..n).map(|p| coefficient(0, 0, p)).collect());
        let v_g = v.matmul(&g);
        let ga_v = va.transpose_matmul(&g);
        let ga = antisymmetrizer.transpose_matmul(&g);
        let bare = g.frobenius_dot(&v_g);
        let ov = g.frobenius_dot(&ga);
        let sym = g.frobenius_dot(&ga_v) / ov;
        let transition = |m: &Matrix, i, j| {
            m.as_slice()
                .iter()
                .enumerate()
                .map(|(p, x)| x * coefficient(i, j, p))
                .sum::<f64>()
        };
        let mut out = [0.0; 10];
        out[0] = bare;
        out[1] = sym - bare;
        for (i, e) in ea.iter().enumerate() {
            for (j, f) in eb.iter().enumerate() {
                if i == 0 && j == 0 {
                    continue;
                }
                let sector = match (i > 0, j > 0) {
                    (true, false) => 0,
                    (false, true) => 1,
                    _ => 2,
                };
                let vertex = transition(&v_g, i, j);
                let response = -vertex / (e + f - ea[0] - eb[0]);
                let es = (transition(&ga_v, i, j) - sym * transition(&ga, i, j)) / ov;
                out[2 + sector] += vertex * response;
                out[5 + sector] += (es - vertex) * response;
            }
        }
        let densities = [partner_density(0, &a), partner_density(1, &b)];
        for m in 0..2 {
            let cross_energy: f64 = (1..sizes[m])
                .map(|p| {
                    let delta = mo_energies[m][p] - mo_energies[m][0];
                    -2.0 * field_element(m, p, 0, &densities[1 - m]).powi(2) / delta
                })
                .sum();
            out[8 + m] = mp2_doubles_only(m, coupling) + cross_energy;
        }
        cache.borrow_mut().insert(coupling.to_bits(), out);
        out
    };
    let points = [0, 1]
        .into_iter()
        .flat_map(|channel| (1..=3).map(move |order| (channel, order)))
        .chain((2..8).flat_map(|channel| (1..=2).map(move |order| (channel, order))))
        .chain([(8, 2), (9, 2)])
        .collect::<Vec<_>>();
    let flatten = |c: SaptCorrelationTerms, swap: bool| {
        let permutation = if swap { [1, 0, 2] } else { [0, 1, 2] };
        c.electrostatics
            .into_iter()
            .chain(c.exchange)
            .chain(
                permutation
                    .into_iter()
                    .flat_map(|k| c.second_order_polarization[k]),
            )
            .chain(
                permutation
                    .into_iter()
                    .flat_map(|k| c.second_order_exchange[k]),
            )
            .chain(permutation[..2].iter().map(|&k| c.truncated_induction22[k]))
            .collect::<Vec<_>>()
    };
    let result = flatten(computed, false);
    let reverse = flatten(swapped, true);
    let swap_error = result
        .iter()
        .zip(&reverse)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(swap_error < 1e-12);
    let zero = values(0.0);
    let estimate = |h: f64, channel: usize, n: usize| {
        let plus = values(h)[channel];
        let minus = values(-h)[channel];
        let central = || stencils::Stencil::Second.first(plus, minus, 0.0, 0.0, h);
        match n {
            1 => central(),
            2 => (plus + minus - 2.0 * zero[channel]) / (2.0 * h * h),
            _ => {
                let smaller = stencils::Stencil::Second.first(
                    values(h / 2.0)[channel],
                    values(-h / 2.0)[channel],
                    0.0,
                    0.0,
                    h / 2.0,
                );
                4.0 * (central() - smaller) / (3.0 * h * h)
            }
        }
    };
    let channels = [
        "electrostatic",
        "exchange",
        "ind-A",
        "ind-B",
        "disp",
        "exch-ind-A",
        "exch-ind-B",
        "exch-disp",
        "truncated ind-A",
        "truncated ind-B",
    ];
    let names = points
        .iter()
        .map(|&(channel, order)| format!("fixed orbital {} W^{order}", channels[channel]))
        .collect::<Vec<_>>();
    let stages = [
        ([0.8, 0.65, 0.5, 0.35], stencils::Stencil::Second),
        ([0.5, 0.4, 0.3, 0.2], stencils::Stencil::Second),
        ([1.1, 0.9, 0.7, 0.55], stencils::Stencil::Fourth),
        ([0.7, 0.55, 0.45, 0.35], stencils::Stencil::Fourth),
    ];
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &stages,
        |steps, stencil| {
            points
                .iter()
                .zip(&result)
                .map(|(&(channel, n), expected)| {
                    steps.map(|h| {
                        let e = estimate(h, channel, n);
                        let limit = match stencil {
                            stencils::Stencil::Second => e,
                            stencils::Stencil::Fourth => {
                                (4.0 * estimate(h / 2.0, channel, n) - e) / 3.0
                            }
                        };
                        (limit - expected).abs()
                    })
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("fixed-orbital mixed-order SAPT", &failures);

    let step = 0.25;
    let mut max_residual = 0.0_f64;
    let mut extrapolated = [0.0; 20];
    for (index, (((channel, n), name), measured)) in
        points.into_iter().zip(&names).zip(result).enumerate()
    {
        // Sixth-order Richardson limit of independently diagonalized spectra.
        let fourth = |h| (4.0 * estimate(h / 2.0, channel, n) - estimate(h, channel, n)) / 3.0;
        let sixth = |h| (16.0 * fourth(h / 2.0) - fourth(h)) / 15.0;
        let limit = sixth(step);
        extrapolated[index] = limit;
        if n <= 2 {
            // The lower derivatives also admit a wider, tenth-order
            // comparison before roundoff. Keep the original sixth-order
            // residual and avoid applying the extra extrapolation to the
            // cubic coefficients, whose signed differences cancel more.
            let eighth = |h| (64.0 * sixth(h / 2.0) - sixth(h)) / 63.0;
            let fine = (256.0 * eighth(0.4) - eighth(0.8)) / 255.0;
            extrapolated[index] = fine;
            let difference = (fine - measured).abs();
            println!("{name}: independent tenth-order residual {difference:.3e} Eh");
            assert!(difference < 5e-14);
        }
        let residual = (limit - measured).abs();
        max_residual = max_residual.max(residual);
        println!("{name}: {measured:+.15e} Eh, independent spectral residual {residual:.3e} Eh");
    }
    assert!(max_residual < 2e-11);
    println!("C1 H2 dimer 6-31G fixed-orbital terms: max spectral {max_residual:.3e} Eh, max swap {swap_error:.3e} Eh");
    extrapolated
}

fn relaxed_electrostatic_check(
    fragment: usize,
    molecule: &Molecule,
    basis: &MolBasis,
    field: &Matrix,
    opts: &ScfOptions,
    result: [f64; 2],
) -> [f64; 2] {
    let eri = eri_incore(basis, CoulombKernel::Coulomb, 256).unwrap();
    let mut h0 = kinetic_matrix(basis);
    let v = nuclear_matrix(basis, molecule, None);
    for (h, v) in h0.as_mut_slice().iter_mut().zip(v.as_slice()) {
        *h += v;
    }
    let at_field = |coupling: f64| {
        let mut extra_hcore = field.clone();
        for h in extra_hcore.as_mut_slice() {
            *h *= coupling;
        }
        let mut h = h0.clone();
        for (x, y) in h.as_mut_slice().iter_mut().zip(extra_hcore.as_slice()) {
            *x += y;
        }
        let hf = run_scf(
            molecule,
            basis,
            &ScfOptions {
                extra_hcore: Some(extra_hcore),
                ..opts.clone()
            },
            &ResolvedFunctional::hartree_fock(),
            None,
        )
        .unwrap();
        let c = &hf.mo_coeff[0];
        let one_body = c.transpose_matmul(&h.matmul(c));
        let eri = transform_eri(&eri, c);
        let n = c.cols;
        let two_body = |p: usize, q: usize, r: usize, s: usize| eri[((p * n + q) * n + r) * n + s];
        let states = (1..n)
            .flat_map(|a| (1..n).map(move |b| (a, b)))
            .collect::<Vec<_>>();
        let e = &hf.mo_energy[0];
        let gap = |p, q| e[p] + e[q] - 2.0 * e[0];
        let amp = |p, q| -two_body(p, 0, q, 0) / gap(p, q);
        let second = states
            .iter()
            .map(|&(a, b)| amp(a, b) * two_body(a, 0, b, 0))
            .sum();
        let mut third = 0.0;
        let reference = 2.0 * one_body[(0, 0)] + two_body(0, 0, 0, 0);
        for &(a, b) in &states {
            for &(c, d) in &states {
                let mut w = two_body(a, c, b, d);
                if a == c {
                    w += one_body[(b, d)];
                }
                if b == d {
                    w += one_body[(a, c)];
                }
                if a == c && b == d {
                    w -= reference + gap(a, b);
                }
                third += amp(a, b) * w * amp(c, d);
            }
        }
        [second, third]
    };
    let names = [2, 3].map(|n| format!("fragment {fragment} orbital-relaxed electrostatic W^{n}"));
    let central = |h| {
        let positive = at_field(h);
        let negative = at_field(-h);
        std::array::from_fn::<_, 2, _>(|i| {
            stencils::Stencil::Second.first(positive[i], negative[i], 0.0, 0.0, h)
        })
    };
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &[[8.0, 6.0, 4.0, 3.0], [4.0, 3.0, 2.0, 1.5]],
        |steps| {
            let estimated = steps.map(central);
            (0..2)
                .map(|i| estimated.map(|e| (e[i] - result[i]).abs()))
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("orbital-relaxed MP electrostatics", &failures);
    let a = central(3.0);
    let b = central(1.5);
    let c = central(0.75);
    let limit = std::array::from_fn::<_, 2, _>(|i| (64.0 * c[i] - 20.0 * b[i] + a[i]) / 45.0);
    for i in 0..2 {
        let residual = (limit[i] - result[i]).abs();
        println!(
            "{}: {:+.15e} Eh, reconverged RHF MP residual {residual:.3e} Eh",
            names[i], result[i]
        );
        assert!(residual < 2e-12);
    }
    limit
}

#[test]
fn helium_pairs_obey_induction_and_dispersion_long_range_limits() {
    let set = data::load("def2-svp").unwrap();
    let functional = ResolvedFunctional::hartree_fock();
    let opts = reference();
    let helium = atom(2, 0.0);
    let basis = MolBasis::build(&helium, &set).unwrap();
    let scf = run_scf(&helium, &basis, &opts, &functional, None).unwrap();
    let dipole = dipole_matrices(&basis, Vec3::new(0.0, 0.0, 0.0));
    let c = &scf.mo_coeff[0];
    let dipoles = dipole.map(|d| c.transpose_matmul(&d.matmul(c)));

    // Exact integration of the uncoupled atomic dipole Casimir-Polder
    // expression, independent of the dimer's electron repulsion integrals.
    let oscillator_weights: Vec<f64> = (1..c.cols)
        .map(|v| (4.0 / 3.0) * dipoles.iter().map(|d| d[(0, v)].powi(2)).sum::<f64>())
        .collect();
    let gaps = &scf.mo_energy[0];
    let mut c6 = 0.0;
    for (i, &di) in oscillator_weights.iter().enumerate() {
        for (j, &dj) in oscillator_weights.iter().enumerate() {
            c6 += 1.5 * di * dj / (gaps[i + 1] + gaps[j + 1] - 2.0 * gaps[0]);
        }
    }
    let solver = ResponseSolver::new(&helium, &basis, &functional, None, &scf, &opts).unwrap();
    let dip_x = dipole_matrices(&basis, Vec3::new(0.0, 0.0, 0.0))[0].clone();
    let result = solver.solve(std::slice::from_ref(&dip_x)).unwrap();
    let charge_induction = 0.5
        * result[0]
            .iter()
            .map(|p| p.frobenius_dot(&dip_x))
            .sum::<f64>();
    let config = SaptConfig::default();
    println!("Independent helium references: C6={c6:.15e} Eh Bohr^6, -alpha_x/2={charge_induction:.15e} Eh Bohr^4");
    let evaluate = |b: &Molecule, variant| {
        sapt_interaction_variant(&helium, b, &set, &functional, &opts, None, &config, variant)
            .unwrap()
    };
    for r in [8.0_f64, 12.0, 18.0, 24.0] {
        let neutral = atom(2, r);
        let cation = atom(3, r).with_charge(1);
        let dispersive = evaluate(&neutral, SaptVariant::Sapt0);
        let ionic = evaluate(&cation, SaptVariant::Sapt0);
        let ionic_hl = evaluate(&cation, SaptVariant::Response);
        let scaled_disp = -dispersive.e_disp_coupled * r.powi(6);
        let scaled_ind = ionic.e_ind_a * r.powi(4);
        let ionic_exchange = ionic.e_exch_ind * r.powi(4);
        let hl_exchange = ionic_hl.e_exch_ind * r.powi(4);
        println!("R={r:5.1} Bohr: -R^6 Edisp={scaled_disp:.15e}, R^4 Eind={scaled_ind:.15e}, R^4 Eexch-ind [SAPT(0), HL]=[{ionic_exchange:.15e}, {hl_exchange:.15e}]");
        if r >= 18.0 {
            assert!((scaled_disp - c6).abs() < 1e-9);
            assert!((scaled_ind - charge_induction).abs() < 1e-9);
            assert!(ionic_exchange.abs() < 1e-9);
            assert!(ionic_hl.e_exch_ind.abs() < 1e-13);
            assert!(hl_exchange.abs() < 1e-6 * scaled_ind.abs());
        }
    }
}

#[test]
fn noncoplanar_h2_dimer_matches_explicit_permutations_term_by_term() {
    let monomers = [
        h2([[0.04, 0.12, -0.06], [1.41, -0.31, 0.63]]),
        h2([[1.02, 0.77, 4.62], [1.67, 1.86, 5.15]]),
    ];
    let set = data::load("6-31g").unwrap();
    let functional = ResolvedFunctional::hartree_fock();
    let opts = reference();
    let basis = monomers
        .each_ref()
        .map(|m| MolBasis::build(m, &set).unwrap());
    let scf: [_; 2] = std::array::from_fn(|i| {
        run_scf(&monomers[i], &basis[i], &opts, &functional, None).unwrap()
    });
    let dimer = Molecule::new(monomers.iter().flat_map(|m| m.atoms.clone()).collect());
    let b_dimer = MolBasis::build(&dimer, &set).unwrap();
    let eri = eri_incore(&b_dimer, CoulombKernel::Coulomb, 256).unwrap();
    let n = b_dimer.nao_sph;
    let offsets = [0, basis[0].nao_sph];
    let full_nuclear = dimer.nuclear_repulsion();
    let external_nuclear = full_nuclear
        - monomers
            .iter()
            .map(Molecule::nuclear_repulsion)
            .sum::<f64>();
    let external = [
        nuclear_matrix(&b_dimer, &dimer, Some(&[1.0, 1.0, 0.0, 0.0])),
        nuclear_matrix(&b_dimer, &dimer, Some(&[0.0, 0.0, 1.0, 1.0])),
    ];
    let mut full_core = kinetic_matrix(&b_dimer);
    for v in &external {
        for (x, y) in full_core.as_mut_slice().iter_mut().zip(v.as_slice()) {
            *x += y;
        }
    }
    let overlap = overlap_matrix(&b_dimer);
    let mut c = Matrix::zeros(n, n);
    for i in 0..2 {
        for a in 0..basis[i].nao_sph {
            for b in 0..basis[i].nao_sph {
                c[(offsets[i] + a, offsets[i] + b)] = scf[i].mo_coeff[0][(a, b)];
            }
        }
    }
    let transform = |a: &Matrix| c.transpose_matmul(&a.matmul(&c));
    let integrals = OrbitalIntegrals {
        overlap: transform(&overlap),
        external: external.each_ref().map(transform),
        full_core: transform(&full_core),
        eri: transform_eri(&eri, &c),
        nuclear: [external_nuclear, full_nuclear],
    };
    let fixed_orbital = |a: &Molecule, b: &Molecule| {
        sapt_interaction_diagnostics(
            a,
            b,
            &set,
            &functional,
            &opts,
            None,
            &SaptConfig::default(),
            SaptVariant::Sapt0,
            SaptDiagnostics {
                correlation_terms: true,
                ..SaptDiagnostics::default()
            },
        )
        .unwrap()
        .correlation_terms
        .unwrap()
    };
    let correlated = fixed_orbital(&monomers[0], &monomers[1]);
    let swapped = fixed_orbital(&monomers[1], &monomers[0]);
    for a in 0..2 {
        for k in 0..2 {
            assert!(
                (correlated.relaxed_electrostatics[a][k]
                    - swapped.relaxed_electrostatics[1 - a][k])
                    .abs()
                    < 1e-12
            );
        }
    }
    let oracle_correlation = fixed_orbital_spectral_check(
        &integrals,
        [&scf[0].mo_energy[0], &scf[1].mo_energy[0]],
        offsets,
        correlated,
        swapped,
    );
    let bra = [0, offsets[1], 0, offsets[1]];
    let identity = PERMUTATIONS[0].0;
    let (elst, _) = integrals.permutation(bra, bra, identity, false);
    let (numerator, norm) = integrals.matrix_element(bra, bra, false);
    let first_order = numerator / norm;
    let single_exchange = |ket| {
        -PERMUTATIONS[1..3]
            .iter()
            .map(|&(perm, _)| {
                let (v, s) = integrals.permutation(bra, ket, perm, false);
                v - elst * s
            })
            .sum::<f64>()
    };
    let s2_exchange = single_exchange(bra);
    let mut induction = [0.0; 2];
    let mut exchanged_induction = [0.0; 2];
    let mut s2_induction = [0.0; 2];
    let mut dispersion = 0.0;
    let mut exchanged_dispersion = 0.0;
    let mut s2_dispersion = 0.0;
    let eta = 3.0;
    let mut regularized_oracle = [[0.0; 2]; 2];
    let mut amplitudes: [Vec<f64>; 2] = std::array::from_fn(|x| vec![0.0; basis[x].nao_sph]);
    let mut orbital_relaxation_oracle = [[0.0; 2]; 2];
    for (fragment, relaxed) in orbital_relaxation_oracle.iter_mut().enumerate() {
        let other = 1 - fragment;
        let offset = offsets[fragment];
        let mut potential = Matrix::zeros(basis[fragment].nao_sph, basis[fragment].nao_sph);
        for a in 0..potential.rows {
            for b in 0..potential.cols {
                potential[(a, b)] = external[other][(offset + a, offset + b)];
                for d in 0..basis[other].nao_sph {
                    for e in 0..basis[other].nao_sph {
                        potential[(a, b)] += 2.0
                            * scf[other].mo_coeff[0][(d, 0)]
                            * scf[other].mo_coeff[0][(e, 0)]
                            * eri.get(
                                offset + a,
                                offset + b,
                                offsets[other] + d,
                                offsets[other] + e,
                            );
                    }
                }
            }
        }
        let solver = ResponseSolver::new(
            &monomers[fragment],
            &basis[fragment],
            &functional,
            None,
            &scf[fragment],
            &opts,
        )
        .unwrap();
        *relaxed = relaxed_electrostatic_check(
            fragment,
            &monomers[fragment],
            &basis[fragment],
            &potential,
            &opts,
            correlated.relaxed_electrostatics[fragment],
        );
        let dp = &solver.solve(&[potential.clone()]).unwrap()[0];
        let sm = overlap_matrix(&basis[fragment]);
        let scm = sm.matmul(&scf[fragment].mo_coeff[0]);
        let tangent = scm.transpose_matmul(&dp[0].matmul(&scm));
        let a96 = penetration_quadrature(&basis[fragment], &monomers[other], eta, 96);
        let a192 = penetration_quadrature(&basis[fragment], &monomers[other], eta, 192);
        let quadrature_difference = a96
            .as_slice()
            .iter()
            .zip(a192.as_slice())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        println!("Regularized H2 direction {fragment} AO Laplace quadrature step difference {quadrature_difference:.3e} Eh");
        assert!(quadrature_difference < 1e-11);
        for (x, y) in potential.as_mut_slice().iter_mut().zip(a192.as_slice()) {
            *x -= y;
        }
        let dp_regularized = &solver.solve(&[potential]).unwrap()[0];
        let tangent_regularized = scm.transpose_matmul(&dp_regularized[0].matmul(&scm));
        for slot in [fragment, fragment + 2] {
            for v in 1..basis[fragment].nao_sph {
                let mut ket = bra;
                ket[slot] = offset + v;
                let (numerator, overlap) = integrals.matrix_element(bra, ket, false);
                let (bare, _) = integrals.permutation(bra, ket, identity, false);
                let corrected = (numerator - first_order * overlap) / norm;
                amplitudes[fragment][v] = tangent[(v, 0)];
                exchanged_induction[fragment] += tangent[(v, 0)] * corrected;
                s2_induction[fragment] += tangent[(v, 0)] * single_exchange(ket);
                induction[fragment] += tangent[(v, 0)] * bare;
                regularized_oracle[fragment][0] += tangent_regularized[(v, 0)] * bare;
                regularized_oracle[fragment][1] += tangent_regularized[(v, 0)] * corrected;
            }
        }
    }
    for slot_a in [0, 2] {
        for slot_b in [1, 3] {
            for a in 1..basis[0].nao_sph {
                for b in 1..basis[1].nao_sph {
                    let mut ket = bra;
                    ket[slot_a] = a;
                    ket[slot_b] = offsets[1] + b;
                    let (numerator, overlap) = integrals.matrix_element(bra, ket, false);
                    let (bare, _) = integrals.permutation(bra, ket, identity, false);
                    let corrected = (numerator - first_order * overlap) / norm;
                    // The lower-order subtraction is E_elst + E_exch times
                    // the two-excitation overlap. Both come from permutations.
                    let gaps = scf[0].mo_energy[0][a] - scf[0].mo_energy[0][0]
                        + scf[1].mo_energy[0][b]
                        - scf[1].mo_energy[0][0];
                    dispersion -= bare * bare / gaps;
                    exchanged_dispersion -= bare * corrected / gaps;
                    s2_dispersion -= bare * single_exchange(ket) / gaps;
                }
            }
        }
    }
    let expected = [
        elst,
        first_order - elst,
        induction[0],
        induction[1],
        exchanged_induction.iter().sum::<f64>() - induction.iter().sum::<f64>(),
        dispersion,
        exchanged_dispersion - dispersion,
        first_order + exchanged_induction.iter().sum::<f64>() + exchanged_dispersion,
    ];
    let expected_s2 = [
        elst,
        s2_exchange,
        induction[0],
        induction[1],
        s2_induction.iter().sum::<f64>(),
        dispersion,
        s2_dispersion,
        elst + s2_exchange
            + induction.iter().sum::<f64>()
            + s2_induction.iter().sum::<f64>()
            + dispersion
            + s2_dispersion,
    ];
    let evaluate = |a, b, options, variant| {
        sapt_interaction_variant(
            a,
            b,
            &set,
            &functional,
            options,
            None,
            &SaptConfig::default(),
            variant,
        )
        .unwrap()
    };
    let uhf_opts = ScfOptions {
        reference: dft_rs::Reference::Uhf,
        ..opts.clone()
    };
    let regularize = |a, b, exponent| {
        sapt_interaction_regularized(
            a,
            b,
            &set,
            &functional,
            &opts,
            None,
            &SaptConfig::default(),
            exponent,
        )
        .unwrap()
    };
    let unmodified = evaluate(&monomers[0], &monomers[1], &opts, SaptVariant::Sapt0);
    let exchange_ratio = unmodified.e_exch_ind / (unmodified.e_ind_a + unmodified.e_ind_b);
    let scaled = correlated.truncated_induction22.iter().sum::<f64>() * exchange_ratio;
    assert!((scaled - correlated.truncated_exchange_induction22).abs() < 1e-16);
    println!("C1 H2 dimer 6-31G truncated exchange-induction (22): {scaled:+.15e} Eh");
    let regularized = regularize(&monomers[0], &monomers[1], eta);
    let diagnostic = regularized.regularized_induction.unwrap();
    let swapped = regularize(&monomers[1], &monomers[0], eta)
        .regularized_induction
        .unwrap();
    let narrow = regularize(&monomers[0], &monomers[1], 1e8)
        .regularized_induction
        .unwrap();
    assert_eq!(terms(&regularized), terms(&unmodified));
    for i in 0..2 {
        let bare = regularized_oracle[i][0];
        let combined = regularized_oracle[i][1];
        let transfer = exchanged_induction[i] - combined;
        let calculated = [
            diagnostic.induction[i],
            diagnostic.exchange_induction[i],
            diagnostic.charge_transfer[i],
        ];
        let expected = [bare, combined - bare, transfer];
        let swapped = [
            swapped.induction[1 - i],
            swapped.exchange_induction[1 - i],
            swapped.charge_transfer[1 - i],
        ];
        let err = calculated
            .into_iter()
            .zip(expected)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        let swap = calculated
            .into_iter()
            .zip(swapped)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        println!("Regularized SAPT(0) direction {i}: induction, exchange-induction and charge transfer {calculated:?} Eh, independent maximum {err:.3e}, swap {swap:.3e}");
        assert!(err < 1e-11 && swap < 1e-11);
        assert!(narrow.charge_transfer[i].abs() < 1e-11);
    }
    for (variant, expected, response) in [
        (
            SaptVariant::Sapt0,
            expected,
            [0, 1].map(|i| exchanged_induction[i] - induction[i]),
        ),
        (SaptVariant::Sapt0SingleExchange, expected_s2, s2_induction),
    ] {
        let formal = evaluate(&monomers[0], &monomers[1], &opts, variant);
        let swapped = evaluate(&monomers[1], &monomers[0], &opts, variant);
        let unrestricted = evaluate(&monomers[0], &monomers[1], &uhf_opts, variant);
        let values = terms(&formal);
        let mut swapped_values = terms(&swapped);
        swapped_values.swap(2, 3);
        let unrestricted_values = terms(&unrestricted);
        for (i, measured) in values.into_iter().enumerate() {
            let error = (measured - expected[i]).abs();
            let swap_error = (measured - swapped_values[i]).abs();
            let spin_error = (measured - unrestricted_values[i]).abs();
            println!("{variant:?} term {i}: {measured:+.15e} Eh, permutation error {error:.3e}, swap error {swap_error:.3e}, RHF/UHF error {spin_error:.3e}");
            assert!(error < 1e-11 && swap_error < 1e-11 && spin_error < 1e-11);
        }
        for (direction, (&value, expected)) in formal
            .e_exch_ind_directions
            .iter()
            .zip(response)
            .enumerate()
        {
            let error = (value - expected).abs();
            println!("{variant:?} exchange-induction direction {direction}: {value:+.15e} Eh, permutation error {error:.3e}");
            assert!(error < 1e-11);
        }
        assert!((formal.e_disp_unc - formal.e_disp_coupled).abs() < 1e-14);
    }

    let cp_energy = |keep: Option<usize>| {
        let mut atoms = dimer.atoms.clone();
        if let Some(fragment) = keep {
            for (i, atom) in atoms.iter_mut().enumerate() {
                if i / 2 != fragment {
                    atom.z = 0;
                }
            }
        }
        run_scf(&Molecule::new(atoms), &b_dimer, &opts, &functional, None)
            .unwrap()
            .energy
    };
    let counterpoise = cp_energy(None) - cp_energy(Some(0)) - cp_energy(Some(1));
    let linear = (first_order - elst) / s2_exchange;
    let cubic = linear.powi(3);
    let s2_induction_sum = s2_induction.iter().sum::<f64>();
    let induction_sum = induction.iter().sum::<f64>();
    let correction = counterpoise - first_order - induction_sum - linear * s2_induction_sum;
    let scaled = evaluate(&monomers[0], &monomers[1], &opts, SaptVariant::ScaledSapt0);
    let expected_scaled = [
        elst,
        first_order - elst,
        induction[0],
        induction[1],
        s2_induction_sum * cubic,
        dispersion,
        s2_dispersion * cubic,
        first_order
            + induction_sum
            + (s2_induction_sum + s2_dispersion) * cubic
            + dispersion
            + correction,
    ];
    let scaled_defect = terms(&scaled)
        .iter()
        .zip(expected_scaled)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    assert!(scaled_defect < 1e-11);
    assert!((scaled.exchange_scale.unwrap() - cubic).abs() < 1e-11);
    assert!((scaled.delta_hf.unwrap() - correction).abs() < 1e-11);
    assert!((scaled.e_int_supermolecular - counterpoise).abs() < 1e-11);
    println!(
        "sSAPT0: scale {:.15}, delta-HF={:+.15e} Eh, counterpoise HF={:+.15e} Eh, E={:+.15e} Eh, maximum term error {scaled_defect:.3e} Eh",
        scaled.exchange_scale.unwrap(), scaled.delta_hf.unwrap(), scaled.e_int_supermolecular, scaled.interaction_total(),
    );
    let third = evaluate(
        &monomers[0],
        &monomers[1],
        &opts,
        SaptVariant::Sapt30Uncoupled,
    );
    let third_swapped = evaluate(
        &monomers[1],
        &monomers[0],
        &opts,
        SaptVariant::Sapt30Uncoupled,
    );
    let oracle_third = third_order_oracle(
        &integrals,
        [&scf[0].mo_energy[0], &scf[1].mo_energy[0]],
        offsets,
        &third,
        &third_swapped,
    );

    // Differentiate the HL model by independent evaluations of the
    // four-electron Hamiltonian with each monomer ket and bra polarized.
    // Orbitals occupy four independent spin slots. The comparison subtracts
    // the distinguishable electron product energy at the same amplitudes.
    let higher = evaluate(&monomers[0], &monomers[1], &opts, SaptVariant::Sapt2Plus3);
    let reversed = evaluate(&monomers[1], &monomers[0], &opts, SaptVariant::Sapt2Plus3);
    let terms = higher.sapt2_plus3.unwrap();
    let relaxed = higher.relaxed_induction30.unwrap();
    let independent_ratio = (exchanged_induction.iter().sum::<f64>()
        - induction.iter().sum::<f64>())
        / induction.iter().sum::<f64>();
    let independent = [
        orbital_relaxation_oracle.iter().flatten().sum::<f64>(),
        oracle_correlation[3] + oracle_correlation[4],
        (oracle_correlation[18] + oracle_correlation[19]) * (1.0 + independent_ratio),
        oracle_correlation[10] + oracle_correlation[11],
        oracle_third[1] + oracle_third[2] + oracle_third[4] + oracle_third[5],
    ];
    let observed = [
        terms.correlated_electrostatics,
        terms.correlated_exchange,
        terms.correlated_induction,
        terms.correlated_dispersion,
        terms.cubic_dispersion_and_mixed,
    ];
    for (index, (&measured, reference)) in observed.iter().zip(independent).enumerate() {
        let residual = (measured - reference).abs();
        assert!(residual < 5e-14);
        println!(
            "SAPT2+3 subtotal {index}: {measured:+.15e} Eh, independent residual {residual:.3e} Eh"
        );
    }
    let expected = counterpoise + exchanged_dispersion + independent.iter().sum::<f64>();
    let total = higher.interaction_total();
    let error = (expected - total).abs();
    let exchange_error = (terms.relaxed_exchange_induction30
        - relaxed.induction * oracle_third[3] / oracle_third[0])
        .abs();
    let delta_error =
        (terms.delta_hf_order3 + relaxed.induction + terms.relaxed_exchange_induction30
            - relaxed.delta_hf_order2)
            .abs();
    let swap_error = (total - reversed.interaction_total()).abs();
    assert!(error < 5e-14);
    assert!(exchange_error < 1e-15 && delta_error < 1e-16 && swap_error < 1e-13);
    println!("SAPT2+3 H2 dimer total {total:+.15e} Eh, independent {error:.3e} Eh, swapped {swap_error:.3e} Eh");
    println!("SAPT2+3 relaxed induction {:+.15e} Eh, relaxed exchange {:+.15e} Eh, delta-HF(3) {:+.15e} Eh, ratio discrepancy {exchange_error:.3e} Eh, cancellation {delta_error:.3e} Eh",
        relaxed.induction, terms.relaxed_exchange_induction30, terms.delta_hf_order3);

    let energy = |h: f64, component: usize| {
        let mut rotated = Matrix::identity(n);
        let offset = offsets[component];
        for (i, &co) in amplitudes[component].iter().enumerate().skip(1) {
            rotated[(offset + i, offset)] = h * co;
        }
        let rotate = |matrix: &Matrix| rotated.transpose_matmul(&matrix.matmul(&rotated));
        let mut eri_1 = integrals.eri.clone();
        for p in [0, offsets[1]] {
            for q in [0, offsets[1]] {
                for r in [0, offsets[1]] {
                    for s in [0, offsets[1]] {
                        let mut element = 0.0;
                        for i in 0..n {
                            for j in 0..n {
                                for k in 0..n {
                                    for l in 0..n {
                                        element += rotated[(i, p)]
                                            * rotated[(j, q)]
                                            * rotated[(k, r)]
                                            * rotated[(l, s)]
                                            * integrals.eri(i, j, k, l);
                                    }
                                }
                            }
                        }
                        eri_1[((p * n + q) * n + r) * n + s] = element;
                    }
                }
            }
        }
        let polarized = OrbitalIntegrals {
            overlap: rotate(&integrals.overlap),
            full_core: rotate(&integrals.full_core),
            external: integrals.external.each_ref().map(rotate),
            eri: eri_1,
            nuclear: integrals.nuclear,
        };
        let (hf_energy, s) = polarized.matrix_element(bra, bra, true);
        let (product_energy, p) = polarized.permutation(bra, bra, identity, true);
        hf_energy / s - product_energy / p
    };
    let hl = evaluate(&monomers[0], &monomers[1], &opts, SaptVariant::Response);
    let hl_error = (energy(0.0, 0) - hl.e_exch).abs();
    assert!(hl_error < 1e-11);
    let mut failures = Vec::new();
    let names = ["A", "B"].map(|s| format!("HL coupled exchange-induction {s}"));
    let stages = [
        ([0.16, 0.12, 0.08, 0.06], stencils::Stencil::Second),
        ([2.0, 1.5, 1.0, 0.75], stencils::Stencil::Fourth),
    ];
    stencils::fit_components_staged(
        &names,
        &stages,
        |steps, stencil| {
            (0..2)
                .map(|i| {
                    steps.map(|h| {
                        let first = stencil.first(
                            energy(h, i),
                            energy(-h, i),
                            energy(2.0 * h, i),
                            energy(-2.0 * h, i),
                            h,
                        );
                        (first - hl.e_exch_ind_directions[i]).abs()
                    })
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("HL exchange induction", &failures);
    println!("HL exchange {:+.15e} Eh, first variation {:?} Eh, first-order permutation residual {hl_error:.3e}", hl.e_exch, hl.e_exch_ind_directions);
}
