// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::*;
use crate::acceptance::{components, stencils};
use crate::integrals::analytic::{self, Primitive};
use crate::integrals::dirac::GaussianNucleus;
use crate::periodic::spinor_scf::{
    run_spinor_scf, SpinorEnergyBackend, SpinorFields, SpinorScfConfig,
};
use crate::properties::no_pair::stationary::{DiracCoulombBackend, NoPairBackend};
use stencils::Stencil;

const N: usize = 8;
const C: f64 = 7.4;

fn shells() -> [AnalyticShell<f64>; 2] {
    [
        AnalyticShell::normalized(0, [-0.31, 0.26, 0.12], vec![0.77], vec![1.0]).unwrap(),
        AnalyticShell::normalized(0, [1.19, -0.41, 0.53], vec![0.91], vec![1.0]).unwrap(),
    ]
}

type CartesianProducts = [[Vec<(usize, usize, Complex)>; 4]; 4];

fn cartesian_products(c: f64) -> CartesianProducts {
    let pauli = [
        [[ZERO, ONE], [ONE, ZERO]],
        [[ZERO, NEG_IM], [IM, ZERO]],
        [[ONE, ZERO], [ZERO, NEG]],
    ];
    let expansion = |component: usize, spin: usize| -> [Complex; 4] {
        let mut lanes = [ZERO; 4];
        if component < 2 {
            lanes[0] = if component == spin { ONE } else { ZERO };
        } else {
            for (axis, matrix) in pauli.iter().enumerate() {
                lanes[1 + axis] = IM * matrix[spin][component - 2] * (0.5 / c);
            }
        }
        lanes
    };
    std::array::from_fn(|p| {
        std::array::from_fn(|q| {
            if (p < 2) != (q < 2) {
                return Vec::new();
            }
            let mut terms = Vec::new();
            for a in 0..4 {
                for b in 0..4 {
                    let coefficient = (0..2).fold(ZERO, |sum, spin| {
                        sum + expansion(p, spin)[a].conj() * expansion(q, spin)[b]
                    });
                    if coefficient.abs() > 0.0 {
                        terms.push((a, b, coefficient));
                    }
                }
            }
            terms
        })
    })
}

/// Independent primitive oracle. Expand the physical spinor wavefunctions
/// into Cartesian s/p Gaussians before forming any two-electron product.
/// No centre jets or production quaternion channels are used.
fn dense_integrals(shells: &[AnalyticShell<f64>; 2], c: f64) -> Vec<Complex> {
    let products = cartesian_products(c);
    let primitive = |shell: usize, lane: usize| {
        let basis = &shells[shell];
        let exponent = basis.exponents[0];
        let mut angular = [0; 3];
        let coefficient = if lane == 0 {
            basis.coefficients[0]
        } else {
            angular[lane - 1] = 1;
            2.0 * exponent * basis.coefficients[0]
        };
        (
            Primitive {
                angular,
                exponent,
                center: basis.center,
            },
            coefficient,
        )
    };
    let index = |[p, q, r, s]: [usize; 4]| ((p * N + q) * N + r) * N + s;
    let mut dense = vec![ZERO; N.pow(4)];
    for quartet in 0..16 {
        let spatial = std::array::from_fn::<_, 4, _>(|axis| (quartet >> axis) & 1);
        let mut bare = [0.0; 256];
        for (lane_index, out) in bare.iter_mut().enumerate() {
            let [(a, wa), (b, wb), (c, wc), (d, wd)] = std::array::from_fn(|axis| {
                primitive(spatial[axis], (lane_index >> (2 * axis)) & 3)
            });
            *out = wa
                * wb
                * wc
                * wd
                * analytic::repulsion(&a, &b, &c, &d, GaussianKernel::Coulomb, 0).unwrap();
        }
        for (p, row) in products.iter().enumerate() {
            for (q, left) in row.iter().enumerate() {
                for (r, row) in products.iter().enumerate() {
                    for (s, right) in row.iter().enumerate() {
                        let mut g = ZERO;
                        for &(a, b, x) in left {
                            for &(c, d, y) in right {
                                g += x * y * bare[a + 4 * b + 16 * c + 64 * d];
                            }
                        }
                        let aos = [p, q, r, s];
                        dense[index(std::array::from_fn(|i| 2 * aos[i] + spatial[i]))] = g;
                    }
                }
            }
        }
    }
    dense
}

fn contract(g: &[Complex], density: &BlochMatrix) -> (BlochMatrix, BlochMatrix) {
    let mut direct = BlochMatrix::zeros(N, N);
    let mut exchange = BlochMatrix::zeros(N, N);
    for p in 0..N {
        for q in 0..N {
            let mut j = ZERO;
            let mut k = ZERO;
            for r in 0..N {
                for s in 0..N {
                    let value = entry(density, s, r);
                    j += value * g[((p * N + q) * N + r) * N + s];
                    k += value * g[((p * N + s) * N + r) * N + q];
                }
            }
            accumulate(&mut direct, p, q, j);
            accumulate(&mut exchange, p, q, k);
        }
    }
    (direct, exchange)
}

fn sample_density(phase: f64, count: usize) -> BlochMatrix {
    let mut density = BlochMatrix::zeros(N, N);
    for k in 0..count {
        let vector: [Complex; N] = std::array::from_fn(|i| {
            let angle = phase + ((i + 1) * (k + 1)) as f64;
            Complex::new(0.4 * angle.sin(), 0.31 * (1.7 * angle).cos())
        });
        for i in 0..N {
            for j in 0..N {
                accumulate(&mut density, i, j, vector[i] * vector[j].conj());
            }
        }
    }
    density
}

fn difference(a: &BlochMatrix, b: &BlochMatrix) -> f64 {
    let mut delta = a.clone();
    delta.add_scaled(b, -1.0);
    delta
        .re
        .as_slice()
        .iter()
        .zip(delta.im.as_slice())
        .map(|(&r, &i)| r.hypot(i))
        .fold(0.0, f64::max)
}

fn hermitian_directions() -> Vec<(String, BlochMatrix)> {
    let mut directions = Vec::new();
    for i in 0..N {
        for j in 0..=i {
            let mut direction = BlochMatrix::zeros(N, N);
            direction.re[(i, j)] = 1.0;
            direction.re[(j, i)] = 1.0;
            directions.push((format!("density re[{i},{j}]"), direction));
            if i != j {
                let mut direction = BlochMatrix::zeros(N, N);
                direction.im[(i, j)] = 1.0;
                direction.im[(j, i)] = -1.0;
                directions.push((format!("density im[{i},{j}]"), direction));
            }
        }
    }
    directions
}

#[test]
fn dirac_coulomb_all_spin_blocks_and_density_fits() {
    let shells = shells();
    let dense = dense_integrals(&shells, C);
    let density = sample_density(0.47, 2);
    let fields = electronic_coulomb_exchange(&shells, &density, C).unwrap();
    let (j, k) = contract(&dense, &density);
    let j_error = difference(&fields.direct, &j);
    let k_error = difference(&fields.exchange, &k);
    assert!(j_error.max(k_error) < 1e-13, "J {j_error}, K {k_error}");
    assert!(fields.hermiticity_defect < 1e-13);
    assert!(fields.exchange.im[(0, 7)].abs() > 1e-5);
    let one = electronic_coulomb_exchange(&shells, &sample_density(0.47, 1), C).unwrap();
    assert!((one.direct_energy + one.exchange_energy).abs() < 1e-13);
    let mut fock = fields.direct;
    fock.add_scaled(&fields.exchange, -1.0);
    let quadratic = sample_density(0.81, 3);
    let (names, directions): (Vec<_>, Vec<_>) = hermitian_directions()
        .into_iter()
        .map(|(name, direction)| (format!("Dirac Coulomb {name}"), direction))
        .unzip();
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[([0.06, 0.04, 0.03, 0.02], Stencil::Second)],
        |steps, stencil| {
            directions
                .iter()
                .map(|direction| {
                    let sample = |h: f64| {
                        let mut p = density.clone();
                        p.add_scaled(direction, h);
                        p.add_scaled(&quadratic, h * h);
                        let (mut j, k) = contract(&dense, &p);
                        j.add_scaled(&k, -1.0);
                        0.5 * p.trace_product(&j)
                    };
                    steps.map(|h| {
                        (stencil.first(sample(h), sample(-h), 0.0, 0.0, h)
                            - direction.trace_product(&fock))
                        .abs()
                    })
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures(
        "all independent four-component Hermitian density entries",
        &failures,
    );
    println!(
        "Dirac Coulomb direct={:.17e} exchange={:.17e} direct_defect={j_error:.17e} exchange_defect={k_error:.17e} hermiticity={:.17e} self_cancellation={:.17e}",
        fields.direct_energy, fields.exchange_energy, fields.hermiticity_defect, one.direct_energy + one.exchange_energy,
    );
}

struct Dense<'a> {
    core: &'a DiracCoulombBackend<'a>,
    integrals: &'a [Complex],
    classical: f64,
}
impl SpinorEnergyBackend for Dense<'_> {
    fn overlap(&self) -> &[BlochMatrix] {
        self.core.overlap()
    }
    fn initial_fock(&self) -> &[BlochMatrix] {
        self.core.initial_fock()
    }
    fn electrons(&self) -> f64 {
        self.core.electrons()
    }
    fn quadratic_density_energy(&self) -> bool {
        true
    }
    fn evaluate(&self, density: &[BlochMatrix]) -> Result<SpinorFields> {
        let p = &density[0];
        let (j, k) = contract(self.integrals, p);
        let mut fock = self.initial_fock()[0].clone();
        let mut terms = [0.0; 10];
        terms[0] = self.classical;
        terms[1] = p.trace_product(&fock);
        terms[4] = 0.5 * p.trace_product(&j);
        terms[5] = -0.5 * p.trace_product(&k);
        fock.add_scaled(&j, 1.0);
        fock.add_scaled(&k, -1.0);
        Ok(SpinorFields {
            fock: vec![fock],
            energy: terms.iter().sum(),
            energy_components: terms,
            xc_component_energies: vec![],
            nuclear_soc_energy: 0.0,
            operator_converged: true,
            coulomb_operator_defect: 0.0,
            xc_grid_number_defect: 0.0,
            xc_diagnostics: None,
        })
    }
}

#[test]
fn two_electron_no_pair_dirac_coulomb_scf_agrees_with_dense_oracle() {
    let shells = shells();
    let nucleus = GaussianNucleus {
        center: [-0.21, 0.12, -0.35],
        charge: 2.0,
        rms_radius: 0.17,
    };
    let backend = DiracCoulombBackend::new(&shells, &[nucleus], C, 2).unwrap();
    let oracle = Dense {
        core: &backend,
        integrals: &dense_integrals(&shells, C),
        classical: 0.0,
    };
    let config = SpinorScfConfig {
        overlap_cutoff: 1e-9,
        orbital_tolerance: 1e-11,
        energy_tolerance: 2e-13,
        density_tolerance: 1e-11,
        ..SpinorScfConfig::default()
    };
    let space = backend.electronic_space(config.overlap_cutoff).unwrap();
    let state = run_spinor_scf(&NoPairBackend::new(&space, &backend).unwrap(), &config).unwrap();
    let reference = run_spinor_scf(&NoPairBackend::new(&space, &oracle).unwrap(), &config).unwrap();
    assert!(state.converged && reference.converged);
    let terms = state.fields.energy_components;
    assert!((state.fields.energy - reference.fields.energy).abs() < 1e-12);
    for i in [4, 5] {
        assert!((terms[i] - reference.fields.energy_components[i]).abs() < 1e-12);
    }
    assert!(difference(&state.density[0], &reference.density[0]) < 1e-11);
    println!(
        "Dirac Coulomb SCF energy={:.17e} oracle={:.17e} direct={:.17e} exchange={:.17e} orbital={:.17e} particle={:.17e} iterations={}",
        state.fields.energy, reference.fields.energy, terms[4], terms[5], state.orbital_residual,
        state.particle_number_error, state.iterations,
    );
}

#[test]
fn dirac_coulomb_closed_shell_recovers_nonrelativistic_gaussian_limit() {
    let exponent: f64 = 0.77;
    let center = [0.17, -0.31, 0.12];
    let nucleus = GaussianNucleus {
        center,
        charge: 2.0,
        rms_radius: 0.17,
    };
    let basis = [AnalyticShell::normalized(0, center, vec![exponent], vec![1.0]).unwrap()];
    let omega2 = 1.5 / nucleus.rms_radius.powi(2);
    let factor = 2.0 / std::f64::consts::PI.sqrt();
    let attraction =
        -nucleus.charge * factor * (2.0 * exponent * omega2 / (2.0 * exponent + omega2)).sqrt();
    let expected = 2.0 * (1.5 * exponent + attraction) + factor * exponent.sqrt();
    let mut measured = [0.0; 4];
    let inverse_c = [1.0 / 25.0, 1.0 / 50.0, 1.0 / 100.0, 1.0 / 200.0];
    let config = SpinorScfConfig {
        overlap_cutoff: 1e-12,
        ..SpinorScfConfig::default()
    };
    for (energy, inverse) in measured.iter_mut().zip(inverse_c) {
        let backend = DiracCoulombBackend::new(&basis, &[nucleus], 1.0 / inverse, 2).unwrap();
        let space = backend.electronic_space(config.overlap_cutoff).unwrap();
        let state =
            run_spinor_scf(&NoPairBackend::new(&space, &backend).unwrap(), &config).unwrap();
        assert!(state.converged && state.orbital_residual < 1e-11);
        *energy = state.fields.energy;
    }
    let errors = measured.map(|energy| (energy - expected).abs());
    assert!(components::component_fit_of_order(
        "Dirac Coulomb nonrelativistic limit",
        &inverse_c,
        &errors,
        2.0
    ));
    println!("Dirac Coulomb nonrelativistic oracle={expected:.17e} energies={measured:?}");
}

#[test]
fn balanced_electric_moments_match_cartesian_gaussian_products() {
    let shells = shells();
    let products = cartesian_products(C);
    let powers = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [2, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 2, 0],
        [0, 1, 1],
        [0, 0, 2],
    ];
    let origin = [0.17, -0.23, 0.41];
    let calculated =
        crate::integrals::dirac::balanced_electric_moments(&shells, C, powers, origin).unwrap();
    let mut defect = [0.0_f64; 9];
    for p in 0..N {
        for q in 0..N {
            let (a, b) = (&shells[p % 2], &shells[q % 2]);
            let (ea, eb) = (a.exponents[0], b.exponents[0]);
            let exponent = ea + eb;
            let centre = std::array::from_fn::<_, 3, _>(|i| {
                (ea * a.center[i] + eb * b.center[i]) / exponent
            });
            let distance2 = (0..3)
                .map(|i| (a.center[i] - b.center[i]).powi(2))
                .sum::<f64>();
            let overlap = a.coefficients[0]
                * b.coefficients[0]
                * (std::f64::consts::PI / exponent).powf(1.5)
                * (-ea * eb * distance2 / exponent).exp();
            let variance = 0.5 / exponent;
            for (term, (matrix, power)) in calculated.iter().zip(powers).enumerate() {
                let expected = products[p / 2][q / 2]
                    .iter()
                    .fold(ZERO, |sum, &(l, m, spin)| {
                        let mut gaussian = 1.0;
                        for axis in 0..3 {
                            // Expand the independent Cartesian Gaussian-product
                            // polynomial about its mean. Only even powers survive.
                            let mut shifts = vec![centre[axis] - origin[axis]; power[axis]];
                            if l == axis + 1 {
                                shifts.push(centre[axis] - a.center[axis]);
                                gaussian *= 2.0 * ea;
                            }
                            if m == axis + 1 {
                                shifts.push(centre[axis] - b.center[axis]);
                                gaussian *= 2.0 * eb;
                            }
                            let central = [1.0, 0.0, variance, 0.0, 3.0 * variance.powi(2)];
                            let moment = (0..(1_usize << shifts.len()))
                                .map(|mask| {
                                    shifts
                                        .iter()
                                        .enumerate()
                                        .filter(|(i, _)| mask & (1 << i) == 0)
                                        .map(|(_, shift)| shift)
                                        .product::<f64>()
                                        * central[mask.count_ones() as usize]
                                })
                                .sum::<f64>();
                            gaussian *= moment;
                        }
                        sum + spin * (overlap * gaussian)
                    });
                let actual = Complex::new(matrix.re[(p, q)], matrix.im[(p, q)]);
                defect[term] = defect[term].max((expected - actual).abs());
            }
        }
    }
    println!("Balanced electric moments Gaussian product defects={defect:?}");
    assert!(defect.iter().all(|&value| value < 2e-14));
}

#[path = "current_tests.rs"]
mod current_tests;
