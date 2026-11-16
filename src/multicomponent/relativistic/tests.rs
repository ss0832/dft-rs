// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::*;
use crate::acceptance::{components, stencils};
use crate::dfpt::orbitals::generalized_orbitals;
use crate::integrals::analytic_basis::{kinetic, overlap};
use crate::integrals::dirac::{restricted_kinetic_balance, GaussianNucleus};
use crate::periodic::spinor_scf::SpinorFields;
use crate::properties::no_pair::no_pair_projector_taylor;
use crate::properties::no_pair::stationary::DiracCoulombBackend;

/// A single electron has no electron–electron interaction.
struct OneElectron {
    metric: [BlochMatrix; 1],
    core: [BlochMatrix; 1],
}

impl SpinorEnergyBackend for OneElectron {
    fn overlap(&self) -> &[BlochMatrix] {
        &self.metric
    }
    fn initial_fock(&self) -> &[BlochMatrix] {
        &self.core
    }
    fn electrons(&self) -> f64 {
        1.0
    }
    fn quadratic_density_energy(&self) -> bool {
        true
    }
    fn evaluate(&self, density: &[BlochMatrix]) -> Result<SpinorFields> {
        let energy = density[0].trace_product(&self.core[0]);
        let mut terms = [0.0; 10];
        terms[0] = energy;
        Ok(SpinorFields {
            fock: self.core.to_vec(),
            energy,
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

fn shell(center: [f64; 3], exponent: f64) -> AnalyticShell<f64> {
    AnalyticShell::normalized(0, center, vec![exponent], vec![1.0]).unwrap()
}

fn gaussian_product(
    first: &AnalyticShell<f64>,
    second: &AnalyticShell<f64>,
    charge: f64,
    rms_radius: f64,
) -> GaussianNucleus<f64> {
    let a = first.exponents[0];
    let b = second.exponents[0];
    GaussianNucleus {
        center: std::array::from_fn(|d| (a * first.center[d] + b * second.center[d]) / (a + b)),
        charge,
        rms_radius: (rms_radius * rms_radius + 1.5 / (a + b)).sqrt(),
    }
}

fn gaussian_repulsion(a: GaussianNucleus<f64>, b: GaussianNucleus<f64>) -> f64 {
    let distance = (0..3)
        .map(|d| (a.center[d] - b.center[d]).powi(2))
        .sum::<f64>()
        .sqrt();
    let omega = (1.5 / (a.rms_radius.powi(2) + b.rms_radius.powi(2))).sqrt();
    a.charge * b.charge * libm::erf(omega * distance) / distance
}

fn prepared_oracle_nucleus(
    shells: &[AnalyticShell<f64>; 2],
    fixed: GaussianNucleus<f64>,
    mass: f64,
    charge: f64,
    radius: f64,
) -> (Matrix, Matrix) {
    let mut metric = Matrix::zeros(2, 2);
    let mut core = Matrix::zeros(2, 2);
    for i in 0..2 {
        for j in 0..2 {
            metric[(i, j)] = overlap(&shells[i], &shells[j]).unwrap().values[0];
            let product = gaussian_product(&shells[i], &shells[j], charge * metric[(i, j)], radius);
            core[(i, j)] = kinetic(&shells[i], &shells[j]).unwrap().values[0] / mass
                + gaussian_repulsion(product, fixed);
        }
    }
    (metric, core)
}

fn attraction_oracle(
    electronic: &[AnalyticShell<f64>; 2],
    nucleus: GaussianNucleus<f64>,
    c: f64,
) -> BlochMatrix {
    let v = restricted_kinetic_balance(electronic, &[nucleus], c, 0).unwrap();
    let mut out = BlochMatrix::zeros(8, 8);
    for p in 0..4 {
        for q in 0..4 {
            if p / 2 == q / 2 {
                out.re[(p, q)] = v.potential.values[2 * (p % 2) + q % 2];
            }
            out.re[(p + 4, q + 4)] = v.small_potential_real.values[4 * p + q];
            out.im[(p + 4, q + 4)] = v.small_potential_imaginary.values[4 * p + q];
        }
    }
    out
}

#[test]
fn joint_no_pair_coulomb_scf_matches_independent_variational_energy() {
    let electronic = [
        shell([-0.31, 0.26, 0.12], 0.77),
        shell([1.19, -0.41, 0.53], 0.91),
    ];
    let quantum = [
        shell([0.35, 1.27, -0.48], 1.19),
        shell([0.57, 0.94, -0.23], 1.82),
    ];
    let fixed = GaussianNucleus {
        center: [-0.61, -0.34, -0.28],
        charge: 1.25,
        rms_radius: 0.12,
    };
    let c: f64 = 11.7;
    let radius: f64 = 0.081;
    let charge = 0.87;
    let integrals = restricted_kinetic_balance(&electronic, &[fixed], c, 0).unwrap();
    let metric = BlochMatrix::from_pair((
        matrix(integrals.overlap),
        matrix(integrals.overlap_imaginary),
    ));
    let mut core = BlochMatrix::from_pair((
        matrix(integrals.hamiltonian_real),
        matrix(integrals.hamiltonian_imaginary),
    ));
    // A specified spin splitting isolates the occupied level.
    for i in 0..core.rows() {
        for j in 0..core.cols() {
            core.re[(i, j)] += if (i / 2) % 2 == 0 { -0.023 } else { 0.023 } * metric.re[(i, j)];
        }
    }
    let space = NoPairSpace::new(&core, &metric, -c * c, 1e-9).unwrap();
    let initial_projector = no_pair_projector_taylor(
        std::slice::from_ref(&core),
        std::slice::from_ref(&metric),
        0,
        -c * c,
        1e-9,
    )
    .unwrap();
    let mut projector_error = space
        .lift_density(&BlochMatrix::from_pair((
            Matrix::identity(4),
            Matrix::zeros(4, 4),
        )))
        .unwrap();
    projector_error.add_scaled(&initial_projector.projector[0], -1.0);
    assert!(max_abs(&projector_error.re).max(max_abs(&projector_error.im)) < 1e-11);
    let (nuclear_metric, nuclear_core) =
        prepared_oracle_nucleus(&quantum, fixed, 19.0, charge, radius);
    let provider = OneElectron {
        metric: [metric],
        core: [core],
    };
    let model = JointNoPairModel {
        electronic: &provider,
        electronic_space: &space,
        electronic_shells: &electronic,
        speed_of_light: c,
        nucleus: BalancedQuantumNucleus {
            shells: &quantum,
            overlap: &nuclear_metric,
            one_body: &nuclear_core,
            charge,
            intrinsic_rms_radius: Some(radius),
        },
    };
    let state = model
        .solve(
            &SpinorScfConfig {
                overlap_cutoff: 1e-9,
                energy_tolerance: 2e-13,
                density_tolerance: 2e-12,
                orbital_tolerance: 2e-12,
                ..SpinorScfConfig::default()
            },
            &MulticomponentOptions {
                tolerance: 5e-11,
                max_iterations: 150,
                mixing: 0.7,
            },
        )
        .unwrap();
    assert!(state.converged);
    assert!(state.iterations > 1);
    assert!(state.reciprocity_residual.abs() < 1e-13);
    assert!(state.projection_residual.abs() < 1e-13);
    assert!(state.nuclear_particle_number_error.abs() < 1e-13);
    assert!(state.electronic.particle_number_error.abs() < 1e-13);

    // Every nuclear Gaussian pair can be replaced by a finite Gaussian
    // charge with the product centre and the convolved intrinsic width.
    // Use the one-electron attraction recurrence as an independent ERI oracle.
    let oracle = |i: usize, j: usize| {
        let product = gaussian_product(
            &quantum[i],
            &quantum[j],
            charge * nuclear_metric[(i, j)],
            radius,
        );
        attraction_oracle(&electronic, product, c)
    };
    let pairs = [[oracle(0, 0), oracle(0, 1)], [oracle(1, 0), oracle(1, 1)]];
    let mut interaction = 0.0;
    for (i, row) in pairs.iter().enumerate() {
        for (j, p) in row.iter().enumerate() {
            let value = state.electronic_density.trace_product(p);
            interaction += state.nuclear_density[(i, j)] * value;
            assert!((state.nuclear_fock[(i, j)] - nuclear_core[(i, j)] - value).abs() < 1e-12);
        }
    }
    let core_energy = state.electronic_density.trace_product(&provider.core[0]);
    assert!((state.electronic_energy - core_energy).abs() < 1e-12);
    assert!((state.interaction_energy - interaction).abs() < 1e-12);
    assert!(
        (state.energy - (core_energy + interaction + state.nuclear_one_body_energy)).abs() < 1e-12
    );
    let orth = Orthogonalizer::canonical(&nuclear_metric, 1e-9).unwrap();
    let electronic_metric = space.project_operator(&provider.metric[0]).unwrap();
    let energy = |angle: f64| {
        let (sine, cosine) = angle.sin_cos();
        let orbital = orth
            .x
            .matmul_seq(&Matrix::from_row_major(2, 1, vec![cosine, sine]));
        let density = projector(orbital.as_slice());
        let mut fock = provider.core[0].clone();
        for (i, row) in pairs.iter().enumerate() {
            for (j, p) in row.iter().enumerate() {
                fock.add_scaled(p, density[(i, j)]);
            }
        }
        let fock = space.project_operator(&fock).unwrap();
        let levels = generalized_orbitals(&fock, &electronic_metric, 1e-9).unwrap();
        levels.energies[0] + density.frobenius_dot(&nuclear_core)
    };
    let step = std::f64::consts::PI / 64.0;
    let best = (0..64)
        .map(|i| i as f64 * step)
        .min_by(|&a, &b| energy(a).total_cmp(&energy(b)))
        .unwrap();
    let (mut lo, mut hi) = (best - step, best + step);
    for _ in 0..40 {
        let a = (2.0 * lo + hi) / 3.0;
        let b = (lo + 2.0 * hi) / 3.0;
        if energy(a) < energy(b) {
            hi = b;
        } else {
            lo = a;
        }
    }
    let variational = energy((lo + hi) / 2.0);
    assert!((state.energy - variational).abs() < 1e-11);
    println!(
        "Joint no-pair energy={:.17e} oracle={variational:.17e} electronic={core_energy:.17e} nuclear={:.17e} interaction={:.17e} reciprocity={:.17e} projection={:.17e} electronic_orbital={:.17e} nuclear_orbital={:.17e} nuclear_density={:.17e} gap={:.17e} iterations={}",
        state.energy, state.nuclear_one_body_energy, state.interaction_energy,
        state.reciprocity_residual, state.projection_residual, state.electronic.orbital_residual, state.nuclear_orbital_residual,
        state.nuclear_density_residual, space.spectral_gap, state.iterations
    );
}

#[test]
fn joint_no_pair_coulomb_two_electrons_and_two_quantum_nuclei() {
    let electronic = [
        shell([-0.31, 0.26, 0.12], 0.77),
        shell([1.19, -0.41, 0.53], 0.91),
    ];
    let quantum = [
        [
            shell([0.35, 1.27, -0.48], 1.19),
            shell([0.57, 0.94, -0.23], 1.82),
        ],
        [
            shell([0.18, -0.61, 0.93], 1.43),
            shell([-0.15, -0.71, 1.07], 1.67),
        ],
    ];
    let fixed = GaussianNucleus {
        center: [-0.61, -0.34, -0.28],
        charge: 1.55,
        rms_radius: 0.12,
    };
    let c = 11.7;
    let masses = [19.0, 27.0];
    let charges = [0.76, 0.64];
    let radii = [0.081, 0.062];
    let prepared: [(Matrix, Matrix); 2] = std::array::from_fn(|i| {
        prepared_oracle_nucleus(&quantum[i], fixed, masses[i], charges[i], radii[i])
    });
    let make = |i: usize| BalancedQuantumNucleus {
        shells: &quantum[i],
        overlap: &prepared[i].0,
        one_body: &prepared[i].1,
        charge: charges[i],
        intrinsic_rms_radius: Some(radii[i]),
    };
    let nuclei = [make(0), make(1)];
    let provider = DiracCoulombBackend::new(&electronic, &[fixed], c, 2).unwrap();
    let config = SpinorScfConfig {
        overlap_cutoff: 1e-9,
        energy_tolerance: 2e-13,
        density_tolerance: 2e-12,
        orbital_tolerance: 2e-12,
        ..SpinorScfConfig::default()
    };
    let space = provider.electronic_space(config.overlap_cutoff).unwrap();
    let options = MulticomponentOptions {
        tolerance: 5e-11,
        max_iterations: 150,
        mixing: 0.7,
    };
    let solve = |nuclei| {
        JointNoPairSystem {
            electronic: &provider,
            electronic_space: &space,
            electronic_shells: &electronic,
            speed_of_light: c,
            nuclei,
        }
        .solve(&config, &options)
        .unwrap()
    };
    let state = solve(&nuclei);
    assert!(state.converged && state.iterations > 1);
    let reversed = solve(&[make(1), make(0)]);
    assert!(reversed.converged);
    assert!((state.energy - reversed.energy).abs() < 1e-12);
    assert!(state.max_electron_nuclear_reciprocity_defect < 1e-13);
    assert!(state.max_nuclear_pair_reciprocity_defect < 1e-13);
    assert!(state.projection_residual.abs() < 1e-13);
    let product = |a: usize, i: usize, j: usize| {
        gaussian_product(
            &quantum[a][i],
            &quantum[a][j],
            charges[a] * prepared[a].0[(i, j)],
            radii[a],
        )
    };
    let pair_value = |i, j, k, l| gaussian_repulsion(product(0, i, j), product(1, k, l));
    let pair_energy = |density: [&Matrix; 2]| {
        let mut energy = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        energy += density[0][(i, j)] * density[1][(k, l)] * pair_value(i, j, k, l);
                    }
                }
            }
        }
        energy
    };
    let density = [&state.nuclei[0].density, &state.nuclei[1].density];
    let oracle_pair = pair_energy(density);
    assert!((state.nuclear_pair_energy - oracle_pair).abs() < 1e-12);
    let mut oracle_interaction = 0.0;
    for (a, result) in state.nuclei.iter().enumerate() {
        assert!(
            result.orbital_residual < options.tolerance
                && result.density_residual < options.tolerance
        );
        assert!(result.particle_number_error.abs() < 1e-13);
        let mut swapped = result.density.clone();
        add(&mut swapped, &reversed.nuclei[1 - a].density, -1.0);
        assert!(max_abs(&swapped) < 1e-11);
        for i in 0..2 {
            for j in 0..2 {
                let field = attraction_oracle(&electronic, product(a, i, j), c);
                let electron = state.electronic_density.trace_product(&field);
                oracle_interaction += electron * result.density[(i, j)];
                let mut pair = 0.0;
                for k in 0..2 {
                    for l in 0..2 {
                        let g = if a == 0 {
                            pair_value(i, j, k, l)
                        } else {
                            pair_value(k, l, i, j)
                        };
                        pair += g * density[1 - a][(k, l)];
                    }
                }
                assert!(
                    (result.fock[(i, j)] - prepared[a].1[(i, j)] - electron - pair).abs() < 1e-12
                );
            }
        }
    }
    let base = provider
        .evaluate(std::slice::from_ref(&state.electronic_density))
        .unwrap();
    let nuclear_one_body: f64 = state.nuclei.iter().map(|n| n.one_body_energy).sum();
    assert!((state.electron_nuclear_energy - oracle_interaction).abs() < 1e-12);
    assert!(
        (state.energy - (base.energy + nuclear_one_body + oracle_interaction + oracle_pair)).abs()
            < 1e-12
    );

    let mut first = NuclearIteration::new(&nuclei[0], config.overlap_cutoff).unwrap();
    let mut second = NuclearIteration::new(&nuclei[1], config.overlap_cutoff).unwrap();
    first.density = density[0].clone();
    second.density = density[1].clone();
    let pair = nuclear_coulomb::pair(
        [&nuclei[0], &nuclei[1]],
        [&first.density, &second.density],
        [&first.offsets, &second.offsets],
    )
    .unwrap();
    let potentials = [&pair.first, &pair.second];
    let slots = [(0, 0), (1, 0), (1, 1)];
    let names: Vec<_> = (0..2)
        .flat_map(|a| slots.map(move |(i, j)| format!("quantum pair density {a}[{i},{j}]")))
        .collect();
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[([0.06, 0.04, 0.03, 0.02], stencils::Stencil::Second)],
        |steps, stencil| {
            (0..2)
                .flat_map(|a| {
                    slots.map(|(i, j)| {
                        steps.map(|h| {
                            let sample = |t: f64| {
                                let mut p = [density[0].clone(), density[1].clone()];
                                p[a][(i, j)] += t;
                                if i != j {
                                    p[a][(j, i)] += t;
                                }
                                add(&mut p[1 - a], density[1 - a], t * t);
                                pair_energy([&p[0], &p[1]])
                            };
                            let derivative = potentials[a][(i, j)] * if i == j { 1.0 } else { 2.0 };
                            (stencil.first(sample(h), sample(-h), 0.0, 0.0, h) - derivative).abs()
                        })
                    })
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures(
        "all independent two-nucleus Coulomb density entries",
        &failures,
    );
    println!(
        "Joint two-nucleus energy={:.17e} reversed={:.17e} electronic={:.17e} nuclear_one_body={nuclear_one_body:.17e} electron_nuclear={:.17e} nuclear_pair={:.17e} pair_oracle={oracle_pair:.17e} electronic_orbital={:.17e} nuclear_orbital={:?} nuclear_density={:?} pair_defect={:.17e} projection={:.17e} iterations={}",
        state.energy, reversed.energy, state.electronic_energy, state.electron_nuclear_energy,
        state.nuclear_pair_energy, state.electronic.orbital_residual,
        state.nuclei.iter().map(|n| n.orbital_residual).collect::<Vec<_>>(),
        state.nuclei.iter().map(|n| n.density_residual).collect::<Vec<_>>(),
        state.max_nuclear_pair_reciprocity_defect, state.projection_residual, state.iterations,
    );
}
