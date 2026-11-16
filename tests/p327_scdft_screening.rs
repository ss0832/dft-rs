// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Raw static screened-pair matrix and independent reciprocal density fits.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::data;
use dft_rs::dfpt::coulomb::{plane_wave_pair, reciprocal_waves};
use dft_rs::dfpt::orbitals::{adjoint, generalized_orbitals, matmul, BandOrbitals};
use dft_rs::dfpt::thermodynamic::divided_difference;
use dft_rs::grid::GridOptions;
use dft_rs::linalg::symmetric_eigen;
use dft_rs::math::Vec3;
use dft_rs::periodic::bloch::BlochMatrix;
use dft_rs::periodic::electron_phonon::screening::{
    screened_pair_coulomb, static_screened_interaction, CoulombScreeningOptions, GammaHead,
};
use dft_rs::periodic::occupations::{OccupationScheme, BOLTZMANN_HARTREE_PER_K};
use dft_rs::periodic::scf::{run_periodic_scf, PeriodicScfOptions};
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::response::complex::Complex;
use dft_rs::{resolve, Atom, DftError, FunctionalSpec, Molecule};
use std::f64::consts::PI;

#[test]
fn head_only_compressibility_and_unscreened_body_are_closed_form() {
    let mut head = BlochMatrix::zeros(1, 1);
    head.re[(0, 0)] = 1.0;
    let volume = 21.0;
    let kt = 0.014;
    let coefficient = -1.0 / (2.0 * volume * kt);
    let result =
        static_screened_interaction(&[Vec3::zero()], &head, &[coefficient], 1e-12).unwrap();
    assert!((result.independent_response.re[(0, 0)] - coefficient).abs() < 1e-15);
    assert!((result.interaction.re[(0, 0)] / volume - 2.0 * kt).abs() < 1e-15);
    assert!(matches!(
        static_screened_interaction(&[Vec3::zero()], &head, &[0.0], 1e-12),
        Err(DftError::Unsupported { feature, .. }) if feature.contains("without thermal compressibility")
    ));
    let bare = Vec3::new(0.3, 0.5, 0.8);
    let result = static_screened_interaction(&[bare], &head, &[0.0], 1e-12).unwrap();
    assert!((result.interaction.re[(0, 0)] - 4.0 * PI / bare.norm2()).abs() < 1e-13);
    println!(
        "RPA scalar limits: bare={:.14e} Hartree Bohr^3, thermal head={:.14e} Hartree, chi0={coefficient:.14e} Bohr^-3/Hartree",
        result.interaction.re[(0, 0)],
        2.0 * kt,
    );
}

fn at(matrix: &BlochMatrix, i: usize, j: usize) -> Complex {
    Complex::new(matrix.re[(i, j)], matrix.im[(i, j)])
}

fn add(matrix: &mut BlochMatrix, i: usize, j: usize, value: Complex) {
    matrix.re[(i, j)] += value.re;
    matrix.im[(i, j)] += value.im;
}

/// Explicit pivoted complex inverse for the few-mode oracle.
fn independent_inverse(matrix: &[Vec<Complex>]) -> Vec<Vec<Complex>> {
    let n = matrix.len();
    let mut augmented: Vec<_> = (0..n)
        .map(|i| {
            matrix[i]
                .iter()
                .copied()
                .chain((0..n).map(|j| Complex::from(if i == j { 1.0 } else { 0.0 })))
                .collect::<Vec<_>>()
        })
        .collect();
    for j in 0..n {
        let pivot = (j..n)
            .max_by(|&a, &b| augmented[a][j].abs().total_cmp(&augmented[b][j].abs()))
            .unwrap();
        augmented.swap(j, pivot);
        let scale = augmented[j][j];
        assert!(scale.abs() > 0.0);
        for value in &mut augmented[j] {
            *value = *value / scale;
        }
        let lead = augmented[j].clone();
        for (i, row) in augmented.iter_mut().enumerate() {
            if i != j {
                let multiplier = row[j];
                for (value, &reference) in row.iter_mut().zip(&lead) {
                    *value -= multiplier * reference;
                }
            }
        }
    }
    augmented.into_iter().map(|row| row[n..].to_vec()).collect()
}

struct Reference<'a> {
    cell: &'a Cell,
    points: &'a [Vec3],
    bands: Vec<BandOrbitals>,
    energies: Vec<f64>,
    offsets: Vec<usize>,
    volume: f64,
    kt: f64,
    mu: f64,
}

struct Transfer {
    waves: Vec<Vec3>,
    pairs: Vec<(usize, usize)>,
    forms: BlochMatrix,
}

impl Reference<'_> {
    fn occupation(&self, energy: f64) -> f64 {
        1.0 / (1.0 + ((energy - self.mu) / self.kt).exp())
    }

    fn volume_weight(&self) -> f64 {
        2.0 / (self.volume * self.points.len() as f64)
    }

    fn transfer(&self, shift: usize, cutoff: f64) -> Transfer {
        let nk = self.points.len();
        let reduced = shift as f64 / nk as f64;
        let q = self.cell.lattice().reciprocal()[0]
            * if reduced > 0.5 {
                reduced - 1.0
            } else {
                reduced
            };
        let mut waves = reciprocal_waves(self.cell, q, cutoff).unwrap();
        if shift == 0 {
            waves.insert(0, Vec3::zero());
        }
        let mut pairs = Vec::new();
        for k in 0..nk {
            let next = (k + shift) % nk;
            let start = self.offsets[next];
            for a in 0..self.bands[next].energies.len() {
                for b in 0..self.bands[k].energies.len() {
                    pairs.push((start + a, self.offsets[k] + b));
                }
            }
        }
        let mut forms = BlochMatrix::zeros(waves.len(), pairs.len());
        for (g, &wave) in waves.iter().enumerate() {
            let mut column = 0;
            for (k, &point) in self.points.iter().enumerate() {
                let next = (k + shift) % nk;
                let ket = &self.bands[k].coefficients;
                let bra = &self.bands[next].coefficients;
                let integral = if wave == Vec3::zero() {
                    BlochMatrix::from_pair(self.cell.overlap(point))
                } else {
                    plane_wave_pair(
                        self.cell,
                        point,
                        q,
                        wave,
                        self.cell.hartree_real_space_cutoff(),
                    )
                    .unwrap()
                };
                let projected =
                    matmul(&adjoint(bra).unwrap(), &matmul(&integral, ket).unwrap()).unwrap();
                for a in 0..projected.rows() {
                    for b in 0..projected.cols() {
                        add(&mut forms, g, column, at(&projected, a, b).conj());
                        column += 1;
                    }
                }
            }
        }
        Transfer {
            waves,
            pairs,
            forms,
        }
    }

    fn density(
        &self,
        transfer: &Transfer,
        source: usize,
        phase: Complex,
        step: f64,
    ) -> Vec<Complex> {
        let n = self.energies.len();
        let mut hamiltonian = BlochMatrix::zeros(n, n);
        for (i, &energy) in self.energies.iter().enumerate() {
            hamiltonian.re[(i, i)] = energy;
        }
        // Add a physical Hermitian source. For nonzero q, the adjoint acts
        // in the opposite transfer sector and does not enter the sampled
        // transition density at first order.
        for (j, &(a, b)) in transfer.pairs.iter().enumerate() {
            let element = at(&transfer.forms, source, j).conj() * phase * step;
            add(&mut hamiltonian, a, b, element);
            add(&mut hamiltonian, b, a, element.conj());
        }
        let (values, vectors) = symmetric_eigen(&hamiltonian.real_embedding()).unwrap();
        let occupations: Vec<_> = values.iter().map(|&e| self.occupation(e)).collect();
        (0..transfer.waves.len())
            .map(|g| {
                transfer
                    .pairs
                    .iter()
                    .enumerate()
                    .fold(Complex::ZERO, |sum, (t, &(a, b))| {
                        let mut p = Complex::ZERO;
                        for (i, &occupation) in occupations.iter().enumerate() {
                            p += Complex::new(vectors[(a, i)], vectors[(n + a, i)])
                                * (vectors[(b, i)] * occupation);
                        }
                        sum + at(&transfer.forms, g, t) * p * self.volume_weight()
                    })
            })
            .collect()
    }

    fn screened_oracle(&self, transfer: &Transfer) -> Vec<Vec<Complex>> {
        let n = transfer.waves.len();
        let fermi_secants: Vec<_> = transfer
            .pairs
            .iter()
            .map(|&(a, b)| {
                let (ea, eb) = (self.energies[a], self.energies[b]);
                let (fa, fb) = (self.occupation(ea), self.occupation(eb));
                let slope = if (ea - eb).abs() < 1e-11 {
                    let f = self.occupation(0.5 * (ea + eb));
                    -f * (1.0 - f) / self.kt
                } else {
                    (fa - fb) / (ea - eb)
                };
                slope * self.volume_weight()
            })
            .collect();
        let metric: Vec<Vec<_>> = (0..n)
            .map(|g| {
                (0..n)
                    .map(|h| {
                        let chi =
                            fermi_secants
                                .iter()
                                .enumerate()
                                .fold(Complex::ZERO, |sum, (i, &d)| {
                                    sum + at(&transfer.forms, g, i)
                                        * at(&transfer.forms, h, i).conj()
                                        * d
                                });
                        let bare = if g == h {
                            transfer.waves[g].norm2() / (4.0 * PI)
                        } else {
                            0.0
                        };
                        Complex::from(bare) - chi
                    })
                    .collect()
            })
            .collect();
        independent_inverse(&metric)
    }
}

fn mode_derivatives(
    fixture: &Reference<'_>,
    transfer: &Transfer,
    shift: usize,
    step: f64,
    stencil: stencils::Stencil,
) -> Vec<Vec<Complex>> {
    let n = transfer.waves.len();
    let imaginary_unit = Complex::new(0.0, 1.0);
    let mut derivative = vec![vec![Complex::ZERO; n]; n];
    for source in 0..n {
        let direction = |phase| {
            let plus = fixture.density(transfer, source, phase, step);
            let minus = fixture.density(transfer, source, phase, -step);
            let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                (
                    fixture.density(transfer, source, phase, 2.0 * step),
                    fixture.density(transfer, source, phase, -2.0 * step),
                )
            } else {
                (vec![Complex::ZERO; n], vec![Complex::ZERO; n])
            };
            (0..n)
                .map(|i| {
                    Complex::new(
                        stencil.first(plus[i].re, minus[i].re, plus2[i].re, minus2[i].re, step),
                        stencil.first(plus[i].im, minus[i].im, plus2[i].im, minus2[i].im, step),
                    )
                })
                .collect::<Vec<_>>()
        };
        let real_source = direction(Complex::ONE);
        if shift == 0 {
            let imaginary_source = direction(imaginary_unit);
            for i in 0..n {
                derivative[i][source] =
                    (real_source[i] - imaginary_unit * imaginary_source[i]) * 0.5;
            }
        } else {
            for i in 0..n {
                derivative[i][source] = real_source[i];
            }
        }
    }
    derivative
}

#[test]
fn asymmetric_lithium_hydride_all_pair_entries_and_reciprocal_density_components() {
    let cell = Cell::new(
        Molecule::new(vec![
            Atom {
                z: 3,
                position: Vec3::new(1.21, 0.8, 1.1),
            },
            Atom {
                z: 1,
                position: Vec3::new(3.68, 1.48, 2.0),
            },
        ]),
        Lattice::new(
            Vec3::new(6.5, 0.21, -0.13),
            Vec3::new(0.32, 7.2, 0.39),
            Vec3::new(0.27, -0.34, 7.8),
        ),
        &data::load("sto-3g").unwrap(),
        4.4,
    )
    .unwrap();
    let functional = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let temperature_kelvin = 40000.0;
    let settings = PeriodicScfOptions {
        k_mesh: [3, 1, 1],
        occupation: OccupationScheme::FermiDirac { temperature_kelvin },
        e_tol: 1e-12,
        p_tol: 1e-11,
        conv_tol_grad: 1e-10,
        max_iter: 250,
        grid_shell: 1,
        grid: GridOptions {
            radial_override: Some(12),
            angular_override: Some(9),
            prune: false,
            weight_cutoff: 0.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let reference = run_periodic_scf(&cell, &functional, &settings).unwrap();
    assert!(reference.converged);
    let bands: Vec<_> = reference.fock[0]
        .iter()
        .zip(&reference.overlap)
        .map(|(h, s)| generalized_orbitals(h, s, settings.orth_threshold).unwrap())
        .collect();
    let mut offsets = vec![0];
    let mut energies = Vec::new();
    for point in &bands {
        energies.extend_from_slice(&point.energies);
        offsets.push(energies.len());
    }
    let fixture = Reference {
        cell: &cell,
        points: &reference.k_points,
        bands,
        energies,
        offsets,
        volume: cell.lattice().volume(),
        kt: temperature_kelvin * BOLTZMANN_HARTREE_PER_K,
        mu: reference.chemical_potential[0].unwrap(),
    };
    let cutoff = 1.05;
    let screened = screened_pair_coulomb(
        &cell,
        &functional,
        &reference,
        &settings,
        &CoulombScreeningOptions {
            reciprocal_cutoff: cutoff,
            gamma_head: GammaHead::Thermal,
            tolerance: 1e-10,
        },
    )
    .unwrap();
    let neutral = screened_pair_coulomb(
        &cell,
        &functional,
        &reference,
        &settings,
        &CoulombScreeningOptions {
            reciprocal_cutoff: cutoff,
            gamma_head: GammaHead::Neutral,
            tolerance: 1e-10,
        },
    )
    .unwrap();
    assert!(neutral.gamma_head.is_none());
    let mut max_pair_defect = 0.0_f64;
    let mut max_neutral_defect = 0.0_f64;
    let mut max_inverse_defect = 0.0_f64;
    let mut failures = Vec::new();
    let stages = [
        ([0.006, 0.004, 0.002, 0.001], stencils::Stencil::Second),
        ([0.01, 0.007, 0.005, 0.003], stencils::Stencil::Second),
        ([0.022, 0.016, 0.011, 0.008], stencils::Stencil::Fourth),
    ];
    for shift in 0..3 {
        let transfer = fixture.transfer(shift, cutoff);
        let coefficients: Vec<_> = transfer
            .pairs
            .iter()
            .map(|&(a, b)| {
                divided_difference(
                    fixture.energies[a],
                    fixture.energies[b],
                    fixture.mu,
                    fixture.kt,
                )
                .unwrap()
                    * fixture.volume_weight()
            })
            .collect();
        let operator =
            static_screened_interaction(&transfer.waves, &transfer.forms, &coefficients, 1e-10)
                .unwrap();
        let independent = fixture.screened_oracle(&transfer);
        let n = transfer.waves.len();
        assert_eq!(screened.reciprocal_modes[shift], n);
        assert_eq!(neutral.reciprocal_modes[shift], n - usize::from(shift == 0));
        for (g, row) in independent.iter().enumerate() {
            for (h, &element) in row.iter().enumerate() {
                max_inverse_defect =
                    max_inverse_defect.max((at(&operator.interaction, g, h) - element).abs());
            }
        }
        for (transition, &(a, b)) in transfer.pairs.iter().enumerate() {
            let contract = |first_mode| {
                independent.iter().enumerate().skip(first_mode).fold(
                    Complex::ZERO,
                    |sum, (g, row)| {
                        sum + row.iter().enumerate().skip(first_mode).fold(
                            Complex::ZERO,
                            |inner, (h, &element)| {
                                inner
                                    + at(&transfer.forms, g, transition).conj()
                                        * element
                                        * at(&transfer.forms, h, transition)
                            },
                        )
                    },
                ) / fixture.volume
            };
            max_pair_defect =
                max_pair_defect.max((Complex::from(screened.kernel[(a, b)]) - contract(0)).abs());
            max_neutral_defect = max_neutral_defect.max(
                (Complex::from(neutral.kernel[(a, b)]) - contract(usize::from(shift == 0))).abs(),
            );
        }
        let components: Vec<_> = (0..n)
            .flat_map(|g| {
                (g..n).flat_map(move |h| {
                    [Some((g, h, false)), (g != h).then_some((g, h, true))]
                        .into_iter()
                        .flatten()
                })
            })
            .collect();
        let labels: Vec<_> = components
            .iter()
            .map(|&(g, h, imaginary)| {
                format!(
                    "LiH q={shift} chi0[{g},{h}] {}",
                    if imaginary { "imaginary" } else { "real" }
                )
            })
            .collect();
        stencils::fit_components_staged(
            &labels,
            &stages,
            |steps, stencil| {
                let samples =
                    steps.map(|h| mode_derivatives(&fixture, &transfer, shift, h, stencil));
                components
                    .iter()
                    .map(|&(g, h, imaginary)| {
                        std::array::from_fn(|step| {
                            let defect =
                                samples[step][g][h] - at(&operator.independent_response, g, h);
                            if imaginary {
                                defect.im.abs()
                            } else {
                                defect.re.abs()
                            }
                        })
                    })
                    .collect()
            },
            &mut failures,
        );
    }
    println!("LiH static screened Coulomb: E={:.15e} Eh, mu={:.15e} Eh, q modes={:?}, gamma head={:?} Eh, compressibility={:.6e} Bohr^-3/Eh, all ordered pair defect={max_pair_defect:.6e} Eh, neutral pair defect={max_neutral_defect:.6e} Eh, microscopic W defect={max_inverse_defect:.6e} Eh Bohr^3, reciprocity={:.6e} Eh, inverse={:?}, number={:.6e}", reference.energy, fixture.mu, screened.reciprocal_modes, screened.gamma_head, screened.gamma_compressibility, screened.pair_reciprocity_defect, screened.inverse_residuals, screened.electron_number_residual);
    assert!(max_pair_defect < 2e-10);
    assert!(max_neutral_defect < 2e-10);
    assert!(max_inverse_defect < 1e-8);
    assert!(screened.pair_reciprocity_defect < 1e-10);
    assert!(screened.electron_number_residual < 1e-10);
    components::assert_no_failures("microscopic static RPA susceptibility", &failures);
}

#[test]
#[ignore = "explicit elemental screening cutoff and k-mesh study"]
fn fcc_aluminum_thermal_rpa_screening_cutoff_and_mesh() {
    // Double one primitive vector of fcc Al to retain the true material
    // geometry and a neutral spin-degenerate even-electron reference.
    let a = 4.0495 * dft_rs::constants::ANGSTROM_TO_BOHR;
    let cell = Cell::new(
        Molecule::new(vec![
            Atom {
                z: 13,
                position: Vec3::zero(),
            },
            Atom {
                z: 13,
                position: Vec3::new(0.0, a / 2.0, a / 2.0),
            },
        ]),
        Lattice::new(
            Vec3::new(0.0, a, a),
            Vec3::new(a / 2.0, 0.0, a / 2.0),
            Vec3::new(a / 2.0, a / 2.0, 0.0),
        ),
        &data::load("sto-3g").unwrap(),
        6.0,
    )
    .unwrap();
    let functional = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let temperature_kelvin = 10000.0;
    for k_mesh in [[1, 1, 1], [2, 1, 1]] {
        let settings = PeriodicScfOptions {
            k_mesh,
            occupation: OccupationScheme::FermiDirac { temperature_kelvin },
            e_tol: 1e-10,
            p_tol: 1e-9,
            conv_tol_grad: 1e-8,
            max_iter: 250,
            grid_shell: 1,
            grid: GridOptions {
                radial_override: Some(12),
                angular_override: Some(9),
                prune: false,
                weight_cutoff: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let reference = run_periodic_scf(&cell, &functional, &settings).unwrap();
        assert!(reference.converged);
        let mut previous: Option<dft_rs::linalg::Matrix> = None;
        for cutoff in [1.0, 1.3, 1.6] {
            let screened = screened_pair_coulomb(
                &cell,
                &functional,
                &reference,
                &settings,
                &CoulombScreeningOptions {
                    reciprocal_cutoff: cutoff,
                    gamma_head: GammaHead::Thermal,
                    tolerance: 1e-9,
                },
            )
            .unwrap();
            let change = previous
                .as_ref()
                .map(|coarse| coarse.max_abs_diff(&screened.kernel));
            let head = screened.gamma_head.unwrap();
            println!("fcc Al screening a=4.0495 Angstrom, k={k_mesh:?}, T={temperature_kelvin} K, cutoff={cutoff} Bohr^-1, modes={:?}, E={:.16e} Hartree/cell, mu={:?} Hartree, SCF={:?} Hartree, mesh crossing={}, W00/V={head:.10e} Hartree, compressibility={:.10e} Bohr^-3/Hartree, K00={:.10e} Hartree, cutoff change={change:?} Hartree, charge residual {:.3e}, inverse residuals {:?}, reciprocity {:.3e} Hartree", screened.reciprocal_modes, reference.energy, reference.chemical_potential, reference.orbital_residual, screened.fermi_level_crossing, screened.gamma_compressibility, screened.kernel[(0,0)], screened.electron_number_residual, screened.inverse_residuals, screened.pair_reciprocity_defect);
            assert!(head.is_finite() && head > 0.0);
            assert!(screened.electron_number_residual < 1e-8);
            previous = Some(screened.kernel);
        }
    }
}
