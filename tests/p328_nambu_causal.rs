// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent off-resonant Floquet and damped Liouville Nambu response.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::dfpt::orbitals::matmul;
use dft_rs::linalg::{solve_linear_matrix, symmetric_eigen, Matrix};
use dft_rs::periodic::BlochMatrix;
use dft_rs::response::complex::Complex;
use dft_rs::response::nambu::{nambu_causal_response, NambuProbe, NambuReference};
use dft_rs::response::{Frequency, ResponseSolverConfig};
use dft_rs::DftError;

fn at(matrix: &BlochMatrix, i: usize, j: usize) -> Complex {
    Complex::new(matrix.re[(i, j)], matrix.im[(i, j)])
}

fn set(matrix: &mut BlochMatrix, i: usize, j: usize, value: Complex) {
    matrix.re[(i, j)] = value.re;
    matrix.im[(i, j)] = value.im;
}

fn hermitian(diagonal: [f64; 3], upper: [[f64; 2]; 3]) -> BlochMatrix {
    let mut r = BlochMatrix::zeros(3, 3);
    for (i, value) in diagonal.into_iter().enumerate() {
        r.re[(i, i)] = value;
    }
    for ((i, j), [re, im]) in [(0, 1), (0, 2), (1, 2)].into_iter().zip(upper) {
        let value = Complex::new(re, im);
        set(&mut r, i, j, value);
        set(&mut r, j, i, value.conj());
    }
    r
}

fn skew(upper: [[f64; 2]; 3]) -> BlochMatrix {
    let mut r = BlochMatrix::zeros(3, 3);
    for ((i, j), [re, im]) in [(0, 1), (0, 2), (1, 2)].into_iter().zip(upper) {
        let value = Complex::new(re, im);
        set(&mut r, i, j, value);
        set(&mut r, j, i, -value);
    }
    r
}

/// Explicit test-only BdG assembly, separate from the production builder.
fn bdg(normal: &BlochMatrix, pair: &BlochMatrix) -> BlochMatrix {
    let n = normal.rows();
    let mut h = BlochMatrix::zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            let normal = at(normal, i, j);
            let pair = at(pair, i, j);
            set(&mut h, i, j, normal);
            set(&mut h, n + i, n + j, -normal.conj());
            set(&mut h, i, n + j, pair);
            set(&mut h, n + j, i, pair.conj());
        }
    }
    h
}

fn spectral_matrix(vectors: &Matrix, weights: &[f64]) -> BlochMatrix {
    let n = vectors.rows / 2;
    let mut density = BlochMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut value = Complex::ZERO;
            for (c, &weight) in weights.iter().enumerate() {
                value +=
                    Complex::new(vectors[(i, c)], vectors[(n + i, c)]) * (weight * vectors[(j, c)]);
            }
            set(&mut density, i, j, value);
        }
    }
    density
}

fn twice_real_pencil(m: &BlochMatrix) -> (Vec<f64>, Matrix) {
    symmetric_eigen(&m.real_embedding()).unwrap()
}

struct Fixture {
    normal: BlochMatrix,
    pair: BlochMatrix,
    metric: BlochMatrix,
    probes: [NambuProbe; 3],
    interaction: Matrix,
    chemical_potential: f64,
    kt: f64,
    omega: f64,
}

impl Fixture {
    fn new() -> Self {
        Self {
            normal: hermitian(
                [-0.17, 0.061, 0.202],
                [[0.028, 0.023], [-0.012, 0.009], [0.034, -0.016]],
            ),
            pair: skew([[0.027, 0.008], [0.005, -0.019], [0.017, 0.031]]),
            metric: hermitian(
                [1.04, 1.18, 0.97],
                [[0.013, 0.021], [-0.033, 0.011], [0.029, -0.016]],
            ),
            probes: [
                NambuProbe {
                    normal: hermitian(
                        [0.31, -0.07, 0.11],
                        [[0.18, 0.27], [-0.12, 0.18], [-0.17, 0.05]],
                    ),
                    pairing: skew([[0.08, 0.21], [-0.12, -0.10], [0.22, 0.04]]),
                },
                NambuProbe {
                    normal: hermitian(
                        [0.029, 0.34, -0.24],
                        [[-0.17, 0.28], [0.12, -0.08], [0.30, 0.11]],
                    ),
                    pairing: skew([[0.25, -0.04], [0.19, 0.1], [0.02, -0.21]]),
                },
                NambuProbe {
                    normal: hermitian(
                        [0.14, -0.28, -0.09],
                        [[0.058, -0.19], [-0.029, 0.21], [0.15, 0.23]],
                    ),
                    pairing: skew([[0.18, 0.19], [0.04, 0.37], [-0.11, 0.16]]),
                },
            ],
            interaction: Matrix::from_row_major(
                3,
                3,
                vec![
                    0.018, 0.004, -0.002, 0.004, 0.012, 0.003, -0.002, 0.003, 0.021,
                ],
            ),
            chemical_potential: 0.018,
            kt: 0.083,
            omega: 0.047,
        }
    }

    fn reference(&self) -> NambuReference<'_> {
        NambuReference {
            normal: &self.normal,
            pairing: &self.pair,
            overlap: &self.metric,
            chemical_potential: self.chemical_potential,
            temperature_energy: self.kt,
            overlap_cutoff: 1e-12,
        }
    }

    fn normalized(&self) -> (BlochMatrix, Vec<BlochMatrix>) {
        let n = self.normal.rows();
        let mut metric = BlochMatrix::zeros(2 * n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                let value = at(&self.metric, i, j);
                set(&mut metric, i, j, value);
                set(&mut metric, n + i, n + j, value.conj());
            }
        }
        let (e, vectors) = twice_real_pencil(&metric);
        let inverse_roots: Vec<_> = e.iter().map(|x| x.sqrt().recip()).collect();
        let ortho = spectral_matrix(&vectors, &inverse_roots);
        let congruence =
            |operator: &BlochMatrix| matmul(&matmul(&ortho, operator).unwrap(), &ortho).unwrap();
        let mut shifted = self.normal.clone();
        shifted.add_scaled(&self.metric, -self.chemical_potential);
        let h = congruence(&bdg(&shifted, &self.pair));
        let q = self
            .probes
            .iter()
            .map(|probe| congruence(&bdg(&probe.normal, &probe.pairing)))
            .collect();
        (h, q)
    }
}

fn fermi(energy: f64, kt: f64) -> f64 {
    1.0 / (1.0 + (energy / kt).exp())
}

struct Floquet<'a> {
    system: &'a BlochMatrix,
    probes: &'a [BlochMatrix],
    omega: f64,
    levels: Vec<f64>,
    fermi_populations: Vec<f64>,
    min_gap: f64,
}

impl<'a> Floquet<'a> {
    fn new(system: &'a BlochMatrix, probes: &'a [BlochMatrix], kt: f64, omega: f64) -> Self {
        let normal = twice_real_pencil(system).0;
        let mut levels: Vec<_> = normal
            .chunks_exact(2)
            .flat_map(|pair| {
                let e = pair[0].midpoint(pair[1]);
                [(e - omega, fermi(e, kt)), (e, fermi(e, kt))]
            })
            .collect();
        levels.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (levels, fermi_populations): (Vec<_>, Vec<_>) = levels.into_iter().unzip();
        let min_gap = levels
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .fold(f64::INFINITY, f64::min);
        assert!(min_gap > 1e-4, "resolved Floquet copy spacing {min_gap}");
        Self {
            system,
            probes,
            omega,
            levels,
            fermi_populations,
            min_gap,
        }
    }

    fn expectations(&self, source: &[Complex]) -> Vec<Complex> {
        let n = self.system.rows();
        let mut f = BlochMatrix::zeros(2 * n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                let h = at(self.system, i, j);
                set(
                    &mut f,
                    i,
                    j,
                    h - if i == j {
                        Complex::from(self.omega)
                    } else {
                        Complex::ZERO
                    },
                );
                set(&mut f, n + i, n + j, h);
                let drive = self
                    .probes
                    .iter()
                    .zip(source)
                    .fold(Complex::ZERO, |w, (q, &s)| w + at(q, i, j) * s);
                set(&mut f, i, n + j, drive);
                set(&mut f, n + j, i, drive.conj());
            }
        }
        let (e, coefficients) = twice_real_pencil(&f);
        let occupations: Vec<_> = e
            .iter()
            .enumerate()
            .map(|(i, &value)| {
                assert!(
                    (value - self.levels[i / 2]).abs() < 0.45 * self.min_gap,
                    "finite Floquet perturbation changed the adiabatic level ordering"
                );
                self.fermi_populations[i / 2]
            })
            .collect();
        let density = spectral_matrix(&coefficients, &occupations);
        self.probes
            .iter()
            .map(|probe| {
                (0..n).fold(Complex::ZERO, |sum, i| {
                    sum + (0..n).fold(Complex::ZERO, |line, j| {
                        line + at(probe, j, i) * at(&density, i, n + j)
                    })
                }) * 0.5
            })
            .collect()
    }

    fn vary(&self, driven: usize, step: f64, interaction: Option<&Matrix>) -> Vec<Complex> {
        let p = self.probes.len();
        let mut external = vec![Complex::ZERO; p];
        external[driven] = Complex::from(step);
        let mut expectation = self.expectations(&external);
        if let Some(u) = interaction {
            for _ in 0..100 {
                let fields: Vec<_> = (0..p)
                    .map(|i| (0..p).fold(external[i], |sum, j| sum + expectation[j] * u[(i, j)]))
                    .collect();
                let next = self.expectations(&fields);
                let defect = next
                    .iter()
                    .zip(&expectation)
                    .fold(0.0_f64, |m, (a, b)| m.max((*a - *b).abs()));
                expectation = next;
                if defect < 1e-15 {
                    return expectation;
                }
            }
            panic!("independent self-consistent Floquet field did not converge");
        }
        expectation
    }
}

fn cfg() -> ResponseSolverConfig {
    ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    }
}

#[test]
fn all_off_resonant_components_match_four_step_floquet_derivatives() {
    let f = Fixture::new();
    let (system, probes) = f.normalized();
    let floquet = Floquet::new(&system, &probes, f.kt, f.omega);
    let frequency = [Frequency::Retarded {
        omega: f.omega,
        damping: 0.0,
    }];
    let result = nambu_causal_response(
        f.reference(),
        &f.probes,
        &frequency,
        Some(&f.interaction),
        &cfg(),
    )
    .unwrap();
    let mut failures = Vec::new();
    // The unperturbed Floquet copies approach within 0.001106 Hartree.
    // Use a common resolved interval for all bare and induced components.
    let stages = [
        ([0.0004, 0.0003, 0.0002, 0.0001], stencils::Stencil::Second),
        (
            [0.0003, 0.0002, 0.00015, 0.00008],
            stencils::Stencil::Second,
        ),
        ([0.0004, 0.0003, 0.0002, 0.0001], stencils::Stencil::Fourth),
    ];
    for (model, analytic, coupling) in [
        ("bare", &result.bare[0], None),
        ("induced", &result.response[0], Some(&f.interaction)),
    ] {
        let entries: Vec<_> = (0..probes.len())
            .flat_map(|i| {
                (i..probes.len()).flat_map(move |j| {
                    [Some((i, j, false)), (i != j).then_some((i, j, true))]
                        .into_iter()
                        .flatten()
                })
            })
            .collect();
        let labels: Vec<_> = entries
            .iter()
            .map(|&(i, j, imag)| {
                format!(
                    "{model} Floquet chi[{i},{j}] {}",
                    if imag { "imaginary" } else { "real" }
                )
            })
            .collect();
        stencils::fit_components_staged(
            &labels,
            &stages,
            |steps, stencil| {
                let p = probes.len();
                let samples = steps.map(|h| {
                    (0..p)
                        .map(|driven| {
                            let plus = floquet.vary(driven, h, coupling);
                            let minus = floquet.vary(driven, -h, coupling);
                            let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                                (
                                    floquet.vary(driven, 2.0 * h, coupling),
                                    floquet.vary(driven, -2.0 * h, coupling),
                                )
                            } else {
                                (vec![Complex::ZERO; p], vec![Complex::ZERO; p])
                            };
                            (0..p)
                                .map(|observed| {
                                    Complex::new(
                                        stencil.first(
                                            plus[observed].re,
                                            minus[observed].re,
                                            plus2[observed].re,
                                            minus2[observed].re,
                                            h,
                                        ),
                                        stencil.first(
                                            plus[observed].im,
                                            minus[observed].im,
                                            plus2[observed].im,
                                            minus2[observed].im,
                                            h,
                                        ),
                                    )
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                });
                entries
                    .iter()
                    .map(|&(i, j, imaginary)| {
                        std::array::from_fn(|h| {
                            let defect = samples[h][j][i] - at(analytic, i, j);
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
        assert!(analytic.hermiticity_error() < 1e-12);
    }
    println!("Nambu Floquet omega={:.9e} Hartree, minimum unperturbed copy spacing={:.9e} Hartree, chi00={:.14e} Hartree^-1, eigen/metric defects {:.3e}/{:.3e}, bare residual {:?}, coupled residuals {:?}, stability margin {:.12e}", f.omega, floquet.min_gap, result.response[0].re[(0,0)], result.reference_eigen_residual, result.reference_metric_defect, result.bare_residuals, result.coupled_residuals, result.stability_margin);
    components::assert_no_failures("collisionless Nambu Floquet response", &failures);
}

fn commutator(a: &BlochMatrix, b: &BlochMatrix) -> BlochMatrix {
    let mut commutator = matmul(a, b).unwrap();
    commutator.add_scaled(&matmul(b, a).unwrap(), -1.0);
    commutator
}

#[test]
fn collisionless_static_preserves_small_gap_and_conserved_number() {
    let gap = 5e-10;
    let kt = 0.038;
    let mut normal = BlochMatrix::zeros(2, 2);
    normal.re[(0, 0)] = 0.02;
    normal.re[(1, 1)] = 0.02 + gap;
    let pairs = BlochMatrix::zeros(2, 2);
    let overlap = BlochMatrix {
        re: Matrix::identity(2),
        im: Matrix::zeros(2, 2),
    };
    let mut transition = BlochMatrix::zeros(2, 2);
    transition.re[(0, 1)] = 0.4;
    transition.re[(1, 0)] = 0.4;
    let probes = [
        NambuProbe {
            normal: transition,
            pairing: pairs.clone(),
        },
        NambuProbe {
            normal: overlap.clone(),
            pairing: pairs.clone(),
        },
    ];
    let config = ResponseSolverConfig {
        min_denominator: 1e-6,
        ..cfg()
    };
    let solve = |normal: &BlochMatrix| {
        nambu_causal_response(
            NambuReference {
                normal,
                pairing: &pairs,
                overlap: &overlap,
                chemical_potential: 0.0,
                temperature_energy: kt,
                overlap_cutoff: 1e-12,
            },
            &probes,
            &[
                Frequency::Static,
                Frequency::Retarded {
                    omega: 0.031,
                    damping: 0.006,
                },
            ],
            None,
            &config,
        )
        .unwrap()
    };
    let result = solve(&normal);
    let f = fermi(0.02 + 0.5 * gap, kt);
    let expected = -2.0 * 0.4_f64.powi(2) * f * (1.0 - f) / kt;
    assert!((result.response[0].re[(0, 0)] - expected).abs() < 1e-12);
    for response in &result.response {
        for row in 0..2 {
            assert_eq!(at(response, 1, row), Complex::ZERO);
            assert_eq!(at(response, row, 1), Complex::ZERO);
        }
    }
    normal.re[(1, 1)] = normal.re[(0, 0)];
    assert_eq!(at(&solve(&normal).response[0], 0, 0), Complex::ZERO);
    println!("Nambu resolved small-gap static chi={expected:.14e} Hartree^-1; conserved-number response is zero");
}

#[test]
fn molecular_lithium_hydride_normal_limit_matches_spin_resolved_dipole_response() {
    use dft_rs::basis::{data, MolBasis};
    use dft_rs::driver::method::scf_options;
    use dft_rs::grid::{GridOptions, MolecularGrid};
    use dft_rs::integrals::dipole_matrices;
    use dft_rs::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
    use dft_rs::scf::run_scf;
    use dft_rs::xc::{resolve, FunctionalSpec};
    use dft_rs::{Atom, Molecule, Vec3};

    let origin = Vec3::zero();
    let molecule = Molecule::new(vec![
        Atom {
            z: 3,
            position: origin,
        },
        Atom {
            z: 1,
            position: Vec3::new(0.7, -0.5, 2.9),
        },
    ]);
    let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
    let grid = MolecularGrid::build(
        &molecule,
        &GridOptions {
            level: 3,
            weight_cutoff: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    let dipoles = dipole_matrices(&basis, origin);
    let (omega, eta) = (0.037, 0.004);
    let frequency = Frequency::Retarded {
        omega,
        damping: eta,
    };
    let z = frequency.energy().unwrap();
    for name in ["wb97x-d", "wb97x-v"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let (mut options, _) =
            scf_options(name, "auto", None, 250, 1e-12, Some(1e-10), 1e-11, 128, 2).unwrap();
        options.electronic_temperature = 1800.0 * BOLTZMANN_HARTREE_PER_K;
        options.vv10_grid_weight_cutoff = 0.0;
        let scf = run_scf(&molecule, &basis, &options, &functional, Some(&grid)).unwrap();
        assert!(scf.converged && !scf.unrestricted);
        assert!(
            (-8.5..-7.0).contains(&scf.energy),
            "LiH {name} SCF free energy {}",
            scf.energy
        );
        let chemical_potential = scf.chemical_potentials[0].unwrap();
        assert_eq!(scf.chemical_potentials[1], Some(chemical_potential));
        let n = scf.mo_coeff[0].cols;
        let pair = BlochMatrix::zeros(2 * n, 2 * n);
        let metric = BlochMatrix {
            re: Matrix::identity(2 * n),
            im: Matrix::zeros(2 * n, 2 * n),
        };
        let mut normal = pair.clone();
        for (i, energy) in scf.mo_energy.iter().flatten().enumerate() {
            normal.re[(i, i)] = *energy;
        }
        // Use dipole coordinates divided by one Bohr as probe operators.
        let dipole_mo = scf.mo_coeff.each_ref().map(|c| {
            dipoles
                .each_ref()
                .map(|dipole| c.transpose_matmul(&dipole.matmul_seq(c)))
        });
        let probes = std::array::from_fn::<_, 3, _>(|axis| {
            let mut operator = pair.clone();
            for (spin, dipole) in dipole_mo.iter().enumerate() {
                for i in 0..n {
                    for j in 0..n {
                        operator.re[(spin * n + i, spin * n + j)] = dipole[axis][(i, j)];
                    }
                }
            }
            NambuProbe {
                normal: operator,
                pairing: pair.clone(),
            }
        });
        let reference = NambuReference {
            normal: &normal,
            pairing: &pair,
            overlap: &metric,
            chemical_potential,
            temperature_energy: scf.electronic_temperature,
            overlap_cutoff: 1e-12,
        };
        let result = nambu_causal_response(reference, &probes, &[frequency], None, &cfg()).unwrap();
        let mut maximum_defect = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                let mut expected = Complex::ZERO;
                for (spin, dipole) in dipole_mo.iter().enumerate() {
                    for i in 0..n {
                        for j in 0..n {
                            let population = scf.occupations[spin][j] - scf.occupations[spin][i];
                            let gap = scf.mo_energy[spin][i] - scf.mo_energy[spin][j];
                            expected += (Complex::from(population) / (z - Complex::from(gap)))
                                * (dipole[a][(j, i)] * dipole[b][(i, j)]);
                        }
                    }
                }
                maximum_defect =
                    maximum_defect.max((at(&result.response[0], a, b) - expected).abs());
            }
        }
        assert!(
            maximum_defect < 2e-11,
            "LiH {name} dipole defect={maximum_defect:.3e}"
        );
        let population_defect = (scf.occupations.iter().flatten().sum::<f64>() - 4.0).abs();
        let alpha_zz = -at(&result.response[0], 2, 2);
        println!("LiH {name}/STO-3G no-pairing dipole polarizability omega={omega} eta={eta} Hartree: alpha_zz={:.12e}+i{:.12e} Bohr^2/Hartree, SCF free energy={:.14e}, mu={chemical_potential:.14e}, nonlocal energy={:.14e} Hartree, SCF iterations={}, number defect {population_defect:.3e}, all-component defect {maximum_defect:.3e}, spectral residual {:?}", alpha_zz.re, alpha_zz.im, scf.energy, scf.e_nlc, scf.iterations, result.bare_residuals);
    }
}

fn dense_liouville(
    h: &BlochMatrix,
    probes: &[BlochMatrix],
    kt: f64,
    omega: f64,
    eta: f64,
    interaction: Option<&Matrix>,
) -> BlochMatrix {
    let (e, orbitals) = twice_real_pencil(h);
    let occupations: Vec<_> = e.iter().map(|&e| fermi(e, kt)).collect();
    let p = probes.len();
    let n = h.rows();
    let dim = n * n;
    let ground = spectral_matrix(&orbitals, &occupations);
    let sources: Vec<_> = probes.iter().map(|q| commutator(q, &ground)).collect();
    let z = Complex::new(omega, eta);
    let mut op = BlochMatrix::zeros(dim, dim);
    for k in 0..n {
        for l in 0..n {
            let mut elementary = BlochMatrix::zeros(n, n);
            elementary.re[(k, l)] = 1.0;
            let action = commutator(h, &elementary);
            for i in 0..n {
                for j in 0..n {
                    let mut value = -at(&action, i, j);
                    if i == k && j == l {
                        value += z;
                    }
                    if let Some(u) = interaction {
                        for (a, source) in sources.iter().enumerate() {
                            for (b, probe) in probes.iter().enumerate() {
                                value -= at(source, i, j) * at(probe, l, k) * (0.5 * u[(a, b)]);
                            }
                        }
                    }
                    set(&mut op, i * n + j, k * n + l, value);
                }
            }
        }
    }
    let mut rhs = Matrix::zeros(2 * dim, p);
    for (b, source) in sources.iter().enumerate() {
        for i in 0..n {
            for j in 0..n {
                let value = at(source, i, j);
                rhs[(i * n + j, b)] = value.re;
                rhs[(dim + i * n + j, b)] = value.im;
            }
        }
    }
    let density = solve_linear_matrix(&op.real_embedding(), &rhs).unwrap();
    let mut sus = BlochMatrix::zeros(p, p);
    for (a, probe) in probes.iter().enumerate() {
        for b in 0..p {
            let mut value = Complex::ZERO;
            for i in 0..n {
                for j in 0..n {
                    value += at(probe, j, i)
                        * Complex::new(density[(i * n + j, b)], density[(dim + i * n + j, b)])
                        * 0.5;
                }
            }
            set(&mut sus, a, b, value);
        }
    }
    sus
}

#[test]
fn positive_damping_and_negative_frequencies_match_full_liouville_and_error_limits() {
    let f = Fixture::new();
    let (system, probes) = f.normalized();
    let eta = 0.009;
    let points = [f.omega, -f.omega, 0.0].map(|omega| Frequency::Retarded {
        omega,
        damping: eta,
    });
    let result = nambu_causal_response(
        f.reference(),
        &f.probes,
        &points,
        Some(&f.interaction),
        &cfg(),
    )
    .unwrap();
    let mut max_defect = 0.0_f64;
    let mut max_frequency_defect = 0.0_f64;
    for (reports, interaction) in [
        (&result.bare, None),
        (&result.response, Some(&f.interaction)),
    ] {
        for (&omega, report) in [f.omega, -f.omega, 0.0].iter().zip(reports) {
            let expected = dense_liouville(&system, &probes, f.kt, omega, eta, interaction);
            for i in 0..probes.len() {
                for j in 0..probes.len() {
                    max_defect = max_defect.max((at(report, i, j) - at(&expected, i, j)).abs());
                }
            }
        }
        for i in 0..probes.len() {
            assert!(reports[0].im[(i, i)] < 0.0);
            for j in 0..probes.len() {
                max_frequency_defect = max_frequency_defect
                    .max((at(&reports[0], i, j).conj() - at(&reports[1], i, j)).abs());
            }
        }
    }
    assert!(
        max_defect < 1e-11,
        "full density-space response defect={max_defect:.3e}"
    );
    assert!(max_frequency_defect < 1e-12);
    // Rotating every pairing field by the same phase is a similarity
    // transformation of the Nambu reference and each driven operator.
    let phase = Complex::new(0.618_f64.cos(), 0.618_f64.sin());
    let rotate = |input: &BlochMatrix| {
        let mut rotated = BlochMatrix::zeros(input.rows(), input.cols());
        for i in 0..input.rows() {
            for j in 0..input.cols() {
                set(&mut rotated, i, j, phase * at(input, i, j));
            }
        }
        rotated
    };
    let pairing = rotate(&f.pair);
    let rotated_probes = f.probes.each_ref().map(|probe| NambuProbe {
        normal: probe.normal.clone(),
        pairing: rotate(&probe.pairing),
    });
    let rotated = nambu_causal_response(
        NambuReference {
            pairing: &pairing,
            ..f.reference()
        },
        &rotated_probes,
        &points,
        Some(&f.interaction),
        &cfg(),
    )
    .unwrap();
    let mut gauge_defect = 0.0_f64;
    for (a, b) in result.response.iter().zip(&rotated.response) {
        for i in 0..probes.len() {
            for j in 0..probes.len() {
                gauge_defect = gauge_defect.max((at(a, i, j) - at(b, i, j)).abs());
            }
        }
    }
    assert!(gauge_defect < 1e-12);
    println!("Nambu shared pairing phase rotation defect={gauge_defect:.4e} Hartree^-1");
    println!("Nambu full Liouville damped chi00={:.14e}+i{:.14e} Hartree^-1, dense defect={max_defect:.4e}, signed-frequency defect={max_frequency_defect:.4e}, bare residuals={:?}, coupled residuals={:?}", result.response[0].re[(0,0)],result.response[0].im[(0,0)],result.bare_residuals,result.coupled_residuals);
    let stable = [Frequency::Retarded {
        omega: f.omega,
        damping: eta,
    }];
    let mut unstable = Matrix::identity(probes.len());
    for item in unstable.as_mut_slice() {
        *item *= -100.0;
    }
    assert!(
        matches!(nambu_causal_response(f.reference(),&f.probes,&stable,Some(&unstable),&cfg()),
        Err(DftError::InvalidInput(message)) if message.contains("unstable collisionless Nambu reference"))
    );
    let last = result.modes.len() - 1;
    let pole = result.modes[last] - result.modes[last - 1];
    let exact = [Frequency::Retarded {
        omega: pole,
        damping: 0.0,
    }];
    assert!(
        matches!(nambu_causal_response(f.reference(),&f.probes,&exact,None,&cfg()),
        Err(DftError::LinearAlgebra(message)) if message.contains("unresolved collisionless Nambu pole"))
    );
}
