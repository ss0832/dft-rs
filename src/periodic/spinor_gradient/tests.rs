// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed-density operator tests; these are not converged GKS total forces.
use super::*;
use crate::basis::{g94, MolBasis};
use crate::linalg::Matrix;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::features::PeriodicBoundary;
use crate::periodic::integrals::{bloch, dimensional_image_blocks, Operator};
use crate::periodic::spinor_scf::{pauli_components, spin_diagonal};
use crate::periodic::Lattice;
use crate::response::complex::Complex;
use crate::system::{Atom, Molecule};

const CUTOFF: f64 = 4.0;
const MESH: [usize; 3] = [3, 1, 1];

fn geometry() -> (Molecule, PeriodicGeometry) {
    let molecule = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.13, -0.21, 0.07),
        },
        Atom {
            z: 2,
            position: Vec3::new(1.31, 0.36, -0.41),
        },
    ]);
    let lattice = Lattice::new(
        Vec3::new(4.4, 0., 0.),
        Vec3::new(0.2, 4.8, 0.),
        Vec3::new(0.3, 0.1, 5.2),
    );
    (
        molecule,
        PeriodicGeometry::new(lattice, PeriodicBoundary::FULL_3D).unwrap(),
    )
}

fn basis(molecule: &Molecule) -> MolBasis {
    let set = g94::parse(
        "operator-one-primitive",
        "H 0\nS 1 1.0\n0.73 1.0\n****\nHe 0\nS 1 1.0\n0.73 1.0\n****\n",
    )
    .unwrap();
    MolBasis::build(molecule, &set).unwrap()
}

fn at(matrix: &BlochMatrix, i: usize, j: usize) -> Complex {
    Complex::new(matrix.re[(i, j)], matrix.im[(i, j)])
}

fn metric_dot(a: &[Complex], s: &BlochMatrix, b: &[Complex]) -> Complex {
    let mut value = Complex::ZERO;
    for i in 0..a.len() {
        for j in 0..a.len() {
            value += a[i].conj() * at(s, i, j) * b[j];
        }
    }
    value
}

fn trial_density(mb: &MolBasis, geometry: &PeriodicGeometry) -> Vec<[BlochMatrix; 4]> {
    let overlap = dimensional_image_blocks(mb, geometry, 2.0 * CUTOFF, Operator::Overlap).unwrap();
    geometry
        .k_points(MESH)
        .unwrap()
        .iter()
        .enumerate()
        .map(|(ik, &k)| {
            let metric = spin_diagonal(&BlochMatrix::from_pair(bloch(&overlap, k)));
            let mut occupied: Vec<Vec<Complex>> = Vec::new();
            for orbital in 0..2 {
                let mut v: Vec<_> = (0..4)
                    .map(|i| {
                        let a = (i + 1) as f64;
                        let b = (orbital + 1) as f64;
                        Complex::new(
                            (a * b * 0.63 + ik as f64 * 0.17).sin(),
                            (a * (b + 0.7) * 0.41 - ik as f64 * 0.13).cos(),
                        )
                    })
                    .collect();
                for old in &occupied {
                    let projection = metric_dot(old, &metric, &v);
                    for i in 0..4 {
                        v[i] -= old[i] * projection;
                    }
                }
                let norm = metric_dot(&v, &metric, &v);
                assert!(norm.re > 1e-8 && norm.im.abs() < 1e-12);
                for value in &mut v {
                    *value = Complex::new(value.re / norm.re.sqrt(), value.im / norm.re.sqrt());
                }
                occupied.push(v);
            }
            let mut density = BlochMatrix::zeros(4, 4);
            for i in 0..4 {
                for j in 0..4 {
                    let value = occupied
                        .iter()
                        .fold(Complex::ZERO, |sum, v| sum + v[i] * v[j].conj());
                    density.re[(i, j)] = value.re;
                    density.im[(i, j)] = value.im;
                }
            }
            assert!((density.trace_product(&metric) - 2.0).abs() < 1e-12);
            pauli_components(&density).unwrap()
        })
        .collect()
}

fn images(
    geometry: &PeriodicGeometry,
    density: &[BlochMatrix],
) -> (Vec<ImageBlock>, Vec<ImageBlock>) {
    let points = geometry.k_points(MESH).unwrap();
    let mut real = Vec::new();
    let mut imaginary = Vec::new();
    for translation in geometry.translations_within(2.0 * CUTOFF).unwrap() {
        let mut re = Matrix::zeros(2, 2);
        let mut im = Matrix::zeros(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                for (&k, p) in points.iter().zip(density) {
                    let phase = k.dot(translation);
                    re[(i, j)] += (phase.cos() * p.re[(i, j)] + phase.sin() * p.im[(i, j)])
                        / points.len() as f64;
                    im[(i, j)] += (phase.cos() * p.im[(i, j)] - phase.sin() * p.re[(i, j)])
                        / points.len() as f64;
                }
            }
        }
        real.push(ImageBlock {
            translation,
            matrix: re,
        });
        imaginary.push(ImageBlock {
            translation,
            matrix: im,
        });
    }
    (real, imaginary)
}

fn complex_bloch(real: &[ImageBlock], imaginary: &[ImageBlock], k: Vec3) -> BlochMatrix {
    let (mut re, mut im) = bloch(real, k);
    let (a, b) = bloch(imaginary, k);
    for (r, x) in re.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *r -= x;
    }
    for (r, x) in im.as_mut_slice().iter_mut().zip(a.as_slice()) {
        *r += x;
    }
    BlochMatrix { re, im }
}

fn pauli_block(density: &[[BlochMatrix; 4]], component: usize) -> Vec<BlochMatrix> {
    density.iter().map(|p| p[component].clone()).collect()
}

fn component_energy(
    mb: &MolBasis,
    geometry: &PeriodicGeometry,
    density: &[BlochMatrix],
    kernel: CoulombKernel,
    only_imaginary: bool,
) -> f64 {
    let (real, imaginary) = images(geometry, density);
    let re = if only_imaginary {
        vec![ImageBlock {
            translation: Vec3::zero(),
            matrix: Matrix::zeros(2, 2),
        }]
    } else {
        crate::periodic::exchange::dimensional_exchange_component_images(
            mb, geometry, &real, CUTOFF, kernel, false,
        )
        .unwrap()
    };
    let im = crate::periodic::exchange::dimensional_exchange_component_images(
        mb, geometry, &imaginary, CUTOFF, kernel, true,
    )
    .unwrap();
    let points = geometry.k_points(MESH).unwrap();
    -0.25
        * points
            .iter()
            .zip(density)
            .map(|(&k, p)| p.trace_product(&complex_bloch(&re, &im, k)))
            .sum::<f64>()
        / points.len() as f64
}

fn energy(
    molecule: &Molecule,
    geometry: &PeriodicGeometry,
    density: &[[BlochMatrix; 4]],
    kernel: CoulombKernel,
    only_imaginary: bool,
) -> f64 {
    let mb = basis(molecule);
    (0..4)
        .map(|a| {
            component_energy(
                &mb,
                geometry,
                &pauli_block(density, a),
                kernel,
                only_imaginary,
            )
        })
        .sum()
}

fn analytic(
    molecule: &Molecule,
    geometry: &PeriodicGeometry,
    density: &[[BlochMatrix; 4]],
    kernel: CoulombKernel,
    only_imaginary: bool,
    imaginary_scale: f64,
) -> ExchangeNuclearDerivative {
    let mb = basis(molecule);
    let mut sum = ExchangeNuclearDerivative {
        gradient: vec![Vec3::zero(); 2],
        strain_derivative: [[0.; 3]; 3],
    };
    for component in 0..4 {
        let (mut real, mut imaginary) = images(geometry, &pauli_block(density, component));
        if only_imaginary {
            for block in &mut real {
                block.matrix.as_mut_slice().fill(0.);
            }
        }
        for block in &mut imaginary {
            for value in block.matrix.as_mut_slice() {
                *value *= imaginary_scale;
            }
        }
        let value = complex_exchange_nuclear_derivative(
            &mb,
            geometry,
            2,
            ComplexImageDensity {
                real: &real,
                imaginary: &imaginary,
            },
            CUTOFF,
            kernel,
            -0.25,
        )
        .unwrap();
        for (a, b) in sum.gradient.iter_mut().zip(value.gradient) {
            *a += b;
        }
        for (a, b) in sum
            .strain_derivative
            .iter_mut()
            .flatten()
            .zip(value.strain_derivative.iter().flatten())
        {
            *a += *b;
        }
    }
    sum
}

fn flatten(value: &ExchangeNuclearDerivative) -> Vec<f64> {
    value
        .gradient
        .iter()
        .flat_map(|g| [g.x, g.y, g.z])
        .chain(value.strain_derivative.iter().flatten().copied())
        .collect()
}

fn displaced(
    molecule: &Molecule,
    geometry: &PeriodicGeometry,
    component: usize,
    h: f64,
) -> (Molecule, PeriodicGeometry) {
    let mut molecule = molecule.clone();
    let mut lattice = geometry.lattice().clone();
    if component < 6 {
        let p = &mut molecule.atoms[component / 3].position;
        let value = p.get(component % 3) + h;
        match component % 3 {
            0 => p.x = value,
            1 => p.y = value,
            _ => p.z = value,
        }
    } else {
        let a = (component - 6) / 3;
        let b = (component - 6) % 3;
        for position in molecule
            .atoms
            .iter_mut()
            .map(|a| &mut a.position)
            .chain(lattice.a.iter_mut())
        {
            let value = position.get(a) + h * position.get(b);
            match a {
                0 => position.x = value,
                1 => position.y = value,
                _ => position.z = value,
            }
        }
    }
    (
        molecule,
        PeriodicGeometry::new(lattice, geometry.boundary()).unwrap(),
    )
}

#[test]
fn gks_complex_exchange_sixty_fixed_four_step_fits() {
    let (molecule, geometry) = geometry();
    let density = trial_density(&basis(&molecule), &geometry);
    let mut failures = Vec::new();
    let mut attempted = 0;
    for (label, kernel) in [
        ("full", CoulombKernel::Coulomb),
        ("erf", CoulombKernel::Erf { omega: 0.37 }),
    ] {
        for only_imaginary in [false, true] {
            let domain = if only_imaginary {
                "imaginary"
            } else {
                "all-Pauli"
            };
            let derivative = analytic(&molecule, &geometry, &density, kernel, only_imaginary, 1.);
            let expected = flatten(&derivative);
            let names: Vec<_> = (0..15)
                .map(|i| format!("GKS exchange {label}/{domain} component[{i}]"))
                .collect();
            super::stencils::fit_components_staged(
                &names,
                &[([0.04, 0.02, 0.01, 0.005], super::stencils::Stencil::Second)],
                |steps, stencil| {
                    (0..15).map(|component| std::array::from_fn(|j| {
                    let h = steps[j];
                    let (plus, plus_geometry) = displaced(&molecule, &geometry, component, h);
                    let (minus, minus_geometry) = displaced(&molecule, &geometry, component, -h);
                    let ep = energy(&plus, &plus_geometry, &density, kernel, only_imaginary);
                    let em = energy(&minus, &minus_geometry, &density, kernel, only_imaginary);
                    let reference = stencil.first(ep, em, 0., 0., h);
                    let error = (reference - expected[component]).abs();
                    println!("GKS_EXCHANGE_SIGNED kernel={label} domain={domain} component={component} h={h:.17e} analytic={:.17e} reference={reference:.17e} error={error:.17e}", expected[component]);
                    error
                })).collect()
                },
                &mut failures,
            );
            attempted += 15;
        }
    }
    assert_eq!(attempted, 60);
    super::components::assert_no_failures(
        "complex exchange fixed-density gradient/strain",
        &failures,
    );
}

#[test]
fn gks_exchange_spin_rotation_tiny_current_and_zero_range() {
    let (molecule, geometry) = geometry();
    let density = trial_density(&basis(&molecule), &geometry);
    let mut rotated = density.clone();
    let (sine, cosine) = 0.63_f64.sin_cos();
    for (out, p) in rotated.iter_mut().zip(&density) {
        out[1] = p[1].clone();
        out[1].scale(cosine);
        out[1].add_scaled(&p[3], sine);
        out[3] = p[3].clone();
        out[3].scale(cosine);
        out[3].add_scaled(&p[1], -sine);
    }
    for kernel in [CoulombKernel::Coulomb, CoulombKernel::Erf { omega: 0.37 }] {
        let base = analytic(&molecule, &geometry, &density, kernel, false, 1.);
        let changed = analytic(&molecule, &geometry, &rotated, kernel, false, 1.);
        for (a, b) in flatten(&base).into_iter().zip(flatten(&changed)) {
            assert!((a - b).abs() < 1e-12);
        }
        let normal = analytic(&molecule, &geometry, &density, kernel, true, 1.);
        let tiny = analytic(&molecule, &geometry, &density, kernel, true, 1e-10);
        assert!(flatten(&normal).iter().any(|v| v.abs() > 1e-8));
        for (a, b) in flatten(&normal).into_iter().zip(flatten(&tiny)) {
            assert!((a - b / 1e-20).abs() < 1e-12 * (1. + a.abs()));
        }
        let translation = base.gradient.iter().fold(Vec3::zero(), |a, b| a + *b);
        assert!(translation.norm() < 1e-12);
    }
    let zero = analytic(
        &molecule,
        &geometry,
        &density,
        CoulombKernel::Erf { omega: 0. },
        false,
        1.,
    );
    assert!(flatten(&zero).iter().all(|v| *v == 0.));
}

#[test]
fn gks_exchange_input_guards() {
    let (molecule, geometry) = geometry();
    let mb = basis(&molecule);
    let density = trial_density(&mb, &geometry);
    let (real, imaginary) = images(&geometry, &pauli_block(&density, 0));
    let mut malformed = imaginary.clone();
    malformed[0].matrix[(0, 0)] = 0.1;
    assert!(matches!(
        complex_exchange_nuclear_derivative(
            &mb,
            &geometry,
            2,
            ComplexImageDensity {
                real: &real,
                imaginary: &malformed
            },
            CUTOFF,
            CoulombKernel::Coulomb,
            -0.25
        ),
        Err(DftError::InvalidInput(_))
    ));
    assert!(matches!(
        complex_exchange_nuclear_derivative(
            &mb,
            &geometry,
            2,
            ComplexImageDensity {
                real: &real,
                imaginary: &imaginary
            },
            CUTOFF,
            CoulombKernel::Erf { omega: -1. },
            -0.25
        ),
        Err(DftError::InvalidInput(_))
    ));
}

#[test]
fn gks_fixture_is_genuinely_noncollinear_and_complex() {
    let (molecule, geometry) = geometry();
    let density = trial_density(&basis(&molecule), &geometry);
    let mut gram = [[0.0; 3]; 3];
    let mut imaginary_max = 0.0_f64;
    for p in &density {
        for i in 0..3 {
            for j in 0..3 {
                gram[i][j] += p[i + 1]
                    .re
                    .as_slice()
                    .iter()
                    .zip(p[j + 1].re.as_slice())
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    + p[i + 1]
                        .im
                        .as_slice()
                        .iter()
                        .zip(p[j + 1].im.as_slice())
                        .map(|(a, b)| a * b)
                        .sum::<f64>();
            }
        }
        imaginary_max = p
            .iter()
            .flat_map(|a| a.im.as_slice())
            .map(|a| a.abs())
            .fold(imaginary_max, f64::max);
    }
    let minor = (0..3)
        .flat_map(|i| (i + 1..3).map(move |j| (i, j)))
        .map(|(i, j)| gram[i][i] * gram[j][j] - gram[i][j] * gram[j][i])
        .fold(0.0_f64, f64::max);
    println!("GKS_FIXTURE gram={gram:?} largest_spin_minor={minor:.17e} imaginary_max={imaginary_max:.17e}");
    assert!(
        minor > 1e-6,
        "a common collinear spin axis would have a rank-one Gram matrix"
    );
    assert!(imaginary_max > 1e-4);
}
