// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complex descriptor continuation of the real Gaussian AO basis, with
//! analytic derivatives on a fixed real UKS occupied subspace.

use super::{
    occupied_blocks, orbitals::fermi_frame, parent_orbital_residual, with_uks_sic_reference,
    DescriptorMode, MolecularDensityModel, SelfEnergy,
};
use crate::basis::{aos, cart_components, MolBasis};
use crate::grid::MolecularGrid;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::BlochMatrix;
use crate::pseudo_sub::fit::{projected_lbfgs, OptimizerResult};
use crate::response::complex::Complex;
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

#[derive(Clone, Copy, Debug)]
pub struct ComplexDescriptor {
    pub real: Vec3,
    pub imaginary: Vec3,
}

pub struct ComplexFloRun {
    pub energy: f64,
    pub parent_energy: f64,
    pub correction: f64,
    pub terms: [Vec<SelfEnergy>; 2],
    pub coefficients: [BlochMatrix; 2],
    pub descriptors: [Vec<ComplexDescriptor>; 2],
    /// Logarithms relative to the atomic-unit density 1/Bohr^3.
    pub log_descriptor_densities: [Vec<f64>; 2],
    pub orthogonality_residuals: [f64; 2],
    pub embedding_residuals: [f64; 2],
    pub density_residual: f64,
    pub parent_orbital_residual: f64,
    pub parent_iterations: usize,
    /// Real x,y,z followed by imaginary x,y,z, in Hartree per Bohr.
    pub gradients: Option<[Vec<[f64; 6]>; 2]>,
    pub optimization: Option<OptimizerResult>,
}

struct AoValues {
    values: BlochMatrix,
    gradient: Option<[BlochMatrix; 3]>,
    exponents: Vec<f64>,
}

fn monomial(offset: &[Complex; 3], powers: [i32; 3]) -> Complex {
    let mut value = Complex::ONE;
    for (&x, power) in offset.iter().zip(powers) {
        for _ in 0..power {
            value = value * x;
        }
    }
    value
}

fn complex_aos(
    basis: &MolBasis,
    descriptors: &[ComplexDescriptor],
    gradient: bool,
) -> Result<AoValues> {
    let zeros = || BlochMatrix::zeros(descriptors.len(), basis.nao_sph);
    let mut values = zeros();
    let mut derivative = gradient.then(|| std::array::from_fn(|_| zeros()));
    let mut exponents = Vec::with_capacity(descriptors.len());
    for (i, &point) in descriptors.iter().enumerate() {
        let imaginary2 = point.imaginary.norm2();
        // Subtract the same real exponent for every AO at a descriptor.
        // Its derivative cancels from the normalized Fermi orbital.
        let scale = basis
            .shells
            .iter()
            .map(|shell| {
                let r2 = (point.real - shell.center).norm2() - imaginary2;
                shell
                    .exps
                    .iter()
                    .map(|&alpha| -alpha * r2)
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .fold(f64::NEG_INFINITY, f64::max);
        if !scale.is_finite() {
            return Err(DftError::InvalidInput(
                "complex Fermi descriptor is outside finite Gaussian range".into(),
            ));
        }
        exponents.push(scale);
        for shell in &basis.shells {
            let real = point.real - shell.center;
            let r2 = real.norm2() - imaginary2;
            let cross = 2.0 * real.dot(point.imaginary);
            let offset = std::array::from_fn(|a| Complex::new(real.get(a), point.imaginary.get(a)));
            let mut radial = Complex::ZERO;
            let mut radial_gradient = Complex::ZERO;
            for (&alpha, &coefficient) in shell.exps.iter().zip(&shell.coefs) {
                let (sine, cosine) = (-alpha * cross).sin_cos();
                let term = Complex::new(cosine, sine) * (coefficient * (-alpha * r2 - scale).exp());
                radial += term;
                if gradient {
                    radial_gradient += term * (-2.0 * alpha);
                }
            }
            let transform = aos::cart_to_sph(shell.l);
            for (cartesian, (a, b, c)) in cart_components(shell.l).into_iter().enumerate() {
                let powers = [a, b, c];
                let polynomial = monomial(&offset, powers);
                let value = polynomial * radial;
                let gradients = gradient.then(|| {
                    std::array::from_fn::<_, 3, _>(|axis| {
                        let mut order = powers;
                        let mut result = polynomial * offset[axis] * radial_gradient;
                        if powers[axis] != 0 {
                            order[axis] -= 1;
                            result += monomial(&offset, order) * radial * f64::from(powers[axis]);
                        }
                        result
                    })
                });
                for sph in 0..transform.cols {
                    let j = shell.sph_off + sph;
                    let weight = transform[(cartesian, sph)];
                    values.re[(i, j)] += weight * value.re;
                    values.im[(i, j)] += weight * value.im;
                    if let (Some(derivative), Some(gradients)) = (&mut derivative, gradients) {
                        for (matrix, gradient) in derivative.iter_mut().zip(gradients) {
                            matrix.re[(i, j)] += weight * gradient.re;
                            matrix.im[(i, j)] += weight * gradient.im;
                        }
                    }
                }
            }
        }
    }
    Ok(AoValues {
        values,
        gradient: derivative,
        exponents,
    })
}

fn embed(re: &Matrix, im: Option<&Matrix>) -> Matrix {
    let (rows, cols) = (re.rows, re.cols);
    let mut result = Matrix::zeros(2 * rows, 2 * cols);
    for i in 0..rows {
        for j in 0..cols {
            result[(i, j)] = re[(i, j)];
            result[(i + rows, j + cols)] = re[(i, j)];
            if let Some(im) = im {
                result[(i + rows, j)] = im[(i, j)];
                result[(i, j + cols)] = -im[(i, j)];
            }
        }
    }
    result
}

fn unembed(matrix: &Matrix) -> (BlochMatrix, f64) {
    let (rows, cols) = (matrix.rows / 2, matrix.cols / 2);
    let mut result = BlochMatrix::zeros(rows, cols);
    let mut defect = 0.0_f64;
    for i in 0..rows {
        for j in 0..cols {
            let first = matrix[(i, j)];
            let second = matrix[(i + rows, j + cols)];
            let third = matrix[(i + rows, j)];
            let fourth = -matrix[(i, j + cols)];
            defect = defect
                .max((first - second).abs())
                .max((third - fourth).abs());
            result.re[(i, j)] = first.midpoint(second);
            result.im[(i, j)] = third.midpoint(fourth);
        }
    }
    (result, defect)
}

struct SpinResult {
    terms: Vec<SelfEnergy>,
    coefficients: BlochMatrix,
    log_densities: Vec<f64>,
    gradient: Vec<[f64; 6]>,
    orthogonality: f64,
    embedding_residual: f64,
    density_residual: f64,
}

fn spin_result(
    model: &MolecularDensityModel<'_>,
    overlap: &Matrix,
    occupied: &Matrix,
    descriptors: &[ComplexDescriptor],
    want_gradient: bool,
) -> Result<SpinResult> {
    let ao = complex_aos(model.mb, descriptors, want_gradient)?;
    let occupied_real = embed(occupied, None);
    let frame = fermi_frame(
        &occupied_real,
        overlap,
        &embed(&ao.values.re, Some(&ao.values.im)),
    )?;
    let (coefficients, defect) = unembed(&frame.orbitals.coefficients);
    if defect > 1e-10 {
        return Err(DftError::LinearAlgebra(format!(
            "complex Fermi-Lowdin orbital embedding defect {defect:.6e}"
        )));
    }
    let mut terms = Vec::with_capacity(occupied.cols);
    let empty = Matrix::zeros(occupied.rows, occupied.rows);
    let mut grad = want_gradient.then(|| BlochMatrix::zeros(occupied.rows, occupied.cols));
    for orbital in 0..occupied.cols {
        let column = |m: &Matrix| {
            Matrix::from_row_major(m.rows, 1, (0..m.rows).map(|a| m[(a, orbital)]).collect())
        };
        let real = column(&coefficients.re);
        let imaginary = column(&coefficients.im);
        let mut density = real.leading_columns_gram(1, 1.0);
        super::super::add_scaled(&mut density, &imaginary.leading_columns_gram(1, 1.0), 1.0);
        let (term, potential) = model.single_complex_orbital(&density, &empty)?;
        terms.push(term);
        if let Some(grad) = &mut grad {
            let image_real = potential.matmul_seq(&real);
            let image_imaginary = potential.matmul_seq(&imaginary);
            for a in 0..occupied.rows {
                grad.re[(a, orbital)] = -image_real[(a, 0)];
                grad.im[(a, orbital)] = -image_imaginary[(a, 0)];
            }
        }
    }
    let mut gradient = Vec::new();
    if let (Some(grad), Some(ao_gradient)) = (grad, ao.gradient) {
        // The doubled real inner product counts each complex entry twice.
        // The gradients above therefore carry half of the complex gradient.
        let gradient_values =
            frame.value_gradient(&occupied_real, &embed(&grad.re, Some(&grad.im)));
        let (pulled_back, _) = unembed(&gradient_values);
        let contraction = |d: usize, ao: &BlochMatrix| {
            (0..occupied.rows).fold(Complex::ZERO, |sum, p| {
                let source = (0..occupied.cols).fold(Complex::ZERO, |s, j| {
                    s + Complex::new(pulled_back.re[(d, j)], pulled_back.im[(d, j)])
                        * occupied[(p, j)]
                });
                sum + source.conj() * Complex::new(ao.re[(d, p)], ao.im[(d, p)])
            }) * 2.0
        };
        for d in 0..descriptors.len() {
            let values = ao_gradient.each_ref().map(|a| contraction(d, a));
            gradient.push([
                values[0].re,
                values[1].re,
                values[2].re,
                -values[0].im,
                -values[1].im,
                -values[2].im,
            ]);
        }
        if gradient.iter().flatten().any(|x| !x.is_finite()) {
            return Err(DftError::LinearAlgebra(
                "non-finite complex descriptor force".into(),
            ));
        }
    }
    let density = coefficients.re.leading_columns_gram(occupied.cols, 1.0);
    let imaginary_density = coefficients.im.leading_columns_gram(occupied.cols, 1.0);
    let mut total = density;
    super::super::add_scaled(&mut total, &imaginary_density, 1.0);
    let mut imaginary_density = coefficients.im.matmul_transpose_seq(&coefficients.re);
    let transpose = imaginary_density.transpose();
    super::super::add_scaled(&mut imaginary_density, &transpose, -1.0);
    let density_residual = total
        .max_abs_diff(&occupied.leading_columns_gram(occupied.cols, 1.0))
        .max(
            imaginary_density
                .as_slice()
                .iter()
                .map(|x| x.abs())
                .fold(0.0, f64::max),
        );
    let log_densities = ao
        .exponents
        .iter()
        .zip(&frame.orbitals.descriptor_densities)
        .map(|(&scale, &density)| 2.0 * scale + density.ln())
        .collect::<Vec<_>>();
    if log_densities.iter().any(|x| !x.is_finite()) {
        return Err(DftError::InvalidInput(
            "non-finite complex Fermi descriptor log density".into(),
        ));
    }
    Ok(SpinResult {
        terms,
        coefficients,
        log_densities,
        gradient,
        orthogonality: frame.orbitals.orthogonality_residual,
        embedding_residual: defect,
        density_residual,
    })
}

fn correction(spins: &[SpinResult; 2]) -> f64 {
    -spins
        .iter()
        .flat_map(|s| &s.terms)
        .map(SelfEnergy::total)
        .sum::<f64>()
}

fn at_descriptors(
    model: &MolecularDensityModel<'_>,
    seed: &ScfResult,
    descriptors: [&[ComplexDescriptor]; 2],
    mode: DescriptorMode,
) -> Result<ComplexFloRun> {
    let counts = [seed.n_alpha, seed.n_beta];
    for (spin, count) in counts.into_iter().enumerate() {
        if descriptors[spin].len() != count
            || descriptors[spin]
                .iter()
                .flat_map(|p| p.real.to_array().into_iter().chain(p.imaginary.to_array()))
                .any(|x| !x.is_finite())
        {
            return Err(DftError::InvalidInput(format!(
                "complex FLO-SIC spin {spin} needs {count} finite descriptors"
            )));
        }
    }
    let occupied = occupied_blocks(&seed.mo_coeff, counts);
    let overlap = embed(&seed.overlap, None);
    let evaluate = |d: [&[ComplexDescriptor]; 2], gradient| -> Result<[SpinResult; 2]> {
        Ok([
            spin_result(model, &overlap, &occupied[0], d[0], gradient)?,
            spin_result(model, &overlap, &occupied[1], d[1], gradient)?,
        ])
    };
    let coordinates = |x: &[f64]| {
        let (a, b) = x.split_at(6 * counts[0]);
        [a, b].map(|part| {
            part.chunks_exact(6)
                .map(|p| ComplexDescriptor {
                    real: Vec3::new(p[0], p[1], p[2]),
                    imaginary: Vec3::new(p[3], p[4], p[5]),
                })
                .collect::<Vec<_>>()
        })
    };
    let optimization = if let DescriptorMode::Optimize {
        gradient_tolerance,
        max_iterations,
    } = mode
    {
        let initial: Vec<_> = descriptors
            .iter()
            .flat_map(|d| d.iter())
            .flat_map(|p| p.real.to_array().into_iter().chain(p.imaginary.to_array()))
            .collect();
        Some(projected_lbfgs(
            &initial,
            8,
            max_iterations,
            gradient_tolerance,
            1e-10,
            |x| {
                let d = coordinates(x);
                let spins = evaluate([&d[0], &d[1]], true)?;
                Ok((
                    correction(&spins),
                    spins
                        .into_iter()
                        .flat_map(|s| s.gradient.into_iter().flatten())
                        .collect(),
                ))
            },
            |_| Ok(0),
        )?)
    } else {
        None
    };
    let descriptors = optimization.as_ref().map_or_else(
        || descriptors.map(<[ComplexDescriptor]>::to_vec),
        |x| coordinates(&x.parameters),
    );
    let gradients = !matches!(mode, DescriptorMode::Fixed);
    let mut spins = evaluate([&descriptors[0], &descriptors[1]], gradients)?;
    let correction = correction(&spins);
    if !correction.is_finite() {
        return Err(DftError::InvalidInput(
            "non-finite complex FLO-SIC correction".into(),
        ));
    }
    Ok(ComplexFloRun {
        energy: seed.energy + correction,
        parent_energy: seed.energy,
        correction,
        terms: std::array::from_fn(|s| std::mem::take(&mut spins[s].terms)),
        log_descriptor_densities: std::array::from_fn(|s| {
            std::mem::take(&mut spins[s].log_densities)
        }),
        gradients: gradients
            .then(|| std::array::from_fn(|s| std::mem::take(&mut spins[s].gradient))),
        orthogonality_residuals: spins.each_ref().map(|s| s.orthogonality),
        embedding_residuals: spins.each_ref().map(|s| s.embedding_residual),
        density_residual: spins[0].density_residual.max(spins[1].density_residual),
        coefficients: spins.map(|s| s.coefficients),
        descriptors,
        parent_orbital_residual: parent_orbital_residual(seed),
        parent_iterations: seed.iterations,
        optimization,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn run_complex_descriptors(
    mol: &Molecule,
    basis: &MolBasis,
    options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    descriptors: [&[ComplexDescriptor]; 2],
    mode: DescriptorMode,
) -> Result<ComplexFloRun> {
    if matches!(mode, DescriptorMode::SelfConsistent(_)) {
        return Err(DftError::Unsupported {
            feature: "fully self-consistent complex FLO-SIC".into(),
            missing: "the joint complex occupied-subspace and orbital-dependent Fock optimization"
                .into(),
        });
    }
    with_uks_sic_reference(
        mol,
        basis,
        options,
        functional,
        grid,
        "complex FLO-SIC",
        |model, seed| at_descriptors(model, seed, descriptors, mode),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::c1_geometry::assert_c1;
    use crate::acceptance::stencils::{fit_components_staged, Stencil};
    use crate::driver::method::{prepare_molecular, MolecularMethod};
    use crate::scf::sic::flo_sic::evaluate_descriptors;

    #[test]
    fn complex_fod_real_limit_and_every_cartesian_derivative() {
        let positions = [
            [0.13, -0.27, 0.19],
            [1.77, 0.36, -0.08],
            [2.44, 2.35, 1.20],
            [3.97, 2.88, 1.67],
        ];
        assert_c1(&[1; 4], &positions);
        let system = prepare_molecular(
            &[1; 4],
            &positions,
            &MolecularMethod {
                xc: Some("pbe".into()),
                unrestricted: true,
                grid_level: 0,
                grid_weight_cutoff: 0.0,
                dispersion: "none".into(),
                ..MolecularMethod::default()
            },
        )
        .unwrap();
        let real = [
            [Vec3::new(0.68, 0.07, -0.09), Vec3::new(3.41, 2.31, 1.29)],
            [Vec3::new(0.79, 0.03, 0.11), Vec3::new(3.29, 2.55, 1.44)],
        ];
        let imaginary = [
            [Vec3::new(0.19, -0.22, 0.30), Vec3::new(-0.22, 0.28, 0.24)],
            [Vec3::new(0.25, -0.31, 0.17), Vec3::new(0.23, 0.18, -0.26)],
        ];
        let empty = [[Vec3::zero(); 2]; 2];
        let pair = |imag: [[Vec3; 2]; 2]| {
            std::array::from_fn::<_, 2, _>(|s| {
                std::array::from_fn::<_, 2, _>(|j| ComplexDescriptor {
                    real: real[s][j],
                    imaginary: imag[s][j],
                })
            })
        };
        let complex = pair(imaginary);
        with_uks_sic_reference(
            &system.molecule, &system.basis, &system.options, &system.functional, system.grid.as_ref(),
            "complex FOD acceptance",
            |model, seed| {
                let fixed = |d: &[[ComplexDescriptor; 2]; 2]| at_descriptors(
                    model, seed, [&d[0], &d[1]], DescriptorMode::Fixed,
                );
                let occupied = occupied_blocks(&seed.mo_coeff, [2, 2]);
                let (real_result, _) = evaluate_descriptors(
                    model, seed, &occupied, [&real[0], &real[1]], false, 0.0,
                )?;
                assert!((fixed(&pair(empty))?.correction - real_result.correction).abs() < 1e-11);
                let base = at_descriptors(model, seed, [&complex[0], &complex[1]], DescriptorMode::Gradient)?;
                let conjugate = pair(imaginary.map(|s| s.map(|p| p * -1.0)));
                assert!((fixed(&conjugate)?.correction - base.correction).abs() < 1e-11);
                assert!(base.density_residual < 1e-12);
                assert!(base.orthogonality_residuals.iter().all(|&x| x < 1e-12));
                assert!(base.coefficients.iter().flat_map(|m| m.im.as_slice()).any(|x| x.abs() > 1e-4));
                let targets = base.gradients.as_ref().unwrap();
                let indices: Vec<_> = (0..2).flat_map(|s| (0..2).flat_map(move |d| (0..6).map(move |a| (s, d, a)))).collect();
                let labels: Vec<_> = indices.iter().map(|&(s,d,a)| format!("complex FOD spin={s} descriptor={d} axis={a}")).collect();
                let mut failures = Vec::new();
                let central = |(s, d, a): (usize, usize, usize), h: f64, stencil: Stencil| {
                    let energy = |step| {
                        let mut trial = complex;
                        let descriptor = &mut trial[s][d];
                        let point = if a < 3 { &mut descriptor.real } else { &mut descriptor.imaginary };
                        let mut r = point.to_array();
                        r[a % 3] += step;
                        *point = Vec3::new(r[0], r[1], r[2]);
                        fixed(&trial).unwrap().correction
                    };
                    let (plus2, minus2) = if stencil == Stencil::Fourth { (energy(2.0 * h), energy(-2.0 * h)) } else { (0.0, 0.0) };
                    stencil.first(energy(h), energy(-h), plus2, minus2, h)
                };
                fit_components_staged(
                    &labels,
                    &[
                        ([0.012, 0.008, 0.005, 0.003], Stencil::Second),
                        ([0.006, 0.004, 0.0025, 0.0015], Stencil::Second),
                        ([0.032, 0.024, 0.016, 0.010], Stencil::Fourth),
                    ],
                    |steps, stencil| indices.iter().map(|&(s,d,a)| {
                        steps.map(|h| (central((s,d,a), h, stencil) - targets[s][d][a]).abs())
                    }).collect(),
                    &mut failures,
                );
                assert!(failures.is_empty(), "{failures:?}");
                for &(s,d,a) in &indices {
                    assert!((central((s,d,a), 0.002, Stencil::Fourth) - targets[s][d][a]).abs() < 2e-8);
                }
                println!(
                    "C1 H4 cFLO-SIC E={:.12e}, correction={:.12e}, real correction={:.12e}, density={:.3e}, orthogonality={:?}",
                    base.energy, base.correction, real_result.correction, base.density_residual,
                    base.orthogonality_residuals
                );
                Ok(())
            },
        ).unwrap();
    }
}
