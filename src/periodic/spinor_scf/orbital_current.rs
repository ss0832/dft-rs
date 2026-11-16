// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Static orbital-current response of an insulating periodic spinor reference.

use super::{spin_diagonal, BlochMatrix, SpinorDensityKernel, SpinorDftBackend, SpinorScfReport};
use crate::math::Vec3;
use crate::periodic::field::vertices;
use crate::periodic::magnetic_scf::{add, transverse_frame};
use crate::periodic::phonon::adjoint;
use crate::response::complex::Complex;
use crate::response::krylov::ResponseSolverConfig;
use crate::response::nuclear::OrbitalSusceptibility;
use crate::response::spinor::spinor_operator_response_with_kernel;
use crate::{DftError, Result};

/// Phase-averaged orbital susceptibility in the two transverse directions.
#[derive(Clone, Debug)]
pub struct SpinorOrbitalSusceptibilityPoint {
    pub orbital: OrbitalSusceptibility,
    /// Susceptibility matrix in the reported magnetic-field axes.
    pub chi_transverse: [[f64; 2]; 2],
    /// Phase-averaged finite-q and uniform quadratic energy coefficients.
    pub paramagnetic_transverse: [[f64; 2]; 2],
    pub uniform_transverse: [[f64; 2]; 2],
    pub response_iterations: [[usize; 3]; 2],
    /// Maximum imaginary part and transpose defect before symmetrization.
    pub imaginary_defect: f64,
    pub reciprocity_defect: f64,
}

/// Extend the real-linear operator on Hermitian density changes to the
/// paired complex transition densities used by the response solver.
fn transition_kernel(
    kernel: &dyn SpinorDensityKernel,
    density: &BlochMatrix,
) -> Result<BlochMatrix> {
    let adj = adjoint(density);
    let mut real = density.clone();
    real.add_scaled(&adj, 1.0);
    real.scale(0.5);
    let mut result = kernel.apply(&[real])?.remove(0);
    let mut anti = density.clone();
    anti.add_scaled(&adj, -1.0);
    if anti
        .re
        .as_slice()
        .iter()
        .chain(anti.im.as_slice())
        .any(|&x| x != 0.0)
    {
        anti.scale(0.5);
        let mut imaginary = BlochMatrix {
            re: anti.im,
            im: anti.re,
        };
        for value in imaginary.im.as_mut_slice() {
            *value = -*value;
        }
        let opposite = kernel.apply(&[imaginary])?.remove(0);
        add(&mut result, &opposite, Complex::new(0.0, 1.0));
    }
    Ok(result)
}

impl SpinorDftBackend<'_> {
    /// Orbital vector-potential response for reciprocal waves of this
    /// Gamma supercell. Any prescribed spin bias must be local and fixed
    /// during the orbital perturbation. `replicas` converts the quadratic
    /// energy coefficients to a single primitive cell.
    pub fn orbital_susceptibility_q(
        &self,
        reference: &SpinorScfReport,
        replicas: usize,
        waves: &[[i64; 3]],
        config: &ResponseSolverConfig,
    ) -> Result<Vec<SpinorOrbitalSusceptibilityPoint>> {
        config.validate()?;
        let frame = self.scalar.field_free_nuclear_frame()?;
        if frame.geometry.dimension() != 3 || self.k_points() != [Vec3::zero()] {
            return Err(DftError::Unsupported {
                feature: "periodic spinor orbital susceptibility".into(),
                missing:
                    "the transverse current response outside a three-dimensional Gamma supercell"
                        .into(),
            });
        }
        if !frame.basis.ecps.is_empty()
            || self.nuclear_soc.iter().any(|h| {
                h.re.as_slice()
                    .iter()
                    .chain(h.im.as_slice())
                    .any(|&x| x != 0.0)
            })
        {
            return Err(DftError::Unsupported {
                feature: "periodic spinor orbital susceptibility with nonlocal ECP or nuclear SOC"
                    .into(),
                missing: "the covariant nonlocal-core and nuclear spin-orbit current vertices"
                    .into(),
            });
        }
        if !reference.converged || !reference.fields.operator_converged {
            return Err(DftError::ScfNotConverged {
                iterations: reference.iterations,
                error: reference.orbital_residual,
            });
        }
        if replicas == 0 || waves.is_empty() || waves.contains(&[0; 3]) {
            return Err(DftError::InvalidInput(
                "orbital susceptibility requires a positive replica count and nonzero waves".into(),
            ));
        }
        let ([band], [occupations]) =
            (reference.bands.as_slice(), reference.occupations.as_slice())
        else {
            return Err(DftError::InvalidInput(
                "periodic spinor orbital susceptibility needs one Gamma orbital frame".into(),
            ));
        };
        let occupied = occupations.iter().take_while(|&&f| f == 1.0).count();
        if occupations.len() != band.energies.len()
            || occupied == 0
            || occupied == occupations.len()
            || occupations[occupied..].iter().any(|&f| f != 0.0)
        {
            return Err(DftError::Unsupported {
                feature: "periodic spinor orbital susceptibility".into(),
                missing: "an insulating integer reference with occupied and empty spinors".into(),
            });
        }
        // In each projected spin channel tau_A and j_A acquire the same
        // vector-potential shift. It cancels in tau_A - |j_A|^2/(2 rho),
        // leaving the paramagnetic-current density kernel at zero field.
        let kernel = super::kernel::DensityKernel::new(self, &reference.density)?;
        let induced = |density: &BlochMatrix| transition_kernel(&kernel, density);
        let current = |q| {
            let (_, components) = vertices(
                frame.basis,
                frame.geometry,
                self.k_points(),
                q,
                frame.options.basis_cutoff,
                true,
            )?;
            Ok::<_, DftError>(components.map(|mut axis| axis.remove(0)))
        };
        let flat = current(Vec3::zero())?;
        let reciprocal = frame.geometry.reciprocal();
        let scale = 1.0 / replicas as f64;
        let volume = frame.geometry.lattice().volume().abs() * scale;
        waves.iter().map(|&wave| {
            let q = (0..3).fold(Vec3::zero(), |total, axis| total + reciprocal[axis] * wave[axis] as f64);
            let (polarizations, field_axes) = transverse_frame(q);
            let positive = current(q)?;
            let negative = current(-q)?;
            let n = frame.basis.nao_sph;
            let mut sources = Vec::with_capacity(6);
            // A cosine has Fourier coefficients e/2; the sine has -i e/2
            // at +q. The uniform limit uses A=e and includes both spin blocks.
            for e in polarizations {
                let mut cosine = BlochMatrix::zeros(n, n);
                let mut sine = BlochMatrix::zeros(n, n);
                let mut uniform = BlochMatrix::zeros(n, n);
                for axis in 0..3 {
                    let a = e.get(axis);
                    add(&mut cosine, &positive[axis], Complex::new(0.5 * a, 0.0));
                    add(&mut cosine, &negative[axis], Complex::new(0.5 * a, 0.0));
                    add(&mut sine, &positive[axis], Complex::new(0.0, -0.5 * a));
                    add(&mut sine, &negative[axis], Complex::new(0.0, 0.5 * a));
                    uniform.add_scaled(&flat[axis], a);
                }
                for vertex in [cosine, sine, uniform] {
                    if super::valid(&vertex, n) {
                        sources.push(spin_diagonal(&vertex));
                    } else {
                        return Err(DftError::LinearAlgebra("non-Hermitian orbital current vertex".into()));
                    }
                }
            }
            let response = spinor_operator_response_with_kernel(
                &band.coefficients, &band.energies, occupied, &induced, &sources,
                &[Complex::ZERO], config,
            )?.remove(0);
            let raw = &response.susceptibility;
            let mut imaginary_defect = 0.0_f64;
            let mut reciprocity_defect = 0.0_f64;
            let mut magnitude = 0.0_f64;
            for (i, row) in raw.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    imaginary_defect = imaginary_defect.max(v.im.abs());
                    reciprocity_defect = reciprocity_defect.max((v - raw[j][i]).abs());
                    magnitude = magnitude.max(v.abs());
                }
            }
            let tolerance = 16.0 * config.tolerance * (1.0 + magnitude);
            if imaginary_defect.max(reciprocity_defect) > tolerance {
                return Err(DftError::LinearAlgebra(format!("spinor static orbital response defects exceed {tolerance:e}: imaginary {imaginary_defect:e}, reciprocity {reciprocity_defect:e}")));
            }
            // The operator solve gives the second derivative. Multiply by
            // one half for the quadratic energy coefficient and average phases.
            let coefficient = |a: usize, b: usize, phase: usize| {
                let i = 3 * a + phase;
                let j = 3 * b + phase;
                0.25 * scale * (raw[i][j].re + raw[j][i].re)
            };
            let paramagnetic = std::array::from_fn(|a| std::array::from_fn(|b|
                0.5 * (coefficient(a, b, 0) + coefficient(a, b, 1))));
            let uniform = std::array::from_fn(|a| std::array::from_fn(|b| coefficient(a, b, 2)));
            let chi = std::array::from_fn(|a| std::array::from_fn(|b|
                -4.0 * (paramagnetic[a][b] - 0.5 * uniform[a][b]) / (volume * q.norm2())));
            let diagonal = |m: [[f64; 2]; 2]| [m[0][0], m[1][1]];
            Ok(SpinorOrbitalSusceptibilityPoint {
                orbital: OrbitalSusceptibility {
                    wave,
                    vector: q.to_array(),
                    polarization: polarizations.map(Vec3::to_array),
                    field: field_axes.map(Vec3::to_array),
                    chi: diagonal(chi),
                    paramagnetic: diagonal(paramagnetic),
                    uniform: diagonal(uniform),
                    response_residuals: std::array::from_fn(|a| std::array::from_fn(|p| response.residuals[3*a+p])),
                },
                chi_transverse: chi,
                paramagnetic_transverse: paramagnetic,
                uniform_transverse: uniform,
                response_iterations: std::array::from_fn(|a| std::array::from_fn(|p| response.iterations[3*a+p])),
                imaginary_defect,
                reciprocity_defect,
            })
        }).collect()
    }
}
