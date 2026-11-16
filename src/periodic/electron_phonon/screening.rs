// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Microscopic static RPA screening of the band-decoupled singlet kernel.
//!
//! A pair of bands at k and k+q uses the same analytic Gaussian Bloch
//! plane-wave form as finite-q DFPT. With F_G = <a,k+q|exp(i(q+G)r)|b,k>,
//! the response density uses rho_G = conj(F_G). The independent response
//! is a sum of rho rho† weighted by spin, volume, k weight and the Fermi
//! divided difference. The screened interaction solves
//! (v^-1 - chi0) W = I. The singlet pair matrix element is rho† W rho / V.

use crate::dfpt::coulomb::{plane_wave_pair, reciprocal_waves};
use crate::dfpt::orbitals::{adjoint, generalized_orbitals, matmul, BandOrbitals};
use crate::dfpt::thermodynamic::{divided_difference, fermi};
use crate::linalg::{solve_linear_matrix, Matrix};
use crate::math::Vec3;
use crate::periodic::features::PeriodicBoundary;
use crate::periodic::occupations::{OccupationScheme, BOLTZMANN_HARTREE_PER_K};
use crate::periodic::scf::{PeriodicScfOptions, PeriodicScfResult};
use crate::periodic::{bloch::BlochMatrix, Cell};
use crate::response::complex::Complex;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};
use std::f64::consts::PI;

/// Treatment of the zero reciprocal component at zero transfer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GammaHead {
    /// Static long-wavelength thermal response at fixed chemical potential.
    Thermal,
    /// Fixed-number neutral cell with the constant Coulomb mode removed.
    Neutral,
}

/// Convergence controls of the static screened Coulomb projection.
#[derive(Clone, Copy, Debug)]
pub struct CoulombScreeningOptions {
    /// Spherical cutoff on |q+G| in inverse Bohr.
    pub reciprocal_cutoff: f64,
    pub gamma_head: GammaHead,
    /// Absolute scaled inverse and normalized reference acceptance gate.
    pub tolerance: f64,
}

#[derive(Clone, Debug)]
pub struct StaticScreenedInteraction {
    /// Chi0, row=density reciprocal mode and column=potential source.
    /// Units are Bohr^-3/Hartree.
    pub independent_response: BlochMatrix,
    /// W in Hartree Bohr^3.
    pub interaction: BlochMatrix,
    /// Maximum absolute defect of the diagonally equilibrated inverse.
    pub scaled_inverse_residual: f64,
    /// Maximum defect of the original product (v^-1 - chi0) W - I.
    pub inverse_residual: f64,
    /// Raw relative Hermiticity defect of W.
    pub hermiticity_defect: f64,
}

/// Build the static microscopic RPA operator from transition density forms.
///
/// The complex form matrix has one row per q+G and one column per band
/// transition. Form entries are dimensionless. The nonpositive coefficients
/// include spin and reciprocal volume as well as the Fermi slopes, in
/// Bohr^-3/Hartree. A supplied zero mode is inverted together with its
/// local-field wings and requires nonzero thermal compressibility.
pub fn static_screened_interaction(
    waves: &[Vec3],
    density_forms: &BlochMatrix,
    coefficients: &[f64],
    tolerance: f64,
) -> Result<StaticScreenedInteraction> {
    let n = waves.len();
    if n == 0
        || density_forms.rows() != n
        || density_forms.cols() != coefficients.len()
        || density_forms.im.rows != n
        || density_forms.im.cols != coefficients.len()
        || coefficients.is_empty()
        || coefficients.iter().any(|c| !c.is_finite() || *c > 0.0)
        || waves.iter().any(|v| !v.norm2().is_finite())
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || tolerance >= 1.0
    {
        return Err(DftError::InvalidInput(
            "static RPA screening needs matching finite waves, forms and nonpositive transition coefficients".into(),
        ));
    }
    // -chi0 is the Gram matrix of the weighted reciprocal transition forms.
    let mut weighted = density_forms.clone();
    for (column, &coefficient) in coefficients.iter().enumerate() {
        let factor = (-coefficient).sqrt();
        for row in 0..n {
            weighted.re[(row, column)] *= factor;
            weighted.im[(row, column)] *= factor;
        }
    }
    let mut metric = matmul(&weighted, &adjoint(&weighted)?)?;
    let mut independent_response = metric.clone();
    independent_response.scale(-1.0);
    for (i, wave) in waves.iter().enumerate() {
        metric.re[(i, i)] += wave.norm2() / (4.0 * PI);
        if wave.norm2() == 0.0 && metric.re[(i, i)] == 0.0 {
            return Err(DftError::Unsupported {
                feature: "static SCDFT RPA zero-transfer Coulomb head without thermal compressibility"
                    .into(),
                missing: "a resolved metallic thermal susceptibility or an insulating q-cell Coulomb average (use the explicit neutral-cell gauge only for that finite-cell model)".into(),
            });
        }
    }
    let diagonal: Vec<_> = (0..n).map(|i| metric.re[(i, i)].sqrt()).collect();
    if diagonal.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return Err(DftError::LinearAlgebra(
            "static RPA inverse Coulomb diagonal is nonpositive or non-finite".into(),
        ));
    }
    // Equilibrate by the full inverse-Coulomb diagonal, including the
    // metallic head. Undo the congruence scaling on the solved inverse.
    let scaled = |part: &Matrix| {
        let mut scaled = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                scaled[(i, j)] = part[(i, j)] / diagonal[i] / diagonal[j];
            }
        }
        scaled
    };
    let operator = BlochMatrix {
        re: scaled(&metric.re),
        im: scaled(&metric.im),
    }
    .real_embedding();
    let identity = Matrix::identity(2 * n);
    let inverse = solve_linear_matrix(&operator, &identity)?;
    let scaled_inverse_residual = operator.matmul_seq(&inverse).max_abs_diff(&identity);
    let mut interaction = BlochMatrix::from_real_embedding(&inverse);
    for part in [&mut interaction.re, &mut interaction.im] {
        for i in 0..n {
            for j in 0..n {
                part[(i, j)] = part[(i, j)] / diagonal[i] / diagonal[j];
            }
        }
    }
    let image = matmul(&metric, &interaction)?;
    let inverse_residual = image.re.max_abs_diff(&Matrix::identity(n)).max(
        image
            .im
            .as_slice()
            .iter()
            .fold(0.0_f64, |m, x| m.max(x.abs())),
    );
    let interaction_scale = interaction
        .re
        .as_slice()
        .iter()
        .chain(interaction.im.as_slice())
        .fold(0.0_f64, |m, x| m.max(x.abs()));
    let hermiticity_defect = interaction.hermiticity_error() / interaction_scale;
    if !scaled_inverse_residual.is_finite()
        || scaled_inverse_residual > tolerance
        || !inverse_residual.is_finite()
        || inverse_residual > tolerance
        || !hermiticity_defect.is_finite()
        || hermiticity_defect > tolerance
    {
        return Err(DftError::LinearAlgebra(format!(
            "static RPA screened inverse failed its residual gate (scaled {scaled_inverse_residual:.6e}, raw {inverse_residual:.6e}, Hermiticity {hermiticity_defect:.6e})"
        )));
    }
    Ok(StaticScreenedInteraction {
        independent_response,
        interaction,
        scaled_inverse_residual,
        inverse_residual,
        hermiticity_defect,
    })
}

/// Generated Coulomb kernel in k-major / band-minor order.
#[derive(Clone, Debug)]
pub struct ScreenedPairCoulomb {
    /// Pair repulsion K in Hartree. Band and k weights are excluded.
    pub kernel: Matrix,
    /// Cartesian mesh differences (inverse Bohr) in mesh ordering.
    pub transfers: Vec<Vec3>,
    pub reciprocal_modes: Vec<usize>,
    pub scaled_inverse_residuals: Vec<f64>,
    pub inverse_residuals: Vec<f64>,
    pub inverse_hermiticity_defects: Vec<f64>,
    /// W(0) / volume in Hartree after microscopic local-field screening.
    pub gamma_head: Option<f64>,
    /// -chi0(0,0) before local-field screening, Bohr^-3/Hartree.
    pub gamma_compressibility: f64,
    /// Whether the reference chemical potential lies within a sampled band.
    pub fermi_level_crossing: bool,
    pub occupation_defect: f64,
    pub electron_number_residual: f64,
    pub spin_fock_defect: f64,
    pub time_reversal_fock_defect: f64,
    pub time_reversal_density_defect: f64,
    pub pair_imaginary_defect: f64,
    pub pair_reciprocity_defect: f64,
}

fn address(index: usize, mesh: [usize; 3]) -> [usize; 3] {
    [
        index / (mesh[1] * mesh[2]),
        index / mesh[2] % mesh[1],
        index % mesh[2],
    ]
}

fn translated_index(point: usize, shift: [usize; 3], mesh: [usize; 3]) -> usize {
    let a = address(point, mesh);
    let t = std::array::from_fn::<_, 3, _>(|i| (a[i] + shift[i]) % mesh[i]);
    (t[0] * mesh[1] + t[1]) * mesh[2] + t[2]
}

fn max_bloch_difference(a: &BlochMatrix, b: &BlochMatrix) -> f64 {
    a.re.max_abs_diff(&b.re).max(a.im.max_abs_diff(&b.im))
}

fn reference_bands(
    reference: &PeriodicScfResult,
    options: &PeriodicScfOptions,
) -> Result<Vec<BandOrbitals>> {
    reference.fock[0]
        .iter()
        .zip(&reference.overlap)
        .map(|(fock, overlap)| generalized_orbitals(fock, overlap, options.orth_threshold))
        .collect()
}

fn crosses_fermi_level(bands: &[BandOrbitals], mu: f64) -> bool {
    let rank = bands.iter().map(|o| o.energies.len()).min().unwrap_or(0);
    (0..rank).any(|i| {
        bands.iter().any(|o| o.energies[i] <= mu) && bands.iter().any(|o| o.energies[i] >= mu)
    })
}

fn fixed_number_body(
    forms: &BlochMatrix,
    coefficients: &[f64],
    pairs: &[(usize, usize)],
) -> Option<BlochMatrix> {
    let head: Vec<_> = pairs
        .iter()
        .zip(coefficients)
        .enumerate()
        .filter_map(|(i, ((a, b), &c))| (a == b).then_some((i, -c)))
        .collect();
    let head_metric = head.iter().map(|(_, weight)| weight).sum::<f64>();
    if head_metric == 0.0 {
        return None;
    }
    // Project the constant-density response before the Gram contraction.
    // This subtracts chi_b0 chi_0b / chi_00 from the susceptibility body.
    let mut projected = forms.clone();
    for part in [&mut projected.re, &mut projected.im] {
        for g in 0..forms.rows() {
            let mean = head
                .iter()
                .map(|&(i, weight)| (weight / head_metric) * part[(g, i)])
                .sum::<f64>();
            for &(i, _) in &head {
                part[(g, i)] -= mean;
            }
        }
    }
    Some(projected)
}

/// Generate the time-reversed singlet screened Coulomb matrix on the complete
/// reference mesh. All q differences are evaluated and both pair directions
/// are checked individually. The normal bands stay fixed.
pub fn screened_pair_coulomb(
    cell: &Cell,
    functional: &ResolvedFunctional,
    reference: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    screening: &CoulombScreeningOptions,
) -> Result<ScreenedPairCoulomb> {
    if options.boundary != PeriodicBoundary::FULL_3D {
        return Err(DftError::Unsupported {
            feature: "SCDFT static microscopic screening of a reduced-dimensional cell".into(),
            missing: "dimension-specific reciprocal Coulomb and long-wavelength pairing limits"
                .into(),
        });
    }
    reference.validate_derivative_plan(cell, functional, options)?;
    let mesh = options.k_mesh;
    let expected = cell.k_points(mesh);
    let nk = expected.len();
    let tolerance = screening.tolerance;
    let volume = cell.lattice.volume().abs();
    if nk == 0
        || reference.k_points != expected
        || reference.overlap.len() != nk
        || reference.fock.iter().any(|s| s.len() != nk)
        || reference.occupations.iter().any(|s| s.len() != nk)
        || !screening.reciprocal_cutoff.is_finite()
        || screening.reciprocal_cutoff <= 0.0
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || tolerance >= 1.0
        || !volume.is_finite()
        || volume == 0.0
    {
        return Err(DftError::InvalidInput(
            "SCDFT RPA screening requires a complete matching uniform mesh, finite cell and positive cutoff and tolerance".into(),
        ));
    }
    let OccupationScheme::FermiDirac { temperature_kelvin } = options.occupation else {
        return Err(DftError::Unsupported {
            feature: "SCDFT RPA screening on integer occupations".into(),
            missing: "a zero-temperature metallic Fermi-surface or insulating long-wave quadrature"
                .into(),
        });
    };
    let kt = temperature_kelvin * BOLTZMANN_HARTREE_PER_K;
    let [Some(mu), Some(mu_beta)] = reference.chemical_potential else {
        return Err(DftError::InvalidInput(
            "SCDFT finite-temperature screening requires both chemical potentials".into(),
        ));
    };
    let spin_fock_defect = reference.fock[0]
        .iter()
        .zip(&reference.fock[1])
        .map(|(a, b)| max_bloch_difference(a, b))
        .fold(0.0_f64, f64::max);
    let tr = reference.time_reversal_diagnostics(mesh)?;
    if reference.n_alpha != reference.n_beta
        || spin_fock_defect > tolerance
        || (mu - mu_beta).abs() > tolerance
        || tr.spinful_fock_residual > tolerance
        || tr.spinful_density_residual > tolerance
    {
        return Err(DftError::Unsupported {
            feature: "band-decoupled singlet SCDFT screening of a spin-inequivalent or time-reversal-breaking reference".into(),
            missing: format!(
                "a general spinor pair metric (spin Fock defect {spin_fock_defect:.6e}, time-reversal Fock defect {:.6e}, density defect {:.6e})",
                tr.spinful_fock_residual, tr.spinful_density_residual
            ),
        });
    }
    let bands = reference_bands(reference, options)?;
    let dual = bands
        .iter()
        .map(|band| adjoint(&band.coefficients))
        .collect::<Result<Vec<_>>>()?;
    let mut offsets = vec![0_usize];
    let mut occupation_defect = 0.0_f64;
    let mut number = 0.0;
    let mut population_slope = 0.0;
    for (k, band) in bands.iter().enumerate() {
        let end = offsets[k]
            .checked_add(band.energies.len())
            .ok_or_else(|| DftError::InvalidInput("SCDFT band count overflows".into()))?;
        offsets.push(end);
        if reference
            .occupations
            .iter()
            .any(|s| s[k].len() != band.energies.len())
        {
            return Err(DftError::InvalidInput(
                "SCDFT reference occupations do not match the canonical bands".into(),
            ));
        }
        for (i, &energy) in band.energies.iter().enumerate() {
            let (occupation, slope) = fermi(energy, mu, kt)?;
            number += occupation;
            population_slope += slope;
            for spin in &reference.occupations {
                occupation_defect = occupation_defect.max((spin[k][i] - occupation).abs());
            }
        }
    }
    let charge = (reference.n_alpha + reference.n_beta) as f64;
    let electron_number_residual = (2.0 * number / nk as f64 - charge).abs();
    if occupation_defect > tolerance || electron_number_residual > tolerance * charge.max(1.0) {
        return Err(DftError::LinearAlgebra(format!(
            "SCDFT normal-state occupation defect {occupation_defect:.6e} or number defect {electron_number_residual:.6e} exceeds the screening gate"
        )));
    }
    let n = offsets[nk];
    let mut output = ScreenedPairCoulomb {
        kernel: Matrix::zeros(n, n),
        transfers: Vec::with_capacity(nk),
        reciprocal_modes: Vec::with_capacity(nk),
        scaled_inverse_residuals: Vec::with_capacity(nk),
        inverse_residuals: Vec::with_capacity(nk),
        inverse_hermiticity_defects: Vec::with_capacity(nk),
        gamma_head: None,
        gamma_compressibility: -2.0 * population_slope / (volume * nk as f64),
        fermi_level_crossing: crosses_fermi_level(&bands, mu),
        occupation_defect,
        electron_number_residual,
        spin_fock_defect,
        time_reversal_fock_defect: tr.spinful_fock_residual,
        time_reversal_density_defect: tr.spinful_density_residual,
        pair_imaginary_defect: 0.0,
        pair_reciprocity_defect: 0.0,
    };
    let reciprocal = cell.lattice.reciprocal();
    for transfer in 0..nk {
        let shift = address(transfer, mesh);
        let q = (0..3).fold(Vec3::zero(), |q, i| {
            let offset = shift[i] as f64 / mesh[i] as f64;
            q + reciprocal[i] * if offset > 0.5 { offset - 1.0 } else { offset }
        });
        let include_head = transfer == 0 && screening.gamma_head == GammaHead::Thermal;
        let waves = if include_head {
            let mut waves = vec![Vec3::zero()];
            waves.extend(
                cell.lattice
                    .reciprocal_within(screening.reciprocal_cutoff * (1.0 + 1e-12)),
            );
            waves
        } else {
            reciprocal_waves(cell, q, screening.reciprocal_cutoff)?
        };
        let targets: Vec<_> = (0..nk).map(|k| translated_index(k, shift, mesh)).collect();
        let mut pair_indices = Vec::new();
        let mut coefficients = Vec::new();
        for (k, &target) in targets.iter().enumerate() {
            for (a, &ea) in bands[target].energies.iter().enumerate() {
                for (b, &eb) in bands[k].energies.iter().enumerate() {
                    pair_indices.push((offsets[target] + a, offsets[k] + b));
                    coefficients
                        .push(2.0 * divided_difference(ea, eb, mu, kt)? / (volume * nk as f64));
                }
            }
        }
        let mut forms = BlochMatrix::zeros(waves.len(), coefficients.len());
        for (g, &wave) in waves.iter().enumerate() {
            let mut column = 0;
            for (k, &target) in targets.iter().enumerate() {
                if wave == Vec3::zero() {
                    // The uniform potential is the overlap operator in the
                    // exact same metric used to define the canonical bands.
                    let count = bands[k].energies.len();
                    for a in 0..count {
                        for b in 0..count {
                            forms.re[(g, column)] = if a == b { 1.0 } else { 0.0 };
                            column += 1;
                        }
                    }
                    continue;
                }
                let ao = plane_wave_pair(
                    cell,
                    reference.k_points[k],
                    q,
                    wave,
                    cell.hartree_real_space_cutoff(),
                )?;
                let projected = matmul(&dual[target], &matmul(&ao, &bands[k].coefficients)?)?;
                for (&re, &im) in projected.re.as_slice().iter().zip(projected.im.as_slice()) {
                    forms.re[(g, column)] = re;
                    forms.im[(g, column)] = -im;
                    column += 1;
                }
            }
        }
        let projected = if transfer == 0 && !include_head {
            fixed_number_body(&forms, &coefficients, &pair_indices)
        } else {
            None
        };
        let screened = static_screened_interaction(
            &waves,
            projected.as_ref().unwrap_or(&forms),
            &coefficients,
            tolerance,
        )?;
        if include_head {
            output.gamma_head = Some(screened.interaction.re[(0, 0)] / volume);
        }
        let potential = matmul(&screened.interaction, &forms)?;
        for (column, (a, b)) in pair_indices.into_iter().enumerate() {
            let mut pair = Complex::ZERO;
            for g in 0..waves.len() {
                pair += Complex::new(forms.re[(g, column)], -forms.im[(g, column)])
                    * Complex::new(potential.re[(g, column)], potential.im[(g, column)]);
            }
            let value = pair / volume;
            output.pair_imaginary_defect = output.pair_imaginary_defect.max(value.im.abs());
            output.kernel[(a, b)] = value.re;
        }
        output.transfers.push(q);
        output.reciprocal_modes.push(waves.len());
        output
            .scaled_inverse_residuals
            .push(screened.scaled_inverse_residual);
        output.inverse_residuals.push(screened.inverse_residual);
        output
            .inverse_hermiticity_defects
            .push(screened.hermiticity_defect);
    }
    for i in 0..n {
        for j in 0..i {
            output.pair_reciprocity_defect = output
                .pair_reciprocity_defect
                .max((output.kernel[(i, j)] - output.kernel[(j, i)]).abs());
        }
    }
    let scale = output
        .kernel
        .as_slice()
        .iter()
        .fold(1.0_f64, |s, x| s.max(x.abs()));
    if output
        .kernel
        .as_slice()
        .iter()
        .any(|v| !v.is_finite() || *v < -tolerance * scale)
        || output.pair_imaginary_defect > tolerance * scale
        || output.pair_reciprocity_defect > tolerance * scale
    {
        return Err(DftError::LinearAlgebra(format!(
            "SCDFT screened Coulomb pair matrix failed positivity, reality or reciprocity (imaginary {:.6e}, reciprocity {:.6e})",
            output.pair_imaginary_defect, output.pair_reciprocity_defect
        )));
    }
    Ok(output)
}
