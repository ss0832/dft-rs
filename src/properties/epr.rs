// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Reference-preserving electronic paramagnetic resonance operators.

#[path = "epr/doublet.rs"]
pub mod doublet;
#[path = "epr/spinor.rs"]
pub mod spinor;

use crate::basis::MolBasis;
use crate::constants::{ELECTRON_G_FACTOR, FINE_STRUCTURE};
use crate::grid::MolecularGrid;
use crate::integrals::london::ComplexMatrix;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::response::nuclear::open_shell_operator_response;
use crate::response::{Frequency, ResponseSolverConfig};
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

/// Common-origin, one-electron Breit-Pauli g tensor. Each Cartesian entry
/// retains its independent common-orbital response contraction.
pub struct CommonOrbitalGTensor {
    pub tensor: [[f64; 3]; 3],
    pub mass_correction: f64,
    pub gauge_correction: [[f64; 3]; 3],
    pub spin_orbit: [[f64; 3]; 3],
    pub response_residuals: Vec<f64>,
    pub reference_stationarity: f64,
    pub metric_defect: f64,
}

fn scaled(matrix: &Matrix, coefficient: f64) -> Matrix {
    let mut out = matrix.clone();
    for value in out.as_mut_slice() {
        *value *= coefficient;
    }
    out
}

fn imaginary(matrix: Matrix) -> ComplexMatrix {
    ComplexMatrix {
        re: Matrix::zeros(matrix.rows, matrix.cols),
        im: matrix,
    }
}

fn trace(a: &Matrix, b: &Matrix) -> f64 {
    (0..a.rows)
        .map(|i| (0..a.cols).map(|j| a[(i, j)] * b[(j, i)]).sum::<f64>())
        .sum()
}

/// Static contact/SD hyperfine tensor of a common-orbital open shell.
/// The common orbitals must reproduce both supplied spin densities; an
/// unrestricted density with unrelated occupied spaces is not relabeled.
pub fn common_orbital_hyperfine_tensor(
    basis: &MolBasis,
    reference: &ScfResult,
    common_orbitals: &Matrix,
    center: Vec3,
    tolerance: f64,
) -> Result<crate::properties::hyperfine::HyperfineTensor> {
    let n = basis.nao_sph;
    if !reference.converged
        || !reference.has_integer_occupations()
        || reference.electronic_temperature != 0.0
        || reference.n_alpha <= reference.n_beta
        || common_orbitals.rows != n
        || common_orbitals.cols < reference.n_alpha
        || reference.overlap.rows != n
        || reference.overlap.cols != n
        || reference.density.iter().any(|p| p.rows != n || p.cols != n)
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || common_orbitals.as_slice().iter().any(|v| !v.is_finite())
        || reference.overlap.as_slice().iter().any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput("common-orbital hyperfine requires a converged integer open-shell reference, finite matching common orbitals and positive tolerance".into()));
    }
    let metric = common_orbitals
        .transpose()
        .matmul_seq(&reference.overlap)
        .matmul_seq(common_orbitals);
    let mut defect = 0.0_f64;
    for i in 0..metric.rows {
        for j in 0..metric.cols {
            defect = defect.max((metric[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs());
        }
    }
    for (spin, occupied) in [reference.n_alpha, reference.n_beta]
        .into_iter()
        .enumerate()
    {
        for i in 0..n {
            for j in 0..n {
                let expected = (0..occupied)
                    .map(|p| common_orbitals[(i, p)] * common_orbitals[(j, p)])
                    .sum::<f64>();
                let actual = reference.density[spin][(i, j)];
                if !actual.is_finite() {
                    return Err(DftError::InvalidInput(
                        "nonfinite common-orbital density".into(),
                    ));
                }
                defect = defect.max((expected - actual).abs());
            }
        }
    }
    if !defect.is_finite() || defect > tolerance {
        return Err(DftError::InvalidInput(format!(
            "common-orbital spin-projector/metric defect {defect} exceeds {tolerance}"
        )));
    }
    let mut spin_density = reference.density[0].clone();
    for (value, beta) in spin_density
        .as_mut_slice()
        .iter_mut()
        .zip(reference.density[1].as_slice())
    {
        *value -= beta;
    }
    crate::properties::hyperfine::spin_hyperfine_tensor(
        basis,
        &spin_density,
        center,
        (reference.n_alpha - reference.n_beta) as f64,
    )
}

/// Static ROKS/ROHF g tensor on the constrained common-spatial-orbital
/// manifold. The Zeeman and spin-orbit vertices are Hermitian imaginary
/// operators. The response is the paired common-orbital Hessian, never the
/// unrestricted orbital Hessian. Coulomb/exchange/XC are supplied by the
/// reference's generic functional. London overlap/phase and current-density
/// meta-GGA terms require their own providers and are not inferred here.
#[allow(clippy::too_many_arguments)]
pub fn common_orbital_g_tensor(
    molecule: &Molecule,
    basis: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    common_orbitals: &Matrix,
    options: &ScfOptions,
    origin: Vec3,
    soc_charges: &[f64],
    config: &ResponseSolverConfig,
) -> Result<CommonOrbitalGTensor> {
    config.validate()?;
    if !basis.ecps.is_empty()
        || soc_charges.len() != molecule.len()
        || soc_charges.iter().any(|v| !v.is_finite())
        || !origin.norm().is_finite()
        || reference.n_alpha <= reference.n_beta
    {
        return Err(DftError::InvalidInput(
            "common-orbital g tensor requires all-electron open shells, a finite origin and one finite SOC charge per nucleus".into(),
        ));
    }
    if functional.family == crate::xc::Family::MetaGga || options.skala.is_some() {
        return Err(DftError::Unsupported {
            feature: "common-orbital meta-GGA or learned g tensor".into(),
            missing:
                "the current-density Hessian in the constrained imaginary common-orbital response"
                    .into(),
        });
    }
    let angular = super::nmr::angular_momentum_operator(basis, origin);
    let n = basis.nao_sph;
    let mut orbital = std::array::from_fn::<_, 3, _>(|_| Matrix::zeros(n, n));
    for (atom, charge) in molecule.atoms.iter().zip(soc_charges) {
        let local = super::nmr::paramagnetic_operator(basis, atom.position);
        for axis in 0..3 {
            for (out, value) in orbital[axis]
                .as_mut_slice()
                .iter_mut()
                .zip(local[axis].as_slice())
            {
                *out += charge * value;
            }
        }
    }
    let mut sources: Vec<[ComplexMatrix; 2]> = angular
        .iter()
        .map(|matrix| {
            let vertex = imaginary(scaled(matrix, -0.5));
            [vertex.clone(), vertex]
        })
        .collect();
    sources.extend(
        orbital
            .iter()
            .map(|matrix| [imaginary(matrix.clone()), imaginary(scaled(matrix, -1.0))]),
    );
    let response = open_shell_operator_response(
        molecule,
        basis,
        functional,
        grid,
        reference,
        common_orbitals,
        options,
        &sources,
        Frequency::Static,
        config,
    )?;
    if response.reference_stationarity > config.acceptance()
        || response.metric_defect > config.acceptance()
        || response
            .susceptibility
            .iter()
            .flatten()
            .any(|z| z.im.abs() > config.acceptance())
    {
        return Err(DftError::ResponseNotConverged {
            iterations: 0,
            residual: response
                .reference_stationarity
                .max(response.metric_defect)
                .max(
                    response
                        .susceptibility
                        .iter()
                        .flatten()
                        .map(|z| z.im.abs())
                        .fold(0.0, f64::max),
                ),
        });
    }
    let spin_density = {
        let mut out = reference.density[0].clone();
        for (value, beta) in out
            .as_mut_slice()
            .iter_mut()
            .zip(reference.density[1].as_slice())
        {
            *value -= beta;
        }
        out
    };
    let factor = FINE_STRUCTURE.powi(2) / (reference.n_alpha - reference.n_beta) as f64;
    let mass_correction = -factor
        * ELECTRON_G_FACTOR
        * trace(&spin_density, &crate::integrals::kinetic_matrix(basis));
    let mut gauge_correction = [[0.0; 3]; 3];
    for (atom, charge) in molecule.atoms.iter().zip(soc_charges) {
        let mixed = super::nmr::shielding_mixed_operator(basis, atom.position, origin, false);
        for field in 0..3 {
            for spin in 0..3 {
                gauge_correction[field][spin] +=
                    factor * charge * trace(&spin_density, &mixed[field][spin]);
            }
        }
    }
    // Tr[(i dP_hat)(i L_hat)] = -Tr[dP_hat L_hat]. The minus
    // sign converts the energy cross susceptibility to the conventional
    // g-tensor spin-orbit contribution, without changing the Hessian sign.
    let spin_orbit = std::array::from_fn(|field| {
        std::array::from_fn(|spin| -factor * response.susceptibility[3 + spin][field].re)
    });
    let tensor = std::array::from_fn(|field| {
        std::array::from_fn(|spin| {
            (if field == spin {
                ELECTRON_G_FACTOR + mass_correction
            } else {
                0.0
            }) + gauge_correction[field][spin]
                + spin_orbit[field][spin]
        })
    });
    Ok(CommonOrbitalGTensor {
        tensor,
        mass_correction,
        gauge_correction,
        spin_orbit,
        response_residuals: response.residuals,
        reference_stationarity: response.reference_stationarity,
        metric_defect: response.metric_defect,
    })
}
