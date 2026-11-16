// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete nuclear/magnetic first-order pair-wavefunction axial tensors.
//! Nuclear and field paths reuse the generic parent and amplitude solvers.
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::integrals::london::{ComplexMatrix, FieldPath};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::properties::nmr::gradient_operator;
use crate::properties::pt2_axial::{PairAxialState, PairStateTangent};
use crate::response::complex::Complex;
use crate::response::correlated_field::correlated_field_taylor;
use crate::response::nuclear::{correlated_molecular_taylor, MolecularDirection};
use crate::response::{ResponseReport, ResponseSolverConfig};
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::double_hybrid::Pt2Coefficients;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

pub struct CorrelatedAxial {
    pub response: ResponseReport<Vec<[f64; 3]>>,
    pub parent: Vec<[f64; 3]>,
    /// Unscaled intermediate-normalized derivative-state channel bilinears,
    /// not additive contributions to the normalized final tensor.
    pub channel_bilinears: [Vec<[f64; 3]>; 3],
    pub norm: f64,
    pub norm_derivatives: Vec<f64>,
    pub magnetic_connections: Vec<[f64; 3]>,
    pub connection_defects: Vec<f64>,
    pub cancellation_defects: Vec<f64>,
    pub amplitude_residuals: Vec<f64>,
    pub orbital_frame_defects: Vec<f64>,
}

struct MagneticTangent {
    connection: [ComplexMatrix; 2],
    amplitudes: [Vec<Complex>; 3],
}

/// All `3N x 3` electronic AAT entries of the normalized first-order pair
/// state. The physical MO nuclear connection includes the ket-basis half
/// derivative; the common-origin magnetic basis has no derivative outside
/// the AO span. Distributed origins select each nucleus's own common
/// origin, without pretending that these are London magnetic derivatives.
/// No finite differences enter the production tensor.
#[allow(clippy::too_many_arguments)]
pub fn correlated_molecular_axial_response(
    molecule: &Molecule,
    basis: &MolBasis,
    parent: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    options: &ScfOptions,
    coefficients: Pt2Coefficients,
    origin: Vec3,
    distributed: bool,
    memory_mb: usize,
    config: &ResponseSolverConfig,
) -> Result<CorrelatedAxial> {
    coefficients.validate()?;
    let n = basis.nao_sph;
    let dimensions = reference.mo_energy.each_ref().map(Vec::len);
    let mut residuals = Vec::new();
    let mut amplitude_residuals = Vec::new();
    let mut orbital_frame_defects = Vec::new();
    let mut background: Option<[Vec<f64>; 3]> = None;
    let count = if distributed { molecule.len() } else { 1 };
    let mut fields = Vec::with_capacity(count);
    for centre in 0..count {
        let gauge = if distributed {
            molecule.atoms[centre].position
        } else {
            origin
        };
        let mut axes = Vec::with_capacity(3);
        for b in 0..3 {
            let mut direction = [0.0; 3];
            direction[b] = 1.0;
            let value = correlated_field_taylor(
                molecule,
                basis,
                parent,
                None,
                grid,
                reference,
                options,
                coefficients,
                &FieldPath {
                    electric: [0.0; 3],
                    magnetic: direction,
                    origin: gauge.to_array(),
                    london: false,
                    order: 1,
                },
                None,
                memory_mb,
                config,
            )?;
            residuals.extend(&value.parent.parent.residuals);
            residuals.extend(&value.parent.parent.projector_defects);
            residuals.extend(&value.parent.parent.stationarity_defects);
            residuals.push(value.parent.parent.reference_defect);
            amplitude_residuals.extend(&value.amplitude_residuals);
            orbital_frame_defects.push(value.parent.orbital_frame_defect);
            let base = value.amplitudes[0]
                .each_ref()
                .map(|v| v.iter().map(|z| z.re).collect::<Vec<_>>());
            if let Some(background) = &background {
                let defect = background
                    .iter()
                    .flatten()
                    .zip(base.iter().flatten())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                residuals.push(defect);
            } else {
                background = Some(base);
            }
            let connection = std::array::from_fn(|s| {
                let left = reference.mo_coeff[s].transpose_matmul(&reference.overlap);
                ComplexMatrix {
                    re: left.matmul(&value.parent.orbitals[s][1].re),
                    im: left.matmul(&value.parent.orbitals[s][1].im),
                }
            });
            axes.push(MagneticTangent {
                connection,
                amplitudes: value.amplitudes[1].clone(),
            });
        }
        fields.push(axes);
    }
    let background = background.ok_or_else(|| {
        DftError::InvalidInput("an axial response requires at least one nucleus".into())
    })?;
    let occupied = [reference.n_alpha, reference.n_beta];
    let state = PairAxialState::with_memory_limit(dimensions, occupied, &background, memory_mb)?;
    let gradient = gradient_operator(basis);
    let owner = basis.sph_ao_atoms();
    let dim = 3 * molecule.len();
    let mut tensors = vec![[0.0; 3]; dim];
    let mut parent_tensor = tensors.clone();
    let mut channel_bilinears = std::array::from_fn(|_| tensors.clone());
    let mut norm = 1.0;
    let mut norm_derivatives = vec![0.0; dim];
    let mut magnetic_connections = tensors.clone();
    let mut connection_defects = Vec::new();
    let mut cancellation_defects = Vec::new();
    for x in 0..dim {
        let (atom, axis) = (x / 3, x % 3);
        let mut direction = vec![[0.0; 3]; molecule.len()];
        direction[atom][axis] = 1.0;
        let nuclear = correlated_molecular_taylor(
            molecule,
            basis,
            parent,
            None,
            grid,
            reference,
            options,
            coefficients,
            &MolecularDirection::nuclear(direction),
            1,
            config,
        )?;
        residuals.extend(&nuclear.response_residuals);
        residuals.extend(&nuclear.projector_defects);
        residuals.extend(&nuclear.stationarity_defects);
        residuals.extend(&nuclear.metric_defects);
        residuals.extend(nuclear.number_defects.iter().flatten());
        residuals.push(nuclear.reference_energy_defect);
        residuals.push(nuclear.reference_fock_defect);
        amplitude_residuals.extend(&nuclear.amplitude_residuals);
        orbital_frame_defects.push(nuclear.orbital_frame_defect);
        let background_defect = nuclear.amplitudes[0]
            .iter()
            .flatten()
            .zip(background.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        residuals.push(background_defect);
        let mut half = Matrix::zeros(n, n);
        for mu in 0..n {
            for nu in 0..n {
                if owner[nu] == atom {
                    half[(mu, nu)] = -gradient[axis][(mu, nu)];
                }
            }
        }
        let connection = std::array::from_fn(|s| {
            let c = &reference.mo_coeff[s];
            let mut k = c
                .transpose_matmul(&reference.overlap)
                .matmul(&nuclear.orbitals[s][1]);
            let basis = c.transpose_matmul(&half).matmul(c);
            for (a, b) in k.as_mut_slice().iter_mut().zip(basis.as_slice()) {
                *a += b;
            }
            ComplexMatrix {
                re: k,
                im: Matrix::zeros(dimensions[s], dimensions[s]),
            }
        });
        let amplitudes = nuclear.amplitudes[1]
            .each_ref()
            .map(|v| v.iter().copied().map(Complex::from).collect());
        for b in 0..3 {
            let magnetic = &fields[if distributed { atom } else { 0 }][b];
            let value = state.overlap(
                &PairStateTangent {
                    connection: &connection,
                    amplitudes: &amplitudes,
                },
                &PairStateTangent {
                    connection: &magnetic.connection,
                    amplitudes: &magnetic.amplitudes,
                },
                coefficients,
                config.acceptance(),
            )?;
            tensors[x][b] = value.value;
            parent_tensor[x][b] = value.parent;
            for c in 0..3 {
                channel_bilinears[c][x][b] = value.channel_bilinears[c];
            }
            norm = value.norm;
            norm_derivatives[x] = value.norm_derivative;
            magnetic_connections[x][b] = value.magnetic_connection;
            connection_defects.push(value.connection_defect);
            cancellation_defects.push(value.cancellation_defect);
        }
    }
    let converged = residuals
        .iter()
        .chain(&amplitude_residuals)
        .chain(&orbital_frame_defects)
        .chain(&connection_defects)
        .chain(&cancellation_defects)
        .all(|r| r.is_finite() && r.abs() <= config.acceptance());
    Ok(CorrelatedAxial {
        response:ResponseReport {
            value:tensors,shape:vec![molecule.len(),3,3],functional:parent.description.clone(),
            reference:crate::response::diagnostics::reference_diagnostics(reference)?,
            units:"e*bohr/hbar",axis_convention:"3*nucleus + nuclear axis, magnetic field axis; imaginary overlap",
            model:"normalized first-order pair-wavefunction AAT; complete parent orbital and PT2 amplitude derivatives; nuclear ket-basis connection; common-origin or atom-distributed common origins; current-invariant meta-GGA magnetic kernel; integer collinear populations",
            converged,residuals,iterations:Vec::new(),warnings:vec!["This is the normalized MP1 wavefunction AAT model; London distributed-basis derivatives and spin-Zeeman terms are separate models.".into()],
        },parent:parent_tensor,channel_bilinears,norm,norm_derivatives,magnetic_connections,
        connection_defects,cancellation_defects,amplitude_residuals,orbital_frame_defects,
    })
}
