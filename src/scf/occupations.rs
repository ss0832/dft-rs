// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Functional-independent orbital occupation constraints in the physical AO metric.
//! State selection changes the occupied subspace, never the energy functional.
use super::{run_scf_impl, ScfOptions, ScfResult};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::Matrix;
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

/// Prescribed spin-orbital populations in ascending physical Fock energy order.
/// Trailing omitted columns have zero population. The orbital coefficients
/// relax at fixed populations; no thermal or ensemble-XC weight optimization
/// is implied by applying an ordinary density functional to this density.
#[derive(Clone, Debug)]
pub struct FixedOccupations {
    pub populations: [Vec<f64>; 2],
}

impl FixedOccupations {
    pub(crate) fn validated(
        &self,
        rank: usize,
        counts: [usize; 2],
        restricted: bool,
    ) -> Result<[Vec<f64>; 2]> {
        let mut populations = self.populations.clone();
        for spin in 0..2 {
            let f = &mut populations[spin];
            if f.len() > rank
                || f.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                || (f.iter().sum::<f64>() - counts[spin] as f64).abs() > 1e-10
            {
                return Err(DftError::InvalidInput(
                    "fixed populations must lie in [0,1], fit the retained MO rank and sum to the molecular electron count of each spin".into(),
                ));
            }
            f.resize(rank, 0.0);
        }
        if restricted && populations[0] != populations[1] {
            return Err(DftError::InvalidInput(
                "different alpha/beta fixed populations require an unrestricted reference".into(),
            ));
        }
        Ok(populations)
    }
}

/// Which occupied projector defines the next maximum-overlap selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverlapReference {
    /// Projector MOM: use the orbitals selected on the previous SCF cycle.
    Previous,
    /// Projector IMOM: retain the explicitly supplied initial occupied space.
    Initial,
}
impl OverlapReference {
    pub fn id(self) -> &'static str {
        match self {
            Self::Previous => "mom",
            Self::Initial => "imom",
        }
    }
}

/// Zero-based columns of a converged reference, separately for alpha/beta.
/// Each list describes the complete occupied subspace, including core orbitals.
#[derive(Clone, Debug)]
pub struct DeterminantOccupation {
    pub occupied: [Vec<usize>; 2],
    pub reference: OverlapReference,
}

#[derive(Clone, Debug)]
pub struct OccupationDiagnostics {
    pub mode: OverlapReference,
    pub initial_energy: f64,
    pub occupied_indices: [Vec<usize>; 2],
    /// Squared projections onto the reference occupied subspace, without floors.
    pub overlap_weights: [Vec<f64>; 2],
    /// Last occupied minus first omitted projector weight; None at an empty
    /// or full spin space. A zero gap identifies a genuinely tied selection.
    pub boundary_gaps: [Option<f64>; 2],
    /// Maximum element of the physical-Fock selected density map in the
    /// orthonormal AO frame, measured independently of the DIIS commutator.
    pub density_residual: f64,
}

fn columns(c: &Matrix, indices: &[usize]) -> Matrix {
    let mut selected = Matrix::zeros(c.rows, indices.len());
    for i in 0..c.rows {
        for (j, &k) in indices.iter().enumerate() {
            selected[(i, j)] = c[(i, k)];
        }
    }
    selected
}
pub(crate) struct SelectedOrbitals {
    pub coefficients: Matrix,
    pub energies: Vec<f64>,
    pub weights: Vec<f64>,
    pub gap: Option<f64>,
}
pub(crate) struct OccupationTracker {
    config: DeterminantOccupation,
    targets: [Matrix; 2],
    diagnostics: OccupationDiagnostics,
}
impl OccupationTracker {
    fn new(config: &DeterminantOccupation, state: &ScfResult) -> Result<Self> {
        if !state.converged
            || state.electronic_temperature != 0.0
            || !state.has_integer_occupations()
            || !state.energy.is_finite()
            || state.overlap.as_slice().iter().any(|x| !x.is_finite())
        {
            return Err(DftError::InvalidInput(
                "determinant tracking needs a converged zero-temperature reference".into(),
            ));
        }
        for spin in 0..2 {
            let c = &state.mo_coeff[spin];
            let indices = &config.occupied[spin];
            if c.rows != state.overlap.rows
                || c.cols != state.mo_energy[spin].len()
                || indices.iter().any(|&i| i >= c.cols)
                || indices
                    .iter()
                    .enumerate()
                    .any(|(i, index)| indices[..i].contains(index))
                || c.as_slice().iter().any(|x| !x.is_finite())
            {
                return Err(DftError::InvalidInput("occupied indices must be distinct valid reference columns in the same finite AO metric".into()));
            }
            let metric = c.transpose_matmul(&state.overlap.matmul_seq(c));
            if metric.max_abs_diff(&Matrix::identity(c.cols)) > 1e-8 {
                return Err(DftError::InvalidInput(
                    "occupation reference orbitals must be orthonormal in the AO overlap".into(),
                ));
            }
        }
        Ok(Self {
            config: config.clone(),
            targets: std::array::from_fn(|s| columns(&state.mo_coeff[s], &config.occupied[s])),
            diagnostics: OccupationDiagnostics {
                mode: config.reference,
                initial_energy: state.energy,
                occupied_indices: config.occupied.clone(),
                overlap_weights: std::array::from_fn(|_| Vec::new()),
                boundary_gaps: [None; 2],
                density_residual: f64::INFINITY,
            },
        })
    }
    pub(crate) fn counts(&self) -> [usize; 2] {
        self.config.occupied.each_ref().map(|v| v.len())
    }
    pub(crate) fn seed(&self) -> [Matrix; 2] {
        self.targets.each_ref().map(|c| c.matmul_transpose_seq(c))
    }
    pub(crate) fn preview(
        &self,
        spin: usize,
        c: &Matrix,
        energies: &[f64],
        overlap: &Matrix,
    ) -> Result<SelectedOrbitals> {
        if spin >= 2
            || c.rows != overlap.rows
            || overlap.rows != overlap.cols
            || c.cols != energies.len()
            || self.targets[spin].rows != c.rows
            || self.targets[spin].cols > c.cols
            || c.as_slice().iter().chain(energies).any(|x| !x.is_finite())
        {
            return Err(DftError::InvalidInput(
                "overlap selection needs matching finite AO orbitals and energies".into(),
            ));
        }
        let projection = self.targets[spin].transpose_matmul(&overlap.matmul_seq(c));
        let mut weights: Vec<_> = (0..c.cols)
            .map(|j| {
                (
                    (0..projection.rows)
                        .map(|i| projection[(i, j)].powi(2))
                        .sum::<f64>(),
                    j,
                )
            })
            .collect();
        weights.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let n = self.targets[spin].cols;
        let gap = if n > 0 && n < c.cols {
            Some(weights[n - 1].0 - weights[n].0)
        } else {
            None
        };
        let mut chosen: Vec<_> = weights[..n].iter().map(|v| v.1).collect();
        chosen.sort_unstable();
        let selected_weights = chosen
            .iter()
            .map(|&j| weights.iter().find(|w| w.1 == j).unwrap().0)
            .collect();
        let order: Vec<_> = chosen
            .iter()
            .copied()
            .chain((0..c.cols).filter(|j| !chosen.contains(j)))
            .collect();
        Ok(SelectedOrbitals {
            coefficients: columns(c, &order),
            energies: order.iter().map(|&j| energies[j]).collect(),
            weights: selected_weights,
            gap,
        })
    }
    pub(crate) fn commit(&mut self, spin: usize, selection: &SelectedOrbitals) {
        self.diagnostics.overlap_weights[spin] = selection.weights.clone();
        self.diagnostics.boundary_gaps[spin] = selection.gap;
        if self.config.reference == OverlapReference::Previous {
            self.targets[spin] = columns(
                &selection.coefficients,
                &(0..self.targets[spin].cols).collect::<Vec<_>>(),
            );
        }
    }
    pub(crate) fn set_residual(&mut self, residual: f64) {
        self.diagnostics.density_residual = residual;
    }
}

/// Relax an explicitly selected determinant using the ordinary variational
/// SCF energy/Fock builder. The overlap score is sum_i |C_occ^T S C_j|^2;
/// it is invariant under occupied-space rotations and orbital signs. Returned
/// orbitals place the occupied block first, with energy order inside each block.
/// This is a state-specific delta-SCF determinant, not a spin projection or
/// a TDDFT excitation. Reference and current geometry must share their AO metric.
pub fn run_scf_with_occupations(
    mol: &Molecule,
    basis: &MolBasis,
    options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    occupation: &DeterminantOccupation,
) -> Result<ScfResult> {
    if options.electronic_temperature != 0.0
        || options.electron_offset != [0.0; 2]
        || options.fixed_occupations.is_some()
    {
        return Err(DftError::InvalidInput("integer determinant tracking cannot be combined with Fermi smearing, fixed populations or fractional electron offsets".into()));
    }
    let overlap = crate::integrals::overlap_matrix(basis);
    if overlap.rows != reference.overlap.rows || overlap.max_abs_diff(&reference.overlap) > 1e-10 {
        return Err(DftError::InvalidInput("occupation reference and current basis must have the same AO overlap; geometry transport requires cross-overlap integrals".into()));
    }
    let mut tracker = OccupationTracker::new(occupation, reference)?;
    let seed = tracker.seed();
    if options.reference != super::Reference::Uhf
        && occupation.occupied[0].len() == occupation.occupied[1].len()
        && seed[0].max_abs_diff(&seed[1]) > 1e-10
    {
        return Err(DftError::InvalidInput(
            "different alpha/beta occupied subspaces require an unrestricted reference".into(),
        ));
    }
    let mut result = run_scf_impl(
        mol,
        basis,
        options,
        functional,
        grid,
        Some(&seed),
        Some(&mut tracker),
    )?;
    result.occupation_tracking = Some(tracker.diagnostics);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(
        reference: OverlapReference,
        occupied: usize,
    ) -> (OccupationTracker, Matrix, Matrix) {
        let s = Matrix::from_row_major(3, 3, vec![2.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.7]);
        let mut c = Matrix::identity(3);
        for i in 0..3 {
            c[(i, i)] = s[(i, i)].sqrt().recip();
        }
        let indices: Vec<_> = (0..occupied).collect();
        let tracker = OccupationTracker {
            config: DeterminantOccupation {
                occupied: [indices.clone(), indices.clone()],
                reference,
            },
            targets: [columns(&c, &indices), columns(&c, &indices)],
            diagnostics: OccupationDiagnostics {
                mode: reference,
                initial_energy: 0.0,
                occupied_indices: [indices.clone(), indices],
                overlap_weights: [Vec::new(), Vec::new()],
                boundary_gaps: [None; 2],
                density_residual: f64::INFINITY,
            },
        };
        (tracker, s, c)
    }
    fn rotate(c: &Matrix, angle: f64, i: usize, j: usize) -> Matrix {
        let mut r = Matrix::identity(c.cols);
        let (s, co) = angle.sin_cos();
        r[(i, i)] = co;
        r[(j, j)] = co;
        r[(i, j)] = -s;
        r[(j, i)] = s;
        c.matmul_seq(&r)
    }

    #[test]
    fn previous_and_initial_projectors_select_different_followed_branches() {
        let (mut previous, s, c) = fixture(OverlapReference::Previous, 1);
        let (mut initial, _, _) = fixture(OverlapReference::Initial, 1);
        let first = rotate(&c, 40.0_f64.to_radians(), 0, 1);
        let second = rotate(&c, 80.0_f64.to_radians(), 0, 1);
        let energies = [0.0, 1.0, 2.0];
        for tracker in [&mut previous, &mut initial] {
            let selection = tracker.preview(0, &first, &energies, &s).unwrap();
            assert_eq!(selection.energies[0], 0.0);
            tracker.commit(0, &selection);
        }
        assert_eq!(
            previous
                .preview(0, &second, &energies, &s)
                .unwrap()
                .energies[0],
            0.0
        );
        assert_eq!(
            initial.preview(0, &second, &energies, &s).unwrap().energies[0],
            1.0
        );
    }

    #[test]
    fn projector_selection_preserves_metric_and_occupied_rotation_invariance() {
        let (mut tracker, s, c) = fixture(OverlapReference::Initial, 2);
        let candidate = rotate(&c, 0.43, 0, 2);
        let energy = [-1.2, -0.2, 0.8];
        let before = tracker.preview(0, &candidate, &energy, &s).unwrap();
        tracker.targets[0] = rotate(&tracker.targets[0], 0.79, 0, 1);
        let mut signed = candidate.clone();
        for i in 0..signed.rows {
            signed[(i, 0)] *= -1.0;
        }
        let after = tracker.preview(0, &signed, &energy, &s).unwrap();
        assert_eq!(before.energies, after.energies);
        for (a, b) in before.weights.iter().zip(&after.weights) {
            assert!((a - b).abs() < 1e-14);
        }
        let metric = after
            .coefficients
            .transpose_matmul(&s.matmul_seq(&after.coefficients));
        assert!(metric.max_abs_diff(&Matrix::identity(3)) < 1e-14);
    }
}
