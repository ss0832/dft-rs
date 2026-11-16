// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Frozen occupied-space reference in the full molecular AO basis. Besides
//! generation diagnostics, the explicit experimental frozen-scalar-action-v1
//! model retains these full endpoint evaluators. It is not a compressed
//! reduced potential and has no claimed computational compression advantage.
//!
//! Restricted compatibility APIs use spin-summed P_B=2 B B^T and P_A=2 A A^T.
//! Collinear APIs keep independent unit-occupation alpha/beta projectors.
//! Active and frozen spaces are S-orthogonal within each spin. Both Focks
//! differentiate the same scalar at total active-plus-frozen spin densities.

mod collinear;
mod scf;
pub use collinear::{CollinearActiveSpaceScf, CollinearFrozenEvaluation, CollinearFrozenScf};

use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::integrals::{
    dipole_matrices, kinetic_matrix, nuclear_matrix, overlap_matrix, CoulombKernel,
};
use crate::ks::xc_contribution;
use crate::linalg::{symmetric_eigen, Matrix, Orthogonalizer};
use crate::math::Vec3;
use crate::properties::natural_bond::{natural_bond_analysis, NboKind, NboOptions, NboSpin};
use crate::response::diagnostics::reference_diagnostics;
use crate::scf::diis::Diis;
use crate::scf::{JkEngine, Reference, ScfOptions, ScfResult};
use crate::system::{Atom, Molecule};
use crate::xc::{Family, ResolvedFunctional};
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct MethylBoundary {
    pub carbon: usize,
    pub hydrogens: [usize; 3],
    /// Explicit retained carbon at the C(sp3)-Me attachment port.
    pub port_atom: usize,
}

#[derive(Clone, Debug)]
pub struct ExtractionOptions {
    /// Fixed Lewis threshold; no threshold search is performed along a path.
    pub lewis_threshold: f64,
    pub reference_residual_tolerance: f64,
    pub subspace_tolerance: f64,
    pub maximum_external_population: f64,
}

impl Default for ExtractionOptions {
    fn default() -> Self {
        Self {
            lewis_threshold: 0.95,
            reference_residual_tolerance: 1e-11,
            subspace_tolerance: 1e-9,
            maximum_external_population: 0.2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FrozenSpaceDiagnostics {
    pub selected_parent_labels: Vec<String>,
    pub excluded_port_label: String,
    pub selected_populations: Vec<f64>,
    pub external_population_fractions: Vec<f64>,
    pub occupied_reconstruction_error: f64,
    pub orbital_orthonormality_error: f64,
    pub frozen_idempotency_error: f64,
    pub active_idempotency_error: f64,
    pub active_frozen_overlap: f64,
    pub frozen_electron_count: f64,
    pub active_electron_count: f64,
    pub nlmo_residual: f64,
    pub lewis_threshold: f64,
    /// A candidate can be inspected, but cannot enter the reference solver.
    pub ambiguous: bool,
}

/// Frozen and active reference SUBSPACES. Columns inside each space may be
/// freely rotated; neither MO number nor sign defines the partition.
#[derive(Clone)]
pub struct FrozenSelection {
    pub boundary: MethylBoundary,
    pub frozen_orbitals: Matrix,
    pub active_reference_orbitals: Matrix,
    pub frozen_density: Matrix,
    pub active_reference_density: Matrix,
    pub diagnostics: FrozenSpaceDiagnostics,
    pub full_electrons: usize,
    pub frozen_electrons: usize,
    pub active_electrons: usize,
    pub ionic_charge: i32,
    pub full_reference_energy: f64,
    pub full_reference_fock: Matrix,
    signature: u64,
}

fn invalid<T>(message: impl Into<String>) -> Result<T> {
    Err(DftError::InvalidInput(message.into()))
}

fn largest(matrix: &Matrix) -> f64 {
    matrix
        .as_slice()
        .iter()
        .fold(0.0_f64, |v, x| v.max(x.abs()))
}

fn add(target: &mut Matrix, source: &Matrix, factor: f64) {
    for (a, &b) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
        *a += factor * b;
    }
}

fn scaled(source: &Matrix, factor: f64) -> Matrix {
    let mut result = source.clone();
    for x in result.as_mut_slice() {
        *x *= factor;
    }
    result
}

fn columns(matrix: &Matrix, indices: &[usize]) -> Matrix {
    let mut result = Matrix::zeros(matrix.rows, indices.len());
    for (j, &index) in indices.iter().enumerate() {
        for i in 0..matrix.rows {
            result[(i, j)] = matrix[(i, index)];
        }
    }
    result
}

fn metric_gram(c: &Matrix, s: &Matrix) -> Matrix {
    c.transpose_matmul_seq(&s.matmul_seq(c))
}

fn density(c: &Matrix, occupied: usize) -> Matrix {
    let occupied = columns(c, &(0..occupied).collect::<Vec<_>>());
    scaled(&occupied.matmul_transpose_seq(&occupied), 2.0)
}

fn idempotency(p: &Matrix, s: &Matrix) -> f64 {
    let mut error = p.matmul_seq(&s.matmul_seq(p));
    add(&mut error, p, -2.0);
    largest(&error)
}

fn finite_symmetric(p: &Matrix, n: usize) -> bool {
    p.rows == n
        && p.cols == n
        && p.as_slice().iter().all(|x| x.is_finite())
        && p.max_abs_diff(&p.transpose()) < 1e-10
}

fn signature(molecule: &Molecule, basis: &MolBasis) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    let mut feed = |value: u64| {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    };
    feed(molecule.len() as u64);
    feed(molecule.charge as u64);
    feed(molecule.multiplicity as u64);
    for atom in &molecule.atoms {
        feed(u64::from(atom.z));
        for x in atom.position.to_array() {
            feed(x.to_bits());
        }
    }
    feed(basis.nao_sph as u64);
    for shell in &basis.shells {
        feed(u64::from(shell.l));
        feed(shell.atom as u64);
        feed(shell.sph_off as u64);
        for x in shell.center.to_array() {
            feed(x.to_bits());
        }
        feed(shell.exps.len() as u64);
        for &x in shell.exps.iter().chain(&shell.coefs) {
            feed(x.to_bits());
        }
    }
    hash
}

/// Select C 1s and the three explicitly named internal C-H occupied NLMOs.
/// The boundary C-C occupied orbital remains active. Ambiguous populations
/// are returned as diagnostics and are refused by the SCF constructor.
pub fn extract_methyl_frozen_space(
    molecule: &Molecule,
    basis: &MolBasis,
    reference: &ScfResult,
    boundary: MethylBoundary,
    options: &ExtractionOptions,
) -> Result<FrozenSelection> {
    if !options.lewis_threshold.is_finite()
        || options.lewis_threshold <= 0.0
        || options.lewis_threshold > 1.0
        || !options.reference_residual_tolerance.is_finite()
        || options.reference_residual_tolerance <= 0.0
        || !options.subspace_tolerance.is_finite()
        || options.subspace_tolerance <= 0.0
        || !options.maximum_external_population.is_finite()
        || !(0.0..1.0).contains(&options.maximum_external_population)
    {
        return invalid("invalid frozen-space extraction thresholds");
    }
    let mut atoms = vec![boundary.carbon, boundary.port_atom];
    atoms.extend(boundary.hydrogens);
    let mut unique = atoms.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != 5
        || atoms.iter().any(|&a| a >= molecule.len())
        || molecule.atoms[boundary.carbon].z != 6
        || molecule.atoms[boundary.port_atom].z != 6
        || boundary.hydrogens.iter().any(|&a| molecule.atoms[a].z != 1)
    {
        return invalid("Me extraction requires one carbon, three distinct named hydrogens and a distinct retained carbon port");
    }
    if reference.unrestricted
        || reference.n_alpha != reference.n_beta
        || molecule.multiplicity != 1
        || !basis.ecps.is_empty()
        || basis.n_core_electrons != 0
        || !reference.has_integer_occupations()
    {
        return invalid(
            "initial Me extraction requires an all-electron closed-shell integer RKS reference",
        );
    }
    let diagnostics = reference_diagnostics(reference)?;
    if !reference.converged || diagnostics.orbital_residual > options.reference_residual_tolerance {
        return Err(DftError::ScfNotConverged {
            iterations: reference.iterations,
            error: diagnostics.orbital_residual,
        });
    }
    if reference.n_alpha + reference.n_beta != molecule.n_electrons()? || reference.n_alpha < 4 {
        return invalid(
            "full-reference electron count is inconsistent with the molecule or Me frozen space",
        );
    }
    let analysis = natural_bond_analysis(
        molecule,
        basis,
        reference,
        NboSpin::Total,
        &NboOptions {
            lewis_threshold: Some(options.lewis_threshold),
            ..NboOptions::default()
        },
    )?;
    if analysis.channels.len() != 1 {
        return invalid("closed-shell Me extraction requires one total-density NBO channel");
    }
    let channel = &analysis.channels[0];
    let orbitals = analysis
        .nao
        .nao_coefficients
        .matmul_seq(&channel.nlmo_coefficients);
    if orbitals.cols != reference.n_alpha {
        return invalid(
            "the selected natural Lewis structure does not span the full occupied space",
        );
    }
    let mut selected = Vec::new();
    let mut labels = Vec::new();
    let mut populations = Vec::new();
    let mut leakage = Vec::new();
    let mut port_label = None;
    let mut core_count = 0;
    let mut hydrogen_counts = [0usize; 3];
    for (column, nlmo) in channel.nlmos.iter().enumerate() {
        let nbo = channel
            .nbos
            .get(nlmo.parent)
            .ok_or_else(|| DftError::InvalidInput("NLMO parent index is invalid".into()))?;
        let is_core = nbo.kind == NboKind::Core && nbo.atoms == [boundary.carbon];
        let mut is_internal = false;
        if nbo.kind == NboKind::Bond && nbo.atoms.len() == 2 && nbo.atoms.contains(&boundary.carbon)
        {
            for (h, &atom) in boundary.hydrogens.iter().enumerate() {
                if nbo.atoms.contains(&atom) {
                    hydrogen_counts[h] += 1;
                    is_internal = true;
                }
            }
            if nbo.atoms.contains(&boundary.port_atom) {
                port_label = Some(nbo.label.clone());
            }
        }
        if is_core {
            core_count += 1;
        }
        if is_core || is_internal {
            selected.push(column);
            labels.push(nbo.label.clone());
            populations.push(nlmo.occupancy);
            let internal: f64 = std::iter::once(boundary.carbon)
                .chain(boundary.hydrogens)
                .map(|atom| nlmo.atomic_percentages[atom] / 100.0)
                .sum();
            leakage.push(1.0 - internal);
        }
    }
    if core_count != 1
        || hydrogen_counts != [1, 1, 1]
        || selected.len() != 4
        || port_label.is_none()
    {
        return invalid(format!("Me internal occupied-space selection is not unique: core={core_count}, C-H counts={hydrogen_counts:?}, frozen={}, boundary bond found={}", selected.len(), port_label.is_some()));
    }
    let active_indices: Vec<_> = (0..orbitals.cols)
        .filter(|i| !selected.contains(i))
        .collect();
    let frozen = columns(&orbitals, &selected);
    let active = columns(&orbitals, &active_indices);
    let frozen_density = density(&frozen, 4);
    let active_density = density(&active, active.cols);
    let reconstructed = density(&orbitals, orbitals.cols);
    let s = &reference.overlap;
    let reconstruction = reconstructed.max_abs_diff(&reference.p_total());
    let orthogonality = metric_gram(&orbitals, s).max_abs_diff(&Matrix::identity(orbitals.cols));
    let frozen_idempotency = idempotency(&frozen_density, s);
    let active_idempotency = idempotency(&active_density, s);
    let exclusion = largest(&frozen.transpose_matmul_seq(&s.matmul_seq(&active)));
    let frozen_count = frozen_density.frobenius_dot(s);
    let active_count = active_density.frobenius_dot(s);
    let ambiguous = leakage
        .iter()
        .any(|&p| p > options.maximum_external_population)
        || populations
            .iter()
            .any(|&p| (p - 2.0).abs() > options.subspace_tolerance)
        || [
            reconstruction,
            orthogonality,
            frozen_idempotency,
            active_idempotency,
            exclusion,
            (frozen_count - 8.0).abs(),
            (active_count - 2.0 * active.cols as f64).abs(),
        ]
        .iter()
        .any(|&e| !e.is_finite() || e > options.subspace_tolerance);
    Ok(FrozenSelection {
        boundary,
        frozen_orbitals: frozen,
        active_reference_orbitals: active,
        frozen_density,
        active_reference_density: active_density,
        diagnostics: FrozenSpaceDiagnostics {
            selected_parent_labels: labels,
            excluded_port_label: port_label.unwrap(),
            selected_populations: populations,
            external_population_fractions: leakage,
            occupied_reconstruction_error: reconstruction,
            orbital_orthonormality_error: orthogonality,
            frozen_idempotency_error: frozen_idempotency,
            active_idempotency_error: active_idempotency,
            active_frozen_overlap: exclusion,
            frozen_electron_count: frozen_count,
            active_electron_count: active_count,
            nlmo_residual: channel.nlmo_residual,
            lewis_threshold: channel.lewis_threshold,
            ambiguous,
        },
        full_electrons: reference.n_alpha + reference.n_beta,
        frozen_electrons: 8,
        active_electrons: 2 * active_indices.len(),
        ionic_charge: 1,
        full_reference_energy: reference.energy,
        full_reference_fock: reference.fock[0].clone(),
        signature: signature(molecule, basis),
    })
}

pub struct FrozenDensityEvaluation {
    pub energy: f64,
    pub electronic_energy: f64,
    pub one_electron_energy: f64,
    pub hartree_energy: f64,
    pub xc_energy: f64,
    pub nuclear_energy: f64,
    /// Density-independent reference removed from `energy`. Nuclear and
    /// electronic component energies keep their physical, unshifted meaning.
    pub reference_energy: f64,
    pub fock: Matrix,
    pub active_electrons: f64,
    pub frozen_electrons: f64,
    pub grid_total_electrons: f64,
}

/// Off-shell scalar density functional and its exact first density derivative.
/// The frozen density is fixed; the XC is evaluated on TOTAL density here.
/// Both generation and the explicit full frozen scalar model use this action.
/// It is not the independently compressed static potential.
pub struct FrozenDensityEvaluator<'a> {
    molecule: &'a Molecule,
    basis: &'a MolBasis,
    grid: &'a MolecularGrid,
    functional: &'a ResolvedFunctional,
    engine: JkEngine,
    overlap: Matrix,
    h_core: Matrix,
    frozen_density: Matrix,
    frozen_spin_density: [Matrix; 2],
    restricted: bool,
}

/// Fixed-AO electric perturbation. The same operator enters the scalar energy
/// and its density derivative; the frozen electrons also couple to the field.
struct FrozenElectricPerturbation {
    operator: Matrix,
    nuclear_energy: f64,
}

impl<'a> FrozenDensityEvaluator<'a> {
    pub fn overlap(&self) -> &Matrix {
        &self.overlap
    }
    pub fn h_core(&self) -> &Matrix {
        &self.h_core
    }
    pub fn frozen_density(&self) -> &Matrix {
        &self.frozen_density
    }
    pub fn new(
        molecule: &'a Molecule,
        basis: &'a MolBasis,
        grid: &'a MolecularGrid,
        functional: &'a ResolvedFunctional,
        options: &ScfOptions,
        frozen_density: &Matrix,
    ) -> Result<Self> {
        if molecule.multiplicity != 1 || matches!(options.reference, Reference::Uhf) {
            return invalid(
                "restricted frozen scalar requires a closed-shell restricted reference",
            );
        }
        Self::new_inner(
            molecule,
            basis,
            grid,
            functional,
            options,
            [scaled(frozen_density, 0.5), scaled(frozen_density, 0.5)],
            true,
        )
    }

    /// Collinear frozen action with independent integer spin projectors. No
    /// frozen polarization or target-state reconstruction occurs here.
    pub fn new_collinear(
        molecule: &'a Molecule,
        basis: &'a MolBasis,
        grid: &'a MolecularGrid,
        functional: &'a ResolvedFunctional,
        options: &ScfOptions,
        frozen_spin_density: [&Matrix; 2],
    ) -> Result<Self> {
        if matches!(options.reference, Reference::Rhf) {
            return invalid(
                "collinear frozen scalar requires an automatic or unrestricted reference",
            );
        }
        Self::new_inner(
            molecule,
            basis,
            grid,
            functional,
            options,
            [
                frozen_spin_density[0].clone(),
                frozen_spin_density[1].clone(),
            ],
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        molecule: &'a Molecule,
        basis: &'a MolBasis,
        grid: &'a MolecularGrid,
        functional: &'a ResolvedFunctional,
        options: &ScfOptions,
        frozen_spin_density: [Matrix; 2],
        restricted: bool,
    ) -> Result<Self> {
        if !basis.ecps.is_empty()
            || basis.n_core_electrons != 0
            || !matches!(functional.family, Family::Lda | Family::Gga)
            || functional.hybrid.cx != 0.0
            || functional.hybrid.clr != 0.0
            || functional.required_nonlocal_term().is_some()
            || options.electronic_temperature != 0.0
            || options.electron_offset != [0.0; 2]
            || options.occupation.is_some()
            || options.fixed_occupations.is_some()
            || options.extra_hcore.is_some()
            || options.external_field.is_some()
            || options.pcm.is_some()
            || options.vv10.is_some()
            || options.skala.is_some()
            || options.asymptotic_correction.is_some()
            || options.density_fit.is_some()
            || options.kernel != CoulombKernel::Coulomb
        {
            return invalid("frozen scalar requires isolated all-electron integer semilocal DFT without additional models");
        }
        if grid.is_empty()
            || frozen_spin_density
                .iter()
                .any(|p| !finite_symmetric(p, basis.nao_sph))
        {
            return invalid("frozen scalar requires finite symmetric spin densities and a nonempty molecular grid");
        }
        let (alpha, beta) = molecule.n_alpha_beta()?;
        let overlap = overlap_matrix(basis);
        // The legacy restricted checks act on P=2p. Preserve their absolute
        // thresholds when expressing the same projector in unit-spin form.
        let lower_tolerance = if restricted { 0.5e-10 } else { 1e-10 };
        let count_tolerance = if restricted { 0.5e-9 } else { 1e-9 };
        let projector_tolerance = if restricted { 0.25e-8 } else { 1e-8 };
        for (p, count) in frozen_spin_density.iter().zip([alpha, beta]) {
            let particles = p.frobenius_dot(&overlap);
            let mut defect = p.matmul_seq(&overlap.matmul_seq(p));
            add(&mut defect, p, -1.);
            if particles < -lower_tolerance
                || particles > count as f64 + count_tolerance
                || (particles - particles.round()).abs() > 1e-9
                || largest(&defect) > projector_tolerance
            {
                return invalid("each frozen spin density must be an integer occupied projector in the full AO metric");
            }
        }
        let mut frozen_density = frozen_spin_density[0].clone();
        add(&mut frozen_density, &frozen_spin_density[1], 1.);
        let mut core = kinetic_matrix(basis);
        add(&mut core, &nuclear_matrix(basis, molecule, None), 1.);
        Ok(Self {
            molecule,
            basis,
            grid,
            functional,
            engine: JkEngine::from_options(molecule, basis, options)?,
            overlap,
            h_core: core,
            frozen_density,
            frozen_spin_density,
            restricted,
        })
    }

    pub fn evaluate(&self, active_density: &Matrix) -> Result<FrozenDensityEvaluation> {
        self.evaluate_with_reference(active_density, 0.)
    }

    /// Remove a common density-independent energy reference in the same
    /// compensated accumulation as the electronic and nuclear terms. This
    /// avoids rounding a large total before evaluating a small active energy.
    /// It does not alter the Fock or any physical SCF stopping threshold.
    pub fn evaluate_with_reference(
        &self,
        active_density: &Matrix,
        reference_energy: f64,
    ) -> Result<FrozenDensityEvaluation> {
        if !self.restricted || !finite_symmetric(active_density, self.basis.nao_sph) {
            return invalid("spin-summed frozen evaluation requires a restricted scalar and finite symmetric active density");
        }
        let half = scaled(active_density, 0.5);
        let value = self.evaluate_collinear_with_reference([&half, &half], reference_energy)?;
        Ok(FrozenDensityEvaluation {
            energy: value.energy,
            electronic_energy: value.electronic_energy,
            one_electron_energy: value.one_electron_energy,
            hartree_energy: value.hartree_energy,
            xc_energy: value.xc_energy,
            nuclear_energy: value.nuclear_energy,
            reference_energy: value.reference_energy,
            fock: value.fock[0].clone(),
            active_electrons: value.active_electrons.iter().sum(),
            frozen_electrons: value.frozen_electrons.iter().sum(),
            grid_total_electrons: value.grid_total_electrons,
        })
    }

    /// Both spin Focks are first derivatives of this same compensated scalar.
    /// Densities use unit spin occupations; the saved reference is removed
    /// before the large total is rounded, including in unrestricted SCF.
    pub fn evaluate_collinear_with_reference(
        &self,
        active: [&Matrix; 2],
        reference_energy: f64,
    ) -> Result<CollinearFrozenEvaluation> {
        if !reference_energy.is_finite()
            || active
                .iter()
                .any(|p| !finite_symmetric(p, self.basis.nao_sph))
        {
            return invalid("collinear active densities and reference must be finite");
        }
        let spin: [Matrix; 2] = std::array::from_fn(|s| {
            let mut p = active[s].clone();
            add(&mut p, &self.frozen_spin_density[s], 1.);
            p
        });
        let mut total = spin[0].clone();
        add(&mut total, &spin[1], 1.);
        let (j, _) = self
            .engine
            .build(self.basis, &total, [&spin[0], &spin[1]], false)?;
        let xc = xc_contribution(self.functional, self.basis, self.grid, [&spin[0], &spin[1]])?;
        let mut one_sum = crate::linalg::AccurateSum::default();
        let mut hartree_sum = crate::linalg::AccurateSum::default();
        let mut energy_sum = crate::linalg::AccurateSum::default();
        for ((&p, &h), &j) in total
            .as_slice()
            .iter()
            .zip(self.h_core.as_slice())
            .zip(j.as_slice())
        {
            one_sum.add_product(p, h);
            hartree_sum.add_product(0.5 * p, j);
            energy_sum.add_product(p, h);
            energy_sum.add_product(0.5 * p, j);
        }
        energy_sum.add_product(1., xc.exc);
        let electronic = energy_sum.value();
        let nuclear = self.molecule.nuclear_repulsion();
        energy_sum.add_product(1., nuclear);
        energy_sum.add_product(-1., reference_energy);
        let fock = std::array::from_fn(|s| {
            let mut f = self.h_core.clone();
            add(&mut f, &j, 1.);
            add(&mut f, &xc.vxc[s], 1.);
            f
        });
        if !energy_sum.value().is_finite()
            || fock
                .iter()
                .any(|f| f.as_slice().iter().any(|x| !x.is_finite()))
        {
            return invalid("collinear frozen scalar produced a nonfinite energy or Fock");
        }
        Ok(CollinearFrozenEvaluation {
            energy: energy_sum.value(),
            electronic_energy: electronic,
            one_electron_energy: one_sum.value(),
            hartree_energy: hartree_sum.value(),
            xc_energy: xc.exc,
            nuclear_energy: nuclear,
            reference_energy,
            fock,
            active_electrons: std::array::from_fn(|s| active[s].frobenius_dot(&self.overlap)),
            frozen_electrons: std::array::from_fn(|s| {
                self.frozen_spin_density[s].frobenius_dot(&self.overlap)
            }),
            grid_total_electrons: xc.n_elec,
        })
    }

    fn electric_perturbation(
        &self,
        field: Vec3,
        origin: Vec3,
    ) -> Result<FrozenElectricPerturbation> {
        if field
            .to_array()
            .into_iter()
            .chain(origin.to_array())
            .any(|v| !v.is_finite())
        {
            return invalid("frozen-reference electric field and origin must be finite");
        }
        let mut operator = Matrix::zeros(self.basis.nao_sph, self.basis.nao_sph);
        for (axis, dipole) in dipole_matrices(self.basis, origin).iter().enumerate() {
            add(&mut operator, dipole, field.get(axis));
        }
        let nuclear_energy = self
            .molecule
            .atoms
            .iter()
            .map(|atom| -f64::from(atom.z) * (atom.position - origin).dot(field))
            .sum();
        Ok(FrozenElectricPerturbation {
            operator,
            nuclear_energy,
        })
    }

    fn evaluate_perturbed(
        &self,
        active_density: &Matrix,
        perturbation: Option<&FrozenElectricPerturbation>,
    ) -> Result<FrozenDensityEvaluation> {
        let mut evaluation = self.evaluate(active_density)?;
        if let Some(field) = perturbation {
            let electronic = active_density.frobenius_dot(&field.operator)
                + self.frozen_density.frobenius_dot(&field.operator);
            evaluation.one_electron_energy += electronic;
            evaluation.electronic_energy += electronic;
            evaluation.nuclear_energy += field.nuclear_energy;
            evaluation.energy += electronic + field.nuclear_energy;
            add(&mut evaluation.fock, &field.operator, 1.0);
            if !evaluation.energy.is_finite()
                || evaluation.fock.as_slice().iter().any(|v| !v.is_finite())
            {
                return invalid("frozen-reference electric perturbation is not finite");
            }
        }
        Ok(evaluation)
    }

    /// Fixed-geometry diagnostic energy on the total active-plus-frozen density.
    /// This does not regenerate or polarize the frozen occupied subspace.
    pub fn evaluate_in_electric_field(
        &self,
        active_density: &Matrix,
        field: Vec3,
        origin: Vec3,
    ) -> Result<FrozenDensityEvaluation> {
        let perturbation = self.electric_perturbation(field, origin)?;
        self.evaluate_perturbed(active_density, Some(&perturbation))
    }

    /// Total static dipole in e Bohr, the negative explicit field derivative.
    /// A stationary active solution is required for a relaxed energy derivative.
    pub fn electric_dipole(&self, active_density: &Matrix, origin: Vec3) -> Result<Vec3> {
        if !finite_symmetric(active_density, self.basis.nao_sph)
            || origin.to_array().iter().any(|v| !v.is_finite())
        {
            return invalid(
                "frozen-reference dipole requires a finite symmetric density and origin",
            );
        }
        let nuclear = self.molecule.atoms.iter().fold(Vec3::zero(), |sum, atom| {
            sum + (atom.position - origin) * f64::from(atom.z)
        });
        let dipoles = dipole_matrices(self.basis, origin);
        let electronic: [f64; 3] = std::array::from_fn(|axis| {
            active_density.frobenius_dot(&dipoles[axis])
                + self.frozen_density.frobenius_dot(&dipoles[axis])
        });
        Ok(nuclear - Vec3::new(electronic[0], electronic[1], electronic[2]))
    }
}

#[derive(Clone, Debug)]
pub struct FrozenIteration {
    pub iteration: usize,
    pub energy: f64,
    pub projected_orbital_residual: f64,
    pub density_change: Option<f64>,
    pub energy_change: Option<f64>,
}

pub struct FrozenReferenceScf {
    pub active_density: Matrix,
    pub active_orbitals: Matrix,
    pub orbital_energies: Vec<f64>,
    pub evaluation: FrozenDensityEvaluation,
    pub history: Vec<FrozenIteration>,
    pub projected_orbital_residual: f64,
    pub active_idempotency_error: f64,
    pub active_frozen_overlap: f64,
    pub particle_number_error: f64,
}

/// Exact fixed-B constrained reference. Five spatial orbitals are occupied for
/// the ethane example. It never changes the ordinary molecular SCF routine.
pub struct FrozenReferenceModel<'a> {
    evaluator: FrozenDensityEvaluator<'a>,
    complement: Matrix,
    active_electrons: usize,
    frozen_orbitals: Matrix,
    options: ScfOptions,
    selection: FrozenSelection,
}

/// Reusable constrained SCF geometry for a scalar active-density functional.
/// The callback returns both energy and its derivative in the same AO chart.
/// This is the existing frozen-reference solver, with unchanged stopping gates.
pub struct ActiveSpaceScf<'a> {
    pub overlap: &'a Matrix,
    pub complement: &'a Matrix,
    pub frozen_orbitals: &'a Matrix,
    pub active_electrons: usize,
    pub options: &'a ScfOptions,
    pub initial_hcore: &'a Matrix,
}

/// A full frozen scalar functional represented in one fixed raw AO chart.
/// Both energy and Fock retain the full active-plus-frozen density, including
/// nonlinear Hartree/XC terms. This is not a fit to a converged Fock matrix.
pub struct MappedFrozenAction<'a, 'b> {
    evaluator: &'a FrozenDensityEvaluator<'b>,
    chart: &'a super::reduction_support::OrthogonalReducedChart,
    reference_energy: f64,
}

impl<'a, 'b> MappedFrozenAction<'a, 'b> {
    /// `reference_energy` is a density-independent constant. A reaction must
    /// use the identical value for both occurrences of the same fragment;
    /// it cannot remove state-dependent internal strain or polarization.
    pub fn new(
        evaluator: &'a FrozenDensityEvaluator<'b>,
        chart: &'a super::reduction_support::OrthogonalReducedChart,
        reference_energy: f64,
    ) -> Result<Self> {
        let n = evaluator.overlap.rows;
        if !reference_energy.is_finite()
            || chart.transform.rows != n
            || chart.transform.cols != chart.raw_overlap.rows
            || !finite_symmetric(&chart.raw_overlap, chart.transform.cols)
            || chart.transform.as_slice().iter().any(|x| !x.is_finite())
        {
            return invalid(
                "mapped frozen action requires a compatible finite chart and reference constant",
            );
        }
        let mapped_metric = chart
            .transform
            .transpose_matmul_seq(&evaluator.overlap.matmul_seq(&chart.transform));
        let exclusion = evaluator
            .frozen_density
            .matmul_seq(&evaluator.overlap.matmul_seq(&chart.transform));
        if mapped_metric.max_abs_diff(&chart.raw_overlap) > 1e-9 || largest(&exclusion) > 1e-9 {
            return invalid("mapped frozen action metric or frozen-density exclusion differs from its evaluator");
        }
        Ok(Self {
            evaluator,
            chart,
            reference_energy,
        })
    }

    pub fn raw_overlap(&self) -> &Matrix {
        &self.chart.raw_overlap
    }

    /// Pull the Fock back with the map dual to the lifted density. The fixed
    /// reference is subtracted before this result enters the active SCF loop.
    pub fn evaluate(&self, raw_density: &Matrix) -> Result<FrozenDensityEvaluation> {
        let lifted = self.chart.lift_density(raw_density)?;
        let mut result = self
            .evaluator
            .evaluate_with_reference(&lifted, self.reference_energy)?;
        result.fock = self.chart.pullback_operator(&result.fock)?;
        validate_evaluation(&result, self.chart.transform.cols)?;
        Ok(result)
    }
}

/// Convex combination of density-dependent actions in a fixed common chart.
/// Endpoint maps and frozen projectors are fixed while weights vary. No chart
/// derivative is omitted: a moving-map parameterization is a different model.
/// Metric equality is checked; the caller must also supply the same ordered
/// physical raw AO functions, whose identity cannot be inferred from S alone.
pub struct WeightedFrozenAction<'a, 'b, 'c> {
    terms: &'a [MappedFrozenAction<'b, 'c>],
    weights: &'a [f64],
}

pub struct WeightedFrozenEvaluation {
    pub evaluation: FrozenDensityEvaluation,
    /// Each term is evaluated at this same density, before scalar weighting.
    pub term_energies: Vec<f64>,
}

impl WeightedFrozenEvaluation {
    /// Envelope-theorem derivatives for w0=1-sum(q), wi=q[i-1]. A stationary
    /// common density is required for a relaxed derivative of the SCF energy.
    pub fn independent_weight_gradient(&self) -> Vec<f64> {
        self.term_energies[1..]
            .iter()
            .map(|e| e - self.term_energies[0])
            .collect()
    }
}

impl<'a, 'b, 'c> WeightedFrozenAction<'a, 'b, 'c> {
    pub fn new(terms: &'a [MappedFrozenAction<'b, 'c>], weights: &'a [f64]) -> Result<Self> {
        if terms.is_empty()
            || terms.len() != weights.len()
            || weights.iter().any(|w| !w.is_finite() || *w < 0.)
            || (weights.iter().sum::<f64>() - 1.).abs() > 1e-12
        {
            return invalid("weighted frozen action requires matching nonempty terms and finite simplex weights");
        }
        let overlap = terms[0].raw_overlap();
        if terms.iter().any(|term| {
            let other = term.raw_overlap();
            other.rows != overlap.rows || other.max_abs_diff(overlap) > 1e-10
        }) {
            return invalid("weighted frozen action terms must use the same fixed raw AO metric");
        }
        Ok(Self { terms, weights })
    }

    /// Evaluate full scalar energies and their consistently weighted Focks.
    /// Frozen-electron diagnostics are weighted model quantities; the active
    /// determinant retains its separate integer electron count in SCF.
    pub fn evaluate(&self, density: &Matrix) -> Result<WeightedFrozenEvaluation> {
        self.evaluate_selected(density, true)
    }

    /// SCF needs only the weighted scalar and Fock. An exactly zero weight
    /// contributes neither, so avoid evaluating that full reference system.
    /// Use `evaluate` at the stationary density for all weight derivatives.
    pub fn evaluate_scalar(&self, density: &Matrix) -> Result<FrozenDensityEvaluation> {
        Ok(self.evaluate_selected(density, false)?.evaluation)
    }

    fn evaluate_selected(
        &self,
        density: &Matrix,
        retain_terms: bool,
    ) -> Result<WeightedFrozenEvaluation> {
        let n = self.terms[0].raw_overlap().rows;
        let mut result = FrozenDensityEvaluation {
            energy: 0.,
            electronic_energy: 0.,
            one_electron_energy: 0.,
            hartree_energy: 0.,
            xc_energy: 0.,
            nuclear_energy: 0.,
            reference_energy: 0.,
            fock: Matrix::zeros(n, n),
            active_electrons: 0.,
            frozen_electrons: 0.,
            grid_total_electrons: 0.,
        };
        let mut term_energies = Vec::with_capacity(self.terms.len());
        for (term, &weight) in self.terms.iter().zip(self.weights) {
            if weight == 0. && !retain_terms {
                continue;
            }
            let value = term.evaluate(density)?;
            if retain_terms {
                term_energies.push(value.energy);
            }
            result.energy += weight * value.energy;
            result.electronic_energy += weight * value.electronic_energy;
            result.one_electron_energy += weight * value.one_electron_energy;
            result.hartree_energy += weight * value.hartree_energy;
            result.xc_energy += weight * value.xc_energy;
            result.nuclear_energy += weight * value.nuclear_energy;
            result.reference_energy += weight * value.reference_energy;
            result.active_electrons += weight * value.active_electrons;
            result.frozen_electrons += weight * value.frozen_electrons;
            result.grid_total_electrons += weight * value.grid_total_electrons;
            add(&mut result.fock, &value.fock, weight);
        }
        validate_evaluation(&result, n)?;
        Ok(WeightedFrozenEvaluation {
            evaluation: result,
            term_energies,
        })
    }
}

fn validate_evaluation(evaluation: &FrozenDensityEvaluation, dimension: usize) -> Result<()> {
    if !evaluation.energy.is_finite() || !finite_symmetric(&evaluation.fock, dimension) {
        return invalid(
            "the active scalar must return a finite energy and symmetric same-chart Fock matrix",
        );
    }
    Ok(())
}

impl ActiveSpaceScf<'_> {
    /// Restricted compatibility wrapper over the shared constrained spin loop.
    pub fn solve(
        &self,
        initial_active_density: Option<&Matrix>,
        mut evaluate: impl FnMut(&Matrix) -> Result<FrozenDensityEvaluation>,
    ) -> Result<FrozenReferenceScf> {
        if self.active_electrons == 0 || self.active_electrons % 2 != 0 {
            return invalid("restricted active scalar requires a nonzero even electron count");
        }
        let spaces = [scf::Channel {
            overlap: self.overlap,
            complement: self.complement,
            frozen: self.frozen_orbitals,
            core: self.initial_hcore,
            occupied: self.active_electrons / 2,
            occupation: 2.,
        }];
        let seed = initial_active_density.map(|p| [p]);
        let solved = scf::solve(&spaces, self.options, seed.as_ref().map(|p| &p[..]), |p| {
            evaluate(&p[0])
        })?;
        let channel = solved.channels.into_iter().next().unwrap();
        Ok(FrozenReferenceScf {
            active_density: channel.density,
            active_orbitals: channel.orbitals,
            orbital_energies: channel.energies,
            evaluation: solved.evaluation,
            history: solved.history,
            projected_orbital_residual: solved.residual,
            active_idempotency_error: channel.idempotency,
            active_frozen_overlap: channel.frozen_overlap,
            particle_number_error: channel.particle_error,
        })
    }
}

impl<'a> FrozenReferenceModel<'a> {
    pub fn evaluator(&self) -> &FrozenDensityEvaluator<'a> {
        &self.evaluator
    }
    pub fn complement(&self) -> &Matrix {
        &self.complement
    }
    pub fn active_electrons(&self) -> usize {
        self.active_electrons
    }
    pub fn new(
        molecule: &'a Molecule,
        basis: &'a MolBasis,
        grid: &'a MolecularGrid,
        functional: &'a ResolvedFunctional,
        options: &ScfOptions,
        selection: &FrozenSelection,
    ) -> Result<Self> {
        if selection.signature != signature(molecule, basis) || selection.diagnostics.ambiguous {
            return invalid(
                "frozen-space geometry/basis mismatch or ambiguous occupied-space extraction",
            );
        }
        if options.max_iter == 0
            || [
                options.e_tol,
                options.p_tol,
                options.conv_tol_grad,
                options.orth_threshold,
            ]
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
        {
            return invalid("constrained-reference SCF needs a positive iteration limit and finite positive tolerances");
        }
        let n = basis.nao_sph;
        if selection.frozen_electrons != 8
            || selection.ionic_charge != 1
            || selection.full_electrons != molecule.n_electrons()?
            || selection.full_electrons.checked_sub(8) != Some(selection.active_electrons)
            || selection.frozen_orbitals.rows != n
            || selection.frozen_orbitals.cols != 4
            || selection
                .frozen_orbitals
                .as_slice()
                .iter()
                .any(|x| !x.is_finite())
            || selection.active_reference_orbitals.rows != n
            || selection.active_reference_orbitals.cols != selection.active_electrons / 2
            || selection
                .active_reference_orbitals
                .as_slice()
                .iter()
                .any(|x| !x.is_finite())
            || !finite_symmetric(&selection.full_reference_fock, n)
            || !finite_symmetric(&selection.frozen_density, n)
            || !finite_symmetric(&selection.active_reference_density, n)
            || selection
                .frozen_density
                .max_abs_diff(&density(&selection.frozen_orbitals, 4))
                > 1e-9
            || selection.active_reference_density.max_abs_diff(&density(
                &selection.active_reference_orbitals,
                selection.active_reference_orbitals.cols,
            )) > 1e-9
        {
            return invalid("frozen-selection dimensions, electron ledger or occupied projectors were modified inconsistently");
        }
        let evaluator = FrozenDensityEvaluator::new(
            molecule,
            basis,
            grid,
            functional,
            options,
            &selection.frozen_density,
        )?;
        let reference = evaluator.evaluate(&selection.active_reference_density)?;
        if (reference.energy - selection.full_reference_energy).abs() > 1e-10
            || reference.fock.max_abs_diff(&selection.full_reference_fock) > 1e-9
        {
            return invalid("functional, quadrature or physical reference differs from the extracted full-system state");
        }
        let orthogonalizer = Orthogonalizer::canonical(&evaluator.overlap, options.orth_threshold)?;
        if orthogonalizer.n_dropped != 0 {
            return invalid("frozen reference requires a fixed full-rank AO template");
        }
        let x = orthogonalizer.x;
        let b = x.transpose_matmul_seq(&evaluator.overlap.matmul_seq(&selection.frozen_orbitals));
        let mut projector = Matrix::identity(x.cols);
        add(&mut projector, &b.matmul_transpose_seq(&b), -1.0);
        let (values, vectors) = symmetric_eigen(&projector)?;
        let frozen = selection.frozen_orbitals.cols;
        if values[..frozen].iter().any(|v| v.abs() > 1e-8)
            || values[frozen..].iter().any(|v| (v - 1.0).abs() > 1e-8)
        {
            return invalid("the frozen projector does not define the expected fixed-rank orthogonal complement");
        }
        let complement = x.matmul_seq(&columns(&vectors, &(frozen..x.cols).collect::<Vec<_>>()));
        if metric_gram(&complement, &evaluator.overlap)
            .max_abs_diff(&Matrix::identity(complement.cols))
            > 1e-9
        {
            return invalid("the frozen-space complement is not orthonormal in the AO metric");
        }
        if selection.active_electrons == 0
            || selection.active_electrons % 2 != 0
            || selection.active_electrons / 2 > complement.cols
        {
            return invalid(
                "the fixed even active electron count does not fit the constrained reference space",
            );
        }
        Ok(Self {
            evaluator,
            complement,
            active_electrons: selection.active_electrons,
            frozen_orbitals: selection.frozen_orbitals.clone(),
            options: options.clone(),
            selection: selection.clone(),
        })
    }

    /// None uses an independent constrained core-Hamiltonian guess. A supplied
    /// seed is an AO determinant density and cannot alter the frozen projector.
    pub fn solve(&self, initial_active_density: Option<&Matrix>) -> Result<FrozenReferenceScf> {
        self.solve_perturbed(initial_active_density, None)
    }

    /// Resolve only active electrons in the unchanged full-AO frozen complement.
    /// This reference diagnostic separates frozen internal response from later
    /// operator/basis compression; it is not a production fragment response.
    pub fn solve_in_electric_field(
        &self,
        field: Vec3,
        origin: Vec3,
        initial_active_density: Option<&Matrix>,
    ) -> Result<FrozenReferenceScf> {
        let perturbation = self.evaluator.electric_perturbation(field, origin)?;
        self.solve_perturbed(initial_active_density, Some(&perturbation))
    }

    fn solve_perturbed(
        &self,
        initial_active_density: Option<&Matrix>,
        perturbation: Option<&FrozenElectricPerturbation>,
    ) -> Result<FrozenReferenceScf> {
        let mut h_core = self.evaluator.h_core.clone();
        if let Some(field) = perturbation {
            add(&mut h_core, &field.operator, 1.0);
        }
        ActiveSpaceScf {
            overlap: &self.evaluator.overlap,
            complement: &self.complement,
            frozen_orbitals: &self.frozen_orbitals,
            active_electrons: self.active_electrons,
            options: &self.options,
            initial_hcore: &h_core,
        }
        .solve(initial_active_density, |active| {
            self.evaluator.evaluate_perturbed(active, perturbation)
        })
    }

    /// Explicit electrostatic potential of removed nuclei minus the frozen
    /// electron cloud. This is a reference target, not a production evaluator.
    pub fn frozen_core_potential(&self, points: &[Vec3]) -> Result<Vec<f64>> {
        let selection = &self.selection;
        let removed: Vec<_> = std::iter::once(selection.boundary.carbon)
            .chain(selection.boundary.hydrogens)
            .collect();
        let mut potential = Vec::with_capacity(points.len());
        for &point in points {
            if point.to_array().iter().any(|x| !x.is_finite())
                || removed
                    .iter()
                    .any(|&a| (point - self.evaluator.molecule.atoms[a].position).norm() < 1e-10)
            {
                return invalid("frozen-core electrostatic probes must be finite and separated from removed nuclei");
            }
            let probe = Molecule::new(vec![Atom {
                z: 1,
                position: point,
            }]);
            let unit = nuclear_matrix(self.evaluator.basis, &probe, Some(&[1.0]));
            let nuclear: f64 = removed
                .iter()
                .map(|&a| {
                    let atom = &self.evaluator.molecule.atoms[a];
                    f64::from(atom.z) / (point - atom.position).norm()
                })
                .sum();
            potential.push(nuclear + selection.frozen_density.frobenius_dot(&unit));
        }
        Ok(potential)
    }

    pub fn frozen_core_dipole(&self, origin: Vec3) -> Result<Vec3> {
        let selection = &self.selection;
        if origin.to_array().iter().any(|x| !x.is_finite()) {
            return invalid("frozen-core dipole requires a finite origin");
        }
        let nuclear = std::iter::once(selection.boundary.carbon)
            .chain(selection.boundary.hydrogens)
            .fold(Vec3::zero(), |sum, a| {
                let atom = &self.evaluator.molecule.atoms[a];
                sum + (atom.position - origin) * f64::from(atom.z)
            });
        let dipoles = dipole_matrices(self.evaluator.basis, origin);
        Ok(nuclear
            - Vec3::new(
                selection.frozen_density.frobenius_dot(&dipoles[0]),
                selection.frozen_density.frobenius_dot(&dipoles[1]),
                selection.frozen_density.frobenius_dot(&dipoles[2]),
            ))
    }
}

pub struct StaticResidualTarget {
    pub removed_nuclear: Matrix,
    pub frozen_hartree: Matrix,
    pub nonadditive_xc: Matrix,
    pub reference_pauli: Matrix,
    /// Subtract the chosen sigma-derived electronic operator before fitting
    /// local/projector coefficients. Retained nuclei and T are absent here.
    pub total: Matrix,
    pub core_reference_energy: f64,
    pub internal_reference_energy: f64,
    pub reconstruction_error: f64,
    pub frozen_pauli_energy: f64,
}

impl FrozenReferenceModel<'_> {
    /// Hermitian finite reference Pauli target: F+V=Q^T F Q+S B K B^T S,
    /// Q=I-B B^T S, K=kappa I. The exact constrained SCF above does not use
    /// this static approximation or require a large level shift.
    pub fn static_residual_target(&self, kappa: f64) -> Result<StaticResidualTarget> {
        let selection = &self.selection;
        if !kappa.is_finite() || kappa <= 0.0 {
            return invalid("static residual requires a positive finite frozen-block level");
        }
        let basis = self.evaluator.basis;
        let mol = self.evaluator.molecule;
        let s = &self.evaluator.overlap;
        let pa = &selection.active_reference_density;
        let pb = &selection.frozen_density;
        let half_b = scaled(pb, 0.5);
        let half_a = scaled(pa, 0.5);
        let (jb, _) = self
            .evaluator
            .engine
            .build(basis, pb, [&half_b, &half_b], false)?;
        let (ja, _) = self
            .evaluator
            .engine
            .build(basis, pa, [&half_a, &half_a], false)?;
        let xc_a = xc_contribution(
            self.evaluator.functional,
            basis,
            self.evaluator.grid,
            [&half_a, &half_a],
        )?;
        let mut pt = pa.clone();
        add(&mut pt, pb, 1.0);
        let half_t = scaled(&pt, 0.5);
        let xc_t = xc_contribution(
            self.evaluator.functional,
            basis,
            self.evaluator.grid,
            [&half_t, &half_t],
        )?;
        let mut delta_xc = xc_t.vxc[0].clone();
        add(&mut delta_xc, &xc_a.vxc[0], -1.0);
        let b = &selection.frozen_orbitals;
        let sb = s.matmul_seq(b);
        let mut q = Matrix::identity(b.rows);
        add(&mut q, &b.matmul_transpose_seq(&sb), -1.0);
        let mut pauli = q.transpose_matmul_seq(&selection.full_reference_fock.matmul_seq(&q));
        add(&mut pauli, &selection.full_reference_fock, -1.0);
        add(&mut pauli, &sb.matmul_transpose_seq(&sb), kappa);
        let removed: Vec<_> = std::iter::once(selection.boundary.carbon)
            .chain(selection.boundary.hydrogens)
            .collect();
        let z_removed: Vec<_> = mol
            .atoms
            .iter()
            .enumerate()
            .map(|(a, atom)| {
                if removed.contains(&a) {
                    f64::from(atom.z)
                } else {
                    0.0
                }
            })
            .collect();
        let z_retained: Vec<_> = mol
            .atoms
            .iter()
            .enumerate()
            .map(|(a, atom)| {
                if removed.contains(&a) {
                    0.0
                } else {
                    f64::from(atom.z)
                }
            })
            .collect();
        let removed_nuclear = nuclear_matrix(basis, mol, Some(&z_removed));
        let retained_nuclear = nuclear_matrix(basis, mol, Some(&z_retained));
        let mut total = removed_nuclear.clone();
        add(&mut total, &jb, 1.0);
        add(&mut total, &delta_xc, 1.0);
        add(&mut total, &pauli, 1.0);
        let mut core = pb.frobenius_dot(&retained_nuclear);
        for a in 0..mol.len() {
            for c in a + 1..mol.len() {
                if !(removed.contains(&a) && removed.contains(&c)) {
                    core += f64::from(mol.atoms[a].z) * f64::from(mol.atoms[c].z)
                        / (mol.atoms[a].position - mol.atoms[c].position).norm();
                }
            }
        }
        let mut active_h = kinetic_matrix(basis);
        add(&mut active_h, &retained_nuclear, 1.0);
        add(&mut active_h, &total, 1.0);
        let without_constant =
            pa.frobenius_dot(&active_h) + 0.5 * pa.frobenius_dot(&ja) + xc_a.exc + core;
        let internal = selection.full_reference_energy - without_constant;
        Ok(StaticResidualTarget {
            removed_nuclear,
            frozen_hartree: jb,
            nonadditive_xc: delta_xc,
            frozen_pauli_energy: pa.frobenius_dot(&pauli),
            reference_pauli: pauli,
            total,
            core_reference_energy: core,
            internal_reference_energy: internal,
            reconstruction_error: (without_constant + internal - selection.full_reference_energy)
                .abs(),
        })
    }
}
