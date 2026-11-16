// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Self-consistent HF and Kohn-Sham calculations on a uniform k mesh.
//!
//! Every one-electron, density and Fock matrix remains a complex Hermitian
//! matrix at its own k point.  The only real-space detour is the inverse
//! Bloch transform needed by the localized Hartree and exact-exchange
//! contractions; their image blocks are transformed back to every k before
//! diagonalization.  This is therefore a finite-k SCF, not a Gamma-point
//! SCF followed by band interpolation.

use crate::error::{DftError, Result};
use crate::grid::GridOptions;
use crate::integrals::CoulombKernel;
use crate::linalg::{Matrix, Orthogonalizer};
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::exchange::{exchange_image_blocks, minimum_image_blocks};
use crate::periodic::integrals::bloch;
use crate::periodic::occupations::{
    solve_occupations, solve_occupations_during_scf, OccupationScheme,
};
use crate::periodic::xc::{xc_contribution_at_grid, AoDerivativeOrder, PeriodicXcGrid};
use crate::periodic::Cell;
use crate::scf::{Guess, Reference};
use crate::xc::ResolvedFunctional;

#[derive(Clone, Debug)]
pub struct PeriodicScfOptions {
    /// Explicit periodic VV10 operator and physical direct-sum cutoff.
    pub vv10: Option<crate::periodic::nonlocal::PeriodicVv10Options>,
    /// Uniform, unshifted Monkhorst-Pack mesh.
    pub k_mesh: [usize; 3],
    /// Which lattice axes are physically periodic.  Partial boundaries are
    /// represented explicitly; their standalone dimensional Coulomb kernels
    /// must still be connected to every electronic SCF operator.
    pub boundary: crate::periodic::features::PeriodicBoundary,
    pub occupation: OccupationScheme,
    pub reference: Reference,
    /// Initial Fock construction. `Sad` superposes the molecular atomic
    /// densities in every cell, builds one periodic HF Fock from that seed,
    /// then returns to the ordinary finite-k occupation solver. `Core` stays
    /// the default to preserve the pre-v0.1.1 periodic starting point.
    pub guess: Guess,
    pub max_iter: usize,
    pub e_tol: f64,
    pub p_tol: f64,
    pub conv_tol_grad: f64,
    pub use_diis: bool,
    /// Use convex-hull ADIIS while the physical commutator is large, then
    /// hand over to CDIIS near convergence.
    /// Use zero-temperature ADIIS for integer occupations while the
    /// commutator error exceeds `adiis_until_error`. Fermi-Dirac calculations
    /// use CDIIS because this ADIIS model does not include entropy curvature.
    pub use_adiis: bool,
    pub adiis_until_error: f64,
    pub diis_size: usize,
    pub damping: f64,
    pub damping_until_error: f64,
    /// Virtual-space shift in Hartree while the physical commutator residual
    /// exceeds `level_shift_until`. The shift is removed automatically, so a
    /// converged result is canonical for the unshifted physical Fock.
    pub level_shift: f64,
    pub level_shift_until: f64,
    pub orth_threshold: f64,
    /// Localized exact-exchange cutoff in Bohr.  `None` uses the cell's AO
    /// overlap cutoff.  Converge this value for production calculations.
    pub exchange_cutoff: Option<f64>,
    /// Bloch-AO cutoff for XC quadrature.  `None` uses the cell cutoff.
    pub xc_ao_cutoff: Option<f64>,
    pub grid: GridOptions,
    pub grid_shell: i64,
    /// The caller adds the defining pair dispersion lattice sum of a
    /// functional that carries it by definition (`periodic::dispersion`).
    /// Without this acknowledgement such a functional is refused, so that an
    /// energy never silently lacks its defining term.
    pub pair_dispersion_from_caller: bool,
    /// Learned functional evaluated on the exchange-correlation grid in
    /// addition to the semilocal parts of the functional.
    pub skala: Option<std::sync::Arc<crate::skala::SkalaModel>>,
}

impl Default for PeriodicScfOptions {
    fn default() -> Self {
        Self {
            vv10: None,
            k_mesh: [1, 1, 1],
            boundary: crate::periodic::features::PeriodicBoundary::FULL_3D,
            occupation: OccupationScheme::Insulator,
            reference: Reference::Auto,
            guess: Guess::Core,
            max_iter: 100,
            e_tol: 1e-9,
            p_tol: 1e-8,
            conv_tol_grad: 3.162_277_660_168_379_5e-5,
            use_diis: true,
            use_adiis: true,
            adiis_until_error: 0.1,
            diis_size: 8,
            damping: 0.25,
            damping_until_error: 0.2,
            level_shift: 0.0,
            level_shift_until: 1e-4,
            orth_threshold: 1e-7,
            exchange_cutoff: None,
            xc_ao_cutoff: None,
            grid: GridOptions::default(),
            grid_shell: 2,
            pair_dispersion_from_caller: false,
            skala: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct XcPartSignature {
    coefficient: u64,
    name: String,
    kind: crate::xc::Kind,
    family: crate::xc::Family,
    parameters: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FunctionalSignature {
    description: String,
    family: crate::xc::Family,
    hybrid: [u64; 3],
    required_nonlocal: Option<crate::xc::RequiredNonlocalTerm>,
    parts: Vec<XcPartSignature>,
}

impl FunctionalSignature {
    fn new(functional: &ResolvedFunctional) -> Self {
        Self {
            description: functional.description.clone(),
            family: functional.family,
            hybrid: [
                functional.hybrid.cx.to_bits(),
                functional.hybrid.clr.to_bits(),
                functional.hybrid.omega.to_bits(),
            ],
            required_nonlocal: functional.required_nonlocal_kind(),
            parts: functional
                .parts
                .iter()
                .map(|(coefficient, part)| XcPartSignature {
                    coefficient: coefficient.to_bits(),
                    name: part.name().to_string(),
                    kind: part.kind(),
                    family: part.family(),
                    parameters: part.params().into_iter().map(f64::to_bits).collect(),
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GridSignature {
    level: usize,
    prune: bool,
    scheme: crate::grid::AngularScheme,
    radial_override: Option<usize>,
    angular_override: Option<usize>,
    per_element: Vec<(u8, usize, usize)>,
    per_atom: Vec<(usize, usize, usize)>,
    weight_cutoff: u64,
}

impl GridSignature {
    fn new(options: &GridOptions) -> Self {
        Self {
            level: options.level,
            prune: options.prune,
            scheme: options.scheme,
            radial_override: options.radial_override,
            angular_override: options.angular_override,
            per_element: options.per_element.clone(),
            per_atom: options.per_atom.clone(),
            weight_cutoff: options.weight_cutoff.to_bits(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct XcDerivativePlan {
    ao_cutoff: u64,
    grid_shell: i64,
    grid: GridSignature,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PeriodicDerivativePlan {
    vv10: Option<[u64; 3]>,
    cell_identity: u64,
    functional: FunctionalSignature,
    k_mesh: [usize; 3],
    boundary: crate::periodic::features::PeriodicBoundary,
    exchange_cutoff: Option<u64>,
    xc: Option<XcDerivativePlan>,
}

impl PeriodicDerivativePlan {
    fn new(cell: &Cell, functional: &ResolvedFunctional, options: &PeriodicScfOptions) -> Self {
        let exchange_cutoff =
            (functional.hybrid.cx != 0.0 || functional.hybrid.clr != 0.0).then(|| {
                options
                    .exchange_cutoff
                    .unwrap_or(cell.overlap_cutoff)
                    .to_bits()
            });
        let xc =
            (!functional.parts.is_empty() || options.skala.is_some() || options.vv10.is_some())
                .then(|| XcDerivativePlan {
                    ao_cutoff: options
                        .xc_ao_cutoff
                        .unwrap_or(cell.overlap_cutoff)
                        .to_bits(),
                    grid_shell: options.grid_shell,
                    grid: GridSignature::new(&options.grid),
                });
        Self {
            vv10: options.vv10.map(|v| {
                [
                    v.parameters.b.to_bits(),
                    v.parameters.c.to_bits(),
                    v.cutoff.to_bits(),
                ]
            }),
            cell_identity: cell.identity(),
            functional: FunctionalSignature::new(functional),
            k_mesh: options.k_mesh,
            boundary: options.boundary,
            exchange_cutoff,
            xc,
        }
    }
}

/// Converged periodic wavefunction.  Fractional occupations are present only
/// when explicitly requested through [`OccupationScheme::FermiDirac`].
#[derive(Clone, Debug)]
pub struct PeriodicScfResult {
    /// Internal energy `E` including nuclei.
    pub energy: f64,
    /// Mermin free energy `F = E - T S` (equal to `energy` for an insulator).
    pub free_energy: f64,
    /// Positive `T S` contribution in Hartree per cell.
    pub entropy_term: f64,
    pub e_nuc: f64,
    pub e_elec: f64,
    pub e_xc: f64,
    /// Included in `e_xc`, in Hartree per cell.
    pub e_nlc: f64,
    pub k_points: Vec<crate::math::Vec3>,
    pub mo_energy: [Vec<Vec<f64>>; 2],
    pub occupations: [Vec<Vec<f64>>; 2],
    pub chemical_potential: [Option<f64>; 2],
    pub density: [Vec<BlochMatrix>; 2],
    /// `W(k) = C f epsilon C^dagger`, required by the Pulay force when
    /// occupations are fractional.
    pub energy_weighted_density: [Vec<BlochMatrix>; 2],
    pub fock: [Vec<BlochMatrix>; 2],
    pub overlap: Vec<BlochMatrix>,
    pub h_core: Vec<BlochMatrix>,
    pub n_alpha: usize,
    pub n_beta: usize,
    pub unrestricted: bool,
    /// This constraint applies separately to each spin and is not physical
    /// spinful time reversal. Finite orbital fields set it to false.
    pub orbital_conjugation_constraint: bool,
    pub grid_electron_count: Option<f64>,
    /// Accepted maximum canonical-metric SCF residual, in Hartree.
    pub orbital_residual: Option<f64>,
    pub converged: bool,
    pub iterations: usize,
    pub(crate) derivative_plan: Option<PeriodicDerivativePlan>,
}

/// Raw covariance defects on an unshifted, complete uniform Bloch mesh.
/// Complex conjugation within one collinear spin channel is an orbital
/// antiunitary symmetry. Physical spin-1/2 time reversal also exchanges
/// alpha and beta. A magnetic collinear state can retain the former and
/// break the latter, including when its net spin population is zero.
#[derive(Clone, Debug)]
pub struct CollinearTimeReversalDiagnostics {
    pub orbital_density_residual: [f64; 2],
    pub orbital_fock_residual: [f64; 2],
    pub spinful_density_residual: f64,
    pub spinful_fock_residual: f64,
}

/// Measure the supplied matrices without imposing a symmetry or modifying
/// the reference. The mesh ordering is (ix * ny + iy) * nz + iz.
pub fn collinear_time_reversal_diagnostics(
    density: &[Vec<BlochMatrix>; 2],
    fock: &[Vec<BlochMatrix>; 2],
    mesh: [usize; 3],
) -> Result<CollinearTimeReversalDiagnostics> {
    let count = mesh.iter().try_fold(1_usize, |n, x| n.checked_mul(*x));
    let n = density[0].first().map_or(0, BlochMatrix::rows);
    if mesh.contains(&0)
        || count.is_none()
        || n == 0
        || density.iter().chain(fock).any(|spin| {
            Some(spin.len()) != count
                || spin.iter().any(|m| {
                    m.re.rows != n
                        || m.re.cols != n
                        || m.im.rows != n
                        || m.im.cols != n
                        || m.re
                            .as_slice()
                            .iter()
                            .chain(m.im.as_slice())
                            .any(|v| !v.is_finite())
                })
        })
    {
        return Err(DftError::InvalidInput(
            "time-reversal diagnostics require a complete non-empty uniform mesh and \
            finite square spin density/Fock matrices with a common AO shape"
                .into(),
        ));
    }
    let [nx, ny, nz] = mesh;
    let index = |i: usize, j: usize, k: usize| (i * ny + j) * nz + k;
    let defect = |a: &BlochMatrix, b: &BlochMatrix| {
        a.re.as_slice()
            .iter()
            .zip(b.re.as_slice())
            .map(|(x, y)| (x - y).abs())
            .chain(
                a.im.as_slice()
                    .iter()
                    .zip(b.im.as_slice())
                    .map(|(x, y)| (x + y).abs()),
            )
            .fold(0.0_f64, f64::max)
    };
    let mut report = CollinearTimeReversalDiagnostics {
        orbital_density_residual: [0.0; 2],
        orbital_fock_residual: [0.0; 2],
        spinful_density_residual: 0.0,
        spinful_fock_residual: 0.0,
    };
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let left = index(i, j, k);
                let right = index((nx - i) % nx, (ny - j) % ny, (nz - k) % nz);
                for spin in 0..2 {
                    report.orbital_density_residual[spin] = report.orbital_density_residual[spin]
                        .max(defect(&density[spin][left], &density[spin][right]));
                    report.orbital_fock_residual[spin] = report.orbital_fock_residual[spin]
                        .max(defect(&fock[spin][left], &fock[spin][right]));
                    report.spinful_density_residual = report
                        .spinful_density_residual
                        .max(defect(&density[spin][left], &density[1 - spin][right]));
                    report.spinful_fock_residual = report
                        .spinful_fock_residual
                        .max(defect(&fock[spin][left], &fock[1 - spin][right]));
                }
            }
        }
    }
    Ok(report)
}

impl PeriodicScfResult {
    pub fn time_reversal_diagnostics(
        &self,
        mesh: [usize; 3],
    ) -> Result<CollinearTimeReversalDiagnostics> {
        collinear_time_reversal_diagnostics(&self.density, &self.fock, mesh)
    }
    pub fn total_density(&self) -> Vec<BlochMatrix> {
        self.density[0]
            .iter()
            .zip(&self.density[1])
            .map(|(a, b)| {
                let mut total = a.clone();
                total.add_scaled(b, 1.0);
                total
            })
            .collect()
    }

    pub(crate) fn validate_derivative_request(
        &self,
        cell: &Cell,
        functional: &ResolvedFunctional,
        options: &PeriodicScfOptions,
    ) -> Result<()> {
        if options.vv10.is_some() {
            return Err(DftError::Unsupported {
                feature: "finite-k derivatives of a periodic nonlocal-correlation reference".into(),
                missing: "the nonlocal operator terms on this response backend".into(),
            });
        }
        self.validate_derivative_plan(cell, functional, options)
    }

    pub(crate) fn validate_derivative_plan(
        &self,
        cell: &Cell,
        functional: &ResolvedFunctional,
        options: &PeriodicScfOptions,
    ) -> Result<()> {
        if !self.converged {
            return Err(DftError::InvalidInput(
                "periodic analytic derivatives require a converged SCF result".to_string(),
            ));
        }
        let requested = PeriodicDerivativePlan::new(cell, functional, options);
        match &self.derivative_plan {
            Some(original) if original == &requested => Ok(()),
            Some(original) if original.cell_identity != requested.cell_identity => {
                Err(DftError::InvalidInput(
                    "periodic analytic derivatives were requested for a different Cell than the SCF result; rebuild and rerun the SCF for this geometry/lattice"
                        .to_string(),
                ))
            }
            Some(original) if original.functional != requested.functional => {
                Err(DftError::InvalidInput(
                    "periodic analytic derivatives must use the same resolved functional as the SCF result"
                        .to_string(),
                ))
            }
            Some(_) => Err(DftError::InvalidInput(
                "periodic analytic derivatives must use the same k mesh, boundary, exchange cutoff, and XC grid plan as the SCF result"
                    .to_string(),
            )),
            None => Err(DftError::InvalidInput(
                "periodic analytic derivatives require an SCF result carrying its validated calculation plan"
                    .to_string(),
            )),
        }
    }
}

pub(crate) fn validate_options(options: &PeriodicScfOptions) -> Result<()> {
    options.boundary.validate_k_mesh(options.k_mesh)?;
    crate::periodic::features::PeriodicCapabilities::current()
        .validate_boundary(options.boundary)?;
    validate_solver_options(options)
}

pub(crate) fn validate_solver_options(options: &PeriodicScfOptions) -> Result<()> {
    options.boundary.validate_k_mesh(options.k_mesh)?;
    options.occupation.validate()?;
    if let Some(vv10) = options.vv10 {
        vv10.validate()?;
    }
    if options.max_iter == 0 || options.diis_size == 0 {
        return Err(DftError::InvalidInput(
            "periodic SCF max_iter and diis_size must be positive".to_string(),
        ));
    }
    for (name, cutoff) in [
        ("exact-exchange", options.exchange_cutoff),
        ("XC AO", options.xc_ao_cutoff),
    ] {
        if let Some(value) = cutoff {
            if !(value.is_finite() && value > 0.0) {
                return Err(DftError::InvalidInput(format!(
                    "periodic {name} cutoff must be finite and positive"
                )));
            }
        }
    }
    if !([options.e_tol, options.p_tol, options.conv_tol_grad]
        .into_iter()
        .all(|value| value.is_finite() && value > 0.0)
        && options.orth_threshold.is_finite()
        && options.orth_threshold > 0.0
        && options.orth_threshold <= 1.0
        && options.damping.is_finite()
        && (0.0..1.0).contains(&options.damping)
        && options.damping_until_error.is_finite()
        && options.damping_until_error >= 0.0
        && options.adiis_until_error.is_finite()
        && options.adiis_until_error >= 0.0
        && options.level_shift.is_finite()
        && options.level_shift >= 0.0
        && options.level_shift_until.is_finite()
        && options.level_shift_until >= 0.0)
    {
        return Err(DftError::InvalidInput(
            "periodic SCF tolerances must be finite and positive, orth_threshold \
                must be in (0,1], damping must be in [0,1), and \
                ADIIS/damping/level-shift controls must be finite and non-negative"
                .to_string(),
        ));
    }
    if !(options.grid.weight_cutoff.is_finite() && options.grid.weight_cutoff >= 0.0) {
        return Err(DftError::InvalidInput(
            "periodic grid weight_cutoff must be finite and non-negative".to_string(),
        ));
    }
    Ok(())
}

fn solve_orbitals(
    fock: &BlochMatrix,
    orthogonalizer: &Orthogonalizer,
) -> Result<(Vec<f64>, Matrix)> {
    let embedded = fock.real_embedding();
    let (energies2, coeff2) = orthogonalizer.solve_roothaan(&embedded)?;
    if orthogonalizer.n_kept % 2 != 0 {
        return Err(DftError::LinearAlgebra(
            "complex overlap embedding retained an odd-dimensional subspace".to_string(),
        ));
    }
    let energies = energies2
        .chunks_exact(2)
        .map(|pair| 0.5 * (pair[0] + pair[1]))
        .collect();
    Ok((energies, coeff2))
}

/// Add `lambda (I-P)` in the orthonormal Bloch-AO basis without storing MO
/// coefficients from a previous iteration. In the original nonorthogonal AO
/// representation this is `lambda (S-S P S)`. The doubled-real embedding is
/// an algebra homomorphism for complex matrix products, so the same identity
/// applies at finite k.
fn level_shifted_fock(
    fock: &BlochMatrix,
    overlap: &BlochMatrix,
    density: &BlochMatrix,
    lambda: f64,
) -> BlochMatrix {
    debug_assert!(lambda >= 0.0 && lambda.is_finite());
    let n = fock.rows();
    let s = overlap.real_embedding();
    let p = density.real_embedding();
    let sps = s.matmul_seq(&p).matmul_seq(&s);
    let mut shifted = fock.real_embedding();
    for i in 0..shifted.rows {
        for j in 0..shifted.cols {
            shifted[(i, j)] += lambda * (s[(i, j)] - sps[(i, j)]);
        }
    }
    let mut out = BlochMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            out.re[(i, j)] = 0.5 * (shifted[(i, j)] + shifted[(n + i, n + j)]);
            out.im[(i, j)] = 0.5 * (shifted[(n + i, j)] - shifted[(i, n + j)]);
        }
    }
    out.make_hermitian();
    out
}

fn weighted_density_from_embedding(
    n: usize,
    coefficients: &Matrix,
    weights: &[f64],
) -> BlochMatrix {
    debug_assert!(2 * weights.len() <= coefficients.cols);
    let mut density = BlochMatrix::zeros(n, n);
    for (band, &weight) in weights.iter().enumerate() {
        for duplicate in 0..2 {
            let column = 2 * band + duplicate;
            for i in 0..n {
                let real_left = weight * coefficients[(i, column)];
                let imaginary_left = weight * coefficients[(n + i, column)];
                for j in 0..n {
                    let right = coefficients[(j, column)];
                    density.re[(i, j)] += real_left * right;
                    density.im[(i, j)] += imaginary_left * right;
                }
            }
        }
    }
    density.make_hermitian();
    density
}

#[derive(Clone)]
struct DensitySet {
    energies: Vec<Vec<f64>>,
    occupations: Vec<Vec<f64>>,
    density: Vec<BlochMatrix>,
    weighted: Vec<BlochMatrix>,
    chemical_potential: Option<f64>,
    entropy_term: f64,
}

fn solve_density_set_with_policy(
    focks: &[BlochMatrix],
    orthogonalizers: &[Orthogonalizer],
    n_electrons: usize,
    scheme: OccupationScheme,
    final_state: bool,
) -> Result<DensitySet> {
    let solved: Vec<(Vec<f64>, Matrix)> = focks
        .iter()
        .zip(orthogonalizers)
        .map(|(fock, orthogonalizer)| solve_orbitals(fock, orthogonalizer))
        .collect::<Result<_>>()?;
    let energies: Vec<Vec<f64>> = solved.iter().map(|(energy, _)| energy.clone()).collect();
    let occupation = if final_state {
        solve_occupations(&energies, n_electrons as f64, scheme)?
    } else {
        solve_occupations_during_scf(&energies, n_electrons as f64, scheme)?
    };
    let mut density = Vec::with_capacity(focks.len());
    let mut weighted = Vec::with_capacity(focks.len());
    for (ik, (_, coefficients)) in solved.iter().enumerate() {
        density.push(weighted_density_from_embedding(
            focks[ik].rows(),
            coefficients,
            &occupation.occupations[ik],
        ));
        let energy_weights: Vec<f64> = occupation.occupations[ik]
            .iter()
            .zip(&energies[ik])
            .map(|(&f, &energy)| f * energy)
            .collect();
        weighted.push(weighted_density_from_embedding(
            focks[ik].rows(),
            coefficients,
            &energy_weights,
        ));
    }
    Ok(DensitySet {
        energies,
        occupations: occupation.occupations,
        density,
        weighted,
        chemical_potential: occupation.chemical_potential,
        entropy_term: occupation.entropy_term,
    })
}

fn solve_density_set(
    focks: &[BlochMatrix],
    orthogonalizers: &[Orthogonalizer],
    n_electrons: usize,
    scheme: OccupationScheme,
) -> Result<DensitySet> {
    solve_density_set_with_policy(focks, orthogonalizers, n_electrons, scheme, true)
}

fn solve_density_set_during_scf(
    focks: &[BlochMatrix],
    orthogonalizers: &[Orthogonalizer],
    n_electrons: usize,
    scheme: OccupationScheme,
) -> Result<DensitySet> {
    solve_density_set_with_policy(focks, orthogonalizers, n_electrons, scheme, false)
}

pub(crate) fn conjugate_mesh_index(index: usize, [nx, ny, nz]: [usize; 3]) -> usize {
    let (i, j, k) = (index / (ny * nz), (index / nz) % ny, index % nz);
    ((nx - i) % nx * ny + (ny - j) % ny) * nz + (nz - k) % nz
}

pub(crate) fn time_reversal_symmetrize(matrices: &mut [BlochMatrix], mesh: [usize; 3]) {
    let [nx, ny, nz] = mesh;
    let index = |i: usize, j: usize, k: usize| (i * ny + j) * nz + k;
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let left = index(i, j, k);
                let right = conjugate_mesh_index(left, mesh);
                if right < left {
                    continue;
                }
                if left == right {
                    for value in matrices[left].im.as_mut_slice() {
                        *value = 0.0;
                    }
                    matrices[left].make_hermitian();
                } else {
                    let (before, from_right) = matrices.split_at_mut(right);
                    let a = &mut before[left];
                    let b = &mut from_right[0];
                    for p in 0..a.re.as_slice().len() {
                        let re = 0.5 * (a.re.as_slice()[p] + b.re.as_slice()[p]);
                        let im = 0.5 * (a.im.as_slice()[p] - b.im.as_slice()[p]);
                        a.re.as_mut_slice()[p] = re;
                        b.re.as_mut_slice()[p] = re;
                        a.im.as_mut_slice()[p] = im;
                        b.im.as_mut_slice()[p] = -im;
                    }
                    a.make_hermitian();
                    b.make_hermitian();
                }
            }
        }
    }
}

fn density_pairs(density: &[BlochMatrix]) -> Vec<(Matrix, Matrix)> {
    density
        .iter()
        .map(|matrix| (matrix.re.clone(), matrix.im.clone()))
        .collect()
}

fn total_density(density: &[Vec<BlochMatrix>; 2]) -> Vec<BlochMatrix> {
    density[0]
        .iter()
        .zip(&density[1])
        .map(|(alpha, beta)| {
            let mut total = alpha.clone();
            total.add_scaled(beta, 1.0);
            total
        })
        .collect()
}

fn density_rms_change(
    next: [&[BlochMatrix]; 2],
    current: [&[BlochMatrix]; 2],
    active_spins: usize,
    nao: usize,
) -> f64 {
    let elements = 2 * nao * nao;
    let mut change2 = 0.0;
    let mut count = 0usize;
    for spin in 0..active_spins {
        for (next_k, current_k) in next[spin].iter().zip(current[spin]) {
            let rms = next_k.rms_diff(current_k);
            change2 += rms * rms * elements as f64;
            count += elements;
        }
    }
    if count == 0 {
        0.0
    } else {
        (change2 / count as f64).sqrt()
    }
}

pub(crate) fn commutator_error(
    fock: &BlochMatrix,
    density: &BlochMatrix,
    overlap: &BlochMatrix,
    orthogonalizer: &Orthogonalizer,
) -> Vec<f64> {
    let f = fock.real_embedding();
    let p = density.real_embedding();
    let s = overlap.real_embedding();
    let fps = f.matmul_seq(&p.matmul_seq(&s));
    let spf = s.matmul_seq(&p.matmul_seq(&f));
    let mut commutator = fps;
    for (value, rhs) in commutator.as_mut_slice().iter_mut().zip(spf.as_slice()) {
        *value -= rhs;
    }
    orthogonalizer
        .x
        .transpose_matmul_seq(&commutator.matmul_seq(&orthogonalizer.x))
        .as_slice()
        .to_vec()
}

/// Thermal SCF also needs occupation stationarity. A diagonal Fock and
/// density can commute while their Fermi occupations remain inconsistent.
/// Use the physical Fock's number-constrained Fermi map, in the overlap
/// metric, as an additional DIIS residual. Scaling by kBT gives Hartree
/// units without changing the Fock or its physical denominators.
fn thermal_density_error(
    target: &BlochMatrix,
    current: &BlochMatrix,
    overlap: &BlochMatrix,
    orthogonalizer: &Orthogonalizer,
    kbt: f64,
) -> Vec<f64> {
    let mut delta = target.clone();
    delta.add_scaled(current, -1.0);
    let sx = overlap.real_embedding().matmul_seq(&orthogonalizer.x);
    let error = sx.transpose_matmul_seq(&delta.real_embedding().matmul_seq(&sx));
    error.as_slice().iter().map(|x| x * kbt).collect()
}

pub(crate) struct PeriodicDiis {
    capacity: usize,
    focks: Vec<Vec<BlochMatrix>>,
    errors: Vec<Vec<f64>>,
}

/// Complex finite-k ADIIS over the convex hull of stored densities/Focks.
/// The uniform Brillouin-zone weight is common to every objective term and
/// therefore cancels from the minimizing coefficients.
struct PeriodicAdiis {
    capacity: usize,
    densities: Vec<Vec<BlochMatrix>>,
    focks: Vec<Vec<BlochMatrix>>,
}

impl PeriodicAdiis {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            densities: Vec::new(),
            focks: Vec::new(),
        }
    }

    fn push(&mut self, densities: Vec<BlochMatrix>, focks: Vec<BlochMatrix>) {
        debug_assert_eq!(densities.len(), focks.len());
        self.densities.push(densities);
        self.focks.push(focks);
        if self.focks.len() > self.capacity {
            self.densities.remove(0);
            self.focks.remove(0);
        }
    }

    fn dot(a: &[BlochMatrix], b: &[BlochMatrix]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x.trace_product(y)).sum()
    }

    fn differences(a: &[BlochMatrix], b: &[BlochMatrix]) -> Vec<BlochMatrix> {
        a.iter()
            .zip(b)
            .map(|(left, right)| {
                let mut value = left.clone();
                value.add_scaled(right, -1.0);
                value
            })
            .collect()
    }

    fn extrapolate(&self) -> Option<Vec<BlochMatrix>> {
        let count = self.focks.len();
        if count < 2 {
            return None;
        }
        let last = count - 1;
        let newest_density = &self.densities[last];
        let newest_fock = &self.focks[last];
        let density_differences: Vec<_> = self
            .densities
            .iter()
            .map(|density| Self::differences(density, newest_density))
            .collect();
        let fock_differences: Vec<_> = self
            .focks
            .iter()
            .map(|fock| Self::differences(fock, newest_fock))
            .collect();
        let mut g = vec![0.0; count];
        let mut h = vec![0.0; count * count];
        for i in 0..count {
            g[i] = Self::dot(&density_differences[i], newest_fock);
            for j in 0..count {
                h[i * count + j] = Self::dot(&density_differences[i], &fock_differences[j]);
            }
        }
        for i in 0..count {
            for j in (i + 1)..count {
                let average = 0.5 * (h[i * count + j] + h[j * count + i]);
                h[i * count + j] = average;
                h[j * count + i] = average;
            }
        }
        let coefficients = crate::scf::adiis::minimize_simplex(&g, &h, count)?;
        let template = self.focks.first()?;
        let mut result: Vec<BlochMatrix> = template
            .iter()
            .map(|matrix| BlochMatrix::zeros(matrix.rows(), matrix.cols()))
            .collect();
        for (coefficient, stored) in coefficients.iter().zip(&self.focks) {
            for (target, source) in result.iter_mut().zip(stored) {
                target.add_scaled(source, *coefficient);
            }
        }
        Some(result)
    }
}

fn periodic_adiis_eligible(options: &PeriodicScfOptions) -> bool {
    options.use_diis && options.use_adiis && options.occupation == OccupationScheme::Insulator
}

fn periodic_adiis_selected(options: &PeriodicScfOptions, max_error: f64) -> bool {
    periodic_adiis_eligible(options) && max_error > options.adiis_until_error
}

impl PeriodicDiis {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            focks: Vec::new(),
            errors: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, focks: Vec<BlochMatrix>, error: Vec<f64>) {
        self.focks.push(focks);
        self.errors.push(error);
        if self.focks.len() > self.capacity {
            self.focks.remove(0);
            self.errors.remove(0);
        }
    }

    pub(crate) fn extrapolate(&self) -> Option<Vec<BlochMatrix>> {
        let count = self.focks.len();
        if count < 2 {
            return None;
        }
        let dim = count + 1;
        let mut b = Matrix::zeros(dim, dim);
        let mut scale = 0.0_f64;
        for i in 0..count {
            for j in 0..count {
                let value: f64 = self.errors[i]
                    .iter()
                    .zip(&self.errors[j])
                    .map(|(a, b)| a * b)
                    .sum();
                b[(i, j)] = value;
                scale = scale.max(value.abs());
            }
        }
        if scale == 0.0 {
            return None;
        }
        for i in 0..count {
            for j in 0..count {
                b[(i, j)] /= scale;
            }
            b[(i, count)] = -1.0;
            b[(count, i)] = -1.0;
        }
        let mut rhs = vec![0.0; dim];
        rhs[count] = -1.0;
        let coefficients = crate::linalg::solve_linear(&b, &rhs).ok()?;
        let template = self.focks.first()?;
        let mut result: Vec<BlochMatrix> = template
            .iter()
            .map(|matrix| BlochMatrix::zeros(matrix.rows(), matrix.cols()))
            .collect();
        for (coefficient, stored) in coefficients.iter().zip(&self.focks) {
            for (target, source) in result.iter_mut().zip(stored) {
                target.add_scaled(source, *coefficient);
            }
        }
        Some(result)
    }
}

pub(crate) struct Fields {
    pub fock: [Vec<BlochMatrix>; 2],
    pub e_elec: f64,
    pub e_xc: f64,
    pub e_nlc: f64,
    pub grid_electron_count: Option<f64>,
}

pub(crate) fn build_fields(
    cell: &Cell,
    functional: &ResolvedFunctional,
    k_points: &[crate::math::Vec3],
    density: &[Vec<BlochMatrix>; 2],
    h_core: &[BlochMatrix],
    xc_grid: Option<&PeriodicXcGrid>,
    exchange_cutoff: f64,
    restricted: bool,
) -> Result<Fields> {
    build_fields_at_mesh(
        cell,
        functional,
        k_points,
        density,
        k_points,
        h_core,
        xc_grid,
        xc_grid,
        exchange_cutoff,
        restricted,
    )
}

fn build_fields_at_mesh(
    cell: &Cell,
    functional: &ResolvedFunctional,
    k_points: &[crate::math::Vec3],
    density: &[Vec<BlochMatrix>; 2],
    target_points: &[crate::math::Vec3],
    h_core: &[BlochMatrix],
    xc_grid: Option<&PeriodicXcGrid>,
    target_grid: Option<&PeriodicXcGrid>,
    exchange_cutoff: f64,
    restricted: bool,
) -> Result<Fields> {
    let nk = k_points.len();
    let target_nk = target_points.len();
    let inv_nk = 1.0 / nk as f64;
    let p_total = total_density(density);
    let interaction_cutoff = cell.hartree_real_space_cutoff().max(exchange_cutoff);
    // C and D are both inside the interaction supercell, so D-C can reach
    // twice its radius.  Supplying only home/first-shell D(T) silently
    // truncates J and K even if their own integral cutoff is converged.
    let total_images = crate::logger::timed("pbc-fields: density images", || {
        cell.density_images_from_k(k_points, &density_pairs(&p_total), 2.0 * interaction_cutoff)
    })?;
    let hartree_blocks = crate::logger::timed("pbc-fields: Hartree", || {
        cell.hartree_image_blocks(&total_images)
    })?;
    let hartree: Vec<BlochMatrix> = target_points
        .iter()
        .map(|k| BlochMatrix::from_pair(bloch(&hartree_blocks, *k)))
        .collect();

    let want_exchange = functional.hybrid.cx != 0.0;
    let want_lr = functional.hybrid.clr != 0.0;
    let mut exchange: [Vec<BlochMatrix>; 2] = std::array::from_fn(|_| Vec::new());
    let mut exchange_lr: [Vec<BlochMatrix>; 2] = std::array::from_fn(|_| Vec::new());
    if want_exchange || want_lr {
        let spin_count = if restricted { 1 } else { 2 };
        for spin in 0..spin_count {
            // Every density image and every exchange block once: the
            // minimum images of the Born-von Karman supercell of the mesh.
            let images = minimum_image_blocks(
                cell,
                k_points,
                cell.density_images_from_k(
                    k_points,
                    &density_pairs(&density[spin]),
                    (2.0 * exchange_cutoff).max(
                        crate::periodic::exchange_quartets::ExchangeCells::density_image_radius(
                            &cell.basis,
                            exchange_cutoff,
                        ),
                    ),
                )?,
            );
            if want_exchange {
                let blocks = minimum_image_blocks(
                    cell,
                    k_points,
                    crate::logger::timed("pbc-fields: exchange", || {
                        exchange_image_blocks(
                            &cell.basis,
                            &cell.lattice,
                            &images,
                            exchange_cutoff,
                            CoulombKernel::Coulomb,
                        )
                    }),
                );
                exchange[spin] = target_points
                    .iter()
                    .map(|k| BlochMatrix::from_pair(bloch(&blocks, *k)))
                    .collect();
            }
            if want_lr {
                let blocks = minimum_image_blocks(
                    cell,
                    k_points,
                    exchange_image_blocks(
                        &cell.basis,
                        &cell.lattice,
                        &images,
                        exchange_cutoff,
                        CoulombKernel::Erf {
                            omega: functional.hybrid.omega,
                        },
                    ),
                );
                exchange_lr[spin] = target_points
                    .iter()
                    .map(|k| BlochMatrix::from_pair(bloch(&blocks, *k)))
                    .collect();
            }
        }
        if restricted {
            exchange[1] = exchange[0].clone();
            exchange_lr[1] = exchange_lr[0].clone();
        }
    }

    let xc = match (xc_grid, target_grid) {
        (Some(grid), Some(target)) => Some(crate::logger::timed("pbc-fields: XC", || {
            xc_contribution_at_grid(functional, grid, density, target)
        })?),
        (None, None) => None,
        _ => {
            return Err(DftError::InvalidInput(
                "reference and target XC grids must be supplied together".into(),
            ))
        }
    };
    let mut fock: [Vec<BlochMatrix>; 2] = std::array::from_fn(|_| Vec::with_capacity(target_nk));
    for spin in 0..2 {
        for ik in 0..target_nk {
            let mut matrix = h_core[ik].clone();
            matrix.add_scaled(&hartree[ik], 1.0);
            if want_exchange {
                matrix.add_scaled(&exchange[spin][ik], -functional.hybrid.cx);
            }
            if want_lr {
                matrix.add_scaled(&exchange_lr[spin][ik], -functional.hybrid.clr);
            }
            if let Some(xc) = &xc {
                matrix.add_scaled(&xc.vxc[spin][ik], 1.0);
            }
            matrix.make_hermitian();
            fock[spin].push(matrix);
        }
    }
    if restricted {
        fock[1] = fock[0].clone();
    }

    let mut e_elec = 0.0;
    // Only the original integration mesh defines a total energy. Shifted
    // meshes evaluate the same field and are never used as an energy quadrature.
    for ik in 0..if target_points == k_points { nk } else { 0 } {
        e_elec += inv_nk * p_total[ik].trace_product(&h_core[ik]);
        e_elec += 0.5 * inv_nk * p_total[ik].trace_product(&hartree[ik]);
        if want_exchange {
            for spin in 0..2 {
                e_elec -= 0.5
                    * inv_nk
                    * functional.hybrid.cx
                    * density[spin][ik].trace_product(&exchange[spin][ik]);
            }
        }
        if want_lr {
            for spin in 0..2 {
                e_elec -= 0.5
                    * inv_nk
                    * functional.hybrid.clr
                    * density[spin][ik].trace_product(&exchange_lr[spin][ik]);
            }
        }
    }
    let e_xc = xc.as_ref().map(|value| value.exc).unwrap_or(0.0);
    e_elec += e_xc;
    Ok(Fields {
        fock,
        e_elec,
        e_xc,
        e_nlc: xc.as_ref().map_or(0.0, |value| value.nonlocal_energy),
        grid_electron_count: xc.map(|value| value.n_elec),
    })
}

/// Build the quadrature used for the physical Fock on a stationary mesh.
pub(crate) fn build_scf_xc_grid(
    cell: &Cell,
    functional: &ResolvedFunctional,
    options: &PeriodicScfOptions,
    k_points: &[crate::math::Vec3],
) -> Result<Option<PeriodicXcGrid>> {
    if functional.parts.is_empty() && options.skala.is_none() && options.vv10.is_none() {
        return Ok(None);
    }
    let order = AoDerivativeOrder::for_scf(functional.family);
    // VV10 depends on the total density gradient even with an LDA parent.
    let order = if options.vv10.is_some() {
        order.max(AoDerivativeOrder::Gradient)
    } else {
        order
    };
    let mut grid = PeriodicXcGrid::build(
        cell,
        k_points,
        &options.grid,
        options.grid_shell,
        options.xc_ao_cutoff.unwrap_or(cell.overlap_cutoff),
        order,
    )?;
    if let Some(model) = &options.skala {
        grid = grid.with_learned(model.clone(), cell);
    }
    if let Some(settings) = options.vv10 {
        grid = grid.with_vv10(cell, settings)?;
    }
    Ok(Some(grid))
}

/// Reconstruct the converged reference Fock operator at arbitrary Cartesian
/// k points, including incommensurate shifts. Hartree, exchange, and XC use
/// the original reference density and integration mesh.
pub fn reference_fock_at(
    cell: &Cell,
    functional: &ResolvedFunctional,
    reference: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    target_points: &[crate::math::Vec3],
) -> Result<[Vec<BlochMatrix>; 2]> {
    reference.validate_derivative_request(cell, functional, options)?;
    fock_at_density(
        cell,
        functional,
        &reference.k_points,
        &reference.density,
        options,
        target_points,
        !reference.unrestricted,
    )
}

/// Evaluate the nonlinear Fock operator for a supplied spin density. Source
/// and target meshes may differ; neither density nor particle number is
/// projected. This does not assert self-consistency of the supplied density.
pub fn fock_at_density(
    cell: &Cell,
    functional: &ResolvedFunctional,
    source_points: &[crate::math::Vec3],
    density: &[Vec<BlochMatrix>; 2],
    options: &PeriodicScfOptions,
    target_points: &[crate::math::Vec3],
    restricted: bool,
) -> Result<[Vec<BlochMatrix>; 2]> {
    validate_options(options)?;
    let n = cell.nao();
    if source_points.is_empty()
        || target_points.is_empty()
        || source_points
            .iter()
            .chain(target_points)
            .any(|k| !k.x.is_finite() || !k.y.is_finite() || !k.z.is_finite())
        || density.iter().any(|channel| {
            channel.len() != source_points.len()
                || channel.iter().any(|p| {
                    p.rows() != n
                        || p.cols() != n
                        || p.re
                            .as_slice()
                            .iter()
                            .chain(p.im.as_slice())
                            .any(|v| !v.is_finite())
                })
        })
    {
        return Err(DftError::InvalidInput(
            "reference field requires finite target k points".into(),
        ));
    }
    let xc_cutoff = options.xc_ao_cutoff.unwrap_or(cell.overlap_cutoff);
    let grid = build_scf_xc_grid(cell, functional, options, source_points)?;
    let target_grid = grid
        .as_ref()
        .map(|g| g.at_k_points(cell, target_points, xc_cutoff))
        .transpose()?;
    let h_core: Vec<_> = target_points
        .iter()
        .map(|&k| BlochMatrix::from_pair(cell.core_hamiltonian(k)))
        .collect();
    Ok(build_fields_at_mesh(
        cell,
        functional,
        source_points,
        density,
        target_points,
        &h_core,
        grid.as_ref(),
        target_grid.as_ref(),
        options.exchange_cutoff.unwrap_or(cell.overlap_cutoff),
        restricted,
    )?
    .fock)
}

/// Generalized Wolfsberg-Helmholtz Fock on each complex Bloch AO metric.
/// The diagonal is the real core-Hamiltonian diagonal and the off-diagonal
/// factor multiplies the complete complex overlap, so this reduces exactly
/// to the molecular implementation at Gamma and obeys F(-k)=F(k)*.
pub(crate) fn gwh_guess_focks(h_core: &[BlochMatrix], overlap: &[BlochMatrix]) -> Vec<BlochMatrix> {
    const K: f64 = 1.75;
    h_core
        .iter()
        .zip(overlap)
        .map(|(h, s)| {
            let n = h.rows();
            let mut f = BlochMatrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        f.re[(i, i)] = h.re[(i, i)];
                    } else {
                        let factor = 0.5 * K * (h.re[(i, i)] + h.re[(j, j)]);
                        f.re[(i, j)] = factor * s.re[(i, j)];
                        f.im[(i, j)] = factor * s.im[(i, j)];
                    }
                }
            }
            f.make_hermitian();
            f
        })
        .collect()
}

/// Build the finite-k Fock whose eigenvectors define the initial density.
///
/// SAD deliberately follows the molecular convention: the on-the-fly atomic
/// densities are spin averaged, placed identically in every primitive cell,
/// and used to build an HF Fock regardless of the target functional. The raw
/// superposition is not rescaled or returned as an SCF iterate; the ordinary
/// mesh-global occupation solve below turns the Fock into a determinant (or
/// a thermal density) with the requested charge and spin populations.
fn initial_guess_focks(
    cell: &Cell,
    k_points: &[crate::math::Vec3],
    overlap: &[BlochMatrix],
    h_core: &[BlochMatrix],
    guess: Guess,
    exchange_cutoff: f64,
) -> Result<Vec<BlochMatrix>> {
    match guess {
        Guess::Core => Ok(h_core.to_vec()),
        Guess::Gwh => Ok(gwh_guess_focks(h_core, overlap)),
        Guess::Sad => {
            // Only failure of the bounded atomic SAD calculation is a reason
            // to fall back. An error from the periodic HF field build is a
            // genuine periodic-calculation error and must propagate.
            let total = match crate::scf::sad::sad_density(cell.molecule(), cell.basis()) {
                Ok(total) => total,
                Err(_) => return Ok(gwh_guess_focks(h_core, overlap)),
            };
            let mut spin = BlochMatrix {
                re: total,
                im: Matrix::zeros(cell.nao(), cell.nao()),
            };
            spin.scale(0.5);
            let spin_density = vec![spin; k_points.len()];
            let density = [spin_density.clone(), spin_density];
            let fields = build_fields(
                cell,
                &ResolvedFunctional::hartree_fock(),
                k_points,
                &density,
                h_core,
                None,
                exchange_cutoff,
                true,
            )?;
            Ok(fields.fock[0].clone())
        }
    }
}

/// Run periodic HF, pure KS-DFT, or hybrid KS-DFT on a uniform k mesh.
///
/// Integer insulating and variational finite-temperature Fermi-Dirac
/// occupations share the same finite-k Fock build.  The latter solves a
/// chemical potential over the complete mesh and converges the Mermin free
/// energy rather than silently treating a fractional density as a zero-T
/// determinant.
pub fn run_periodic_scf(
    cell: &Cell,
    functional: &ResolvedFunctional,
    options: &PeriodicScfOptions,
) -> Result<PeriodicScfResult> {
    validate_options(options)?;
    crate::linalg::ensure_parallelism();
    // A defining pair-additive dispersion term is a function of the geometry
    // alone: it is not part of the SCF, and a caller that adds it with
    // `periodic::dispersion` says so in the options. Every other nonlocal
    // term needs its own periodic implementation.
    if let Some(term) = functional.required_nonlocal_kind().filter(|term| {
        !term.is_supplied(options.vv10.is_some(), options.pair_dispersion_from_caller)
    }) {
        return Err(DftError::Unsupported {
            feature: format!("periodic SCF for `{}`", functional.description),
            missing: format!("a periodic implementation of its defining {} nonlocal term (nonlocal term is not yet periodic)", term.name()),
        });
    }
    let n_electrons = cell.n_electrons()?;
    let (n_alpha, n_beta) = cell.molecule.n_alpha_beta_for(n_electrons)?;
    let restricted = match options.reference {
        Reference::Auto => n_alpha == n_beta,
        Reference::Rhf => {
            if n_alpha != n_beta {
                return Err(DftError::InvalidInput(format!(
                    "restricted periodic SCF requested for {n_alpha} alpha/{n_beta} beta electrons"
                )));
            }
            true
        }
        Reference::Uhf => false,
    };
    let k_points = cell.k_points(options.k_mesh);
    let (overlap, h_core) = crate::logger::timed("pbc-scf: overlap and core Hamiltonian", || {
        let overlap: Vec<BlochMatrix> = k_points
            .iter()
            .map(|k| BlochMatrix::from_pair(cell.overlap(*k)))
            .collect();
        let h_core: Vec<BlochMatrix> = k_points
            .iter()
            .map(|k| BlochMatrix::from_pair(cell.core_hamiltonian(*k)))
            .collect();
        (overlap, h_core)
    });
    let exchange_cutoff = options.exchange_cutoff.unwrap_or(cell.overlap_cutoff);
    let guess_focks = initial_guess_focks(
        cell,
        &k_points,
        &overlap,
        &h_core,
        options.guess,
        exchange_cutoff,
    )?;

    let xc_grid = crate::logger::timed("pbc-scf: XC grid", || {
        build_scf_xc_grid(cell, functional, options, &k_points)
    })?;
    // Geometry and lattice are immutable throughout this SCF. The Ewald
    // nuclear energy is therefore a loop invariant and can be expensive.
    let e_nuc = crate::logger::timed("pbc-scf: nuclear energy", || cell.nuclear_energy());
    let cached_k = k_points.clone();
    let cached_h = h_core.clone();
    run_scf_iterations(
        ScfIterationInput {
            k_points,
            overlap,
            h_core,
            guess_focks,
            n_alpha,
            n_beta,
            restricted,
            enforce_orbital_conjugation: true,
            e_nuc,
            derivative_plan: Some(PeriodicDerivativePlan::new(cell, functional, options)),
        },
        options,
        |density| {
            build_fields(
                cell,
                functional,
                &cached_k,
                density,
                &cached_h,
                xc_grid.as_ref(),
                exchange_cutoff,
                restricted,
            )
        },
    )
}

/// Shared occupation, metric, mixing and physical-stationarity iteration.
pub(crate) struct ScfIterationInput {
    pub k_points: Vec<crate::math::Vec3>,
    pub overlap: Vec<BlochMatrix>,
    pub h_core: Vec<BlochMatrix>,
    pub guess_focks: Vec<BlochMatrix>,
    pub n_alpha: usize,
    pub n_beta: usize,
    pub restricted: bool,
    pub enforce_orbital_conjugation: bool,
    pub e_nuc: f64,
    pub derivative_plan: Option<PeriodicDerivativePlan>,
}

pub(crate) fn run_scf_iterations(
    input: ScfIterationInput,
    options: &PeriodicScfOptions,
    mut evaluate: impl FnMut(&[Vec<BlochMatrix>; 2]) -> Result<Fields>,
) -> Result<PeriodicScfResult> {
    let ScfIterationInput {
        k_points,
        overlap,
        h_core,
        guess_focks,
        n_alpha,
        n_beta,
        restricted,
        enforce_orbital_conjugation,
        e_nuc,
        derivative_plan,
    } = input;
    let nao = overlap[0].rows();
    let orthogonalizers: Vec<Orthogonalizer> = overlap
        .iter()
        .map(|matrix| Orthogonalizer::canonical(&matrix.real_embedding(), options.orth_threshold))
        .collect::<Result<_>>()?;
    // An initial guess need not yet have the indirect gap of the converged
    // insulator. Defer only that one check until finalization.
    let alpha =
        solve_density_set_during_scf(&guess_focks, &orthogonalizers, n_alpha, options.occupation)?;
    let beta = if restricted {
        alpha.clone()
    } else {
        solve_density_set_during_scf(&guess_focks, &orthogonalizers, n_beta, options.occupation)?
    };
    let mut mo_energy = [alpha.energies, beta.energies];
    let mut occupation_values = [alpha.occupations, beta.occupations];
    let mut chemical_potential = [alpha.chemical_potential, beta.chemical_potential];
    let mut entropy_term = alpha.entropy_term + beta.entropy_term;
    let mut density = [alpha.density, beta.density];
    let mut energy_weighted_density = [alpha.weighted, beta.weighted];
    if enforce_orbital_conjugation {
        time_reversal_symmetrize(&mut density[0], options.k_mesh);
        time_reversal_symmetrize(&mut density[1], options.k_mesh);
        time_reversal_symmetrize(&mut energy_weighted_density[0], options.k_mesh);
        time_reversal_symmetrize(&mut energy_weighted_density[1], options.k_mesh);
    }

    let active_spins = if restricted { 1 } else { 2 };
    let mut diis = PeriodicDiis::new(options.diis_size);
    let mut adiis = PeriodicAdiis::new(options.diis_size);
    if options.use_diis
        && options.use_adiis
        && matches!(options.occupation, OccupationScheme::FermiDirac { .. })
    {
        crate::logger::log(
            "[pbc-scf] finite-temperature occupations use CDIIS; zero-temperature ADIIS has no entropy curvature",
        );
    }
    let mut previous_free_energy = f64::INFINITY;
    let mut density_change = f64::INFINITY;
    let mut iterations = 0;
    let mut converged = false;
    let mut orbital_residual = None;
    let mut final_fields = None;
    let mut require_accelerated_step = false;

    for iteration in 0..options.max_iter {
        let allow_canonical_check = !require_accelerated_step;
        require_accelerated_step = false;
        iterations = iteration + 1;
        let fields = evaluate(&density)?;
        let total_energy = fields.e_elec + e_nuc;
        let free_energy = total_energy - entropy_term;
        let energy_change = (free_energy - previous_free_energy).abs();
        let mut errors = Vec::new();
        let mut max_error = 0.0_f64;
        let mut active_focks = Vec::new();
        for spin in 0..active_spins {
            let thermal_target =
                if matches!(options.occupation, OccupationScheme::FermiDirac { .. }) {
                    Some(
                        solve_density_set_during_scf(
                            &fields.fock[spin],
                            &orthogonalizers,
                            if spin == 0 { n_alpha } else { n_beta },
                            options.occupation,
                        )?
                        .density,
                    )
                } else {
                    None
                };
            for ik in 0..k_points.len() {
                let mut error = commutator_error(
                    &fields.fock[spin][ik],
                    &density[spin][ik],
                    &overlap[ik],
                    &orthogonalizers[ik],
                );
                if let Some(target) = &thermal_target {
                    error.extend(thermal_density_error(
                        &target[ik],
                        &density[spin][ik],
                        &overlap[ik],
                        &orthogonalizers[ik],
                        options.occupation.temperature_kelvin()
                            * crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K,
                    ));
                }
                max_error = error
                    .iter()
                    .fold(max_error, |largest, value| largest.max(value.abs()));
                errors.extend(error);
                active_focks.push(fields.fock[spin][ik].clone());
            }
        }
        crate::logger::log(&format!(
            "[pbc-scf] iter {iterations:>3} E={total_energy:>20.12} \
                F={free_energy:>20.12} dF={energy_change:>10.3e} \
                residual={max_error:>10.3e} rms(dP)={density_change:>10.3e}"
        ));
        if iteration > 0
            && allow_canonical_check
            && energy_change < options.e_tol
            && max_error < options.conv_tol_grad
            && density_change < options.p_tol
        {
            // DIIS diagonalizes an extrapolated Fock, so its stored orbital
            // metadata need not belong to the physical F[P] tested above.
            // Canonicalize once against that final physical Fock. This also
            // performs the public indirect-gap validation for an insulator.
            let mut final_alpha = solve_density_set(
                &fields.fock[0],
                &orthogonalizers,
                n_alpha,
                options.occupation,
            )?;
            let mut final_beta = if restricted {
                final_alpha.clone()
            } else {
                solve_density_set(
                    &fields.fock[1],
                    &orthogonalizers,
                    n_beta,
                    options.occupation,
                )?
            };
            if enforce_orbital_conjugation {
                time_reversal_symmetrize(&mut final_alpha.density, options.k_mesh);
                time_reversal_symmetrize(&mut final_alpha.weighted, options.k_mesh);
                time_reversal_symmetrize(&mut final_beta.density, options.k_mesh);
                time_reversal_symmetrize(&mut final_beta.weighted, options.k_mesh);
            }

            let canonical_density_change = density_rms_change(
                [&final_alpha.density, &final_beta.density],
                [&density[0], &density[1]],
                active_spins,
                nao,
            );
            let canonical_entropy = final_alpha.entropy_term + final_beta.entropy_term;
            let canonical_free_energy = total_energy - canonical_entropy;
            let canonical_energy_change = (canonical_free_energy - previous_free_energy).abs();
            let canonical_converged =
                canonical_density_change < options.p_tol && canonical_energy_change < options.e_tol;

            let DensitySet {
                energies: alpha_energies,
                occupations: alpha_occupations,
                density: alpha_density,
                weighted: alpha_weighted,
                chemical_potential: alpha_mu,
                entropy_term: _,
            } = final_alpha;
            let DensitySet {
                energies: beta_energies,
                occupations: beta_occupations,
                density: beta_density,
                weighted: beta_weighted,
                chemical_potential: beta_mu,
                entropy_term: _,
            } = final_beta;
            mo_energy = [alpha_energies, beta_energies];
            occupation_values = [alpha_occupations, beta_occupations];
            chemical_potential = [alpha_mu, beta_mu];
            density = [alpha_density, beta_density];
            energy_weighted_density = [alpha_weighted, beta_weighted];
            entropy_term = canonical_entropy;
            density_change = canonical_density_change;
            previous_free_energy = canonical_free_energy;

            if canonical_converged {
                // Canonicalization changes P. The field above belongs to the
                // preceding P, so accepting that stored pair would hide its
                // physical SCF residual. Rebuild F[P] and the energy at the
                // exact density that will be returned, then validate its
                // canonical occupation map and thermodynamic consistency.
                let refreshed = evaluate(&density)?;
                let mut targets = [
                    solve_density_set(
                        &refreshed.fock[0],
                        &orthogonalizers,
                        n_alpha,
                        options.occupation,
                    )?,
                    solve_density_set(
                        &refreshed.fock[1],
                        &orthogonalizers,
                        n_beta,
                        options.occupation,
                    )?,
                ];
                if enforce_orbital_conjugation {
                    for target in &mut targets {
                        time_reversal_symmetrize(&mut target.density, options.k_mesh);
                        time_reversal_symmetrize(&mut target.weighted, options.k_mesh);
                    }
                }
                let occupation_change = density_rms_change(
                    [&targets[0].density, &targets[1].density],
                    [&density[0], &density[1]],
                    active_spins,
                    nao,
                );
                let mut physical_residual = 0.0_f64;
                for spin in 0..active_spins {
                    for ik in 0..k_points.len() {
                        let mut error = commutator_error(
                            &refreshed.fock[spin][ik],
                            &density[spin][ik],
                            &overlap[ik],
                            &orthogonalizers[ik],
                        );
                        if matches!(options.occupation, OccupationScheme::FermiDirac { .. }) {
                            error.extend(thermal_density_error(
                                &targets[spin].density[ik],
                                &density[spin][ik],
                                &overlap[ik],
                                &orthogonalizers[ik],
                                options.occupation.temperature_kelvin()
                                    * crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K,
                            ));
                        }
                        physical_residual = error
                            .iter()
                            .fold(physical_residual, |largest, value| largest.max(value.abs()));
                    }
                }
                let refreshed_entropy = targets[0].entropy_term + targets[1].entropy_term;
                // canonical_entropy belongs to the returned density, whose
                // occupations were formed before this physical field rebuild.
                let refreshed_free_energy = refreshed.e_elec + e_nuc - canonical_entropy;
                let refreshed_energy_change = (refreshed_free_energy - canonical_free_energy).abs();
                let entropy_change = (refreshed_entropy - canonical_entropy).abs();
                crate::logger::log(&format!("[pbc-scf] physical final check F={refreshed_free_energy:.12} residual={physical_residual:.3e} rms(dP)={occupation_change:.3e} dF={refreshed_energy_change:.3e} d(TS)={entropy_change:.3e}"));
                if physical_residual < options.conv_tol_grad
                    && occupation_change < options.p_tol
                    && refreshed_energy_change < options.e_tol
                    && entropy_change < options.e_tol
                {
                    for (spin, target) in targets.into_iter().enumerate() {
                        mo_energy[spin] = target.energies;
                        occupation_values[spin] = target.occupations;
                        chemical_potential[spin] = target.chemical_potential;
                        energy_weighted_density[spin] = target.weighted;
                    }
                    converged = true;
                    orbital_residual = Some(physical_residual);
                    final_fields = Some(refreshed);
                    break;
                }
            }
            require_accelerated_step = true;
            crate::logger::log(&format!(
                "[pbc-scf] final canonicalization requires another iteration: dF={canonical_energy_change:.3e} rms(dP)={canonical_density_change:.3e}"
            ));
            continue;
        }

        if options.use_diis {
            diis.push(active_focks.clone(), errors);
            if periodic_adiis_eligible(options) {
                let active_densities: Vec<BlochMatrix> = (0..active_spins)
                    .flat_map(|spin| density[spin].iter().cloned())
                    .collect();
                adiis.push(active_densities, active_focks.clone());
            }
        }
        let mut work = if periodic_adiis_selected(options, max_error) {
            adiis.extrapolate().unwrap_or(active_focks)
        } else if options.use_diis {
            diis.extrapolate().unwrap_or(active_focks)
        } else {
            active_focks
        };
        if options.level_shift > 0.0 && max_error > options.level_shift_until {
            for spin in 0..active_spins {
                for ik in 0..k_points.len() {
                    let index = spin * k_points.len() + ik;
                    work[index] = level_shifted_fock(
                        &work[index],
                        &overlap[ik],
                        &density[spin][ik],
                        options.level_shift,
                    );
                }
            }
        }
        let mut new_density: [Vec<BlochMatrix>; 2] = std::array::from_fn(|_| Vec::new());
        let mut new_weighted: [Vec<BlochMatrix>; 2] = std::array::from_fn(|_| Vec::new());
        let mut new_energies: [Vec<Vec<f64>>; 2] = std::array::from_fn(|_| Vec::new());
        let mut new_occupations: [Vec<Vec<f64>>; 2] = std::array::from_fn(|_| Vec::new());
        let mut new_chemical_potential = [None, None];
        let mut new_entropy_term = 0.0;
        for spin in 0..active_spins {
            let start = spin * k_points.len();
            let n_electrons = if spin == 0 { n_alpha } else { n_beta };
            let solved = solve_density_set_during_scf(
                &work[start..start + k_points.len()],
                &orthogonalizers,
                n_electrons,
                options.occupation,
            )?;
            new_energies[spin] = solved.energies;
            new_occupations[spin] = solved.occupations;
            new_weighted[spin] = solved.weighted;
            new_chemical_potential[spin] = solved.chemical_potential;
            new_entropy_term += solved.entropy_term;
            for (ik, mut next) in solved.density.into_iter().enumerate() {
                // Mixing density matrices destroys the one-to-one relation
                // between P, occupations and the noninteracting entropy.
                // Keep finite-temperature iterates variational; DIIS remains
                // available for their convergence. Integer occupations have
                // zero entropy and may use the traditional density damping.
                if options.occupation == OccupationScheme::Insulator
                    && options.damping > 0.0
                    && max_error > options.damping_until_error
                {
                    next.scale(1.0 - options.damping);
                    next.add_scaled(&density[spin][ik], options.damping);
                    next.make_hermitian();
                }
                new_density[spin].push(next);
            }
            if enforce_orbital_conjugation {
                time_reversal_symmetrize(&mut new_density[spin], options.k_mesh);
                time_reversal_symmetrize(&mut new_weighted[spin], options.k_mesh);
            }
        }
        if restricted {
            new_density[1] = new_density[0].clone();
            new_weighted[1] = new_weighted[0].clone();
            new_energies[1] = new_energies[0].clone();
            new_occupations[1] = new_occupations[0].clone();
            new_chemical_potential[1] = new_chemical_potential[0];
            new_entropy_term *= 2.0;
        }
        density_change = density_rms_change(
            [&new_density[0], &new_density[1]],
            [&density[0], &density[1]],
            active_spins,
            nao,
        );
        density = new_density;
        energy_weighted_density = new_weighted;
        mo_energy = new_energies;
        occupation_values = new_occupations;
        chemical_potential = new_chemical_potential;
        entropy_term = new_entropy_term;
        previous_free_energy = free_energy;
        final_fields = Some(fields);
    }

    if !converged {
        return Err(DftError::ScfNotConverged {
            iterations,
            error: previous_free_energy,
        });
    }
    // The returned Fock and energy were rebuilt at the returned density.
    // Canonical occupations and orbital metadata satisfy the same physical
    // residual, density and free-energy gates; no later density replacement occurs.
    let fields = final_fields.expect("a converged iteration has fields");
    Ok(PeriodicScfResult {
        orbital_conjugation_constraint: enforce_orbital_conjugation,
        energy: fields.e_elec + e_nuc,
        free_energy: fields.e_elec + e_nuc - entropy_term,
        entropy_term,
        e_nuc,
        e_elec: fields.e_elec,
        e_xc: fields.e_xc,
        e_nlc: fields.e_nlc,
        k_points,
        mo_energy,
        occupations: occupation_values,
        chemical_potential,
        density,
        energy_weighted_density,
        fock: fields.fock,
        overlap,
        h_core,
        n_alpha,
        n_beta,
        unrestricted: !restricted,
        grid_electron_count: fields.grid_electron_count,
        orbital_residual,
        converged,
        iterations,
        derivative_plan,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thermal_residual_detects_commuting_occupation_error_in_the_metric() {
        let mut s = BlochMatrix::zeros(2, 2);
        s.re[(0, 0)] = 2.0;
        s.re[(1, 1)] = 3.0;
        let orth = Orthogonalizer::canonical(&s.real_embedding(), 1e-9).unwrap();
        let mut f = BlochMatrix::zeros(2, 2);
        f.re[(0, 0)] = -0.4;
        f.re[(1, 1)] = 1.2;
        let mut current = BlochMatrix::zeros(2, 2);
        current.re[(0, 0)] = 0.4 / 2.0;
        current.re[(1, 1)] = 0.6 / 3.0;
        let mut target = BlochMatrix::zeros(2, 2);
        target.re[(0, 0)] = 0.3 / 2.0;
        target.re[(1, 1)] = 0.7 / 3.0;
        assert!(commutator_error(&f, &current, &s, &orth)
            .iter()
            .all(|x| x.abs() < 1e-15));
        let error = thermal_density_error(&target, &current, &s, &orth, 0.1);
        let max = error.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        assert!((max - 0.01).abs() < 1e-14);
        assert!((error.iter().map(|x| x * x).sum::<f64>() - 0.0004).abs() < 1e-14);
        assert!(thermal_density_error(&current, &current, &s, &orth, 0.1)
            .iter()
            .all(|x| *x == 0.0));
    }
    use crate::basis::data;
    use crate::math::Vec3;
    use crate::periodic::Lattice;
    use crate::system::{Atom, Molecule};
    use crate::xc::{resolve, FunctionalSpec};

    #[test]
    fn periodic_options_reject_nonfinite_or_unphysical_controls() {
        for options in [
            PeriodicScfOptions {
                e_tol: f64::INFINITY,
                ..PeriodicScfOptions::default()
            },
            PeriodicScfOptions {
                orth_threshold: f64::NAN,
                ..PeriodicScfOptions::default()
            },
            PeriodicScfOptions {
                damping_until_error: -1.0,
                ..PeriodicScfOptions::default()
            },
            PeriodicScfOptions {
                adiis_until_error: f64::NAN,
                ..PeriodicScfOptions::default()
            },
            PeriodicScfOptions {
                level_shift: f64::NAN,
                ..PeriodicScfOptions::default()
            },
            PeriodicScfOptions {
                level_shift_until: -1.0,
                ..PeriodicScfOptions::default()
            },
            PeriodicScfOptions {
                exchange_cutoff: Some(-2.0),
                ..PeriodicScfOptions::default()
            },
        ] {
            assert!(validate_options(&options).is_err());
        }
        let mut bad_grid = PeriodicScfOptions::default();
        bad_grid.grid.weight_cutoff = f64::INFINITY;
        assert!(validate_options(&bad_grid).is_err());
    }

    #[test]
    fn periodic_adiis_selector_respects_temperature_threshold_and_switches() {
        let mut options = PeriodicScfOptions::default();
        assert!(periodic_adiis_selected(&options, 0.100_001));
        assert!(!periodic_adiis_selected(&options, 0.1));

        options.occupation = OccupationScheme::FermiDirac {
            temperature_kelvin: 300.0,
        };
        assert!(!periodic_adiis_selected(&options, 1.0));
        options.occupation = OccupationScheme::Insulator;
        options.use_diis = false;
        assert!(!periodic_adiis_selected(&options, 1.0));
        options.use_diis = true;
        options.use_adiis = false;
        assert!(!periodic_adiis_selected(&options, 1.0));
    }

    #[test]
    fn periodic_adiis_matches_a_complex_brute_force_objective() {
        let q = BlochMatrix {
            re: Matrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, -1.0]),
            im: Matrix::from_row_major(2, 2, vec![0.0, -0.5, 0.5, 0.0]),
        };
        // R is orthogonal to Q only when both the real and imaginary pieces
        // enter Re Tr[Q^dagger R]. Dropping Im would move the optimum.
        let mut r = q.clone();
        for value in r.re.as_mut_slice() {
            *value *= 0.25;
        }
        for value in r.im.as_mut_slice() {
            *value *= -1.0;
        }
        let d0 = q.clone();
        let d1 = BlochMatrix::zeros(2, 2);
        let mut f0 = r.clone();
        f0.add_scaled(&q, 0.7);
        let mut f1 = r.clone();
        f1.add_scaled(&q, -0.3);

        let mut adiis = PeriodicAdiis::new(2);
        adiis.push(vec![d0.clone()], vec![f0.clone()]);
        adiis.push(vec![d1.clone()], vec![f1.clone()]);
        let out = adiis.extrapolate().unwrap();
        let coefficient = (out[0].re[(0, 0)] - f1.re[(0, 0)]) / (f0.re[(0, 0)] - f1.re[(0, 0)]);

        let manual_dot = |a: &BlochMatrix, b: &BlochMatrix| {
            a.re.as_slice()
                .iter()
                .zip(b.re.as_slice())
                .map(|(x, y)| x * y)
                .chain(
                    a.im.as_slice()
                        .iter()
                        .zip(b.im.as_slice())
                        .map(|(x, y)| x * y),
                )
                .sum::<f64>()
        };
        let mut dd = d0;
        dd.add_scaled(&d1, -1.0);
        let mut df = f0.clone();
        df.add_scaled(&f1, -1.0);
        let g = manual_dot(&dd, &f1);
        let h = manual_dot(&dd, &df);
        let objective = |c: f64| 2.0 * g * c + h * c * c;
        let brute = (0..=10_000)
            .map(|i| i as f64 / 10_000.0)
            .min_by(|a, b| objective(*a).total_cmp(&objective(*b)))
            .unwrap();

        assert!((coefficient - brute).abs() < 2e-3);
        assert!(objective(coefficient) <= objective(0.0) + 1e-12);
        assert!(objective(coefficient) <= objective(1.0) + 1e-12);
        assert!(out[0].im.as_slice().iter().any(|value| value.abs() > 0.1));
        assert!(out[0].hermiticity_error() < 1e-14);

        let mut capacity_one = PeriodicAdiis::new(1);
        capacity_one.push(vec![d1.clone()], vec![f1.clone()]);
        capacity_one.push(vec![d1], vec![f1]);
        assert!(capacity_one.extrapolate().is_none());
    }

    #[test]
    fn periodic_level_shift_targets_the_complex_virtual_projector() {
        let overlap = BlochMatrix {
            re: Matrix::identity(2),
            im: Matrix::zeros(2, 2),
        };
        // Projector onto c=(1,i)/sqrt(2), so both the real and imaginary
        // blocks of I-P are observable.
        let density = BlochMatrix {
            re: Matrix::from_row_major(2, 2, vec![0.5, 0.0, 0.0, 0.5]),
            im: Matrix::from_row_major(2, 2, vec![0.0, -0.5, 0.5, 0.0]),
        };
        let fock = BlochMatrix {
            re: Matrix::from_row_major(2, 2, vec![-0.7, 0.13, 0.13, 0.2]),
            im: Matrix::from_row_major(2, 2, vec![0.0, -0.08, 0.08, 0.0]),
        };
        let lambda = 0.6;
        let shifted = level_shifted_fock(&fock, &overlap, &density, lambda);
        for i in 0..2 {
            for j in 0..2 {
                let identity = if i == j { 1.0 } else { 0.0 };
                let want_re = fock.re[(i, j)] + lambda * (identity - density.re[(i, j)]);
                let want_im = fock.im[(i, j)] - lambda * density.im[(i, j)];
                assert!((shifted.re[(i, j)] - want_re).abs() < 1e-14);
                assert!((shifted.im[(i, j)] - want_im).abs() < 1e-14);
            }
        }
        assert!(shifted.hermiticity_error() < 1e-14);
        assert_eq!(level_shifted_fock(&fock, &overlap, &density, 0.0), fock);
    }

    /// Classification: exact/sanity. The raw neutral-atom superposition has
    /// the correct complete-mesh electron count, but is used only to build an
    /// HF Fock. Re-diagonalization must produce finite-k determinants with the
    /// requested occupations, Hermiticity and time-reversal covariance.
    #[test]
    fn periodic_sad_seed_builds_a_physical_finite_k_guess() {
        assert_eq!(PeriodicScfOptions::default().guess, Guess::Core);
        let molecule = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(-0.7, 0.13, -0.09),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.7, -0.13, 0.09),
            },
        ]);
        let set = data::load("sto-3g").unwrap();
        let cell = Cell::new(
            molecule,
            Lattice::new(
                Vec3::new(6.0, 0.0, 0.0),
                Vec3::new(0.0, 12.0, 0.0),
                Vec3::new(0.0, 0.0, 12.0),
            ),
            &set,
            6.1,
        )
        .unwrap();
        let k_points = cell.k_points([3, 1, 1]);
        let overlap: Vec<_> = k_points
            .iter()
            .map(|k| BlochMatrix::from_pair(cell.overlap(*k)))
            .collect();
        let h_core: Vec<_> = k_points
            .iter()
            .map(|k| BlochMatrix::from_pair(cell.core_hamiltonian(*k)))
            .collect();

        let total_sad = crate::scf::sad::sad_density(cell.molecule(), cell.basis()).unwrap();
        let raw_total_count = overlap
            .iter()
            .map(|s| {
                let raw_spin = BlochMatrix {
                    re: total_sad.clone(),
                    im: Matrix::zeros(cell.nao(), cell.nao()),
                };
                raw_spin.trace_product(s)
            })
            .sum::<f64>()
            * (2.0 * 0.5 / k_points.len() as f64);
        assert!((raw_total_count - cell.n_electrons().unwrap() as f64).abs() < 1e-9);

        let sad_fock =
            initial_guess_focks(&cell, &k_points, &overlap, &h_core, Guess::Sad, 6.1).unwrap();
        assert!(sad_fock
            .iter()
            .zip(&h_core)
            .any(|(sad, core)| sad.re.max_abs_diff(&core.re) > 1e-8
                || sad.im.max_abs_diff(&core.im) > 1e-8));
        for ik in 0..k_points.len() {
            // Time reversal is modulo a reciprocal lattice vector: on this
            // unshifted [0, 1/3, 2/3] mesh the non-Gamma partners are 1 <-> 2.
            let partner = (k_points.len() - ik) % k_points.len();
            assert!(sad_fock[ik].re.max_abs_diff(&sad_fock[partner].re) < 2e-12);
            assert!(sad_fock[ik]
                .im
                .as_slice()
                .iter()
                .zip(sad_fock[partner].im.as_slice())
                .all(|(a, b)| (a + b).abs() < 2e-12));
        }

        let orthogonalizers: Vec<_> = overlap
            .iter()
            .map(|s| Orthogonalizer::canonical(&s.real_embedding(), 1e-8).unwrap())
            .collect();
        let solved =
            solve_density_set(&sad_fock, &orthogonalizers, 1, OccupationScheme::Insulator).unwrap();
        for ik in 0..k_points.len() {
            let p = solved.density[ik].real_embedding();
            let s = overlap[ik].real_embedding();
            let psp = p.matmul_seq(&s).matmul_seq(&p);
            assert!((solved.density[ik].trace_product(&overlap[ik]) - 1.0).abs() < 2e-10);
            assert!(solved.density[ik].hermiticity_error() < 2e-12);
            assert!(psp.max_abs_diff(&p) < 2e-10);
        }
    }

    #[test]
    fn derivative_plan_tracks_only_energy_defining_inputs() {
        let molecule = Molecule::new(vec![Atom {
            z: 2,
            position: Vec3::zero(),
        }]);
        let set = data::load("sto-3g").unwrap();
        let cell = Cell::new(molecule.clone(), Lattice::cubic(10.0), &set, 7.0).unwrap();
        let rebuilt = Cell::new(molecule, Lattice::cubic(10.0), &set, 7.0).unwrap();
        let hf = ResolvedFunctional::hartree_fock();
        let base_options = PeriodicScfOptions::default();
        let base = PeriodicDerivativePlan::new(&cell, &hf, &base_options);

        let mut convergence_only = base_options.clone();
        convergence_only.max_iter += 17;
        convergence_only.e_tol *= 0.1;
        assert_eq!(
            base,
            PeriodicDerivativePlan::new(&cell, &hf, &convergence_only)
        );

        let mut explicit_same_cutoff = base_options.clone();
        explicit_same_cutoff.exchange_cutoff = Some(cell.overlap_cutoff());
        assert_eq!(
            base,
            PeriodicDerivativePlan::new(&cell, &hf, &explicit_same_cutoff)
        );

        let mut changed_cutoff = base_options.clone();
        changed_cutoff.exchange_cutoff = Some(cell.overlap_cutoff() * 1.1);
        assert_ne!(
            base,
            PeriodicDerivativePlan::new(&cell, &hf, &changed_cutoff)
        );
        assert_ne!(
            base,
            PeriodicDerivativePlan::new(&rebuilt, &hf, &base_options),
            "a separately rebuilt Cell must not accept another Cell's SCF result"
        );

        let pbe = resolve(&FunctionalSpec::Combined("pbe".to_string())).unwrap();
        let pbe_plan = PeriodicDerivativePlan::new(&cell, &pbe, &base_options);
        assert_ne!(base, pbe_plan);
        let mut changed_grid = base_options;
        changed_grid.grid.level += 1;
        assert_ne!(
            pbe_plan,
            PeriodicDerivativePlan::new(&cell, &pbe, &changed_grid)
        );
    }

    fn weighted_density_reference(n: usize, coefficients: &Matrix, weights: &[f64]) -> BlochMatrix {
        let mut embedded = Matrix::zeros(2 * n, 2 * n);
        for (band, &weight) in weights.iter().enumerate() {
            for duplicate in 0..2 {
                let column = 2 * band + duplicate;
                for i in 0..2 * n {
                    let left = weight * coefficients[(i, column)];
                    for j in 0..2 * n {
                        embedded[(i, j)] += left * coefficients[(j, column)];
                    }
                }
            }
        }
        let mut density = BlochMatrix::from_real_embedding(&embedded);
        density.make_hermitian();
        density
    }

    fn fp_sw_error(
        fock: &BlochMatrix,
        density: &BlochMatrix,
        overlap: &BlochMatrix,
        weighted: &BlochMatrix,
    ) -> f64 {
        let fp = fock.real_embedding().matmul_seq(&density.real_embedding());
        let sw = overlap
            .real_embedding()
            .matmul_seq(&weighted.real_embedding());
        fp.max_abs_diff(&sw)
    }

    #[test]
    fn direct_complex_blocks_match_the_old_full_embedding_gram() {
        let coefficients = Matrix::from_row_major(
            4,
            4,
            vec![
                0.71, -0.12, 0.33, 0.28, -0.41, 0.63, 0.19, -0.52, 0.24, 0.31, -0.67, 0.38, 0.49,
                -0.22, 0.46, 0.57,
            ],
        );
        let weights = [0.83, -0.27];
        let direct = weighted_density_from_embedding(2, &coefficients, &weights);
        let reference = weighted_density_reference(2, &coefficients, &weights);
        assert!(direct.re.max_abs_diff(&reference.re) < 2e-15);
        assert!(direct.im.max_abs_diff(&reference.im) < 2e-15);
    }

    #[test]
    fn unfinished_scf_may_cross_a_gap_but_final_occupations_may_not() {
        let focks = vec![
            BlochMatrix {
                re: Matrix::from_row_major(2, 2, vec![-1.0, 0.0, 0.0, -0.9]),
                im: Matrix::zeros(2, 2),
            },
            BlochMatrix {
                re: Matrix::from_row_major(2, 2, vec![0.2, 0.0, 0.0, 0.3]),
                im: Matrix::zeros(2, 2),
            },
        ];
        let overlap = BlochMatrix {
            re: Matrix::identity(2),
            im: Matrix::zeros(2, 2),
        };
        let orth = Orthogonalizer::canonical(&overlap.real_embedding(), 1e-12).unwrap();
        let orthogonalizers = vec![orth.clone(), orth];
        assert!(
            solve_density_set(&focks, &orthogonalizers, 1, OccupationScheme::Insulator,).is_err()
        );
        let intermediate =
            solve_density_set_during_scf(&focks, &orthogonalizers, 1, OccupationScheme::Insulator)
                .unwrap();
        assert_eq!(intermediate.occupations, vec![vec![1.0, 0.0]; 2]);
    }

    #[test]
    fn complex_generalized_density_has_the_right_occupation() {
        let s = BlochMatrix {
            re: Matrix::from_row_major(2, 2, vec![1.0, 0.1, 0.1, 1.0]),
            im: Matrix::from_row_major(2, 2, vec![0.0, 0.08, -0.08, 0.0]),
        };
        let f = BlochMatrix {
            re: Matrix::from_row_major(2, 2, vec![-0.8, 0.25, 0.25, 0.4]),
            im: Matrix::from_row_major(2, 2, vec![0.0, -0.12, 0.12, 0.0]),
        };
        let orth = Orthogonalizer::canonical(&s.real_embedding(), 1e-10).unwrap();
        let solved = solve_density_set(
            std::slice::from_ref(&f),
            std::slice::from_ref(&orth),
            1,
            OccupationScheme::Insulator,
        )
        .unwrap();
        let density = &solved.density[0];
        assert!((density.trace_product(&s) - 1.0).abs() < 2e-12);
        assert!(density.hermiticity_error() < 1e-13);
        assert!(fp_sw_error(&f, density, &s, &solved.weighted[0]) < 3e-12);
        let error = commutator_error(&f, density, &s, &orth);
        assert!(error.iter().all(|value| value.abs() < 2e-12));
    }

    #[test]
    fn finite_k_hybrid_scf_is_stationary_and_number_conserving() {
        // One occupied band per spin makes the answer independent of an
        // initial guess while still exercising nontrivial S(k), J(k), K(k)
        // and Vxc(k) on a time-reversal-complete three-point mesh.
        let molecule = Molecule::new(vec![Atom {
            z: 2,
            position: Vec3::zero(),
        }]);
        let set = data::load("sto-3g").unwrap();
        let cell = Cell::new(
            molecule,
            Lattice::new(
                Vec3::new(6.0, 0.0, 0.0),
                Vec3::new(0.0, 12.0, 0.0),
                Vec3::new(0.0, 0.0, 12.0),
            ),
            &set,
            6.1,
        )
        .unwrap();
        let functional = resolve(&FunctionalSpec::Combined("pbe0".to_string())).unwrap();
        let options = PeriodicScfOptions {
            k_mesh: [3, 1, 1],
            max_iter: 12,
            damping: 0.0,
            exchange_cutoff: Some(6.1),
            xc_ao_cutoff: Some(6.1),
            grid_shell: 1,
            grid: GridOptions {
                radial_override: Some(12),
                angular_override: Some(9),
                prune: false,
                ..GridOptions::default()
            },
            ..PeriodicScfOptions::default()
        };
        let result = run_periodic_scf(&cell, &functional, &options).unwrap();
        assert!(result.converged);
        assert!(result.energy.is_finite());
        assert!(result.e_xc.abs() > 1e-5);
        assert_eq!(result.k_points.len(), 3);
        let total = result.total_density();
        for ik in 0..3 {
            let electrons = total[ik].trace_product(&result.overlap[ik]);
            assert!(
                (electrons - 2.0).abs() < 2e-10,
                "k point {ik} has {electrons} electrons"
            );
            for spin in 0..2 {
                let orth = Orthogonalizer::canonical(
                    &result.overlap[ik].real_embedding(),
                    options.orth_threshold,
                )
                .unwrap();
                let residual = commutator_error(
                    &result.fock[spin][ik],
                    &result.density[spin][ik],
                    &result.overlap[ik],
                    &orth,
                );
                assert!(residual.iter().all(|value| value.abs() < 1e-9));
            }
        }
        let integrated = result.grid_electron_count.unwrap();
        assert!(
            (integrated - 2.0).abs() < 8e-2,
            "periodic grid integrates {integrated} electrons"
        );
        assert!(
            result.density[0][1]
                .re
                .max_abs_diff(&result.density[0][2].re)
                < 1e-12
        );
        assert!(result.density[0][1]
            .im
            .as_slice()
            .iter()
            .zip(result.density[0][2].im.as_slice())
            .all(|(a, b)| (a + b).abs() < 1e-12));
    }

    #[test]
    fn finite_k_r2scan_meta_gga_scf_is_stationary() {
        // The one occupied band is enough to exercise rho, grad-rho and
        // tau from every k point, including the v_tau generalized-KS term,
        // without making this normal regression gate exchange-dominated.
        let molecule = Molecule::new(vec![Atom {
            z: 2,
            position: Vec3::zero(),
        }]);
        let set = data::load("sto-3g").unwrap();
        let cell = Cell::new(
            molecule,
            Lattice::new(
                Vec3::new(6.0, 0.0, 0.0),
                Vec3::new(0.0, 12.0, 0.0),
                Vec3::new(0.0, 0.0, 12.0),
            ),
            &set,
            6.1,
        )
        .unwrap();
        let functional = resolve(&FunctionalSpec::Combined("r2scan".to_string())).unwrap();
        let options = PeriodicScfOptions {
            k_mesh: [3, 1, 1],
            max_iter: 12,
            damping: 0.0,
            xc_ao_cutoff: Some(6.1),
            grid_shell: 1,
            grid: GridOptions {
                radial_override: Some(9),
                angular_override: Some(7),
                prune: false,
                ..GridOptions::default()
            },
            ..PeriodicScfOptions::default()
        };
        let result = run_periodic_scf(&cell, &functional, &options).unwrap();
        assert!(result.converged);
        assert!(result.energy.is_finite());
        assert!(result.e_xc.abs() > 1e-5);
        assert_eq!(result.k_points.len(), 3);
        for ik in 0..3 {
            let total = result.density[0][ik].trace_product(&result.overlap[ik])
                + result.density[1][ik].trace_product(&result.overlap[ik]);
            assert!((total - 2.0).abs() < 2e-10);
            let orth = Orthogonalizer::canonical(
                &result.overlap[ik].real_embedding(),
                options.orth_threshold,
            )
            .unwrap();
            let residual = commutator_error(
                &result.fock[0][ik],
                &result.density[0][ik],
                &result.overlap[ik],
                &orth,
            );
            assert!(residual.iter().all(|value| value.abs() < 1e-9));
        }
        assert!(result.fock[0][1].re.max_abs_diff(&result.fock[0][2].re) < 2e-10);
        assert!(result.fock[0][1]
            .im
            .as_slice()
            .iter()
            .zip(result.fock[0][2].im.as_slice())
            .all(|(a, b)| (a + b).abs() < 2e-10));
    }

    /// At fixed electron number, the derivative of the noninteracting
    /// Mermin free energy with respect to the AO metric is `-Tr(W dS)`, with
    /// `W = C f epsilon C^dagger`.  This catches the tempting but incorrect
    /// `P F P` expression, which contains `f^2` under smearing.
    #[test]
    fn fractional_occupation_pulay_matrix_is_free_energy_derivative() {
        let fock = BlochMatrix {
            re: Matrix::from_row_major(2, 2, vec![-0.6, 0.17, 0.17, 0.25]),
            im: Matrix::from_row_major(2, 2, vec![0.0, -0.09, 0.09, 0.0]),
        };
        let overlap0 = BlochMatrix {
            re: Matrix::from_row_major(2, 2, vec![1.0, 0.18, 0.18, 1.1]),
            im: Matrix::from_row_major(2, 2, vec![0.0, 0.04, -0.04, 0.0]),
        };
        let ds = BlochMatrix {
            re: Matrix::from_row_major(2, 2, vec![0.08, -0.03, -0.03, -0.05]),
            im: Matrix::zeros(2, 2),
        };
        let scheme = OccupationScheme::FermiDirac {
            temperature_kelvin: 30_000.0,
        };
        let evaluate = |amount: f64| {
            let mut overlap = overlap0.clone();
            overlap.add_scaled(&ds, amount);
            let orth = Orthogonalizer::canonical(&overlap.real_embedding(), 1e-10).unwrap();
            let solved =
                solve_density_set(std::slice::from_ref(&fock), &[orth], 1, scheme).unwrap();
            let band_energy: f64 = solved.occupations[0]
                .iter()
                .zip(&solved.energies[0])
                .map(|(&occupation, &energy)| occupation * energy)
                .sum();
            (band_energy - solved.entropy_term, solved)
        };
        let (_, at_zero) = evaluate(0.0);
        let analytic = -at_zero.weighted[0].trace_product(&ds);
        let h = 1e-5;
        let finite_difference = (evaluate(h).0 - evaluate(-h).0) / (2.0 * h);
        assert!(
            (analytic - finite_difference).abs() < 2e-8,
            "fractional Pulay derivative {analytic} vs FD {finite_difference}"
        );
        assert!((at_zero.density[0].trace_product(&overlap0) - 1.0).abs() < 2e-12);
        assert!(at_zero.entropy_term > 0.0);
    }

    #[test]
    fn finite_temperature_scf_conserves_electrons_and_returns_mermin_energy() {
        let molecule = Molecule::new(vec![Atom {
            z: 2,
            position: Vec3::zero(),
        }]);
        let set = data::load("6-31g").unwrap();
        let cell = Cell::new(molecule, Lattice::cubic(24.0), &set, 8.0).unwrap();
        let functional = ResolvedFunctional::hartree_fock();
        let options = PeriodicScfOptions {
            occupation: OccupationScheme::FermiDirac {
                temperature_kelvin: 100_000.0,
            },
            max_iter: 40,
            damping: 0.1,
            exchange_cutoff: Some(8.0),
            ..PeriodicScfOptions::default()
        };
        let result = run_periodic_scf(&cell, &functional, &options).unwrap();
        assert!(result.entropy_term > 1e-8);
        assert!((result.free_energy - (result.energy - result.entropy_term)).abs() < 1e-13);
        for spin in 0..2 {
            let count: f64 = result.occupations[spin][0].iter().sum();
            assert!((count - 1.0).abs() < 2e-12);
            assert!(result.occupations[spin][0]
                .iter()
                .any(|&occupation| occupation > 1e-8 && occupation < 1.0 - 1e-8));
            assert!(result.chemical_potential[spin].is_some());
        }
        let electrons = result.total_density()[0].trace_product(&result.overlap[0]);
        assert!((electrons - 2.0).abs() < 3e-11);
        for spin in 0..2 {
            assert!(
                fp_sw_error(
                    &result.fock[spin][0],
                    &result.density[spin][0],
                    &result.overlap[0],
                    &result.energy_weighted_density[spin][0],
                ) < 2e-9
            );
        }
    }

    #[test]
    fn periodic_ecp_energy_scf_uses_the_explicit_valence_count() {
        let molecule = Molecule::new(vec![Atom {
            z: 11,
            position: Vec3::zero(),
        }])
        .with_multiplicity(2);
        let set = data::load("lanl2dz").unwrap();
        let cell = Cell::new(molecule, Lattice::cubic(24.0), &set, 8.0).unwrap();
        let plan = crate::periodic::features::PeriodicEcpPlan::from_cell(&cell).unwrap();
        assert_eq!(plan.centers_per_cell, 1);
        assert_eq!(plan.core_electrons_per_cell, 10);
        assert_eq!(cell.n_electrons().unwrap(), 1);
        let functional = ResolvedFunctional::hartree_fock();
        let options = PeriodicScfOptions {
            max_iter: 12,
            damping: 0.0,
            exchange_cutoff: Some(8.0),
            ..PeriodicScfOptions::default()
        };
        let result = run_periodic_scf(&cell, &functional, &options).unwrap();
        assert_eq!((result.n_alpha, result.n_beta), (1, 0));
        assert!(result.energy.is_finite());
        assert!((result.total_density()[0].trace_product(&result.overlap[0]) - 1.0).abs() < 2e-11);
        let forces =
            crate::periodic::forces::analytic_forces(&cell, &functional, &result, &options)
                .unwrap();
        assert!(forces.total[0].norm() < 2e-8);
        assert!((forces.electronic[0] + forces.nuclear[0] - forces.total[0]).norm() < 1e-12);
    }
}
