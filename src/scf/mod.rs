// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Self-consistent field driver, written spin-resolved from the start:
//! restricted (RHF) runs are the closed-shell fast path of the same loop
//! that serves UHF — and later RKS/UKS, where the exchange factor and the
//! XC contribution come from the resolved functional.

pub mod adiis;
pub(crate) mod car_parrinello;
pub mod constrained;
pub mod core_hole;
pub mod diis;
pub mod divide_conquer;
mod fields;
pub mod fock;
mod globalize;
pub mod guess;
pub mod hubbard;
pub mod magnetic;
pub mod occupations;
mod progress;
pub mod ri;
pub mod roks;
pub mod sad;
pub mod sic;
pub mod soscf;
pub(crate) mod trah;
pub use trah::OrbitalOptimizer;

use crate::basis::MolBasis;
use crate::error::{DftError, Result};
use crate::integrals::{overlap_matrix, CoulombKernel};
use crate::linalg::{Matrix, Orthogonalizer};
use crate::system::Molecule;
pub use fock::{EriMode, JkEngine};

/// Default memory budget (MiB) below which the SCF stores the ERI tensor
/// instead of recomputing it. Deliberately modest: the point is to catch
/// the common small-to-medium case, not to trade a hang for a swap storm.
///
/// It is a DEFAULT rather than a constant because the cliff is steep and
/// lands in awkward places. Ferrocene in 6-31G with an ECP on the metal is
/// 132 AOs and needs 2317 MiB — 13% over — so it falls back to recomputing
/// every integral, which is far slower. Raising the budget is then the
/// whole fix, so it has to be reachable (`ScfOptions::incore_mb`,
/// `--incore-mb`).
pub const DEFAULT_INCORE_MB: usize = 2048;
pub use guess::Guess;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Reference {
    /// RHF for closed shells, UHF otherwise.
    #[default]
    Auto,
    Rhf,
    Uhf,
}

#[derive(Clone, Debug)]
pub struct ScfOptions {
    pub occupation: Option<occupations::DeterminantOccupation>,
    pub fixed_occupations: Option<occupations::FixedOccupations>,
    pub reference: Reference,
    pub max_iter: usize,
    /// |ΔE| convergence threshold (Hartree).
    /// Energy convergence threshold — PySCF''s `conv_tol`.
    pub e_tol: f64,
    /// rms(ΔP) convergence threshold.
    pub p_tol: f64,
    pub use_diis: bool,
    pub diis_size: usize,
    /// Damping factor applied while the DIIS error is above
    /// `damping_until_error`.
    pub damping: f64,
    pub damping_until_error: f64,
    /// Use ADIIS while the DIIS error is above `adiis_until_error`, then
    /// hand over to CDIIS. ADIIS interpolates on the convex hull of the
    /// stored densities and so cannot overshoot; CDIIS converges faster
    /// once the residual is small.
    pub use_adiis: bool,
    pub adiis_until_error: f64,
    /// Enable orbital globalization. Auto retains the inexpensive quadratic
    /// Newton endgame and uses the full-kernel trust-region augmented Hessian
    /// on stalled nonlinear integer SCF. Missing optional kernels retain
    /// energy/Fock quasi-Newton globalization. Forced TRAH requires the complete
    /// analytic kernel and never changes the physical convergence thresholds.
    /// Thermal, selected populations and nonvariational corrections keep
    /// their original algorithms unless an incompatible forced mode is requested.
    pub use_soscf: bool,
    /// Integer-manifold globalization; convergence thresholds are unchanged.
    pub orbital_optimizer: OrbitalOptimizer,
    pub soscf_from_error: f64,
    /// Gradient (commutator) convergence threshold — PySCF's conv_tol_grad.
    /// PySCF defaults it to sqrt(conv_tol) and so does ScfOptions::default;
    /// set it explicitly to override.
    pub conv_tol_grad: f64,
    /// Trust radius on |kappa|: a Newton step is only as good as its
    /// quadratic model.
    pub soscf_trust: f64,
    pub soscf_max_cg: usize,
    /// Level shift (Hartree) added to the virtual space while the DIIS
    /// error is above `level_shift_until` (auto-released so the converged
    /// state is a genuine aufbau solution of the unshifted Fock).
    pub level_shift: f64,
    pub level_shift_until: f64,
    pub guess: Guess,
    pub eri_mode: EriMode,
    pub kernel: CoulombKernel,
    /// Overlap eigenvalue cutoff for linear-dependence removal.
    pub orth_threshold: f64,
    /// Density fitting for the Coulomb build (RI-J): `Some(aux basis name)`
    /// enables it (e.g. "def2-universal-jfit"); `None` = exact J (default).
    /// Exchange K always stays on the exact path.
    pub density_fit: Option<String>,
    /// Uniform external electric field (a.u.). Adds `+E·r` to H_core (the
    /// electronic −μ̂·E with μ̂ = −r) and `−Σ Z_A R_A·E` to the nuclear
    /// energy, so `dE/dE_ax = −μ_ax` exactly.
    pub external_field: Option<crate::math::Vec3>,
    /// Gradient-regulated asymptotic correction of the exchange-correlation
    /// potential with this shift (Hartree; ionization potential plus the
    /// highest occupied orbital energy of the bulk functional), see
    /// [`crate::ks::grac_correction`]. Potential only: the energy stays that
    /// of the functional.
    pub asymptotic_correction: Option<f64>,
    /// Extra one-electron operator added to H_core (embedding potentials,
    /// SAPT induction finite-difference gates, ...).
    ///
    /// CONTRACT: this matrix must be independent of the nuclear coordinates.
    /// It contributes to the total energy through `Tr[P·extra]`, exactly
    /// like any other H_core term, and to nothing else. `assemble_gradient`
    /// (`gradient.rs`), `analytic_hessian_full` (`hessian/analytic.rs`) and
    /// `tda_gradient`/the CPKS response kernel (`response/tddft_grad.rs`)
    /// do not read this field and do not need to: a geometry-independent
    /// additive constant in H_core has zero nuclear derivative, and being
    /// density-independent it also drops out of the response kernel, so it
    /// cannot bias a polarizability, Hessian, or excited-state gradient.
    ///
    /// Do NOT put a geometry-dependent operator here. The type is a plain
    /// `Matrix` with no way to express or carry its own dR — the
    /// gradient/Hessian/excited-state code would silently omit whatever
    /// that operator's true nuclear derivative should have been, producing
    /// a wrong analytic gradient with no error. A geometry-dependent
    /// one-electron term needs its own derivative contribution wired into
    /// the paths above instead. `src/pcm.rs`'s C-PCM/IEF-PCM reaction field
    /// is the worked example: it is rebuilt from the density every SCF
    /// iteration and carries its own analytic gradient/Hessian machinery
    /// (`cpcm_gradient`, `switch_gradient`, `PcmSolver::gradient`, ...) — it
    /// is deliberately not plumbed through `extra_hcore`.
    pub extra_hcore: Option<crate::linalg::Matrix>,
    /// Memory budget (MiB) for holding the ERI tensor in core. Applies to
    /// the automatic Direct→in-core switch; see [`DEFAULT_INCORE_MB`] for
    /// why this is worth being able to raise.
    pub incore_mb: usize,
    /// Broken-symmetry start: rotate the beta HOMO/LUMO pair of the guess
    /// by this angle (DEGREES) before building the beta density.
    ///
    /// Only meaningful together with `Reference::Uhf` on a closed shell.
    /// An unrestricted singlet started from the restricted guess converges
    /// straight back to the restricted solution — correct for a normal
    /// closed shell, and useless for a diradical or a dissociating bond,
    /// where the low-energy solution has the two spins localized on
    /// opposite centres. Mixing the guess is what lets the SCF find it.
    /// `0.0` disables the rotation, so `Reference::Uhf` alone reproduces
    /// the restricted answer exactly and can be used to check the
    /// unrestricted code path.
    pub guess_mix: f64,
    /// Self-consistent VV10: the nonlocal potential enters the Fock matrix
    /// every iteration (variational total energy; exact analytic
    /// derivatives). `None` leaves VV10 to the caller (post-SCF add-on).
    pub vv10: Option<crate::nlc::Vv10Params>,
    /// Grid level for the (coarse) VV10 quadrature when `vv10` is set.
    pub vv10_grid_level: usize,
    /// Independent auxiliary-grid weight pruning; zero keeps every point.
    pub vv10_grid_weight_cutoff: f64,
    /// A learned exchange-correlation model evaluated on the molecular grid
    /// (Skala); its energy and Kohn-Sham matrices enter every iteration.
    pub skala: Option<std::sync::Arc<crate::skala::SkalaModel>>,
    /// Continuum solvation (C-PCM/COSMO family, optionally with the SMD
    /// non-electrostatic term): the reaction field is rebuilt from the
    /// current density every iteration, so the total energy is variational
    /// and its derivatives are exact at the converged density.
    pub pcm: Option<crate::pcm::PcmOptions>,
    /// Electronic temperature `k_B T` in Hartree. Zero (the default) keeps
    /// integer aufbau occupations. A positive value gives Fermi-Dirac
    /// occupations with the electron count of each spin channel fixed
    /// (canonical per spin), and the reported `energy` becomes the Mermin
    /// free energy `E - T S`, which is the variational quantity whose
    /// analytic nuclear gradient `assemble_gradient` returns.
    pub electronic_temperature: f64,
    /// Fractional change of the alpha and beta electron counts relative to
    /// the integer counts implied by charge and multiplicity. Nonzero values
    /// need a positive electronic temperature (Fermi-Dirac occupations) and
    /// are used for fractional-charge ensembles and their derivatives.
    pub electron_offset: [f64; 2],
}

impl Default for ScfOptions {
    fn default() -> Self {
        Self {
            occupation: None,
            fixed_occupations: None,
            reference: Reference::Auto,
            max_iter: 128,
            // PySCF's default conv_tol.
            e_tol: 1e-9,
            p_tol: 1e-8,
            use_diis: true,
            diis_size: 8,
            damping: 0.3,
            damping_until_error: 0.25,
            use_adiis: true,
            adiis_until_error: 0.1,
            use_soscf: true,
            orbital_optimizer: OrbitalOptimizer::Auto,
            soscf_from_error: 1e-2,
            // PySCF: conv_tol_grad defaults to sqrt(conv_tol).
            conv_tol_grad: 3.162_277_660_168_379_5e-5,
            soscf_trust: 0.2,
            soscf_max_cg: 20,
            level_shift: 0.0,
            level_shift_until: 1e-4,
            guess: Guess::Sad,
            incore_mb: DEFAULT_INCORE_MB,
            guess_mix: 0.0,
            eri_mode: EriMode::default(),
            kernel: CoulombKernel::Coulomb,
            orth_threshold: 1e-7,
            density_fit: None,
            external_field: None,
            asymptotic_correction: None,
            extra_hcore: None,
            vv10: None,
            vv10_grid_level: 3,
            vv10_grid_weight_cutoff: 1e-14,
            skala: None,
            pcm: None,
            electronic_temperature: 0.0,
            electron_offset: [0.0; 2],
        }
    }
}

impl ScfOptions {
    /// Dedicated VV10 quadrature shared by SCF, target energies and response.
    /// Molecular driver grid_weight_cutoff applies to this grid as well.
    pub fn nonlocal_grid(&self, molecule: &Molecule) -> Result<crate::grid::MolecularGrid> {
        if !self.vv10_grid_weight_cutoff.is_finite() || self.vv10_grid_weight_cutoff < 0.0 {
            return Err(DftError::InvalidInput(
                "the VV10 quadrature weight cutoff must be finite and nonnegative".into(),
            ));
        }
        crate::grid::MolecularGrid::build(
            molecule,
            &crate::grid::GridOptions {
                level: self.vv10_grid_level,
                weight_cutoff: self.vv10_grid_weight_cutoff,
                ..Default::default()
            },
        )
    }

    /// Use the same dense-tensor memory policy for a reference and its response.
    /// The estimate budgets the full n^4 tensor, without a symmetry discount.
    /// Explicit in-core limits and density fitting retain their own policy.
    pub(crate) fn effective_eri_mode(&self, nao: usize) -> EriMode {
        if let Some(remaining) = crate::resources::remaining_bytes() {
            // Keep room for two full/range tensors, quartet assembly and
            // matrix workspaces. Direct recomputation retains the caller's
            // original screening policy when this conservative bound fails.
            let estimate = nao.checked_pow(4).and_then(|n| n.checked_mul(128));
            if estimate.is_none_or(|bytes| bytes > remaining / 2) {
                return self.eri_mode;
            }
        }
        if matches!(self.eri_mode, EriMode::Direct { .. }) && self.density_fit.is_none() {
            let need_mb = nao
                .checked_pow(4)
                .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
                .and_then(|bytes| bytes.checked_add(1024 * 1024 - 1))
                .map(|bytes| bytes / (1024 * 1024));
            if let Some(need) = need_mb.filter(|need| *need <= self.incore_mb) {
                if let Some(limit_mb) = need.max(1).checked_add(8) {
                    return EriMode::InCore { limit_mb };
                }
            }
        }
        self.eri_mode
    }
}

/// Fermi-Dirac occupations `f_i = 1 / (1 + exp((e_i - mu) / kt))` of
/// `energies` holding `electrons` electrons at `kt` (Hartree), with `mu`
/// fixed by bisection. Returns the occupations, `mu` and the entropy
/// `S / k_B = -sum_i [f ln f + (1 - f) ln (1 - f)]`.
pub fn fermi_dirac_occupations(
    energies: &[f64],
    electrons: f64,
    kt: f64,
) -> Result<(Vec<f64>, f64, f64)> {
    let n = energies.len() as f64;
    if !(kt.is_finite() && kt > 0.0 && (0.0..=n).contains(&electrons))
        || energies.iter().any(|e| !e.is_finite())
    {
        return Err(DftError::InvalidInput(
            "Fermi-Dirac occupations need kt > 0, finite energies and 0 <= N <= n_mo".into(),
        ));
    }
    let occupation = |e: f64, mu: f64| {
        let x = (e - mu) / kt;
        if x > 0.0 {
            let t = (-x).exp();
            t / (1.0 + t)
        } else {
            1.0 / (1.0 + x.exp())
        }
    };
    let count = |mu: f64| energies.iter().map(|&e| occupation(e, mu)).sum::<f64>();
    let lowest = energies.iter().copied().fold(f64::INFINITY, f64::min);
    let highest = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (mut lo, mut hi) = (lowest - 60.0 * kt - 1.0, highest + 60.0 * kt + 1.0);
    let mu = if electrons == 0.0 {
        lo
    } else if electrons == n {
        hi
    } else {
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if count(mid) < electrons {
                lo = mid;
            } else {
                hi = mid;
            }
            if hi - lo <= 4.0 * f64::EPSILON * mid.abs().max(1.0) {
                break;
            }
        }
        0.5 * (lo + hi)
    };
    let mut occupations: Vec<f64> = energies.iter().map(|&e| occupation(e, mu)).collect();
    // Remove the bisection remainder so the electron count is exact.
    let error = occupations.iter().sum::<f64>() - electrons;
    let weight: f64 = occupations.iter().map(|f| f * (1.0 - f)).sum();
    if weight > 0.0 {
        for f in &mut occupations {
            *f -= error * *f * (1.0 - *f) / weight;
        }
    }
    let entropy = occupations
        .iter()
        .map(|&f| {
            let mut s = 0.0;
            if f > 0.0 && f < 1.0 {
                s -= f * f.ln() + (1.0 - f) * (1.0 - f).ln();
            }
            s
        })
        .sum();
    Ok((occupations, mu, entropy))
}

/// `P = C diag(f) C^T` for occupation numbers `f`.
pub fn occupied_density(c: &Matrix, occupations: &[f64]) -> Matrix {
    let n = c.rows;
    let mut p = Matrix::zeros(n, n);
    for (i, &f) in occupations.iter().enumerate().take(c.cols) {
        if f == 0.0 {
            continue;
        }
        for mu in 0..n {
            let cm = f * c[(mu, i)];
            if cm == 0.0 {
                continue;
            }
            for nu in 0..n {
                p[(mu, nu)] += cm * c[(nu, i)];
            }
        }
    }
    p
}

/// Full SCF solution — deliberately rich: MO coefficients, densities and
/// intermediate matrices are part of the public result for debugging and
/// research use (wavefunction dump API).
#[derive(Clone, Debug)]
pub struct ScfResult {
    pub occupation_tracking: Option<occupations::OccupationDiagnostics>,
    /// Fresh fixed-population density-map residual in the orthonormal AO frame.
    pub fixed_occupation_residual: Option<f64>,
    /// Largest active effective-Fock element at an accepted common-orbital solution.
    pub common_orbital_residual: Option<f64>,
    pub energy: f64,
    pub e_nuc: f64,
    pub e_elec: f64,
    /// Ascending MO energies for Aufbau states; tracked determinants place
    /// occupied orbitals first, with energy order inside occupied/virtual blocks.
    pub mo_energy: [Vec<f64>; 2],
    /// MO coefficient columns per spin (`nao × n_mo`, CᵀSC = I).
    pub mo_coeff: [Matrix; 2],
    /// Spin densities `P^α`, `P^β`.
    pub density: [Matrix; 2],
    /// Converged Fock matrices per spin.
    pub fock: [Matrix; 2],
    pub overlap: Matrix,
    pub h_core: Matrix,
    pub n_alpha: usize,
    pub n_beta: usize,
    pub unrestricted: bool,
    /// Spin moment of the independent-orbital occupation ensemble; reduces
    /// to the determinant expression for integer populations.
    pub s_squared: f64,
    /// XC energy of the converged density (0 for HF).
    pub e_xc: f64,
    /// Self-consistent VV10 nonlocal correlation energy (0 unless
    /// `ScfOptions::vv10` was set).
    pub e_nlc: f64,
    /// Continuum-solvation polarization energy (0 unless
    /// `ScfOptions::pcm` was set); the non-electrostatic SMD term is
    /// added on top of the SCF total by the caller.
    pub e_pcm: f64,
    /// Numerically integrated electron count (grid-quality diagnostic;
    /// `None` for grid-free runs).
    pub grid_electron_count: Option<f64>,
    pub converged: bool,
    pub iterations: usize,
    /// Occupation number of every MO per spin channel (1 or 0 for integer
    /// aufbau occupations).
    pub occupations: [Vec<f64>; 2],
    /// `k_B T` in Hartree (0 for integer or nonthermal fixed populations).
    pub electronic_temperature: f64,
    /// Electronic entropy `S / k_B`, summed over both spin channels; `energy`
    /// contains `- electronic_temperature * electronic_entropy`.
    pub electronic_entropy: f64,
    /// Per-spin Fermi levels of a smeared reference (`None` at zero
    /// temperature).
    pub chemical_potentials: [Option<f64>; 2],
}

impl ScfResult {
    /// Whether the reported populations describe the leading-column integer
    /// determinant required by occupied/virtual-only response algorithms.
    pub fn has_integer_occupations(&self) -> bool {
        (0..2).all(|spin| {
            let count = [self.n_alpha, self.n_beta][spin];
            self.occupations[spin].len() == self.mo_coeff[spin].cols
                && self.occupations[spin]
                    .iter()
                    .enumerate()
                    .all(|(i, &f)| f == if i < count { 1.0 } else { 0.0 })
        })
    }

    pub fn p_total(&self) -> Matrix {
        let mut p = self.density[0].clone();
        for (t, b) in p.as_mut_slice().iter_mut().zip(self.density[1].as_slice()) {
            *t += *b;
        }
        p
    }

    pub fn homo_lumo(&self) -> (Option<f64>, Option<f64>) {
        if !self.has_integer_occupations() {
            return (None, None);
        }
        let occ = self.n_alpha;
        let e = &self.mo_energy[0];
        let homo = occ.checked_sub(1).and_then(|i| e.get(i)).copied();
        let lumo = e.get(occ).copied();
        (homo, lumo)
    }

    /// Lowest virtual minus highest occupied energy, separately for each spin.
    /// Occupied and virtual blocks remain distinct after orbital optimization;
    /// a negative value reports a non-Aufbau stationary determinant. This is
    /// neither an orbital-stability test nor a ground-state certificate.
    pub fn occupied_virtual_gaps(&self) -> [Option<f64>; 2] {
        if !self.has_integer_occupations() || self.common_orbital_residual.is_some() {
            return [None; 2];
        }
        std::array::from_fn(|spin| {
            let occupied = [self.n_alpha, self.n_beta][spin];
            let energies = &self.mo_energy[spin];
            let highest = energies.get(..occupied)?.iter().copied().reduce(f64::max)?;
            let lowest = energies.get(occupied..)?.iter().copied().reduce(f64::min)?;
            Some(lowest - highest)
        })
    }
}

/// Run a Hartree–Fock SCF (no functional parts, full exact exchange).
pub fn run_hf(mol: &Molecule, mb: &MolBasis, opts: &ScfOptions) -> Result<ScfResult> {
    run_scf(
        mol,
        mb,
        opts,
        &crate::xc::ResolvedFunctional::hartree_fock(),
        None,
    )
}

/// Run an SCF with the resolved functional: RHF/UHF (no parts, cx = 1),
/// pure DFT (cx = 0, XC quadrature), hybrids (both). A grid is required
/// whenever the functional has XC parts.
pub fn run_scf(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &crate::xc::ResolvedFunctional,
    grid: Option<&crate::grid::MolecularGrid>,
) -> Result<ScfResult> {
    run_scf_with_density(mol, mb, opts, functional, grid, None)
}

/// SCF from an optional determinant density, used by orbital-stability descent.
/// The seed changes only the initial iterate, never the functional or operator.
pub fn run_scf_with_density(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &crate::xc::ResolvedFunctional,
    grid: Option<&crate::grid::MolecularGrid>,
    initial_density: Option<&[Matrix; 2]>,
) -> Result<ScfResult> {
    if crate::resources::current_usage().is_some() {
        crate::resources::catch_allocations(|| {
            run_scf_with_density_inner(mol, mb, opts, functional, grid, initial_density)
        })
    } else {
        run_scf_with_density_inner(mol, mb, opts, functional, grid, initial_density)
    }
}

fn run_scf_with_density_inner(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &crate::xc::ResolvedFunctional,
    grid: Option<&crate::grid::MolecularGrid>,
    initial_density: Option<&[Matrix; 2]>,
) -> Result<ScfResult> {
    if let Some(occupation) = &opts.occupation {
        if opts.electronic_temperature != 0.0
            || opts.electron_offset != [0.0; 2]
            || opts.fixed_occupations.is_some()
        {
            return Err(DftError::InvalidInput("integer determinant tracking cannot be combined with Fermi smearing or fractional electron offsets".into()));
        }
        if initial_density.is_some() {
            return Err(DftError::InvalidInput("an occupation-selected state needs explicit reference orbitals, not an additional determinant seed".into()));
        }
        let mut ground_options = opts.clone();
        ground_options.occupation = None;
        let reference = run_scf_impl(mol, mb, &ground_options, functional, grid, None, None)?;
        return occupations::run_scf_with_occupations(
            mol, mb, opts, functional, grid, &reference, occupation,
        );
    }
    run_scf_impl(mol, mb, opts, functional, grid, initial_density, None)
}

fn run_scf_impl(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &crate::xc::ResolvedFunctional,
    grid: Option<&crate::grid::MolecularGrid>,
    initial_density: Option<&[Matrix; 2]>,
    tracking: Option<&mut occupations::OccupationTracker>,
) -> Result<ScfResult> {
    run_scf_in_potential(
        mol,
        mb,
        opts,
        functional,
        grid,
        initial_density,
        tracking,
        None,
    )
}

/// Electronic inner solve at a fixed quantum-particle density. This scoped
/// operator is never installed in `ScfOptions::extra_hcore`: ordinary nuclear
/// derivatives cannot mistake a density-dependent embedding for a constant.
#[derive(Clone, Copy)]
pub(crate) struct FrozenPotential<'a> {
    pub operator: &'a Matrix,
    pub coulomb: &'a JkEngine,
    pub long_range: Option<&'a JkEngine>,
    /// Optional complete classical energy at this fixed geometry and field.
    /// It changes iteration energies and final totals, never the Fock matrix.
    /// None preserves ordinary point-nucleus and embedding behavior.
    pub classical_energy: Option<f64>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_scf_in_potential(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &crate::xc::ResolvedFunctional,
    grid: Option<&crate::grid::MolecularGrid>,
    initial_density: Option<&[Matrix; 2]>,
    tracking: Option<&mut occupations::OccupationTracker>,
    potential: Option<FrozenPotential<'_>>,
) -> Result<ScfResult> {
    run_scf_in_potential_with_completion(
        mol,
        mb,
        opts,
        functional,
        grid,
        initial_density,
        tracking,
        potential,
        None,
        true,
    )
}

/// Retain the final finite iterate for the typed SCF boundary. The ordinary
/// public solvers keep their convergence requirement. Occupation tracking
/// needs a separately converged reference and does not use this trial route.
pub(crate) fn run_scf_trial(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &crate::xc::ResolvedFunctional,
    grid: Option<&crate::grid::MolecularGrid>,
) -> Result<ScfResult> {
    if opts.occupation.is_some() {
        return Err(DftError::Unsupported {
            feature: "retaining an unconverged occupation-tracked SCF iterate".into(),
            missing: "a checked tracked-reference trial-state contract".into(),
        });
    }
    let evaluate = || {
        run_scf_in_potential_with_completion(
            mol, mb, opts, functional, grid, None, None, None, None, false,
        )
    };
    if crate::resources::current_usage().is_some() {
        crate::resources::catch_allocations(evaluate)
    } else {
        evaluate()
    }
}

#[allow(clippy::too_many_arguments)]
fn run_scf_in_potential_with_completion(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &crate::xc::ResolvedFunctional,
    grid: Option<&crate::grid::MolecularGrid>,
    initial_density: Option<&[Matrix; 2]>,
    mut tracking: Option<&mut occupations::OccupationTracker>,
    potential: Option<FrozenPotential<'_>>,
    extension: Option<fields::DensityExtension<'_>>,
    require_convergence: bool,
) -> Result<ScfResult> {
    crate::logger::scf_settings(opts);
    crate::linalg::ensure_parallelism();
    let cx = functional.hybrid.cx;
    let clr = functional.hybrid.clr;
    let has_xc = !functional.parts.is_empty();
    if opts.asymptotic_correction.is_some()
        && (!has_xc || grid.is_none() || functional.family == crate::xc::Family::MetaGga)
    {
        return Err(DftError::InvalidInput(
            "the asymptotic correction applies to local and gradient-corrected density \
             functionals on the molecular grid"
                .into(),
        ));
    }
    if opts.skala.is_some() && grid.is_none() {
        return Err(DftError::InvalidInput(
            "a learned exchange-correlation model needs the molecular grid".into(),
        ));
    }
    if has_xc && grid.is_none() {
        return Err(DftError::InvalidInput(format!(
            "functional `{}` needs a quadrature grid",
            functional.description
        )));
    }
    let scf_t0 = std::time::Instant::now();
    let n_elec_eff = {
        let n = mol.n_electrons()?;
        let core = mb.n_core_electrons as usize;
        n.checked_sub(core).ok_or_else(|| {
            DftError::InvalidInput(format!(
                "ECP core electrons ({core}) exceed the total electron count ({n})"
            ))
        })?
    };
    let expected = mol.n_alpha_beta_for(n_elec_eff)?;
    let [n_alpha, n_beta] = tracking
        .as_ref()
        .map_or([expected.0, expected.1], |t| t.counts());
    if n_alpha + n_beta != n_elec_eff || n_alpha.abs_diff(n_beta) != expected.0.abs_diff(expected.1)
    {
        return Err(DftError::InvalidInput("selected determinant occupations must match the molecular charge and absolute spin projection".into()));
    }
    let restricted = match opts.reference {
        Reference::Auto => n_alpha == n_beta,
        Reference::Rhf => {
            if n_alpha != n_beta {
                return Err(DftError::InvalidInput(format!(
                    "RHF requested for an open shell ({n_alpha}α/{n_beta}β)"
                )));
            }
            true
        }
        Reference::Uhf => false,
    };

    let s = overlap_matrix(mb);
    let z_eff = mb.effective_charges(mol);
    let (mut h_core, classical_energy) = fields::one_electron(mol, mb, opts, &z_eff)?;
    if let Some(extra) = potential {
        let extra = extra.operator;
        if extra.rows != mb.nao_sph
            || extra.cols != mb.nao_sph
            || extra.as_slice().iter().any(|v| !v.is_finite())
        {
            return Err(DftError::InvalidInput(
                "the frozen multicomponent potential must be a finite electronic AO matrix".into(),
            ));
        }
        for (h, value) in h_core.as_mut_slice().iter_mut().zip(extra.as_slice()) {
            *h += value;
        }
    }
    let orth = Orthogonalizer::canonical(&s, opts.orth_threshold)?;
    if orth.n_dropped > 0 {
        crate::logger::warn(&format!(
            "dft-rs: dropped {} nearly linearly dependent AO combination(s) (min kept σ = {:.3e})",
            orth.n_dropped, orth.min_kept_eigenvalue
        ));
    }
    // Cache a dense ERI tensor when it fits the caller's memory budget.
    // Response kernels use this same selection instead of returning to
    // repeated direct integral generation after a cached reference solve.
    let mut eff_opts = opts.clone();
    eff_opts.eri_mode = opts.effective_eri_mode(mb.nao_sph);
    if let (EriMode::Direct { .. }, EriMode::InCore { limit_mb }) =
        (opts.eri_mode, eff_opts.eri_mode)
    {
        crate::logger::log(&format!(
            "[scf] ERI tensor fits in {} MiB: using in-core",
            limit_mb - 8,
        ));
    }
    let opts = &eff_opts;
    let owned_jk;
    let jk = if let Some(potential) = potential {
        potential.coulomb
    } else {
        owned_jk = JkEngine::from_options(mol, mb, opts)?;
        &owned_jk
    };
    // Long-range exchange engine for range-separated hybrids.
    let owned_lr;
    let jk_lr = if let Some(potential) = potential {
        potential.long_range
    } else if clr != 0.0 {
        owned_lr = JkEngine::new(
            mb,
            opts.eri_mode,
            CoulombKernel::Erf {
                omega: functional.hybrid.omega,
            },
        )?;
        Some(&owned_lr)
    } else {
        None
    };
    let e_nuc = match potential.and_then(|p| p.classical_energy) {
        Some(energy) if energy.is_finite() => energy,
        Some(_) => {
            return Err(DftError::InvalidInput(
                "a scoped classical nuclear energy must be finite".into(),
            ))
        }
        None => classical_energy,
    };
    // Self-consistent VV10: dedicated coarse grid, built once.
    // Continuum solvation: cavity, S factorization and surface integrals
    // depend only on the geometry, so they are built once.
    let pcm_solver = match &opts.pcm {
        Some(p) => Some(crate::pcm::PcmSolver::build(mol, mb, &z_eff, p)?),
        None => None,
    };
    let nlc_grid = match opts.vv10 {
        Some(_) => Some(opts.nonlocal_grid(mol)?),
        None => None,
    };

    // Initial guess. SAD is a DENSITY guess: superpose converged atoms and
    // build the Fock from that, which starts the SCF with the right
    // electron count already in the right shells. The one-electron guesses
    // go through a guess Fock instead. If the atomic SCF fails for any
    // reason the guess quietly falls back rather than failing the run — a
    // worse starting point costs iterations, not correctness.
    let f0 = if initial_density.is_some() {
        h_core.clone()
    } else {
        match opts.guess {
            Guess::Sad => match sad::sad_density(mol, mb) {
                Ok(p_sad) => {
                    let mut p_half = p_sad.clone();
                    for v in p_half.as_mut_slice().iter_mut() {
                        *v *= 0.5;
                    }
                    let (j, k) = jk.build(mb, &p_sad, [&p_half, &p_half], true)?;
                    let mut f = h_core.clone();
                    for (fv, (jv, kv)) in f
                        .as_mut_slice()
                        .iter_mut()
                        .zip(j.as_slice().iter().zip(k[0].as_slice()))
                    {
                        *fv += *jv - *kv;
                    }
                    f
                }
                Err(_) => guess::initial_fock(Guess::Gwh, &h_core, &s),
            },
            other => guess::initial_fock(other, &h_core, &s),
        }
    };
    let (_, c0) = orth.solve_roothaan(&f0)?;
    let fixed_populations = opts
        .fixed_occupations
        .as_ref()
        .map(|f| f.validated(c0.cols, [n_alpha, n_beta], restricted))
        .transpose()?;
    if fixed_populations.is_some()
        && (opts.electronic_temperature != 0.0
            || opts.electron_offset != [0.0; 2]
            || tracking.is_some()
            || initial_density.is_some())
    {
        return Err(DftError::InvalidInput("fixed populations cannot be combined with thermal smearing, electron offsets or determinant tracking/seeds".into()));
    }
    let population_density = |c: &Matrix, spin: usize| match &fixed_populations {
        Some(f) => occupied_density(c, &f[spin]),
        None => c.leading_columns_gram([n_alpha, n_beta][spin].min(c.cols), 1.0),
    };
    let mut density = [population_density(&c0, 0), {
        // Optional broken-symmetry start: a 2x2 rotation of the beta
        // HOMO/LUMO pair. Being orthogonal, it keeps CᵀSC = I, so the
        // guess stays a legitimate determinant and only its symmetry
        // is broken.
        let mut cb = c0.clone();
        if opts.guess_mix != 0.0 && n_beta >= 1 && n_beta < cb.cols {
            let th = opts.guess_mix.to_radians();
            let (cs, sn) = (th.cos(), th.sin());
            let (h, l) = (n_beta - 1, n_beta);
            for mu in 0..cb.rows {
                let (a, b) = (cb[(mu, h)], cb[(mu, l)]);
                cb[(mu, h)] = cs * a + sn * b;
                cb[(mu, l)] = -sn * a + cs * b;
            }
        }
        population_density(&cb, 1)
    }];
    if let Some(seed) = initial_density {
        for spin in 0..2 {
            let p = &seed[spin];
            if p.rows != s.rows || p.cols != s.cols || p.as_slice().iter().any(|v| !v.is_finite()) {
                return Err(DftError::InvalidInput(
                    "SCF initial density has invalid AO dimensions or values".into(),
                ));
            }
            let occupied = if spin == 0 { n_alpha } else { n_beta };
            if p.max_abs_diff(&p.transpose()) > 1e-8
                || p.matmul_seq(&s).matmul_seq(p).max_abs_diff(p) > 1e-8
                || (p.frobenius_dot(&s) - occupied as f64).abs() > 1e-8
            {
                return Err(DftError::InvalidInput("SCF initial density must be a symmetric particle-conserving determinant projector".into()));
            }
        }
        if restricted && seed[0].max_abs_diff(&seed[1]) > 1e-8 {
            return Err(DftError::InvalidInput(
                "restricted SCF needs equal alpha and beta seed densities".into(),
            ));
        }
        density = seed.clone();
    }
    let nspin = if restricted { 1 } else { 2 };
    let kt = opts.electronic_temperature;
    if !(kt.is_finite() && kt >= 0.0) {
        return Err(DftError::InvalidInput(
            "electronic temperature must be finite and non-negative".into(),
        ));
    }
    if opts.orbital_optimizer == OrbitalOptimizer::Trah
        && (!opts.use_soscf
            || kt > 0.0
            || fixed_populations.is_some()
            || tracking.is_some()
            || opts.asymptotic_correction.is_some())
    {
        return Err(DftError::Unsupported {
            feature: "forced trust-region augmented-Hessian SCF".into(),
            missing: "an enabled variational integer-occupation orbital manifold".into(),
        });
    }
    let smeared = kt > 0.0;
    let offset = opts.electron_offset;
    if offset.iter().any(|x| !x.is_finite()) || (!smeared && offset != [0.0; 2]) {
        return Err(DftError::InvalidInput(
            "fractional electron offsets must be finite and need a positive electronic temperature"
                .into(),
        ));
    }
    if restricted && offset[0] != offset[1] {
        return Err(DftError::InvalidInput(
            "a restricted reference needs equal alpha and beta electron offsets".into(),
        ));
    }
    // Occupations and entropy of the current `density`; the integer guess
    // starts with zero entropy.
    let mut entropy = [0.0_f64; 2];
    let mut fermi_levels: [Option<f64>; 2] = [None, None];
    let mut diis = diis::Diis::new(opts.diis_size);
    let mut adiis = adiis::Adiis::new(opts.diis_size);
    let mut e_last = f64::INFINITY;
    let fixed_integral_tensor = match opts.eri_mode {
        EriMode::InCore { .. } => true,
        EriMode::Direct { screen_threshold } => screen_threshold == 0.0,
    };
    let quadratic_energy = fixed_integral_tensor
        && !has_xc
        && !matches!(extension, Some(fields::DensityExtension::Hubbard(_)))
        && opts.skala.is_none()
        && opts.vv10.is_none()
        && opts.pcm.is_none()
        && kt == 0.0;
    let mut previous_quadratic: Option<([Matrix; 2], [Matrix; 2])> = None;
    let mut orbital_energy_work: Option<f64> = None;
    let mut last_scf_error = f64::INFINITY;
    let mut last_p_delta = f64::INFINITY;
    let mut converged = false;
    let mut orbital_optimization_active = false;
    let mut newton_stagnation = 0usize;
    let mut previous_newton_residual = f64::INFINITY;
    let mut newton_suspended = false;
    let mut iterations = 0;
    let mut focks;
    let mut mo_energy = [Vec::new(), Vec::new()];
    let mut mo_coeff = [Matrix::zeros(0, 0), Matrix::zeros(0, 0)];

    let model = fields::MolecularDensityModel {
        mol,
        mb,
        opts,
        functional,
        grid,
        h_core: &h_core,
        extension,
        jk,
        jk_lr,
        pcm_solver: pcm_solver.as_ref(),
        nlc_grid: nlc_grid.as_ref(),
        restricted,
    };

    let mut trust_region = trah::TrustRegion::new(opts.soscf_trust)?;
    let mut trust_progress = progress::ResidualProgress::default();
    let mut trust_active = false;
    let mut trust_unavailable = false;
    let mut globalizer: Option<globalize::CollinearGlobalizer> = None;
    let mut globalization_suspended = false;
    let mut local_progress = progress::ResidualProgress::default();
    let mut globalization_entry_residual = f64::INFINITY;

    for iter in 0..opts.max_iter {
        iterations = iter + 1;
        let fields = model.evaluate(&density)?;
        focks = fields.fock;
        let e_elec = fields.electronic_energy;
        let e_total = e_elec + e_nuc - kt * (entropy[0] + entropy[1]);
        // For a quadratic density functional this identity is exact. It
        // resolves an arbitrarily small energy change without subtracting
        // two large molecular energies, and retains the requested e_tol.
        let energy_difference = if quadratic_energy {
            let difference = previous_quadratic
                .as_ref()
                .map_or(f64::INFINITY, |(old_p, old_f)| {
                    let mut work = 0.0;
                    for spin in 0..2 {
                        for k in 0..density[spin].as_slice().len() {
                            work += 0.5
                                * (density[spin].as_slice()[k] - old_p[spin].as_slice()[k])
                                * (focks[spin].as_slice()[k] + old_f[spin].as_slice()[k]);
                        }
                    }
                    work
                });
            previous_quadratic = Some((density.clone(), focks.clone()));
            difference
        } else {
            e_total - e_last
        };
        let signed_energy_change = orbital_energy_work.take().unwrap_or(energy_difference);

        // DIIS bookkeeping and extrapolation.
        let active = &focks[..nspin];
        let dens = &density[..nspin];
        let mut population_defect = 0.0_f64;
        let err = if smeared || fixed_populations.is_some() {
            // A commutator vanishes for any occupations diagonal in F.
            // It cannot establish the prescribed populations: a stale DIIS Fock
            // can keep the density unchanged with incorrect populations.
            // Use the complete physical population-map residual for extrapolation
            // and test it independently before accepting convergence.
            let sx = s.matmul_seq(&orth.x);
            let mut errors = Vec::new();
            for spin in 0..nspin {
                let fo = orth.x.transpose_matmul(&focks[spin].matmul_seq(&orth.x));
                let (eps, co) = crate::linalg::symmetric_eigen(&fo)?;
                let n_occ = if spin == 0 { n_alpha } else { n_beta };
                let occupations = match &fixed_populations {
                    Some(f) => f[spin].clone(),
                    None => fermi_dirac_occupations(&eps, n_occ as f64 + offset[spin], kt)?.0,
                };
                let c = orth.x.matmul_seq(&co);
                let mut delta = occupied_density(&c, &occupations);
                for (value, current) in delta
                    .as_mut_slice()
                    .iter_mut()
                    .zip(density[spin].as_slice())
                {
                    *value -= current;
                }
                let residual = sx.transpose_matmul(&delta.matmul_seq(&sx));
                for &value in residual.as_slice() {
                    population_defect = population_defect.max(value.abs());
                    errors.push(value);
                }
            }
            diis.push_with_errors(active, errors);
            population_defect
        } else {
            diis.push(active, dens, &s, &orth.x)
        };
        let mut occupation_defect = 0.0_f64;
        if let Some(tracker) = tracking.as_deref_mut() {
            let sx = s.matmul_seq(&orth.x);
            for spin in 0..2 {
                let (eps, c) = orth.solve_roothaan(&focks[spin])?;
                let selected = tracker.preview(spin, &c, &eps, &s)?;
                let count = if spin == 0 { n_alpha } else { n_beta };
                let mut delta = selected.coefficients.leading_columns_gram(count, 1.0);
                for (v, current) in delta
                    .as_mut_slice()
                    .iter_mut()
                    .zip(density[spin].as_slice())
                {
                    *v -= current;
                }
                let residual = sx.transpose_matmul(&delta.matmul_seq(&sx));
                occupation_defect = occupation_defect.max(
                    residual
                        .as_slice()
                        .iter()
                        .copied()
                        .map(f64::abs)
                        .fold(0.0, f64::max),
                );
            }
            tracker.set_residual(occupation_defect);
        }
        let err = if tracking.is_some() {
            err.max(occupation_defect)
        } else {
            err
        };
        last_scf_error = err;
        crate::logger::log(&format!(
            "[scf] iter {iterations:>3}  E = {e_total:>20.12}  dE = {:>10.3e}  |SCF residual| = {err:>10.3e}  rms(dP) = {last_p_delta:>10.3e}",
            signed_energy_change
        ));
        // A tiny unitary rotation can cease to change the represented AO
        // projector while its physical commutator still fails the requested
        // gate. Rebuild a fresh physical-Fock/CDIIS sequence instead of
        // repeatedly applying unresolved Newton steps. Convergence remains
        // subject to the original energy, density and residual tests.
        if quadratic_energy
            && opts.use_soscf
            && orbital_optimization_active
            && !newton_suspended
            && err > opts.conv_tol_grad
            && signed_energy_change.abs() < opts.e_tol
            && last_p_delta < opts.p_tol
            && err >= 0.7 * previous_newton_residual
        {
            newton_stagnation += 1;
        } else {
            newton_stagnation = 0;
        }
        previous_newton_residual = err;
        if newton_stagnation >= 4 {
            newton_suspended = true;
            orbital_optimization_active = false;
            diis.reset();
            adiis = adiis::Adiis::new(opts.diis_size);
            crate::logger::log(&format!(
                "[scf] unresolved Newton projector steps; restart physical Fock/CDIIS at residual={err:.12e}"
            ));
        }
        // An accepted orbital rotation minimizes on the integer determinant
        // manifold. Reoccupying the eigenvectors of its physical Fock is a
        // different operation: a non-Aufbau stationary determinant can have
        // a finite Aufbau-map change even as all three SCF tests vanish.
        // Retain the original energy, actual density-step and fresh orbital
        // residual gates. Stability and occupation ordering are reported
        // separately; convergence does not assert the electronic ground state.
        if trust_active
            && iter > 0
            && signed_energy_change.abs() < opts.e_tol
            && err < opts.conv_tol_grad
            && last_p_delta < opts.p_tol
            && population_defect < opts.p_tol
            && occupation_defect < opts.p_tol
        {
            crate::logger::log(&format!(
                "[scf] stationary trust-region determinant dE={signed_energy_change:.12e} orbital_residual={err:.12e} density_rms={last_p_delta:.12e}"
            ));
            converged = true;
            break;
        }
        // Do not perturb a physical fixed point with another extrapolated
        // diagonalization. Reevaluate the actual Fock population map first;
        // retaining an already stationary density makes its next quadratic
        // energy change exactly zero, without a tolerance floor or subtracting
        // roundoff-sized AO projectors against large core energies.
        if opts.asymptotic_correction.is_none()
            && !smeared
            && fixed_populations.is_none()
            && tracking.is_none()
            && mo_coeff[0].cols > 0
            && err < opts.conv_tol_grad
        {
            let sx = s.matmul_seq(&orth.x);
            let mut map_norm_squared = 0.0;
            let mut map_rms_squared = 0.0;
            for spin in 0..nspin {
                let (_, c) = orth.solve_roothaan(&focks[spin])?;
                let mut delta = c.leading_columns_gram([n_alpha, n_beta][spin], 1.0);
                for (value, current) in delta
                    .as_mut_slice()
                    .iter_mut()
                    .zip(density[spin].as_slice())
                {
                    *value -= current;
                }
                let map = sx.transpose_matmul_seq(&delta).matmul_seq(&sx);
                map_norm_squared += map.as_slice().iter().map(|x| x * x).sum::<f64>();
                map_rms_squared += delta.as_slice().iter().map(|x| x * x).sum::<f64>()
                    / delta.as_slice().len() as f64;
            }
            let map_norm = map_norm_squared.sqrt();
            let map_rms = (map_rms_squared / nspin as f64).sqrt();
            if map_norm < opts.p_tol && map_rms < opts.p_tol {
                crate::logger::log(&format!(
                    "[scf] physical density map metric_norm={map_norm:.12e} AO_rms={map_rms:.12e}"
                ));
                last_p_delta = map_rms;
                e_last = e_total;
                if iter > 0 && signed_energy_change.abs() < opts.e_tol {
                    converged = true;
                    break;
                }
                continue;
            }
        }
        // A scaled augmented-Hessian root includes negative curvature instead
        // of stopping a Newton CG solve at its first indefinite direction.
        let trust_stalled = trust_progress.has_stalled(err, 6);
        let wants_trust = opts.orbital_optimizer == OrbitalOptimizer::Trah
            || (opts.orbital_optimizer == OrbitalOptimizer::Auto
                && !quadratic_energy
                && (trust_stalled || trust_active));
        if opts.use_soscf
            && wants_trust
            && !trust_unavailable
            && !smeared
            && fixed_populations.is_none()
            && tracking.is_none()
            && opts.asymptotic_correction.is_none()
            && mo_coeff[0].cols > 0
            // A small orbital residual does not yet certify the density
            // fixed point. Keep the accepted trust-region path active while
            // the physical-map gate above still fails; a stale extrapolated
            // Fock can otherwise kick a nearly converged determinant away.
            && (err > opts.conv_tol_grad || trust_active)
        {
            // A mixed density cannot share the integer orbital Hessian. In
            // forced mode damping is disabled below; automatic mode waits for
            // the existing physical determinant rather than projecting it.
            let determinant = (0..nspin).all(|spin| {
                mo_coeff[spin]
                    .leading_columns_gram([n_alpha, n_beta][spin], 1.0)
                    .max_abs_diff(&density[spin])
                    < 1e-10
            });
            if determinant {
                let reference = globalize::trial_reference(
                    &density,
                    &mo_coeff,
                    &mo_energy,
                    &focks,
                    &s,
                    &h_core,
                    [n_alpha, n_beta],
                    nspin,
                );
                match globalize::trust_step(
                    &model,
                    &reference,
                    e_elec,
                    quadratic_energy,
                    &mut trust_region,
                ) {
                    Ok(Some(step)) => {
                        density = step.density;
                        mo_coeff = step.coefficients;
                        mo_energy = step.energies;
                        last_p_delta = step.density_rms_change;
                        orbital_energy_work = Some(step.energy_work);
                        e_last = e_total;
                        trust_active = true;
                        globalizer = None;
                        diis.reset();
                        adiis = adiis::Adiis::new(opts.diis_size);
                        continue;
                    }
                    Ok(None) => {
                        trust_active = false;
                        // A fresh physical Fock iteration defines a new model;
                        // do not carry a collapsed rejected radius into it.
                        trust_region = trah::TrustRegion::new(opts.soscf_trust)?;
                    }
                    Err(DftError::Unsupported { .. })
                        if opts.orbital_optimizer == OrbitalOptimizer::Auto =>
                    {
                        trust_unavailable = true;
                        crate::logger::log("[scf] complete TRAH kernel unavailable; retain energy/Fock globalization");
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        // A complete variational energy/Fock is sufficient for manifold
        // globalization. No approximate or incomplete XC Hessian is used.
        // This fallback is shared with the complex-spinor SCF optimizer.
        if !quadratic_energy
            && opts.use_soscf
            && !globalization_suspended
            && !smeared
            && fixed_populations.is_none()
            && tracking.is_none()
            && opts.asymptotic_correction.is_none()
            && mo_coeff[0].cols > 0
        {
            // Measure improvement across a whole window. Comparing each
            // step to the immediately preceding best mislabels a sequence
            // of useful smaller reductions as stagnation.
            let local_stalled = globalizer.is_none() && local_progress.has_stalled(err, 6);
            let globalization_stalled = opts.use_diis
                && opts.diis_size >= 2
                && globalizer
                    .as_mut()
                    .is_some_and(|optimizer| optimizer.needs_local_recovery(err, opts.diis_size));
            if globalizer.is_some()
                && (err < 0.01 * globalization_entry_residual || globalization_stalled)
            {
                if globalization_stalled {
                    // Decreasing total energies alone do not establish
                    // stationarity. Retire a stalled orbital minimizer and
                    // retain the unmodified physical energy/density/residual
                    // gates during a fresh local DIIS iteration.
                    globalization_suspended = true;
                    crate::logger::log(&format!(
                        "[scf] orbital residual plateau; restart physical Fock/DIIS at residual={err:.12e}"
                    ));
                } else {
                    crate::logger::log(&format!(
                        "[scf] resume local CDIIS residual={err:.6e} entry_residual={globalization_entry_residual:.6e}"
                    ));
                }
                globalizer = None;
                local_progress = progress::ResidualProgress::default();
                diis.reset();
                adiis = adiis::Adiis::new(opts.diis_size);
            }
            if globalizer.is_none()
                && !globalization_suspended
                && local_stalled
                && err > opts.conv_tol_grad
            {
                // Damped mixtures do not lie on the integer manifold.
                // Wait for a determinant rather than projecting its density.
                let determinant = (0..nspin).all(|spin| {
                    density[spin]
                        .matmul_seq(&s)
                        .matmul_seq(&density[spin])
                        .max_abs_diff(&density[spin])
                        < 1e-10
                });
                if determinant {
                    globalizer = Some(globalize::CollinearGlobalizer::new(
                        &mo_coeff,
                        [n_alpha, n_beta],
                        nspin,
                    ));
                    globalization_entry_residual = err;
                    diis.reset();
                    adiis = adiis::Adiis::new(opts.diis_size);
                    crate::logger::log("[scf] start occupied-subspace quasi-Newton globalization");
                }
            }
            if let Some(optimizer) = &mut globalizer {
                if let Some(step) = optimizer.step(&model, &density, &focks, e_elec, &s, &orth)? {
                    density = step.density;
                    mo_coeff = step.coefficients;
                    last_p_delta = step.density_rms_change;
                    e_last = e_total;
                    continue;
                }
                // Failed line searches start a fresh local subspace and
                // never count as physical convergence.
                globalizer = None;
                local_progress = progress::ResidualProgress::default();
                diis.reset();
                adiis = adiis::Adiis::new(opts.diis_size);
            }
        }
        // ADIIS far from convergence, CDIIS near it. CDIIS drives the
        // RESIDUAL to zero, which is right at the solution and unreliable
        // away from it -- the extrapolated Fock can be worse than any of
        // its inputs. ADIIS instead interpolates on the convex hull of the
        // stored densities, so it cannot overshoot; but its objective is
        // only a variational ESTIMATE, so it stalls where CDIIS shines.
        // Switching on the error is what gets both behaviours.
        adiis.push(dens, active);
        let extrapolated = if opts.use_diis {
            let use_adiis = opts.use_adiis && err > opts.adiis_until_error;
            if use_adiis {
                adiis.extrapolate()
            } else {
                diis.extrapolate()
            }
        } else {
            None
        };
        let work = extrapolated.unwrap_or_else(|| active.to_vec());

        // SOSCF, taken HERE rather than after the diagonalization: `focks`
        // is the true Fock of the current density and `mo_coeff` are the
        // orbitals that produced it, so gradient and orbitals belong to
        // the same point. Taking the step after diagonalizing instead
        // pairs new orbitals with the previous density's Fock, which is
        // not a Newton step at all -- measured, that made benzene WORSE
        // (HF 11 -> 17 iterations). At this position no extra Fock build
        // is needed.
        let mut p_delta_soscf = 0.0;
        let mut soscf_taken = false;
        // The second-order step assumes an idempotent determinant; a
        // Fermi-smeared density is iterated by (A)DIIS alone.
        if opts.use_soscf
            && !newton_suspended
            // This cached Newton implementation currently supplies the
            // complete quadratic Coulomb/exchange density kernel. Other
            // terms keep the ordinary variational SCF/DIIS iteration until
            // their full kernels are connected here; no partial Newton
            // operator is substituted for them.
            && quadratic_energy
            && tracking.is_none()
            && fixed_populations.is_none()
            && !smeared
            && (err < opts.soscf_from_error || orbital_optimization_active)
            && err > opts.conv_tol_grad
            && mo_coeff[0].cols > 0
        {
            let oh = soscf::OrbitalHessian {
                c: &mo_coeff,
                eps: &mo_energy,
                n_occ: [n_alpha, n_beta],
                spins: nspin,
            };
            if !oh.is_empty() {
                let g = oh.gradient(&focks);
                let hx = |kap: &[f64]| -> Result<Vec<f64>> {
                    let dp = oh.delta_density(kap);
                    let go = fock::coulomb_exchange_response(mb, jk, jk_lr, cx, clr, &dp)?;
                    let mut out = oh.project(&go);
                    let fixed = oh.fock_commutator(kap, &focks);
                    for (o, k) in out.iter_mut().zip(fixed) {
                        *o += k;
                    }
                    Ok(out)
                };
                if let Ok(mut kappa) = soscf::newton_direction(&oh, &g, hx, opts.soscf_max_cg, 1e-3)
                {
                    if kappa.iter().all(|k| *k == 0.0) {
                        // The CG routine stops at negative curvature. Its
                        // zero iterate is not a converged SCF solution: try
                        // a true downhill preconditioned gradient with the
                        // same energy-tested trust-region line search.
                        let diagonal = oh.diagonal();
                        for i in 0..kappa.len() {
                            kappa[i] = -g[i] / diagonal[i];
                        }
                    }
                    let kn: f64 = kappa.iter().map(|v| v * v).sum::<f64>().sqrt();
                    if kn > 0.0 && kn.is_finite() {
                        let scale = if kn > opts.soscf_trust {
                            opts.soscf_trust / kn
                        } else {
                            1.0
                        };
                        let nao = mb.nao_sph;
                        let mut accepted = None;
                        let mut fraction = scale;
                        for backtrack in 0..32 {
                            let step: Vec<_> = kappa.iter().map(|v| v * fraction).collect();
                            let initial_work: f64 = g.iter().zip(&step).map(|(a, b)| a * b).sum();
                            if initial_work >= 0.0 {
                                break;
                            }
                            let mut rot = [Matrix::zeros(0, 0), Matrix::zeros(0, 0)];
                            soscf::rotate(&oh, &step, &mut rot);
                            let delta = oh.finite_density_change(&step);
                            let induced =
                                fock::coulomb_exchange_response(mb, jk, jk_lr, cx, clr, &delta)?;
                            let trial_fock: [Matrix; 2] = std::array::from_fn(|s| {
                                Matrix::from_row_major(
                                    nao,
                                    nao,
                                    focks[s]
                                        .as_slice()
                                        .iter()
                                        .zip(induced[s].as_slice())
                                        .map(|(a, b)| a + b)
                                        .collect(),
                                )
                            });
                            let energy_work =
                                oh.quadratic_energy_work(&step, &focks, &delta, &induced);
                            // Armijo decrease of the actual quadratic energy.
                            // A gradient norm can increase while descending
                            // away from a saddle; it is not a line-search merit.
                            if energy_work <= 1e-4 * initial_work {
                                if backtrack > 0 {
                                    crate::logger::log(&format!(
                                        "[scf] Newton backtracks {backtrack} step {fraction:.6e}"
                                    ));
                                }
                                accepted = Some((rot, energy_work, trial_fock));
                                break;
                            }
                            fraction *= 0.5;
                        }
                        if let Some((rot, energy_work, trial_fock)) = accepted {
                            orbital_energy_work = Some(energy_work);
                            // Retain variational optimization while escaping
                            // a saddle, even if its gradient temporarily grows
                            // past the initial DIIS/Newton handover threshold.
                            orbital_optimization_active = true;
                            for spin in 0..nspin {
                                let n_occ = if spin == 0 { n_alpha } else { n_beta };
                                let p_new =
                                    rot[spin].leading_columns_gram(n_occ.min(rot[spin].cols), 1.0);
                                let mut d2 = 0.0;
                                for (pn, po) in
                                    p_new.as_slice().iter().zip(density[spin].as_slice())
                                {
                                    d2 += (pn - po) * (pn - po);
                                }
                                p_delta_soscf += d2 / (p_new.rows * p_new.cols) as f64;
                                density[spin] = p_new;
                                mo_coeff[spin] = rot[spin].clone();
                                let f = rot[spin]
                                    .transpose_matmul_seq(&trial_fock[spin])
                                    .matmul_seq(&rot[spin]);
                                mo_energy[spin] = (0..f.rows).map(|i| f[(i, i)]).collect();
                            }
                            if restricted {
                                density[1] = density[0].clone();
                                mo_coeff[1] = mo_coeff[0].clone();
                                mo_energy[1] = mo_energy[0].clone();
                            }
                            soscf_taken = true;
                        }
                    }
                }
            }
        }

        // Convergence test before the next diagonalization (energy + last ΔP).
        let de = signed_energy_change.abs();
        e_last = e_total;

        let mut p_delta_rms = p_delta_soscf;
        for spin in 0..(if soscf_taken { 0 } else { nspin }) {
            let mut f_o = orth.x.transpose_matmul(&work[spin].matmul(&orth.x));
            if opts.level_shift > 0.0 && err > opts.level_shift_until {
                // Shift virtuals in the orthonormal basis: F += λ(I − P_o).
                let sx = s.matmul(&orth.x);
                let p_o = sx.transpose_matmul(&density[spin].matmul(&sx));
                for i in 0..f_o.rows {
                    for jj in 0..f_o.cols {
                        let proj = if i == jj { 1.0 } else { 0.0 };
                        f_o[(i, jj)] += opts.level_shift * (proj - p_o[(i, jj)]);
                    }
                }
            }
            let (mut eps, c_o) = crate::linalg::symmetric_eigen(&f_o)?;
            let mut c = orth.x.matmul(&c_o);
            if let Some(tracker) = tracking.as_deref_mut() {
                let selected = tracker.preview(spin, &c, &eps, &s)?;
                tracker.commit(spin, &selected);
                if restricted {
                    tracker.commit(1, &selected);
                }
                eps = selected.energies;
                c = selected.coefficients;
            }
            let n_occ = if spin == 0 { n_alpha } else { n_beta };
            let mut p_new = if let Some(f) = &fixed_populations {
                occupied_density(&c, &f[spin])
            } else if smeared {
                let (occupations, mu, s) =
                    fermi_dirac_occupations(&eps, n_occ as f64 + offset[spin], kt)?;
                entropy[spin] = s;
                fermi_levels[spin] = Some(mu);
                occupied_density(&c, &occupations)
            } else {
                c.leading_columns_gram(n_occ.min(c.cols), 1.0)
            };
            // Damping while the error is large.
            if opts.damping > 0.0
                && err > opts.damping_until_error
                && opts.orbital_optimizer != OrbitalOptimizer::Trah
            {
                for (pn, po) in p_new
                    .as_mut_slice()
                    .iter_mut()
                    .zip(density[spin].as_slice())
                {
                    *pn = (1.0 - opts.damping) * *pn + opts.damping * *po;
                }
            }
            let mut delta2 = 0.0;
            for (pn, po) in p_new.as_slice().iter().zip(density[spin].as_slice()) {
                delta2 += (pn - po) * (pn - po);
            }
            p_delta_rms += delta2 / (p_new.rows * p_new.cols) as f64;
            density[spin] = p_new;
            mo_energy[spin] = eps;
            mo_coeff[spin] = c;
        }
        if restricted {
            density[1] = density[0].clone();
            mo_energy[1] = mo_energy[0].clone();
            mo_coeff[1] = mo_coeff[0].clone();
            entropy[1] = entropy[0];
            fermi_levels[1] = fermi_levels[0];
        }
        let p_delta_rms = (p_delta_rms / nspin as f64).sqrt();
        last_p_delta = p_delta_rms;

        // Require both the energy change and the orbital-gradient norm,
        // where the gradient is the same commutator DIIS drives to zero.
        // The density change is kept as a third, looser guard because a
        // stalled iteration can show a small dE and a small gradient while
        // still moving the density around.
        if iter > 0
            && de < opts.e_tol
            && err < opts.conv_tol_grad
            && p_delta_rms < opts.p_tol
            && population_defect < opts.p_tol
            && occupation_defect < opts.p_tol
        {
            converged = true;
            break;
        }
    }

    if !converged && require_convergence {
        return Err(DftError::ScfNotConverged {
            iterations,
            error: last_scf_error,
        });
    }
    // The same physical evaluator supplies iterations, line searches and reports.
    let fields = model.evaluate(&density)?;
    let e_elec = fields.electronic_energy;
    let e_xc = fields.xc_energy;
    let e_nlc = fields.nonlocal_energy;
    let e_pcm = fields.solvent_energy;
    let grid_electron_count = fields.grid_electron_count;
    focks = fields.fock;

    // Orbital rotations retain the occupied projector but do not keep the
    // occupied/virtual blocks canonical. Recanonicalize within each block
    // without changing the stationary density or selected occupation space.
    if tracking.is_none() && fixed_populations.is_none() && !smeared {
        for spin in 0..nspin {
            let f = mo_coeff[spin]
                .transpose_matmul_seq(&focks[spin])
                .matmul_seq(&mo_coeff[spin]);
            let no = [n_alpha, n_beta][spin];
            let n = f.rows;
            let mut rotation = Matrix::zeros(n, n);
            let mut energies = vec![0.0; n];
            for (start, end) in [(0, no), (no, n)] {
                if start == end {
                    continue;
                }
                let block = Matrix::from_row_major(
                    end - start,
                    end - start,
                    (start..end)
                        .flat_map(|i| {
                            (start..end).map({
                                let f = &f;
                                move |j| f[(i, j)]
                            })
                        })
                        .collect(),
                );
                let (e, c) = crate::linalg::symmetric_eigen(&block)?;
                for i in start..end {
                    energies[i] = e[i - start];
                    for j in start..end {
                        rotation[(i, j)] = c[(i - start, j - start)];
                    }
                }
            }
            mo_coeff[spin] = mo_coeff[spin].matmul_seq(&rotation);
            mo_energy[spin] = energies;
        }
        if restricted {
            mo_coeff[1] = mo_coeff[0].clone();
            mo_energy[1] = mo_energy[0].clone();
        }
    }
    let mut occupations: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    let mut total_entropy = 0.0;
    for spin in 0..2 {
        let n_occ = if spin == 0 { n_alpha } else { n_beta };
        occupations[spin] = if let Some(f) = &fixed_populations {
            f[spin].clone()
        } else if smeared {
            let (f, _, s) =
                fermi_dirac_occupations(&mo_energy[spin], n_occ as f64 + offset[spin], kt)?;
            total_entropy += s;
            f
        } else {
            (0..mo_energy[spin].len())
                .map(|i| if i < n_occ { 1.0 } else { 0.0 })
                .collect()
        };
    }

    // Wick's independent-orbital occupation ensemble gives an additional
    // number-fluctuation term for fractional populations. For determinants
    // this reduces exactly to Sz(Sz+1)+N_beta-Tr(P_alpha S P_beta S).
    let counts = occupations.each_ref().map(|f| f.iter().sum::<f64>());
    let sz = 0.5 * (counts[0] - counts[1]);
    let variance: f64 = occupations.iter().flatten().map(|f| f * (1.0 - f)).sum();
    let cross = density[0]
        .matmul_seq(&s)
        .frobenius_dot(&s.matmul_seq(&density[1]));
    let s_squared = sz * sz + 0.5 * (counts[0] + counts[1]) - cross + 0.25 * variance;
    let fixed_occupation_residual = if let Some(f) = &fixed_populations {
        let sx = s.matmul_seq(&orth.x);
        let mut maximum = 0.0_f64;
        for spin in 0..2 {
            let (_, c) = orth.solve_roothaan(&focks[spin])?;
            let mut delta = occupied_density(&c, &f[spin]);
            for (v, p) in delta
                .as_mut_slice()
                .iter_mut()
                .zip(density[spin].as_slice())
            {
                *v -= p;
            }
            let defect = sx.transpose_matmul(&delta.matmul_seq(&sx));
            maximum = maximum.max(
                defect
                    .as_slice()
                    .iter()
                    .copied()
                    .map(f64::abs)
                    .fold(0.0, f64::max),
            );
        }
        if maximum > opts.p_tol {
            converged = false;
            if require_convergence {
                return Err(DftError::ScfNotConverged {
                    iterations,
                    error: maximum,
                });
            }
        }
        Some(maximum)
    } else {
        None
    };
    // Report the same density and energy retained in the result. At the
    // iteration limit, e_last belongs to the preceding density update.
    let energy = e_elec + e_nuc - kt * total_entropy;
    let status = if converged {
        "converged"
    } else {
        "stopped without convergence"
    };
    crate::logger::log(&format!(
        "[scf] {status} in {iterations} iterations, {:.3} s (E = {energy:.12})",
        scf_t0.elapsed().as_secs_f64()
    ));
    if !restricted {
        crate::logger::log(&format!("[uks] <S^2>={s_squared:.12} (hbar^2)"));
    }
    Ok(ScfResult {
        occupation_tracking: None,
        fixed_occupation_residual,
        common_orbital_residual: None,
        energy,
        occupations,
        electronic_temperature: kt,
        electronic_entropy: total_entropy,
        chemical_potentials: fermi_levels,
        e_nuc,
        e_elec,
        mo_energy,
        mo_coeff,
        density,
        fock: focks,
        overlap: s,
        h_core,
        n_alpha,
        n_beta,
        unrestricted: !restricted,
        s_squared,
        e_xc,
        e_nlc,
        e_pcm,
        grid_electron_count,
        converged,
        iterations,
    })
}
