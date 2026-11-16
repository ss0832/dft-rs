// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Symmetry-adapted perturbation theory on HF or KS monomers.
//!
//! # Which SAPT is this?
//!
//! [`SaptVariant::Sapt0`] evaluates SAPT through second order in the
//! intermolecular interaction at zeroth order in intramonomer correlation.
//! It uses field-free HF determinants, the full intermolecular
//! antisymmetrizer, coupled induction and uncoupled orbital-pair dispersion.
//! [`SaptVariant::Response`] is the default response model. It uses
//! HF or KS monomers, Heitler-London exchange and its first variation,
//! coupled induction, and a transition-density projection for
//! exchange-dispersion. TDA dispersion is always available for the response
//! model, with optional full response and a potential-only KS asymptotic
//! correction. The response exchange model is distinct from the formal
//! SAPT(0) antisymmetrizer and from conventional SAPT(DFT).
//! [`SaptVariant::KsFdds`] uses KS determinants with the full
//! antisymmetrizer, coupled static induction and dispersion from the
//! imaginary-frequency density response. Its exchange-dispersion uses
//! uncoupled orbital-pair amplitudes.
//!
//! Perturbation terms use monomer-centred orbital spaces. Supermolecular
//! diagnostics evaluate the dimer and the ghost monomers in the dimer basis.
//!
//! Open- and closed-shell monomers both run: every stage is spin-resolved,
//! and forcing UKS on a closed shell reproduces the restricted numbers
//! term by term (`sapt_terms_are_reference_independent_for_a_closed_shell`).
//!
//! # Terms
//!
//! - `E_elst^(1)`: exact electrostatic interaction of the unperturbed
//!   monomer densities,
//!   `Tr[P^A v^B] + Tr[P^B v^A] + Tr[P^A J[P^B]] + E_nn^{AB}`.
//! - `E_exch^(1)` for the response model: the Heitler-London expression
//!   through the Löwdin nonorthogonal
//!   single-determinant energy. Per spin, the union of the (mutually
//!   nonorthogonal) occupied monomer orbitals defines
//!   `D^s = C^s (C^sᵀ S C^s)⁻¹ C^sᵀ` and
//!   `E_exch = E^{HF-expr}[D] − E^{HF-expr}[A] − E^{HF-expr}[B] − E_elst`,
//!   evaluated with the HF interaction operator on the monomer determinants.
//! - `E_ind^(2)` (coupled, both directions): monomer X's CPKS response (the
//!   shared `ResponseSolver`, so hybrid/RSH/meta kernels, SC-VV10 and a
//!   continuum reaction field all apply) to the FULL electrostatic
//!   potential of the partner, `ω^Y = v^Y + J[P^Y]`; `E = ½ Tr[ΔP^X ω^Y]`
//!   per direction. The λ-scaled `extra_hcore` finite-difference gate pins
//!   the factor.
//! - `E_exch-ind^(2)` for the response model: first variation of this
//!   Heitler-London exchange model in the two coupled induction directions.
//! - `E_disp^(2)` uncoupled: `−Σ (ia|jb)²/Δε` over monomer spin-orbital
//!   excitations, equivalent to a factor of four with restricted spatial
//!   orbitals.
//! - `E_disp^(2)` coupled: the same sum over the monomers' TDA spectra.
//!   Driving the identical machinery with UNCOUPLED excitations reproduces
//!   the closed form to 1e-8, which is what fixes the spin factor instead
//!   of it being a guessed coefficient.
//! - `E_exch-disp^(2)` for the response model: the coupled sum with each spin channel's transition
//!   density projected by `1 − ρ_A^s S ρ_B^s S`, minus the unprojected one.
//! - `E_int` supermolecular (counterpoise-corrected in the dimer basis) and
//!   `delta` = supermolecular minus the coupled SAPT sum (including
//!   exchange-induction and exchange-dispersion). The residual compares
//!   two models and can contain physical dispersion absent from the
//!   supermolecular reference.

use crate::basis::{BasisSet, MolBasis};
use crate::error::{DftError, Result};
use crate::grid::{GridOptions, MolecularGrid};
use crate::integrals::one_electron::{kinetic_matrix, nuclear_matrix, overlap_matrix};
use crate::linalg::Matrix;
use crate::response::ResponseSolver;
use crate::scf::{run_scf, JkEngine, ScfOptions};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;

mod antisymmetry;
mod correlation;
mod correlation_kernels;
mod dispersion;
mod exchange_induction;
mod partition;
mod relaxed_induction;
mod single_exchange;
mod third_order;
mod truncated_induction;

pub use correlation::SaptCorrelationTerms;
use exchange_induction::OccupiedProjector;
pub use third_order::SaptThirdOrder;

/// Intermolecular decomposition selected for a pair of monomers.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SaptVariant {
    /// Coupled monomer response with Heitler-London exchange and projected
    /// exchange-dispersion.
    #[default]
    Response,
    /// SAPT(0) on Hartree-Fock monomers with the full antisymmetrizer,
    /// coupled induction and uncoupled orbital-pair dispersion.
    Sapt0,
    /// SAPT(0) with exchange reduced to one interfragment electron pair.
    Sapt0SingleExchange,
    /// Empirical sSAPT0 using third-power scaling of the single-pair
    /// second-order exchange terms and the first-power delta-HF reference.
    ScaledSapt0,
    /// SAPT through third intermolecular order with uncoupled canonical HF
    /// monomer orbitals and the full intermolecular antisymmetrizer.
    Sapt30Uncoupled,
    /// SAPT2+3 with the full intermolecular antisymmetrizer.
    Sapt2Plus3,
    /// Second-order KS SAPT with coupled FDDS and uncoupled KS exchange-dispersion.
    KsFdds,
}

/// Optional extensions of the SAPT decomposition.
#[derive(Clone, Debug, Default)]
pub struct SaptConfig {
    /// Gradient-regulated asymptotic correction of the exchange-correlation
    /// potential of monomer A and B with these shifts (Hartree; ionization
    /// potential plus highest occupied orbital energy), Kohn-Sham monomers.
    pub asymptotic_shift: [Option<f64>; 2],
    /// Coupled dispersion and exchange-dispersion from the full (RPA)
    /// response of the monomers, as a sum over states and as the
    /// Casimir-Polder integral of the frequency-dependent susceptibilities.
    pub rpa_dispersion: bool,
    /// Gauss-Legendre points of the Casimir-Polder integral (0: 24); the
    /// integral is repeated with twice as many points.
    pub casimir_polder_points: usize,
    /// Charge-transfer energy: induction with each monomer in the dimer
    /// basis minus induction in the monomer basis.
    pub charge_transfer: bool,
    /// Atom-pair partition of the electrostatic and induction energies on
    /// the natural localized molecular orbitals of the monomers.
    pub partition: bool,
}

/// Additional terms reported alongside a selected SAPT decomposition.
#[derive(Clone, Copy, Debug, Default)]
pub struct SaptDiagnostics {
    /// Gaussian nucleus damping for the regularized second-order calculation.
    pub regularization_eta: Option<f64>,
    /// Coupled third-order induction and the second-order delta-HF remainder.
    pub relaxed_induction30: bool,
    /// Mixed intramonomer orders with fixed RHF orbitals.
    pub correlation_terms: bool,
}

/// Higher-order energy terms entering SAPT2+3. Energies are in Hartree.
#[derive(Clone, Copy, Debug)]
pub struct Sapt2Plus3 {
    pub relaxed_exchange_induction30: f64,
    pub delta_hf_order3: f64,
    pub correlated_electrostatics: f64,
    pub correlated_exchange: f64,
    pub correlated_induction: f64,
    pub correlated_dispersion: f64,
    pub cubic_dispersion_and_mixed: f64,
}

impl Sapt2Plus3 {
    fn correlation_and_dispersion(self) -> f64 {
        self.correlated_electrostatics
            + self.correlated_exchange
            + self.correlated_induction
            + self.correlated_dispersion
            + self.cubic_dispersion_and_mixed
    }
}

/// Orbital-relaxed third-order induction and the lower-order HF remainder.
#[derive(Clone, Copy, Debug)]
pub struct SaptRelaxedInduction30 {
    /// Third-order polarization energy in Hartree.
    pub induction: f64,
    /// Counterpoise HF minus first-order and coupled second-order induction
    /// components, with their exchange corrections, in Hartree.
    pub delta_hf_order2: f64,
}

/// Coupled dispersion from the full response of the monomers.
#[derive(Clone, Debug)]
pub struct RpaDispersion {
    /// Sum over RPA states with the transition vectors `X + Y`.
    pub e_disp: f64,
    /// Exchange-dispersion from the chosen model. This is the projection
    /// correction for `Response` and uncoupled KS exchange for `KsFdds`.
    pub e_exch_disp: f64,
    /// Casimir-Polder integral of the coupled susceptibilities.
    pub e_disp_casimir_polder: f64,
    /// Change of the integral between the two quadratures.
    pub casimir_polder_change: f64,
    /// Casimir-Polder integral of the uncoupled susceptibilities (equal to
    /// the closed-form uncoupled dispersion for a converged quadrature).
    pub e_disp_casimir_polder_uncoupled: f64,
    pub points: usize,
}

/// Charge transfer from the basis of the partner.
#[derive(Clone, Debug)]
pub struct ChargeTransfer {
    /// Induction of A and B with each monomer in the dimer basis.
    pub e_ind_dimer_basis: [f64; 2],
    /// Dimer-basis minus monomer-basis induction, both directions.
    pub e_ct: f64,
}

/// Atom-pair partition `[atom of A][atom of B]` (Hartree).
#[derive(Clone, Debug)]
pub struct SaptPartition {
    pub electrostatic: Matrix,
    /// Induction of A in the potential of B.
    pub induction_a: Matrix,
    /// Induction of B in the potential of A.
    pub induction_b: Matrix,
    /// Largest `|sum - total|` of the three partitions.
    pub residual: f64,
}

pub struct SaptResult {
    pub e_monomer_a: f64,
    pub e_monomer_b: f64,
    /// First-order electrostatics.
    pub e_elst: f64,
    /// First-order exchange from the selected model.
    pub e_exch: f64,
    /// Coupled induction, A polarized by B's field.
    pub e_ind_a: f64,
    /// Coupled induction, B polarized by A's field.
    pub e_ind_b: f64,
    /// Exchange-dispersion correction to the selected dispersion model.
    pub e_exch_disp: f64,
    /// Exchange-induction summed over the two monomer responses.
    pub e_exch_ind: f64,
    /// Contributions with A and B polarized, respectively.
    pub e_exch_ind_directions: [f64; 2],
    /// Multiplicative second-order exchange scale in sSAPT0.
    pub exchange_scale: Option<f64>,
    /// Counterpoise HF remainder: scaled second order for sSAPT0 and
    /// third order for SAPT2+3.
    pub delta_hf: Option<f64>,
    /// Uncoupled third-order polarization and exchange components.
    pub third_order: Option<SaptThirdOrder>,
    /// Optional higher-order energies entering the SAPT2+3 selector.
    pub sapt2_plus3: Option<Sapt2Plus3>,
    /// Optional regularized induction with full intermolecular antisymmetrization.
    pub regularized_induction: Option<SaptRegularizedInduction>,
    pub relaxed_induction30: Option<SaptRelaxedInduction30>,
    /// Mixed-order coefficients from the fixed-orbital determinant expansion.
    pub correlation_terms: Option<SaptCorrelationTerms>,
    /// The sum-over-states dispersion driven by UNCOUPLED excitations.
    /// Calibration probe: it must equal `e_disp_unc`, and their ratio is
    /// exactly the spin factor missing from `e_disp_coupled`.
    pub e_disp_sos_uncoupled: f64,
    /// Selected second-order dispersion. The response model uses TDA
    /// spectra, while SAPT(0) uses uncoupled orbital-pair excitations.
    pub e_disp_coupled: f64,
    /// Supermolecular interaction energy of the SAME reference the
    /// monomers were computed with, evaluated in the DIMER basis for both
    /// the dimer and (through ghost centres) each monomer, i.e. already
    /// counterpoise corrected.
    pub e_int_supermolecular: f64,
    /// `delta` = supermolecular minus [`Self::interaction_total`]: everything
    /// the truncated expansion misses (higher-order response and correlation
    /// absent from the reference).
    pub delta: f64,
    /// Uncoupled second-order dispersion,
    /// −4 Σ_{ia∈A, jb∈B} (ia|jb)²/(ε_a−ε_i+ε_b−ε_j), from the monomer
    /// (HF or KS) orbitals and eigenvalues.
    pub e_disp_unc: f64,
    /// Asymptotic-correction shifts applied to the monomers.
    pub asymptotic_shift: [Option<f64>; 2],
    pub rpa: Option<RpaDispersion>,
    pub charge_transfer: Option<ChargeTransfer>,
    pub partition: Option<SaptPartition>,
}

/// Regularized second-order induction at one Gaussian damping exponent.
/// Directional arrays are A polarized by B followed by B polarized by A.
#[derive(Clone, Copy, Debug)]
pub struct SaptRegularizedInduction {
    /// Gaussian damping parameter, in inverse Bohr squared.
    pub eta: f64,
    /// Induction with regularized amplitudes and the unmodified energy operator.
    pub induction: [f64; 2],
    /// Exchange-induction with the same regularized amplitudes.
    pub exchange_induction: [f64; 2],
    /// Unregularized minus regularized total induction.
    pub charge_transfer: [f64; 2],
}

impl SaptResult {
    /// Interaction energy including the selected dispersion model.
    pub fn interaction_total(&self) -> f64 {
        if let Some(terms) = self.sapt2_plus3 {
            // In the published delta-HF(3) grouping, the first-order and
            // relaxed second-/third-order induction terms cancel against
            // the HF remainder. Evaluate the exactly reduced expression.
            return self.e_int_supermolecular
                + self.e_disp_coupled
                + self.e_exch_disp
                + terms.correlation_and_dispersion();
        }
        self.e_elst
            + self.e_exch
            + self.e_ind_a
            + self.e_ind_b
            + self.e_exch_ind
            + self.e_disp_coupled
            + self.e_exch_disp
            + self.delta_hf.unwrap_or_default()
            + self.third_order.map_or(0.0, SaptThirdOrder::total)
    }

    /// The same total with the dispersion and exchange-dispersion of the
    /// full (RPA) monomer response, when it was computed.
    pub fn interaction_total_rpa(&self) -> Option<f64> {
        self.rpa.as_ref().map(|rpa| {
            self.e_elst
                + self.e_exch
                + self.e_ind_a
                + self.e_ind_b
                + self.e_exch_ind
                + rpa.e_disp
                + rpa.e_exch_disp
        })
    }

    /// Stage-1 calibration total using uncoupled dispersion and omitting the
    /// exchange-induction and exchange-dispersion corrections.
    ///
    /// This diagnostic preserves the historical sum used to calibrate the
    /// uncoupled sum-over-states machinery. It is not the reported SAPT total.
    pub fn interaction_total_uncoupled_calibration(&self) -> f64 {
        self.e_elst + self.e_exch + self.e_ind_a + self.e_ind_b + self.e_disp_unc
    }
}

/// AO transition density of one excited state: `P_mn = sum_ai X_ai C_ma C_ni`.
pub(crate) fn transition_density(
    x: &Matrix,
    c: &Matrix,
    no: usize,
    nv: usize,
    nao: usize,
) -> Matrix {
    let mut p = Matrix::zeros(nao, nao);
    for a_v in 0..nv {
        for i in 0..no {
            let xv = x[(a_v, i)];
            if xv == 0.0 {
                continue;
            }
            for mu in 0..nao {
                let cma = c[(mu, no + a_v)];
                if cma == 0.0 {
                    continue;
                }
                for nu in 0..nao {
                    p[(mu, nu)] += xv * cma * c[(nu, i)];
                }
            }
        }
    }
    p
}

/// Second-order dispersion as a sum over monomer excitations:
/// `-spin_factor sum_nm V_nm^2 / (w_n + w_m)` with `V_nm` the Coulomb
/// coupling of the two AO transition densities. Shared by the coupled path
/// (TDA spectra) and its uncoupled-limit calibration (single (a,i)
/// excitations), which is what fixes the spin factor without guessing.
///
/// `spin_factor` is `f_A * f_B`, with `f = 2` for a monomer whose
/// excitation basis is SPATIAL (the singlet TDA basis of a closed-shell
/// monomer: one state per spatial (i,a), so the two spin channels have
/// been folded into it) and `f = 1` for a SPIN-RESOLVED basis (each spin
/// channel enumerated separately, so the count is already right).
///
/// The two conventions agree in the closed-shell limit, which is the
/// check that pins them: a closed-shell system run spin-resolved gives
/// four identical (alpha/beta x alpha/beta) state pairs at `f = 1`, the
/// same total as one pair at `f = 2` per side. Transition densities are
/// always UNIT-weighted (`P = C_a C_i^T` summed over occupied spin
/// channels) — the spin counting lives here and nowhere else.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispersion_sum_over_states(
    wa: &[f64],
    pa: &[Matrix],
    wb: &[f64],
    pb: &[Matrix],
    eri: &crate::integrals::eri::EriTensor,
    nao_a: usize,
    nao_b: usize,
    spin_factor: f64,
) -> f64 {
    let mut acc = 0.0;
    for (n, p_n) in pa.iter().enumerate() {
        // Coulomb field of A's transition density on B's AO block.
        let mut field = Matrix::zeros(nao_b, nao_b);
        for lam in 0..nao_b {
            for sg in 0..nao_b {
                let mut v = 0.0;
                for mu in 0..nao_a {
                    for nu in 0..nao_a {
                        let pv = p_n[(mu, nu)];
                        if pv != 0.0 {
                            v += pv * eri.get(mu, nu, lam + nao_a, sg + nao_a);
                        }
                    }
                }
                field[(lam, sg)] = v;
            }
        }
        for (m, p_m) in pb.iter().enumerate() {
            let v_nm: f64 = p_m
                .as_slice()
                .iter()
                .zip(field.as_slice())
                .map(|(a, b)| a * b)
                .sum();
            acc -= spin_factor * v_nm * v_nm / (wa[n] + wb[m]);
        }
    }
    acc
}

/// HF energy expression (electronic part) for a given closed-shell-like
/// density: `Tr[Ph] + ½Tr[P J[P]] − ½ Σ_s Tr[P_s K[P_s]]`.
fn hf_energy_and_fock(
    jk: &JkEngine,
    mb: &MolBasis,
    h: &Matrix,
    p_tot: &Matrix,
    p_spin: [&Matrix; 2],
) -> Result<(f64, [Matrix; 2])> {
    let (j, k) = jk.build(mb, p_tot, p_spin, true)?;
    let mut e = p_tot.frobenius_dot(h) + 0.5 * p_tot.frobenius_dot(&j);
    for spin in 0..2 {
        e -= 0.5 * p_spin[spin].frobenius_dot(&k[spin]);
    }
    let fock = std::array::from_fn(|spin| {
        let mut f = h.clone();
        for ((f, &j), &k) in f
            .as_mut_slice()
            .iter_mut()
            .zip(j.as_slice())
            .zip(k[spin].as_slice())
        {
            *f += j - k;
        }
        f
    });
    Ok((e, fock))
}

/// Embed a monomer-basis matrix block into the dimer AO space at `off`.
fn embed(m: &Matrix, nao_dimer: usize, off: usize) -> Matrix {
    let mut out = Matrix::zeros(nao_dimer, nao_dimer);
    for r in 0..m.rows {
        for c in 0..m.cols {
            out[(r + off, c + off)] = m[(r, c)];
        }
    }
    out
}

/// Closed-shell SAPT (stage 1) on two monomers described by `functional`.
/// Monomer-centered basis: each monomer SCF runs in its own AO set; the
/// interaction terms are assembled in the concatenated dimer AO space.
#[allow(clippy::too_many_arguments)]
pub fn sapt_interaction(
    mol_a: &Molecule,
    mol_b: &Molecule,
    set: &BasisSet,
    functional: &ResolvedFunctional,
    opts: &ScfOptions,
    grid_opts: Option<&GridOptions>,
) -> Result<SaptResult> {
    sapt_interaction_with(
        mol_a,
        mol_b,
        set,
        functional,
        opts,
        grid_opts,
        &SaptConfig::default(),
    )
}

/// SAPT with the extensions of `config`.
#[allow(clippy::too_many_arguments)]
pub fn sapt_interaction_with(
    mol_a: &Molecule,
    mol_b: &Molecule,
    set: &BasisSet,
    functional: &ResolvedFunctional,
    opts: &ScfOptions,
    grid_opts: Option<&GridOptions>,
    config: &SaptConfig,
) -> Result<SaptResult> {
    sapt_interaction_variant(
        mol_a,
        mol_b,
        set,
        functional,
        opts,
        grid_opts,
        config,
        SaptVariant::Response,
    )
}

/// Evaluate the selected decomposition with explicit monomer and response controls.
#[allow(clippy::too_many_arguments)]
pub fn sapt_interaction_variant(
    mol_a: &Molecule,
    mol_b: &Molecule,
    set: &BasisSet,
    functional: &ResolvedFunctional,
    opts: &ScfOptions,
    grid_opts: Option<&GridOptions>,
    config: &SaptConfig,
    variant: SaptVariant,
) -> Result<SaptResult> {
    sapt_interaction_diagnostics(
        mol_a,
        mol_b,
        set,
        functional,
        opts,
        grid_opts,
        config,
        variant,
        SaptDiagnostics::default(),
    )
}

/// Formal SAPT(0) and regularized second-order induction with one fixed exponent.
#[allow(clippy::too_many_arguments)]
pub fn sapt_interaction_regularized(
    mol_a: &Molecule,
    mol_b: &Molecule,
    set: &BasisSet,
    functional: &ResolvedFunctional,
    opts: &ScfOptions,
    grid_opts: Option<&GridOptions>,
    config: &SaptConfig,
    eta: f64,
) -> Result<SaptResult> {
    sapt_interaction_regularized_variant(
        mol_a,
        mol_b,
        set,
        functional,
        opts,
        grid_opts,
        config,
        SaptVariant::Sapt0,
        eta,
    )
}

/// Evaluate regularized induction with the full antisymmetrizer and a
/// coupled HF or KS monomer response.
#[allow(clippy::too_many_arguments)]
pub fn sapt_interaction_regularized_variant(
    mol_a: &Molecule,
    mol_b: &Molecule,
    set: &BasisSet,
    functional: &ResolvedFunctional,
    opts: &ScfOptions,
    grid_opts: Option<&GridOptions>,
    config: &SaptConfig,
    variant: SaptVariant,
    eta: f64,
) -> Result<SaptResult> {
    sapt_interaction_diagnostics(
        mol_a,
        mol_b,
        set,
        functional,
        opts,
        grid_opts,
        config,
        variant,
        SaptDiagnostics {
            regularization_eta: Some(eta),
            ..SaptDiagnostics::default()
        },
    )
}

/// Evaluate the selected SAPT model and optional diagnostic terms.
#[allow(clippy::too_many_arguments)]
pub fn sapt_interaction_diagnostics(
    mol_a: &Molecule,
    mol_b: &Molecule,
    set: &BasisSet,
    functional: &ResolvedFunctional,
    opts: &ScfOptions,
    grid_opts: Option<&GridOptions>,
    config: &SaptConfig,
    variant: SaptVariant,
    diagnostics: SaptDiagnostics,
) -> Result<SaptResult> {
    if let Some(eta) = diagnostics.regularization_eta {
        if !matches!(variant, SaptVariant::Sapt0 | SaptVariant::KsFdds) {
            return Err(DftError::InvalidInput(
                "SAPT regularization requires sapt0 or sapt-ks-fdds".into(),
            ));
        }
        if !eta.is_finite() || eta < 0.0 {
            return Err(DftError::InvalidInput(
                "SAPT regularization requires a finite nonnegative eta in Bohr^-2".into(),
            ));
        }
    }
    let higher_order = variant == SaptVariant::Sapt2Plus3;
    if diagnostics.relaxed_induction30 && !higher_order && variant != SaptVariant::Sapt0 {
        return Err(DftError::InvalidInput(
            "relaxed third-order HF induction requires sapt0".into(),
        ));
    }
    if diagnostics.correlation_terms && !higher_order && variant != SaptVariant::Sapt0 {
        return Err(DftError::InvalidInput(
            "fixed-orbital SAPT correlation requires sapt0".into(),
        ));
    }
    let ks_fdds = variant == SaptVariant::KsFdds;
    if ks_fdds {
        let h = functional.hybrid;
        let local_tail = h.cx == 0.0
            && h.clr == 0.0
            && functional.family != crate::xc::Family::MetaGga
            && config
                .asymptotic_shift
                .iter()
                .all(|s| s.is_some_and(f64::is_finite));
        let long_range_tail = h.clr != 0.0
            && h.omega > 0.0
            && (h.cx + h.clr - 1.0).abs() < 1e-12
            && config.asymptotic_shift.iter().all(Option::is_none);
        if functional.parts.is_empty()
            || (functional.required_nonlocal_kind() == Some(crate::xc::RequiredNonlocalTerm::Vv10)
                && opts.vv10.is_none())
            || !config.rpa_dispersion
            || !(local_tail || long_range_tail)
            || opts.occupation.is_some()
            || opts.fixed_occupations.is_some()
            || opts.electronic_temperature != 0.0
            || opts.external_field.is_some()
            || opts.extra_hcore.is_some()
            || opts.pcm.is_some()
            || opts.skala.is_some()
        {
            return Err(DftError::InvalidInput(
                concat!(
                    "sapt-ks-fdds requires field-free integer Aufbau monomers with their required ",
                    "density-correlation kernel, coupled FDDS, and two finite GRAC shifts for ",
                    "pure LDA/GGA or full long-range Fock exchange without GRAC shifts",
                )
                .into(),
            ));
        }
    }
    if variant != SaptVariant::Response
        && !ks_fdds
        && (!functional.parts.is_empty()
            || functional.hybrid != crate::xc::HybridParams::HF
            || config.rpa_dispersion
            || config.asymptotic_shift.iter().any(Option::is_some)
            || opts.occupation.is_some()
            || opts.fixed_occupations.is_some()
            || opts.electronic_temperature != 0.0
            || opts.external_field.is_some()
            || opts.extra_hcore.is_some()
            || opts.pcm.is_some()
            || opts.vv10.is_some()
            || opts.skala.is_some())
    {
        return Err(DftError::InvalidInput(
            "SAPT(0) requires field-free integer Aufbau Hartree-Fock monomers, vacuum, \
             and uncoupled orbital-pair dispersion"
                .into(),
        ));
    }
    if variant == SaptVariant::Sapt30Uncoupled && (config.partition || config.charge_transfer) {
        return Err(DftError::InvalidInput(
            "SAPT(30) uses uncoupled induction; its atom-pair partition and charge-transfer diagnostic are unavailable".into(),
        ));
    }
    // Open-shell monomers are handled by running every determinant-level
    // step spin-resolved, exactly as the SCF driver does; a closed-shell
    // pair takes the same path with the alpha block reused for beta, so
    // the RHF numbers are reproduced bit-for-bit (gated in p12_sapt).
    // ---- monomer SCFs (own basis, own grid) ----
    let mb_a = MolBasis::build(mol_a, set)?;
    let mb_b = MolBasis::build(mol_b, set)?;
    if !mb_a.ecps.is_empty() || !mb_b.ecps.is_empty() {
        return Err(DftError::Unsupported {
            feature: "SAPT with ECP basis sets".into(),
            missing: "consistent effective nuclear charges and semi-local ECP intermolecular \
                operators; refusing to return an incomplete decomposition"
                .into(),
        });
    }
    let needs_grid = !functional.parts.is_empty();
    let default_gopts = GridOptions::default();
    let gopts = grid_opts.unwrap_or(&default_gopts);
    let grid_a = if needs_grid {
        Some(MolecularGrid::build(mol_a, gopts)?)
    } else {
        None
    };
    let grid_b = if needs_grid {
        Some(MolecularGrid::build(mol_b, gopts)?)
    } else {
        None
    };
    let corrected = |shift: Option<f64>| ScfOptions {
        asymptotic_correction: shift,
        ..opts.clone()
    };
    let scf_a = run_scf(
        mol_a,
        &mb_a,
        &corrected(config.asymptotic_shift[0]),
        functional,
        grid_a.as_ref(),
    )?;
    let scf_b = run_scf(
        mol_b,
        &mb_b,
        &corrected(config.asymptotic_shift[1]),
        functional,
        grid_b.as_ref(),
    )?;
    // The second-order terms need a unique first-order response of each
    // monomer, which a zero or negative internal orbital-Hessian curvature
    // denies: a partially filled degenerate shell (the hole may turn freely
    // within the shell) or a saddle point. Whether the interaction operator
    // happens to have a component along such a mode depends on the
    // arbitrary orientation of the degenerate solution, so the size of the
    // rotation below alone cannot detect it; the curvature is checked first.
    for (who, mol, mb, grid, scf) in [
        ("A", mol_a, &mb_a, grid_a.as_ref(), &scf_a),
        ("B", mol_b, &mb_b, grid_b.as_ref(), &scf_b),
    ] {
        use crate::response::stability::{
            stability_from_state, StabilityChannel, StabilityOptions,
        };
        let channel = if scf.unrestricted {
            StabilityChannel::UnrestrictedInternal
        } else {
            StabilityChannel::InternalReal
        };
        let options = StabilityOptions {
            n_roots: 1,
            channels: vec![channel],
            verify_lowest: false,
            ..Default::default()
        };
        let report = stability_from_state(mol, mb, functional, grid, scf, opts, &options)?;
        let block = &report.value.blocks[0];
        if block.classification != "stable" {
            let lowest = block.modes.first().map_or(f64::NAN, |m| m.eigenvalue);
            return Err(DftError::InvalidInput(format!(
                "SAPT: monomer {who} has non-positive curvature of its internal orbital                  Hessian (lowest eigenvalue {lowest:.3e} Hartree, resolution {:.1e}, {}): its                  first-order response is not unique. This is a partially filled degenerate                  shell or a saddle point of the SCF, which needs a multireference treatment                  rather than SAPT on a single determinant.",
                block.resolution, block.classification
            )));
        }
    }

    // ---- dimer AO space ----
    let mut atoms = mol_a.atoms.clone();
    atoms.extend(mol_b.atoms.iter().cloned());
    // The dimer's spin state is fixed by the monomers, not by whatever the
    // caller happened to set on the combined molecule: the supermolecular
    // and ghost-monomer references must describe the SAME determinant the
    // SAPT terms are built from, or `delta` compares two different states.
    // High-spin coupling — alpha and beta counts simply add — is the one
    // choice consistent with the product of the two monomer determinants.
    let mult_d = (mol_a.multiplicity - 1) + (mol_b.multiplicity - 1) + 1;
    let mol_d = Molecule::new(atoms)
        .with_charge(mol_a.charge + mol_b.charge)
        .with_multiplicity(mult_d);
    let mb_d = MolBasis::build(&mol_d, set)?;
    let nao_a = mb_a.nao_sph;
    let nao_d = mb_d.nao_sph;
    let na = mol_a.len();

    let p_a = embed(&scf_a.p_total(), nao_d, 0);
    let p_b = embed(&scf_b.p_total(), nao_d, nao_a);

    // Nuclear-attraction operators of each monomer's nuclei (dimer basis).
    let z_d: Vec<f64> = mol_d.atoms.iter().map(|a| a.z as f64).collect();
    let mut za = z_d.clone();
    let mut zb = z_d.clone();
    for (k, (a, b)) in za.iter_mut().zip(zb.iter_mut()).enumerate() {
        if k < na {
            *b = 0.0;
        } else {
            *a = 0.0;
        }
    }
    let v_a = nuclear_matrix(&mb_d, &mol_d, Some(&za));
    let v_b = nuclear_matrix(&mb_d, &mol_d, Some(&zb));

    // Cross nuclear repulsion.
    let mut e_nn_ab = 0.0;
    for i in 0..na {
        for j in na..mol_d.len() {
            let r = (mol_d.atoms[i].position - mol_d.atoms[j].position).norm();
            e_nn_ab += z_d[i] * z_d[j] / r;
        }
    }

    // Counterpoise uses a common dimer basis and preserves the monomer spin states.
    let counterpoise = || -> Result<_> {
        let ghost = |keep_a: bool| -> Molecule {
            let mut atoms = mol_d.atoms.clone();
            for (k, at) in atoms.iter_mut().enumerate() {
                let is_a = k < na;
                if is_a != keep_a {
                    at.z = 0; // basis functions stay, nucleus and electrons go
                }
            }
            // Charge AND spin follow the monomer being kept: a ghost monomer
            // must be the same electronic state as the real one, or the
            // counterpoise subtraction removes a different energy than the one
            // the basis-set superposition added.
            let keep = if keep_a { mol_a } else { mol_b };
            Molecule::new(atoms)
                .with_charge(keep.charge)
                .with_multiplicity(keep.multiplicity)
        };
        let grid_d = if needs_grid {
            Some(MolecularGrid::build(&mol_d, gopts)?)
        } else {
            None
        };
        let e_dimer = run_scf(&mol_d, &mb_d, opts, functional, grid_d.as_ref())?.energy;
        let mol_a_ghost = ghost(true);
        let mol_b_ghost = ghost(false);
        let scf_a_cp = run_scf(&mol_a_ghost, &mb_d, opts, functional, grid_d.as_ref())?;
        let scf_b_cp = run_scf(&mol_b_ghost, &mb_d, opts, functional, grid_d.as_ref())?;
        Ok((
            e_dimer,
            [mol_a_ghost, mol_b_ghost],
            [scf_a_cp, scf_b_cp],
            grid_d,
        ))
    };

    let limit_mb = match opts.eri_mode {
        crate::scf::EriMode::InCore { limit_mb } => limit_mb,
        _ => opts.incore_mb,
    };
    let two_electron_integrals = || {
        crate::integrals::eri::eri_incore(&mb_d, crate::integrals::CoulombKernel::Coulomb, limit_mb)
    };
    let s_d = overlap_matrix(&mb_d);
    if variant == SaptVariant::Sapt30Uncoupled {
        let eri = two_electron_integrals()?;
        let interaction = antisymmetry::Interaction {
            eri: &eri,
            overlap: &s_d,
            external: [&v_a, &v_b],
            nuclear: e_nn_ab,
        };
        let terms = third_order::terms(&interaction, [&scf_a, &scf_b], limit_mb)?;
        let [e_ind_a, e_ind_b, dispersion] = terms.second_bare;
        let e_exch_ind_directions = [
            terms.second_antisymmetrized[0] - e_ind_a,
            terms.second_antisymmetrized[1] - e_ind_b,
        ];
        let (dimer, _, [a_cp, b_cp], _) = counterpoise()?;
        let e_int_supermolecular = dimer - a_cp.energy - b_cp.energy;
        let mut result = SaptResult {
            e_monomer_a: scf_a.energy,
            e_monomer_b: scf_b.energy,
            e_elst: terms.electrostatics,
            e_exch: terms.first - terms.electrostatics,
            e_ind_a,
            e_ind_b,
            e_exch_disp: terms.second_antisymmetrized[2] - dispersion,
            e_exch_ind: e_exch_ind_directions.iter().sum(),
            e_exch_ind_directions,
            exchange_scale: None,
            delta_hf: None,
            third_order: Some(terms.third),
            sapt2_plus3: None,
            regularized_induction: None,
            relaxed_induction30: None,
            correlation_terms: None,
            e_disp_sos_uncoupled: dispersion,
            e_disp_coupled: dispersion,
            e_int_supermolecular,
            delta: 0.0,
            e_disp_unc: dispersion,
            asymptotic_shift: [None, None],
            rpa: None,
            charge_transfer: None,
            partition: None,
        };
        result.delta = result.e_int_supermolecular - result.interaction_total();
        return Ok(result);
    }

    let jk_d = JkEngine::from_options(&mol_d, &mb_d, opts)?;
    let half = |p: &Matrix| -> Matrix {
        let mut h = p.clone();
        for v in h.as_mut_slice() {
            *v *= 0.5;
        }
        h
    };
    let (j_b, _) = jk_d.build(&mb_d, &p_b, [&half(&p_b), &half(&p_b)], false)?;
    let (j_a, _) = jk_d.build(&mb_d, &p_a, [&half(&p_a), &half(&p_a)], false)?;

    // ---- E_elst^(1) ----
    let e_elst =
        p_a.frobenius_dot(&v_b) + p_b.frobenius_dot(&v_a) + p_a.frobenius_dot(&j_b) + e_nn_ab;

    // ---- E_exch^(1): Heitler-London via the Loewdin density ----
    // Occupations per spin. A restricted monomer contributes the same
    // spatial orbitals to both spin channels.
    let occ = |scf: &crate::scf::ScfResult, spin: usize| -> usize {
        if spin == 0 {
            scf.n_alpha
        } else {
            scf.n_beta
        }
    };
    fn mo(scf: &crate::scf::ScfResult, spin: usize) -> &Matrix {
        if scf.unrestricted {
            &scf.mo_coeff[spin]
        } else {
            &scf.mo_coeff[0]
        }
    }
    // Loewdin-orthogonalized Heitler-London density, per spin:
    // D^s = C^s (C^s^T S C^s)^{-1} C^s^T, summed over the two channels.
    // Closed shell is the special case D^alpha = D^beta, recovering the
    // familiar factor of two.
    let hl_spin = |spin| {
        let (oa, ob) = (occ(&scf_a, spin), occ(&scf_b, spin));
        let nocc = oa + ob;
        let (ca, cb) = (mo(&scf_a, spin), mo(&scf_b, spin));
        let mut c_un = Matrix::zeros(nao_d, nocc);
        for i in 0..oa {
            for mu in 0..nao_a {
                c_un[(mu, i)] = ca[(mu, i)];
            }
        }
        for i in 0..ob {
            for mu in 0..mb_b.nao_sph {
                c_un[(mu + nao_a, oa + i)] = cb[(mu, i)];
            }
        }
        OccupiedProjector::new(c_un, &s_d)
    };
    let hl = [hl_spin(0)?, hl_spin(1)?];
    let d_hl_spin = [&hl[0].density, &hl[1].density];
    let d_hl = {
        let mut d = hl[0].density.clone();
        for (t, b) in d.as_mut_slice().iter_mut().zip(d_hl_spin[1].as_slice()) {
            *t += *b;
        }
        d
    };
    let h_d = {
        let mut h = kinetic_matrix(&mb_d);
        let v_full = nuclear_matrix(&mb_d, &mol_d, None);
        for (hv, vv) in h.as_mut_slice().iter_mut().zip(v_full.as_slice()) {
            *hv += *vv;
        }
        h
    };
    let (e_hl_el, fock_hl) = hf_energy_and_fock(&jk_d, &mb_d, &h_d, &d_hl, d_hl_spin)?;
    let e_hl = e_hl_el + mol_d.nuclear_repulsion();
    // Monomer HF-expression energies on the SAME determinants (in the
    // dimer space the padded blocks reproduce the monomer values, so the
    // cheap monomer-basis evaluation is exact).
    let jk_a = JkEngine::from_options(mol_a, &mb_a, opts)?;
    let jk_b = JkEngine::from_options(mol_b, &mb_b, opts)?;
    let h_a = {
        let mut h = kinetic_matrix(&mb_a);
        let v = nuclear_matrix(&mb_a, mol_a, None);
        for (hv, vv) in h.as_mut_slice().iter_mut().zip(v.as_slice()) {
            *hv += *vv;
        }
        h
    };
    let h_b = {
        let mut h = kinetic_matrix(&mb_b);
        let v = nuclear_matrix(&mb_b, mol_b, None);
        for (hv, vv) in h.as_mut_slice().iter_mut().zip(v.as_slice()) {
            *hv += *vv;
        }
        h
    };
    let pa_m = scf_a.p_total();
    let pb_m = scf_b.p_total();
    let (e_a_hf_el, fock_a_hf) = hf_energy_and_fock(
        &jk_a,
        &mb_a,
        &h_a,
        &pa_m,
        [&scf_a.density[0], &scf_a.density[1]],
    )?;
    let e_a_hf = e_a_hf_el + mol_a.nuclear_repulsion();
    let (e_b_hf_el, fock_b_hf) = hf_energy_and_fock(
        &jk_b,
        &mb_b,
        &h_b,
        &pb_m,
        [&scf_b.density[0], &scf_b.density[1]],
    )?;
    let e_b_hf = e_b_hf_el + mol_b.nuclear_repulsion();
    let hl_exchange = e_hl - e_a_hf - e_b_hf - e_elst;

    // ---- E_ind^(2), coupled ----
    // Electrostatic potential of B on A's AO block (and vice versa).
    let omega_b_full = {
        let mut o = v_b.clone();
        for (ov, jv) in o.as_mut_slice().iter_mut().zip(j_b.as_slice()) {
            *ov += *jv;
        }
        o
    };
    let omega_a_full = {
        let mut o = v_a.clone();
        for (ov, jv) in o.as_mut_slice().iter_mut().zip(j_a.as_slice()) {
            *ov += *jv;
        }
        o
    };
    let block = |m: &Matrix, off: usize, nao: usize| -> Matrix {
        let mut out = Matrix::zeros(nao, nao);
        for r in 0..nao {
            for c in 0..nao {
                out[(r, c)] = m[(r + off, c + off)];
            }
        }
        out
    };
    let omega_b_on_a = block(&omega_b_full, 0, nao_a);
    let omega_a_on_b = block(&omega_a_full, nao_a, mb_b.nao_sph);

    // Every second-order term below is built on the FIRST-ORDER orbital
    // rotation U, and second-order perturbation theory is only meaningful
    // while that rotation is small. When the monomer's orbital Hessian is
    // nearly singular — a partially filled degenerate shell, whose
    // symmetry-broken SCF solution is a saddle rather than a minimum — U
    // blows up while every individual orbital-energy denominator still
    // looks healthy, so a gap test does NOT catch it. Measuring U itself
    // does, and it is the quantity whose smallness the expansion actually
    // assumes.
    //
    // The recovered amplitudes are `U = C_v^T S dP^s S C_o`; the same
    // density tangents enter the Heitler-London exchange derivative.
    let max_rotation =
        |scf: &crate::scf::ScfResult, s: &Matrix, dp: &[Matrix; 2], nao: usize| -> f64 {
            let mut worst = 0.0_f64;
            let spins = if scf.unrestricted { 2 } else { 1 };
            for spin in 0..spins {
                let c = mo(scf, spin);
                let no = occ(scf, spin);
                let sdps = s.matmul(&dp[spin].matmul(s));
                for i in 0..no {
                    for a_v in no..c.cols {
                        let mut t = 0.0;
                        for mu in 0..nao {
                            let cma = c[(mu, a_v)];
                            if cma == 0.0 {
                                continue;
                            }
                            for nu in 0..nao {
                                t += cma * sdps[(mu, nu)] * c[(nu, i)];
                            }
                        }
                        worst = worst.max(t.abs());
                    }
                }
            }
            worst
        };
    // `|U| < 1` is the loosest statement of "the rotation is still small",
    // and the threshold is deliberately set there rather than tighter.
    // Genuinely strong induction reaches a few tenths — a lithium atom
    // (polarizability ~164 a.u.) beside a water gives 0.26, which is
    // physics and must not be refused. A singular orbital Hessian, by
    // contrast, overshoots by orders of magnitude: the triplet-oxygen case
    // that motivated this check lands at 19 in STO-3G and 96 in 6-31G. The
    // two regimes are not close, so the test needs no fine tuning.
    const MAX_U: f64 = 1.0;
    let rotation_check = |who: &str, u: f64| -> Result<()> {
        if u.is_finite() && u < MAX_U {
            return Ok(());
        }
        Err(DftError::InvalidInput(format!(
            "SAPT: the first-order orbital rotation of monomer {who} reaches \
             |U|max = {u:.3e}, outside the small-rotation regime the \
             second-order expansion assumes, so its induction, \
             exchange-induction and coupled-dispersion terms would be \
             meaningless numbers rather than energies. This is the signature \
             of a nearly singular orbital Hessian — typically a partially \
             filled degenerate shell whose symmetry-broken SCF solution is a \
             saddle point, which needs a multireference treatment rather than \
             SAPT on a single determinant. Note that the orbital-energy gaps \
             can look perfectly healthy in this situation."
        )))
    };

    let solver_a = ResponseSolver::new(mol_a, &mb_a, functional, grid_a.as_ref(), &scf_a, opts)?;
    let dp_a = &solver_a.solve(std::slice::from_ref(&omega_b_on_a))?[0];
    let mut dp_a_tot = dp_a[0].clone();
    for (t, b) in dp_a_tot.as_mut_slice().iter_mut().zip(dp_a[1].as_slice()) {
        *t += *b;
    }
    let e_ind_a = 0.5 * dp_a_tot.frobenius_dot(&omega_b_on_a);
    let s_a = overlap_matrix(&mb_a);
    rotation_check("A", max_rotation(&scf_a, &s_a, dp_a, nao_a))?;

    let solver_b = ResponseSolver::new(mol_b, &mb_b, functional, grid_b.as_ref(), &scf_b, opts)?;
    let dp_b = &solver_b.solve(std::slice::from_ref(&omega_a_on_b))?[0];
    let mut dp_b_tot = dp_b[0].clone();
    for (t, b) in dp_b_tot.as_mut_slice().iter_mut().zip(dp_b[1].as_slice()) {
        *t += *b;
    }
    let e_ind_b = 0.5 * dp_b_tot.frobenius_dot(&omega_a_on_b);
    let s_b = overlap_matrix(&mb_b);
    rotation_check("B", max_rotation(&scf_b, &s_b, dp_b, mb_b.nao_sph))?;

    // ---- atom-pair partition of electrostatics and induction ----
    let sapt_partition = if config.partition {
        let nb = mol_b.len();
        let local_a = partition::local_orbitals(mol_a, &mb_a, &scf_a)?;
        let local_b = partition::local_orbitals(mol_b, &mb_b, &scf_b)?;
        let rho_a = partition::atom_densities(&local_a, na, nao_d, 0);
        let rho_b = partition::atom_densities(&local_b, nb, nao_d, nao_a);
        let attraction = |atom: usize| -> Matrix {
            let mut q = vec![0.0; mol_d.len()];
            q[atom] = z_d[atom];
            nuclear_matrix(&mb_d, &mol_d, Some(&q))
        };
        let va_atoms: Vec<Matrix> = (0..na).map(attraction).collect();
        let vb_atoms: Vec<Matrix> = (na..na + nb).map(attraction).collect();
        let coulomb = |rho: &Matrix| -> Result<Matrix> {
            Ok(jk_d.build(&mb_d, rho, [&half(rho), &half(rho)], false)?.0)
        };
        let ja: Vec<Matrix> = rho_a.iter().map(coulomb).collect::<Result<_>>()?;
        let jb: Vec<Matrix> = rho_b.iter().map(coulomb).collect::<Result<_>>()?;
        let mut electrostatic = Matrix::zeros(na, nb);
        for a in 0..na {
            for b in 0..nb {
                let r = (mol_d.atoms[a].position - mol_d.atoms[na + b].position).norm();
                electrostatic[(a, b)] = rho_a[a].frobenius_dot(&jb[b])
                    + rho_a[a].frobenius_dot(&vb_atoms[b])
                    + rho_b[b].frobenius_dot(&va_atoms[a])
                    + z_d[a] * z_d[na + b] / r;
            }
        }
        let plus = |x: &Matrix, y: &Matrix| -> Matrix {
            let mut m = x.clone();
            for (p, q) in m.as_mut_slice().iter_mut().zip(y.as_slice()) {
                *p += q;
            }
            m
        };
        let on_a: Vec<Matrix> = (0..nb)
            .map(|b| block(&plus(&vb_atoms[b], &jb[b]), 0, nao_a))
            .collect();
        let on_b: Vec<Matrix> = (0..na)
            .map(|a| block(&plus(&va_atoms[a], &ja[a]), nao_a, mb_b.nao_sph))
            .collect();
        let induction_a =
            partition::induction_partition(&solver_a, &scf_a, &s_a, &local_a, na, &on_a)?;
        let induction_b =
            partition::induction_partition(&solver_b, &scf_b, &s_b, &local_b, nb, &on_b)?
                .transpose();
        let total = |m: &Matrix| -> f64 { m.as_slice().iter().sum() };
        let residual = (total(&electrostatic) - e_elst)
            .abs()
            .max((total(&induction_a) - e_ind_a).abs())
            .max((total(&induction_b) - e_ind_b).abs());
        Some(SaptPartition {
            electrostatic,
            induction_a,
            induction_b,
            residual,
        })
    } else {
        None
    };

    // ---- E_disp^(2), uncoupled ----
    let eri = two_electron_integrals()?;
    let nao_b = mb_b.nao_sph;
    // The uncoupled dispersion is a sum over SPIN-ORBITAL excitations
    // (i^sA -> a^sA) x (j^sB -> b^sB). For closed shells all four (sA, sB)
    // channels are identical, which is where the familiar factor of four
    // comes from; open shell needs them separately. Restricted monomers
    // therefore run the alpha channel once and weight it by two per side.
    let sp_a = if scf_a.unrestricted { 2 } else { 1 };
    let sp_b = if scf_b.unrestricted { 2 } else { 1 };
    let wt_a = if scf_a.unrestricted { 1.0 } else { 2.0 };
    let wt_b = if scf_b.unrestricted { 1.0 } else { 2.0 };
    let mut e_disp_unc = 0.0;
    for spin_a in 0..sp_a {
        for spin_b in 0..sp_b {
            let c_a = mo(&scf_a, spin_a);
            let c_b = mo(&scf_b, spin_b);
            let eps_a = if scf_a.unrestricted {
                &scf_a.mo_energy[spin_a]
            } else {
                &scf_a.mo_energy[0]
            };
            let eps_b = if scf_b.unrestricted {
                &scf_b.mo_energy[spin_b]
            } else {
                &scf_b.mo_energy[0]
            };
            let nmo_a = c_a.cols;
            let nmo_b = c_b.cols;
            let (no_a, nv_a) = (occ(&scf_a, spin_a), nmo_a - occ(&scf_a, spin_a));
            let (no_b, nv_b) = (occ(&scf_b, spin_b), nmo_b - occ(&scf_b, spin_b));
            // X[ia][λσ] = Σ_{μν∈A} C_μi C_νa (μν|λσ): per B AO pair, transform the
            // A-block matrix.
            let mut x = vec![0.0; no_a * nv_a * nao_b * nao_b];
            for lam in 0..nao_b {
                for sg in 0..nao_b {
                    let mut mblock = Matrix::zeros(nao_a, nao_a);
                    for mu in 0..nao_a {
                        for nu in 0..nao_a {
                            mblock[(mu, nu)] = eri.get(mu, nu, lam + nao_a, sg + nao_a);
                        }
                    }
                    for i in 0..no_a {
                        for a_v in 0..nv_a {
                            let mut acc = 0.0;
                            for mu in 0..nao_a {
                                let cmi = c_a[(mu, i)];
                                if cmi == 0.0 {
                                    continue;
                                }
                                for nu in 0..nao_a {
                                    acc += cmi * c_a[(nu, no_a + a_v)] * mblock[(mu, nu)];
                                }
                            }
                            x[((i * nv_a + a_v) * nao_b + lam) * nao_b + sg] = acc;
                        }
                    }
                }
            }
            for i in 0..no_a {
                for a_v in 0..nv_a {
                    let de_a = eps_a[no_a + a_v] - eps_a[i];
                    for j in 0..no_b {
                        for b_v in 0..nv_b {
                            let de_b = eps_b[no_b + b_v] - eps_b[j];
                            let mut iajb = 0.0;
                            for lam in 0..nao_b {
                                let clj = c_b[(lam, j)];
                                if clj == 0.0 {
                                    continue;
                                }
                                for sg in 0..nao_b {
                                    iajb += clj
                                        * c_b[(sg, no_b + b_v)]
                                        * x[((i * nv_a + a_v) * nao_b + lam) * nao_b + sg];
                                }
                            }
                            e_disp_unc -= wt_a * wt_b * iajb * iajb / (de_a + de_b);
                        }
                    }
                }
            }
        }
    }

    // Every stage from here on reads its orbitals through `mo`/`occ`, which
    // give the alpha channel for a restricted monomer and the real per-spin
    // ones otherwise. There are no alpha-only bindings left to accidentally
    // use on an open-shell system.

    let (e_dimer, [mol_a_ghost, mol_b_ghost], [scf_a_cp, scf_b_cp], grid_d) = counterpoise()?;
    let (e_a_cp, e_b_cp) = (scf_a_cp.energy, scf_b_cp.energy);

    // ---- charge transfer: induction with each monomer in the dimer basis ----
    let charge_transfer = if config.charge_transfer {
        // Kohn-Sham monomers with an asymptotic correction are re-converged
        // with it in the dimer basis; otherwise the ghost-basis references
        // of the counterpoise energy are the dimer-basis monomers.
        let rerun =
            |ghost: &Molecule, shift: Option<f64>| -> Result<Option<crate::scf::ScfResult>> {
                shift
                    .map(|_| run_scf(ghost, &mb_d, &corrected(shift), functional, grid_d.as_ref()))
                    .transpose()
            };
        let own_a = rerun(&mol_a_ghost, config.asymptotic_shift[0])?;
        let own_b = rerun(&mol_b_ghost, config.asymptotic_shift[1])?;
        let scf_a_d = own_a.as_ref().unwrap_or(&scf_a_cp);
        let scf_b_d = own_b.as_ref().unwrap_or(&scf_b_cp);
        let potential = |scf: &crate::scf::ScfResult, v: &Matrix| -> Result<Matrix> {
            let p = scf.p_total();
            let (j, _) = jk_d.build(&mb_d, &p, [&scf.density[0], &scf.density[1]], false)?;
            let mut m = v.clone();
            for (x, y) in m.as_mut_slice().iter_mut().zip(j.as_slice()) {
                *x += y;
            }
            Ok(m)
        };
        let induction =
            |ghost: &Molecule, scf: &crate::scf::ScfResult, omega: &Matrix| -> Result<f64> {
                let solver =
                    ResponseSolver::new(ghost, &mb_d, functional, grid_d.as_ref(), scf, opts)?;
                let dp = &solver.solve(std::slice::from_ref(omega))?[0];
                Ok(0.5 * (dp[0].frobenius_dot(omega) + dp[1].frobenius_dot(omega)))
            };
        let ind_a = induction(&mol_a_ghost, scf_a_d, &potential(scf_b_d, &v_b)?)?;
        let ind_b = induction(&mol_b_ghost, scf_b_d, &potential(scf_a_d, &v_a)?)?;
        Some(ChargeTransfer {
            e_ind_dimer_basis: [ind_a, ind_b],
            e_ct: ind_a + ind_b - e_ind_a - e_ind_b,
        })
    } else {
        None
    };
    let e_int_supermolecular = e_dimer - e_a_cp - e_b_cp;

    // First variation of the Heitler-London exchange model at the
    // unperturbed orbitals. The polarized occupied projectors satisfy the
    // electron-number constraints through the retained order. Finite
    // polarization energies would also contain higher-order terms.
    let interaction = antisymmetry::Interaction {
        eri: &eri,
        overlap: &s_d,
        external: [&v_a, &v_b],
        nuclear: e_nn_ab,
    };
    let mut exchange_scale = None;
    let mut delta_hf = None;
    let formal = match variant {
        SaptVariant::Response | SaptVariant::Sapt30Uncoupled => None,
        SaptVariant::Sapt0 | SaptVariant::Sapt2Plus3 | SaptVariant::KsFdds => {
            Some(antisymmetry::terms(
                &interaction,
                &hl,
                [&scf_a, &scf_b],
                [dp_a, dp_b],
                [&s_a, &s_b],
                nao_a,
            )?)
        }
        SaptVariant::Sapt0SingleExchange | SaptVariant::ScaledSapt0 => {
            let mut terms = single_exchange::terms(
                &interaction,
                [&scf_a, &scf_b],
                [dp_a, dp_b],
                [&s_a, &s_b],
                e_elst,
                [e_ind_a, e_ind_b],
                limit_mb,
            )?;
            if variant == SaptVariant::ScaledSapt0 {
                let first_order =
                    antisymmetry::first_order(&interaction, &hl, [scf_a.n_alpha, scf_a.n_beta]);
                let induction = [e_ind_a, e_ind_b];
                let raw = [0, 1].map(|i| terms.induction_with_exchange[i] - induction[i]);
                let scale = single_exchange::scaling_ratio(
                    first_order - e_elst,
                    terms.first_order - e_elst,
                    raw,
                    terms.exchange_dispersion,
                )?;
                let cubic = scale.powi(3);
                // Parker et al., Eqs. 4, 6 and 7: the delta-HF subtraction
                // retains first-power scaling when the explicit
                // second-order exchange terms use the third power.
                delta_hf = Some(
                    e_int_supermolecular
                        - first_order
                        - e_ind_a
                        - e_ind_b
                        - scale * raw.iter().sum::<f64>(),
                );
                exchange_scale = Some(cubic);
                terms.first_order = first_order;
                terms.induction_with_exchange = [0, 1].map(|i| induction[i] + cubic * raw[i]);
                terms.exchange_dispersion *= cubic;
            }
            Some(terms)
        }
    };
    let (e_exch, e_exch_ind_directions) = if let Some(terms) = &formal {
        (
            terms.first_order - e_elst,
            [
                terms.induction_with_exchange[0] - e_ind_a,
                terms.induction_with_exchange[1] - e_ind_b,
            ],
        )
    } else {
        let fragment_derivative = |fragment| {
            let (dp, s_mon, fock_mon, potential, offset) = if fragment == 0 {
                (dp_a, &s_a, &fock_a_hf, &omega_b_on_a, 0)
            } else {
                (dp_b, &s_b, &fock_b_hf, &omega_a_on_b, nao_a)
            };
            (0..2)
                .map(|s| {
                    let direction = hl[s].polarization_direction(&dp[s], s_mon, &s_d, offset);
                    let derivative = hl[s].energy_derivative(&direction, &fock_hl[s]);
                    derivative - dp[s].frobenius_dot(&fock_mon[s]) - dp[s].frobenius_dot(potential)
                })
                .sum::<f64>()
        };
        (hl_exchange, [0, 1].map(fragment_derivative))
    };
    let induction_a = relaxed_induction::Monomer {
        basis: &mb_a,
        scf: &scf_a,
        jk: &jk_a,
        overlap: &s_a,
        density_first: dp_a,
        perturbation: &omega_b_on_a,
    };
    let induction_b = relaxed_induction::Monomer {
        basis: &mb_b,
        scf: &scf_b,
        jk: &jk_b,
        overlap: &s_b,
        density_first: dp_b,
        perturbation: &omega_a_on_b,
    };
    let relaxed_induction30 = if diagnostics.relaxed_induction30 || higher_order {
        let a = relaxed_induction::monomer(&induction_a)?;
        let b = relaxed_induction::monomer(&induction_b)?;
        let t_a = embed(&dp_a_tot, nao_d, 0);
        let t_b = embed(&dp_b_tot, nao_d, nao_a);
        let (j, _) = jk_d.build(&mb_d, &t_b, [&t_b, &t_b], false)?;
        let mutual = t_a.frobenius_dot(&j);
        Some(SaptRelaxedInduction30 {
            induction: a.polarization + b.polarization + mutual + a.intramonomer + b.intramonomer,
            delta_hf_order2: e_int_supermolecular
                - e_elst
                - e_exch
                - e_ind_a
                - e_ind_b
                - e_exch_ind_directions.iter().sum::<f64>(),
        })
    } else {
        None
    };
    let (correlation_terms, third_order) = if diagnostics.correlation_terms || higher_order {
        if [&scf_a, &scf_b]
            .iter()
            .any(|scf| scf.unrestricted || scf.n_alpha != scf.n_beta)
        {
            return Err(DftError::InvalidInput(
                "fixed-orbital SAPT correlation requires closed-shell RHF monomers".into(),
            ));
        }
        if !matches!(
            opts.eri_mode,
            crate::scf::EriMode::InCore { .. } | crate::scf::EriMode::Direct { .. }
        ) {
            return Err(DftError::InvalidInput(
                "fixed-orbital SAPT correlation requires exact Coulomb integrals in the RHF monomers".into(),
            ));
        }
        let (correlation, third) = correlation::terms(
            &interaction,
            [&induction_a, &induction_b],
            [&h_a, &h_b],
            [e_ind_a + e_ind_b, e_exch_ind_directions.iter().sum()],
            higher_order,
            limit_mb,
        )?;
        (Some(correlation), third)
    } else {
        (None, None)
    };
    let sapt2_plus3 = match (correlation_terms, third_order, relaxed_induction30) {
        (Some(correlated), Some(cubic), Some(relaxed)) => {
            // Patkowski (2010), Eq. 39: estimate the relaxed exchange
            // induction by the ratio of cubic polarization energies.
            let exchange = if cubic.exchange_induction == 0.0 {
                0.0
            } else {
                cubic.exchange_induction * (relaxed.induction / cubic.induction)
            };
            if !exchange.is_finite() {
                return Err(DftError::InvalidInput(
                    "relaxed cubic exchange-induction scale is singular".into(),
                ));
            }
            let remainder = relaxed.delta_hf_order2 - relaxed.induction - exchange;
            delta_hf = Some(remainder);
            Some(Sapt2Plus3 {
                relaxed_exchange_induction30: exchange,
                delta_hf_order3: remainder,
                correlated_electrostatics: correlated.relaxed_electrostatics.iter().flatten().sum(),
                correlated_exchange: correlated.exchange[..2].iter().sum(),
                correlated_induction: correlated.truncated_induction22.iter().sum::<f64>()
                    + correlated.truncated_exchange_induction22,
                correlated_dispersion: correlated.second_order_polarization[2].iter().sum(),
                cubic_dispersion_and_mixed: cubic.dispersion
                    + cubic.induction_dispersion
                    + cubic.exchange_dispersion
                    + cubic.exchange_induction_dispersion,
            })
        }
        _ => None,
    };
    let regularized_induction = diagnostics
        .regularization_eta
        .map(|eta| -> Result<_> {
            let bare = [&omega_b_on_a, &omega_a_on_b];
            let references = [&mb_a, &mb_b];
            let charges = [&zb, &za];
            let solve = |index: usize, solver: &ResponseSolver<'_>| -> Result<[Matrix; 2]> {
                let damping = crate::integrals::one_electron::nuclear_penetration_matrix(
                    references[index],
                    &mol_d,
                    charges[index],
                    eta,
                );
                if damping.as_slice().iter().any(|v| !v.is_finite()) {
                    return Err(DftError::InvalidInput(
                        "SAPT regularization produced a nonfinite damped nuclear operator".into(),
                    ));
                }
                let mut source = bare[index].clone();
                for (value, &short) in source.as_mut_slice().iter_mut().zip(damping.as_slice()) {
                    *value -= short;
                }
                Ok(solver.solve(&[source])?.remove(0))
            };
            let tangents = [solve(0, &solver_a)?, solve(1, &solver_b)?];
            let induction = std::array::from_fn(|i| {
                0.5 * tangents[i]
                    .iter()
                    .map(|d| d.frobenius_dot(bare[i]))
                    .sum::<f64>()
            });
            let combined = antisymmetry::coupled_induction(
                &interaction,
                &hl,
                [scf_a.n_alpha, scf_a.n_beta],
                tangents.each_ref(),
                [&s_a, &s_b],
                nao_a,
            );
            let unregularized = [e_ind_a, e_ind_b];
            Ok(SaptRegularizedInduction {
                eta,
                induction,
                exchange_induction: std::array::from_fn(|i| combined[i] - induction[i]),
                charge_transfer: std::array::from_fn(|i| {
                    unregularized[i] + e_exch_ind_directions[i] - combined[i]
                }),
            })
        })
        .transpose()?;

    let spin_weight = |scf: &crate::scf::ScfResult| if scf.unrestricted { 1.0 } else { 2.0 };
    let spin_factor = spin_weight(&scf_a) * spin_weight(&scf_b);

    // AO transition density of one state, summed over its spin channels and
    // optionally projected channel by channel.
    //
    // Unit weight in every channel: the spin counting is the `spin_factor`
    // of the sum-over-states, not a scale on the density. Folding it in
    // here instead would square it, since the Coulomb coupling is bilinear
    // in the two densities.
    let td_density = |x: &[Matrix; 2],
                      scf: &crate::scf::ScfResult,
                      nao: usize,
                      proj: Option<&[Matrix; 2]>|
     -> Matrix {
        let mut p = Matrix::zeros(nao, nao);
        let spins = if scf.unrestricted { 2 } else { 1 };
        for spin in 0..spins {
            let c = mo(scf, spin);
            let no = occ(scf, spin);
            let nv = c.cols - no;
            if no == 0 || nv == 0 {
                continue;
            }
            let t = transition_density(&x[spin], c, no, nv, nao);
            // Pauli quenching is a one-particle projection and therefore
            // acts within a spin channel; projecting the spin-summed
            // density instead would let A's alpha transition see B's beta
            // occupied space.
            let t = match proj {
                Some(q) => q[spin].matmul(&t).matmul(&q[spin].transpose()),
                None => t,
            };
            for (pv, tv) in p.as_mut_slice().iter_mut().zip(t.as_slice()) {
                *pv += tv;
            }
        }
        p
    };

    // Use one construction for the full-response spectral and imaginary-frequency
    // dispersion. The legacy model also uses the transition-density projection.
    let full_response = |projectors: Option<(&[Matrix; 2], &[Matrix; 2])>,
                         exchange_uncoupled: f64|
     -> Result<RpaDispersion> {
        use crate::response::tddft::{TdaSolver, TdaSpin};
        let spin_mode = |scf: &crate::scf::ScfResult| {
            if scf.unrestricted {
                TdaSpin::SpinResolved
            } else {
                TdaSpin::Singlet
            }
        };
        let td_a = TdaSolver::new(
            mol_a,
            &mb_a,
            functional,
            grid_a.as_ref(),
            &scf_a,
            opts,
            spin_mode(&scf_a),
        )?;
        let td_b = TdaSolver::new(
            mol_b,
            &mb_b,
            functional,
            grid_b.as_ref(),
            &scf_b,
            opts,
            spin_mode(&scf_b),
        )?;
        // Dense A + B and A - B in the unit transition basis, shared by the
        // RPA states and the Casimir-Polder susceptibilities.
        let (apb_a, amb_a) = td_a.orbital_hessians()?;
        let (apb_b, amb_b) = td_b.orbital_hessians()?;
        let rpa_a = td_a.rpa_from_hessians(&apb_a, &amb_a, td_a.dim())?;
        let rpa_b = td_b.rpa_from_hessians(&apb_b, &amb_b, td_b.dim())?;
        let wa_r: Vec<f64> = rpa_a.iter().map(|s| s.energy).collect();
        let wb_r: Vec<f64> = rpa_b.iter().map(|s| s.energy).collect();
        let densities = |states: &[crate::response::tddft::TdaState],
                         scf: &crate::scf::ScfResult,
                         nao: usize,
                         proj: Option<&[Matrix; 2]>|
         -> Vec<Matrix> {
            states
                .iter()
                .map(|s| td_density(&s.x, scf, nao, proj))
                .collect()
        };
        let e_disp = dispersion_sum_over_states(
            &wa_r,
            &densities(&rpa_a, &scf_a, nao_a, None),
            &wb_r,
            &densities(&rpa_b, &scf_b, nao_b, None),
            &eri,
            nao_a,
            nao_b,
            spin_factor,
        );
        let exchange_dispersion = if let Some((pa, pb)) = projectors {
            dispersion_sum_over_states(
                &wa_r,
                &densities(&rpa_a, &scf_a, nao_a, Some(pa)),
                &wb_r,
                &densities(&rpa_b, &scf_b, nao_b, Some(pb)),
                &eri,
                nao_a,
                nao_b,
                spin_factor,
            ) - e_disp
        } else {
            exchange_uncoupled
        };
        // Unit transition densities and orbital gaps in the same order.
        let units = |scf: &crate::scf::ScfResult, nao: usize| -> (Vec<f64>, Vec<Matrix>) {
            let mut w = Vec::new();
            let mut p = Vec::new();
            let spins = if scf.unrestricted { 2 } else { 1 };
            for spin in 0..spins {
                let c = mo(scf, spin);
                let eps = &scf.mo_energy[if scf.unrestricted { spin } else { 0 }];
                let no = occ(scf, spin);
                let nv = c.cols - no;
                for a_v in 0..nv {
                    for i in 0..no {
                        let mut x = Matrix::zeros(nv, no);
                        x[(a_v, i)] = 1.0;
                        p.push(transition_density(&x, c, no, nv, nao));
                        w.push(eps[no + a_v] - eps[i]);
                    }
                }
            }
            (w, p)
        };
        let (gaps_a, units_a) = units(&scf_a, nao_a);
        let (gaps_b, units_b) = units(&scf_b, nao_b);
        let coupling = dispersion::coupling_matrix(&units_a, &units_b, &eri, nao_a, nao_b);
        let points = if config.casimir_polder_points == 0 {
            24
        } else {
            config.casimir_polder_points
        };
        let (e_cp, change) = dispersion::casimir_polder(
            &dispersion::Response::coupled(&apb_a, &amb_a)?,
            &dispersion::Response::coupled(&apb_b, &amb_b)?,
            &coupling,
            spin_factor,
            points,
        )?;
        let (e_cp_unc, _) = dispersion::casimir_polder(
            &dispersion::Response::Uncoupled(gaps_a),
            &dispersion::Response::Uncoupled(gaps_b),
            &coupling,
            spin_factor,
            points,
        )?;
        Ok(RpaDispersion {
            e_disp,
            e_exch_disp: exchange_dispersion,
            e_disp_casimir_polder: e_cp,
            casimir_polder_change: change,
            e_disp_casimir_polder_uncoupled: e_cp_unc,
            points: 2 * points,
        })
    };

    let (e_disp_coupled, e_disp_sos_uncoupled, e_exch_disp, rpa) = if let Some(terms) = formal {
        if ks_fdds {
            let response = full_response(None, terms.exchange_dispersion)?;
            (
                response.e_disp_casimir_polder,
                terms.dispersion,
                terms.exchange_dispersion,
                Some(response),
            )
        } else {
            (
                terms.dispersion,
                terms.dispersion,
                terms.exchange_dispersion,
                None,
            )
        }
    } else {
        // ---- E_disp^(2), coupled (sum over TDA states) ----
        // E = -4 sum_{n in A, m in B} V_nm^2 / (w_n + w_m) with
        // V_nm = sum_{mu nu in A, lam sig in B} P^n_mu nu P^m_lam sig (mu nu|lam sig)
        // and the transition density P^n_mu nu = sum_ai X^n_ai C_mu,a C_nu,i.
        // No transcribed formula: switching the monomer response off collapses
        // X to a single (a,i) and reproduces the uncoupled sum above, which is
        // what pins the spin factor.
        // Open-shell monomers need the SPIN-RESOLVED excitation basis: the
        // singlet mode ties the beta block to the alpha one, which only holds
        // when the two spin channels share their orbitals. The transition
        // density is then a sum over both channels rather than twice the alpha
        // one, and the closed-shell answer is recovered exactly because the
        // two channels coincide there.
        // Spatial (singlet) basis counts 2 per side; a spin-resolved one
        // counts 1 because it already enumerates both channels.

        // The monomer TDA spectra, solved ONCE and shared by the coupled
        // dispersion and its exchange quenching. Sharing is not just cheaper:
        // the exchange term is a DIFFERENCE of two sums over these states, so
        // letting each stage run its own eigensolve would fold any discrepancy
        // between the two solves straight into the answer.
        let (states_a, states_b) = {
            use crate::response::tddft::{TdaSolver, TdaSpin};
            let spin_mode = |scf: &crate::scf::ScfResult| {
                if scf.unrestricted {
                    TdaSpin::SpinResolved
                } else {
                    TdaSpin::Singlet
                }
            };
            // Dimension of the excitation basis actually being solved: one
            // block per spin channel when spin-resolved, each with its own
            // occupied/virtual split (which need not match between channels).
            let n_states = |scf: &crate::scf::ScfResult| -> usize {
                let spins = if scf.unrestricted { 2 } else { 1 };
                (0..spins)
                    .map(|s| {
                        let no = occ(scf, s);
                        no * (mo(scf, s).cols - no)
                    })
                    .sum()
            };
            (
                TdaSolver::new(
                    mol_a,
                    &mb_a,
                    functional,
                    grid_a.as_ref(),
                    &scf_a,
                    opts,
                    spin_mode(&scf_a),
                )?
                .solve_dense(n_states(&scf_a))?,
                TdaSolver::new(
                    mol_b,
                    &mb_b,
                    functional,
                    grid_b.as_ref(),
                    &scf_b,
                    opts,
                    spin_mode(&scf_b),
                )?
                .solve_dense(n_states(&scf_b))?,
            )
        };

        let wa: Vec<f64> = states_a.iter().map(|s| s.energy).collect();
        let wb: Vec<f64> = states_b.iter().map(|s| s.energy).collect();

        let e_disp_coupled = {
            let pa: Vec<Matrix> = states_a
                .iter()
                .map(|s| td_density(&s.x, &scf_a, nao_a, None))
                .collect();
            let pb: Vec<Matrix> = states_b
                .iter()
                .map(|s| td_density(&s.x, &scf_b, nao_b, None))
                .collect();
            dispersion_sum_over_states(&wa, &pa, &wb, &pb, &eri, nao_a, nao_b, spin_factor)
        };
        // Same machinery driven by the UNCOUPLED excitations (X = e_{ai},
        // w = eps_a - eps_i). This must reproduce `e_disp_unc` exactly; any
        // gap is the spin factor of the transition density, so the ratio
        // calibrates the coupled value rather than being guessed.
        // Spin-resolved when the monomer is, exactly as the coupled path is,
        // so the equality with `e_disp_unc` gates the OPEN-shell counting too
        // and not only the closed-shell one.
        let e_disp_sos_uncoupled = {
            let build = |scf: &crate::scf::ScfResult, nao: usize| {
                let mut w = Vec::new();
                let mut p = Vec::new();
                let spins = if scf.unrestricted { 2 } else { 1 };
                for spin in 0..spins {
                    let c = mo(scf, spin);
                    let eps = if scf.unrestricted {
                        &scf.mo_energy[spin]
                    } else {
                        &scf.mo_energy[0]
                    };
                    let no = occ(scf, spin);
                    let nv = c.cols - no;
                    for a_v in 0..nv {
                        for i in 0..no {
                            let mut x = Matrix::zeros(nv, no);
                            x[(a_v, i)] = 1.0;
                            p.push(transition_density(&x, c, no, nv, nao));
                            w.push(eps[no + a_v] - eps[i]);
                        }
                    }
                }
                (w, p)
            };
            let (wa, pa) = build(&scf_a, nao_a);
            let (wb, pb) = build(&scf_b, nao_b);
            dispersion_sum_over_states(&wa, &pa, &wb, &pb, &eri, nao_a, nao_b, spin_factor)
        };

        // ---- E_exch-disp^(2) ----
        // Dispersion is quenched by the Pauli exclusion between the monomers:
        // the transition density of A must not leak into B's occupied space.
        // Projecting it out with (1 - rho_B S) — which carries the overlap to
        // ALL orders, so no S^2 truncation — and recomputing the same
        // sum-over-states gives the quenched dispersion; the difference from
        // the unprojected value is the exchange-dispersion term.
        let e_exch_disp = {
            // Overlap block between the monomer AO sets.
            let s_ab = {
                let mut m = Matrix::zeros(nao_a, nao_b);
                for mu in 0..nao_a {
                    for nu in 0..nao_b {
                        m[(mu, nu)] = s_d[(mu, nu + nao_a)];
                    }
                }
                m
            };
            let s_ba = s_ab.transpose();
            // rho_X^s = C_occ^s (C_occ^s)^T, idempotent under S within the
            // monomer. Per spin, because the exclusion the projector enforces
            // is between electrons of the SAME spin.
            let rho = |scf: &crate::scf::ScfResult, spin: usize, nao: usize| -> Matrix {
                let no = occ(scf, spin);
                let src = mo(scf, spin);
                let mut c = Matrix::zeros(nao, no);
                for mu in 0..nao {
                    for i in 0..no {
                        c[(mu, i)] = src[(mu, i)];
                    }
                }
                c.matmul_transpose_seq(&c)
            };
            let minus_from_identity = |t: &Matrix, n: usize| -> Matrix {
                let mut m = Matrix::zeros(n, n);
                for mu in 0..n {
                    m[(mu, mu)] = 1.0;
                }
                for mu in 0..n {
                    for nu in 0..n {
                        m[(mu, nu)] -= t[(mu, nu)];
                    }
                }
                m
            };
            // P' = (1 - rho_A^s S_ab rho_B^s S_ba) P^s, and the mirror on B.
            let mut proj_a = [Matrix::zeros(nao_a, nao_a), Matrix::zeros(nao_a, nao_a)];
            let mut proj_b = [Matrix::zeros(nao_b, nao_b), Matrix::zeros(nao_b, nao_b)];
            for spin in 0..2 {
                let ra = rho(&scf_a, spin, nao_a);
                let rb = rho(&scf_b, spin, nao_b);
                proj_a[spin] =
                    minus_from_identity(&ra.matmul(&s_ab).matmul(&rb).matmul(&s_ba), nao_a);
                proj_b[spin] =
                    minus_from_identity(&rb.matmul(&s_ba).matmul(&ra).matmul(&s_ab), nao_b);
            }
            let pa: Vec<Matrix> = states_a
                .iter()
                .map(|s| td_density(&s.x, &scf_a, nao_a, Some(&proj_a)))
                .collect();
            let pb: Vec<Matrix> = states_b
                .iter()
                .map(|s| td_density(&s.x, &scf_b, nao_b, Some(&proj_b)))
                .collect();
            // Same states, same spin factor as the coupled dispersion this is
            // differenced against — the difference is then purely the
            // projection, which is what the term is defined to be.
            let quenched =
                dispersion_sum_over_states(&wa, &pa, &wb, &pb, &eri, nao_a, nao_b, spin_factor);
            (quenched - e_disp_coupled, proj_a, proj_b)
        };
        let (e_exch_disp, proj_a, proj_b) = e_exch_disp;

        let rpa = config
            .rpa_dispersion
            .then(|| full_response(Some((&proj_a, &proj_b)), 0.0))
            .transpose()?;

        (e_disp_coupled, e_disp_sos_uncoupled, e_exch_disp, rpa)
    };

    let mut result = SaptResult {
        e_exch_disp,
        e_exch_ind: e_exch_ind_directions.iter().sum(),
        e_exch_ind_directions,
        exchange_scale,
        delta_hf,
        third_order,
        sapt2_plus3,
        regularized_induction,
        relaxed_induction30,
        correlation_terms,
        e_disp_sos_uncoupled,
        e_disp_coupled,
        e_int_supermolecular,
        delta: 0.0,
        e_monomer_a: scf_a.energy,
        e_monomer_b: scf_b.energy,
        e_elst,
        e_exch,
        e_ind_a,
        e_ind_b,
        e_disp_unc,
        asymptotic_shift: config.asymptotic_shift,
        rpa,
        charge_transfer,
        partition: sapt_partition,
    };
    result.delta = result.e_int_supermolecular - result.interaction_total();
    Ok(result)
}
