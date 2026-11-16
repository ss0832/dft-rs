// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Phase 8 gate (Stage A): the analytic CPHF Hessian against the numerical
//! Hessian (central differences of the analytic gradient) for HF.

use dft_rs::basis::{data, MolBasis};
use dft_rs::dispersion::Dispersion;
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::hessian::{hessian_with_fallback, numerical_hessian, HessianMethod};
use dft_rs::math::Vec3;
use dft_rs::scf::{run_scf, EriMode, Reference, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec, ResolvedFunctional};

fn water() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.05, 0.2217),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.03, 1.4309, -0.8867),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, -1.4309, -0.8867),
        },
    ])
}

fn opts() -> ScfOptions {
    ScfOptions {
        e_tol: 1e-12,
        p_tol: 1e-10,
        max_iter: 200,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    }
}

fn compare(molecule: &Molecule, reference: Reference, functional: Option<&str>, tag: &str) {
    compare_with(molecule, reference, functional, tag, |_| {});
}

fn compare_with(
    molecule: &Molecule,
    reference: Reference,
    functional: Option<&str>,
    tag: &str,
    tweak: impl Fn(&mut ScfOptions),
) {
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(molecule, &set).unwrap();
    let func = match functional {
        None => ResolvedFunctional::hartree_fock(),
        Some(name) => resolve(&FunctionalSpec::Combined(name.into())).unwrap(),
    };
    let mut o = opts();
    o.reference = reference;
    tweak(&mut o);
    let grid_opts = GridOptions::default();
    let grid = if func.parts.is_empty() {
        None
    } else {
        Some(MolecularGrid::build(molecule, &grid_opts).unwrap())
    };
    let scf = run_scf(molecule, &mb, &o, &func, grid.as_ref()).unwrap();
    let full =
        dft_rs::hessian::analytic_hessian_full(molecule, &mb, &o, &func, &scf, grid.as_ref())
            .unwrap();
    let h_an = full.hessian;
    let num = numerical_hessian(
        molecule,
        &set,
        &o,
        &func,
        functional.unwrap_or("hf"),
        if func.parts.is_empty() {
            None
        } else {
            Some(&grid_opts)
        },
        &Dispersion::None,
        1e-3,
    )
    .unwrap();
    let h_num = num.hessian;
    let dim = 3 * molecule.len();
    let mut max_diff = 0.0f64;
    for r in 0..dim {
        for c in 0..dim {
            max_diff = max_diff.max((h_an[(r, c)] - h_num[(r, c)]).abs());
        }
    }
    assert!(
        max_diff < 2e-6,
        "{tag}: max |analytic - numerical| = {max_diff:.3e}"
    );
    // Translational sum rules: rows of the analytic Hessian must sum to
    // zero over atoms for each fixed (b, y).
    for x in 0..3 {
        for col in 0..dim {
            let s: f64 = (0..molecule.len()).map(|a| h_an[(3 * a + x, col)]).sum();
            assert!(s.abs() < 1e-6, "{tag}: translation sum {s:.3e}");
        }
    }
    // Hessian symmetry: mixed second derivatives of the energy commute,
    // H = H^T exactly (up to CPKS solver tolerance). NOTE: the analytic
    // assembly in `hessian::analytic` force-symmetrizes its result before
    // returning (averaging each off-diagonal pair), so on unmodified code
    // this residual is exactly 0.0 by construction rather than an emergent
    // property of the CPKS solve — this is a permanent regression gate on
    // that guarantee (and on the response-contraction code that feeds it)
    // staying in place, not a live probe of the interchange theorem itself.
    let sym_diff = h_an.max_abs_diff(&h_an.transpose());
    assert!(sym_diff < 1e-10, "{tag}: max |H - H^T| = {sym_diff:.3e}");
    // Analytic APT vs the numerical dipole derivatives.
    let mut apt_diff = 0.0f64;
    for x in 0..3 {
        for col in 0..dim {
            apt_diff = apt_diff
                .max((full.dipole_derivatives[(x, col)] - num.dipole_derivatives[(x, col)]).abs());
        }
    }
    assert!(apt_diff < 5e-6, "{tag}: max APT diff {apt_diff:.3e}");
    // APT translational sum rule: Σ_A ∂μ_i/∂R_{A,j} = Q·δ_ij. Displacing
    // every atom (and its atom-centred basis) by the same infinitesimal
    // vector is a rigid translation of the whole system, under which μ
    // shifts by Q·t (Q = net molecular charge); summing the per-atom
    // partials of μ over A is exactly that directional derivative. Every
    // molecule gated in this file is neutral, so this must be the zero
    // matrix; the Q·I case (net-charged system) is checked separately in
    // `apt_translational_sum_rule_holds_for_a_cation` below.
    let charge = molecule.charge as f64;
    let mut apt_sum_diff = 0.0f64;
    for i in 0..3 {
        for j in 0..3 {
            let s: f64 = (0..molecule.len())
                .map(|a| full.dipole_derivatives[(i, 3 * a + j)])
                .sum();
            let want = if i == j { charge } else { 0.0 };
            apt_sum_diff = apt_sum_diff.max((s - want).abs());
        }
    }
    assert!(
        apt_sum_diff < 1e-8,
        "{tag}: APT translational sum rule max|Σ_A ∂μ/∂R - Qδ| = {apt_sum_diff:.3e}"
    );
}

#[test]
fn hf_analytic_hessian_matches_numerical_rhf() {
    compare(&water(), Reference::Auto, None, "RHF");
}

/// A uniform field contributes both the fixed-origin dipole-integral second
/// derivative to the skeleton and its first derivative to every CPKS right
/// hand side.  This end-to-end comparison also includes the linear nuclear
/// field energy through the numerical reference.
#[test]
fn hf_analytic_hessian_in_uniform_field_matches_numerical() {
    compare_with(
        &water(),
        Reference::Auto,
        None,
        "RHF + uniform field",
        |options| options.external_field = Some(Vec3::new(0.0017, -0.0009, 0.0023)),
    );
}

#[test]
fn hf_analytic_hessian_matches_numerical_uhf() {
    compare(&water(), Reference::Uhf, None, "UHF");
}

#[test]
fn svwn_analytic_hessian_matches_numerical() {
    compare(&water(), Reference::Auto, Some("svwn5"), "SVWN5");
}

#[test]
fn blyp_analytic_hessian_matches_numerical() {
    compare(&water(), Reference::Auto, Some("blyp"), "BLYP");
}

#[test]
fn b3lyp_analytic_hessian_matches_numerical() {
    compare(&water(), Reference::Auto, Some("b3lyp"), "B3LYP");
}

/// Range-separated hybrid GGA: exercises the K_lr derivative Fock,
/// LR skeleton second derivatives, and the RSH CPKS coupling.
#[test]
fn wb97xd_analytic_hessian_matches_numerical() {
    compare(&water(), Reference::Auto, Some("wb97x-d"), "wB97X-D");
}

/// Pure meta-GGA: exercises the tau-channel XC Hessian terms and the
/// meta response kernel in the CPKS.
#[test]
fn m06l_analytic_hessian_matches_numerical() {
    compare(&water(), Reference::Auto, Some("m06l"), "M06-L");
}

/// Hybrid meta-GGA: tau terms + exact-exchange machinery together.
#[test]
fn m062x_analytic_hessian_matches_numerical() {
    compare(&water(), Reference::Auto, Some("m06-2x"), "M06-2X");
}

/// SC-VV10 RSH GGA (ωB97X-V): the fully analytic -V Hessian — VV10
/// skeleton (Gram + kernel R² motion), D_a V_nlc in the CPKS RHS, and the
/// exact nonlocal f_nlc kernel in the response operator — against the
/// numerical Hessian of the exact analytic gradients.
///
/// The NLC quadrature is deliberately coarse (level 1): analytic and
/// numerical sides share the same grid, so derivative consistency — what
/// this gate tests — is grid independent, while the O(m²) VV10 kernel
/// cost (paid 18× by the numerical side) drops by orders of magnitude.
/// Absolute VV10 accuracy is gated separately against PySCF in p9.
#[test]
fn wb97xv_sc_vv10_analytic_hessian_matches_numerical() {
    compare_with(
        &water(),
        Reference::Auto,
        Some("wb97x-v"),
        "ωB97X-V+SC-VV10",
        |o| {
            o.vv10 = Some(dft_rs::nlc::Vv10Params::WB97XV);
            o.vv10_grid_level = 1;
        },
    );
}

/// SC-VV10 RSH meta-GGA (ωB97M-V): the tau-channel Hessian terms and the
/// meta response kernel combined with the VV10 machinery.
#[test]
fn wb97mv_sc_vv10_analytic_hessian_matches_numerical() {
    compare_with(
        &water(),
        Reference::Auto,
        Some("wb97m-v"),
        "ωB97M-V+SC-VV10",
        |o| {
            o.vv10 = Some(dft_rs::nlc::Vv10Params::WB97XV);
            o.vv10_grid_level = 1;
        },
    );
}

/// APT translational sum rule for a NET-CHARGED system. Every case above is
/// neutral (Σ_A ∂μ_i/∂R_{A,j} = 0, gated inline in `compare_with` at zero
/// extra cost since the APT is already computed there). A cation is the
/// asymmetric case the same identity predicts: Σ must equal +Q·δ_ij, not
/// the zero matrix — the discriminator a bug that always returns "zero" (or
/// that only happens to cancel for neutral systems) would fail. This test
/// skips the expensive numerical-Hessian FD cross-check (already exercised
/// above) and only runs the analytic CPKS/APT pass on water/STO-3G with one
/// electron removed, so it stays cheap despite being open-shell.
#[test]
fn apt_translational_sum_rule_holds_for_a_cation() {
    let mol = water().with_charge(1).with_multiplicity(2);
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let func = ResolvedFunctional::hartree_fock();
    let mut o = opts();
    o.reference = Reference::Auto; // n_alpha != n_beta here, so this is UHF.
    o.level_shift = 0.2;
    let scf = run_scf(&mol, &mb, &o, &func, None).unwrap();
    let full = dft_rs::hessian::analytic_hessian_full(&mol, &mb, &o, &func, &scf, None).unwrap();

    let charge = mol.charge as f64;
    let mut apt_sum_diff = 0.0f64;
    for i in 0..3 {
        for j in 0..3 {
            let s: f64 = (0..mol.len())
                .map(|a| full.dipole_derivatives[(i, 3 * a + j)])
                .sum();
            let want = if i == j { charge } else { 0.0 };
            apt_sum_diff = apt_sum_diff.max((s - want).abs());
        }
    }
    assert!(
        apt_sum_diff < 1e-7,
        "cation APT translational sum rule max|Σ_A ∂μ/∂R - Qδ| = {apt_sum_diff:.3e}"
    );
}

/// End-to-end wiring gate for empirical dispersion in the analytic Hessian
/// path (`--disp d3bj frequencies`): the CLI sums `analytic_hessian_full`
/// (purely electronic) with `dispersion_hessian` (the dual-seeded empirical
/// correction from `src/dispersion/mod.rs`), exactly reproduced here,
/// against the numerical Hessian of the analytic-gradient-plus-dispersion-
/// gradient sum (`numerical_hessian` with a non-`None` `Dispersion`, i.e.
/// the path the CLI used exclusively before this Hessian existed). Water/HF
/// is the smallest system already used throughout this file and converges
/// reliably; "hf" is tabulated for D3(BJ).
#[test]
fn d3bj_dispersion_plus_analytic_hessian_matches_numerical() {
    let molecule = water();
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&molecule, &set).unwrap();
    let func = ResolvedFunctional::hartree_fock();
    let o = opts();
    let scf = run_scf(&molecule, &mb, &o, &func, None).unwrap();
    let (best, method) = hessian_with_fallback(
        &molecule,
        &set,
        &mb,
        &o,
        &func,
        "hf",
        None,
        None,
        &scf,
        &Dispersion::D3Bj,
        1e-3,
    )
    .unwrap();
    assert_eq!(method, HessianMethod::Analytic);
    let h_an = best.hessian;
    let num = numerical_hessian(
        &molecule,
        &set,
        &o,
        &func,
        "hf",
        None,
        &Dispersion::D3Bj,
        1e-3,
    )
    .unwrap();
    let h_num = num.hessian;
    let dim = 3 * molecule.len();
    let mut max_diff = 0.0f64;
    for r in 0..dim {
        for c in 0..dim {
            max_diff = max_diff.max((h_an[(r, c)] - h_num[(r, c)]).abs());
        }
    }
    assert!(
        max_diff < 2e-6,
        "D3(BJ) analytic+dispersion vs numerical: max diff = {max_diff:.3e}"
    );
}
