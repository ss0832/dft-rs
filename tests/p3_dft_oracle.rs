// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Phase 3 gate: LDA (RKS/UKS) energies against the frozen PySCF oracle,
//! grid diagnostics, and functional-resolution CLI behavior.

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::scf::{run_scf, EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

fn water() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.0, 0.2217),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 1.4309, -0.8867),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, -1.4309, -0.8867),
        },
    ])
}

fn tight() -> ScfOptions {
    ScfOptions {
        e_tol: 1e-10,
        p_tol: 1e-8,
        max_iter: 200,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    }
}

struct DftCase {
    name: String,
    basis: String,
    /// Label used for the tolerance rules and failure messages: the
    /// combined name, or `x-c` for a split specification.
    functional: String,
    spec: FunctionalSpec,
    molecule: Molecule,
    e_tot: f64,
    density_fit: Option<String>,
}

fn load_dft_cases() -> Vec<DftCase> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data");
    let mut cases = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap().flatten() {
        let fname = entry.file_name().to_string_lossy().to_string();
        if !fname.ends_with("_dft.csv") {
            continue;
        }
        let text = std::fs::read_to_string(entry.path()).unwrap();
        let mut basis = String::new();
        let mut functional = String::new();
        // A CSV carries either `functional,<name>` (combined) or the pair
        // `x,<name>` / `c,<name>` — the split tables have entries no
        // combined functional reaches at their registry parameters.
        let mut x_name: Option<String> = None;
        let mut c_name: Option<String> = None;
        let mut charge = 0i32;
        let mut mult = 1usize;
        let mut atoms = Vec::new();
        let mut e_tot = f64::NAN;
        let mut density_fit = None;
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let p: Vec<&str> = line.split(',').collect();
            match p[0] {
                "basis" => basis = p[1].into(),
                "functional" => functional = p[1].into(),
                "x" => x_name = Some(p[1].into()),
                "c" => c_name = Some(p[1].into()),
                "density_fit" => density_fit = Some(p[1].to_string()),
                "charge" => charge = p[1].parse().unwrap(),
                "mult" => mult = p[1].parse().unwrap(),
                "atom" => atoms.push(Atom {
                    z: p[1].parse().unwrap(),
                    position: Vec3::new(
                        p[2].parse().unwrap(),
                        p[3].parse().unwrap(),
                        p[4].parse().unwrap(),
                    ),
                }),
                "e_tot" => e_tot = p[1].parse().unwrap(),
                other => panic!("unknown key {other}"),
            }
        }
        let spec = match (x_name, c_name) {
            (Some(x), Some(c)) => {
                assert!(
                    functional.is_empty(),
                    "{fname}: `functional` and `x`/`c` are mutually exclusive"
                );
                functional = format!("{x}-{c}");
                FunctionalSpec::Split { x, c }
            }
            (None, None) => {
                assert!(!functional.is_empty(), "{fname}: no functional key");
                FunctionalSpec::Combined(functional.clone())
            }
            _ => panic!("{fname}: a split specification needs both `x` and `c`"),
        };
        cases.push(DftCase {
            name: fname.trim_end_matches("_dft.csv").into(),
            basis,
            functional,
            spec,
            molecule: Molecule::new(atoms)
                .with_charge(charge)
                .with_multiplicity(mult),
            e_tot,
            density_fit,
        });
    }
    assert!(
        cases.len() >= 9,
        "expected 9+ DFT cases, found {}",
        cases.len()
    );
    cases
}

/// LDA energies vs the frozen PySCF oracle. Both sides are run at
/// (near-)grid-converged resolution, so the comparison bounds functional
/// implementation + grid quality together.
#[test]
fn lda_energies_match_pyscf_oracle() {
    let mut failures = Vec::new();
    for case in load_dft_cases() {
        let set = data::load(&case.basis).unwrap();
        let mb = MolBasis::build(&case.molecule, &set).unwrap();
        let func = resolve(&case.spec).unwrap();
        // The ωB97 SR-LSDA attenuation and the M06 w-series have sharper
        // spatial structure — grid-converge harder for those cases.
        // P86's Phi ~ |grad rho|/rho^{7/6} exponential is likewise sharp,
        // as is the wPBEh hole (s-rescaling kink at s = 1, EG branch switch).
        let sharp = case.functional.starts_with("wb97")
            || case.functional.starts_with("m06")
            || case.functional.starts_with("revm06")
            || case.functional == "b97mv"
            || case.functional == "bp86"
            || case.functional.starts_with("hse")
            || case.functional.starts_with("lc-wpbe")
            || case.functional == "r2scan";
        // `lc-wpbe-hjs` is deliberately absent: it uses the HJS hole, not
        // wPBEh, so it is unaffected and still gated at full tolerance.
        // That it still passes is itself evidence the deviation is confined
        // to `wpbeh_scale_s`.
        const DELIBERATELY_DEVIATES_FROM_LIBXC: &[&str] = &["hse03", "hse06", "lc-wpbe"];
        let level = if sharp { 9 } else { 8 };
        // SCAN's alpha interpolation is grid-sensitive (with the former
        // product angular grid our CH3 energy moved by 4e-4 between levels 5
        // and 9 and needed 200 radial points); the standard level-9 grid is
        // the oracle's own grid, so SCAN runs on it directly. So does every
        // functional built on SCAN: SCAN0 on level 8 missed its level-9
        // oracle by -2.27e-6 (water) and -4.51e-6 (CH3).
        let scan = case.functional.starts_with("scan");
        let gopts = GridOptions {
            level: if scan { 9 } else { level },
            ..GridOptions::default()
        };
        let grid = MolecularGrid::build(&case.molecule, &gopts).unwrap();
        let mut opts = tight();
        if case.molecule.multiplicity > 1 {
            opts.level_shift = 0.2;
        }
        opts.density_fit = case.density_fit.clone();
        match run_scf(&case.molecule, &mb, &opts, &func, Some(&grid)) {
            Ok(scf) => {
                let ne_expected = case.molecule.n_electrons().unwrap() as f64;
                let ne = scf.grid_electron_count.unwrap();
                if (ne - ne_expected).abs() > 1e-7 {
                    failures.push(format!("{}: ∫ρ = {ne} vs {ne_expected}", case.name));
                }
                // ωB97 family: pointwise agreement is 1e-8, but the SCF
                // totals carry a ~1e-6 residual (published-digit coefficient
                // truncation scale) — gated separately. The wPBEh hole's
                // C¹-discontinuous s-rescaling (kink at s = 1) converges
                // slowly in quadrature: level-8/9 totals straddle the oracle
                // at ±2e-6 while the pointwise pins hold at 1e-9.
                // r²SCAN's α-interpolation branch structure is likewise
                // quadrature-sensitive (level-8/9 totals straddle the
                // oracle at ±2e-6; pointwise pins hold at 1e-9).
                let e_tol = if case.functional.starts_with("wb97")
                    || case.functional.starts_with("hse")
                    || case.functional == "lc-wpbe"
                    || case.functional == "r2scan"
                {
                    2.5e-6
                } else {
                    5e-7
                };
                let e_tol = if case.functional.starts_with("m06")
                    || case.functional.starts_with("revm06")
                    || case.functional == "b97mv"
                {
                    1e-6
                } else {
                    e_tol
                };
                // SCAN: the oracle itself moves by 9e-6 (water) and 1.5e-5
                // (CH3) between PySCF grids of level 7, 9, (300, 1454) and
                // (500, 2030); pointwise values agree with libxc to 1e-13.
                let e_tol = if case.functional == "scan" {
                    2e-5
                } else {
                    e_tol
                };
                // Functionals built on the wPBEh hole no longer reproduce
                // libxc's total, deliberately. libxc's s-rescaling
                // short-circuits (`s < 1 -> s`, `s > 15 -> smax`) are
                // discontinuous by -5.141e-4 and +1.616e-3, and this project
                // treats a discontinuity as a defect even where the reference
                // implementation shares it, so `wpbeh_scale_s` ships the
                // smooth offset-corrected form instead. Measured cost:
                // ~1.0e-4 to 1.8e-4 Ha, i.e. about 0.1 kcal/mol, systematic.
                //
                // The rows stay, and so does a check: the difference must
                // remain SMALL and bounded. What is no longer asserted is
                // agreement to the 2.5e-6 gate. The hole model itself is
                // still pinned pointwise against these same libxc references
                // in `src/xc/sr_x.rs`, evaluated through libxc's own
                // rescaling, so only the choice of rescaling is exempt --
                // and that has its own reference-free gate.
                let deviates = DELIBERATELY_DEVIATES_FROM_LIBXC.contains(&case.functional.as_str());
                let effective_tol = if deviates { 5e-4 } else { e_tol };
                if (scf.energy - case.e_tot).abs() > effective_tol {
                    failures.push(format!(
                        "{}: E {} vs oracle {} (Δ = {:.3e}{})",
                        case.name,
                        scf.energy,
                        case.e_tot,
                        scf.energy - case.e_tot,
                        if deviates {
                            ", and this functional's deviation from libxc is supposed to stay \
                             within 5e-4 -- a larger one is a real regression, not the known \
                             s-rescaling difference"
                        } else {
                            ""
                        }
                    ));
                }
            }
            Err(e) => failures.push(format!("{}: SCF failed: {e}", case.name)),
        }
    }
    assert!(
        failures.is_empty(),
        "DFT oracle failures:\n{}",
        failures.join("\n")
    );
}

/// Range-separated hybrid limits (internal gates for LC functionals):
/// LC-BLYP(ω → 0) = BLYP exactly (K_lr → 0, SR-B88 → B88), and a pure
/// long-range HF with ω → ∞ equals full HF.
#[test]
fn rsh_limits() {
    let mol = water();
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let grid = MolecularGrid::build(&mol, &GridOptions::default()).unwrap();
    let opts = tight();

    let lc_w0 = resolve(&FunctionalSpec::Combined("lc-blyp".into()))
        .unwrap()
        .with_omega(1e-7)
        .unwrap();
    let blyp = resolve(&FunctionalSpec::Combined("blyp".into())).unwrap();
    let e_lc = run_scf(&mol, &mb, &opts, &lc_w0, Some(&grid))
        .unwrap()
        .energy;
    let e_blyp = run_scf(&mol, &mb, &opts, &blyp, Some(&grid))
        .unwrap()
        .energy;
    assert!(
        (e_lc - e_blyp).abs() < 1e-6,
        "LC-BLYP(ω→0) {e_lc} vs BLYP {e_blyp}"
    );

    // Pure LR-HF with huge ω ≡ full HF (erf → 1).
    use dft_rs::xc::{HybridParams, ResolvedFunctional};
    let lr_only = ResolvedFunctional::custom(
        "LR-HF test",
        Vec::new(),
        HybridParams {
            cx: 0.0,
            clr: 1.0,
            omega: 1e5,
        },
        dft_rs::xc::Family::Lda,
    );
    let hf = ResolvedFunctional::hartree_fock();
    let e_lr = run_scf(&mol, &mb, &opts, &lr_only, None).unwrap().energy;
    let e_hf = run_scf(&mol, &mb, &opts, &hf, None).unwrap().energy;
    assert!((e_lr - e_hf).abs() < 1e-6, "LR-HF(ω→∞) {e_lr} vs HF {e_hf}");
}

/// Convergence probe: SVWN5 water must converge with the default
/// acceleration stack; energy must be in the known SVWN5/STO-3G range.
#[test]
fn svwn5_water_converges() {
    let mol = water();
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let func = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    let grid = MolecularGrid::build(&mol, &GridOptions::default()).unwrap();
    let scf = run_scf(&mol, &mb, &tight(), &func, Some(&grid)).unwrap();
    assert!(scf.converged);
    // ∫ρ must reproduce the electron count at the default grid level.
    let ne = scf.grid_electron_count.unwrap();
    assert!((ne - 10.0).abs() < 5e-5, "∫ρ = {ne}");
    assert!(
        (-74.9..=-74.5).contains(&scf.energy),
        "unexpected SVWN5 energy {}",
        scf.energy
    );

    // Direct-mode J/K at the CONVERGED density must match in-core at the
    // production screening threshold (diagnoses screening-induced SCF
    // stalls).
    use dft_rs::integrals::CoulombKernel;
    use dft_rs::scf::JkEngine;
    let p_tot = scf.p_total();
    let incore = JkEngine::new(
        &mb,
        EriMode::InCore { limit_mb: 512 },
        CoulombKernel::Coulomb,
    )
    .unwrap();
    let direct = JkEngine::new(
        &mb,
        EriMode::Direct {
            screen_threshold: 1e-12,
        },
        CoulombKernel::Coulomb,
    )
    .unwrap();
    let (j1, k1) = incore
        .build(&mb, &p_tot, [&scf.density[0], &scf.density[1]], true)
        .unwrap();
    let (j2, k2) = direct
        .build(&mb, &p_tot, [&scf.density[0], &scf.density[1]], true)
        .unwrap();
    assert!(
        j1.max_abs_diff(&j2) < 1e-9,
        "direct J deviates by {}",
        j1.max_abs_diff(&j2)
    );
    assert!(k1[0].max_abs_diff(&k2[0]) < 1e-9, "direct K deviates");
}
