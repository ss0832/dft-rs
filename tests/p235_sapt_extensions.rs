// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Acceptance of the SAPT extensions: full-response (RPA) dispersion and the
//! Casimir-Polder integral, asymptotically corrected Kohn-Sham monomers,
//! charge transfer from the dimer-centred basis and the atom-pair partition.
//!
//! Oracles: the Casimir-Polder integral of the coupled susceptibilities
//! against the independent sum over RPA states (matrix inversion along the
//! imaginary axis versus diagonalization) and of the uncoupled
//! susceptibilities against the closed-form uncoupled dispersion, per
//! functional class; the dimer-basis induction behind the charge transfer
//! against a four-step fit of central differences of the ghost-basis
//! monomer energy in the scaled partner potential; the exponential decay
//! of the charge transfer against the algebraic decay of the induction; the
//! asymptotic correction against the ionization potential of a separate
//! cation calculation (Koopmans alignment of the highest occupied orbital)
//! and its exact linearity in the shift; the partition sums and the
//! transposition symmetry under exchange of the monomers; the public driver
//! route against the library call.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::basis::{data, MolBasis};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::integrals::one_electron::nuclear_matrix;
use dft_rs::sapt::{sapt_interaction_with, SaptConfig, SaptResult};
use dft_rs::scf::{run_scf, EriMode, JkEngine, ScfOptions};
use dft_rs::{resolve, Atom, FunctionalSpec, Matrix, Molecule, Vec3};

fn water_a() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 1.43, 1.108),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, -1.43, 1.108),
        },
    ])
}

fn water_b(shift: f64) -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.0, 5.4 + shift),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 1.43, 6.5 + shift),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, -1.43, 6.5 + shift),
        },
    ])
}

fn opts() -> ScfOptions {
    ScfOptions {
        e_tol: 1e-12,
        p_tol: 1e-10,
        max_iter: 300,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    }
}

fn analysis(
    a: &Molecule,
    b: &Molecule,
    basis: &str,
    functional: &str,
    config: &SaptConfig,
) -> SaptResult {
    let set = data::load(basis).unwrap();
    let functional = resolve(&FunctionalSpec::Combined(functional.into())).unwrap();
    sapt_interaction_with(a, b, &set, &functional, &opts(), None, config).unwrap()
}

/// The coupled Casimir-Polder integral equals the sum over RPA states and
/// the uncoupled one the closed-form uncoupled dispersion, for Hartree-Fock,
/// local, gradient-corrected and hybrid monomers.
#[test]
fn casimir_polder_matches_sum_over_states_per_functional_class() {
    let config = SaptConfig {
        rpa_dispersion: true,
        casimir_polder_points: 24,
        ..SaptConfig::default()
    };
    for functional in ["hf", "svwn5", "pbe", "pbe0"] {
        let r = analysis(&water_a(), &water_b(0.0), "6-31g", functional, &config);
        let rpa = r.rpa.as_ref().unwrap();
        let coupled = (rpa.e_disp_casimir_polder - rpa.e_disp).abs();
        let uncoupled = (rpa.e_disp_casimir_polder_uncoupled - r.e_disp_unc).abs();
        println!(
            "{functional}: RPA sum over states {:.12e}, Casimir-Polder {:.12e} (|diff| {coupled:.3e}, \
             quadrature change {:.3e}, {} points); uncoupled closed form {:.12e}, integral {:.12e} \
             (|diff| {uncoupled:.3e}); TDA dispersion {:.12e}; RPA exchange-dispersion {:.12e}",
            rpa.e_disp,
            rpa.e_disp_casimir_polder,
            rpa.casimir_polder_change,
            rpa.points,
            r.e_disp_unc,
            rpa.e_disp_casimir_polder_uncoupled,
            r.e_disp_coupled,
            rpa.e_exch_disp
        );
        assert!(rpa.e_disp < 0.0 && rpa.e_exch_disp > 0.0);
        assert!(
            coupled < 1e-9 * rpa.e_disp.abs().max(1e-6),
            "{functional}: {coupled}"
        );
        assert!(
            uncoupled < 1e-9 * r.e_disp_unc.abs().max(1e-6),
            "{functional}: {uncoupled}"
        );
        assert!(rpa.casimir_polder_change < 1e-8 * rpa.e_disp.abs());
        // The extension leaves every original component unchanged.
        let plain = analysis(
            &water_a(),
            &water_b(0.0),
            "6-31g",
            functional,
            &SaptConfig::default(),
        );
        for (x, y) in [
            (r.e_elst, plain.e_elst),
            (r.e_exch, plain.e_exch),
            (r.e_ind_a, plain.e_ind_a),
            (r.e_disp_coupled, plain.e_disp_coupled),
            (r.e_exch_disp, plain.e_exch_disp),
        ] {
            assert!((x - y).abs() < 1e-12);
        }
    }
}

/// Dimer-basis potential of B (nuclei and its ghost-basis density) and the
/// ghost-basis monomer A.
fn dimer_basis_problem() -> (Molecule, MolBasis, Matrix) {
    let set = data::load("6-31g").unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let (a, b) = (water_a(), water_b(0.0));
    let na = a.len();
    let mut atoms = a.atoms.clone();
    atoms.extend(b.atoms.iter().cloned());
    let dimer = Molecule::new(atoms);
    let mb = MolBasis::build(&dimer, &set).unwrap();
    let ghost = |keep_a: bool| {
        let mut atoms = dimer.atoms.clone();
        for (k, atom) in atoms.iter_mut().enumerate() {
            if (k < na) != keep_a {
                atom.z = 0;
            }
        }
        Molecule::new(atoms)
    };
    let scf_b = run_scf(&ghost(false), &mb, &opts(), &functional, None).unwrap();
    let mut charges: Vec<f64> = dimer.atoms.iter().map(|atom| atom.z as f64).collect();
    for q in charges.iter_mut().take(na) {
        *q = 0.0;
    }
    let mut omega = nuclear_matrix(&mb, &dimer, Some(&charges));
    let jk = JkEngine::from_options(&dimer, &mb, &opts()).unwrap();
    let p = scf_b.p_total();
    let (j, _) = jk
        .build(&mb, &p, [&scf_b.density[0], &scf_b.density[1]], false)
        .unwrap();
    for (x, y) in omega.as_mut_slice().iter_mut().zip(j.as_slice()) {
        *x += y;
    }
    (ghost(true), mb, omega)
}

/// The dimer-basis induction of A behind the charge transfer against central
/// differences of the ghost-basis energy of A in the scaled potential of B:
/// `(E(l) + E(-l) - 2 E(0)) / (2 l^2) = E_ind + O(l^2)`.
#[test]
fn charge_transfer_induction_matches_scaled_potential_differences() {
    let config = SaptConfig {
        charge_transfer: true,
        ..SaptConfig::default()
    };
    let r = analysis(&water_a(), &water_b(0.0), "6-31g", "hf", &config);
    let ct = r.charge_transfer.as_ref().unwrap();
    println!(
        "dimer-basis induction A {:.12e}, B {:.12e}; monomer-basis {:.12e}, {:.12e}; charge \
         transfer {:.6e}",
        ct.e_ind_dimer_basis[0], ct.e_ind_dimer_basis[1], r.e_ind_a, r.e_ind_b, ct.e_ct
    );
    assert!(ct.e_ct < 0.0);
    assert!(ct.e_ind_dimer_basis[0] < r.e_ind_a && ct.e_ind_dimer_basis[1] < r.e_ind_b);
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let (ghost_a, mb, omega) = dimer_basis_problem();
    let energy = |l: f64| -> f64 {
        let mut o = opts();
        o.e_tol = 1e-13;
        o.p_tol = 1e-11;
        if l != 0.0 {
            let mut m = omega.clone();
            for v in m.as_mut_slice() {
                *v *= l;
            }
            o.extra_hcore = Some(m);
        }
        run_scf(&ghost_a, &mb, &o, &functional, None)
            .unwrap()
            .energy
    };
    let e0 = energy(0.0);
    let steps = [0.08, 0.06, 0.04, 0.03];
    let errors: Vec<f64> = steps
        .iter()
        .map(|&l| {
            let fd = (energy(l) + energy(-l) - 2.0 * e0) / (2.0 * l * l);
            (fd - ct.e_ind_dimer_basis[0]).abs()
        })
        .collect();
    let fit = convergence::error_power_fit(&steps, &errors);
    println!(
        "dimer-basis induction of A: steps {steps:?} errors {errors:?} p={:.6} R2={:.8} C={:.3e}",
        fit.exponent, fit.r_squared, fit.prefactor
    );
    assert!((fit.exponent - 2.0).abs() < 0.2 && fit.r_squared > 0.999);
}

/// The charge transfer decays exponentially with the separation, the
/// induction algebraically.
#[test]
fn charge_transfer_decays_faster_than_induction() {
    let config = SaptConfig {
        charge_transfer: true,
        ..SaptConfig::default()
    };
    let at = |shift: f64| analysis(&water_a(), &water_b(shift), "6-31g", "hf", &config);
    let values: Vec<(f64, f64, f64)> = [0.0, 2.0, 4.0, 6.0]
        .iter()
        .map(|&s| {
            let r = at(s);
            (s, r.charge_transfer.unwrap().e_ct, r.e_ind_a + r.e_ind_b)
        })
        .collect();
    for w in values.windows(2) {
        let ct_ratio = (w[0].1 / w[1].1).ln();
        let ind_ratio = (w[0].2 / w[1].2).ln();
        println!(
            "shift {:.1} -> {:.1} Bohr: charge transfer {:.6e} -> {:.6e} (log ratio {ct_ratio:.4}), \
             induction {:.6e} -> {:.6e} (log ratio {ind_ratio:.4})",
            w[0].0, w[1].0, w[0].1, w[1].1, w[0].2, w[1].2
        );
        assert!(w[0].1 < 0.0 && w[1].1 < 0.0);
        assert!(ct_ratio > ind_ratio);
    }
    assert!(values[3].1.abs() < 1e-3 * values[0].1.abs());
}

/// The corrected highest occupied orbital of a gradient-corrected monomer
/// moves to minus the ionization potential of a separate cation
/// calculation; the correction is exactly linear in the shift.
#[test]
fn asymptotic_correction_aligns_the_highest_occupied_orbital() {
    let set = data::load("6-31g").unwrap();
    let functional = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let neutral = water_a();
    let cation = water_a().with_charge(1).with_multiplicity(2);
    let mb = MolBasis::build(&neutral, &set).unwrap();
    let grid = MolecularGrid::build(&neutral, &GridOptions::default()).unwrap();
    let scf = run_scf(&neutral, &mb, &opts(), &functional, Some(&grid)).unwrap();
    let ion = run_scf(&cation, &mb, &opts(), &functional, Some(&grid)).unwrap();
    let ip = ion.energy - scf.energy;
    let homo = scf.mo_energy[0][scf.n_alpha - 1];
    let shift = ip + homo;
    let corrected_opts = ScfOptions {
        asymptotic_correction: Some(shift),
        ..opts()
    };
    let corrected = run_scf(&neutral, &mb, &corrected_opts, &functional, Some(&grid)).unwrap();
    let corrected_homo = corrected.mo_energy[0][corrected.n_alpha - 1];
    println!(
        "PBE water: IP {ip:.8}, HOMO {homo:.8} (misalignment {:.3e}), shift {shift:.8}, corrected \
         HOMO {corrected_homo:.8} (misalignment {:.3e}); energy {:.10} -> {:.10}",
        (homo + ip).abs(),
        (corrected_homo + ip).abs(),
        scf.energy,
        corrected.energy
    );
    assert!(shift > 0.0);
    assert!((corrected_homo + ip).abs() < 0.25 * (homo + ip).abs());
    // The functional energy of the corrected density lies above the
    // variational minimum.
    assert!(corrected.energy > scf.energy);
    // Linearity in the shift.
    let p = [&scf.density[0], &scf.density[1]];
    let at = |s: f64| dft_rs::ks::grac_correction(&functional, &mb, &grid, p, s).unwrap();
    let (g0, g1, g2) = (at(0.0), at(0.1), at(0.3));
    let mut worst = 0.0_f64;
    for spin in 0..2 {
        for k in 0..g0[spin].as_slice().len() {
            let (x0, x1, x2) = (
                g0[spin].as_slice()[k],
                g1[spin].as_slice()[k],
                g2[spin].as_slice()[k],
            );
            worst = worst.max(((x2 - x0) - 3.0 * (x1 - x0)).abs());
        }
    }
    println!("linearity residual of the correction in the shift: {worst:.3e}");
    assert!(worst < 1e-12);
    // Hartree-Fock and meta-GGA monomers are refused.
    for name in ["hf", "tpss"] {
        let f = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        assert!(run_scf(&neutral, &mb, &corrected_opts, &f, Some(&grid)).is_err());
    }
}

/// Atom-pair partitions sum to their totals and transpose under exchange
/// of the monomers; asymptotically corrected monomers enter the same terms.
#[test]
fn partition_sums_and_monomer_exchange_symmetry() {
    let config = SaptConfig {
        partition: true,
        ..SaptConfig::default()
    };
    for functional in ["hf", "pbe0"] {
        let r = analysis(&water_a(), &water_b(0.0), "6-31g", functional, &config);
        let s = analysis(&water_b(0.0), &water_a(), "6-31g", functional, &config);
        let p = r.partition.as_ref().unwrap();
        let q = s.partition.as_ref().unwrap();
        let mut worst = 0.0_f64;
        for (x, y) in [
            (&p.electrostatic, &q.electrostatic),
            (&p.induction_a, &q.induction_b),
            (&p.induction_b, &q.induction_a),
        ] {
            let t = y.transpose();
            for (u, v) in x.as_slice().iter().zip(t.as_slice()) {
                worst = worst.max((u - v).abs());
            }
        }
        println!(
            "{functional}: partition residual {:.3e} (swapped {:.3e}); exchange asymmetry {worst:.3e}; \
             electrostatic O(A)-O(B) {:.6e}, O(A)-H(B) {:.6e}",
            p.residual,
            q.residual,
            p.electrostatic[(0, 0)],
            p.electrostatic[(0, 1)]
        );
        assert!(p.residual < 1e-9 && q.residual < 1e-9);
        assert!(worst < 1e-7);
    }
    let shifted = SaptConfig {
        asymptotic_shift: [Some(0.08), Some(0.08)],
        partition: true,
        charge_transfer: true,
        rpa_dispersion: true,
        casimir_polder_points: 16,
    };
    let plain = analysis(
        &water_a(),
        &water_b(0.0),
        "6-31g",
        "pbe",
        &SaptConfig::default(),
    );
    let r = analysis(&water_a(), &water_b(0.0), "6-31g", "pbe", &shifted);
    let rpa = r.rpa.as_ref().unwrap();
    println!(
        "PBE monomers with the asymptotic correction: electrostatic {:.8e} (uncorrected {:.8e}), \
         induction {:.8e} ({:.8e}), TDA dispersion {:.8e} ({:.8e}), RPA {:.8e}, partition residual {:.3e}",
        r.e_elst,
        plain.e_elst,
        r.e_ind_a + r.e_ind_b,
        plain.e_ind_a + plain.e_ind_b,
        r.e_disp_coupled,
        plain.e_disp_coupled,
        rpa.e_disp,
        r.partition.as_ref().unwrap().residual
    );
    assert!(r.partition.as_ref().unwrap().residual < 1e-9);
    assert!((rpa.e_disp_casimir_polder - rpa.e_disp).abs() < 1e-9 * rpa.e_disp.abs());
    // The corrected references are the monomers of the decomposition: their
    // energies equal separate corrected SCF calculations (the first run of
    // this test asserted instead that the correction lowers the dispersion;
    // with this arbitrary shift in the 6-31G basis it raised it by 2 %,
    // which no identity forbids).
    let set = data::load("6-31g").unwrap();
    let pbe = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let corrected_opts = ScfOptions {
        asymptotic_correction: Some(0.08),
        ..opts()
    };
    for (molecule, energy, uncorrected) in [
        (water_a(), r.e_monomer_a, plain.e_monomer_a),
        (water_b(0.0), r.e_monomer_b, plain.e_monomer_b),
    ] {
        let mb = MolBasis::build(&molecule, &set).unwrap();
        let grid = MolecularGrid::build(&molecule, &GridOptions::default()).unwrap();
        let scf = run_scf(&molecule, &mb, &corrected_opts, &pbe, Some(&grid)).unwrap();
        println!(
            "corrected monomer energy {energy:.12} (separate SCF {:.12}, uncorrected {uncorrected:.12})",
            scf.energy
        );
        assert!((energy - scf.energy).abs() < 1e-10);
        assert!(energy > uncorrected);
    }
}

/// The public driver route reports the library values.
#[test]
fn driver_route_reports_the_library_values() {
    let (a, b) = (water_a(), water_b(0.0));
    let atoms: Vec<&Atom> = a.atoms.iter().chain(b.atoms.iter()).collect();
    let system = System::molecule(
        atoms.iter().map(|atom| atom.z).collect(),
        atoms
            .iter()
            .map(|atom| [atom.position.x, atom.position.y, atom.position.z])
            .collect(),
    );
    let controls = Controls::from_value(&Value::Object(vec![
        ("xc".into(), Value::Str("hf".into())),
        ("basis".into(), Value::Str("6-31g".into())),
        ("conv_tol".into(), Value::Float(1e-12)),
        (
            "fragments".into(),
            Value::Array([0, 0, 0, 1, 1, 1].iter().map(|&f| Value::Int(f)).collect()),
        ),
        ("casimir_polder_points".into(), Value::Int(16)),
    ]))
    .unwrap();
    let report = run("sapt-analysis", &system, &controls).unwrap();
    let config = SaptConfig {
        rpa_dispersion: true,
        casimir_polder_points: 16,
        charge_transfer: true,
        partition: true,
        ..SaptConfig::default()
    };
    let r = analysis(&a, &b, "6-31g", "hf", &config);
    let number = |path: &[&str]| -> f64 {
        let mut v = &report;
        for key in path {
            v = v.get(key).unwrap();
        }
        v.as_f64().unwrap()
    };
    let rows = [
        (number(&["electrostatic"]), r.e_elst),
        (number(&["exchange"]), r.e_exch),
        (number(&["interaction_energy"]), r.interaction_total()),
        (
            number(&["interaction_energy_rpa"]),
            r.interaction_total_rpa().unwrap(),
        ),
        (
            number(&["rpa", "casimir_polder"]),
            r.rpa.as_ref().unwrap().e_disp_casimir_polder,
        ),
        (
            number(&["charge_transfer", "energy"]),
            r.charge_transfer.as_ref().unwrap().e_ct,
        ),
        (
            number(&["partition", "residual"]),
            r.partition.as_ref().unwrap().residual,
        ),
    ];
    for (x, y) in rows {
        assert!((x - y).abs() < 1e-8, "{x} vs {y}");
    }
}
