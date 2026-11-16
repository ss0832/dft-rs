// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! SAPT stage-1 gates: A↔B relabeling symmetry, the coupled-induction
//! finite-difference identity (λ-scaled partner potential through
//! `extra_hcore`), sign/decay sanity, and the frozen electrostatics
//! oracle (added alongside tools/oracle/gen_sapt.py).

use dft_rs::basis::data;
use dft_rs::math::Vec3;
use dft_rs::sapt::sapt_interaction;
use dft_rs::scf::{run_scf, EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

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
        e_tol: 1e-11,
        p_tol: 1e-9,
        max_iter: 200,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    }
}

/// SAPT's intermolecular operators currently use all-electron nuclear
/// potentials. An ECP monomer must therefore be rejected before any SCF or
/// component evaluation instead of returning a numerically plausible but
/// physically incomplete decomposition.
#[test]
fn sapt_rejects_ecp_basis_before_returning_incomplete_components() {
    let set = data::load("lanl2dz").unwrap();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let na = Molecule::new(vec![Atom {
        z: 11,
        position: Vec3::new(0.0, 0.0, 0.0),
    }])
    .with_multiplicity(2);
    let h = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::new(0.0, 0.0, 6.0),
    }])
    .with_multiplicity(2);

    let message = match sapt_interaction(&na, &h, &set, &func, &opts(), None) {
        Ok(_) => panic!("SAPT must not return incomplete ECP interaction components"),
        Err(error) => error.to_string(),
    };
    assert!(message.contains("SAPT"), "unexpected error: {message}");
    assert!(message.contains("ECP"), "unexpected error: {message}");
    assert!(
        message.contains("effective nuclear charges") && message.contains("semi-local"),
        "the rejection must explain the missing ECP physics: {message}"
    );
}

/// Every SAPT term must be INDEPENDENT of whether the closed-shell
/// monomers were solved restricted or unrestricted.
///
/// This is the gate on the whole spin-resolved path. Forcing UKS on a
/// singlet gives the same determinant by a different route: the alpha and
/// beta channels are separately enumerated, the transition densities are
/// summed over channels rather than doubled, the exchange-dispersion
/// projector is built per spin, and the sum-over-states carries a spin
/// factor of 1 per side instead of 2. If any of those countings is off,
/// the two columns disagree — and a factor of 4 or 16 in the dispersion is
/// exactly the kind of error that still LOOKS like an energy.
#[test]
fn sapt_terms_are_reference_independent_for_a_closed_shell() {
    let set = data::load("sto-3g").unwrap();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let (a, b) = (water_a(), water_b(0.0));

    let r = sapt_interaction(&a, &b, &set, &func, &opts(), None).unwrap();
    let u_opts = ScfOptions {
        reference: dft_rs::Reference::Uhf,
        ..opts()
    };
    let u = sapt_interaction(&a, &b, &set, &func, &u_opts, None).unwrap();

    for (name, x, y) in [
        ("E(monomer A)", r.e_monomer_a, u.e_monomer_a),
        ("E(monomer B)", r.e_monomer_b, u.e_monomer_b),
        ("elst", r.e_elst, u.e_elst),
        ("exch", r.e_exch, u.e_exch),
        ("ind A<-B", r.e_ind_a, u.e_ind_a),
        ("ind B<-A", r.e_ind_b, u.e_ind_b),
        ("exch-ind", r.e_exch_ind, u.e_exch_ind),
        ("disp uncoupled", r.e_disp_unc, u.e_disp_unc),
        (
            "disp sum-over-states",
            r.e_disp_sos_uncoupled,
            u.e_disp_sos_uncoupled,
        ),
        ("disp coupled", r.e_disp_coupled, u.e_disp_coupled),
        ("exch-disp", r.e_exch_disp, u.e_exch_disp),
        (
            "supermolecular",
            r.e_int_supermolecular,
            u.e_int_supermolecular,
        ),
        (
            "coupled interaction total",
            r.interaction_total(),
            u.interaction_total(),
        ),
        (
            "uncoupled calibration total",
            r.interaction_total_uncoupled_calibration(),
            u.interaction_total_uncoupled_calibration(),
        ),
        ("delta", r.delta, u.delta),
    ] {
        // Every SAPT term is a DIFFERENCE of total energies of order
        // 100 Hartree, so an absolute floor near machine precision on that
        // scale belongs in the tolerance: the two reference paths sum the
        // same quantities in a different order and cannot agree more
        // closely than that. The relative part is what actually catches a
        // bug — a mis-counted spin factor is a factor of 2, 4 or 16, not
        // parts in 10^8.
        assert!(
            (x - y).abs() <= 1e-8 * x.abs() + 1e-12,
            "{name}: restricted {x:.12e} vs unrestricted {y:.12e}"
        );
    }
}

/// A monomer with a partially filled degenerate shell has no meaningful
/// linear response, and SAPT must say so rather than report the noise as
/// an energy. A triplet oxygen ATOM is the canonical case: its beta 2p
/// shell is one electron in three degenerate orbitals, and the
/// symmetry-broken SCF solution is a saddle point of the orbital Hessian.
///
/// Note what does NOT catch this: every orbital-energy gap looks healthy
/// (0.21 Hartree in STO-3G), because the near-zero curvature lives in the
/// COUPLED Hessian, not in the denominators. Left unguarded it reported an
/// exchange-induction of +133 Hartree as if it were an energy.
///
/// Two guards can fire here, and either is a correct outcome: the CPKS
/// solver rejects a non-positive-curvature direction (the reference is a
/// saddle), or SAPT rejects the resulting runaway rotation. What the test
/// pins is that a NUMBER is never reported, and that whatever comes back
/// explains itself.
#[test]
fn sapt_refuses_a_monomer_whose_response_has_blown_up() {
    const ANG: f64 = 1.889_726_124_6;
    let set = data::load("sto-3g").unwrap();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let water = Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 0.957 * ANG),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.927 * ANG, 0.0, -0.240 * ANG),
        },
    ]);
    let o_triplet = Molecule::new(vec![Atom {
        z: 8,
        position: Vec3::new(0.0, 0.0, 2.900 * ANG),
    }])
    .with_multiplicity(3);
    let msg = match sapt_interaction(&water, &o_triplet, &set, &func, &opts(), None) {
        Ok(_) => panic!("a degenerate open shell must be refused, not reported"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("orbital rotation") || msg.contains("non-positive curvature"),
        "the refusal must name the physical cause: {msg}"
    );
}

#[test]
fn sapt_hf_water_dimer_consistency() {
    let set = data::load("sto-3g").unwrap();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let o = opts();
    let a = water_a();
    let b = water_b(0.0);
    let res = sapt_interaction(&a, &b, &set, &func, &o, None).unwrap();

    // Frozen PySCF electrostatics oracle (independent integral stack).
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/oracle_data/sapt_water_dimer_elst.csv"),
    )
    .unwrap();
    let mut e_elst_ref = f64::NAN;
    let (mut ea_ref, mut eb_ref) = (f64::NAN, f64::NAN);
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let p: Vec<&str> = line.split(',').collect();
        match p[0] {
            "e_elst" => e_elst_ref = p[1].parse().unwrap(),
            "e_monomer_a" => ea_ref = p[1].parse().unwrap(),
            "e_monomer_b" => eb_ref = p[1].parse().unwrap(),
            _ => {}
        }
    }
    assert!(
        (res.e_monomer_a - ea_ref).abs() < 1e-8,
        "monomer A energy drifted"
    );
    assert!(
        (res.e_monomer_b - eb_ref).abs() < 1e-8,
        "monomer B energy drifted"
    );
    assert!(
        (res.e_elst - e_elst_ref).abs() < 1e-9,
        "E_elst {} vs PySCF oracle {}",
        res.e_elst,
        e_elst_ref
    );

    // Sanity: exchange repulsive, induction attractive, elst attractive
    // for this H-bond-like orientation.
    assert!(res.e_exch > 0.0, "exch {}", res.e_exch);
    assert!(
        res.e_ind_a < 0.0 && res.e_ind_b < 0.0,
        "induction must be attractive: {} {}",
        res.e_ind_a,
        res.e_ind_b
    );
    assert!(res.e_elst < 0.0, "elst {}", res.e_elst);

    assert!(
        res.e_disp_unc < 0.0,
        "dispersion must be attractive: {}",
        res.e_disp_unc
    );
    // Coupled dispersion: attractive, and screened relative to the
    // uncoupled sum (the monomer response raises the excitation energies,
    // so |E_disp| shrinks) while staying the same order of magnitude.
    assert!(
        res.e_disp_coupled < 0.0,
        "coupled dispersion must be attractive: {}",
        res.e_disp_coupled
    );
    // Calibration: the SAME sum-over-states machinery driven by uncoupled
    // excitations must reproduce the closed-form uncoupled dispersion.
    // Their ratio is the spin factor of the transition density.
    let cal = res.e_disp_sos_uncoupled / res.e_disp_unc;
    eprintln!(
        "dispersion calibration: sum-over-states/closed-form = {cal:.6} \
         (coupled {:.6e}, uncoupled {:.6e})",
        res.e_disp_coupled, res.e_disp_unc
    );
    assert!(
        (cal - 1.0).abs() < 1e-8,
        "sum-over-states dispersion is off the closed form by a factor {cal}"
    );
    // The amplitude sets are both orthonormal, so the total transition
    // weight is invariant and ONLY the denominators differ: coupling
    // therefore acts purely through the excitation energies. Singlet CIS
    // energies sit BELOW the orbital gaps (the -J electron-hole
    // attraction), so the coupled dispersion is the larger one.
    let ratio = res.e_disp_coupled / res.e_disp_unc;
    assert!(
        (1.0..2.5).contains(&ratio),
        "singlet coupling lowers the excitation energies, so coupled \
         dispersion should exceed the uncoupled value: ratio {ratio}"
    );

    // The derivative of the Heitler-London exchange can have either sign.
    // These two references are Richardson extrapolations from independently
    // reconverged monomer fields at steps 0.6 and 0.3 (see response.md).
    let directional_reference = [-1.7417142520336095e-5, 1.0286652613168538e-5];
    for (direction, (&actual, expected)) in res
        .e_exch_ind_directions
        .iter()
        .zip(directional_reference)
        .enumerate()
    {
        assert!(
            (actual - expected).abs() < 3e-12,
            "exchange-induction direction {direction}: measured {actual:.15e}, reference {expected:.15e}"
        );
    }
    let ind_total = (res.e_ind_a + res.e_ind_b).abs();
    assert!(
        res.e_exch_ind.abs() < 3.0 * ind_total,
        "exchange-induction {} dwarfs the induction {ind_total} it corrects",
        res.e_exch_ind
    );

    // Exchange-dispersion quenches dispersion: repulsive (positive) and
    // smaller in magnitude than the dispersion it damps.
    assert!(
        res.e_exch_disp > 0.0,
        "exchange-dispersion must be repulsive: {}",
        res.e_exch_disp
    );
    assert!(
        res.e_exch_disp < res.e_disp_coupled.abs(),
        "exchange-dispersion {} exceeds the dispersion {} it quenches",
        res.e_exch_disp,
        res.e_disp_coupled
    );

    // Supermolecular cross-check: the counterpoise-corrected HF interaction
    // energy of an H-bonded dimer is attractive. Independently pin both
    // public totals to their documented component sums and delta to the
    // exact supermolecular remainder.
    assert!(
        res.e_int_supermolecular < 0.0,
        "supermolecular interaction should bind: {}",
        res.e_int_supermolecular
    );
    let coupled_sum = res.e_elst
        + res.e_exch
        + res.e_ind_a
        + res.e_ind_b
        + res.e_exch_ind
        + res.e_disp_coupled
        + res.e_exch_disp;
    let uncoupled_calibration_sum =
        res.e_elst + res.e_exch + res.e_ind_a + res.e_ind_b + res.e_disp_unc;
    assert_eq!(res.interaction_total(), coupled_sum);
    assert_eq!(
        res.interaction_total_uncoupled_calibration(),
        uncoupled_calibration_sum
    );
    assert_eq!(
        res.delta,
        res.e_int_supermolecular - res.interaction_total()
    );

    // A↔B relabeling: scalar components, totals, and delta are invariant;
    // the two directional induction components swap.
    let swapped = sapt_interaction(&b, &a, &set, &func, &o, None).unwrap();
    assert!((res.e_elst - swapped.e_elst).abs() < 1e-10);
    assert!((res.e_exch - swapped.e_exch).abs() < 1e-9);
    assert!((res.e_ind_a - swapped.e_ind_b).abs() < 1e-9);
    assert!((res.e_ind_b - swapped.e_ind_a).abs() < 1e-9);
    assert!((res.e_disp_unc - swapped.e_disp_unc).abs() < 1e-9);
    assert!((res.e_exch_ind - swapped.e_exch_ind).abs() < 2e-9);
    assert!((res.e_disp_coupled - swapped.e_disp_coupled).abs() < 2e-9);
    assert!((res.e_exch_disp - swapped.e_exch_disp).abs() < 2e-9);
    assert!((res.interaction_total() - swapped.interaction_total()).abs() < 2e-9);
    assert!(
        (res.interaction_total_uncoupled_calibration()
            - swapped.interaction_total_uncoupled_calibration())
        .abs()
            < 2e-9
    );
    assert!((res.delta - swapped.delta).abs() < 2e-9);

    // Large separation: exchange dies exponentially, elst decays to the
    // multipole tail, dispersion ~ R^-6 (compare two separations by the
    // center-of-charge distance ratio).
    let far = sapt_interaction(&a, &water_b(14.0), &set, &func, &o, None).unwrap();
    assert!(far.e_exch.abs() < 1e-10, "far exch {}", far.e_exch);
    // Exchange-induction dies with the OVERLAP, but it is evaluated on
    // polarized orbitals whose polarization only decays as a multipole
    // series, so the long-range remnant is set by the residual induction
    // rather than by the (exponentially dead) exchange. Require it to be
    // negligible against the near-contact value instead of exactly zero.
    assert!(
        far.e_exch_disp.abs() < 1e-3 * res.e_exch_disp.abs(),
        "exchange-dispersion should collapse at long range: {} vs {}",
        far.e_exch_disp,
        res.e_exch_disp
    );
    assert!(
        far.e_exch_ind.abs() < 1e-3 * res.e_exch_ind.abs(),
        "exchange-induction should collapse at long range: {} vs {}",
        far.e_exch_ind,
        res.e_exch_ind
    );
    assert!(far.e_elst.abs() < 0.1 * res.e_elst.abs());
    let far2 = sapt_interaction(&a, &water_b(24.0), &set, &func, &o, None).unwrap();
    // O-O distances 19.4 and 29.4 Bohr: C6 estimate from each.
    let c6_1 = -far.e_disp_unc * 19.4_f64.powi(6);
    let c6_2 = -far2.e_disp_unc * 29.4_f64.powi(6);
    let ratio = c6_1 / c6_2;
    assert!(
        (0.6..1.6).contains(&ratio),
        "dispersion is not ~R^-6: C6 estimates {c6_1} vs {c6_2}"
    );
}

/// Coupled induction against the finite-difference identity: adding the
/// λ-scaled partner potential to the monomer H_core, the quadratic energy
/// coefficient must equal E_ind.
#[test]
fn sapt_coupled_induction_matches_finite_difference() {
    use dft_rs::basis::MolBasis;
    use dft_rs::integrals::one_electron::nuclear_matrix;
    use dft_rs::linalg::Matrix;
    use dft_rs::scf::JkEngine;

    let set = data::load("sto-3g").unwrap();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let o = opts();
    let a = water_a();
    let b = water_b(0.0);
    let res = sapt_interaction(&a, &b, &set, &func, &o, None).unwrap();

    // Rebuild ω^B on A's AO block exactly as sapt_interaction does.
    let mb_a = MolBasis::build(&a, &set).unwrap();
    let mut atoms = a.atoms.clone();
    atoms.extend(b.atoms.iter().cloned());
    let d = Molecule::new(atoms);
    let mb_d = MolBasis::build(&d, &set).unwrap();
    let scf_b = run_scf(&b, &MolBasis::build(&b, &set).unwrap(), &o, &func, None).unwrap();
    let nao_d = mb_d.nao_sph;
    let nao_a = mb_a.nao_sph;
    let mut p_b = Matrix::zeros(nao_d, nao_d);
    let pbm = scf_b.p_total();
    for r in 0..pbm.rows {
        for c in 0..pbm.cols {
            p_b[(r + nao_a, c + nao_a)] = pbm[(r, c)];
        }
    }
    let mut zb: Vec<f64> = d.atoms.iter().map(|at| at.z as f64).collect();
    for z in zb.iter_mut().take(a.len()) {
        *z = 0.0;
    }
    let v_b = nuclear_matrix(&mb_d, &d, Some(&zb));
    let jk = JkEngine::from_options(&d, &mb_d, &o).unwrap();
    let mut half = p_b.clone();
    for v in half.as_mut_slice() {
        *v *= 0.5;
    }
    let (j_b, _) = jk.build(&mb_d, &p_b, [&half, &half], false).unwrap();
    let mut omega = Matrix::zeros(nao_a, nao_a);
    for r in 0..nao_a {
        for c in 0..nao_a {
            omega[(r, c)] = v_b[(r, c)] + j_b[(r, c)];
        }
    }

    let e0 = run_scf(&a, &mb_a, &o, &func, None).unwrap().energy;
    let lam = 0.05;
    let e_at = |l: f64| -> f64 {
        let mut ol = o.clone();
        let mut ext = omega.clone();
        for v in ext.as_mut_slice() {
            *v *= l;
        }
        ol.extra_hcore = Some(ext);
        run_scf(&a, &mb_a, &ol, &func, None).unwrap().energy
    };
    let fd = (e_at(lam) + e_at(-lam) - 2.0 * e0) / (2.0 * lam * lam);
    assert!(
        (res.e_ind_a - fd).abs() < 5e-7,
        "coupled induction {} vs FD {}",
        res.e_ind_a,
        fd
    );
}

/// Every SAPT component is a scalar observable and must be invariant when
/// both monomers undergo the same rigid rotation and translation. This is a
/// whole-pipeline check on AO rotations, intermonomer integrals, response,
/// and the component bookkeeping, not merely on the final sum.
#[test]
fn every_sapt_component_is_invariant_under_rigid_motion() {
    let rotate = |p: Vec3| {
        let axis = Vec3::new(1.0, -2.0, 0.7);
        let u = axis * (1.0 / axis.norm());
        let angle: f64 = 0.731;
        p * angle.cos() + u.cross(p) * angle.sin() + u * (u.dot(p) * (1.0 - angle.cos()))
    };
    let transform = |mol: Molecule| {
        let charge = mol.charge;
        let multiplicity = mol.multiplicity;
        Molecule::new(
            mol.atoms
                .into_iter()
                .map(|atom| Atom {
                    z: atom.z,
                    position: rotate(atom.position) + Vec3::new(2.3, -1.7, 0.4),
                })
                .collect(),
        )
        .with_charge(charge)
        .with_multiplicity(multiplicity)
    };
    let set = data::load("sto-3g").unwrap();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let original = sapt_interaction(&water_a(), &water_b(0.0), &set, &func, &opts(), None).unwrap();
    let moved = sapt_interaction(
        &transform(water_a()),
        &transform(water_b(0.0)),
        &set,
        &func,
        &opts(),
        None,
    )
    .unwrap();
    for (name, x, y) in [
        ("electrostatics", original.e_elst, moved.e_elst),
        ("exchange", original.e_exch, moved.e_exch),
        ("induction A", original.e_ind_a, moved.e_ind_a),
        ("induction B", original.e_ind_b, moved.e_ind_b),
        ("exchange-induction", original.e_exch_ind, moved.e_exch_ind),
        (
            "uncoupled dispersion",
            original.e_disp_unc,
            moved.e_disp_unc,
        ),
        (
            "coupled dispersion",
            original.e_disp_coupled,
            moved.e_disp_coupled,
        ),
        (
            "exchange-dispersion",
            original.e_exch_disp,
            moved.e_exch_disp,
        ),
        (
            "supermolecular",
            original.e_int_supermolecular,
            moved.e_int_supermolecular,
        ),
        (
            "coupled interaction total",
            original.interaction_total(),
            moved.interaction_total(),
        ),
        (
            "uncoupled calibration total",
            original.interaction_total_uncoupled_calibration(),
            moved.interaction_total_uncoupled_calibration(),
        ),
        ("delta", original.delta, moved.delta),
    ] {
        assert!((x - y).abs() < 2e-9, "{name}: {x:.12e} vs {y:.12e}");
    }
}
