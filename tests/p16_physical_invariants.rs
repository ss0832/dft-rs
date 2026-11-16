// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Physical invariants of the Hartree-Fock foundation.
//!
//! These gates deliberately do not use frozen values from another program.
//! They check consequences of the variational many-electron theory: most
//! hold EXACTLY regardless of basis quality (cancellation of one-electron
//! self-interaction, particle-number/idempotency constraints on a Slater
//! determinant, stationarity of a converged SCF solution, Euclidean
//! invariance of an isolated molecule, Koopmans' frozen-orbital ionization
//! identity, MO-coefficient orthonormality, ascending MO-energy order,
//! dipole origin-(in)dependence). One (the virial ratio) holds only in the
//! basis-set limit and is checked as a shrinking-deviation trend instead of
//! an equality. Each test's doc comment states which kind it is.

use dft_rs::basis::{data, MolBasis};
use dft_rs::gradient::{assemble_gradient, numerical_gradient};
use dft_rs::integrals::kinetic_matrix;
use dft_rs::linalg::{Matrix, Orthogonalizer};
use dft_rs::math::Vec3;
use dft_rs::scf::{run_hf, run_scf, EriMode, Guess, Reference, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::ResolvedFunctional;

fn tight_options() -> ScfOptions {
    ScfOptions {
        e_tol: 1e-12,
        p_tol: 1e-10,
        conv_tol_grad: 1e-9,
        max_iter: 300,
        guess: Guess::Core,
        eri_mode: EriMode::InCore { limit_mb: 256 },
        ..ScfOptions::default()
    }
}

fn water(atoms: impl Fn(Vec3) -> Vec3) -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: atoms(Vec3::new(0.31, -0.27, 0.19)),
        },
        Atom {
            z: 1,
            position: atoms(Vec3::new(1.76, 0.18, -0.43)),
        },
        Atom {
            z: 1,
            position: atoms(Vec3::new(-0.54, 1.37, 0.72)),
        },
    ])
}

/// A one-electron determinant has no electron-electron interaction: its
/// Hartree self-repulsion is cancelled exactly by same-spin exchange.
/// Consequently HF must reduce to the lowest generalized eigenvalue of the
/// one-electron core Hamiltonian in the same finite basis.  This is a sharper
/// gate than comparison to an external SCF code because any shared spin-factor
/// convention error would otherwise be reproduced by both totals.
#[test]
fn one_electron_hf_is_exactly_self_interaction_free() {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::new(0.23, -0.41, 0.67),
    }])
    .with_multiplicity(2);
    let basis = data::load("def2-svp").unwrap();
    let mb = MolBasis::build(&mol, &basis).unwrap();
    let scf = run_hf(&mol, &mb, &tight_options()).unwrap();

    let orth = Orthogonalizer::canonical(&scf.overlap, 1e-7).unwrap();
    let (one_electron_levels, _) = orth.solve_roothaan(&scf.h_core).unwrap();
    let expected = one_electron_levels[0];

    assert!(scf.unrestricted, "a doublet must use the unrestricted path");
    assert_eq!((scf.n_alpha, scf.n_beta), (1, 0));
    assert!(
        (scf.energy - expected).abs() < 2e-11,
        "one-electron HF contains a self-interaction: E_HF = {}, E_core = {}",
        scf.energy,
        expected
    );
    // J and K need not be identical as full AO matrices: exchange is
    // non-local.  Their difference must instead annihilate the occupied
    // orbital, (J - K)|phi> = 0.
    let mut self_interaction_action = vec![0.0; scf.h_core.rows];
    for mu in 0..scf.h_core.rows {
        for nu in 0..scf.h_core.cols {
            self_interaction_action[mu] +=
                (scf.fock[0][(mu, nu)] - scf.h_core[(mu, nu)]) * scf.mo_coeff[0][(nu, 0)];
        }
    }
    let action_norm = self_interaction_action
        .iter()
        .map(|x| x.abs())
        .fold(0.0, f64::max);
    assert!(
        action_norm < 2e-11,
        "(J-K) did not annihilate the occupied orbital: {action_norm:e}"
    );
    assert!(
        scf.density[1].as_slice().iter().all(|x| x.abs() < 1e-15),
        "the empty beta channel acquired density"
    );
    assert!(
        scf.energy >= -0.5 - 1e-12,
        "finite-basis hydrogen violated the variational lower bound: {}",
        scf.energy
    );
}

/// A converged single determinant is an N-electron projector in the AO
/// metric, and its density commutes with its Fock operator in that metric.
/// These are the Euler-Lagrange conditions of the constrained HF variational
/// problem, not implementation-specific convergence diagnostics.
#[test]
fn converged_density_is_a_stationary_n_electron_projector() {
    let mol = water(|r| r);
    let basis = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &basis).unwrap();
    let scf = run_hf(&mol, &mb, &tight_options()).unwrap();

    let electron_count = scf.p_total().frobenius_dot(&scf.overlap);
    assert!(
        (electron_count - 10.0).abs() < 2e-10,
        "Tr[P S] = {electron_count}, expected 10"
    );

    for spin in 0..2 {
        let p = &scf.density[spin];
        let psp = p.matmul(&scf.overlap).matmul(p);
        assert!(
            psp.max_abs_diff(p) < 2e-9,
            "spin-{spin} density is not idempotent in the AO metric"
        );

        // Generalized commutator F P S - S P F = 0.
        let fps = scf.fock[spin].matmul(p).matmul(&scf.overlap);
        let spf = scf.overlap.matmul(p).matmul(&scf.fock[spin]);
        assert!(
            fps.max_abs_diff(&spf) < 2e-8,
            "spin-{spin} SCF solution is not stationary: max|FPS-SPF| = {}",
            fps.max_abs_diff(&spf)
        );
    }
}

/// An isolated non-relativistic molecule has no preferred origin or
/// orientation.  The arbitrary rotation exercises complete p shells and is
/// followed by a translation, so this checks total-energy assembly as well as
/// angular integral transformations and nuclear repulsion.
#[test]
fn hf_energy_is_invariant_under_rigid_motion() {
    let axis = Vec3::new(1.0, -2.0, 3.0);
    let inv_norm = 1.0 / axis.norm();
    let u = axis * inv_norm;
    let angle: f64 = 0.731;
    let (c, s) = (angle.cos(), angle.sin());
    let shift = Vec3::new(4.2, -1.7, 0.83);
    let rotate_translate = |r: Vec3| {
        // Rodrigues rotation formula.
        r * c + u.cross(r) * s + u * (u.dot(r) * (1.0 - c)) + shift
    };

    let basis = data::load("6-31g*").unwrap();
    let mol_a = water(|r| r);
    let mol_b = water(rotate_translate);
    let mb_a = MolBasis::build(&mol_a, &basis).unwrap();
    let mb_b = MolBasis::build(&mol_b, &basis).unwrap();
    let mut opts = tight_options();
    opts.reference = Reference::Rhf;
    let e_a = run_hf(&mol_a, &mb_a, &opts).unwrap().energy;
    let e_b = run_hf(&mol_b, &mb_b, &opts).unwrap().energy;

    assert!(
        (e_a - e_b).abs() < 2e-10,
        "rigid motion changed the HF energy: {e_a} vs {e_b}"
    );
}

/// Size consistency is essential for a molecular electronic-structure
/// method: two non-interacting fragments must have the sum of the fragment
/// energies.  RHF is known not to have the right H2 dissociation limit, so
/// this gate deliberately uses the broken-symmetry UHF solution with one
/// electron localized on each atom.
#[test]
fn unrestricted_hf_has_the_correct_h2_dissociation_limit() {
    let basis = data::load("sto-3g").unwrap();
    let atom = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }])
    .with_multiplicity(2);
    let atom_basis = MolBasis::build(&atom, &basis).unwrap();
    let e_atom = run_hf(&atom, &atom_basis, &tight_options()).unwrap().energy;

    let separation = 20.0;
    let dimer = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(-0.5 * separation, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.5 * separation, 0.0, 0.0),
        },
    ]);
    let dimer_basis = MolBasis::build(&dimer, &basis).unwrap();
    let mut opts = tight_options();
    opts.reference = Reference::Uhf;
    opts.guess_mix = 45.0;
    // The dissociation limit is exactly degenerate.  Newton acceleration is
    // unnecessary there and a deliberately singular orbital Hessian is not
    // a useful object for this size-consistency gate.
    opts.use_soscf = false;
    let result = run_hf(&dimer, &dimer_basis, &opts).unwrap();

    assert!(
        (result.energy - 2.0 * e_atom).abs() < 2e-9,
        "UHF is not size consistent at H2 dissociation: E(H2) = {}, 2E(H) = {}",
        result.energy,
        2.0 * e_atom
    );
    assert!(
        (result.s_squared - 1.0).abs() < 2e-9,
        "the separated alpha/beta determinant should have <S^2> = 1, got {}",
        result.s_squared
    );
}

/// `ScfOptions::extra_hcore` is contracted to be independent of the nuclear
/// coordinates: it contributes to the energy through `Tr[P.extra]` and to
/// NOTHING else, so `assemble_gradient` (and the Hessian / excited-state
/// paths behind it) never differentiate it.  That is only a sound thing to
/// do because a fixed AO-space matrix has an identically zero nuclear
/// derivative.  This is an EXACT invariant: it builds a constant operator
/// out of AO indices alone (never out of a nuclear position, so it is,
/// by construction, the very same matrix at every displaced geometry the
/// numerical side below visits — precisely what "geometry independent"
/// means), and checks both halves of the contract in one gate:
///   1. the operator must actually move the energy (guards against a
///      no-op that would make the rest of the test pass vacuously even if
///      `extra_hcore` were silently dropped somewhere), and
///   2. the analytic gradient of the perturbed system — which never reads
///      `extra_hcore` — must still reproduce the numerical gradient of the
///      SAME perturbed total energy.
///
/// A geometry-DEPENDENT operator smuggled through this field would fail
/// exactly point 2: its missing dR/dR term would show up as a systematic
/// analytic/numerical mismatch, not as noise.
#[test]
fn extra_hcore_is_gradient_invisible_because_it_is_geometry_independent() {
    let mol = water(|r| r);
    let basis_set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &basis_set).unwrap();
    let hf = ResolvedFunctional::hartree_fock();

    // Smooth, deterministic function of the AO indices only. Built from
    // index products rather than from any geometric or overlap quantity,
    // so it is manifestly not a disguised multiple of S and manifestly
    // carries no dependence on nuclear position.
    let n = mb.nao_sph;
    let scale = 1e-3;
    let mut extra = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            extra[(i, j)] = scale * (((i + 1) * (j + 1)) as f64).sqrt().sin();
        }
    }
    // The construction above is already symmetric in (i, j), but symmetrize
    // explicitly so the test's validity does not rest on that being true.
    for i in 0..n {
        for j in 0..i {
            let avg = 0.5 * (extra[(i, j)] + extra[(j, i)]);
            extra[(i, j)] = avg;
            extra[(j, i)] = avg;
        }
    }

    let opts = tight_options();
    let mut opts_extra = opts.clone();
    opts_extra.extra_hcore = Some(extra);

    let e_base = run_hf(&mol, &mb, &opts).unwrap().energy;
    let scf_extra = run_scf(&mol, &mb, &opts_extra, &hf, None).unwrap();
    assert!(
        (scf_extra.energy - e_base).abs() > 1e-6,
        "extra_hcore did not move the energy -- this gate would be vacuous: \
         E_base = {e_base}, E_extra = {}",
        scf_extra.energy
    );

    let analytic = assemble_gradient(&mol, &mb, &hf, None, &scf_extra, &opts_extra).unwrap();
    let numerical = numerical_gradient(&mol, &basis_set, &opts_extra, &hf, None, 1e-4).unwrap();
    for (a, (g, f)) in analytic.iter().zip(&numerical).enumerate() {
        let d = (*g - *f).norm();
        assert!(
            d < 1e-6,
            "atom {a}: analytic gradient (extra_hcore-blind) {g:?} vs FD of \
             the extra_hcore-perturbed energy {f:?}, |delta| = {d:.2e}"
        );
    }
}
/// ASYMPTOTIC: the electronic virial theorem `-V/T -> 2` is a property of
/// the EXACT wavefunction (it follows from the Hellmann-Feynman-style
/// coordinate-scaling stationarity condition at the true minimum); a finite
/// Gaussian basis does not span that stationary point, so a single basis is
/// not expected to hit 2 exactly. This gate instead checks the one thing
/// that must be true in any basis: the deviation `|(-V/T) - 2|` shrinks
/// monotonically as sto-3g -> 6-31g -> def2-svp improves the basis, and
/// every value stays in a sane band. `T = Tr[P * T_matrix]` is computed
/// from the public `kinetic_matrix` and the converged total density; `V =
/// E_total - T` folds in everything else (nuclear attraction, electron
/// repulsion, and nuclear repulsion) since HF exposes no separate
/// potential-energy accessor.
#[test]
fn hf_virial_ratio_deviation_shrinks_as_the_basis_improves() {
    let mol = water(|r| r);
    let mut opts = tight_options();
    opts.reference = Reference::Rhf;
    let mut deviations = Vec::new();
    for basis_name in ["sto-3g", "6-31g", "def2-svp"] {
        let basis = data::load(basis_name).unwrap();
        let mb = MolBasis::build(&mol, &basis).unwrap();
        let scf = run_hf(&mol, &mb, &opts).unwrap();
        let t = scf.p_total().frobenius_dot(&kinetic_matrix(&mb));
        let v = scf.energy - t;
        let ratio = -v / t;
        assert!(
            (1.9..=2.1).contains(&ratio),
            "{basis_name}: virial ratio {ratio} is outside the 1.9-2.1 sanity band"
        );
        deviations.push((basis_name, (ratio - 2.0).abs()));
    }
    for pair in deviations.windows(2) {
        let (name_a, dev_a) = pair[0];
        let (name_b, dev_b) = pair[1];
        assert!(
            dev_b < dev_a,
            "virial deviation did not shrink going {name_a} ({dev_a:e}) -> {name_b} ({dev_b:e})"
        );
    }
}

/// EXACT, HF-only. Koopmans' theorem: `-eps_HOMO` equals the FROZEN-orbital
/// ionization energy `E(N-1, same orbitals) - E(N)`. This is an algebraic
/// identity of a single Slater determinant (Szabo & Ostlund 3.98-3.100),
/// not an approximation, PROVIDED the cation is built without reoptimizing
/// the orbitals. The electronic energy functional this crate's SCF driver
/// already uses,
///   E[P] = Tr[P h] + (1/2) Tr[P_tot J(P_tot)] - (1/2) sum_s Tr[P_s K(P_s)],
/// is evaluated here directly on a frozen-orbital cation density P' (the
/// HOMO electron removed from the alpha channel only, beta channel
/// untouched -- a doublet built from the SAME MO coefficients as the
/// neutral) via `JkEngine`, WITHOUT ever re-diagononalizing a Fock matrix.
/// Re-optimizing the orbitals would instead give the (generally different)
/// "relaxed" ionization energy. The identity is specific to HF exchange
/// (K linear in the density that builds it); DFT's Vxc is a nonlinear
/// functional of the density, so this is deliberately not tested there --
/// only Janak's theorem (dE/dn = eps) holds for DFT, and only in the
/// fractional-occupation limit.
#[test]
fn koopmans_theorem_is_exact_for_frozen_orbital_hf_ionization() {
    let mol = water(|r| r);
    let basis = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &basis).unwrap();
    let mut opts = tight_options();
    opts.reference = Reference::Rhf;
    let scf = run_hf(&mol, &mb, &opts).unwrap();
    assert!(
        !scf.unrestricted,
        "Koopmans' theorem here assumes a closed-shell RHF reference"
    );

    let homo = scf.n_alpha - 1;
    let eps_homo = scf.mo_energy[0][homo];

    // Frozen-orbital cation density: the leading `homo` columns are the
    // occupied MOs BELOW the HOMO, so this drops exactly the HOMO electron
    // from the alpha channel; the beta channel (= the neutral's, since RHF
    // has density[0] == density[1]) is untouched.
    let p_alpha_cation = scf.mo_coeff[0].leading_columns_gram(homo, 1.0);
    let p_beta_cation = scf.density[1].clone();
    let mut p_tot_cation = p_alpha_cation.clone();
    for (t, b) in p_tot_cation
        .as_mut_slice()
        .iter_mut()
        .zip(p_beta_cation.as_slice())
    {
        *t += *b;
    }

    let jk = dft_rs::scf::JkEngine::new(
        &mb,
        EriMode::InCore { limit_mb: 256 },
        dft_rs::integrals::CoulombKernel::Coulomb,
    )
    .unwrap();
    let (j, k) = jk
        .build(&mb, &p_tot_cation, [&p_alpha_cation, &p_beta_cation], true)
        .unwrap();
    let e_elec_cation = p_tot_cation.frobenius_dot(&scf.h_core)
        + 0.5 * p_tot_cation.frobenius_dot(&j)
        - 0.5 * (p_alpha_cation.frobenius_dot(&k[0]) + p_beta_cation.frobenius_dot(&k[1]));
    // Same geometry as the neutral, so nuclear repulsion is unchanged.
    let e_cation_frozen = e_elec_cation + scf.e_nuc;

    let ip_frozen = e_cation_frozen - scf.energy;
    assert!(
        (ip_frozen - (-eps_homo)).abs() < 1e-8,
        "Koopmans identity violated: IP_frozen = {ip_frozen}, -eps_HOMO = {}",
        -eps_homo
    );
}

/// MO energies come back sorted ascending. EXACT: `symmetric_eigen` is
/// documented to return ascending eigenvalues, so this is a corollary of
/// the eigendecomposition itself, not a physical law. Because the SCF
/// driver always occupies the LEADING `n_occ` columns of that sorted
/// eigenvector set (aufbau filling by construction), a converged
/// closed-shell ground state must then have every occupied orbital at or
/// below every virtual one; asserting max(occupied) < min(virtual) is a
/// SANITY check specific to this well-behaved case, not a general claim
/// that HOMO-LUMO gaps are always positive (degenerate or open-shell
/// systems can have an arbitrarily small or vanishing gap).
#[test]
fn mo_energies_are_sorted_and_aufbau_fills_the_lowest_orbitals() {
    let mol = water(|r| r);
    let basis = data::load("def2-svp").unwrap();
    let mb = MolBasis::build(&mol, &basis).unwrap();
    let mut opts = tight_options();
    opts.reference = Reference::Rhf;
    let scf = run_hf(&mol, &mb, &opts).unwrap();

    for w in scf.mo_energy[0].windows(2) {
        assert!(
            w[0] <= w[1],
            "MO energies are not ascending: {} follows {}",
            w[1],
            w[0]
        );
    }

    let occupied_max = scf.mo_energy[0][..scf.n_alpha]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let virtual_min = scf.mo_energy[0][scf.n_alpha..]
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    assert!(
        occupied_max < virtual_min,
        "an occupied orbital ({occupied_max}) is not below the lowest virtual ({virtual_min})"
    );
    let (homo, lumo) = scf.homo_lumo();
    assert_eq!(homo, Some(occupied_max));
    assert_eq!(lumo, Some(virtual_min));
}

/// EXACT. The dipole moment of a NEUTRAL charge distribution does not
/// depend on where the origin is placed: `mu(O) = mu(0) - Q*O` for total
/// charge Q, and Q=0 removes all O-dependence. `dipole_moment` always
/// integrates about the fixed AO-basis origin, so instead of moving the
/// origin this moves the MOLECULE by `d` in that fixed frame and reruns the
/// SCF; by the rigid-motion invariance this file already establishes, that
/// reproduces the identical electron distribution translated by `d`, so
///   mu'(0) = sum_A Z_A(R_A+d) - Tr[P'(r+d)]
///          = mu(0) + d * (sum_A Z_A - N_elec) = mu(0) + d * Q_total
/// exactly (N_elec = Tr[P S] is exact by the idempotency/particle-number
/// identity checked elsewhere in this file). For a NEUTRAL molecule (Q=0)
/// this collapses to mu' = mu -- testing ONLY that case would pass even
/// with a completely wrong sign or a missing Q*d term entirely, so a CATION
/// (Q=+1) is tested too, where the predicted shift is a full `d`, not noise.
#[test]
fn dipole_moment_is_origin_independent_for_neutrals_and_shifts_by_charge_times_translation() {
    let basis = data::load("sto-3g").unwrap();
    let shift = Vec3::new(3.1, -2.4, 1.7);

    // Neutral: origin-independent, mu' = mu exactly (Q = 0).
    let mut opts = tight_options();
    opts.reference = Reference::Rhf;
    let mol_a = water(|r| r);
    let mol_b = water(|r| r + shift);
    let mb_a = MolBasis::build(&mol_a, &basis).unwrap();
    let mb_b = MolBasis::build(&mol_b, &basis).unwrap();
    let scf_a = run_hf(&mol_a, &mb_a, &opts).unwrap();
    let scf_b = run_hf(&mol_b, &mb_b, &opts).unwrap();
    let mu_a = dft_rs::properties::dipole_moment(&mol_a, &mb_a, &scf_a);
    let mu_b = dft_rs::properties::dipole_moment(&mol_b, &mb_b, &scf_b);
    assert!(
        (mu_a - mu_b).norm() < 2e-8,
        "neutral dipole is origin-dependent: {mu_a:?} vs {mu_b:?}"
    );

    // Cation: mu'(d) = mu(0) + d * Q, Q = +1.
    let cation_a = water(|r| r).with_charge(1).with_multiplicity(2);
    let cation_b = water(|r| r + shift).with_charge(1).with_multiplicity(2);
    let mut uopts = tight_options();
    uopts.reference = Reference::Uhf;
    let cmb_a = MolBasis::build(&cation_a, &basis).unwrap();
    let cmb_b = MolBasis::build(&cation_b, &basis).unwrap();
    let cscf_a = run_hf(&cation_a, &cmb_a, &uopts).unwrap();
    let cscf_b = run_hf(&cation_b, &cmb_b, &uopts).unwrap();
    let cmu_a = dft_rs::properties::dipole_moment(&cation_a, &cmb_a, &cscf_a);
    let cmu_b = dft_rs::properties::dipole_moment(&cation_b, &cmb_b, &cscf_b);
    let predicted = cmu_a + shift; // Q_total = +1
    assert!(
        (cmu_b - predicted).norm() < 2e-8,
        "cation dipole shift != d*Q: measured shift {:?}, expected {:?}",
        cmu_b - cmu_a,
        shift
    );
}

/// EXACT. MO coefficients solve the generalized eigenproblem `F C = S C
/// eps` with eigenvectors orthonormal in the AO overlap metric: `CtSC = I`.
/// This is a linear-algebra identity of `Orthogonalizer::solve_roothaan`
/// (`symmetric_eigen` applied to the canonically-orthogonalized Fock,
/// transformed back by X) -- it holds for any solved generalized
/// eigenproblem, converged or not, so it is checked as an exact gate.
/// Covers both spin channels for a RESTRICTED run (where density[1] and
/// mo_coeff[1] are aliased copies of channel 0 -- this path could silently
/// carry a stale or malformed matrix without the aliasing itself being
/// wrong) and for an UNRESTRICTED run with genuinely different alpha/beta
/// occupations, which is otherwise untested in Rust (previously only
/// `python/tests/test_ase.py` checked this).
#[test]
fn mo_coefficients_are_orthonormal_in_the_overlap_metric_both_spins_both_references() {
    let identity_check = |c: &dft_rs::linalg::Matrix, s: &dft_rs::linalg::Matrix, label: &str| {
        let ctsc = c.transpose_matmul(&s.matmul(c));
        let identity = dft_rs::linalg::Matrix::identity(ctsc.rows);
        let max_err = ctsc.max_abs_diff(&identity);
        assert!(max_err < 1e-9, "{label}: max|CtSC - I| = {max_err:e}");
        max_err
    };

    // Restricted: closed-shell water. alpha and beta mo_coeff are aliased.
    let mol = water(|r| r);
    let basis = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &basis).unwrap();
    let mut opts = tight_options();
    opts.reference = Reference::Rhf;
    let scf = run_hf(&mol, &mb, &opts).unwrap();
    assert!(!scf.unrestricted);
    identity_check(&scf.mo_coeff[0], &scf.overlap, "restricted alpha");
    identity_check(&scf.mo_coeff[1], &scf.overlap, "restricted beta");

    // Unrestricted: an open-shell doublet (Li atom) with genuinely
    // different alpha (2) and beta (1) occupations, so both channels solve
    // an independent generalized eigenproblem.
    let atom = Molecule::new(vec![Atom {
        z: 3,
        position: Vec3::zero(),
    }])
    .with_multiplicity(2);
    let atom_basis = data::load("sto-3g").unwrap();
    let atom_mb = MolBasis::build(&atom, &atom_basis).unwrap();
    let mut uopts = tight_options();
    uopts.reference = Reference::Uhf;
    let uscf = run_hf(&atom, &atom_mb, &uopts).unwrap();
    assert!(uscf.unrestricted);
    assert_ne!(uscf.n_alpha, uscf.n_beta);
    identity_check(&uscf.mo_coeff[0], &uscf.overlap, "unrestricted alpha");
    identity_check(&uscf.mo_coeff[1], &uscf.overlap, "unrestricted beta");
}

/// Forcing the unrestricted path on a closed shell reproduces the restricted
/// answer exactly, on every rung of the functional ladder.
///
/// Classification: **exact**. The restricted formulation is the closed-shell
/// special case of the unrestricted one in this code, not a separate
/// implementation, so with equal alpha and beta occupations and no guess
/// mixing the two must converge to the same determinant. Any difference is a
/// spin-channel bug, not a tolerance question.
///
/// This is the sharpest available test of the empty- and equal-channel
/// branches. The ferromagnetic branches added to PBE-C and the LDA
/// correlations key on a per-channel density cutoff, and the spin-scaling
/// relation is exercised throughout; a branch taken on the wrong channel, or
/// a same-spin term double counted, changes the unrestricted total while
/// leaving the restricted one untouched. A single-functional check would
/// miss it, because each rung reaches the spin machinery differently: LDA
/// through the zeta interpolation, GGA through the per-spin gradient
/// channels, meta through tau, and RSH through the screened same-spin hole.
///
/// Also asserts <S^2> = 0 for the closed-shell unrestricted solution, which
/// is exact for a determinant with identical alpha and beta orbitals and is
/// what distinguishes a genuine restricted solution from a broken-symmetry
/// one that happens to have the same energy.
#[test]
fn unrestricted_reproduces_restricted_for_a_closed_shell_on_every_rung() {
    let mol = water(|r| r);
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let gopts = dft_rs::grid::GridOptions {
        level: 4,
        ..dft_rs::grid::GridOptions::default()
    };
    let grid = dft_rs::grid::MolecularGrid::build(&mol, &gopts).unwrap();

    // One representative per rung, plus a screened hybrid, because each
    // reaches the spin-resolved code by a different route.
    for name in [
        "svwn5",
        "blyp",
        "b3lyp",
        "tpss",
        "m06l",
        "cam-b3lyp",
        "hse06",
    ] {
        let func = dft_rs::xc::resolve(&dft_rs::xc::FunctionalSpec::Combined(name.into())).unwrap();

        let mut ropts = tight_options();
        ropts.reference = Reference::Rhf;
        let r = run_scf(&mol, &mb, &ropts, &func, Some(&grid)).unwrap();

        let mut uopts = tight_options();
        uopts.reference = Reference::Uhf;
        uopts.guess_mix = 0.0; // no symmetry breaking: must fall back onto RHF
        let u = run_scf(&mol, &mb, &uopts, &func, Some(&grid)).unwrap();

        assert!(
            r.converged && u.converged,
            "{name}: an SCF did not converge (restricted {}, unrestricted {})",
            r.converged,
            u.converged
        );
        assert!(
            u.unrestricted,
            "{name}: Reference::Uhf did not take the UKS path"
        );
        assert!(
            (r.energy - u.energy).abs() < 1e-9,
            "{name}: restricted {} vs unrestricted {} (Δ = {:.3e}). These are the same \
             determinant; a difference means a spin channel is handled inconsistently \
             between the two paths.",
            r.energy,
            u.energy,
            r.energy - u.energy
        );
        assert!(
            u.s_squared.abs() < 1e-9,
            "{name}: closed-shell unrestricted solution has <S^2> = {}, so it is not the \
             restricted determinant even though its energy may match",
            u.s_squared
        );
    }
}
/// Size consistency, decomposed by what each method actually promises.
///
/// Lumping these together as one "size consistency" assertion would be
/// wrong, because the methods do not make the same promise:
///
/// - a **semilocal functional** is strictly additive over non-overlapping
///   densities, so `E(A...B)` at large separation equals `E(A) + E(B)`
///   exactly (**exact**, up to the residual overlap and the SCF threshold);
/// - **dispersion** is a pairwise interaction that is SUPPOSED to survive at
///   large R, decaying as R^-6. Demanding it vanish would be demanding the
///   correction not work (**asymptotic**, and asserted as a decay rate).
///
/// The existing suite tests only the UHF H2 dissociation limit, which is a
/// different statement again (spin symmetry breaking), and nothing covers
/// DFT at all.
///
/// He...Ne is used rather than two copies of one atom: identical fragments
/// can mask an error that cancels between them, and both are closed-shell so
/// no symmetry breaking is involved.
#[test]
fn semilocal_functionals_are_size_consistent_while_dispersion_decays_as_r6() {
    let set = data::load("sto-3g").unwrap();
    let opts = tight_options();
    let gopts = dft_rs::grid::GridOptions {
        level: 4,
        ..dft_rs::grid::GridOptions::default()
    };

    let atom = |z: u8, x: f64| {
        Molecule::new(vec![Atom {
            z,
            position: Vec3::new(x, 0.0, 0.0),
        }])
    };
    let pair = |r: f64| {
        Molecule::new(vec![
            Atom {
                z: 2,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            Atom {
                z: 10,
                position: Vec3::new(r, 0.0, 0.0),
            },
        ])
    };

    for name in ["svwn5", "blyp", "tpss"] {
        let func = dft_rs::xc::resolve(&dft_rs::xc::FunctionalSpec::Combined(name.into())).unwrap();
        let e_of = |m: &Molecule| {
            let b = MolBasis::build(m, &set).unwrap();
            let g = dft_rs::grid::MolecularGrid::build(m, &gopts).unwrap();
            run_scf(m, &b, &opts, &func, Some(&g)).unwrap().energy
        };
        let e_he = e_of(&atom(2, 0.0));
        let e_ne = e_of(&atom(10, 0.0));
        // 60 Bohr: far outside any STO-3G overlap, so the fragments are
        // genuinely non-interacting rather than merely weakly interacting.
        let e_pair = e_of(&pair(60.0));
        assert!(
            (e_pair - (e_he + e_ne)).abs() < 1e-8,
            "{name}: E(He...Ne, 60 bohr) = {e_pair} but E(He) + E(Ne) = {} (Δ = {:.3e}). \
             A semilocal functional is additive over non-overlapping densities; a residual \
             means the grid partition or the XC integration is leaking between fragments.",
            e_he + e_ne,
            e_pair - (e_he + e_ne)
        );
    }

    // Dispersion must NOT vanish -- it must decay as R^-6. Doubling R has to
    // cut the interaction by close to 2^6 = 64.
    for kind in [
        dft_rs::dispersion::Dispersion::D2,
        dft_rs::dispersion::Dispersion::D3Bj,
    ] {
        let e_at = |r: f64| {
            let m = pair(r);
            let inter = dft_rs::dispersion::dispersion_energy(&m, &kind, "blyp").unwrap();
            let a = dft_rs::dispersion::dispersion_energy(&atom(2, 0.0), &kind, "blyp").unwrap();
            let b = dft_rs::dispersion::dispersion_energy(&atom(10, 0.0), &kind, "blyp").unwrap();
            inter - a - b
        };
        let near = e_at(12.0);
        let far = e_at(24.0);
        assert!(
            near < 0.0 && far < 0.0,
            "{kind:?}: dispersion must be attractive (got {near}, {far})"
        );
        let ratio = near / far;
        assert!(
            (32.0..=128.0).contains(&ratio),
            "{kind:?}: doubling the separation changed the dispersion interaction by a \
             factor {ratio}, but an R^-6 tail requires close to 64. This is the property \
             that distinguishes a working long-range correction from one that has been \
             damped away."
        );
    }
}
/// Molecular point-group symmetry shows up in the spectrum, and nothing in
/// the code is told about it.
///
/// Classification: **exact** for Hartree-Fock, **asymptotic** for a
/// functional that needs a grid — and the difference is the interesting
/// part, so both are measured.
///
/// Methane is Td. Its valence occupied orbitals are one a1 and a THREE-FOLD
/// degenerate t2 set, and its dipole vanishes identically. Neither fact is
/// input anywhere: no symmetry detection, no symmetry-adapted basis, no
/// point-group labels. The degeneracy has to emerge from the angular
/// integrals, the contraction normalisation and the SCF all being right
/// together. A transposed spherical-harmonic block or a mis-scaled p
/// component splits t2 while leaving every existing test — energies against
/// an oracle, sum rules, idempotency — perfectly happy, because they are all
/// invariant under exactly the kind of error that lifts a degeneracy.
///
/// For HF the only limit is the SCF threshold. For a grid-based functional
/// the atom-centred product quadrature is NOT invariant under the Td
/// operations, so the degeneracy is split by angular truncation and is
/// recovered only as the grid is refined — the same mechanism that limits
/// the rotational sum rule in `p5_gradient.rs`. Asserting the DFT case as
/// exact would be asserting an isotropy the quadrature does not have.
#[test]
fn methane_t2_orbitals_are_degenerate_and_its_dipole_vanishes() {
    // Td: carbon at the centre, hydrogens on alternating cube corners.
    let a = 1.1859; // C-H, Bohr-scaled below
    let mol = Molecule::new(vec![
        Atom {
            z: 6,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(a, a, a),
        },
        Atom {
            z: 1,
            position: Vec3::new(a, -a, -a),
        },
        Atom {
            z: 1,
            position: Vec3::new(-a, a, -a),
        },
        Atom {
            z: 1,
            position: Vec3::new(-a, -a, a),
        },
    ]);
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let opts = tight_options();

    // --- Hartree-Fock: exact up to the SCF threshold ---
    let hf = dft_rs::xc::resolve(&dft_rs::xc::FunctionalSpec::Combined("hf".into())).unwrap();
    let scf = run_scf(&mol, &mb, &opts, &hf, None).unwrap();
    let eps = &scf.mo_energy[0];
    // STO-3G methane: 1a1 (C 1s), 2a1, then the 1t2 triple. Five occupied.
    assert_eq!(
        scf.n_alpha, 5,
        "methane/STO-3G should have 5 occupied orbitals"
    );
    let t2 = [eps[2], eps[3], eps[4]];
    let spread =
        t2.iter().cloned().fold(f64::MIN, f64::max) - t2.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        spread < 1e-9,
        "HF: the t2 orbitals of Td methane are split by {spread:e} ({t2:?}). Nothing tells \
         this code about symmetry, so the degeneracy is emergent -- a split means the \
         angular integrals or the contraction normalisation are not consistent across \
         the three Cartesian directions."
    );

    // The dipole of a Td molecule vanishes by symmetry.
    let mu = dft_rs::properties::dipole_moment(&mol, &mb, &scf);
    assert!(
        mu.norm() < 1e-9,
        "HF: Td methane has dipole {mu:?} (|mu| = {}), but symmetry forbids it",
        mu.norm()
    );

    // --- DFT: split by angular quadrature, must shrink with refinement ---
    let svwn = dft_rs::xc::resolve(&dft_rs::xc::FunctionalSpec::Combined("svwn5".into())).unwrap();
    let mut splits = Vec::new();
    for level in [1usize, 3, 5] {
        let gopts = dft_rs::grid::GridOptions {
            level,
            ..dft_rs::grid::GridOptions::default()
        };
        let grid = dft_rs::grid::MolecularGrid::build(&mol, &gopts).unwrap();
        let s = run_scf(&mol, &mb, &opts, &svwn, Some(&grid)).unwrap();
        let e = &s.mo_energy[0];
        let t = [e[2], e[3], e[4]];
        splits.push(
            t.iter().cloned().fold(f64::MIN, f64::max) - t.iter().cloned().fold(f64::MAX, f64::min),
        );
    }
    assert!(
        splits[2] < splits[0],
        "SVWN5: the t2 splitting did not shrink with grid level: {splits:?}. The product \
         angular grid is not invariant under the Td operations, so a residual split is \
         expected -- but it must be quadrature truncation, which refines away, rather \
         than a fixed error, which does not."
    );
    assert!(
        splits[2] < 1e-5,
        "SVWN5: t2 still split by {} at grid level 5",
        splits[2]
    );
}
/// Structural properties of the matrices an SCF hands back.
///
/// Classification: **exact** throughout. These are properties of the objects
/// themselves, not of any limit, so a violation is a construction bug.
///
/// Individually each is almost trivial; collectively they close a gap. On the
/// PERIODIC side this codebase gates hermiticity and positive-definiteness
/// thoroughly (`bloch_matrices_are_hermitian_and_reduce_at_gamma`,
/// `overlap_is_positive_definite_across_the_brillouin_zone`, the image-block
/// relations). On the MOLECULAR side there were exactly two fragments: the
/// polarizability tensor's symmetry and the antisymmetry of `dP/dB`. Nothing
/// checked that the overlap is positive definite, that the Fock and density
/// matrices are symmetric, or that the polarizability is positive definite.
///
/// The overlap one is not academic. `S` is what
/// `Orthogonalizer::canonical` decomposes, and a non-positive eigenvalue
/// there is the difference between "this basis is nearly linearly dependent"
/// -- handled, with a warning -- and "the overlap was built wrong", which
/// would otherwise surface much later as an unexplained SCF failure.
#[test]
fn scf_matrices_have_the_structure_their_definitions_require() {
    let mol = water(|r| r);
    let set = data::load("6-31g*").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let opts = tight_options();
    let scf = run_hf(&mol, &mb, &opts).unwrap();

    // S is a Gram matrix of linearly independent functions: symmetric and
    // positive definite.
    let n = scf.overlap.rows;
    let mut s_asym = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            s_asym = s_asym.max((scf.overlap[(i, j)] - scf.overlap[(j, i)]).abs());
        }
    }
    assert!(
        s_asym < 1e-14,
        "overlap is not symmetric: max|S - S^T| = {s_asym:e}"
    );
    let (s_eig, _) = dft_rs::linalg::symmetric_eigen(&scf.overlap).unwrap();
    let s_min = s_eig.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        s_min > 0.0,
        "the overlap matrix has a non-positive eigenvalue ({s_min:e}); it is a Gram \
         matrix, so this is a construction error rather than linear dependence -- \
         near-dependence gives a SMALL positive eigenvalue, which the canonical \
         orthogonalizer already handles"
    );

    // F and P are symmetric per spin.
    for spin in 0..2 {
        for (label, m) in [("fock", &scf.fock[spin]), ("density", &scf.density[spin])] {
            let mut worst = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    worst = worst.max((m[(i, j)] - m[(j, i)]).abs());
                }
            }
            assert!(
                worst < 1e-12,
                "spin-{spin} {label} matrix is not symmetric: max|A - A^T| = {worst:e}"
            );
        }
    }

    // The static polarizability of a stable closed-shell ground state is
    // symmetric AND positive definite: applying a field can only lower the
    // energy, so the response to it cannot be negative in any direction.
    let alpha = dft_rs::response::polarizability::polarizability(
        &mol,
        &mb,
        &dft_rs::xc::resolve(&dft_rs::xc::FunctionalSpec::Combined("hf".into())).unwrap(),
        None,
        &scf,
        &opts,
    )
    .unwrap();
    let mut a = dft_rs::linalg::Matrix::zeros(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            a[(i, j)] = alpha[i][j];
        }
    }
    let mut a_asym = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            a_asym = a_asym.max((a[(i, j)] - a[(j, i)]).abs());
        }
    }
    assert!(a_asym < 1e-6, "polarizability is not symmetric: {a_asym:e}");
    let (a_eig, _) = dft_rs::linalg::symmetric_eigen(&a).unwrap();
    let a_min = a_eig.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        a_min > 0.0,
        "the polarizability tensor has a non-positive eigenvalue ({a_min:e}); a stable \
         ground state cannot be destabilised by an applied field, so every principal \
         component must be positive"
    );
}
