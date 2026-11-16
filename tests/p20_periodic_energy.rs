// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Physical gates on the periodic TOTAL ENERGY and its forces.
//!
//! The periodic module already pins each operator against an identity of its
//! own (Madelung constants, splitting-parameter independence, finite
//! differences at a generic `k`). What those cannot see is an error that
//! lives in the *assembly*: a background convention counted twice, a
//! Brillouin-zone weight that is right for one mesh only, a Pulay term that
//! happens to vanish for the symmetric geometries used to test the pieces.
//!
//! The three gates here are statements about the total energy that have to
//! hold no matter how the code is organized:
//!
//! 1. a neutral cell made large enough is the isolated molecule,
//! 2. a supercell is exactly as many primitive cells as it contains, once the
//!    k mesh is folded to match,
//! 3. the forces of a translationally invariant lattice sum to zero, and the
//!    energy does not move when the cell contents are translated.
//!
//! None uses a frozen number from another program — there is no periodic
//! reference implementation this project is allowed to diff against — and
//! none can be satisfied by a code that merely reproduces its own
//! conventions.
//!
//! **On cost, and why these are `#[ignore]`d.** Each of these runs several
//! full periodic SCFs, and a periodic SCF is minutes rather than seconds
//! even for two hydrogens in a minimal basis: the Ewald-split Coulomb
//! operator is rebuilt over its own real-space and reciprocal shells at
//! every k point and every iteration, and that cost is set by the lattice,
//! not by the number of basis functions. They therefore follow the same
//! convention as the existing periodic force gates in `periodic::forces` —
//! excluded from the default run and executed explicitly:
//!
//! ```console
//! cargo test --release --test p20_periodic_energy -- --ignored
//! ```
//!
//! Measured on a 16-thread desktop: roughly ten minutes each.
//!
//! Within that budget the settings are still chosen for economy. The first
//! two tests use Hartree-Fock, so there is no XC grid at all and what is
//! under test — Bloch phases, Brillouin-zone weights, the electrostatic
//! conventions, finite-k exact exchange — is tested without paying for
//! quadrature; the supercell invariance of the XC grid itself is gated
//! separately, inside `periodic::grid`. The third needs a functional and
//! uses the deliberately coarse grid the other periodic force gates use,
//! because the sum rule and translation invariance are exact at any grid
//! resolution: the Becke weights are a partition of unity at every level.

use dft_rs::basis::data;
use dft_rs::basis::MolBasis;
use dft_rs::grid::GridOptions;
use dft_rs::math::Vec3;
use dft_rs::periodic::forces::analytic_forces;
use dft_rs::periodic::scf::{run_periodic_scf, PeriodicScfOptions};
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::scf::{run_scf, EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

/// Coarse but unpruned: the identities under test do not improve with grid
/// resolution, so resolution is only cost here.
fn coarse_grid() -> GridOptions {
    GridOptions {
        radial_override: Some(9),
        angular_override: Some(7),
        prune: false,
        ..GridOptions::default()
    }
}

fn hydrogen_molecule() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, -0.7),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 0.7),
        },
    ])
}

/// A neutral cell, made large enough, must reproduce the isolated molecule.
///
/// This is the sharpest single test of the electrostatic conventions,
/// because the periodic total energy is assembled from pieces that
/// individually have no molecular limit at all. `E_nuc` carries a
/// neutralising background and stays far from the molecular nuclear
/// repulsion; `V(k)` carries a `G = 0` constant times the overlap. Only the
/// sum is physical, and it reaches the molecular energy only if every one of
/// those conventions is consistent with the others.
///
/// The approach is checked, not just the endpoint. H2 is neutral and has no
/// dipole, so the leading interaction with its own images is
/// quadrupole-quadrupole and the residual must fall steeply with cell size.
/// A convention error would instead leave a size-INDEPENDENT offset, which a
/// single-cell comparison could not distinguish from slow convergence
/// however tight its tolerance.
#[test]
#[ignore = "three periodic SCFs over a cell-size series; run explicitly with --ignored"]
fn neutral_cell_converges_to_the_molecular_energy() {
    let molecule = hydrogen_molecule();
    let set = data::load("sto-3g").unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();

    let mb = MolBasis::build(&molecule, &set).unwrap();
    let molecular = run_scf(
        &molecule,
        &mb,
        &ScfOptions {
            e_tol: 1e-12,
            p_tol: 1e-10,
            eri_mode: EriMode::InCore { limit_mb: 256 },
            ..ScfOptions::default()
        },
        &functional,
        None,
    )
    .unwrap()
    .energy;

    let options = PeriodicScfOptions {
        e_tol: 1e-11,
        p_tol: 1e-9,
        ..PeriodicScfOptions::default()
    };
    let mut errors = Vec::new();
    for edge in [10.0, 14.0, 18.0] {
        // The cutoff stays below every edge, so the AO image lists are the
        // home cell alone and the residual is purely electrostatic — which
        // is precisely the part whose conventions this test exists to check.
        let cell = Cell::new(molecule.clone(), Lattice::cubic(edge), &set, 8.0).unwrap();
        let result = run_periodic_scf(&cell, &functional, &options).unwrap();
        assert!(
            result.converged,
            "periodic SCF did not converge at L={edge}"
        );
        errors.push((edge, result.energy - molecular));
    }

    let (largest_edge, largest_error) = *errors.last().unwrap();
    assert!(
        largest_error.abs() < 1e-4,
        "L={largest_edge} cell is {largest_error:e} Ha from the molecular energy \
         {molecular}; the full series was {errors:?}"
    );
    for pair in errors.windows(2) {
        assert!(
            pair[1].1.abs() < 0.6 * pair[0].1.abs(),
            "the finite-cell residual is not decaying with cell size, which is \
             what a mismatched background convention looks like: {errors:?}"
        );
    }
}

/// A supercell contains an integer number of primitive cells, so its energy
/// is that multiple of the primitive energy — provided the k mesh is folded
/// by the same factor, since a supercell's Brillouin zone is correspondingly
/// smaller.
///
/// This is what ties the finite-k machinery together. The two calculations
/// visit different code paths throughout: different numbers of k points,
/// different image-block counts, different matrix dimensions. Nothing but a
/// correct Bloch phase, a correct Brillouin-zone weight and a correct
/// inverse transform makes them agree, and a mesh-dependent weighting error
/// — invisible to any single-mesh test — shows up here immediately.
///
/// Hartree-Fock, so that finite-k EXACT EXCHANGE is included: of the pieces
/// involved it is the one most easily written in a way that is right at
/// Gamma and wrong elsewhere. The XC grid's own supercell invariance is
/// gated separately, in `periodic::grid`.
///
/// The cutoff has to exceed the SUPERCELL lattice vector, not the primitive
/// one. Otherwise the two calculations do not describe the same physical
/// neighbour set: a pair the primitive reaches through its `a` translation
/// sits in the supercell's `2a` translation, and a cutoff between the two
/// silently keeps it in one calculation and drops it from the other.
#[test]
#[ignore = "primitive and doubled-cell periodic SCFs; run explicitly with --ignored"]
fn supercell_energy_is_extensive_under_band_folding() {
    let spacing = 5.5;
    let vacuum = 13.0;
    let cutoff = 12.0;
    let set = data::load("sto-3g").unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();

    let primitive_lattice = Lattice::new(
        Vec3::new(spacing, 0.0, 0.0),
        Vec3::new(0.0, vacuum, 0.0),
        Vec3::new(0.0, 0.0, vacuum),
    );
    let primitive_molecule = Molecule::new(vec![Atom {
        z: 2,
        position: Vec3::zero(),
    }]);
    let doubled_molecule = Molecule::new(vec![
        Atom {
            z: 2,
            position: Vec3::zero(),
        },
        Atom {
            z: 2,
            position: Vec3::new(spacing, 0.0, 0.0),
        },
    ]);

    let base = PeriodicScfOptions {
        e_tol: 1e-11,
        p_tol: 1e-9,
        exchange_cutoff: Some(cutoff),
        ..PeriodicScfOptions::default()
    };

    let primitive_cell =
        Cell::new(primitive_molecule, primitive_lattice.clone(), &set, cutoff).unwrap();
    let primitive = run_periodic_scf(
        &primitive_cell,
        &functional,
        &PeriodicScfOptions {
            k_mesh: [4, 1, 1],
            ..base.clone()
        },
    )
    .unwrap();

    let doubled_cell = Cell::new(
        doubled_molecule,
        primitive_lattice.supercell([2, 1, 1]),
        &set,
        cutoff,
    )
    .unwrap();
    let doubled = run_periodic_scf(
        &doubled_cell,
        &functional,
        &PeriodicScfOptions {
            k_mesh: [2, 1, 1],
            ..base
        },
    )
    .unwrap();

    assert!(
        primitive.converged && doubled.converged,
        "periodic SCF did not converge"
    );
    let difference = doubled.energy - 2.0 * primitive.energy;
    assert!(
        difference.abs() < 5e-8,
        "supercell energy {} is not twice the primitive energy {} \
         (difference {difference:e} Ha)",
        doubled.energy,
        primitive.energy
    );
}

/// Two consequences of the same symmetry, on a cell with genuinely non-zero
/// forces.
///
/// A lattice sum is invariant under translating everything it contains, so
/// (a) the total energy cannot depend on where the contents sit relative to
/// the cell origin, and (b) the forces must sum to zero. Part (b) is the
/// stronger of the two: every term of the periodic force — Pulay, Ewald,
/// Hartree, XC — contributes to that sum, and a term differentiated with
/// respect to the wrong centre survives the existing finite-difference gates
/// (which displace one atom at a time) while breaking this one.
///
/// The geometry is deliberately asymmetric so that the individual forces are
/// large. A symmetric cell would satisfy the sum rule for the trivial reason
/// that every force in it is separately zero.
#[test]
#[ignore = "two periodic SCFs plus an analytic force build; run explicitly with --ignored"]
fn periodic_forces_sum_to_zero_and_the_energy_is_translation_invariant() {
    let set = data::load("sto-3g").unwrap();
    let functional = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    let lattice = Lattice::new(
        Vec3::new(6.0, 0.0, 0.0),
        Vec3::new(0.0, 12.0, 0.0),
        Vec3::new(0.0, 0.0, 12.0),
    );
    let molecule = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.13, -0.21, 0.09),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.71, 0.34, -0.17),
        },
    ]);
    let options = PeriodicScfOptions {
        k_mesh: [2, 1, 1],
        e_tol: 1e-11,
        p_tol: 1e-9,
        xc_ao_cutoff: Some(6.1),
        grid_shell: 1,
        grid: coarse_grid(),
        ..PeriodicScfOptions::default()
    };

    let cell = Cell::new(molecule.clone(), lattice.clone(), &set, 6.1).unwrap();
    let result = run_periodic_scf(&cell, &functional, &options).unwrap();
    assert!(result.converged);
    let forces = analytic_forces(&cell, &functional, &result, &options).unwrap();

    let magnitude = forces
        .total
        .iter()
        .map(|f| f.norm())
        .fold(0.0_f64, f64::max);
    assert!(
        magnitude > 1e-3,
        "the test geometry is too close to equilibrium to say anything: |F|max = {magnitude:e}"
    );
    let sum = forces
        .total
        .iter()
        .fold(Vec3::zero(), |total, f| total + *f);
    assert!(
        sum.norm() < 1e-7 * magnitude.max(1.0) + 1e-9,
        "periodic forces do not sum to zero: {sum:?} against |F|max = {magnitude:e}"
    );

    // The same invariance seen from the energy side. The shift is a generic
    // fraction of no lattice vector, so no image-list symmetry absorbs it,
    // and it is large enough to move both atoms out of the home cell.
    let shift = Vec3::new(2.37, -1.14, 0.83);
    let mut translated = molecule;
    for atom in &mut translated.atoms {
        atom.position += shift;
    }
    let translated_cell = Cell::new(translated, lattice, &set, 6.1).unwrap();
    let translated_result = run_periodic_scf(&translated_cell, &functional, &options).unwrap();
    assert!(
        (translated_result.energy - result.energy).abs() < 2e-9,
        "translating the cell contents changed the energy: {} vs {}",
        translated_result.energy,
        result.energy
    );
}
