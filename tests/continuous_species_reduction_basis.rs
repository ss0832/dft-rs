// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Fixed one-center AO alternatives and reference occupied-space capture.
pub use dft_rs::{
    basis, grid, integrals, ks, linalg, math, properties, response, scf, system, xc, DftError,
    Result,
};
#[path = "../src/pseudo_sub/reduction_basis.rs"]
pub mod reduction_basis;
pub use dft_rs::pseudo_sub::reduction_reference;

use basis::{data, MolBasis};
use grid::{GridOptions, MolecularGrid};
use integrals::overlap_matrix;
use linalg::Matrix;
use math::Vec3;
use reduction_basis::*;
use reduction_reference::{extract_methyl_frozen_space, ExtractionOptions, MethylBoundary};
use scf::{run_scf, EriMode, Guess, ScfOptions};
use system::{Atom, Molecule};
use xc::{resolve, FunctionalSpec};

fn full_molecule() -> Molecule {
    let points = [
        [-1.46, -0.14, 0.23],
        [-2.13, 1.69, 0.68],
        [-2.28, -1.57, 1.31],
        [-2.19, -0.43, -1.65],
        [1.35, 0.17, -0.12],
        [2.05, 2.08, 0.16],
        [2.10, -0.92, 1.45],
        [2.01, -0.84, -1.67],
    ];
    Molecule::new(
        points
            .iter()
            .enumerate()
            .map(|(i, p)| Atom {
                z: if i == 0 || i == 4 { 6 } else { 1 },
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    )
}

fn reduced_molecule() -> Molecule {
    let full = full_molecule();
    let mut atoms = full.atoms[..4].to_vec();
    atoms.push(Atom {
        z: 1,
        position: full.atoms[4].position,
    });
    Molecule::new(atoms)
}

#[test]
fn fixed_valence_shells_order_labels_and_host_identity() {
    let reduced = reduced_molecule();
    let common = data::load("sto-3g").unwrap();
    let old = build_reduction_basis(&reduced, 4, OneCenterBasisTemplate::Hydrogen1s).unwrap();
    let new = build_reduction_basis(&reduced, 4, OneCenterBasisTemplate::CarbonValenceSp).unwrap();
    let ordinary = MolBasis::build(&reduced, &common).unwrap();
    assert_eq!((old.nao_sph, new.nao_sph), (9, 12));
    assert_eq!((old.nao_cart, new.nao_cart), (9, 12));
    assert!(overlap_matrix(&old).max_abs_diff(&overlap_matrix(&ordinary)) < 1e-14);
    let old_host: Vec<_> = old.shells.iter().filter(|s| s.atom != 4).collect();
    let new_host: Vec<_> = new.shells.iter().filter(|s| s.atom != 4).collect();
    assert_eq!(old_host.len(), new_host.len());
    for (a, b) in old_host.iter().zip(&new_host) {
        assert_eq!(
            (a.l, a.atom, a.sph_off, a.cart_off),
            (b.l, b.atom, b.sph_off, b.cart_off)
        );
        assert_eq!(a.exps, b.exps);
        assert_eq!(a.coefs, b.coefs);
        assert_eq!(a.center, b.center);
    }
    let site: Vec<_> = new.shells.iter().filter(|s| s.atom == 4).collect();
    assert_eq!(site.len(), 2);
    assert_eq!(
        (site[0].l, site[0].sph_off, site[1].l, site[1].sph_off),
        (0, 8, 1, 9)
    );
    let placed_carbon = MolBasis::build(
        &Molecule::new(vec![Atom {
            z: 6,
            position: reduced.atoms[4].position,
        }]),
        &common,
    )
    .unwrap();
    for (i, shell) in site.iter().enumerate() {
        assert_eq!(shell.exps, placed_carbon.shells[1 + i].exps);
        assert_eq!(shell.coefs, placed_carbon.shells[1 + i].coefs);
        assert_eq!(shell.center, reduced.atoms[4].position);
    }
    assert_ne!(site[0].exps, placed_carbon.shells[0].exps);
    assert_eq!(reduced.atoms[4].z, 1);
    for template in [
        OneCenterBasisTemplate::Hydrogen1s,
        OneCenterBasisTemplate::CarbonValenceSp,
        OneCenterBasisTemplate::CarbonCoreValenceSp,
    ] {
        assert_eq!(
            OneCenterBasisTemplate::from_id(template.id()).unwrap(),
            template
        );
        assert_eq!(fixed_site_basis(template).unwrap().z, 1);
    }
    assert_ne!(H1S_TEMPLATE_ID, VALENCE_SP_TEMPLATE_ID);
    assert_ne!(CORE_VALENCE_SP_TEMPLATE_ID, H1S_TEMPLATE_ID);
    assert_ne!(CORE_VALENCE_SP_TEMPLATE_ID, VALENCE_SP_TEMPLATE_ID);
    assert_eq!(
        (ELECTRONIC_OPERATOR_COEFFICIENTS, ELECTROSTATIC_COEFFICIENTS),
        (39, 12)
    );
    assert!(OneCenterBasisTemplate::from_id("unversioned-one-center").is_err());
    assert!(build_reduction_basis(&reduced, 0, OneCenterBasisTemplate::CarbonValenceSp).is_err());
    assert!(build_reduction_basis(&reduced, 5, OneCenterBasisTemplate::CarbonValenceSp).is_err());
}

#[test]
fn fixed_core_valence_shells_preserve_labels_hosts_and_electron_ledger() {
    let reduced = reduced_molecule();
    let common = data::load("sto-3g").unwrap();
    let h1s = build_reduction_basis(&reduced, 4, OneCenterBasisTemplate::Hydrogen1s).unwrap();
    let core_sp =
        build_reduction_basis(&reduced, 4, OneCenterBasisTemplate::CarbonCoreValenceSp).unwrap();
    assert_eq!((core_sp.nao_sph, core_sp.nao_cart), (13, 13));
    let old_host: Vec<_> = h1s.shells.iter().filter(|s| s.atom != 4).collect();
    let host: Vec<_> = core_sp.shells.iter().filter(|s| s.atom != 4).collect();
    assert_eq!(old_host.len(), host.len());
    for (a, b) in old_host.iter().zip(host) {
        assert_eq!(
            (a.l, a.atom, a.sph_off, a.cart_off),
            (b.l, b.atom, b.sph_off, b.cart_off)
        );
        assert_eq!(a.exps, b.exps);
        assert_eq!(a.coefs, b.coefs);
        assert_eq!(a.center, b.center);
    }
    let site: Vec<_> = core_sp.shells.iter().filter(|s| s.atom == 4).collect();
    let carbon = MolBasis::build(
        &Molecule::new(vec![Atom {
            z: 6,
            position: reduced.atoms[4].position,
        }]),
        &common,
    )
    .unwrap();
    assert_eq!(
        site.iter()
            .map(|s| (s.l, s.sph_off, s.cart_off))
            .collect::<Vec<_>>(),
        [(0, 8, 8), (0, 9, 9), (1, 10, 10)]
    );
    for (a, b) in site.iter().zip(&carbon.shells) {
        assert_eq!(a.exps, b.exps);
        assert_eq!(a.coefs, b.coefs);
        assert_eq!(a.center, b.center);
    }
    assert_eq!(
        fixed_site_basis(OneCenterBasisTemplate::CarbonCoreValenceSp)
            .unwrap()
            .z,
        1
    );
    assert_eq!(reduced.atoms[4].z, 1);
    // Select occupancy from the explicit frozen-electron ledger, never from
    // the carbon origin of a contraction or the number of retained AOs.
    use dft_rs::driver::continuous_species::reduction_linear::{
        ElectronLedger, FrozenFragmentSpec,
    };
    let full = full_molecule();
    let ledger = ElectronLedger::new(
        &full.atoms.iter().map(|a| a.z).collect::<Vec<_>>(),
        18,
        &[FrozenFragmentSpec {
            explicit_atoms: vec![4, 5, 6, 7],
            frozen_electrons: 8,
        }],
    )
    .unwrap();
    assert_eq!(
        (
            ledger.full_electron_count,
            ledger.frozen_electron_count,
            ledger.active_electron_count
        ),
        (18, 8, 10)
    );
    assert_eq!(
        (
            ledger.fragments[0].effective_ionic_charge,
            ledger.reduced_total_charge
        ),
        (1, 0)
    );
    assert_eq!(
        (ELECTRONIC_OPERATOR_COEFFICIENTS, ELECTROSTATIC_COEFFICIENTS),
        (39, 12)
    );
    let options = dft_rs::pseudo_sub::reduction::GenerationOptions::default();
    assert_eq!(options.basis_template.id(), H1S_TEMPLATE_ID);
}

#[test]
fn overlap_metric_and_cross_basis_reciprocity() {
    let reduced = reduced_molecule();
    let old = build_reduction_basis(&reduced, 4, OneCenterBasisTemplate::Hydrogen1s).unwrap();
    let new = build_reduction_basis(&reduced, 4, OneCenterBasisTemplate::CarbonValenceSp).unwrap();
    let core =
        build_reduction_basis(&reduced, 4, OneCenterBasisTemplate::CarbonCoreValenceSp).unwrap();
    for basis in [&old, &new, &core] {
        let condition = overlap_condition(basis, 1e-12).unwrap();
        assert_eq!(condition.ao_count, basis.nao_sph);
        assert!(condition.minimum_eigenvalue > 0.0);
        assert!(condition.maximum_eigenvalue.is_finite());
        assert!(condition.condition_number.is_finite() && condition.condition_number >= 1.0);
        assert_eq!(
            condition.condition_number,
            condition.maximum_eigenvalue / condition.minimum_eigenvalue
        );
        assert!(
            cross_basis_overlap(basis, basis)
                .unwrap()
                .max_abs_diff(&overlap_matrix(basis))
                < 1e-14
        );
    }
    let cross = cross_basis_overlap(&old, &new).unwrap();
    let reverse = cross_basis_overlap(&new, &old).unwrap();
    assert_eq!((cross.rows, cross.cols), (9, 12));
    assert!(cross.max_abs_diff(&reverse.transpose()) < 1e-14);
    let cross_core = cross_basis_overlap(&new, &core).unwrap();
    let reverse_core = cross_basis_overlap(&core, &new).unwrap();
    assert_eq!((cross_core.rows, cross_core.cols), (12, 13));
    assert!(cross_core.max_abs_diff(&reverse_core.transpose()) < 1e-14);
    let atom = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::new(0.31, -0.27, 0.43),
    }]);
    let atom_basis =
        build_reduction_basis(&atom, 0, OneCenterBasisTemplate::CarbonValenceSp).unwrap();
    assert!(overlap_matrix(&atom_basis).max_abs_diff(&Matrix::identity(4)) < 1e-13);
    assert!(overlap_condition(&new, 0.0).is_err());
}

#[test]
fn canonical_active_occupied_capture_compares_h1s_and_fixed_valence_sp() {
    let molecule = full_molecule();
    let reference_basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
    let grid = MolecularGrid::build(
        &molecule,
        &GridOptions {
            level: 1,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let options = ScfOptions {
        guess: Guess::Core,
        max_iter: 300,
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-11,
        eri_mode: EriMode::InCore { limit_mb: 64 },
        ..ScfOptions::default()
    };
    let functional = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let reference = run_scf(
        &molecule,
        &reference_basis,
        &options,
        &functional,
        Some(&grid),
    )
    .unwrap();
    let selection = extract_methyl_frozen_space(
        &molecule,
        &reference_basis,
        &reference,
        MethylBoundary {
            carbon: 4,
            hydrogens: [5, 6, 7],
            port_atom: 0,
        },
        &ExtractionOptions::default(),
    )
    .unwrap();
    assert_eq!(
        (
            selection.full_electrons,
            selection.frozen_electrons,
            selection.active_electrons
        ),
        (18, 8, 10)
    );
    assert!(!selection.diagnostics.ambiguous);
    let occupied = &selection.active_reference_orbitals;
    assert_eq!((occupied.rows, occupied.cols), (16, 5));
    let reduced = reduced_molecule();
    let old = build_reduction_basis(&reduced, 4, OneCenterBasisTemplate::Hydrogen1s).unwrap();
    let new = build_reduction_basis(&reduced, 4, OneCenterBasisTemplate::CarbonValenceSp).unwrap();
    let core =
        build_reduction_basis(&reduced, 4, OneCenterBasisTemplate::CarbonCoreValenceSp).unwrap();
    let old_capture =
        closed_shell_occupied_capture(&reference_basis, occupied, &old, 1e-12).unwrap();
    let new_capture =
        closed_shell_occupied_capture(&reference_basis, occupied, &new, 1e-12).unwrap();
    let core_capture =
        closed_shell_occupied_capture(&reference_basis, occupied, &core, 1e-12).unwrap();
    let complete =
        closed_shell_occupied_capture(&reference_basis, occupied, &reference_basis, 1e-12).unwrap();
    assert!(complete.uncaptured_electrons.abs() < 1e-10);
    assert!(complete.minimum_squared_capture > 1.0 - 1e-10);
    assert!(complete.projected_orbitals.max_abs_diff(occupied) < 1e-11);
    for (template, capture) in [
        (OneCenterBasisTemplate::Hydrogen1s, &old_capture),
        (OneCenterBasisTemplate::CarbonValenceSp, &new_capture),
        (OneCenterBasisTemplate::CarbonCoreValenceSp, &core_capture),
    ] {
        assert_eq!(
            (capture.spatial_orbitals, capture.active_electrons),
            (5, 10)
        );
        assert_eq!(capture.squared_principal_captures.len(), 5);
        assert_eq!(
            (
                capture.projected_orbitals.rows,
                capture.projected_orbitals.cols
            ),
            (capture.overlap.ao_count, 5)
        );
        assert!(capture.projection_relative_residual < 1e-12);
        assert!(capture.reference_metric_defect < 1e-9);
        assert!(
            capture
                .projected_metric
                .max_abs_diff(&capture.projected_metric.transpose())
                < 1e-12
        );
        assert!((capture.captured_electrons + capture.uncaptured_electrons - 10.0).abs() < 1e-12);
        assert!(capture
            .squared_principal_captures
            .iter()
            .all(|&v| (-1e-9..=1.0 + 1e-9).contains(&v)));
        println!("fixed common AO capture template={} fullAO=16 reducedAO={} centerAO={} activeNe={} capturedNe={:.16e} lostNe={:.16e} minimumSquaredCapture={:.16e} spectrum={:?} minS={:.16e} maxS={:.16e} conditionS={:.16e} equationResidual={:.9e} referenceMetricDefect={:.9e} electronicCoefficients={} sigmaCoefficients={} chemicalAccuracyStatus=not-established",
            template.id(),capture.overlap.ao_count,template.site_ao_count(),capture.active_electrons,
            capture.captured_electrons,capture.uncaptured_electrons,capture.minimum_squared_capture,
            capture.squared_principal_captures,capture.overlap.minimum_eigenvalue,capture.overlap.maximum_eigenvalue,
            capture.overlap.condition_number,capture.projection_relative_residual,capture.reference_metric_defect,
            ELECTRONIC_OPERATOR_COEFFICIENTS,ELECTROSTATIC_COEFFICIENTS);
    }
    // Captures belong to the occupied subspace, not to numbered or signed MOs.
    let mut rotation = Matrix::identity(5);
    let angle = 0.37_f64;
    rotation[(0, 0)] = angle.cos();
    rotation[(0, 1)] = -angle.sin();
    rotation[(1, 0)] = angle.sin();
    rotation[(1, 1)] = angle.cos();
    let rotated = closed_shell_occupied_capture(
        &reference_basis,
        &occupied.matmul_seq(&rotation),
        &new,
        1e-12,
    )
    .unwrap();
    assert!((rotated.captured_electrons - new_capture.captured_electrons).abs() < 1e-12);
    for (a, b) in rotated
        .squared_principal_captures
        .iter()
        .zip(&new_capture.squared_principal_captures)
    {
        assert!((a - b).abs() < 1e-12);
    }
    println!("fixed template capture comparison lostElectronChange={:.16e} minimumSquaredCaptureChange={:.16e} AOChange=3 noBasisWidthFit=true noCoefficientDimensionChange=true",new_capture.uncaptured_electrons-old_capture.uncaptured_electrons,new_capture.minimum_squared_capture-old_capture.minimum_squared_capture);
    println!("core-s basis capture comparison lostElectronChange={:.16e} minimumSquaredCaptureChange={:.16e} AOChange=1 frozenNe=8 activeNe=10 ionicCharge=1 noBasisWidthFit=true noCoefficientDimensionChange=true",core_capture.uncaptured_electrons-new_capture.uncaptured_electrons,core_capture.minimum_squared_capture-new_capture.minimum_squared_capture);
    // No claim that improving this projection establishes a good bond curve.
}
