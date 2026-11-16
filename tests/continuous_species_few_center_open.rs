// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Candidate-only transfer of immutable closed-shell reference coefficients to
//! fixed-spin reduced UKS. This is not a localized-reference projection or a
//! chemical-accuracy, periodic or open-frozen-core acceptance claim.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/reduced_fragment.rs"]
pub mod fixture;

use dft_rs::driver::continuous_species::few_center::{
    evaluate, fixed_dictionary_id, FewCenterEvaluation, FixedFewCenterCoefficients,
    PRIMITIVE_FITTED_OPERATOR_TEMPLATE_ID,
};
use dft_rs::driver::continuous_species::reduction_linear::{ElectronLedger, FrozenFragmentSpec};
use dft_rs::math::Vec3;
use dft_rs::pseudo_sub::reduction::{generate_methyl, ProjectionRule, ReducedFragment};
use dft_rs::pseudo_sub::reduction_basis::OneCenterBasisTemplate;
use dft_rs::pseudo_sub::reduction_geometry::RigidExpansion;
use dft_rs::pseudo_sub::reduction_roles::CenterTemplate;
use dft_rs::response::ResponseSolverConfig;
use dft_rs::system::Molecule;
use dft_rs::Result;
use std::collections::BTreeMap;
use std::sync::OnceLock;

fn field() -> Vec3 {
    Vec3::new(0.0013, -0.0009, 0.0017)
}
fn origin() -> Vec3 {
    Vec3::new(0.11, 0.07, -0.09)
}
struct Reference {
    model: ReducedFragment,
    coefficients: FixedFewCenterCoefficients,
    legacy_json: String,
}
fn reference() -> &'static Reference {
    static REFERENCE: OnceLock<Reference> = OnceLock::new();
    REFERENCE.get_or_init(|| {
        let (source, spec) = fixture::ethane();
        let mut options = fixture::construction_options();
        options.basis_template = OneCenterBasisTemplate::CarbonCoreValenceSp;
        options.projection_rule = ProjectionRule::PhysicalSupportPauliNormalized;
        let model = generate_methyl(&source, spec, options).unwrap();
        assert_eq!(model.construction_fingerprint, "fnv1a64:5fbcbd6310839430");
        assert_eq!(
            (
                model.options.electrostatic_points.len(),
                model.options.electrostatic_weight
            ),
            (47, 1e6)
        );
        let coefficients = FixedFewCenterCoefficients {
            template: CenterTemplate::CoreSp,
            operator_template_id: PRIMITIVE_FITTED_OPERATOR_TEMPLATE_ID.into(),
            functional_id: model.options.functional.clone(),
            dictionary_id: fixed_dictionary_id(&model.options).unwrap(),
            gaussian_charges: model.gaussian_charges,
            gaussian_body_dipoles: std::array::from_fn(|i| {
                let p = model.gaussian_body_dipoles[i];
                Vec3::new(p[0], p[1], p[2])
            }),
            local_amplitudes: model.local_amplitudes,
            nonlocal_coefficients: model.nonlocal_coefficients,
            internal_energy_reference: model.internal_energy_reference,
            reference_id: format!(
                "baseline05-spin-transfer-not-localized-fit:{}",
                model.construction_fingerprint
            ),
        };
        coefficients.validate().unwrap();
        let legacy_json = model.to_json().unwrap();
        Reference {
            model,
            coefficients,
            legacy_json,
        }
    })
}
struct System {
    source: Molecule,
    map: RigidExpansion,
    ledger: ElectronLedger,
}
fn system(charge: i32, multiplicity: usize) -> System {
    let r = reference();
    let mut source = r.model.source.clone();
    source.charge = charge;
    source.multiplicity = multiplicity;
    let map = RigidExpansion::methyl_with_internal_offsets(
        &source,
        r.model.spec.clone(),
        r.model.geometry_map.internal_offsets(),
    )
    .unwrap();
    let ledger = ElectronLedger::new(
        &source.atoms.iter().map(|a| a.z).collect::<Vec<_>>(),
        source.n_electrons().unwrap(),
        &[FrozenFragmentSpec {
            explicit_atoms: r.model.spec.explicit_atoms.clone(),
            frozen_electrons: 8,
        }],
    )
    .unwrap();
    System {
        source,
        map,
        ledger,
    }
}
fn run(
    s: &System,
    y: &[Vec3],
    f: Vec3,
    o: Vec3,
    spins: Option<[usize; 2]>,
    response: bool,
) -> Result<FewCenterEvaluation> {
    let config = ResponseSolverConfig::default();
    evaluate(
        &s.source,
        &s.map,
        &s.ledger,
        CenterTemplate::CoreSp,
        &reference().model.options,
        &reference().coefficients,
        y,
        f,
        o,
        spins,
        response.then_some(&config),
    )
}
fn unchanged() {
    assert_eq!(
        reference().model.to_json().unwrap(),
        reference().legacy_json
    );
    assert_eq!(
        reference().coefficients.internal_energy_reference.to_bits(),
        reference().model.internal_energy_reference.to_bits()
    );
}

#[test]
fn fixed_doublet_counts_spin_density_repetition_and_charged_translation() {
    let s = system(1, 2);
    let y = s.map.reference_coordinates();
    let snapshot = reference()
        .coefficients
        .to_value()
        .unwrap()
        .to_json_pretty();
    let first = run(&s, &y, field(), origin(), Some([5, 4]), true).unwrap();
    assert_eq!(
        (
            first.full_electrons,
            first.frozen_electrons,
            first.active_electrons
        ),
        (17, 8, 9)
    );
    assert_eq!(first.spin_electrons, [5, 4]);
    assert_eq!(first.frozen_spin_electrons, [4, 4]);
    assert_eq!(first.full_spin_electrons, [9, 8]);
    assert!(first.unrestricted);
    assert_eq!(
        first.spin_status,
        "fixed-active-spin-sector-with-frozen-singlet-core"
    );
    assert!((first.active_density_trace - 9.).abs() < 1e-9);
    assert!((first.active_spin_trace - 1.).abs() < 1e-9);
    assert!((first.physical_total_charge - 1.).abs() < 1e-12);
    assert_eq!((first.ao_count, first.response_variables), (13, 76));
    assert_eq!(first.candidate_status, "candidate-not-chemically-validated");
    assert_eq!(
        first.response_status,
        "active-only-no-internal-polarization"
    );
    assert_eq!((first.gradient.len(), first.forces.len()), (5, 5));
    assert!((first.energy - first.electronic_energy - first.classical_energy).abs() < 1e-10);
    assert!(first.scf_orbital_residual <= 1e-11);
    let residuals = first.response_residuals.as_ref().unwrap();
    assert_eq!(residuals.len(), 4);
    assert_eq!(residuals[0], 0.);
    assert!(residuals
        .iter()
        .all(|r| r.is_finite() && *r <= ResponseSolverConfig::default().tolerance));
    let second = run(&s, &y, field(), origin(), Some([5, 4]), false).unwrap();
    assert_eq!(second.spin_electrons, [5, 4]);
    assert!((second.active_spin_trace - 1.).abs() < 1e-9);
    assert!((first.energy - second.energy).abs() < 1e-11);
    assert!((first.model_dipole - second.model_dipole).norm() < 1e-9);
    let translation = Vec3::new(0.23, -0.17, 0.31);
    let shifted: Vec<_> = y.iter().map(|&p| p + translation).collect();
    let moved = run(&s, &shifted, field(), origin(), Some([5, 4]), false).unwrap();
    let q = first.physical_total_charge;
    assert!((moved.energy - first.energy + q * field().dot(translation)).abs() < 1e-9);
    assert!((moved.model_dipole - first.model_dipole - translation * q).norm() < 1e-8);
    for (a, b) in first.gradient.iter().zip(&moved.gradient) {
        assert!((*a - *b).norm() < 1e-8);
    }
    let summed = first.gradient.iter().fold(Vec3::zero(), |a, &b| a + b);
    assert!((summed + field() * q).norm() < 1e-7);
    let origin_shift = Vec3::new(-0.19, 0.13, 0.27);
    let reorigin = run(
        &s,
        &y,
        field(),
        origin() + origin_shift,
        Some([5, 4]),
        false,
    )
    .unwrap();
    assert!((reorigin.energy - first.energy - q * field().dot(origin_shift)).abs() < 1e-9);
    assert!((reorigin.model_dipole - first.model_dipole + origin_shift * q).norm() < 1e-8);
    println!("few-center UKS baseline05-transfer E={:.16e} Q={} full/frozen/active=17/8/9 alpha/beta=5/4 frozen=4/4 full=9/8 spinTrace={:.16e} AO=13 responseVariables=76 CPKS={residuals:?} alpha={:?} no_localized_fit=true chemical_accuracy_unestablished=true",first.energy,q,first.active_spin_trace,first.active_model_polarizability);
    assert_eq!(
        snapshot,
        reference()
            .coefficients
            .to_value()
            .unwrap()
            .to_json_pretty()
    );
    unchanged();
}

#[test]
fn wrong_electron_spin_multiplicity_and_frozen_ledgers_are_rejected() {
    let s = system(1, 2);
    let y = s.map.reference_coordinates();
    for spins in [None, Some([5, 5]), Some([4, 5]), Some([6, 3])] {
        assert!(
            run(&s, &y, field(), origin(), spins, false).is_err(),
            "invalid sector {spins:?}"
        );
    }
    let mut wrong_ne = system(1, 2);
    wrong_ne.ledger.active_electron_count = 8;
    assert!(run(&wrong_ne, &y, field(), origin(), Some([5, 4]), false).is_err());
    let mut wrong_frozen = system(1, 2);
    wrong_frozen.ledger.frozen_electron_count = 6;
    assert!(run(&wrong_frozen, &y, field(), origin(), Some([5, 4]), false).is_err());
    let mut wrong_fragment = system(1, 2);
    wrong_fragment.ledger.fragments[0].frozen_electrons = 6;
    assert!(run(&wrong_fragment, &y, field(), origin(), Some([5, 4]), false).is_err());
    let mut wrong_multiplicity = system(1, 2);
    wrong_multiplicity.source.multiplicity = 4;
    assert!(run(
        &wrong_multiplicity,
        &y,
        field(),
        origin(),
        Some([5, 4]),
        false
    )
    .is_err());
    let mut wrong_charge = system(1, 2);
    wrong_charge.source.charge = 0;
    assert!(run(&wrong_charge, &y, field(), origin(), Some([5, 4]), false).is_err());
    unchanged();
}

fn scf_options() -> dft_rs::scf::ScfOptions {
    dft_rs::scf::ScfOptions {
        guess: dft_rs::scf::Guess::Core,
        max_iter: 300,
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-11,
        eri_mode: dft_rs::scf::EriMode::InCore { limit_mb: 64 },
        ..Default::default()
    }
}

#[test]
fn balanced_uks_reduces_to_rks_and_zero_extra_operator_reduces_to_ordinary_role_scf() {
    use dft_rs::grid::{GridOptions, MolecularGrid};
    use dft_rs::pseudo_sub::reduction_roles::layout;
    use dft_rs::xc::{resolve, FunctionalSpec};
    let s = system(0, 1);
    let y = s.map.reference_coordinates();
    let rks = run(&s, &y, field(), origin(), None, false).unwrap();
    let uks = run(&s, &y, field(), origin(), Some([5, 5]), false).unwrap();
    assert!(!rks.unrestricted && uks.unrestricted);
    assert_eq!((rks.response_variables, uks.response_variables), (40, 80));
    assert!((uks.active_spin_trace).abs() < 1e-9);
    assert!((rks.energy - uks.energy).abs() < 1e-10);
    assert!((rks.model_dipole - uks.model_dipole).norm() < 1e-8);
    for (a, b) in rks.gradient.iter().zip(&uks.gradient) {
        assert!((*a - *b).norm() < 1e-7);
    }
    // Ordinary identity is a separate zero-extra-operator boundary, not a
    // claim that the Me transfer coefficients represent a real ordinary atom.
    let zero = FixedFewCenterCoefficients {
        template: CenterTemplate::CoreSp,
        operator_template_id: PRIMITIVE_FITTED_OPERATOR_TEMPLATE_ID.into(),
        functional_id: reference().model.options.functional.clone(),
        dictionary_id: fixed_dictionary_id(&reference().model.options).unwrap(),
        gaussian_charges: [0.; 3],
        gaussian_body_dipoles: [Vec3::zero(); 3],
        local_amplitudes: [0.; 3],
        nonlocal_coefficients: [0.; 36],
        internal_energy_reference: 0.,
        reference_id: "ordinary-role-zero-extra-operator-identity".into(),
    };
    let reduced = evaluate(
        &s.source,
        &s.map,
        &s.ledger,
        CenterTemplate::CoreSp,
        &reference().model.options,
        &zero,
        &y,
        field(),
        origin(),
        None,
        None,
    )
    .unwrap();
    let roles = layout(&s.map, &s.source, &y, CenterTemplate::CoreSp).unwrap();
    let basis = roles.basis().unwrap();
    let mol = &roles.basis_role_molecule;
    assert_eq!(mol.n_electrons().unwrap(), 10);
    let grid = MolecularGrid::build(
        mol,
        &GridOptions {
            level: reference().model.options.grid_level,
            weight_cutoff: 0.,
            ..Default::default()
        },
    )
    .unwrap();
    let functional = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let mut options = scf_options();
    options.external_field = Some(field());
    let ordinary = dft_rs::scf::run_scf(mol, &basis, &options, &functional, Some(&grid)).unwrap();
    let diagnostics = dft_rs::response::diagnostics::reference_diagnostics(&ordinary).unwrap();
    assert!(ordinary.converged && diagnostics.orbital_residual <= 1e-11);
    let gradient = dft_rs::gradient::assemble_gradient(
        mol,
        &basis,
        &functional,
        Some(&grid),
        &ordinary,
        &options,
    )
    .unwrap();
    let gradient = roles.pullback(&gradient).unwrap();
    assert!((reduced.energy - ordinary.energy).abs() < 1e-10);
    assert!(
        (reduced.model_dipole - dft_rs::properties::dipole_moment(mol, &basis, &ordinary)).norm()
            < 1e-8
    );
    for (a, b) in reduced.gradient.iter().zip(&gradient) {
        assert!((*a - *b).norm() < 1e-7);
    }
    unchanged();
}

#[test]
fn two_evaluation_uks_continuation_counts_both_occupied_spaces_independently() {
    use dft_rs::driver::continuous_species::{
        ChargeChart, ChargeParameter, ContinuousSpeciesModel, SpeciesFunctional,
    };
    use dft_rs::grid::{GridOptions, MolecularGrid};
    use dft_rs::pseudo_sub::reduction_roles::layout;
    let s = system(1, 2);
    let y = s.map.reference_coordinates();
    let roles = layout(&s.map, &s.source, &y, CenterTemplate::CoreSp).unwrap();
    let basis = roles.basis().unwrap();
    let overlap = dft_rs::integrals::overlap_matrix(&basis);
    let mol = roles.basis_role_molecule;
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: reference().model.options.grid_level,
            weight_cutoff: 0.,
            ..Default::default()
        },
    )
    .unwrap();
    let chart = ChargeChart {
        reference_charges: roles.ionic_charges,
        conserve_total_charge: true,
        parameters: vec![ChargeParameter {
            name: "null-continuation-regression".into(),
            unit: "dimensionless".into(),
            scale: 1.,
            bounds: [-1., 1.],
            charge_direction: vec![0.; mol.len()],
        }],
    };
    // This low-level diagnostic explicitly isolates occupied-space bookkeeping;
    // it does not pretend that the wrapper returns an SCF continuation state.
    let model = ContinuousSpeciesModel::new_with_spin(
        mol,
        basis,
        grid,
        [5, 4],
        chart,
        SpeciesFunctional::Pbe,
        scf_options(),
        origin(),
    )
    .unwrap();
    let first = model.evaluate(&[0.], field(), None).unwrap();
    assert!(first.occupied_subspace_overlap.is_none());
    let second = model.evaluate(&[0.], field(), Some(&first)).unwrap();
    assert_eq!(first.spin_electron_counts(), [5, 4]);
    assert_eq!(second.spin_electron_counts(), [5, 4]);
    let occupied = second
        .occupied_subspace_overlap
        .expect("both occupied-space diagnostics required");
    assert!((occupied[0] - 5.).abs() < 1e-8);
    assert!((occupied[1] - 4.).abs() < 1e-8);
    assert!((second.total_density().frobenius_dot(&overlap) - 9.).abs() < 1e-9);
    assert!((second.spin_density().frobenius_dot(&overlap) - 1.).abs() < 1e-9);
    assert!(second.continuation_density_distance.unwrap() < 1e-8);
    assert!((first.energy() - second.energy()).abs() < 1e-11);
    println!("fixed UKS two-evaluation occupied singular-weight sums={occupied:?}, expected[5,4]; separate simple null-chart model, not fragment accuracy");
    unchanged();
}

#[test]
fn all_fifteen_uks_gradients_three_dipoles_and_nine_coupled_active_alphas_have_four_step_fits() {
    let s = system(1, 2);
    let model = &s;
    let y = s.map.reference_coordinates();
    let baseline = run(model, &y, field(), origin(), Some([5, 4]), true).unwrap();
    let alpha = baseline
        .active_model_polarizability
        .expect("coupled UKS active-model alpha required");
    assert!(alpha.iter().flatten().all(|x| x.is_finite()));
    let residuals = baseline.response_residuals.as_ref().unwrap();
    assert_eq!(residuals.len(), 4);
    assert_eq!(residuals[0], 0.);
    assert!(residuals
        .iter()
        .all(|r| r.is_finite() && *r <= ResponseSolverConfig::default().tolerance));
    println!("UKS independent derivatives active-only alpha={alpha:?} coupledCPKS={residuals:?} frozen_internal_response=absent");
    let field = field();
    // Both original intervals remain recorded. The finer interval resolves
    // the observed nonasymptotic alpha[0,1] truncation error without changing
    // the production response or the shared convergence-fit thresholds.
    let intervals = [
        [0.02, 0.01, 0.005, 0.0025],
        [0.04, 0.02, 0.01, 0.005],
        [0.01, 0.005, 0.0025, 0.00125],
    ];
    let names: Vec<_> = (0..27)
        .map(|i| {
            if i < 15 {
                format!("few-center UKS gradient[{},{}]", i / 3, i % 3)
            } else if i < 18 {
                format!("few-center UKS dipole[{}]", i - 15)
            } else {
                format!(
                    "few-center UKS active-only alpha[{},{}]",
                    (i - 18) / 3,
                    (i - 18) % 3
                )
            }
        })
        .collect();
    let mut cache = BTreeMap::<(usize, u64), FewCenterEvaluation>::new();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &intervals,
        |steps| {
            let mut errors = vec![[0.; 4]; 27];
            for component in 0..27 {
                let (channel, analytic) = if component < 15 {
                    (
                        component,
                        baseline.gradient[component / 3].get(component % 3),
                    )
                } else if component < 18 {
                    (
                        15 + component - 15,
                        baseline.model_dipole.get(component - 15),
                    )
                } else {
                    (
                        15 + (component - 18) % 3,
                        alpha[(component - 18) / 3][(component - 18) % 3],
                    )
                };
                let mut estimates = [0.; 4];
                for (k, &h) in steps.iter().enumerate() {
                    let mut value = |amount: f64| {
                        let v = cache.entry((channel, amount.to_bits())).or_insert_with(|| {
                            let mut coords = y.clone();
                            let mut f = field;
                            if channel < 15 {
                                let axis = channel % 3;
                                let mut p = coords[channel / 3].to_array();
                                p[axis] += amount;
                                coords[channel / 3] = Vec3::new(p[0], p[1], p[2]);
                            } else {
                                let mut p = f.to_array();
                                p[channel - 15] += amount;
                                f = Vec3::new(p[0], p[1], p[2]);
                            }
                            run(model, &coords, f, origin(), Some([5,4]), false).unwrap_or_else(|e| {
                                panic!(
                                    "few-center UKS derivative channel={channel} amount={amount:.16e}: {e}"
                                )
                            })
                        });
                        if component < 18 {
                            v.energy
                        } else {
                            v.model_dipole.get((component - 18) / 3)
                        }
                    };
                    estimates[k] = (value(h) - value(-h)) / (2. * h)
                        * if (15..18).contains(&component) {
                            -1.
                        } else {
                            1.
                        };
                    errors[component][k] = (estimates[k] - analytic).abs();
                }
                let extrapolated = (4. * estimates[3] - estimates[2]) / 3.;
                println!("{} interval={steps:?} analytic={analytic:.16e} finestAbsoluteError={:.16e} RichardsonError={:.16e} no_floor=true",names[component],errors[component][3],(extrapolated-analytic).abs());
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures(
        "few-center UKS candidate all15 force/3 dipole/9 active alpha components",
        &failures,
    );
    unchanged();
}
