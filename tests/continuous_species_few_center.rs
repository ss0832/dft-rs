// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Controlled variational/force wiring gates, not a chemical model fit.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::driver::continuous_species::few_center::{
    evaluate, fixed_dictionary_id, FixedFewCenterChart, FixedFewCenterCoefficients,
    PRIMITIVE_FITTED_OPERATOR_TEMPLATE_ID,
};
use dft_rs::driver::continuous_species::reduction_linear::{ElectronLedger, FrozenFragmentSpec};
use dft_rs::math::Vec3;
use dft_rs::pseudo_sub::reduction::GenerationOptions;
use dft_rs::pseudo_sub::reduction_basis::OneCenterBasisTemplate;
use dft_rs::pseudo_sub::reduction_geometry::{AttachmentPort, FragmentSpec, RigidExpansion};
use dft_rs::pseudo_sub::reduction_roles::{layout, CenterTemplate};
use dft_rs::system::{Atom, Molecule};

fn inputs() -> (Molecule, RigidExpansion, ElectronLedger, GenerationOptions) {
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
    let source = Molecule::new(
        points
            .iter()
            .enumerate()
            .map(|(i, p)| Atom {
                z: if i == 0 || i == 4 { 6 } else { 1 },
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    );
    let map = RigidExpansion::methyl(
        &source,
        FragmentSpec {
            explicit_atoms: vec![4, 5, 6, 7],
            retained_atoms: vec![0, 1, 2, 3],
            port: AttachmentPort {
                fragment_atom: 4,
                host_atom: 0,
                reference_atom: 1,
            },
            single_bonds: vec![[0, 1], [0, 2], [0, 3], [0, 4], [4, 5], [4, 6], [4, 7]],
            electronic_centers: 1,
        },
    )
    .unwrap();
    let ledger = ElectronLedger::new(
        &source.atoms.iter().map(|a| a.z).collect::<Vec<_>>(),
        source.n_electrons().unwrap(),
        &[FrozenFragmentSpec {
            explicit_atoms: vec![4, 5, 6, 7],
            frozen_electrons: 8,
        }],
    )
    .unwrap();
    let dictionary = GenerationOptions {
        basis_template: OneCenterBasisTemplate::CarbonCoreValenceSp,
        ..GenerationOptions::default()
    };
    (source, map, ledger, dictionary)
}

fn controlled(t: CenterTemplate) -> FixedFewCenterCoefficients {
    // Explicit synthetic operators exercise every frame derivative. They are
    // neither fitted nor reused from a generated candidate of another template.
    let mut d = [0.0; 36];
    for (i, v) in d.iter_mut().enumerate() {
        *v = 0.004 * ((i % 7) as f64 - 2.5);
    }
    let dictionary = GenerationOptions {
        basis_template: OneCenterBasisTemplate::CarbonCoreValenceSp,
        ..GenerationOptions::default()
    };
    FixedFewCenterCoefficients {
        template: t,
        operator_template_id: PRIMITIVE_FITTED_OPERATOR_TEMPLATE_ID.into(),
        functional_id: dictionary.functional.clone(),
        dictionary_id: fixed_dictionary_id(&dictionary).unwrap(),
        gaussian_charges: [0.03, -0.02, -0.01],
        gaussian_body_dipoles: [
            Vec3::new(0.004, -0.003, 0.002),
            Vec3::new(-0.002, 0.001, 0.003),
            Vec3::new(0.001, 0.002, -0.001),
        ],
        local_amplitudes: [0.08, -0.03, 0.01],
        nonlocal_coefficients: d,
        internal_energy_reference: -0.37,
        reference_id: format!("controlled-wiring-only-{}", t.id()),
    }
}

#[test]
fn fixed_replacement_chart_preserves_integer_ledgers_and_operator_space() {
    let (_, map, ledger, dictionary) = inputs();
    let chart =
        FixedFewCenterChart::new(&dictionary, CenterTemplate::CoreSp, &map, &ledger, [5, 5])
            .unwrap();
    assert_eq!(chart.active_electron_count(), 10);
    assert_eq!(chart.spin_electron_counts(), [5, 5]);
    assert_eq!(chart.total_ao_functions(), 13);
    assert_eq!(chart.geometry_map().retained_atoms(), map.retained_atoms());
    let mut ledgers = Vec::new();
    for (name, frozen, nuclear, full) in [("Me", 8, 9, 18), ("Et", 16, 17, 26), ("tBu", 32, 33, 42)]
    {
        let reference = dft_rs::pseudo_sub::presets::get(name)
            .unwrap()
            .methyl_capped_reference()
            .unwrap();
        let accounting = ElectronLedger::new(
            &reference
                .molecule
                .atoms
                .iter()
                .map(|a| a.z)
                .collect::<Vec<_>>(),
            reference.molecule.n_electrons().unwrap(),
            &[FrozenFragmentSpec {
                explicit_atoms: reference.fragment_atoms,
                frozen_electrons: frozen,
            }],
        )
        .unwrap();
        assert_eq!(accounting.fragments[0].nuclear_charge, nuclear);
        assert_eq!(accounting.fragments[0].effective_ionic_charge, 1);
        assert_eq!(accounting.full_electron_count, full);
        assert_eq!(accounting.active_electron_count, 10);
        assert_eq!(accounting.retained_nuclear_charge, 9);
        assert_eq!(accounting.full_total_charge, 0);
        println!(
            "{name}: full={full}, frozen={frozen}, active=10, fragment_nuclear={nuclear}, core=+1"
        );
        ledgers.push(accounting);
    }
    // These three operator fixtures test algebra and chart compatibility only.
    // They are not generated or chemically fitted Me/Et/tBu reduced models.
    let coefficients: Vec<_> = (0..3)
        .map(|i| {
            let mut c = controlled(CenterTemplate::CoreSp);
            c.reference_id = format!("synthetic-replacement-contract-{i}");
            let factor = 0.7 + 0.4 * i as f64;
            c.gaussian_charges[0] *= factor;
            c.gaussian_charges[1] *= factor;
            c.gaussian_charges[2] = -(c.gaussian_charges[0] + c.gaussian_charges[1]);
            for v in &mut c.gaussian_body_dipoles {
                *v = *v * factor;
            }
            for v in c
                .local_amplitudes
                .iter_mut()
                .chain(&mut c.nonlocal_coefficients)
            {
                *v *= factor;
            }
            c.internal_energy_reference *= factor;
            c
        })
        .collect();
    let endpoints: Vec<_> = coefficients
        .iter()
        .zip(&ledgers)
        .map(|(c, l)| chart.endpoint(c, l, [5, 5], 13).unwrap())
        .collect();
    assert_eq!(
        endpoints
            .iter()
            .map(|e| e.realization_ledger().frozen_electron_count)
            .collect::<Vec<_>>(),
        vec![8, 16, 32]
    );
    let weights = [0.2, 0.3, 0.5];
    let result = chart.interpolate(&endpoints, &weights).unwrap();
    let flatten = |c: &FixedFewCenterCoefficients| {
        c.gaussian_charges
            .iter()
            .copied()
            .chain(c.gaussian_body_dipoles.iter().flat_map(|v| v.to_array()))
            .chain(c.local_amplitudes)
            .chain(c.nonlocal_coefficients)
            .chain([c.internal_energy_reference])
            .collect::<Vec<_>>()
    };
    let values: Vec<_> = coefficients.iter().map(&flatten).collect();
    for (i, value) in flatten(&result).into_iter().enumerate() {
        let expected = weights
            .iter()
            .enumerate()
            .map(|(j, w)| w * values[j][i])
            .sum::<f64>();
        assert!((value - expected).abs() < 1e-16, "coefficient {i}");
    }
    assert_eq!(
        result.gaussian_charges[2],
        -(result.gaussian_charges[0] + result.gaussian_charges[1])
    );
    assert_eq!(
        flatten(&chart.interpolate(&endpoints, &[0.0, 1.0, 0.0]).unwrap()),
        flatten(&coefficients[1])
    );
    for weights in [[0.2, 0.3, 0.4], [-0.1, 0.4, 0.7], [f64::NAN, 0.0, 1.0]] {
        assert!(chart.interpolate(&endpoints, &weights).is_err());
    }
    assert!(chart
        .endpoint(&coefficients[0], &ledgers[0], [6, 4], 13)
        .is_err());
    assert!(chart
        .endpoint(&coefficients[0], &ledgers[0], [5, 5], 14)
        .is_err());
    for variant in 0..5 {
        let mut c = coefficients[0].clone();
        match variant {
            0 => c.template = CenterTemplate::CoreSpH,
            1 => c.dictionary_id.push_str("-different-widths"),
            2 => c.functional_id = "lda".into(),
            3 => c.operator_template_id.push_str("-contracted"),
            _ => c.gaussian_charges[2] += f64::EPSILON,
        }
        assert!(chart.endpoint(&c, &ledgers[0], [5, 5], 13).is_err());
    }
    let mut wrong_ledger = ledgers[0].clone();
    wrong_ledger.fragments[0].effective_ionic_charge = 2;
    assert!(chart
        .endpoint(&coefficients[0], &wrong_ledger, [5, 5], 13)
        .is_err());
    let other_chart =
        FixedFewCenterChart::new(&dictionary, CenterTemplate::CoreSp, &map, &ledger, [5, 5])
            .unwrap();
    assert_eq!(
        chart.connection_chart_id(),
        other_chart.connection_chart_id()
    );
    assert!(other_chart.interpolate(&endpoints, &weights).is_err());
    let mut changed_dictionary = dictionary.clone();
    changed_dictionary.s_exponents[0] *= 1.01;
    assert!(FixedFewCenterChart::new(
        &changed_dictionary,
        CenterTemplate::CoreSp,
        &map,
        &ledger,
        [5, 5]
    )
    .is_err());
}

#[test]
fn controlled_role_counts_ledger_and_energy_constant() {
    let (source, map, ledger, dictionary) = inputs();
    let y = map.reference_coordinates();
    for t in [
        CenterTemplate::CoreSp,
        CenterTemplate::CoreSpH,
        CenterTemplate::CoreSpHH,
    ] {
        let p = controlled(t);
        let s = evaluate(
            &source,
            &map,
            &ledger,
            t,
            &dictionary,
            &p,
            &y,
            Vec3::zero(),
            Vec3::zero(),
            None,
            None,
        )
        .unwrap();
        println!("controlled {} E={:.16e} roleCount={} AO={} grid={} Ne={} trace={:.16e} Q={:.16e} residual={:.9e}",t.id(),s.energy,s.role_sites,s.ao_count,s.grid_points,s.active_electrons,s.active_density_trace,s.physical_total_charge,s.scf_orbital_residual);
        assert_eq!(s.ao_count, 12 + t.center_count());
        assert_eq!(s.role_sites, 4 + t.center_count());
        assert_eq!(
            (s.active_electrons, s.frozen_electrons, s.full_electrons),
            (10, 8, 18)
        );
        assert_eq!(s.spin_electrons, [5, 5]);
        assert!(s.physical_total_charge.abs() < 1e-10);
        assert!((s.active_density_trace - 10.0).abs() < 1e-9);
        assert_eq!(s.gradient.len(), 5);
        assert_eq!(s.fragment_internal_energy, p.internal_energy_reference);
        assert_eq!(
            s.energy,
            s.energy_without_internal_reference + s.fragment_internal_energy
        );
        assert_eq!(
            s.classical_energy,
            s.classical_energy_without_internal_reference + s.fragment_internal_energy
        );
        let mut shifted = p.clone();
        shifted.internal_energy_reference += 0.123;
        let b = evaluate(
            &source,
            &map,
            &ledger,
            t,
            &dictionary,
            &shifted,
            &y,
            Vec3::zero(),
            Vec3::zero(),
            None,
            None,
        )
        .unwrap();
        assert!((b.energy - s.energy - 0.123).abs() < 1e-10);
        assert_eq!(
            s.energy_without_internal_reference,
            b.energy_without_internal_reference
        );
        assert_eq!(
            s.classical_energy_without_internal_reference,
            b.classical_energy_without_internal_reference
        );
        assert_eq!(s.scf_iterations, b.scf_iterations);
        assert_eq!(s.scf_orbital_residual, b.scf_orbital_residual);
        for (a, b) in s.gradient.iter().zip(b.gradient) {
            assert!((*a - b).norm() < 1e-9);
        }
        if t == CenterTemplate::CoreSp {
            use dft_rs::driver::continuous_species::few_center_driver::{
                FewCenterModelRecord, FewCenterReference,
            };
            use dft_rs::driver::{Controls, System, Value};
            let record = FewCenterModelRecord::new(
                source.clone(),
                map.fragment_spec().unwrap().clone(),
                t,
                dft_rs::driver::continuous_species::SpeciesFunctional::Pbe,
                dictionary.grid_level,
                FewCenterReference::Rks,
                [5, 5],
                p,
            )
            .unwrap();
            let system = System::molecule(
                record.primary_numbers(),
                y.iter().map(|v| v.to_array()).collect(),
            );
            let controls = Controls::from_value(
                &Value::object()
                    .with("fragment_mode", "few-evaluate")
                    .with("fragment_model_json", record.to_json().unwrap())
                    .with("cpu_threads", 1),
            )
            .unwrap();
            let report = dft_rs::driver::run("continuous-species", &system, &controls).unwrap();
            for (name, expected) in [
                (
                    "energy_without_internal_reference",
                    s.energy_without_internal_reference,
                ),
                (
                    "classical_energy_without_internal_reference",
                    s.classical_energy_without_internal_reference,
                ),
                ("fragment_internal_energy", s.fragment_internal_energy),
            ] {
                assert!((report.get(name).unwrap().as_f64().unwrap() - expected).abs() < 1e-11);
                assert_eq!(
                    report.get("units").unwrap().get(name).unwrap().as_str(),
                    Some("hartree")
                );
            }
            assert_eq!(
                report.get("scf_energy_convention").unwrap().as_str(),
                Some(dft_rs::driver::continuous_species::FRAGMENT_SCF_ENERGY_CONVENTION)
            );
        }
    }
}

fn force_components(t: CenterTemplate) {
    let (source, map, ledger, dictionary) = inputs();
    let mut y = map.reference_coordinates();
    y[4] += Vec3::new(0.11, -0.07, 0.04);
    y[1] += Vec3::new(0.04, 0.03, 0.02);
    let p = controlled(t);
    let field = Vec3::new(0.0013, -0.0009, 0.0017);
    let origin = Vec3::new(0.11, 0.07, -0.09);
    let analytic = evaluate(
        &source,
        &map,
        &ledger,
        t,
        &dictionary,
        &p,
        &y,
        field,
        origin,
        None,
        None,
    )
    .unwrap();
    let names: Vec<_> = (0..5)
        .flat_map(|a| (0..3).map(move |i| format!("{} actualGeneralizedGradient[{a},{i}]", t.id())))
        .collect();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &[[0.02, 0.01, 0.005, 0.0025], [0.04, 0.02, 0.01, 0.005]],
        |steps| {
            let mut errors = vec![[0.0; 4]; 15];
            for a in 0..5 {
                for i in 0..3 {
                    for (k, &h) in steps.iter().enumerate() {
                        let energy = |delta| {
                            let mut displaced = y.clone();
                            let mut pos = displaced[a].to_array();
                            pos[i] += delta;
                            displaced[a] = Vec3::new(pos[0], pos[1], pos[2]);
                            evaluate(
                                &source,
                                &map,
                                &ledger,
                                t,
                                &dictionary,
                                &p,
                                &displaced,
                                field,
                                origin,
                                None,
                                None,
                            )
                            .unwrap()
                            .energy
                        };
                        errors[3 * a + i][k] = ((energy(h) - energy(-h)) / (2.0 * h)
                            - analytic.gradient[a].get(i))
                        .abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("few-center controlled QC generalized forces", &failures);
}

#[test]
fn controlled_c13_force_components() {
    force_components(CenterTemplate::CoreSp);
}
#[test]
fn controlled_c14_force_components() {
    force_components(CenterTemplate::CoreSpH);
}
#[test]
fn controlled_c15_force_components() {
    force_components(CenterTemplate::CoreSpHH);
}

#[test]
fn template_and_provenance_refusals_precede_scf() {
    let (source, map, ledger, dictionary) = inputs();
    let y = map.reference_coordinates();
    let t = CenterTemplate::CoreSpH;
    let p = controlled(CenterTemplate::CoreSp);
    assert!(evaluate(
        &source,
        &map,
        &ledger,
        t,
        &dictionary,
        &p,
        &y,
        Vec3::zero(),
        Vec3::zero(),
        None,
        None
    )
    .is_err());
    let mut p = controlled(t);
    p.reference_id.clear();
    assert!(p.validate().is_err());
    p = controlled(t);
    p.gaussian_charges[0] += 0.01;
    assert!(p.validate().is_err());
    for incompatible in [
        "contracted-Me8-fitted-gaussian3-local3-v1",
        "primitive-projector8-exact-sigma-oracle-v1",
        "",
    ] {
        let mut p = controlled(t);
        p.operator_template_id = incompatible.into();
        assert!(evaluate(
            &source,
            &map,
            &ledger,
            t,
            &dictionary,
            &p,
            &y,
            Vec3::zero(),
            Vec3::zero(),
            None,
            None
        )
        .is_err());
    }
    let mut p = controlled(t);
    p.functional_id = "lda".into();
    assert!(evaluate(
        &source,
        &map,
        &ledger,
        t,
        &dictionary,
        &p,
        &y,
        Vec3::zero(),
        Vec3::zero(),
        None,
        None
    )
    .is_err());
    p = controlled(t);
    p.dictionary_id = "fnv1a64:wrong-dictionary".into();
    assert!(evaluate(
        &source,
        &map,
        &ledger,
        t,
        &dictionary,
        &p,
        &y,
        Vec3::zero(),
        Vec3::zero(),
        None,
        None
    )
    .is_err());
    p.dictionary_id.clear();
    assert!(p.validate().is_err());
    assert!(layout(&map, &source, &y[..4], t).is_err());
}

#[test]
fn operator_recipe_binding_is_canonical_and_excludes_integration_resolution() {
    let (_, _, _, d) = inputs();
    let id = fixed_dictionary_id(&d).unwrap();
    let mut alias = d.clone();
    alias.functional = "PBE".into();
    assert_eq!(fixed_dictionary_id(&alias).unwrap(), id);
    alias.grid_level += 1;
    alias.pauli_level *= 2.0;
    alias.regularization *= 2.0;
    assert_eq!(fixed_dictionary_id(&alias).unwrap(), id);
    alias.local_exponents[0] *= 1.01;
    assert!(fixed_dictionary_id(&alias).is_err());
    let p = controlled(CenterTemplate::CoreSp);
    let value = p.to_value().unwrap();
    assert_eq!(
        value.get("operator_template_id").unwrap().as_str(),
        Some(PRIMITIVE_FITTED_OPERATOR_TEMPLATE_ID)
    );
    assert_eq!(value.get("functional_id").unwrap().as_str(), Some("pbe"));
    assert_eq!(
        value.get("dictionary_id").unwrap().as_str(),
        Some(id.as_str())
    );
}
