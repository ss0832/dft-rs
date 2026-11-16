// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Diagnostic decomposition of an unaccepted fixed candidate; no refitting.
use dft_rs::basis::{data, MolBasis};
use dft_rs::driver::continuous_species::gaussian::gaussian_potential;
use dft_rs::driver::continuous_species::reduced::evaluate_candidate;
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::integrals::nuclear_matrix;
use dft_rs::linalg::{symmetric_eigen, Matrix};
use dft_rs::math::Vec3;
use dft_rs::pseudo_sub::reduction::{generate_methyl, GenerationOptions};
use dft_rs::pseudo_sub::reduction_geometry::{AttachmentPort, FragmentSpec};
use dft_rs::pseudo_sub::reduction_kernels::{
    charge_operator, core_interactions, electrostatic_potential, projector_overlaps,
    symmetric_projector_coefficients, ProjectorTemplate,
};
use dft_rs::pseudo_sub::reduction_reference::{
    extract_methyl_frozen_space, ExtractionOptions, FrozenReferenceModel, MethylBoundary,
};
use dft_rs::scf::{run_scf, EriMode, Guess, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

fn source() -> Molecule {
    let positions = [
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
        positions
            .iter()
            .enumerate()
            .map(|(i, p)| Atom {
                z: if i == 0 || i == 4 { 6 } else { 1 },
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    )
}
fn spec() -> FragmentSpec {
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
    }
}
fn opts() -> ScfOptions {
    ScfOptions {
        guess: Guess::Core,
        max_iter: 300,
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-11,
        eri_mode: EriMode::InCore { limit_mb: 64 },
        ..ScfOptions::default()
    }
}
fn grid(m: &Molecule) -> MolecularGrid {
    MolecularGrid::build(
        m,
        &GridOptions {
            level: 1,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        },
    )
    .unwrap()
}
fn add(a: &mut Matrix, b: &Matrix, s: f64) {
    for (x, y) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *x += s * y;
    }
}
fn norm(a: &Matrix) -> f64 {
    a.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn range(label: &str, c: &[f64]) {
    println!(
        "coefficients {label} count={} min={:.12e} max={:.12e} l2={:.12e} raw={c:?}",
        c.len(),
        c.iter().copied().fold(f64::INFINITY, f64::min),
        c.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        c.iter().map(|x| x * x).sum::<f64>().sqrt()
    );
}

#[test]
fn fixed_candidate_failure_channels() {
    let source = source();
    let candidate = generate_methyl(&source, spec(), GenerationOptions::default()).unwrap();
    let saved = candidate.to_json().unwrap();
    if let Ok(path) = std::env::var("DFT_RS_FRAGMENT_DIAGNOSTIC_MODEL") {
        std::fs::write(path, &saved).unwrap();
    } else {
        println!("candidate_raw_json={saved}");
    }
    range("neutral-sigma-monopoles", &candidate.gaussian_charges);
    range(
        "body-dipoles",
        &candidate
            .gaussian_body_dipoles
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
    );
    range("local", &candidate.local_amplitudes);
    range("nonlocal-upper", &candidate.nonlocal_coefficients);
    let coordinates = candidate.geometry_map.reference_coordinates();
    let center_index = candidate.geometry_map.electronic_center_index().unwrap();
    let frame = candidate
        .geometry_map
        .body_frame_kinematics(&coordinates)
        .unwrap()
        .unwrap();
    let direction = (coordinates[center_index] - coordinates[0])
        * (1.0 / (coordinates[center_index] - coordinates[0]).norm());
    let sites = candidate
        .gaussian_sites(frame.center, frame.orientation)
        .unwrap();
    let basis = MolBasis::build(&source, &data::load("sto-3g").unwrap()).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let options = opts();
    let quadrature = grid(&source);
    let full = run_scf(&source, &basis, &options, &functional, Some(&quadrature)).unwrap();
    let selection = extract_methyl_frozen_space(
        &source,
        &basis,
        &full,
        MethylBoundary {
            carbon: 4,
            hydrogens: [5, 6, 7],
            port_atom: 0,
        },
        &ExtractionOptions::default(),
    )
    .unwrap();
    let frozen = FrozenReferenceModel::new(
        &source,
        &basis,
        &quadrature,
        &functional,
        &options,
        &selection,
    )
    .unwrap();
    // Metric-only projection distinguishes the 9-AO limit from an operator fit.
    let mut reduced_atoms: Vec<_> = candidate
        .spec
        .retained_atoms
        .iter()
        .map(|&i| source.atoms[i].clone())
        .collect();
    reduced_atoms.push(Atom {
        z: 1,
        position: frame.center,
    });
    let reduced_basis = MolBasis::build(
        &Molecule::new(reduced_atoms),
        &data::load("sto-3g").unwrap(),
    )
    .unwrap();
    let mut cross = Matrix::zeros(basis.nao_sph, reduced_basis.nao_sph);
    for left in &basis.shells {
        for right in &reduced_basis.shells {
            use dft_rs::integrals::analytic_basis::{self, AnalyticShell};
            let a = AnalyticShell::from_placed(left, left.center.to_array()).unwrap();
            let b = AnalyticShell::from_placed(right, right.center.to_array()).unwrap();
            let block = analytic_basis::overlap(&a, &b).unwrap();
            for i in 0..block.shape[0] {
                for j in 0..block.shape[1] {
                    cross[(left.sph_off + i, right.sph_off + j)] =
                        block.values[i * block.shape[1] + j];
                }
            }
        }
    }
    let reduced_overlap = dft_rs::integrals::overlap_matrix(&reduced_basis);
    let (spectrum, vectors) = symmetric_eigen(&reduced_overlap).unwrap();
    let mut inverse = Matrix::zeros(reduced_basis.nao_sph, reduced_basis.nao_sph);
    for k in 0..spectrum.len() {
        for i in 0..inverse.rows {
            for j in 0..inverse.cols {
                inverse[(i, j)] += vectors[(i, k)] * vectors[(j, k)] / spectrum[k];
            }
        }
    }
    let overlap = selection
        .active_reference_orbitals
        .transpose_matmul_seq(&cross);
    let capture = overlap.matmul_seq(&inverse).matmul_transpose_seq(&overlap);
    let captures = symmetric_eigen(&capture).unwrap().0;
    let missing = 2.0 * (captures.len() as f64 - captures.iter().sum::<f64>());
    println!("active-reference-basis-projection fullAO={} reducedAO={} occupied_capture_eigenvalues={captures:?} missing_electron_metric_weight={missing:.12e}; metric-only-not-refitted",basis.nao_sph,reduced_basis.nao_sph);
    let mut probes: Vec<_> = candidate
        .spec
        .retained_atoms
        .iter()
        .map(|&i| source.atoms[i].position)
        .collect();
    let mut labels: Vec<_> = candidate
        .spec
        .retained_atoms
        .iter()
        .map(|i| format!("retained-core-{i}"))
        .collect();
    for r in [0.6, 0.9, 1.2, 1.6, 2.0, 2.4, 3.0] {
        probes.push(frame.center - direction * r);
        labels.push(format!("port-axis-{r}"));
    }
    for axis in 0..3 {
        for sign in [-1.0, 1.0] {
            for r in [1.1, 2.2] {
                probes.push(frame.center + frame.orientation.col[axis] * (sign * r));
                labels.push(format!("body-axis-{axis}-{sign}-{r}"));
            }
        }
    }
    let exact = frozen.frozen_core_potential(&probes).unwrap();
    let corr = electrostatic_potential(&sites, &probes).unwrap();
    let mut exact_cross = 0.0;
    let mut fit_cross = 0.0;
    let mut point_cross = 0.0;
    let mut correction_cross = 0.0;
    for i in 0..probes.len() {
        let point = 1.0 / (probes[i] - frame.center).norm();
        let fit = point + corr[i];
        println!("sigma-ESP label={} exact={:.12e} point={:.12e} gaussian={:.12e} fitted={:.12e} defect={:.12e}",labels[i],exact[i],point,corr[i],fit,fit-exact[i]);
        if i < 4 {
            let z = f64::from(source.atoms[candidate.spec.retained_atoms[i]].z);
            exact_cross += z * exact[i];
            fit_cross += z * fit;
            point_cross += z * point;
            correction_cross += z * corr[i];
            println!("sigma-core-boundary retained={} Z={z} exact={:.12e} fitted={:.12e} energy_defect={:.12e}",candidate.spec.retained_atoms[i],z*exact[i],z*fit,z*(fit-exact[i]));
        }
    }
    let cores: Vec<_> = candidate
        .spec
        .retained_atoms
        .iter()
        .map(|&i| (source.atoms[i].position, f64::from(source.atoms[i].z)))
        .collect();
    let core = core_interactions(&sites, &cores).unwrap();
    assert!((core.energy - correction_cross).abs() < 1e-12);
    println!("sigma-core-reference exact_cross={exact_cross:.12e} baseline_point_cross={point_cross:.12e} Gaussian_cross={correction_cross:.12e} fitted_cross={fit_cross:.12e} defect={:.12e}",fit_cross-exact_cross);
    for offset in [-0.001, 0.001] {
        let shifted_probes: Vec<_> = probes[..4]
            .iter()
            .map(|&p| p - direction * offset)
            .collect();
        let exact = frozen.frozen_core_potential(&shifted_probes).unwrap();
        let shifted_sites = candidate
            .gaussian_sites(frame.center + direction * offset, frame.orientation)
            .unwrap();
        let gaussian = core_interactions(&shifted_sites, &cores).unwrap().energy;
        let point: f64 = cores
            .iter()
            .map(|(r, z)| z / (*r - frame.center - direction * offset).norm())
            .sum();
        let exact: f64 = exact.iter().zip(&cores).map(|(v, (_, z))| v * z).sum();
        println!("sigma-core-rigid-translation offset={offset:.9e} exact_cross={exact:.12e} fitted_cross={:.12e} defect={:.12e}",point+gaussian,point+gaussian-exact);
    }
    let target = frozen
        .static_residual_target(candidate.options.pauli_level)
        .unwrap();
    let sigma = charge_operator(&basis, source.len(), &sites)
        .unwrap()
        .operator;
    let mut zpoint = vec![0.0; 8];
    zpoint[4] = 1.0;
    let point = nuclear_matrix(&basis, &source, Some(&zpoint));
    let mut fitted_sigma = point.clone();
    add(&mut fitted_sigma, &sigma, 1.0);
    let mut exact_sigma = target.removed_nuclear.clone();
    add(&mut exact_sigma, &target.frozen_hartree, 1.0);
    let template = ProjectorTemplate {
        s_exponents: candidate.options.s_exponents,
        p_exponents: candidate.options.p_exponents,
        orientation: frame.orientation,
    };
    let overlaps = projector_overlaps(&basis, source.len(), frame.center, &template).unwrap();
    let nonlocal = overlaps
        .operator(&symmetric_projector_coefficients(&candidate.nonlocal_coefficients).unwrap())
        .unwrap()
        .operator;
    let mut local = Matrix::zeros(16, 16);
    for i in 0..3 {
        add(
            &mut local,
            &gaussian_potential(
                &basis,
                source.len(),
                frame.center,
                candidate.options.local_exponents[i],
            )
            .unwrap()
            .operator,
            candidate.local_amplitudes[i],
        );
    }
    let mut electrostatic_residual = exact_sigma.clone();
    add(&mut electrostatic_residual, &fitted_sigma, -1.0);
    let mut total_residual = target.total.clone();
    add(&mut total_residual, &fitted_sigma, -1.0);
    add(&mut total_residual, &local, -1.0);
    add(&mut total_residual, &nonlocal, -1.0);
    let (eigenvalues, vectors) = symmetric_eigen(frozen.evaluator().overlap()).unwrap();
    let mut x = Matrix::zeros(16, 16);
    for i in 0..16 {
        for j in 0..16 {
            for k in 0..16 {
                x[(i, j)] += vectors[(i, k)] * vectors[(j, k)] / eigenvalues[k].sqrt();
            }
        }
    }
    for (label, m) in [
        ("exact-sigma", &exact_sigma),
        ("fitted-point+Gaussian", &fitted_sigma),
        ("electrostatic-residual", &electrostatic_residual),
        ("nonadditive-XC-target", &target.nonadditive_xc),
        ("reference-Pauli-target", &target.reference_pauli),
        ("fitted-local", &local),
        ("fitted-nonlocal", &nonlocal),
        ("total-residual", &total_residual),
    ] {
        let full_norm = norm(&x.matmul_seq(m).matmul_seq(&x));
        let occupied_norm = norm(
            &selection
                .active_reference_orbitals
                .transpose_matmul_seq(&m.matmul_seq(&selection.active_reference_orbitals)),
        );
        let complement_norm = norm(
            &frozen
                .complement()
                .transpose_matmul_seq(&m.matmul_seq(frozen.complement())),
        );
        let frozen_norm = norm(
            &selection
                .frozen_orbitals
                .transpose_matmul_seq(&m.matmul_seq(&selection.frozen_orbitals)),
        );
        println!("operator-channel {label} fullAOmetricNorm={full_norm:.12e} activeOccupiedNorm={occupied_norm:.12e} activeComplementNorm={complement_norm:.12e} frozenNorm={frozen_norm:.12e} TrPA={:.12e} TrPB={:.12e}",selection.active_reference_density.frobenius_dot(m),selection.frozen_density.frobenius_dot(m));
    }
    for offset in [-0.05, 0.0, 0.05] {
        let mut y = coordinates.clone();
        y[center_index] += direction * offset;
        let reduced =
            evaluate_candidate(&candidate, &y, Vec3::zero(), Vec3::new(0.11, 0.07, -0.09)).unwrap();
        let explicit = candidate.geometry_map.expand(&y).unwrap();
        let mb = MolBasis::build(&explicit, &data::load("sto-3g").unwrap()).unwrap();
        let g = grid(&explicit);
        let full = run_scf(&explicit, &mb, &options, &functional, Some(&g)).unwrap();
        println!("curve-channel offset={offset:.9e} Efull={:.12e} Efull_elec={:.12e} Efull_class={:.12e} Ereduced={:.12e} Ereduced_elec={:.12e} Ereduced_class={:.12e} E0={:.12e} reduced_class_withoutE0={:.12e} force_stretch_reduced={:.12e}",full.energy,full.e_elec,full.e_nuc,reduced.energy,reduced.electronic_energy,reduced.classical_energy,candidate.internal_energy_reference,reduced.classical_energy-candidate.internal_energy_reference,reduced.forces[center_index].dot(direction));
    }
    assert_eq!(saved, candidate.to_json().unwrap());
    println!("diagnostic status=chemical-failure-under-investigation coefficients=fixed shape=not-generated no-refit no-E0-reset no-acceptance-promotion");
}

#[test]
fn fixed_esp_construction_variant_curve() {
    use dft_rs::gradient::assemble_gradient;
    use dft_rs::pseudo_sub::reduction_geometry::RigidExpansion;
    let source = source();
    let map = RigidExpansion::methyl(&source, spec()).unwrap();
    let reference = map.reference_coordinates();
    let center_index = map.electronic_center_index().unwrap();
    let frame = map.body_frame_kinematics(&reference).unwrap().unwrap();
    let direction = (reference[center_index] - reference[0])
        * (1.0 / (reference[center_index] - reference[0]).norm());
    // Prescribed BEFORE inspecting this candidate's energies: no search/holdout.
    let mut points: Vec<_> = (0..4).map(|i| source.atoms[i].position).collect();
    for atom in 0..4 {
        for axis in 0..3 {
            for sign in [-1.0, 1.0] {
                let mut displacement = [0.0; 3];
                displacement[axis] = sign * 0.05;
                points.push(
                    source.atoms[atom].position
                        + Vec3::new(displacement[0], displacement[1], displacement[2]),
                );
            }
        }
    }
    for r in [0.6, 0.9, 1.2, 1.6, 2.0, 2.4, 3.0] {
        points.push(frame.center - direction * r);
    }
    for axis in 0..3 {
        for sign in [-1.0, 1.0] {
            for r in [1.1, 2.2] {
                points.push(frame.center + frame.orientation.col[axis] * (sign * r));
            }
        }
    }
    assert_eq!(points.len(), 47);
    let options = GenerationOptions {
        electrostatic_points: points.clone(),
        electrostatic_weight: 1e6,
        ..GenerationOptions::default()
    };
    let candidate = generate_methyl(&source, spec(), options).unwrap();
    let saved = candidate.to_json().unwrap();
    if let Ok(path) = std::env::var("DFT_RS_FRAGMENT_DIAGNOSTIC_MODEL") {
        std::fs::write(path, &saved).unwrap();
    } else {
        println!("ESP-variant_raw_json={saved}");
    }
    println!("ESP-variant constructionpoints={} weight={:.12e} fingerprint={} sigmaDataRank={}/{} electronicDataRank={}/{} E0={:.12e} status=construction-candidate-only",
        points.len(),candidate.options.electrostatic_weight,candidate.construction_fingerprint,candidate.diagnostics.sigma_projection.data_rank,candidate.diagnostics.sigma_projection.free_parameters,
        candidate.diagnostics.electronic_projection.data_rank,candidate.diagnostics.electronic_projection.free_parameters,candidate.internal_energy_reference);
    range("ESP-variant-monopoles", &candidate.gaussian_charges);
    range(
        "ESP-variant-bodydipoles",
        &candidate
            .gaussian_body_dipoles
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
    );
    range("ESP-variant-local", &candidate.local_amplitudes);
    range("ESP-variant-nonlocal", &candidate.nonlocal_coefficients);
    let basis = MolBasis::build(&source, &data::load("sto-3g").unwrap()).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let options = opts();
    let quadrature = grid(&source);
    let full = run_scf(&source, &basis, &options, &functional, Some(&quadrature)).unwrap();
    let selection = extract_methyl_frozen_space(
        &source,
        &basis,
        &full,
        MethylBoundary {
            carbon: 4,
            hydrogens: [5, 6, 7],
            port_atom: 0,
        },
        &ExtractionOptions::default(),
    )
    .unwrap();
    let frozen = FrozenReferenceModel::new(
        &source,
        &basis,
        &quadrature,
        &functional,
        &options,
        &selection,
    )
    .unwrap();
    let exact = frozen.frozen_core_potential(&points).unwrap();
    let sites = candidate
        .gaussian_sites(frame.center, frame.orientation)
        .unwrap();
    let corr = electrostatic_potential(&sites, &points).unwrap();
    let mut coredefect = 0.0;
    let mut residual2 = 0.0;
    let mut max = 0.0_f64;
    for i in 0..points.len() {
        let fitted = 1.0 / (points[i] - frame.center).norm() + corr[i];
        let error = fitted - exact[i];
        residual2 += error * error;
        max = max.max(error.abs());
        if i < 4 {
            coredefect += f64::from(source.atoms[i].z) * error;
        }
        println!("ESP-variant physical-projection point={i} exact={:.12e} fitted={fitted:.12e} defect={error:.12e}",exact[i]);
    }
    println!("ESP-variant coreEnergyDefect={coredefect:.12e} physicalPotentialRMSE={:.12e} physicalPotentialMax={max:.12e} constructionOnly=true",(residual2/points.len() as f64).sqrt());
    for offset in [-0.001, 0.001] {
        let shifted: Vec<_> = points[..4]
            .iter()
            .map(|&p| p - direction * offset)
            .collect();
        let exact = frozen.frozen_core_potential(&shifted).unwrap();
        let sites = candidate
            .gaussian_sites(frame.center + direction * offset, frame.orientation)
            .unwrap();
        let corrections = electrostatic_potential(&sites, &points[..4]).unwrap();
        let mut exactcore = 0.0;
        let mut fittedcore = 0.0;
        for i in 0..4 {
            let z = f64::from(source.atoms[i].z);
            exactcore += z * exact[i];
            fittedcore +=
                z * (1.0 / (points[i] - frame.center - direction * offset).norm() + corrections[i]);
        }
        println!("ESP-variant rigidCore offset={offset:.12e} exact={exactcore:.12e} fitted={fittedcore:.12e} defect={:.12e}",fittedcore-exactcore);
    }
    let mut curves = Vec::new();
    for offset in [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20] {
        let mut y = reference.clone();
        y[center_index] += direction * offset;
        let reduced =
            evaluate_candidate(&candidate, &y, Vec3::zero(), Vec3::new(0.11, 0.07, -0.09)).unwrap();
        let explicit = map.expand(&y).unwrap();
        let mb = MolBasis::build(&explicit, &data::load("sto-3g").unwrap()).unwrap();
        let g = grid(&explicit);
        let full = run_scf(&explicit, &mb, &options, &functional, Some(&g)).unwrap();
        let fullgradient =
            assemble_gradient(&explicit, &mb, &functional, Some(&g), &full, &options).unwrap();
        let fullforces: Vec<_> = fullgradient.iter().map(|&g| g * -1.0).collect();
        let forces = map.pullback_forces(&y, &fullforces).unwrap();
        let fullforce = forces[center_index].dot(direction);
        let reducedforce = reduced.forces[center_index].dot(direction);
        println!("ESP-variant curve offset={offset:.12e} Efull={:.12e} Efull_elec={:.12e} Efull_class={:.12e} Ereduced={:.12e} Ereduced_elec={:.12e} Ereduced_class={:.12e} forceFull={fullforce:.12e} forceReduced={reducedforce:.12e}",full.energy,full.e_elec,full.e_nuc,reduced.energy,reduced.electronic_energy,reduced.classical_energy);
        curves.push((offset, full.energy, reduced.energy, fullforce, reducedforce));
    }
    let zero = curves[4];
    let mut mae = 0.0;
    let mut force2 = 0.0;
    let mut max = 0.0_f64;
    for &(offset, full, reduced, ff, rf) in &curves {
        let relative = (reduced - zero.2) - (full - zero.1);
        mae += relative.abs();
        force2 += (rf - ff).powi(2);
        max = max.max(relative.abs());
        println!(
            "ESP-variant relative offset={offset:.12e} error={relative:.12e} forceError={:.12e}",
            rf - ff
        );
    }
    println!("ESP-variant summary points=9 relativeEnergyMAE={:.12e} relativeEnergyMax={max:.12e} stretchForceRMSE={:.12e} referenceAbsoluteOffset={:.12e} acceptance=not-established no-G-fit no-E0-reset no-parameter-search",mae/9.0,(force2/9.0).sqrt(),zero.2-zero.1);
    assert_eq!(saved, candidate.to_json().unwrap());
}
