// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Nine-point rigid attachment scan with fixed generated coefficients.
//! Numerical convergence is required; chemical accuracy is measured, not assumed.
use dft_rs::basis::{data, MolBasis};
use dft_rs::driver::continuous_species::reduced::{evaluate_candidate, ReducedEvaluation};
use dft_rs::gradient::assemble_gradient;
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::pseudo_sub::reduction::{generate_methyl, GenerationOptions, ReducedFragment};
use dft_rs::pseudo_sub::reduction_geometry::{AttachmentPort, FragmentSpec};
use dft_rs::response::diagnostics::reference_diagnostics;
use dft_rs::scf::{run_scf, EriMode, Guess, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};
use dft_rs::{DftError, Result};
use std::sync::OnceLock;

fn source_molecule() -> Molecule {
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

fn candidate() -> &'static ReducedFragment {
    static CANDIDATE: OnceLock<ReducedFragment> = OnceLock::new();
    CANDIDATE.get_or_init(|| {
        let spec = FragmentSpec {
            explicit_atoms: vec![4, 5, 6, 7],
            retained_atoms: vec![0, 1, 2, 3],
            port: AttachmentPort {
                fragment_atom: 4,
                host_atom: 0,
                reference_atom: 1,
            },
            single_bonds: vec![[0, 1], [0, 2], [0, 3], [0, 4], [4, 5], [4, 6], [4, 7]],
            electronic_centers: 1,
        };
        generate_methyl(&source_molecule(), spec, GenerationOptions::default())
            .expect("canonical C1 ethane candidate must be generated once before the rigid scan")
    })
}

struct FullReference {
    energy: f64,
    /// J_L^T F_full, not a subset of full atom forces.
    generalized_forces: Vec<Vec3>,
    ao_count: usize,
    grid_points: usize,
    orbital_residual: f64,
}

fn explicit_reference(model: &ReducedFragment, coordinates: &[Vec3]) -> Result<FullReference> {
    let molecule = model.geometry_map.expand(coordinates)?;
    if molecule.n_electrons()? != 18 {
        return Err(DftError::InvalidInput(
            "rigid explicit ethane reference must retain all 18 electrons".into(),
        ));
    }
    let basis = MolBasis::build(&molecule, &data::load("sto-3g")?)?;
    let grid = MolecularGrid::build(
        &molecule,
        &GridOptions {
            level: model.options.grid_level,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        },
    )?;
    let options = ScfOptions {
        guess: Guess::Core,
        max_iter: 300,
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-11,
        eri_mode: EriMode::InCore { limit_mb: 64 },
        ..ScfOptions::default()
    };
    let functional = resolve(&FunctionalSpec::Combined("pbe".into()))?;
    let scf = run_scf(&molecule, &basis, &options, &functional, Some(&grid))?;
    let diagnostics = reference_diagnostics(&scf)?;
    if !scf.converged || diagnostics.orbital_residual > options.conv_tol_grad {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: diagnostics.orbital_residual,
        });
    }
    let gradient = assemble_gradient(&molecule, &basis, &functional, Some(&grid), &scf, &options)?;
    let full_forces: Vec<_> = gradient.iter().map(|&g| g * -1.0).collect();
    let generalized_forces = model
        .geometry_map
        .pullback_forces(coordinates, &full_forces)?;
    Ok(FullReference {
        energy: scf.energy,
        generalized_forces,
        ao_count: basis.nao_sph,
        grid_points: grid.len(),
        orbital_residual: diagnostics.orbital_residual,
    })
}

struct CurvePoint {
    offset: f64,
    distance: f64,
    full: FullReference,
    reduced: ReducedEvaluation,
}

#[test]
fn fixed_nine_point_attachment_curve_reports_relative_energies_and_pulled_back_forces() {
    let model = candidate();
    assert_eq!(model.options.functional, "pbe");
    let saved = model.to_json().unwrap();
    let reference = model.geometry_map.reference_coordinates();
    let center = model.geometry_map.electronic_center_index().unwrap();
    let host = model
        .geometry_map
        .retained_atoms()
        .iter()
        .position(|&a| a == model.spec.port.host_atom)
        .unwrap();
    let bond = reference[center] - reference[host];
    let reference_distance = bond.norm();
    let direction = bond * (1.0 / reference_distance);
    let offsets = [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20];
    let mut points = Vec::new();
    let mut numerical_failures = Vec::new();
    for (index, &offset) in offsets.iter().enumerate() {
        let mut coordinates = reference.clone();
        coordinates[center] += direction * offset;
        let result = explicit_reference(model, &coordinates).and_then(|full| {
            // Fixed coefficients are consumed directly. No generate/refit call occurs here.
            let reduced = evaluate_candidate(
                model,
                &coordinates,
                Vec3::zero(),
                Vec3::new(0.11, 0.07, -0.09),
            )?;
            Ok(CurvePoint {
                offset,
                distance: reference_distance + offset,
                full,
                reduced,
            })
        });
        match result {
            Ok(point) => {
                assert_eq!(
                    (
                        point.reduced.full_electrons,
                        point.reduced.frozen_electrons,
                        point.reduced.active_electrons
                    ),
                    (18, 8, 10)
                );
                assert!(point.reduced.physical_total_charge.abs() < 1e-12);
                assert!((point.reduced.active_density_trace - 10.0).abs() < 1e-9);
                assert_eq!((point.full.ao_count, point.reduced.ao_count), (16, 9));
                println!("rigid attachment raw index={index} offset={offset:.9e} R={:.16e} Efull={:.16e} Ereduced={:.16e} fullSCF={:.9e} reducedSCF={:.9e}",
                    point.distance,point.full.energy,point.reduced.energy,point.full.orbital_residual,point.reduced.scf_orbital_residual);
                points.push(point);
            }
            Err(error) => {
                println!("rigid attachment numerical failure index={index} offset={offset:.16e}: {error}");
                numerical_failures.push((index, error.to_string()));
            }
        }
    }
    assert!(
        numerical_failures.is_empty(),
        "rigid attachment scan has explicit numerical failures: {numerical_failures:?}"
    );
    assert_eq!(points.len(), 9);
    let center_point = points.iter().find(|p| p.offset == 0.0).unwrap();
    let full_reference_energy = center_point.full.energy;
    let reduced_reference_energy = center_point.reduced.energy;
    let mut relative_errors = Vec::new();
    let mut stretch_force_errors = Vec::new();
    let mut generalized_squared = 0.0;
    let mut generalized_components = 0;
    for point in &points {
        let full_relative = point.full.energy - full_reference_energy;
        let reduced_relative = point.reduced.energy - reduced_reference_energy;
        let relative_error = reduced_relative - full_relative;
        let full_stretch_force = point.full.generalized_forces[center].dot(direction);
        let reduced_stretch_force = point.reduced.forces[center].dot(direction);
        let stretch_error = reduced_stretch_force - full_stretch_force;
        relative_errors.push(relative_error.abs());
        stretch_force_errors.push(stretch_error);
        let mut maximum_force_error = 0.0_f64;
        assert_eq!(
            point.full.generalized_forces.len(),
            point.reduced.forces.len()
        );
        for (full, reduced) in point
            .full
            .generalized_forces
            .iter()
            .zip(&point.reduced.forces)
        {
            let error = *reduced - *full;
            for axis in 0..3 {
                maximum_force_error = maximum_force_error.max(error.get(axis).abs());
                generalized_squared += error.get(axis).powi(2);
                generalized_components += 1;
            }
        }
        println!("rigid attachment comparison offset={:.9e} R={:.16e} dEfull={full_relative:.16e} dEreduced={reduced_relative:.16e} relativeEnergyError={relative_error:.16e} forceFull={full_stretch_force:.16e} forceReduced={reduced_stretch_force:.16e} stretchForceError={stretch_error:.16e} maxGeneralizedForceError={maximum_force_error:.16e} energy_units=hartree force_units=hartree/bohr",
            point.offset,point.distance);
    }
    let mae = relative_errors.iter().sum::<f64>() / 9.0;
    let maximum = relative_errors.into_iter().fold(0.0_f64, f64::max);
    let force_rms = (stretch_force_errors.iter().map(|e| e * e).sum::<f64>() / 9.0).sqrt();
    let generalized_rms = (generalized_squared / generalized_components as f64).sqrt();
    println!("rigid attachment summary points=9 numerical_failures=0 relativeEnergyMAE={mae:.16e} relativeEnergyMax={maximum:.16e} stretchForceRMSE={force_rms:.16e} allGeneralizedForceRMSE={generalized_rms:.16e} fullNe=18 frozenNe=8 activeNe=10 fullAO=16 reducedAO=9 fullOccupied=9 reducedOccupied=5 fullResponseVariables=63 reducedResponseVariables=20 fullGrid={} reducedGrid={} localTerms=3 gaussianChargeTerms=3 nonlocalProjectors=8 field=zero internalGeometry=rigid frozenCandidate={} chemicalAccuracyStatus=measured-not-accepted shape={} response={}",
        center_point.full.grid_points,center_point.reduced.grid_points,model.construction_fingerprint,model.shape_status(),model.response_status());
    assert_eq!(saved, model.to_json().unwrap());
    // There is intentionally no assertion against an unestablished chemical
    // accuracy goal. These errors decide the next representation changes.
}
