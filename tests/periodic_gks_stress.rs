// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent relaxed energy differences for the same full spinor model.
use dft_rs::basis::{g94, MolBasis};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::math::Vec3;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::scf_boundary::DimensionalScfOptions;
use dft_rs::periodic::spinor_scf::{
    from_pauli, pauli_components, run_spinor_scf, BiasedSpinorBackend, SpinorDftBackend,
    SpinorEnergyBackend, SpinorScfConfig,
};
use dft_rs::periodic::{BlochMatrix, Lattice, PeriodicGeometry};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

#[path = "support/components.rs"]
pub mod components;
#[path = "support/convergence.rs"]
pub mod convergence;
#[path = "support/stencils.rs"]
pub mod stencils;

const BASIS_TEXT: &str = "H 0\nS 1 1.0\n0.8 1.0\n****\nHe 0\nS 1 1.0\n1.1 1.0\n****\n";

fn model(
    coordinate: usize,
    amount: f64,
) -> (
    Molecule,
    MolBasis,
    PeriodicGeometry,
    DimensionalScfOptions,
    SpinorScfConfig,
) {
    let mut mol = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.13, -0.21, 0.07),
        },
        Atom {
            z: 2,
            position: Vec3::new(2.0, 0.45, -0.31),
        },
    ])
    .with_multiplicity(2);
    let affine = |r: Vec3| {
        if (9..12).contains(&coordinate) {
            let axis = Vec3::new(
                f64::from(coordinate == 9),
                f64::from(coordinate == 10),
                f64::from(coordinate == 11),
            );
            let (sine, cosine) = amount.sin_cos();
            return r * cosine + axis.cross(r) * sine + axis * (axis.dot(r) * (1.0 - cosine));
        }
        let mut x = r.to_array();
        if coordinate < 9 {
            x[coordinate / 3] += amount * x[coordinate % 3];
        }
        Vec3::new(x[0], x[1], x[2])
    };
    for atom in &mut mol.atoms {
        atom.position = affine(atom.position);
    }
    let set = g94::parse("distinct-site-gradient", BASIS_TEXT).unwrap();
    let basis = MolBasis::build(&mol, &set).unwrap();
    let geometry = PeriodicGeometry::new(
        Lattice::new(
            affine(Vec3::new(8.4, 0.7, 0.3)),
            affine(Vec3::new(0.4, 8.9, 0.6)),
            affine(Vec3::new(0.2, -0.5, 9.2)),
        ),
        PeriodicBoundary::FULL_3D,
    )
    .unwrap();
    let mut options = DimensionalScfOptions::for_boundary(geometry.boundary());
    options.basis_cutoff = 3.5;
    options.grid_cutoff = 6.;
    options.scf.exchange_cutoff = Some(3.5);
    options.scf.xc_ao_cutoff = Some(12.);
    options.scf.grid.level = 4;
    options.scf.grid.weight_cutoff = 0.;
    options.electrostatics.alpha = Some(0.5);
    options.electrostatics.real_cutoff = Some(12.);
    options.electrostatics.reciprocal_cutoff = Some(7.);
    options.electrostatics.quadrature_order = 16;
    let config = SpinorScfConfig {
        max_iter: 300,
        energy_tolerance: 1e-12,
        density_tolerance: 1e-11,
        orbital_tolerance: 1e-11,
        damping: 0.2,
        ..SpinorScfConfig::default()
    };
    (mol, basis, geometry, options, config)
}
fn calculation(
    name: &str,
    coordinate: usize,
    amount: f64,
    derivative: bool,
) -> (f64, Option<[f64; 9]>) {
    let saved = std::env::var_os("DFT_RS_GKS_STRESS_VALIDATION_DIR").map(|p| {
        let p = std::path::PathBuf::from(p);
        std::fs::create_dir_all(&p).unwrap();
        p.join(format!(
            "{name}-{coordinate}-{:016x}.json",
            amount.to_bits()
        ))
    });
    let identity = std::env::var("DFT_RS_GKS_VALIDATION_ID").unwrap_or_default();
    if !derivative {
        if let Some(path) = saved.as_ref().filter(|p| p.exists()) {
            let value = Value::parse_json(&std::fs::read_to_string(path).unwrap()).unwrap();
            assert!(!identity.is_empty());
            assert_eq!(
                value.get("identity").and_then(Value::as_str),
                Some(identity.as_str())
            );
            assert_eq!(value.get("converged"), Some(&Value::Bool(true)));
            for (key, tol) in [
                ("orbital_residual", 1e-11),
                ("occupation_consistency", 1e-11),
                ("particle_number_error", 1e-11),
            ] {
                assert!(value.get(key).and_then(Value::as_f64).unwrap() < tol);
            }
            return (value.get("energy").and_then(Value::as_f64).unwrap(), None);
        }
    }
    let (mol, basis, geometry, options, config) = model(coordinate, amount);
    let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
    let backend = SpinorDftBackend::new(
        &mol,
        &basis,
        &geometry,
        &functional,
        &options,
        None,
        vec![BlochMatrix::zeros(2 * basis.nao_sph, 2 * basis.nao_sph)],
    )
    .unwrap();
    let mut seed = backend.initial_fock().to_vec();
    let axis = [0.4, -0.3, 0.75_f64.sqrt()];
    for f in &mut seed {
        let mut pc = pauli_components(f).unwrap();
        for a in 0..basis.nao_sph {
            for c in 0..3 {
                pc[c + 1].re[(a, a)] += 0.06 * (a + 1) as f64 * axis[c];
            }
        }
        *f = from_pauli(&pc).unwrap();
    }
    let seeded = BiasedSpinorBackend::new(&backend, None, Some(seed)).unwrap();
    let report = run_spinor_scf(&seeded, &config).unwrap();

    assert!(report.converged && report.fields.operator_converged);
    assert!(report
        .fields
        .xc_diagnostics
        .as_ref()
        .is_none_or(|d| d.converged));
    println!("GKS_STRESS_SCF functional={name} coordinate={coordinate} amount={amount:.17e} energy={:.17e} orbital={:.17e} occupation={:.17e} number={:.17e}",report.fields.energy,report.orbital_residual,report.occupation_consistency,report.particle_number_error);
    let record = Value::object()
        .with("identity", identity)
        .with("converged", true)
        .with("energy", report.fields.energy)
        .with("orbital_residual", report.orbital_residual)
        .with("occupation_consistency", report.occupation_consistency)
        .with("particle_number_error", report.particle_number_error);
    if let Some(path) = saved {
        if !path.exists() {
            let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
            std::fs::write(&tmp, record.to_json()).unwrap();
            std::fs::rename(tmp, path).unwrap();
        }
    }
    let tensor = if derivative {
        let s = backend.stress(&report, &config).unwrap();
        for a in 0..3 {
            for b in 0..3 {
                let sum = s.kinetic[a][b]
                    + s.electrostatic[a][b]
                    + s.full_exchange[a][b]
                    + s.long_range_exchange[a][b]
                    + s.semilocal_xc[a][b]
                    + s.pulay[a][b];
                assert!((sum - s.strain_derivative[a][b]).abs() < 1e-13);
                assert_eq!(
                    (s.strain_derivative[a][b] / geometry.measure()).to_bits(),
                    s.stress[a][b].to_bits()
                );
            }
        }
        println!("GKS_STRESS_DIAGNOSTICS functional={name} orbital={:.17e} occupation={:.17e} number={:.17e} energy={:.17e} fock={:.17e} xc={:.17e} coulomb={:.17e}",s.fresh_orbital_residual,s.fresh_occupation_consistency,s.fresh_number_error,s.reference_energy_defect,s.reference_fock_defect,s.xc_reference_energy_defect,s.coulomb_reference_energy_defect);
        Some(std::array::from_fn(|c| s.strain_derivative[c / 3][c % 3]))
    } else {
        None
    };
    (report.fields.energy, tensor)
}
fn all_components(name: &str) {
    dft_rs::resources::ResourceLimits::new(Some(1),Some(512)).unwrap().install(|| {
        let (_,derivative)=calculation(name,usize::MAX,0.0,true);let derivative=derivative.unwrap();
        let names:Vec<_>=(0..9).map(|c|format!("GKS full {name} strain[{c}]")).collect();let mut failures=Vec::new();
        stencils::fit_components_staged(&names,&[([0.04,0.02,0.01,0.005],stencils::Stencil::Second)],|steps,stencil| {
            (0..9).map(|c|std::array::from_fn(|i| {
                let h=steps[i];let plus=calculation(name,c,h,false).0;let minus=calculation(name,c,-h,false).0;
                let reference=stencil.first(plus,minus,0.0,0.0,h);let analytic=derivative[c];let error=(reference-analytic).abs();
                println!("GKS_STRESS_SIGNED functional={name} component={c} h={h:.17e} analytic={analytic:.17e} reference={reference:.17e} error={error:.17e}");error
            })).collect()
        },&mut failures);
        components::assert_no_failures("all nine stationary GKS strain derivatives",&failures);
    }).unwrap();
}
#[test]
fn hf_all_nine_total_stress_components() {
    all_components("hf");
}

#[test]
fn dedicated_gks_stress_matches_periodic_energy_report() {
    let (molecule, _, geometry, options, scf) = model(usize::MAX, 0.0);
    let system = System {
        numbers: molecule.atoms.iter().map(|atom| atom.z).collect(),
        positions: molecule
            .atoms
            .iter()
            .map(|atom| atom.position.to_array())
            .collect(),
        cell: Some(geometry.lattice().a.map(|vector| vector.to_array())),
        pbc: [true; 3],
    };
    let path = std::env::temp_dir().join(format!("dft_gks_stress_{}.g94", std::process::id()));
    std::fs::write(&path, BASIS_TEXT).unwrap();
    let mut values = Value::object()
        .with("basis", path.to_str().unwrap())
        .with("xc", "hf")
        .with("reference", "gks")
        .with("multiplicity", 2_usize)
        .with("forces", false)
        .with("overlap_cutoff", options.basis_cutoff)
        .with("grid_cutoff", options.grid_cutoff)
        .with("exchange_cutoff", options.scf.exchange_cutoff)
        .with("xc_ao_cutoff", options.scf.xc_ao_cutoff)
        .with("grid_level", options.scf.grid.level)
        .with("grid_weight_cutoff", options.scf.grid.weight_cutoff)
        .with("ewald_alpha", options.electrostatics.alpha)
        .with("ewald_real_cutoff", options.electrostatics.real_cutoff)
        .with(
            "ewald_reciprocal_cutoff",
            options.electrostatics.reciprocal_cutoff,
        )
        .with(
            "electrostatic_order",
            options.electrostatics.quadrature_order,
        )
        .with("max_iter", scf.max_iter)
        .with("conv_tol", scf.energy_tolerance)
        .with("conv_tol_dp", scf.density_tolerance)
        .with("conv_tol_grad", scf.orbital_tolerance)
        .with("damping", scf.damping)
        .with("no_adiis", true)
        .with("cpu_threads", 1_usize)
        .with("memory_mb", 512_usize);
    let dedicated = run("stress", &system, &Controls::from_value(&values).unwrap());
    values.set("stress", true);
    let energy = run(
        "periodic-energy",
        &system,
        &Controls::from_value(&values).unwrap(),
    );
    std::fs::remove_file(path).unwrap();
    let (dedicated, energy) = (dedicated.unwrap(), energy.unwrap());
    assert_eq!(
        dedicated.get("feature").and_then(Value::as_str),
        Some("stress")
    );
    assert_eq!(
        dedicated.get("converged").and_then(Value::as_bool),
        Some(true)
    );
    for key in [
        "stress",
        "strain_derivative",
        "stress_diagnostics",
        "pressure",
    ] {
        assert_eq!(
            dedicated.get(key).unwrap(),
            energy.get(key).unwrap(),
            "{key}"
        );
    }
}

#[test]
fn lda_all_nine_total_stress_components() {
    all_components("svwn5");
}

#[test]
fn pruned_lda_grid_preserves_all_proper_rigid_rotations() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(512))
        .unwrap()
        .install(|| {
            let (reference, strain) = calculation("svwn5", usize::MAX, 0.0, true);
            let strain = strain.unwrap();
            let max_torque = [(7, 5), (2, 6), (3, 1)]
                .into_iter()
                .map(|(a, b)| (strain[a] - strain[b]).abs())
                .fold(0.0, f64::max);
            assert!(max_torque < 2e-11, "stationary torque {max_torque:.12e} Ha");
            // A rigid rotation is exactly flat on the co-rotating grid.
            // Check both signs of every original angle against the reference
            // absolute energy. This identity has no truncation exponent.
            let mut maximum: f64 = 0.0;
            for axis in 0..3 {
                for step in [0.04, 0.02, 0.01, 0.005] {
                    for angle in [-step, step] {
                        let value = calculation("svwn5", 9 + axis, angle, false).0;
                        let error = (value - reference).abs();
                        maximum = maximum.max(error);
                        assert!(error < 2e-11, "axis {axis} angle {angle} error {error:.12e} Ha");
                    }
                }
            }
            println!("pruned SVWN5 bulk proper rotations: largest energy defect {maximum:.12e} Ha, largest torque {max_torque:.12e} Ha");
        })
        .unwrap();
}
#[test]
fn pbe_all_nine_total_stress_components() {
    all_components("pbe");
}
#[test]
fn tpss_all_nine_total_stress_components() {
    all_components("tpss");
}
#[test]
fn pbe0_all_nine_total_stress_components() {
    all_components("pbe0");
}
