// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independently rebuilt magnetic-torus SCF energies against relaxed nuclear
//! energy derivatives. No finite difference is used in the response provider.
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/crystals.rs"]
mod crystals;
use dft_rs::basis::{BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::math::Vec3;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::magnetic_cell::{MagneticCell, MagneticDftBackend, MagneticScfOptions};
use dft_rs::periodic::spinor_scf::{run_spinor_scf, SpinorEnergyBackend, SpinorScfConfig};
use dft_rs::periodic::{Lattice, PeriodicGeometry};
use dft_rs::response::ResponseSolverConfig;
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule};
use std::f64::consts::PI;
const D: [[f64; 3]; 2] = [[0.07, -0.05, -0.3], [-0.11, 0.09, 0.3]];

fn accept(range_exchange: bool, charge: i32, electric: bool, magnetic: bool) {
    let field = if !magnetic {
        Vec3::zero()
    } else if charge != 0 {
        Vec3::new(2.0 * PI / 25.0, 2.0 * PI / 25.0, 2.0 * PI / 25.0)
    } else {
        Vec3::new(0.0, 0.0, 2.0 * PI / 25.0)
    };
    accept_in_cell(
        range_exchange,
        charge,
        electric,
        magnetic,
        Lattice::cubic(5.0),
        field,
    );
}
fn accept_in_cell(
    range_exchange: bool,
    charge: i32,
    electric: bool,
    magnetic: bool,
    lattice: Lattice,
    field: Vec3,
) {
    accept_geometry(
        range_exchange,
        charge,
        electric,
        magnetic,
        lattice,
        field,
        Vec3::zero(),
        D,
    );
}
fn accept_geometry(
    range_exchange: bool,
    charge: i32,
    electric: bool,
    magnetic: bool,
    lattice: Lattice,
    field: Vec3,
    shift: Vec3,
    direction: [[f64; 3]; 2],
) {
    let mut functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    if range_exchange {
        functional.hybrid.cx = 0.65;
        functional.hybrid.clr = 0.25;
        functional.hybrid.omega = 0.3;
    }
    let mut set = BasisSet::new("magnetic-nuclear-scf-validation");
    set.insert(ElementBasis {
        z: 1,
        shells: vec![Shell {
            l: 0,
            exps: vec![0.7],
            coefs: vec![1.0],
        }],
    });
    let mut options = MagneticScfOptions {
        electric_field: if electric {
            [0.0007, 0.0005, -0.001]
        } else {
            [0.0; 3]
        },
        basis_cutoff: 12.0,
        reciprocal_cutoff: 16.0,
        operator_tolerance: 1e-10,
        nuclear_soc_scale: if range_exchange || charge != 0 {
            1.0
        } else {
            0.0
        },
        ..Default::default()
    };
    options.electrostatics.alpha = Some(0.55);
    options.electrostatics.real_cutoff = Some(16.0);
    options.electrostatics.reciprocal_cutoff = Some(8.0);
    if charge != 0 {
        options.electrostatics.compensation =
            dft_rs::periodic::electrostatics::ChargeCompensation::UniformBackground;
    }
    let config = SpinorScfConfig {
        max_iter: 500,
        energy_tolerance: 1e-14,
        density_tolerance: 1e-13,
        orbital_tolerance: 1e-13,
        ..Default::default()
    };
    let response = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let molecule_at = |t: f64| {
        let mut molecule = Molecule::new(
            [[0.1, 0.05, -0.7], [-0.1, -0.05, 0.7]]
                .iter()
                .zip(direction)
                .map(|(r, d)| Atom {
                    z: 1,
                    position: Vec3::new(r[0] + t * d[0], r[1] + t * d[1], r[2] + t * d[2]) + shift,
                })
                .collect(),
        );
        molecule.charge = charge;
        molecule
    };
    let prepare = |t: f64| {
        let molecule = molecule_at(t);
        let basis = MolBasis::build(&molecule, &set).unwrap();
        let cell = MagneticCell::new(
            PeriodicGeometry::new(lattice.clone(), PeriodicBoundary::FULL_3D).unwrap(),
            field,
            Vec3::zero(),
        )
        .unwrap();
        MagneticDftBackend::new(&molecule, &basis, cell, &functional, options.clone(), None)
            .unwrap()
    };
    let backend = prepare(0.0);
    let reference = run_spinor_scf(&backend, &config).unwrap();
    assert!(
        reference.converged,
        "reference residual {} operator {}",
        reference.orbital_residual, reference.fields.operator_converged
    );
    let analytic = backend
        .nuclear_energy_taylor(&reference, &direction, 4, &response)
        .unwrap();
    assert!(analytic
        .response_residuals
        .iter()
        .all(|r| *r <= response.tolerance));
    assert!(analytic.projector_defects.iter().all(|r| *r < 1e-10));
    assert!(analytic.stationarity_defects.iter().all(|r| *r < 1e-10));
    let mut cache = std::collections::BTreeMap::new();
    let mut point = |t: f64| {
        *cache.entry(t.to_bits()).or_insert_with(|| {
        let current=prepare(t);
        let state=run_spinor_scf(&current,&config).unwrap();
        assert!(state.converged);
        if charge!=0 {
            let bare=dft_rs::dfpt::orbitals::generalized_orbitals(&current.initial_fock()[0],&current.overlap()[0],1e-12).unwrap();
            let defect=if !electric {
                (state.fields.energy-state.fields.energy_components[0]-bare.energies[0]).abs()
            } else {
                // Independent occupied one-electron expectation, not the
                // production full-space inverse/determinant construction.
                use dft_rs::response::complex::Complex as Z;
                let molecule=molecule_at(t);
                let basis=MolBasis::build(&molecule,&set).unwrap();
                let coefficients=&state.bands[0].coefficients;
                let charge_centroid=molecule.atoms.iter().fold(Vec3::zero(),|r,a| r+a.position*(a.z as f64))
                    -lattice.a.iter().copied().fold(Vec3::zero(),|r,a| r+a)*0.5*(charge as f64);
                let mut berry=0.0;
                for axis in 0..3 {
                    let wave=current.cell.geometry.lattice().reciprocal()[axis];
                    let q=dft_rs::periodic::spinor_scf::spin_diagonal(&current.cell.fourier(&basis,options.basis_cutoff,wave).unwrap());
                    let mut expectation=Z::ZERO;
                    for i in 0..q.rows() { for j in 0..q.cols() {
                        let ci=Z::new(coefficients.re[(i,0)],coefficients.im[(i,0)]);
                        let cj=Z::new(coefficients.re[(j,0)],coefficients.im[(j,0)]);
                        expectation+=ci.conj()*Z::new(q.re[(i,j)],q.im[(i,j)])*cj;
                    }}
                    let a=current.cell.geometry.lattice().a[axis];
                    let reduced=(options.electric_field[0]*a.x+options.electric_field[1]*a.y+options.electric_field[2]*a.z)/(2.0*PI);
                    berry+=reduced*(expectation.im.atan2(expectation.re)-wave.dot(charge_centroid));
                }
                let density_energy=state.density[0].trace_product(&current.initial_fock()[0]);
                assert!((berry-state.fields.energy_components[8]).abs()<2e-12);
                (state.fields.energy-state.fields.energy_components[0]-density_energy-berry).abs()
            };
            let interaction=state.fields.energy_components[4]+state.fields.energy_components[5];
            let size=state.density[0].rows()/2;
            let coherence=(0..size).flat_map(|i| (0..size).map(move |j| (i,j))).map(|(i,j)| state.density[0].re[(i,size+j)].hypot(state.density[0].im[(i,size+j)])).fold(0.0,f64::max);
            println!("charged magnetic spinor oracle t={t:.16e} energy_defect={defect:.16e} self_interaction={interaction:.16e} spin_coherence={coherence:.16e}");
            assert!(defect<2e-12 && interaction.abs()<2e-12 && coherence>1e-3);
        }
        if electric {
            let dipole=current.electric_dipole(&state.density).unwrap().unwrap();
            let work=(0..3).map(|a| options.electric_field[a]*dipole[a]).sum::<f64>();
            assert!((state.fields.energy_components[8]+work).abs()<2e-12);
        }
        let slope=current.nuclear_energy_slope(&state,&direction).unwrap();
        println!("magnetic nuclear SCF range={range_exchange} charge={charge} electric={electric} magnetic={magnetic} t={t:.16e} energy={:.16e} slope={slope:.16e} residual={:.16e} number={:.16e}",state.fields.energy,state.orbital_residual,state.particle_number_error);
        (state.fields.energy,slope)
    })
    };
    let zero = point(0.0);
    assert!((zero.1 - analytic.energy[1]).abs() < 2e-12);
    let steps = [0.12, 0.09, 0.06, 0.03, 0.0225, 0.015, 0.0075];
    let mut approximations = Vec::new();
    let mut force_estimates = Vec::new();
    for h in steps {
        let (p, m, pp, mm) = (point(h), point(-h), point(2.0 * h), point(-2.0 * h));
        approximations.push([
            (p.0 - m.0) / (2.0 * h),
            (p.0 - 2.0 * zero.0 + m.0) / (h * h),
            (pp.0 - 2.0 * p.0 + 2.0 * m.0 - mm.0) / (2.0 * h * h * h),
            (pp.0 - 4.0 * p.0 + 6.0 * zero.0 - 4.0 * m.0 + mm.0) / (h * h * h * h),
        ]);
        force_estimates.push([
            (p.0 - m.0) / (2.0 * h),
            (p.1 - m.1) / (2.0 * h),
            (p.1 - 2.0 * zero.1 + m.1) / (h * h),
            (pp.1 - 2.0 * p.1 + 2.0 * m.1 - mm.1) / (2.0 * h * h * h),
        ]);
    }
    let mut fits = Vec::new();
    for n in 1..=4 {
        let target = analytic.energy[n] * (1..=n).product::<usize>() as f64;
        let errors: Vec<_> = approximations
            .iter()
            .map(|e| (e[n - 1] - target).abs())
            .collect();
        let coarse = convergence::error_power_fit(&steps[..4], &errors[..4]);
        println!(
            "magnetic nuclear coarse range={range_exchange} charge={charge} electric={electric} magnetic={magnetic} order={n} p={:.9} R2={:.12}",
            coarse.exponent, coarse.r_squared
        );
        let energy_rich =
            ((4.0 * approximations[3][n - 1] - approximations[2][n - 1]) / 3.0 - target).abs();
        println!("magnetic nuclear independent-energy range={range_exchange} charge={charge} electric={electric} magnetic={magnetic} order={n} all_steps={steps:?} all_errors={errors:?} coarse_Richardson={energy_rich:.16e}; fourth energy differences are cancellation-limited on finer steps");
        assert!(energy_rich < 2e-6);
        if n < 4 {
            assert!((coarse.exponent - 2.0).abs() < 0.05 && coarse.r_squared > 0.9999);
        }
        let errors: Vec<_> = force_estimates
            .iter()
            .map(|e| (e[n - 1] - target).abs())
            .collect();
        let interval = if n == 4 { 3..7 } else { 0..4 };
        let fit = convergence::error_power_fit(&steps[interval.clone()], &errors[interval.clone()]);
        let fine = interval.end - 1;
        let ratio = (steps[fine - 1] / steps[fine]).powi(2);
        let rich = ((ratio * force_estimates[fine][n - 1] - force_estimates[fine - 1][n - 1])
            / (ratio - 1.0)
            - target)
            .abs();
        println!("magnetic nuclear fit range={range_exchange} charge={charge} electric={electric} magnetic={magnetic} order={n} target={target:.16e} all_steps={steps:?} all_errors={errors:?} accepted_interval={:?} p={:.9} R2={:.12} Richardson={rich:.16e} response={:?} projector={:?} stationarity={:?}",&steps[interval],fit.exponent,fit.r_squared,analytic.response_residuals,analytic.projector_defects,analytic.stationarity_defects);
        fits.push((n, fit, rich));
    }
    for (n, fit, rich) in fits {
        assert!(
            (fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999 && rich < 2e-6,
            "range={range_exchange} charge={charge} electric={electric} magnetic={magnetic} order={n}"
        );
    }
}
#[test]
fn relaxed_point_nuclear_hf_magnetic_torus_orders_one_through_four() {
    accept(false, 0, false, true);
}
#[test]
fn relaxed_range_exchange_and_generated_soc_magnetic_nuclear_orders() {
    accept(true, 0, false, true);
}

#[test]
fn charged_coherent_spinor_nuclear_orders_with_three_face_flux_and_soc() {
    accept(false, 1, false, true);
}

#[test]
fn joint_bias_hf_magnetic_nuclear_orders_one_through_four() {
    accept(false, 0, true, true);
}
#[test]
fn joint_bias_range_exchange_soc_nuclear_orders_one_through_four() {
    accept(true, 0, true, true);
}
#[test]
fn joint_bias_charged_coherent_spinor_nuclear_orders_one_through_four() {
    accept(false, 1, true, true);
}
#[test]
fn joint_bias_zero_magnetic_limit_electric_nuclear_orders_one_through_four() {
    accept(false, 0, true, false);
}

#[test]
fn joint_bias_nuclear_orders_in_the_remaining_six_crystal_metrics() {
    let names = [
        "cubic",
        "tetragonal",
        "orthorhombic",
        "hexagonal",
        "rhombohedral",
        "monoclinic",
        "triclinic",
    ];
    for (index, rows) in crystals::rows().iter().enumerate().skip(1) {
        let a = rows.map(|r| Vec3::new(r[0], r[1], r[2]));
        let lattice = Lattice::new(a[0], a[1], a[2]);
        let field = a[2] * (2.0 * PI / lattice.volume());
        println!(
            "joint field crystal {} lattice={:?} field={field:?}",
            names[index], lattice.a
        );
        accept_in_cell(false, 0, true, true, lattice, field);
    }
}

#[test]
fn joint_bias_branch_cut_nuclear_orders_have_independent_force_fits() {
    let mut direction = D;
    for d in &mut direction {
        d[2] += 0.07;
    }
    accept_geometry(
        false,
        0,
        true,
        true,
        Lattice::cubic(5.0),
        Vec3::new(0.0, 0.0, 2.0 * PI / 25.0),
        Vec3::new(0.0, 0.0, 1.25),
        direction,
    );
}
