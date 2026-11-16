// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Full Hermitian-density Hessian of the periodic spinor energy.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::math::Vec3;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::scf_boundary::DimensionalScfOptions;
use dft_rs::periodic::spinor_scf::{SpinorDftBackend, SpinorEnergyBackend};
use dft_rs::periodic::{BlochMatrix, Lattice, PeriodicGeometry};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

fn directions() -> Vec<BlochMatrix> {
    // Mix all Hermitian coordinates with an orthonormal cosine transform.
    // This gives all 136 independent Hessian projections without isolating
    // parity-forbidden charge/current entries with structural zero errors.
    let mut generators = Vec::new();
    for i in 0..4 {
        let mut p = BlochMatrix::zeros(4, 4);
        p.re[(i, i)] = 1.0;
        generators.push(p);
        for j in 0..i {
            let mut real = BlochMatrix::zeros(4, 4);
            real.re[(i, j)] = std::f64::consts::FRAC_1_SQRT_2;
            real.re[(j, i)] = std::f64::consts::FRAC_1_SQRT_2;
            generators.push(real);
            let mut imag = BlochMatrix::zeros(4, 4);
            imag.im[(i, j)] = std::f64::consts::FRAC_1_SQRT_2;
            imag.im[(j, i)] = -std::f64::consts::FRAC_1_SQRT_2;
            generators.push(imag);
        }
    }
    (0..16)
        .map(|k| {
            let mut p = BlochMatrix::zeros(4, 4);
            for (i, mode) in generators.iter().enumerate() {
                let angle = std::f64::consts::PI * (i as f64 + 0.5) * k as f64 / 16.0;
                let scale = if k == 0 { 0.25 } else { 0.125_f64.sqrt() };
                p.add_scaled(mode, scale * angle.cos());
            }
            p
        })
        .collect()
}

#[test]
fn noncollinear_periodic_fock_fits_every_density_hessian_component() {
    let molecule = Molecule::new(vec![
        Atom {
            z: 3,
            position: Vec3::new(0.17, -0.29, 0.33),
        },
        Atom {
            z: 1,
            position: Vec3::new(2.89, 0.24, -0.18),
        },
    ]);
    let mut set = BasisSet::new("lithium hydride kernel validation");
    for (z, exponent) in [(3, 0.61), (1, 0.83)] {
        set.insert(ElementBasis {
            z,
            shells: vec![Shell {
                l: 0,
                exps: vec![exponent],
                coefs: vec![1.0],
            }],
        });
    }
    let basis = MolBasis::build(&molecule, &set).unwrap();
    let geometry = PeriodicGeometry::new(
        Lattice::new(
            Vec3::new(7.1, 0.3, -0.2),
            Vec3::new(-0.4, 8.3, 0.7),
            Vec3::new(0.2, -0.6, 8.9),
        ),
        PeriodicBoundary::FULL_3D,
    )
    .unwrap();
    let mut options = DimensionalScfOptions::for_boundary(geometry.boundary());
    options.basis_cutoff = 4.0;
    options.grid_cutoff = 6.0;
    options.scf.xc_ao_cutoff = Some(12.0);
    options.scf.exchange_cutoff = Some(4.0);
    options.scf.grid.level = 1;
    options.scf.grid.weight_cutoff = 0.0;
    let mut functional = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    // Both exchange kernels are active along with local spin-density XC.
    functional.hybrid.cx = 0.18;
    functional.hybrid.clr = 0.29;
    functional.hybrid.omega = 0.31;
    let backend = SpinorDftBackend::new(
        &molecule,
        &basis,
        &geometry,
        &functional,
        &options,
        None,
        vec![BlochMatrix::zeros(4, 4)],
    )
    .unwrap();
    let directions = directions();
    for (i, a) in directions.iter().enumerate() {
        for (j, b) in directions.iter().enumerate() {
            assert!((a.trace_product(b) - f64::from(i == j)).abs() < 3e-15);
        }
    }
    let mut density = BlochMatrix::zeros(4, 4);
    for (i, occupation) in [0.62, 0.79, 0.32, 0.53].into_iter().enumerate() {
        density.re[(i, i)] = occupation;
    }
    for (i, direction) in directions.iter().enumerate() {
        density.add_scaled(direction, 0.035 * (0.7 * (i + 1) as f64).cos());
    }
    let density = [density];
    let kernel = backend.orbital_kernel(&density).unwrap();
    let mut max_reciprocity = 0.0_f64;
    let analytic: Vec<Vec<f64>> = directions
        .iter()
        .map(|source| {
            let potential = kernel.apply(std::slice::from_ref(source)).unwrap();
            directions
                .iter()
                .map(|probe| probe.trace_product(&potential[0]))
                .collect()
        })
        .collect();
    for i in 0..16 {
        for j in 0..i {
            max_reciprocity = max_reciprocity.max((analytic[i][j] - analytic[j][i]).abs());
        }
    }
    assert!(
        max_reciprocity < 1e-13,
        "reciprocity defect {max_reciprocity:e}"
    );
    let coordinates: Vec<_> = (0..16).flat_map(|i| (i..16).map(move |j| (i, j))).collect();
    let names: Vec<_> = coordinates
        .iter()
        .map(|(i, j)| format!("LiH spinor kernel[{i},{j}]"))
        .collect();
    let mut failures = Vec::new();
    let mut cache = std::collections::BTreeMap::new();
    let mut fock = |i: usize, h: f64| {
        cache
            .entry((i, h.to_bits()))
            .or_insert_with(|| {
                let mut trial = density[0].clone();
                trial.add_scaled(&directions[i], h);
                backend.evaluate(&[trial]).unwrap().fock.remove(0)
            })
            .clone()
    };
    let mut max_richardson = 0.0_f64;
    stencils::fit_components_staged(
        &names,
        &[
            ([0.048, 0.032, 0.024, 0.016], stencils::Stencil::Second),
            ([0.016, 0.012, 0.008, 0.006], stencils::Stencil::Second),
            ([0.048, 0.032, 0.024, 0.016], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            coordinates
                .iter()
                .map(|&(i, j)| {
                    let estimate = steps.map(|h| {
                        let work = |p: &BlochMatrix| directions[j].trace_product(p);
                        let p = work(&fock(i, h));
                        let m = work(&fock(i, -h));
                        let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                            (work(&fock(i, 2.0 * h)), work(&fock(i, -2.0 * h)))
                        } else {
                            (0.0, 0.0)
                        };
                        stencil.first(p, m, p2, m2, h)
                    });
                    if stencil == stencils::Stencil::Second && steps[0] == 0.048 {
                        let richardson =
                            ((4.0 * estimate[2] - estimate[0]) / 3.0 - analytic[i][j]).abs();
                        max_richardson = max_richardson.max(richardson);
                    }
                    estimate.map(|v| (v - analytic[i][j]).abs())
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("periodic noncollinear LiH Fock kernel", &failures);
    println!("LiH spinor kernel: H00={:.15e} H0,7={:.15e} max_reciprocity={max_reciprocity:.3e} max_Richardson={max_richardson:.3e}", analytic[0][0], analytic[0][7]);
    assert!(max_richardson < 2e-8);
}
