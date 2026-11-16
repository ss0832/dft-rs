// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Physical complex-field SCF, independent one-electron cancellation,
//! time reversal and four-step variational energy derivative acceptance.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::basis::{BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::math::Vec3;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::magnetic_scf::StaticPeriodicField;
use dft_rs::periodic::scf_boundary::{
    run_dimensional_field_scf, DimensionalScfBackend, DimensionalScfOptions,
};
use dft_rs::periodic::{BlochMatrix, Lattice, PeriodicGeometry};
use dft_rs::response::complex::Complex;
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

fn basis(mol: &Molecule) -> MolBasis {
    let mut set = BasisSet::new("compact-magnetic-oracle");
    set.insert(ElementBasis {
        z: 1,
        shells: vec![
            Shell {
                l: 0,
                exps: vec![0.7],
                coefs: vec![1.0],
            },
            Shell {
                l: 1,
                exps: vec![0.5],
                coefs: vec![1.0],
            },
        ],
    });
    MolBasis::build(mol, &set).unwrap()
}
fn options(boundary: PeriodicBoundary) -> DimensionalScfOptions {
    let mut o = DimensionalScfOptions::for_boundary(boundary);
    o.basis_cutoff = 4.3;
    o.grid_cutoff = 10.0;
    o.scf.exchange_cutoff = Some(4.3);
    o.scf.e_tol = 1e-13;
    o.scf.p_tol = 1e-12;
    o.scf.conv_tol_grad = 1e-11;
    o.scf.max_iter = 150;
    o.electrostatics.alpha = Some(0.55);
    o.electrostatics.real_cutoff = Some(16.0);
    o.electrostatics.reciprocal_cutoff = Some(8.0);
    o
}
fn field(g: &PeriodicGeometry, amplitude: f64) -> StaticPeriodicField {
    StaticPeriodicField {
        wave: g.reciprocal()[0],
        vector_potential: [Complex::ZERO, Complex::new(0.37, 0.12), Complex::ZERO],
        scalar_potential: Complex::ZERO,
        amplitude,
    }
}
#[test]
fn complex_one_electron_hf_cancels_self_interaction_and_passes_four_step_field_fit() {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::new(0.3, -0.1, 0.2),
    }])
    .with_multiplicity(2);
    let mb = basis(&mol);
    let xc = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    for axes in [[true, false, false], [true, true, false], [true; 3]] {
        let b = PeriodicBoundary::new(axes);
        let g = PeriodicGeometry::new(Lattice::cubic(40.0), b).unwrap();
        let o = options(b);
        let source = field(&g, 0.4);
        let report = run_dimensional_field_scf(&mol, &mb, &g, &xc, &o, &source).unwrap();
        assert!(report.converged, "{:?}", report.warnings);
        assert!(!report.reference.orbital_conjugation_constraint);
        let tr = report
            .reference
            .time_reversal_diagnostics(o.scf.k_mesh)
            .unwrap();
        println!(
            "complex HF axes={axes:?} E={} density TR={:?} Fock TR={:?} orbital={} \
            components={} J+K={}",
            report.reference.energy,
            tr.orbital_density_residual,
            tr.orbital_fock_residual,
            report.reference_diagnostics.orbital_residual,
            report.component_energy_defect,
            report.hartree_energy + report.full_exchange_energy
        );
        assert!(tr.orbital_density_residual[0] > 1e-4);
        assert!(report.reference_diagnostics.orbital_residual < 1e-10);
        let eri = dft_rs::integrals::eri_incore(&mb, dft_rs::integrals::CoulombKernel::Coulomb, 16)
            .unwrap();
        let p = &report.reference.density[0][0];
        let n = mb.nao_sph;
        let mut j = BlochMatrix::zeros(n, n);
        let mut k = BlochMatrix::zeros(n, n);
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    for d in 0..n {
                        j.re[(a, b)] += p.re[(c, d)] * eri.get(a, b, c, d);
                        k.re[(a, b)] += p.re[(c, d)] * eri.get(a, c, d, b);
                        k.im[(a, b)] += p.im[(c, d)] * eri.get(a, c, d, b);
                    }
                }
            }
        }
        let free_j = 0.5 * p.trace_product(&j);
        let free_k = -0.5 * p.trace_product(&k);
        println!(
            "free one-electron ERI J={free_j} K={free_k} exchange defect={} source_energy={}",
            report.full_exchange_energy - free_k,
            report.reference.e_nuc
        );
        assert!((free_j + free_k).abs() < 1e-12);
        assert!((report.full_exchange_energy - free_k).abs() < 1e-12);
        // The periodic Hartree includes its lattice/gauge contributions; they
        // need not cancel the localized free-Coulomb exchange component alone.
        // Rebuild the frozen Fock and energy using an independent dense ERI K.
        let backend = DimensionalScfBackend::new(&mol, &mb, &g, &xc, &o)
            .unwrap()
            .with_static_field(&source)
            .unwrap();
        let evaluated = backend.evaluate(&report.reference.density).unwrap();
        let periodic_j = &evaluated.coulomb.hartree.as_ref().unwrap()[0];
        let mut dense = backend.h_core()[0].clone();
        dense.add_scaled(periodic_j, 1.0);
        dense.add_scaled(&k, -1.0);
        assert!(dense.re.max_abs_diff(&report.reference.fock[0][0].re) < 1e-11);
        assert!(dense.im.max_abs_diff(&report.reference.fock[0][0].im) < 1e-11);
        let expected = report.reference.e_nuc
            + p.trace_product(&backend.h_core()[0])
            + 0.5 * p.trace_product(periodic_j)
            + free_k;
        assert!((report.reference.energy - expected).abs() < 1e-12);
        let op = source
            .operators(&mb, &g, backend.k_points(), o.basis_cutoff)
            .unwrap();
        let analytic = report.reference.total_density()[0].trace_product(&op.first_derivative[0]);
        let steps = [0.04, 0.02, 0.01, 0.005];
        let mut errors = Vec::new();
        let mut differences = Vec::new();
        for h in steps {
            let plus =
                run_dimensional_field_scf(&mol, &mb, &g, &xc, &o, &field(&g, source.amplitude + h))
                    .unwrap();
            let minus =
                run_dimensional_field_scf(&mol, &mb, &g, &xc, &o, &field(&g, source.amplitude - h))
                    .unwrap();
            assert!(plus.converged && minus.converged);
            let d = (plus.reference.energy - minus.reference.energy) / (2.0 * h);
            differences.push(d);
            errors.push((d - analytic).abs());
        }
        let fit = convergence::error_power_fit(&steps, &errors);
        println!("complex HF field derivative axes={axes:?} steps={steps:?} errors={errors:?} p={:.9} R2={:.10}",fit.exponent,fit.r_squared);
        assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
        let rich = (4.0 * differences[3] - differences[2]) / 3.0;
        assert!((rich - analytic).abs() < 1e-8);
        let reverse =
            run_dimensional_field_scf(&mol, &mb, &g, &xc, &o, &field(&g, -source.amplitude))
                .unwrap();
        assert!((reverse.reference.energy - report.reference.energy).abs() < 1e-11);
        let p = &report.reference.density[0][0];
        let reversed = &reverse.reference.density[0][0];
        assert!(p.re.max_abs_diff(&reversed.re) < 1e-10);
        for (a, b) in p.im.as_slice().iter().zip(reversed.im.as_slice()) {
            assert!((a + b).abs() < 1e-10);
        }
        let (_, shifted) = backend
            .reference_fock_at(&report.reference.density, &[g.reciprocal()[0] * 0.17])
            .unwrap();
        assert!(shifted[0][0].hermiticity_error() < 1e-11);
    }
}

#[test]
fn spinful_time_reversal_is_not_inferred_from_orbital_conjugation_or_net_spin() {
    use dft_rs::periodic::scf::collinear_time_reversal_diagnostics;
    let mut up = BlochMatrix::zeros(2, 2);
    up.re[(0, 0)] = 1.0;
    let mut down = BlochMatrix::zeros(2, 2);
    down.re[(1, 1)] = 1.0;
    let density = [vec![up.clone()], vec![down.clone()]];
    let original = density.clone();
    let tr = collinear_time_reversal_diagnostics(&density, &density, [1; 3]).unwrap();
    assert_eq!(tr.orbital_density_residual, [0.0; 2]);
    assert_eq!(tr.spinful_density_residual, 1.0);
    assert_eq!(
        up.re[(0, 0)] + up.re[(1, 1)],
        down.re[(0, 0)] + down.re[(1, 1)]
    );
    assert!(density[0][0].re.max_abs_diff(&original[0][0].re) == 0.0);
    let symmetric = [vec![up.clone()], vec![up]];
    assert_eq!(
        collinear_time_reversal_diagnostics(&symmetric, &symmetric, [1; 3])
            .unwrap()
            .spinful_density_residual,
        0.0
    );
    assert!(collinear_time_reversal_diagnostics(&density, &density, [2, 1, 1]).is_err());
}

#[test]
fn noncommensurate_static_fields_require_a_compatible_cell() {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }])
    .with_multiplicity(2);
    let mb = basis(&mol);
    let g = PeriodicGeometry::new(Lattice::cubic(8.0), PeriodicBoundary::FULL_3D).unwrap();
    let mut a = field(&g, 1.0);
    a.wave = a.wave * 0.3;
    assert!(a.operators(&mb, &g, &[Vec3::zero()], 4.3).is_err());
}
