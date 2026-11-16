// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! All complex pair-kernel entries against the retained nuclear tensor
//! contraction. Physical response finite-field fits are retained in p318.
use super::*;
use crate::acceptance::c1_geometry;
use crate::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use crate::driver::{Controls, Value};
use crate::xc::HybridParams;

#[test]
fn fixed_spinor_pair_kernel_matches_every_old_complex_entry() {
    let numbers = [1; 4];
    let positions = [
        [0.17, -0.11, 0.23],
        [1.57, 0.18, 0.33],
        [0.31, 4.74, -0.45],
        [1.60, 5.01, -0.61],
    ];
    c1_geometry::assert_c1(&numbers, &positions);
    let controls = Controls::from_value(
        &Value::object()
            .with("xc", "hf")
            .with("dispersion", "none")
            .with("grid_weight_cutoff", 0.0)
            .with("conv_tol", 1e-12)
            .with("conv_tol_dp", 1e-11)
            .with("conv_tol_grad", 1e-10),
    )
    .unwrap();
    let method = MolecularMethod::from_controls(&controls).unwrap();
    let setup = prepare_molecular(&numbers, &positions, &method).unwrap();
    let scf = run_molecular_reference(&setup).unwrap().scf;
    let reference = crate::response::spinor::embedded_collinear_spinor_reference(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        None,
        &scf,
        &setup.options,
        1e-8,
    )
    .unwrap();
    for (cx, clr, omega) in [(1.0, 0.0, 0.0), (0.23, 0.31, 0.37), (0.0, 0.0, 0.0)] {
        let mut functional = crate::xc::ResolvedFunctional::hartree_fock();
        functional.hybrid = HybridParams { cx, clr, omega };
        let n = 2 * setup.basis.nao_sph;
        let (old, _) = prepare_spinor_path(
            &setup.molecule,
            &setup.basis,
            None,
            &functional,
            &reference,
            &setup.options,
            &BlochMatrix::zeros(n, n),
            &[[0.0; 3]; 4],
            [0.0; 3],
            1,
            &ResponseSolverConfig::default(),
        )
        .unwrap();
        let fast = crate::response::spinor::Interaction::at_spinor_reference(
            &setup.molecule,
            &setup.basis,
            &functional,
            None,
            &setup.options,
            &reference.density[0],
        )
        .unwrap();
        let mut worst = 0.0_f64;
        let mut entries = 0usize;
        // Single AO dyads span all real and imaginary transitions, including
        // non-Hermitian frequency-response sources and every transverse block.
        for imaginary in [false, true] {
            for i in 0..n {
                for j in 0..n {
                    let mut p = BlochMatrix::zeros(n, n);
                    if imaginary {
                        p.im[(i, j)] = 1.0;
                    } else {
                        p.re[(i, j)] = 1.0;
                    }
                    let expected = old.interaction(&[cm(&p)], 0).unwrap();
                    let actual = fast.pair_response(&p).unwrap();
                    worst = worst
                        .max(actual.re.max_abs_diff(&expected.re))
                        .max(actual.im.max_abs_diff(&expected.im));
                    entries += 2 * n * n;
                }
            }
        }
        println!("fixed spinor pair cx={cx} clr={clr} omega={omega} real matrix entries={entries} max defect={worst:.12e}");
        assert!(worst < 3e-13);
    }
}
