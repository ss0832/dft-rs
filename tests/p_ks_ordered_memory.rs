// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! A large grid must not retain one pair of dense potentials per batch.

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::MolecularGrid;
use dft_rs::ks::xc_contribution;
use dft_rs::linalg::Matrix;
use dft_rs::logger::{LogLevel, LogMode, LogOptions};
use dft_rs::math::Vec3;
use dft_rs::resources::{current_usage, ResourceLimits, MIB};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

#[test]
fn complete_large_grid_fits_a_bounded_partial_budget() {
    let mol = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.13, -0.21, 0.19),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.57, 0.16, -0.08),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.19, 1.93, 0.37),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.42, 0.29, 2.28),
        },
    ]);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let n = mb.nao_sph;
    let mut alpha = Matrix::zeros(n, n);
    let mut beta = Matrix::zeros(n, n);
    for i in 0..n {
        alpha[(i, i)] = 0.31 + 0.03 * i as f64;
        beta[(i, i)] = 0.63 * alpha[(i, i)];
    }
    let functional = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let count = 1 << 21;
    let make_grid = |len| MolecularGrid {
        points: vec![Vec3::new(0.37, -0.14, 0.51); len],
        weights: vec![1.0 / count as f64; len],
        raw_weights: vec![0.0; len],
        atom_of: vec![0; len],
    };
    let reference = xc_contribution(&functional, &mb, &make_grid(256), [&alpha, &beta]).unwrap();
    let grid = make_grid(count);
    let log = LogOptions::new(LogMode::None, None, LogLevel::Info).unwrap();
    let limits = ResourceLimits::new(Some(1), Some(1)).unwrap();
    log.run_with_limits(limits, || {
        let actual = xc_contribution(&functional, &mb, &grid, [&alpha, &beta])?;
        let scale = (count / 256) as f64;
        let check = |got: f64, expected: f64| {
            assert!(got.is_finite() && expected.is_finite());
            assert!((got - expected).abs() <= 5e-11 * expected.abs().max(1.0));
        };
        check(actual.exc, scale * reference.exc);
        check(actual.n_elec, scale * reference.n_elec);
        for spin in 0..2 {
            for (got, expected) in actual.vxc[spin]
                .as_slice()
                .iter()
                .zip(reference.vxc[spin].as_slice())
            {
                check(*got, scale * expected);
            }
        }
        let usage = current_usage().unwrap();
        assert!(usage.peak_reserved_bytes <= MIB);
        println!(
            "points={count} managed_peak_bytes={}",
            usage.peak_reserved_bytes
        );
        Ok(())
    })
    .unwrap();
}
