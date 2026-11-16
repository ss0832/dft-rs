// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Resource contracts, including concurrent independent Rayon/faer settings.

use dft_rs::linalg::{symmetric_eigen, Matrix};
use dft_rs::resources::{
    checked_bytes, current_usage, remaining_bytes, reserve, ResourceLimits, MIB,
};
use dft_rs::DftError;
use rayon::prelude::*;

#[test]
fn invalid_and_overflowing_limits_are_rejected_before_work() {
    for (cpu, memory) in [
        (Some(0), None),
        (Some(usize::MAX), None),
        (None, Some(0)),
        (None, Some(usize::MAX)),
    ] {
        assert!(ResourceLimits::new(cpu, memory).is_err());
    }
    assert!(checked_bytes(&[usize::MAX, 8], "test allocation").is_err());
    assert!(current_usage().is_none());
}

#[test]
fn omitted_memory_limits_preserve_unbudgeted_execution() {
    assert!(current_usage().is_none());
    let matrix = ResourceLimits::new(Some(1), None)
        .unwrap()
        .run(|| {
            let matrix = Matrix::identity(32);
            assert_eq!(current_usage().unwrap().memory_mb, None);
            assert_eq!(current_usage().unwrap().peak_reserved_bytes, 0);
            assert_eq!(remaining_bytes(), None);
            Ok(matrix)
        })
        .unwrap();
    assert_eq!(matrix, Matrix::identity(32));
    assert!(current_usage().is_none());
}

#[test]
fn matrix_budget_refusals_and_size_overflows_unwind_to_typed_errors() {
    let limits = ResourceLimits::new(Some(2), Some(1)).unwrap();
    let error = limits.run(|| Ok(Matrix::zeros(1024, 1024))).unwrap_err();
    assert!(matches!(error, DftError::ResourceLimit { .. }));
    let error = limits.run(|| Ok(Matrix::zeros(usize::MAX, 2))).unwrap_err();
    assert!(matches!(error, DftError::InvalidInput(_)));
    assert!(current_usage().is_none());
    limits
        .run(|| {
            assert_eq!(remaining_bytes(), Some(MIB));
            Ok(())
        })
        .unwrap();
}

#[test]
fn leases_count_live_copies_and_release_the_shared_budget() {
    ResourceLimits::new(Some(2), Some(1))
        .unwrap()
        .run(|| {
            let original = Matrix::zeros(100, 100);
            let copy = original.clone();
            assert_eq!(current_usage().unwrap().reserved_bytes, 2 * 100 * 100 * 8);
            drop(copy);
            assert_eq!(current_usage().unwrap().reserved_bytes, 100 * 100 * 8);
            drop(original);
            let lease = reserve(MIB, "cache fixture")?;
            let shared = lease.clone();
            drop(lease);
            assert_eq!(remaining_bytes(), Some(0));
            drop(shared);
            assert_eq!(remaining_bytes(), Some(MIB));
            assert_eq!(current_usage().unwrap().peak_reserved_bytes, MIB);
            Ok(())
        })
        .unwrap();
}

#[test]
fn parallel_workers_share_one_atomic_budget() {
    ResourceLimits::new(Some(2), Some(1))
        .unwrap()
        .run(|| {
            let barrier = std::sync::Barrier::new(2);
            let accepted: Vec<_> = (0..2)
                .into_par_iter()
                .map(|_| {
                    let lease = reserve(700_000, "parallel cache fixture");
                    barrier.wait();
                    let success = lease.is_ok();
                    drop(lease);
                    success
                })
                .collect();
            assert_eq!(accepted.iter().filter(|&&ok| ok).count(), 1);
            assert_eq!(current_usage().unwrap().peak_reserved_bytes, 700_000);
            assert_eq!(remaining_bytes(), Some(MIB));
            Ok(())
        })
        .unwrap();
}

#[test]
fn independent_concurrent_cpu_limits_contain_nested_rayon_and_faer() {
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let jobs: Vec<_> = [1, 3]
        .into_iter()
        .map(|threads| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                ResourceLimits::new(Some(threads), Some(16))
                    .unwrap()
                    .run(|| {
                        barrier.wait();
                        let values: Vec<_> = (0..8)
                            .into_par_iter()
                            .map(|i| {
                                assert_eq!(rayon::current_num_threads(), threads);
                                assert_eq!(current_usage().unwrap().cpu_threads, threads);
                                assert!(rayon::current_thread_index().unwrap() < threads);
                                let a = Matrix::from_row_major(
                                    20,
                                    20,
                                    (0..400).map(|j| ((i + j) as f64 * 0.13).cos()).collect(),
                                );
                                let gram = a.transpose_matmul(&a);
                                gram.as_slice().iter().sum::<f64>()
                            })
                            .collect();
                        let a = Matrix::from_row_major(
                            24,
                            24,
                            (0..576)
                                .map(|i| {
                                    let (r, c) = (i / 24, i % 24);
                                    if r == c {
                                        4.0 + r as f64 * 0.1
                                    } else {
                                        ((r + c) as f64).sin() * 0.01
                                    }
                                })
                                .collect(),
                        );
                        let (eigenvalues, _) = symmetric_eigen(&a)?;
                        Ok((
                            values,
                            eigenvalues,
                            current_usage().unwrap().peak_reserved_bytes,
                        ))
                    })
                    .unwrap()
            })
        })
        .collect();
    let results: Vec<_> = jobs.into_iter().map(|job| job.join().unwrap()).collect();
    for (a, b) in results[0]
        .0
        .iter()
        .zip(&results[1].0)
        .chain(results[0].1.iter().zip(&results[1].1))
    {
        assert!((a - b).abs() < 1e-11);
    }
    assert!(results.iter().all(|r| r.2 > 0 && r.2 <= 16 * MIB));
    println!(
        "one/three-thread nested Rayon/faer peaks: {} / {} managed bytes",
        results[0].2, results[1].2
    );
}

#[test]
fn nested_scopes_inherit_and_cannot_reset_or_enlarge_limits() {
    ResourceLimits::new(Some(2), Some(1))
        .unwrap()
        .run(|| {
            let _lease = reserve(1234, "nested fixture")?;
            ResourceLimits::new(Some(2), None)?.run(|| {
                assert_eq!(remaining_bytes(), Some(MIB - 1234));
                Ok(())
            })?;
            assert!(ResourceLimits::new(Some(3), Some(2))?
                .install(|| ())
                .is_err());
            Ok(())
        })
        .unwrap();
}

#[test]
fn dense_solver_workspace_is_reserved_before_third_party_work() {
    let a = Matrix::identity(64);
    let error = ResourceLimits::new(Some(1), Some(1))
        .unwrap()
        .run(|| symmetric_eigen(&a))
        .unwrap_err();
    assert!(matches!(error, DftError::ResourceLimit { .. }));
}

#[test]
fn unrelated_panics_are_not_hidden_as_resource_errors() {
    let outcome = std::panic::catch_unwind(|| {
        ResourceLimits::new(Some(1), Some(1))
            .unwrap()
            .install(|| panic!("unrelated panic fixture"))
    });
    assert!(outcome.is_err());
}
