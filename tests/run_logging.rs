// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Real logging contracts: scoped workers, I/O errors and physical SCF progress.

use dft_rs::driver::{self, Controls, System, Value};
use dft_rs::logger::{self, LogLevel, LogMode, LogOptions};
use dft_rs::resources::{current_usage, remaining_bytes, reserve, ResourceLimits, MIB};
use dft_rs::DftError;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "dft-rs-log-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        if self.0.parent() == Some(std::env::temp_dir().as_path()) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

#[test]
fn default_does_not_initialize_a_sink_or_create_files() {
    let directory = Directory::new();
    LogOptions::default()
        .run(|| {
            assert!(!logger::is_active());
            logger::log("unconfigured library calculation");
            Ok(())
        })
        .unwrap();
    assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 0);
    assert!(!logger::is_active());
}

#[test]
fn concurrent_runs_keep_distinct_files_ids_and_worker_cpu_limits() {
    let directory = Directory::new();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let jobs: Vec<_> = [1, 2]
        .into_iter()
        .map(|threads| {
            let path = directory.file(&format!("threads-{threads}.log"));
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                LogOptions::to_file(&path)
                    .unwrap()
                    .run_with_limits(ResourceLimits::new(Some(threads), Some(1)).unwrap(), || {
                        barrier.wait();
                        (0..16).into_par_iter().for_each(|index| {
                            assert_eq!(rayon::current_num_threads(), threads);
                            assert!(logger::init(LogMode::Console, None).is_err());
                            logger::log(&format!("worker marker={threads}:{index}"));
                        });
                        Ok(())
                    })
                    .unwrap();
                path
            })
        })
        .collect();
    let mut ids = Vec::new();
    for (threads, job) in [1, 2].into_iter().zip(jobs) {
        let text = std::fs::read_to_string(job.join().unwrap()).unwrap();
        assert_eq!(
            text.matches(&format!("worker marker={threads}:")).count(),
            16
        );
        assert!(!text.contains(&format!("worker marker={}:", 3 - threads)));
        let run_ids: std::collections::HashSet<_> = text
            .lines()
            .map(|line| line.split_whitespace().nth(1).unwrap())
            .collect();
        assert_eq!(run_ids.len(), 1);
        ids.push(run_ids.into_iter().next().unwrap().to_string());
    }
    assert_ne!(ids[0], ids[1]);
    assert!(!logger::is_active());
    assert!(current_usage().is_none());
}

#[test]
fn nested_run_inherits_the_log_and_budget_without_an_additional_cpu_pool() {
    let directory = Directory::new();
    let outer = directory.file("outer.log");
    let inner = directory.file("inner.log");
    LogOptions::to_file(&outer)
        .unwrap()
        .run_with_limits(ResourceLimits::new(Some(2), Some(1)).unwrap(), || {
            let reservation = reserve(MIB / 4, "outer fixture")?;
            logger::log("outer-before");
            LogOptions::default().run(|| {
                assert_eq!(current_usage().unwrap().cpu_threads, 2);
                assert_eq!(remaining_bytes(), Some(3 * MIB / 4));
                (0..4)
                    .into_par_iter()
                    .for_each(|_| logger::log("inherited-inner"));
                Ok(())
            })?;
            let error = LogOptions::to_file(&inner)?.run(|| Ok(())).unwrap_err();
            assert!(error.to_string().contains("inherit the active log context"));
            logger::log("outer-after");
            drop(reservation);
            assert_eq!(remaining_bytes(), Some(MIB));
            Ok(())
        })
        .unwrap();
    let outer = std::fs::read_to_string(outer).unwrap();
    let inner = std::fs::read_to_string(inner).unwrap();
    assert!(outer.contains("outer-before") && outer.contains("outer-after"));
    assert!(outer.contains("inherited-inner"));
    assert!(!inner.contains("outer-before") && inner.contains("status=failed"));
}

#[test]
fn logging_only_scope_allows_the_first_explicit_cpu_and_memory_request() {
    let directory = Directory::new();
    LogOptions::to_file(directory.file("configured-later.log"))
        .unwrap()
        .run(|| {
            ResourceLimits::new(Some(2), Some(1))?.run(|| {
                assert_eq!(current_usage().unwrap().cpu_threads, 2);
                assert_eq!(remaining_bytes(), Some(MIB));
                Ok(())
            })
        })
        .unwrap();
}

#[test]
fn sequential_runs_append_and_concurrent_same_path_is_an_error() {
    let directory = Directory::new();
    let path = directory.file("saved.log");
    LogOptions::to_file(&path)
        .unwrap()
        .run(|| {
            assert!(LogOptions::to_file(&path)?.run(|| Ok(())).is_err());
            logger::log("retained-first-run");
            Ok(())
        })
        .unwrap();
    LogOptions::to_file(&path)
        .unwrap()
        .run(|| {
            logger::log("second-run");
            Ok(())
        })
        .unwrap();
    let text = std::fs::read_to_string(path).unwrap();
    assert!(text.contains("retained-first-run") && text.contains("second-run"));
    assert_eq!(text.matches("run started").count(), 2);
}

#[test]
fn malformed_configuration_and_open_failure_do_not_run_the_calculation() {
    let directory = Directory::new();
    for (path, mode, level) in [
        (Some(""), None, None),
        (None, Some("file"), None),
        (Some("x.log"), Some("console"), None),
        (None, None, Some("verbose")),
    ] {
        assert!(LogOptions::from_strings(
            path.map(str::to_string),
            mode.map(str::to_string),
            level.map(str::to_string)
        )
        .is_err());
    }
    for path in [
        directory.file("missing-parent/log.txt"),
        directory.0.clone(),
    ] {
        let result = LogOptions::to_file(path)
            .unwrap()
            .run::<()>(|| panic!("invalid log destination reached calculation"));
        assert!(matches!(result, Err(DftError::Io(_))));
    }
}

#[test]
fn error_level_filters_progress_and_preserves_the_original_calculation_error() {
    let directory = Directory::new();
    let path = directory.file("failure.log");
    let result = LogOptions::new(LogMode::File, Some(path.clone()), LogLevel::Error)
        .unwrap()
        .run::<()>(|| {
            logger::log("suppressed-progress");
            Err(DftError::ScfNotConverged {
                iterations: 3,
                error: 0.2,
            })
        });
    assert!(matches!(
        result,
        Err(DftError::ScfNotConverged { iterations: 3, .. })
    ));
    let text = std::fs::read_to_string(path).unwrap();
    assert!(!text.contains("suppressed-progress"));
    assert!(text.contains("SCF did not converge after 3 iterations"));
}

#[test]
fn unwinding_preserves_flushed_partial_records_and_restores_context() {
    let directory = Directory::new();
    let path = directory.file("partial.log");
    let result = std::panic::catch_unwind(|| {
        let _ = LogOptions::to_file(&path).unwrap().run::<()>(|| {
            logger::log("last-completed-iteration");
            panic!("fixture cancellation");
        });
    });
    assert!(result.is_err());
    let text = std::fs::read_to_string(path).unwrap();
    assert!(text.contains("last-completed-iteration") && text.contains("interrupted by panic"));
    assert!(!logger::is_active());
}

#[cfg(unix)]
#[test]
fn write_and_flush_failure_is_reported_instead_of_returning_success() {
    let path = std::path::Path::new("/dev/full");
    if path.exists() {
        let error = LogOptions::to_file(path)
            .unwrap()
            .run(|| Ok(1))
            .unwrap_err();
        assert!(matches!(error, DftError::Io(_)));
        assert!(error.to_string().contains("log I/O failed"));
    }
}

fn molecule() -> System {
    System::molecule(vec![1, 1], vec![[0.1, 0.2, 0.3], [0.5, 0.7, 1.5]])
}

#[test]
fn common_driver_saves_actual_scf_settings_iterations_units_and_energy() {
    let directory = Directory::new();
    let mut energies = Vec::new();
    for threads in [1, 2] {
        let path = directory.file(&format!("scf-{threads}.log"));
        let controls = Controls::from_value(
            &Value::object()
                .with("basis", "sto-3g")
                .with("xc", "hf")
                .with("cpu_threads", threads)
                .with("memory_mb", 128)
                .with("log_file", path.to_str().unwrap()),
        )
        .unwrap();
        let report = driver::run("energy", &molecule(), &controls).unwrap();
        energies.push(report.get("energy").unwrap().as_f64().unwrap());
        let text = std::fs::read_to_string(path).unwrap();
        for expected in [
            "SCF settings",
            "[scf] iter",
            "result energy=",
            "hartree",
            "bohr",
            "status=completed",
        ] {
            assert!(text.contains(expected), "missing {expected}");
        }
    }
    assert!((energies[0] - energies[1]).abs() < 1e-12);
}

#[test]
fn common_driver_retains_unconverged_scf_records_and_refuses_a_result() {
    let directory = Directory::new();
    let path = directory.file("unconverged.log");
    let controls = Controls::from_value(
        &Value::object()
            .with("basis", "sto-3g")
            .with("xc", "hf")
            .with("charge", 1)
            .with("max_iter", 1)
            .with("cpu_threads", 1)
            .with("log_file", path.to_str().unwrap()),
    )
    .unwrap();
    let mut system = molecule();
    system.numbers = vec![2, 1];
    let error = driver::run("energy", &system, &controls).unwrap_err();
    assert!(matches!(error, DftError::ScfNotConverged { .. }));
    let text = std::fs::read_to_string(path).unwrap();
    assert!(
        text.contains("[scf] iter")
            && text.contains("SCF did not converge")
            && text.contains("status=failed")
    );
}
