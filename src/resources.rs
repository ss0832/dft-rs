// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Per-calculation CPU ceilings and a shared budget for managed allocations.
//!
//! Private Rayon pools also contain nested faer Rayon jobs. No request changes
//! the global Rayon pool or faer's global policy. The memory ledger counts
//! registered allocations and workspace reservations, not whole-process RSS,
//! caller-owned inputs, Python objects or every third-party allocation.

use crate::{DftError, Result};
use std::cell::{Cell, RefCell};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

pub const MIB: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceLimits {
    pub cpu_threads: Option<usize>,
    pub memory_mb: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceUsage {
    pub cpu_threads: usize,
    pub memory_mb: Option<usize>,
    pub reserved_bytes: usize,
    pub peak_reserved_bytes: usize,
}

struct Budget {
    threads: usize,
    configured: bool,
    limit: Option<usize>,
    used: AtomicUsize,
    peak: AtomicUsize,
    failure: Mutex<Option<DftError>>,
}

thread_local! {
    static ACTIVE: RefCell<Option<Arc<Budget>>> = const { RefCell::new(None) };
    static POOL_LOG_ID: Cell<Option<u64>> = const { Cell::new(None) };
}

struct Lease {
    budget: Arc<Budget>,
    bytes: usize,
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// Keeps one registered allocation charged until its last owner is dropped.
/// Cloning this lease does not describe copying the underlying allocation.
#[derive(Clone, Default)]
pub struct MemoryReservation {
    lease: Option<Arc<Lease>>,
}

impl std::fmt::Debug for MemoryReservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryReservation")
            .field("bytes", &self.bytes())
            .finish()
    }
}

impl MemoryReservation {
    pub fn bytes(&self) -> usize {
        self.lease.as_ref().map_or(0, |lease| lease.bytes)
    }
}

impl ResourceLimits {
    pub fn new(cpu_threads: Option<usize>, memory_mb: Option<usize>) -> Result<Self> {
        let limits = Self {
            cpu_threads,
            memory_mb,
        };
        limits.validate()?;
        Ok(limits)
    }

    fn validate(self) -> Result<()> {
        if cfg!(panic = "abort") && self.memory_mb.is_some() {
            return Err(DftError::Unsupported {
                feature: "managed memory limits with panic=abort".into(),
                missing: "an unwind-enabled build for compatible infallible Matrix constructors"
                    .into(),
            });
        }
        if self
            .cpu_threads
            .is_some_and(|n| n == 0 || n > rayon::max_num_threads())
        {
            return Err(DftError::InvalidInput(format!(
                "cpu_threads must be between 1 and {}",
                rayon::max_num_threads()
            )));
        }
        if self
            .memory_mb
            .is_some_and(|n| n == 0 || n > isize::MAX as usize / MIB)
        {
            return Err(DftError::InvalidInput(
                "memory_mb must be positive and its byte count must fit isize".into(),
            ));
        }
        Ok(())
    }

    pub fn is_configured(self) -> bool {
        self.cpu_threads.is_some() || self.memory_mb.is_some()
    }

    /// Execute in a private pool. Nested calls inherit the same settings and
    /// ledger; reconfiguration inside an active scope is explicitly refused.
    pub fn install<T: Send>(self, work: impl FnOnce() -> T + Send) -> Result<T> {
        self.validate()?;
        if !self.is_configured() && !crate::logger::is_active() {
            return Ok(work());
        }
        let logging = crate::logger::capture();
        let log_id = logging.as_ref().map(crate::logger::LogContext::id);
        // A pool created only to propagate logging is not an explicit resource
        // ceiling. The first real resource request can still configure it.
        let active_budget = ACTIVE
            .with(|slot| slot.borrow().clone())
            .filter(|budget| budget.configured || !self.is_configured());
        let budget = if let Some(active) = active_budget {
            if self.cpu_threads.is_some_and(|n| n != active.threads)
                || self
                    .memory_mb
                    .is_some_and(|n| Some(n) != active.limit.map(|bytes| bytes / MIB))
            {
                return Err(DftError::InvalidInput(
                    "nested calculations inherit cpu_threads and memory_mb; resource reconfiguration requires an independent scope".into()));
            }
            if POOL_LOG_ID.with(Cell::get) == log_id {
                return Ok(work());
            }
            return Err(DftError::InvalidInput(
                "nested calculations inherit the active log context; configure LogOptions::run_with_limits at the outer boundary or use an independent scope".into(),
            ));
        } else {
            // Initialize only the existing process policy, before entering the
            // request pool. Each request leaves that policy unchanged.
            crate::linalg::ensure_parallelism();
            let threads = self.cpu_threads.unwrap_or_else(rayon::current_num_threads);
            Arc::new(Budget {
                threads,
                configured: self.is_configured(),
                limit: self.memory_mb.map(|mb| mb * MIB),
                used: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
                failure: Mutex::new(None),
            })
        };
        let threads = budget.threads;
        crate::logger::log(&format!(
            "resources cpu_threads={threads} memory_mb={:?} scope=managed-allocations-not-RSS",
            budget.limit.map(|bytes| bytes / MIB)
        ));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            // Third-order XC dispatch already requires this stack size.
            // Keeping that work here avoids an unbudgeted helper thread.
            .stack_size(8 * MIB)
            .start_handler(move |_| {
                ACTIVE.with(|slot| *slot.borrow_mut() = Some(budget.clone()));
                POOL_LOG_ID.with(|slot| slot.set(log_id));
                crate::logger::inherit(logging.clone());
            })
            .build()
            .map_err(|error| {
                DftError::InvalidInput(format!("could not create calculation CPU pool: {error}"))
            })?;
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pool.install(work))) {
            Ok(value) => Ok(value),
            Err(payload) => match payload.downcast::<AllocationFailure>() {
                Ok(failure) => Err(failure.0),
                Err(payload) => std::panic::resume_unwind(payload),
            },
        }
    }

    pub fn run<T: Send>(self, work: impl FnOnce() -> Result<T> + Send) -> Result<T> {
        self.install(work)?
    }
}

pub fn current_usage() -> Option<ResourceUsage> {
    ACTIVE.with(|slot| {
        slot.borrow().as_ref().map(|budget| ResourceUsage {
            cpu_threads: budget.threads,
            memory_mb: budget.limit.map(|bytes| bytes / MIB),
            reserved_bytes: budget.used.load(Ordering::Relaxed),
            peak_reserved_bytes: budget.peak.load(Ordering::Relaxed),
        })
    })
}

#[cfg(feature = "python")]
pub(crate) fn scope_is_configured() -> bool {
    ACTIVE.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|budget| budget.configured)
    })
}

pub fn remaining_bytes() -> Option<usize> {
    ACTIVE.with(|slot| {
        slot.borrow().as_ref().and_then(|budget| {
            budget
                .limit
                .map(|limit| limit.saturating_sub(budget.used.load(Ordering::Relaxed)))
        })
    })
}

/// Cap an existing per-component MiB limit without increasing it.
pub fn component_limit_mb(requested: usize) -> usize {
    remaining_bytes().map_or(requested, |bytes| requested.min(bytes / MIB))
}

pub fn checked_bytes(factors: &[usize], operation: &'static str) -> Result<usize> {
    factors
        .iter()
        .try_fold(1usize, |bytes, &factor| bytes.checked_mul(factor))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| DftError::InvalidInput(format!("{operation} allocation size overflows")))
}

pub fn reserve(bytes: usize, operation: &'static str) -> Result<MemoryReservation> {
    let budget = ACTIVE.with(|slot| {
        slot.borrow()
            .as_ref()
            .filter(|budget| budget.limit.is_some())
            .cloned()
    });
    let Some(budget) = budget else {
        return Ok(MemoryReservation::default());
    };
    let limit = budget.limit.expect("memory budget checked");
    let mut used = budget.used.load(Ordering::Relaxed);
    loop {
        let needed = used.checked_add(bytes);
        if needed.is_none_or(|needed| needed > limit) {
            let required = needed.unwrap_or(usize::MAX);
            return Err(DftError::ResourceLimit {
                operation,
                required_mb: required / MIB + usize::from(required % MIB != 0),
                limit_mb: limit / MIB,
            });
        }
        let needed = needed.expect("checked allocation sum");
        match budget
            .used
            .compare_exchange_weak(used, needed, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => {
                budget.peak.fetch_max(needed, Ordering::Relaxed);
                return Ok(MemoryReservation {
                    lease: Some(Arc::new(Lease { budget, bytes })),
                });
            }
            Err(actual) => used = actual,
        }
    }
}

/// Optional caches leave at least half the currently available budget for
/// required workspaces. Reservation races select exact recomputation instead.
pub(crate) fn reserve_cache(
    bytes: usize,
    operation: &'static str,
) -> Result<Option<MemoryReservation>> {
    if bytes == 0 || remaining_bytes().is_some_and(|remaining| bytes > remaining / 2) {
        return Ok(None);
    }
    match reserve(bytes, operation) {
        Ok(lease) => Ok(Some(lease)),
        Err(DftError::ResourceLimit { .. }) => Ok(None),
        Err(error) => Err(error),
    }
}

struct AllocationFailure(DftError);

/// Recover compatibility allocation refusals before crossing an FFI boundary.
/// Ordinary errors and unrelated panic payloads retain their original meaning.
pub(crate) fn catch_allocations<T>(work: impl FnOnce() -> Result<T>) -> Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)) {
        Ok(value) => value,
        Err(payload) => match payload.downcast::<AllocationFailure>() {
            Ok(failure) => Err(failure.0),
            Err(payload) => std::panic::resume_unwind(payload),
        },
    }
}

#[cfg(feature = "python")]
pub(crate) fn take_failure() -> Option<DftError> {
    ACTIVE.with(|slot| {
        slot.borrow().as_ref().and_then(|budget| {
            budget
                .failure
                .lock()
                .expect("resource failure mutex poisoned")
                .take()
        })
    })
}

/// Compatibility adapter for existing infallible Matrix constructors. Only
/// this private payload is recovered by ResourceLimits::install; unrelated
/// panics continue to propagate. resume_unwind avoids invoking the panic hook
/// for a normal budget refusal. Unwinding releases all outstanding leases.
pub(crate) fn reserve_or_unwind(factors: &[usize], operation: &'static str) -> MemoryReservation {
    match checked_bytes(factors, operation).and_then(|bytes| reserve(bytes, operation)) {
        Ok(reservation) => reservation,
        Err(error) => {
            // PyO3 intercepts unwinds at each numerical binding boundary.
            // Retain the typed refusal so the outer Python resource adapter
            // can restore it instead of leaking a generic PanicException.
            ACTIVE.with(|slot| {
                if let Some(budget) = slot.borrow().as_ref() {
                    let copy = match &error {
                        DftError::ResourceLimit {
                            operation,
                            required_mb,
                            limit_mb,
                        } => DftError::ResourceLimit {
                            operation,
                            required_mb: *required_mb,
                            limit_mb: *limit_mb,
                        },
                        _ => DftError::InvalidInput(error.to_string()),
                    };
                    let mut failure = budget
                        .failure
                        .lock()
                        .expect("resource failure mutex poisoned");
                    if failure.is_none() {
                        *failure = Some(copy);
                    }
                }
            });
            std::panic::resume_unwind(Box::new(AllocationFailure(error)))
        }
    }
}

/// Extract the same global options before either the shared or legacy CLI
/// parser. Other arguments retain their original order and spelling.
#[doc(hidden)]
pub fn cli_limits(args: Vec<String>) -> Result<(ResourceLimits, Vec<String>)> {
    let mut cpu = None;
    let mut memory = None;
    let mut rest = Vec::new();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let (key, inline) = arg
            .split_once('=')
            .map_or((arg.as_str(), None), |(k, v)| (k, Some(v)));
        let target = match key {
            "--cpu-threads" | "--cpu_threads" => &mut cpu,
            "--memory-mb" | "--memory_mb" => &mut memory,
            _ => {
                rest.push(arg);
                continue;
            }
        };
        if target.is_some() {
            return Err(DftError::InvalidInput(format!(
                "{key} was given more than once"
            )));
        }
        let value = inline
            .map(str::to_owned)
            .or_else(|| args.next())
            .ok_or_else(|| DftError::InvalidInput(format!("{key} requires a positive integer")))?;
        *target =
            Some(value.parse::<usize>().map_err(|_| {
                DftError::InvalidInput(format!("{key} requires a positive integer"))
            })?);
    }
    Ok((ResourceLimits::new(cpu, memory)?, rest))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_cache_keeps_workspace_and_releases_its_lease() {
        ResourceLimits::new(Some(1), Some(1))
            .unwrap()
            .run(|| {
                assert!(reserve_cache(MIB, "optional fixture")?.is_none());
                let cache = reserve_cache(MIB / 4, "optional fixture")?.unwrap();
                assert_eq!(remaining_bytes(), Some(3 * MIB / 4));
                let workspace = reserve(MIB / 2, "required fixture")?;
                assert!(reserve_cache(MIB / 4, "optional fixture")?.is_none());
                drop(workspace);
                drop(cache);
                assert_eq!(remaining_bytes(), Some(MIB));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn fallible_boundary_recovers_before_ffi_and_releases_allocations() {
        let error = ResourceLimits::new(Some(1), Some(1))
            .unwrap()
            .run(|| catch_allocations(|| Ok(crate::linalg::Matrix::zeros(1024, 1024))))
            .unwrap_err();
        assert!(matches!(error, DftError::ResourceLimit { .. }));
    }

    #[test]
    fn cli_resources_are_global_and_preserve_other_arguments() {
        let (limits, args) = cli_limits(
            [
                "energy",
                "mol.xyz",
                "--cpu-threads=2",
                "--memory-mb",
                "32",
                "--basis",
                "sto-3g",
            ]
            .map(str::to_owned)
            .to_vec(),
        )
        .unwrap();
        assert_eq!(limits, ResourceLimits::new(Some(2), Some(32)).unwrap());
        assert_eq!(args, ["energy", "mol.xyz", "--basis", "sto-3g"]);
    }

    #[test]
    fn cli_invalid_missing_duplicate_and_overflowing_values_fail() {
        for args in [
            vec!["energy", "--cpu-threads"],
            vec!["energy", "--cpu-threads=0"],
            vec!["energy", "--memory-mb=-1"],
            vec!["energy", "--memory-mb=1.5"],
            vec!["energy", "--cpu-threads=2", "--cpu_threads=2"],
            vec!["energy", "--memory-mb=18446744073709551615"],
        ] {
            assert!(cli_limits(args.into_iter().map(str::to_owned).collect()).is_err());
        }
    }
}
