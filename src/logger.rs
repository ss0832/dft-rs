// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Run-local computation progress, SCF iterations and timings.
//! Library calls inherit an enclosing log and create no file by default.
//! The common CLI defaults to stderr; legacy input files default to both
//! stderr and a sibling log file. Explicit options select the sink and level.

use crate::{DftError, Result};
use std::cell::RefCell;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogMode {
    Console,
    File,
    Both,
    None,
}

impl LogMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "console" | "stdout" => Some(Self::Console),
            "file" => Some(Self::File),
            "both" => Some(Self::Both),
            "none" | "quiet" => Some(Self::None),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Error,
    Warn,
    #[default]
    Info,
    Debug,
}

impl LogLevel {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "error" => Some(Self::Error),
            "warn" | "warning" => Some(Self::Warn),
            "info" => Some(Self::Info),
            "debug" => Some(Self::Debug),
            _ => None,
        }
    }
}

/// Default: no output configuration, inheriting an enclosing run.
/// Files append and flush every record. Parents must already exist.
#[derive(Clone, Debug)]
pub struct LogOptions {
    mode: LogMode,
    path: Option<PathBuf>,
    level: LogLevel,
    configured: bool,
}

impl Default for LogOptions {
    fn default() -> Self {
        Self {
            mode: LogMode::None,
            path: None,
            level: LogLevel::Info,
            configured: false,
        }
    }
}

impl LogOptions {
    pub fn new(mode: LogMode, path: Option<PathBuf>, level: LogLevel) -> Result<Self> {
        if matches!(mode, LogMode::File | LogMode::Both) != path.is_some()
            || path.as_ref().is_some_and(|p| p.as_os_str().is_empty())
        {
            return Err(DftError::InvalidInput(
                "log_file is required exactly for log_mode=file or both and must not be empty"
                    .into(),
            ));
        }
        Ok(Self {
            mode,
            path,
            level,
            configured: true,
        })
    }

    pub fn to_file(path: impl Into<PathBuf>) -> Result<Self> {
        Self::new(LogMode::File, Some(path.into()), LogLevel::Info)
    }

    pub fn from_strings(
        path: Option<String>,
        mode: Option<String>,
        level: Option<String>,
    ) -> Result<Self> {
        if path.is_none() && mode.is_none() && level.is_none() {
            return Ok(Self::default());
        }
        let mode = match mode {
            Some(value) => LogMode::parse(&value).ok_or_else(|| {
                DftError::InvalidInput("log_mode must be console, file, both or none".into())
            })?,
            None if path.is_some() => LogMode::File,
            None => LogMode::Console,
        };
        let level = match level {
            Some(value) => LogLevel::parse(&value).ok_or_else(|| {
                DftError::InvalidInput("log_level must be error, warn, info or debug".into())
            })?,
            None => LogLevel::Info,
        };
        Self::new(mode, path.map(PathBuf::from), level)
    }

    pub fn run<T: Send>(self, work: impl FnOnce() -> Result<T> + Send) -> Result<T> {
        self.run_with_limits(crate::resources::ResourceLimits::default(), work)
    }

    /// Logging and resource policy share one execution envelope.
    pub fn run_with_limits<T: Send>(
        self,
        limits: crate::resources::ResourceLimits,
        work: impl FnOnce() -> Result<T> + Send,
    ) -> Result<T> {
        self.install(|| {
            let result = limits.run(work);
            if let Err(error) = &result {
                record_error(&error.to_string());
            }
            result
        })?
    }

    /// Install a context for an adapter with its own error type. Such adapters
    /// call record_error before returning their original error.
    pub fn install<T>(self, work: impl FnOnce() -> T) -> Result<T> {
        if !self.configured {
            return Ok(work());
        }
        let context = self.open().map_err(|error| match error {
            DftError::Io(error) => DftError::Io(std::io::Error::new(
                error.kind(),
                format!("could not open log_file: {error}"),
            )),
            other => other,
        })?;
        let previous = ACTIVE.with(|slot| slot.replace(Some(context.clone())));
        let mut guard = ScopeGuard {
            previous,
            context: context.clone(),
            finished: false,
        };
        log(&format!(
            "run started version={} units=atomic (hartree, bohr)",
            env!("CARGO_PKG_VERSION")
        ));
        let result = work();
        let failed = context
            .0
            .lock()
            .expect("run log lock")
            .calculation_error
            .is_some();
        log(if failed {
            "run finished status=failed"
        } else {
            "run finished status=completed"
        });
        let state = context.0.lock().expect("run log lock");
        let failure = state
            .failure
            .as_ref()
            .map(|message| match &state.calculation_error {
                Some(error) => format!(
                    "calculation log I/O failed: {message}; calculation also failed: {error}"
                ),
                None => format!("calculation log I/O failed: {message}"),
            });
        drop(state);
        guard.finished = true;
        drop(guard);
        match failure {
            Some(message) => Err(DftError::Io(std::io::Error::other(message))),
            None => Ok(result),
        }
    }

    fn open(&self) -> Result<LogContext> {
        let (file, lease) = if let Some(path) = &self.path {
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            let key =
                if path.exists() {
                    path.canonicalize()?
                } else {
                    parent.canonicalize()?.join(path.file_name().ok_or_else(|| {
                        DftError::InvalidInput("log_file must name a file".into())
                    })?)
                };
            let mut paths = active_paths().lock().expect("active log paths lock");
            if !paths.insert(key.clone()) {
                return Err(DftError::InvalidInput(
                    "another active calculation is writing this log_file; use a distinct file per concurrent run".into()));
            }
            let lease = PathLease(key);
            drop(paths);
            (
                Some(OpenOptions::new().create(true).append(true).open(path)?),
                Some(lease),
            )
        } else {
            (None, None)
        };
        Ok(LogContext(Arc::new(Mutex::new(LoggerState {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            mode: self.mode,
            level: self.level,
            file,
            lease,
            start: Instant::now(),
            failure: None,
            calculation_error: None,
            closed: false,
        }))))
    }
}

struct PathLease(PathBuf);
impl Drop for PathLease {
    fn drop(&mut self) {
        if let Ok(mut paths) = active_paths().lock() {
            paths.remove(&self.0);
        }
    }
}

// Only path ownership is global. Neither records nor sinks live here.
fn active_paths() -> &'static Mutex<HashSet<PathBuf>> {
    static PATHS: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    PATHS.get_or_init(|| Mutex::new(HashSet::new()))
}

struct LoggerState {
    id: u64,
    mode: LogMode,
    level: LogLevel,
    file: Option<File>,
    lease: Option<PathLease>,
    start: Instant,
    failure: Option<String>,
    calculation_error: Option<String>,
    closed: bool,
}

#[derive(Clone)]
pub(crate) struct LogContext(Arc<Mutex<LoggerState>>);
impl LogContext {
    pub(crate) fn id(&self) -> u64 {
        self.0.lock().expect("run log lock").id
    }
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
thread_local! {
    static ACTIVE: RefCell<Option<LogContext>> = const { RefCell::new(None) };
}

pub(crate) fn capture() -> Option<LogContext> {
    ACTIVE.with(|slot| slot.borrow().clone())
}
pub(crate) fn inherit(context: Option<LogContext>) {
    ACTIVE.with(|slot| *slot.borrow_mut() = context);
}
pub fn is_active() -> bool {
    ACTIVE.with(|slot| slot.borrow().is_some())
}

pub(crate) fn enabled(level: LogLevel) -> bool {
    capture().is_some_and(|context| {
        let state = context.0.lock().expect("run log lock");
        !state.closed && state.mode != LogMode::None && level <= state.level
    })
}

struct ScopeGuard {
    previous: Option<LogContext>,
    context: LogContext,
    finished: bool,
}
impl Drop for ScopeGuard {
    fn drop(&mut self) {
        if !self.finished {
            record_error("run interrupted by panic; previous records were flushed");
        }
        {
            let mut state = self.context.0.lock().expect("run log lock");
            state.closed = true;
            state.file.take();
            state.lease.take();
        }
        ACTIVE.with(|slot| {
            slot.replace(self.previous.take());
        });
    }
}

/// Calling-thread compatibility initializer; scoped APIs check final I/O errors.
pub fn init(mode: LogMode, path: Option<&Path>) -> std::io::Result<()> {
    if is_active() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "cannot reinitialize an active scoped log; use nested LogOptions",
        ));
    }
    let path = if matches!(mode, LogMode::File | LogMode::Both) {
        path.map(Path::to_path_buf)
    } else {
        None
    };
    let options = LogOptions::new(mode, path, LogLevel::Info).map_err(|error| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, error.to_string())
    })?;
    let context = options
        .open()
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    inherit(Some(context));
    Ok(())
}

fn emit(level: LogLevel, message: &str, console: bool) {
    let Some(context) = capture() else {
        if console && (level <= LogLevel::Warn || std::env::var_os("DFT_RS_SCF_TRACE").is_some()) {
            eprintln!("{message}");
        }
        return;
    };
    let mut state = context.0.lock().expect("run log lock");
    if state.closed {
        return;
    }
    if level == LogLevel::Error && state.calculation_error.is_none() {
        state.calculation_error = Some(message.to_string());
    }
    if level > state.level || state.mode == LogMode::None {
        return;
    }
    let line = format!(
        "[run p{}-r{} {:>9.3}s {level:?}] {}\n",
        std::process::id(),
        state.id,
        state.start.elapsed().as_secs_f64(),
        message.replace('\r', "\\r").replace('\n', "\\n")
    );
    if console && matches!(state.mode, LogMode::Console | LogMode::Both) {
        let mut stderr = std::io::stderr().lock();
        if let Err(error) = stderr
            .write_all(line.as_bytes())
            .and_then(|()| stderr.flush())
        {
            if state.failure.is_none() {
                state.failure = Some(format!("stderr write failed: {error}"));
            }
        }
    }
    if let Some(file) = state.file.as_mut() {
        if let Err(error) = file.write_all(line.as_bytes()).and_then(|()| file.flush()) {
            if state.failure.is_none() {
                state.failure = Some(error.to_string());
            }
        }
    }
}

pub fn log(message: &str) {
    emit(LogLevel::Info, message, true);
}
pub fn debug(message: &str) {
    emit(LogLevel::Debug, message, true);
}
pub fn warn(message: &str) {
    emit(LogLevel::Warn, message, true);
}
pub fn error(message: &str) {
    emit(LogLevel::Error, message, true);
}
/// Keep the caller's existing stderr/error API while adding file detail.
pub fn record_error(message: &str) {
    emit(LogLevel::Error, message, false);
}

/// Check latched I/O failures for callers using the compatibility initializer.
/// Scoped operations perform the same check automatically before returning.
pub fn flush() -> Result<()> {
    let Some(context) = capture() else {
        return Ok(());
    };
    let mut state = context.0.lock().expect("run log lock");
    if let Some(file) = state.file.as_mut() {
        if let Err(error) = file.flush() {
            if state.failure.is_none() {
                state.failure = Some(error.to_string());
            }
        }
    }
    match &state.failure {
        Some(message) => Err(DftError::Io(std::io::Error::other(format!(
            "calculation log I/O failed: {message}"
        )))),
        None => Ok(()),
    }
}

pub(crate) fn scf_settings(options: &crate::scf::ScfOptions) {
    if !enabled(LogLevel::Info) {
        return;
    }
    log(&format!("SCF settings max_iter={} e_tol={:.12e} hartree p_tol={:.12e} residual_tol={:.12e} reference={:?} incore_mb={}",
        options.max_iter, options.e_tol, options.p_tol, options.conv_tol_grad, options.reference, options.incore_mb));
    debug(&format!(
        "SCF method diis={} diis_size={} damping={} soscf={} density_fit={:?}",
        options.use_diis,
        options.diis_size,
        options.damping,
        options.use_soscf,
        options.density_fit
    ));
}

pub fn timed<T>(label: &str, work: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let result = work();
    log(&format!("{label}: {:.3} s", start.elapsed().as_secs_f64()));
    result
}
