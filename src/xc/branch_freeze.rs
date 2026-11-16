// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Test-only freezing of the interpolation branch of rSCAN-type
//! functionals for finite-difference validation.
//!
//! rSCAN switches from its polynomial to the SCAN branch at the regularized
//! indicator 2.5 with three continuous derivatives. When a nucleus or a
//! field moves, quadrature points cross the switch and finite differences of
//! the molecular energy stop following a power law, although the analytic
//! derivative at the reference is exact. With the branch of every point and
//! interpolation site recorded at the reference and replayed in the
//! displaced calculations, the energy is analytic along the path and its
//! derivative at the reference is the analytic one.
//!
//! The recording is keyed by the global quadrature point (the batch start
//! set by the grid loops plus the point inside the batch) and the site
//! (exchange of each spin, correlation). It applies only to the functional
//! instance whose upper switch is the tag, so concurrently running tests are
//! unaffected.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

thread_local! {
    static BATCH: Cell<Option<usize>> = const { Cell::new(None) };
    static POINT: Cell<Option<usize>> = const { Cell::new(None) };
}

/// The recorded branches, sharded by point so that the grid threads rarely
/// wait for each other.
const SHARDS: usize = 64;
static CODES: [Mutex<BTreeMap<(usize, u8), u8>>; SHARDS] =
    [const { Mutex::new(BTreeMap::new()) }; SHARDS];
static ACTIVE: AtomicBool = AtomicBool::new(false);
static REPLAY: AtomicBool = AtomicBool::new(false);
static TAG: AtomicU64 = AtomicU64::new(0);
static REPLAYED: AtomicUsize = AtomicUsize::new(0);

fn clear() {
    for shard in &CODES {
        shard.lock().unwrap().clear();
    }
}

/// Start recording the branches of the functional whose upper switch is `tag`.
pub(crate) fn begin(tag: f64) {
    clear();
    TAG.store(tag.to_bits(), Ordering::SeqCst);
    REPLAY.store(false, Ordering::SeqCst);
    REPLAYED.store(0, Ordering::SeqCst);
    ACTIVE.store(true, Ordering::SeqCst);
}

/// Stop recording and replay the recorded branches.
pub(crate) fn replay() {
    REPLAY.store(true, Ordering::SeqCst);
}

/// Number of recorded (point, site) branches and of replayed decisions.
pub(crate) fn counts() -> (usize, usize) {
    let recorded = CODES.iter().map(|s| s.lock().unwrap().len()).sum();
    (recorded, REPLAYED.load(Ordering::SeqCst))
}

pub(crate) fn end() {
    ACTIVE.store(false, Ordering::SeqCst);
    clear();
}

/// Global index of the first point of the batch being evaluated.
pub(crate) fn set_batch(start: Option<usize>) {
    BATCH.with(|b| b.set(start));
}

/// Mark the point `index` of the current batch.
pub(crate) fn set_point(index: Option<usize>) {
    POINT.with(|p| p.set(index.and_then(|i| BATCH.with(|b| b.get()).map(|s| s + i))));
}

/// The branch to evaluate at `site` for the functional with upper switch
/// `tag`: the automatic one, recorded in record mode or replaced by the
/// recorded one in replay mode.
pub(crate) fn branch(tag: f64, site: u8, automatic: u8) -> u8 {
    if !ACTIVE.load(Ordering::Relaxed) || TAG.load(Ordering::Relaxed) != tag.to_bits() {
        return automatic;
    }
    let Some(point) = POINT.with(|p| p.get()) else {
        return automatic;
    };
    let mut shard = CODES[point % SHARDS].lock().unwrap();
    if REPLAY.load(Ordering::Relaxed) {
        match shard.get(&(point, site)) {
            Some(&code) => {
                REPLAYED.fetch_add(1, Ordering::Relaxed);
                code
            }
            None => automatic,
        }
    } else {
        shard.insert((point, site), automatic);
        automatic
    }
}

#[cfg(test)]
mod tests {
    use super::{begin, counts, end, replay};
    use crate::acceptance::components::assert_no_failures;
    use crate::acceptance::molecular_fits::{
        gradient_and_hessian, polarizability_and_hyperpolarizability, system, WATER,
    };
    use crate::driver::{run, Controls, Value};

    /// The upper switch of the frozen functional: the published 2.5 moved by
    /// 1e-12, which changes no indicator on the grid but keys the table to
    /// this test.
    const TAG: f64 = 2.5 + 1e-12;

    fn controls(extra: &[(&str, Value)]) -> Controls {
        let mut all = vec![
            ("xc".to_string(), Value::Str("rscan".into())),
            (
                "xc_parameters".to_string(),
                Value::object().with("switch_upper", TAG),
            ),
            ("basis".to_string(), Value::Str("sto-3g".into())),
            ("grid_level".to_string(), Value::Int(7)),
            ("conv_tol".to_string(), Value::Float(1e-13)),
            ("conv_tol_grad".to_string(), Value::Float(1e-11)),
            ("grid_weight_cutoff".to_string(), Value::Float(0.0)),
            ("dispersion".to_string(), Value::Str("none".into())),
        ];
        for (k, v) in extra {
            all.retain(|(key, _)| key != k);
            all.push((k.to_string(), v.clone()));
        }
        Controls::from_value(&Value::Object(all)).unwrap()
    }

    /// rSCAN itself, with the published switching interval: every component
    /// of the gradient (9), Hessian (81), polarizability (9) and first
    /// hyperpolarizability (27) against four-step fits of central
    /// differences, with the branch of every quadrature point recorded in
    /// the reference calculations and replayed in the displaced and field
    /// calculations.
    #[test]
    fn frozen_rscan_branches_pass_every_molecular_derivative_fit() {
        begin(TAG);
        for task in ["gradient", "hessian", "polarizability", "beta"] {
            run(task, &system(&WATER), &controls(&[])).unwrap();
        }
        let (recorded, _) = counts();
        replay();
        let mut failures = Vec::new();
        gradient_and_hessian("frozen rSCAN", &controls, &mut failures);
        polarizability_and_hyperpolarizability("frozen rSCAN", &controls, &mut failures);
        let (_, replayed) = counts();
        end();
        println!("frozen rSCAN: {recorded} recorded branches, {replayed} replayed decisions");
        assert!(recorded > 0 && replayed > 0);
        assert_no_failures("frozen rSCAN molecular derivatives", &failures);
    }
}
