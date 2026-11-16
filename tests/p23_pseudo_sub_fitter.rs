// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Exact/sanity gates for the standalone Phase-4 candidate fitter.

use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use dft_rs::pseudo_sub::fit::FitParameterRecord;

fn fitter() -> Command {
    Command::new(env!("CARGO_BIN_EXE_fit_substituent"))
}

#[test]
fn synthetic_cli_recovers_and_round_trips_a_candidate_record() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "dft-rs-fit-substituent-{}-{stamp}.fit",
        std::process::id()
    ));

    let output = fitter()
        .args(["--synthetic-check", "--output"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "fit_substituent failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = fs::read_to_string(&path).unwrap();
    let _ = fs::remove_file(&path);
    let record = FitParameterRecord::from_text(&text).unwrap();
    assert!(record.converged);
    assert!(record.train_rmse < 1e-6);
    assert!(record.validation_rmse < 1e-6);
    assert_eq!(record.parameters.len(), 5);
    assert!(text.contains("status=candidate-unaccepted"));
}

#[test]
fn physical_input_and_unknown_options_are_explicit_refusals() {
    let physical = fitter()
        .args(["--input", "references.csv"])
        .output()
        .unwrap();
    assert!(!physical.status.success());
    let error = String::from_utf8_lossy(&physical.stderr);
    assert!(error.contains("physical pseudo-substituent parameter fitting"));
    assert!(error.contains("reference observable evaluator"));

    let unknown = fitter().arg("--invent-parameters").output().unwrap();
    assert!(!unknown.status.success());
    assert!(String::from_utf8_lossy(&unknown.stderr)
        .contains("unknown fit_substituent option `--invent-parameters`"));
}
