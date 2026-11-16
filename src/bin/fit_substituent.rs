// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Standalone entry point for the pseudo-substituent fitter mechanism.

use dft_rs::error::{DftError, Result};
use dft_rs::pseudo_sub::fit::run_synthetic_recovery_gate;

fn usage() -> &'static str {
    "fit_substituent --synthetic-check [--output candidate.fit]\n\
     [--cpu-threads N] [--memory-mb M]\n\
     \n\
     Runs the required known-parameter recovery gate through the same weighted\n\
     ridge, spectral projection, deterministic split and projected L-BFGS core\n\
     used by future physical fits. Candidate output never enables a named site.\n\
     \n\
     Physical --input fitting is deliberately unavailable until the staged\n\
     density/ESP/repulsion/force/torsion reference evaluator is implemented."
}

fn run() -> Result<()> {
    let (limits, args) = dft_rs::resources::cli_limits(std::env::args().skip(1).collect())?;
    limits.run(move || run_arguments(args))
}

fn run_arguments(args: Vec<String>) -> Result<()> {
    let mut synthetic = false;
    let mut output = None;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(());
            }
            "--synthetic-check" => synthetic = true,
            "--output" => {
                output = Some(args.next().ok_or_else(|| {
                    DftError::InvalidInput("--output requires a path".to_string())
                })?);
            }
            "--input" => {
                let _path = args
                    .next()
                    .ok_or_else(|| DftError::InvalidInput("--input requires a path".to_string()))?;
                return Err(DftError::Unsupported {
                    feature: "physical pseudo-substituent parameter fitting".to_string(),
                    missing: "the staged XYZ/reference observable evaluator (localized density, \
                        masked density residual, ESP, repulsion energies and forces, \
                        Casimir-Polder C6 provenance, and torsion data); a generic \
                        design-matrix input is intentionally not presented as chemistry"
                        .to_string(),
                });
            }
            _ => {
                return Err(DftError::InvalidInput(format!(
                    "unknown fit_substituent option `{arg}`\n{}",
                    usage()
                )));
            }
        }
    }
    if !synthetic {
        return Err(DftError::InvalidInput(format!(
            "no fitter action was selected\n{}",
            usage()
        )));
    }
    let result = run_synthetic_recovery_gate()?;
    let record = result.record.to_text()?;
    if let Some(path) = output {
        std::fs::write(&path, record)?;
        println!("synthetic recovery passed; wrote candidate record to {path}");
    } else {
        print!("{record}");
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
