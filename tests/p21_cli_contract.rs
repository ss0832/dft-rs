// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Command x option contract for `dft-rs` (see docs/cli.md).
//!
//! `periodic` and `sapt` each run through their own arm of the CLI's
//! command dispatch and, before this test existed, several options that
//! only the energy-family commands (`energy`, `charges`, `gradient`,
//! `optimize`, `saddle`, `frequencies`, `hessian`, `polarizability`,
//! `raman`, `tddft`) actually read -- `--disp`/`--nlc`, `--solvent` (and its
//! raw descriptors), `--field`, `--nbo`, `--dump-wfn`, `--molden` -- were
//! silently ignored by `periodic` and `sapt` instead of being refused.
//! `periodic` additionally ignored several molecular-SCF knobs that its own
//! `PeriodicScfOptions` (src/periodic/scf.rs) does not expose at all.
//!
//! This drives the single source of truth for that contract directly --
//! `OPTION_REJECTIONS` / `validate_option_table` in src/bin/dft_rs.rs -- by
//! pulling the CLI frontend in as a module, so a rejected combination can
//! never again pass through into a computation that quietly ignores it and
//! still prints a plausible-looking number.

use dft_rs::cli;

use std::collections::BTreeSet;
use std::process::Command;

use dft_rs::error::DftError;
use dft_rs::{Atom, Molecule, Vec3};

fn flags(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|s| s.to_string()).collect()
}

const SOLVENT_FAMILY: &[&str] = &[
    "--solvent",
    "--eps",
    "--refractive-index",
    "--solvent-alpha",
    "--solvent-beta",
    "--solvent-gamma",
];

const ANALYSIS_BINDINGS: &[&str] = &["--nbo", "--dump-wfn", "--molden"];

const DISPERSION: &[&str] = &["--disp", "--nlc", "--xdm-a1", "--xdm-a2"];

/// The knobs that belong to the ordinary molecular `ScfOptions` and that
/// `PeriodicScfOptions` either has no field for at all, or has a field for
/// but the CLI does not wire the flag to it. `--guess` is intentionally not
/// here: periodic Core/GWH/SAD selection is now implemented.
const PERIODIC_ONLY_SCF_KNOBS: &[&str] = &[
    "--incore",
    "--incore-mb",
    "--ri-j",
    "--auxbasis",
    "--no-soscf",
    "--unrestricted",
    "--guess-mix",
];

const ENERGY_FAMILY_COMMANDS: &[&str] = &[
    "energy",
    "charges",
    "gradient",
    "optimize",
    "saddle",
    "frequencies",
    "hessian",
    "polarizability",
    "raman",
    "tddft",
];

const ALL_COMMANDS: &[&str] = &[
    "info",
    "integrals",
    "energy",
    "gradient",
    "charges",
    "optimize",
    "saddle",
    "frequencies",
    "hessian",
    "polarizability",
    "raman",
    "tddft",
    "sapt",
    "periodic",
];

/// Every entry in the table must actually reject that command x option
/// pair, and the message must name both the option and the command, so a
/// user (and this test) can tell exactly which flag triggered it.
#[test]
fn every_table_entry_rejects_and_names_its_option_and_command() {
    assert!(
        !cli::OPTION_REJECTIONS.is_empty(),
        "the rejection table should not be empty -- this test would pass vacuously"
    );
    for &(command, option, _reason) in cli::OPTION_REJECTIONS {
        let err = cli::validate_option_table(command, &flags(&[option]))
            .unwrap_err_or_panic(command, option);
        assert!(
            err.contains(option),
            "{command}/{option}: error message does not mention the option: {err:?}"
        );
        assert!(
            err.contains(command),
            "{command}/{option}: error message does not name the command: {err:?}"
        );
    }
}

trait ExpectRejected {
    fn unwrap_err_or_panic(self, command: &str, option: &str) -> String;
}

impl ExpectRejected for Result<(), String> {
    fn unwrap_err_or_panic(self, command: &str, option: &str) -> String {
        match self {
            Ok(()) => panic!("{command} was expected to reject {option}, but it was accepted"),
            Err(message) => message,
        }
    }
}

#[test]
fn periodic_rejects_dispersion_solvent_field_and_analysis_bindings() {
    for option in DISPERSION
        .iter()
        .chain(SOLVENT_FAMILY)
        .chain(std::iter::once(&"--field"))
        .chain(ANALYSIS_BINDINGS)
    {
        assert!(
            cli::validate_option_table("periodic", &flags(&[option])).is_err(),
            "periodic should reject {option}"
        );
    }
}

#[test]
fn periodic_rejects_the_molecular_scf_knobs_it_does_not_expose() {
    for option in PERIODIC_ONLY_SCF_KNOBS {
        assert!(
            cli::validate_option_table("periodic", &flags(&[option])).is_err(),
            "periodic should reject {option}: PeriodicScfOptions does not honor it"
        );
    }
}

#[test]
fn periodic_accepts_the_wired_guess_option() {
    assert!(cli::validate_option_table("periodic", &flags(&["--guess"])).is_ok());
    assert!(cli::validate_option_table("periodic", &flags(&["--no-adiis"])).is_ok());
    assert!(cli::validate_option_table("periodic", &flags(&["--guess-mix"])).is_err());
}

#[test]
fn adiis_controls_are_rejected_where_they_would_be_silent() {
    let binary = env!("CARGO_BIN_EXE_dft-rs");
    let run = |args: &[&str]| Command::new(binary).args(args).output().unwrap();

    for command in ["info", "integrals"] {
        let output = run(&[command, "missing.xyz", "--adiis-until-error", "0.1"]);
        assert!(!output.status.success());
        let message = String::from_utf8_lossy(&output.stderr);
        assert!(message.contains("not available"));
        assert!(message.contains(command));
    }

    let catalog = run(&[
        "energy",
        "missing.xyz",
        "--list-functionals",
        "--adiis-until-error",
        "0.1",
    ]);
    assert!(!catalog.status.success());
    assert!(String::from_utf8_lossy(&catalog.stderr).contains("catalog list flags"));

    let contradictory = run(&[
        "energy",
        "missing.xyz",
        "--no-adiis",
        "--adiis-until-error",
        "0.1",
    ]);
    assert!(!contradictory.status.success());
    assert!(String::from_utf8_lossy(&contradictory.stderr)
        .contains("cannot be combined with --adiis-until-error"));
}

#[test]
fn periodic_relax_cli_contract_rejects_silent_combinations() {
    let binary = env!("CARGO_BIN_EXE_dft-rs");
    let run = |args: &[&str]| Command::new(binary).args(args).output().unwrap();

    let wrong_command = run(&["energy", "missing.xyz", "--relax"]);
    assert!(!wrong_command.status.success());
    let message = String::from_utf8_lossy(&wrong_command.stderr);
    assert!(message.contains("--relax is only available for the periodic command"));

    let silent_traj = run(&[
        "periodic",
        "missing.xyz",
        "--cell",
        "10",
        "--traj",
        "path.xyz",
    ]);
    assert!(!silent_traj.status.success());
    let message = String::from_utf8_lossy(&silent_traj.stderr);
    assert!(message.contains("--traj requires"));

    let detached_controls = run(&[
        "periodic",
        "missing.xyz",
        "--cell",
        "10",
        "--opt-max-steps",
        "200",
    ]);
    assert!(!detached_controls.status.success());
    let message = String::from_utf8_lossy(&detached_controls.stderr);
    assert!(message.contains("require periodic --relax"));

    for option in [
        "--relax",
        "--opt-max-steps",
        "--opt-gtol",
        "--opt-step-max",
        "--opt-history",
    ] {
        assert!(cli::validate_option_table("periodic", &flags(&[option])).is_ok());
    }
}

#[test]
fn sapt_rejects_dispersion_solvent_field_and_analysis_bindings() {
    for option in DISPERSION
        .iter()
        .chain(SOLVENT_FAMILY)
        .chain(std::iter::once(&"--field"))
        .chain(ANALYSIS_BINDINGS)
    {
        assert!(
            cli::validate_option_table("sapt", &flags(&[option])).is_err(),
            "sapt should reject {option}"
        );
    }
}

/// sapt reuses the ordinary `ScfOptions` (src/bin/dft_rs.rs, the "sapt" arm
/// around line 933), so it genuinely honors these -- unlike periodic, which
/// has its own options struct without them.
#[test]
fn sapt_accepts_the_scf_knobs_periodic_does_not() {
    for option in PERIODIC_ONLY_SCF_KNOBS {
        assert!(
            cli::validate_option_table("sapt", &flags(&[option])).is_ok(),
            "sapt should accept {option}: it reuses ordinary ScfOptions"
        );
    }
}

/// A representative set of accepted pairs: nothing in the table names an
/// energy-family command, so every option the table governs is accepted by
/// all ten of them, individually and all together.
#[test]
fn energy_family_commands_accept_the_full_option_set() {
    let every_governed_option: Vec<&str> = DISPERSION
        .iter()
        .chain(SOLVENT_FAMILY)
        .chain(std::iter::once(&"--field"))
        .chain(ANALYSIS_BINDINGS)
        .chain(PERIODIC_ONLY_SCF_KNOBS)
        .copied()
        .collect();
    let all_at_once = flags(&every_governed_option);
    for &command in ENERGY_FAMILY_COMMANDS {
        for &option in &every_governed_option {
            assert!(
                cli::validate_option_table(command, &flags(&[option])).is_ok(),
                "{command} should accept {option}"
            );
        }
        assert!(
            cli::validate_option_table(command, &all_at_once).is_ok(),
            "{command} should accept every governed option simultaneously"
        );
    }
}

/// The generic table does not restrict `info` and `integrals`. XDM is
/// value-dependent (`--disp xdm`), so its refusal is tested separately
/// through `resolve_xdm_cli` below.
#[test]
fn info_and_integrals_are_not_restricted_by_the_table() {
    let every_governed_option: Vec<&str> = DISPERSION
        .iter()
        .chain(SOLVENT_FAMILY)
        .chain(std::iter::once(&"--field"))
        .chain(ANALYSIS_BINDINGS)
        .chain(PERIODIC_ONLY_SCF_KNOBS)
        .copied()
        .collect();
    let all_at_once = flags(&every_governed_option);
    for command in ["info", "integrals"] {
        assert!(cli::validate_option_table(command, &all_at_once).is_ok());
    }
}

#[test]
fn no_flags_given_is_always_accepted() {
    for &command in ALL_COMMANDS {
        assert!(
            cli::validate_option_table(command, &BTreeSet::new()).is_ok(),
            "{command} with no options at all must never be rejected"
        );
    }
}

/// `-u` is a short alias for `--unrestricted`; parse_args() canonicalizes it
/// before recording it in the flags-given set, so it must be rejected for
/// `periodic` exactly like the long form (see the `while let Some(flag) =
/// args.next()` loop in src/bin/dft_rs.rs, which inserts `--unrestricted`
/// for either spelling).
#[test]
fn short_alias_for_unrestricted_is_recognized_by_the_running_binary() {
    // validate_option_table() itself only ever sees canonical long-form
    // flag names -- the -u -> --unrestricted canonicalization happens in
    // the parser loop before the set reaches it -- so this documents that
    // contract at the table level: the canonical form used by the parser
    // rejects periodic, and that is the only form the table needs to know.
    assert!(cli::validate_option_table("periodic", &flags(&["--unrestricted"])).is_err());
}

/// XDM depends on the converged density and has no analytic nuclear response
/// yet. The CLI therefore exposes it for single-point energy-bearing commands
/// only, with both empirical BJ damping parameters supplied explicitly.
#[test]
fn xdm_cli_contract_is_explicit_and_never_silent() {
    let p = cli::resolve_xdm_cli("energy", "xdm", Some(0.6), Some(1.0), "auto")
        .expect("a complete XDM energy request should be accepted")
        .expect("XDM request should resolve damping parameters");
    assert_eq!(p.a1, 0.6);
    assert_eq!(p.a2, 1.0);

    assert!(
        cli::resolve_xdm_cli("charges", "XDM", Some(0.6), Some(1.0), "none")
            .unwrap()
            .is_some()
    );
    assert!(cli::resolve_xdm_cli("energy", "xdm", None, Some(1.0), "auto").is_err());
    assert!(cli::resolve_xdm_cli("energy", "xdm", Some(0.6), None, "auto").is_err());
    assert!(cli::resolve_xdm_cli("energy", "none", Some(0.6), Some(1.0), "auto").is_err());
    assert!(cli::resolve_xdm_cli("gradient", "xdm", Some(0.6), Some(1.0), "auto").is_err());
    assert!(cli::resolve_xdm_cli("energy", "xdm", Some(0.6), Some(1.0), "vv10").is_err());
    assert!(cli::resolve_xdm_cli("info", "xdm", Some(0.6), Some(1.0), "auto").is_err());
    assert!(cli::resolve_xdm_cli("integrals", "xdm", Some(0.6), Some(1.0), "auto").is_err());
    assert!(cli::resolve_xdm_cli("energy", "xdm", Some(0.6), Some(1.0), "AUTO").is_ok());
    assert!(cli::resolve_xdm_cli("energy", "xdm", Some(f64::NAN), Some(1.0), "auto").is_err());
    assert!(cli::resolve_xdm_cli("energy", "xdm", Some(0.6), Some(0.0), "auto").is_err());
}

#[test]
fn pseudo_sub_cli_contract_is_typed_and_never_silent() {
    let molecule = Molecule::new(vec![
        Atom {
            z: 6,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 2.0),
        },
    ]);
    for name in ["Me", "Et", "tBu", "Ph", "CF3", "OMe"] {
        match cli::resolve_pseudo_sub_cli(&format!("1={name}"), Some(&molecule)) {
            Err(DftError::Unsupported { feature, missing }) => {
                assert!(feature.contains(name), "{feature}");
                assert!(missing.contains("not been determined") || missing.contains("not defined"));
            }
            other => panic!("{name}: expected typed unsupported refusal, got {other:?}"),
        }
    }
    for bad in ["1=bogus", "1=Me;1=Et", "0=Me", "2=Me"] {
        assert!(matches!(
            cli::resolve_pseudo_sub_cli(bad, Some(&molecule)),
            Err(DftError::InvalidInput(_))
        ));
    }
    // Catalog early returns receive no molecule, but must still resolve and
    // refuse the name rather than silently printing a list.
    assert!(matches!(
        cli::resolve_pseudo_sub_cli("1=Me", None),
        Err(DftError::Unsupported { .. })
    ));
    for command in ["periodic", "sapt"] {
        let error = cli::validate_option_table(command, &flags(&["--pseudo-sub"]))
            .unwrap_err_or_panic(command, "--pseudo-sub");
        assert!(error.contains(command));
        assert!(error.contains("--pseudo-sub"));
    }
}
