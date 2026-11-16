// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Compatibility executable `dft_rs_cli`; identical to `dft-rs`.

fn main() -> std::process::ExitCode {
    std::process::ExitCode::from(
        dft_rs::cli::run_arguments(std::env::args().skip(1).collect()) as u8,
    )
}
