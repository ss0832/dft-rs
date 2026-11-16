// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! License compliance audit — runs as part of every phase gate.
//!
//! Machine-checkable half of the per-phase license audit:
//! 1. every Rust source carries the two-line SPDX + copyright header,
//! 2. no source claims a copyleft license or carries porting-attribution
//!    comments (project policy: none are allowed),
//! 3. direct dependencies stay within the approved pure-Rust allowlist,
//! 4. bundled data files carry a PROVENANCE header,
//! 5. the Apache-2.0 LICENSE file is present.
//!
//! The manual half (no verbatim code from libxc/libint/dftd3/dftd4 etc.,
//! clean data provenance) is a checklist item recorded at each phase close.

use std::fs;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn source_files(subdirs: &[&str], ext: &str) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for sub in subdirs {
        walk(&crate_root().join(sub), &mut files);
    }
    files.retain(|p| p.extension().and_then(|e| e.to_str()) == Some(ext));
    files.sort();
    files
}

#[test]
fn every_rust_source_has_spdx_and_copyright_header() {
    let files = source_files(&["src", "tests", "benches", "examples"], "rs");
    assert!(!files.is_empty(), "no Rust sources found — audit is broken");
    for path in files {
        let text = fs::read_to_string(&path).unwrap();
        let mut lines = text.lines();
        assert_eq!(
            lines.next().map(str::trim),
            Some("// SPDX-License-Identifier: Apache-2.0"),
            "{} must start with the Apache-2.0 SPDX line",
            path.display()
        );
        assert_eq!(
            lines.next().map(str::trim),
            Some("// Copyright (c) 2026 ss0832"),
            "{} must carry the copyright line as its second line",
            path.display()
        );
    }
}

#[test]
fn no_copyleft_claims_or_porting_attribution_comments() {
    // Policy: no source may claim a copyleft license, and porting-origin
    // comments ("adapted from ...") are not written in this project.
    // The needles are assembled at runtime so this test file does not itself
    // contain the forbidden byte sequences.
    let spdx = "SPDX-License-Identifier: ";
    let forbidden: Vec<String> = ["GPL", "LGPL", "MPL"]
        .iter()
        .map(|lic| format!("{spdx}{lic}"))
        .chain(
            ["the author", "pm6-rs", "aTB-rs", "gfn1-rs"]
                .iter()
                .map(|src| format!("{} {src}", ["adapted", "from"].join(" "))),
        )
        .collect();
    for path in source_files(&["src", "tests", "benches", "examples"], "rs") {
        let text = fs::read_to_string(&path).unwrap();
        for needle in &forbidden {
            assert!(
                !text.contains(needle.as_str()),
                "{} contains forbidden text: {needle:?}",
                path.display()
            );
        }
    }
}

#[test]
fn direct_dependencies_stay_in_the_allowlist() {
    const ALLOWED: [&str; 4] = ["libm", "rayon", "faer", "pyo3"];
    let manifest = fs::read_to_string(crate_root().join("Cargo.toml")).unwrap();
    let mut in_deps = false;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            // [dependencies] and target-specific [target.*.dependencies];
            // dev/build dependency sections are intentionally also audited.
            in_deps = trimmed.trim_matches(['[', ']']).ends_with("dependencies");
            continue;
        }
        if !in_deps || trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let name = trimmed
            .split(['=', ' ', '.'])
            .next()
            .unwrap_or_default()
            .trim();
        if name.is_empty() {
            continue;
        }
        assert!(
            ALLOWED.contains(&name),
            "dependency `{name}` is not in the approved allowlist {ALLOWED:?}; \
             adding a dependency requires an explicit user-approved policy change"
        );
    }
}

#[test]
fn bundled_data_files_carry_provenance_headers() {
    // Bundled scientific data (basis sets, dispersion tables, grid node
    // parameters) must document its origin in the first lines of the file.
    let mut data_files = Vec::new();
    walk(&crate_root().join("src"), &mut data_files);
    data_files.retain(|p| {
        matches!(
            p.extension().and_then(|e| e.to_str()),
            Some("g94") | Some("csv") | Some("dat") | Some("json")
        )
    });
    for path in data_files {
        let text = fs::read_to_string(&path).unwrap();
        let head: String = text.lines().take(20).collect::<Vec<_>>().join("\n");
        assert!(
            head.contains("PROVENANCE"),
            "{} lacks a PROVENANCE header in its first 20 lines",
            path.display()
        );
    }
}

#[test]
fn apache_license_file_is_present() {
    let text = fs::read_to_string(crate_root().join("LICENSE")).unwrap();
    assert!(
        text.contains("Apache License") && text.contains("Version 2.0"),
        "LICENSE must be the Apache-2.0 text"
    );
}
