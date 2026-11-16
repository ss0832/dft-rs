// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! GIAO NMR shielding gates.
//!
//! PySCF+pyscf-properties supplies an external full-tensor oracle for HF,
//! LDA/GGA, global hybrids, and RSHs. The additional gates below are internal
//! identities and physical limits that localize failures and cover properties
//! the external implementation cannot independently provide:
//!
//! - **gauge-origin independence.** The entire point of London orbitals.
//!   Move the coordinate origin of the vector potential and the shielding
//!   must not move. Every piece of the assembly (overlap derivative, core
//!   Hamiltonian derivative, two-electron derivative, the mixed `B`-`m`
//!   operator, and the CPHF response that ties them together) carries its
//!   own origin dependence, and they cancel only if every coefficient is
//!   right — a scale error anywhere leaves a residue;
//! - **free-atom limit.** A closed-shell atom with only s occupied orbitals
//!   has no paramagnetic term at all, so its shielding must be exactly
//!   isotropic and equal to Lamb's `(alpha^2/3) <sum 1/r>`, which is
//!   computed here from an independent nuclear-attraction integral;
//! - **one-centre GIAO = common gauge origin.** With every basis function
//!   on the same atom and the origin there, all London phases are 1;
//! - **molecular symmetry.** C2v water must give equal shieldings on the
//!   two hydrogens and a tensor diagonal in the symmetry axes;
//! - **response consistency.** `dP/dB` must satisfy the differentiated
//!   idempotency and normalization constraints of the density matrix.

use dft_rs::basis::{data, MolBasis};
use dft_rs::constants::FINE_STRUCTURE;
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::properties::magnetic::{
    magnetic_response, magnetic_response_with_grid, shielding_tensors, shielding_tensors_with_grid,
    Gauge, Shielding,
};
use dft_rs::scf::{run_scf, EriMode, ScfOptions, ScfResult};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, Family, FunctionalSpec, HybridParams, ResolvedFunctional};

fn tight() -> ScfOptions {
    ScfOptions {
        e_tol: 1e-12,
        p_tol: 1e-11,
        max_iter: 300,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    }
}

fn water() -> Molecule {
    // C2v, in the yz plane, C2 along z.
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.0, 0.2217),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 1.4309, -0.8867),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, -1.4309, -0.8867),
        },
    ])
}

fn skew_water() -> Molecule {
    // Deliberately off every Cartesian symmetry plane: all tensor components
    // are observable in the GGA gauge-origin gate.
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.16, -0.24, 0.31),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.41, 0.38, -0.62),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.72, 1.22, 0.47),
        },
    ])
}

fn shielding_gap(a: &[Shielding], b: &[Shielding]) -> f64 {
    assert_eq!(a.len(), b.len(), "shielding result lengths differ");
    let mut worst = 0.0_f64;
    for (left, right) in a.iter().zip(b) {
        for (x, y) in [
            (&left.tensor, &right.tensor),
            (&left.diamagnetic, &right.diamagnetic),
            (&left.paramagnetic, &right.paramagnetic),
        ] {
            for i in 0..3 {
                for j in 0..3 {
                    let diff = (x[i][j] - y[i][j]).abs();
                    assert!(diff.is_finite(), "non-finite shielding difference");
                    worst = worst.max(diff);
                }
            }
        }
    }
    worst
}

fn parse_csv_record(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut chars = line.chars().peekable();
    let mut quoted = false;
    while let Some(ch) = chars.next() {
        match ch {
            '"' if quoted && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                fields.push(std::mem::take(&mut field));
            }
            _ => field.push(ch),
        }
    }
    assert!(!quoted, "unterminated quoted CSV field: {line}");
    fields.push(field);
    fields
}

struct NmrOracleCase {
    energy: Option<f64>,
    values: [[[[f64; 3]; 3]; 3]; 3],
    seen: [[[[bool; 3]; 3]; 3]; 3],
    entries_per_atom: [usize; 3],
}

impl NmrOracleCase {
    fn new() -> Self {
        Self {
            energy: None,
            values: [[[[0.0; 3]; 3]; 3]; 3],
            seen: [[[[false; 3]; 3]; 3]; 3],
            entries_per_atom: [0; 3],
        }
    }
}

/// Independent PySCF+pyscf-properties GIAO shieldings. The frozen file is in
/// raw atomic units, with PySCF's final two tensor axes already transposed to
/// this crate's `(nuclear magnetic moment, external B)` convention.
#[test]
fn shieldings_match_pyscf_oracle() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/oracle_data/skew_water_sto3g_nmr.csv");
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(!text.is_empty(), "NMR oracle CSV is empty");
    let mut records = text
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'));
    let header = parse_csv_record(records.next().expect("NMR oracle CSV has no header"));
    assert_eq!(
        header,
        [
            "record",
            "functional",
            "pyscf_xc",
            "grid_level",
            "grid_points",
            "grid_prune",
            "scf_energy_hartree",
            "atom_index",
            "atomic_number",
            "part",
            "row_m_axis",
            "column_b_axis",
            "value_au",
            "reference_grid_level",
            "reference_value_au",
            "delta_au",
            "abs_delta_au",
        ],
        "unexpected NMR oracle CSV schema"
    );

    let axis = |value: &str| match value {
        "x" => 0,
        "y" => 1,
        "z" => 2,
        other => panic!("invalid NMR oracle axis {other:?}"),
    };
    let part = |value: &str| match value {
        "total" => 0,
        "dia" => 1,
        "para" => 2,
        other => panic!("invalid NMR oracle part {other:?}"),
    };
    let expected_z = [8_u8, 1, 1];
    let mut cases = std::collections::BTreeMap::<String, NmrOracleCase>::new();
    for line in records {
        let fields = parse_csv_record(line);
        assert_eq!(
            fields.len(),
            header.len(),
            "malformed NMR oracle row: {line}"
        );
        if fields[0] != "tensor" {
            continue;
        }
        let functional = &fields[1];
        let selected_grid = if functional == "hf" {
            fields[3] == "none"
        } else {
            fields[3] == "9"
        };
        if !selected_grid {
            continue;
        }

        let energy: f64 = fields[6].parse().unwrap();
        assert!(energy.is_finite(), "{functional}: non-finite oracle energy");
        let atom: usize = fields[7].parse().unwrap();
        assert!(
            atom < expected_z.len(),
            "{functional}: atom index {atom} out of bounds"
        );
        let z: u8 = fields[8].parse().unwrap();
        assert_eq!(
            z, expected_z[atom],
            "{functional}: wrong atomic number for atom {atom}"
        );
        let part = part(&fields[9]);
        let row = axis(&fields[10]);
        let col = axis(&fields[11]);
        let value: f64 = fields[12].parse().unwrap();
        assert!(
            value.is_finite(),
            "{functional}: non-finite oracle tensor value at atom {atom}, part {part}, ({row},{col})"
        );

        let case = cases
            .entry(functional.clone())
            .or_insert_with(NmrOracleCase::new);
        match case.energy {
            Some(previous) => assert_eq!(
                previous.to_bits(),
                energy.to_bits(),
                "{functional}: inconsistent SCF energies in oracle rows"
            ),
            None => case.energy = Some(energy),
        }
        assert!(
            !case.seen[part][atom][row][col],
            "{functional}: duplicate oracle entry for atom {atom}, part {part}, ({row},{col})"
        );
        case.seen[part][atom][row][col] = true;
        case.values[part][atom][row][col] = value;
        case.entries_per_atom[atom] += 1;
    }

    assert!(
        cases.len() >= 6,
        "expected at least six NMR oracle cases, got {}",
        cases.len()
    );
    for (functional, case) in &cases {
        assert!(case.energy.is_some(), "{functional}: missing oracle energy");
        for atom in 0..expected_z.len() {
            assert_eq!(
                case.entries_per_atom[atom], 27,
                "{functional}: atom {atom} does not have all 27 tensor entries"
            );
            for part in 0..3 {
                for row in 0..3 {
                    for col in 0..3 {
                        assert!(
                            case.seen[part][atom][row][col],
                            "{functional}: missing atom {atom}, part {part}, ({row},{col})"
                        );
                    }
                }
            }
            for row in 0..3 {
                for col in 0..3 {
                    let total = case.values[0][atom][row][col];
                    let pieces = case.values[1][atom][row][col] + case.values[2][atom][row][col];
                    let gap = (total - pieces).abs();
                    assert!(gap.is_finite(), "{functional}: non-finite raw part closure");
                    assert!(
                        gap < 5e-12 * total.abs().max(1.0),
                        "{functional}: raw total != dia + para at atom {atom}, ({row},{col}): {gap:e} au"
                    );
                }
            }
        }
    }

    let mol = skew_water();
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 9,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let opts = tight();
    let ppm = FINE_STRUCTURE * FINE_STRUCTURE * 1e6;
    assert!(ppm.is_finite() && ppm > 0.0);
    for (functional, case) in cases {
        let func = resolve(&FunctionalSpec::Combined(functional.clone())).unwrap();
        let is_hf = functional == "hf";
        let scf = run_scf(
            &mol,
            &mb,
            &opts,
            &func,
            if is_hf { None } else { Some(&grid) },
        )
        .unwrap();
        let reference_energy = case.energy.expect("energy checked above");
        let energy_gap = (scf.energy - reference_energy).abs();
        assert!(
            energy_gap.is_finite(),
            "{functional}: non-finite energy gap"
        );
        assert!(
            energy_gap < 5e-7,
            "{functional}: SCF energy {} vs oracle {} (gap {energy_gap:e} Eh)",
            scf.energy,
            reference_energy
        );
        let got = if is_hf {
            shielding_tensors(&mol, &mb, &func, &scf, &opts, Gauge::Giao, Vec3::zero()).unwrap()
        } else {
            shielding_tensors_with_grid(
                &mol,
                &mb,
                &func,
                Some(&grid),
                &scf,
                &opts,
                Gauge::Giao,
                Vec3::zero(),
            )
            .unwrap()
        };
        assert_eq!(got.len(), mol.len(), "{functional}: shielding atom count");
        let tolerance = if is_hf { 5e-4 } else { 5e-2 };
        for atom in 0..mol.len() {
            for part in 0..3 {
                let matrix = match part {
                    0 => &got[atom].tensor,
                    1 => &got[atom].diamagnetic,
                    _ => &got[atom].paramagnetic,
                };
                for row in 0..3 {
                    for col in 0..3 {
                        let actual = matrix[row][col];
                        let want = case.values[part][atom][row][col] * ppm;
                        let gap = (actual - want).abs();
                        assert!(
                            actual.is_finite() && want.is_finite() && gap.is_finite(),
                            "{functional}: non-finite shielding comparison at atom {atom}, part {part}, ({row},{col})"
                        );
                        assert!(
                            gap < tolerance,
                            "{functional}: atom {atom}, part {part}, ({row},{col}) = {actual:.9} vs oracle {want:.9} ppm (gap {gap:e})"
                        );
                    }
                }
            }
        }
    }
}

fn hf_setup(mol: &Molecule, basis: &str) -> (MolBasis, ResolvedFunctional, ScfResult) {
    let set = data::load(basis).unwrap();
    let mb = MolBasis::build(mol, &set).unwrap();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let opts = tight();
    let scf = run_scf(mol, &mb, &opts, &func, None).unwrap();
    (mb, func, scf)
}

/// The gate. Three unrelated gauge origins, including one far from the
/// molecule, must give the same shielding on every nucleus.
#[test]
fn giao_shielding_is_gauge_origin_independent() {
    let mol = water();
    let (mb, func, scf) = hf_setup(&mol, "sto-3g");
    let opts = tight();
    let origins = [
        Vec3::zero(),
        Vec3::new(0.83, -1.27, 0.41),
        Vec3::new(-4.5, 3.1, -2.7),
    ];
    let mut runs = Vec::new();
    for o in origins {
        runs.push(shielding_tensors(&mol, &mb, &func, &scf, &opts, Gauge::Giao, o).unwrap());
    }
    let base = &runs[0];
    for (idx, run) in runs.iter().enumerate().skip(1) {
        for (atom, (s, s0)) in run.iter().zip(base).enumerate() {
            let mut worst = 0.0_f64;
            for i in 0..3 {
                for j in 0..3 {
                    worst = worst.max((s.tensor[i][j] - s0.tensor[i][j]).abs());
                }
            }
            assert!(
                worst < 1e-5,
                "origin {idx}, atom {atom}: shielding moved by {worst:e} ppm \
                 (iso {} vs {})",
                s.isotropic(),
                s0.isotropic()
            );
        }
    }
    // And the answer must be a real shielding, not a numerical zero.
    let iso_o = base[0].isotropic();
    let (dia, para) = base[0].isotropic_parts();
    println!("H2O / HF / STO-3G  sigma(O) = {iso_o:.3} ppm (dia {dia:.3}, para {para:.3})");
    println!(
        "                   sigma(H) = {:.3} ppm, {:.3} ppm",
        base[1].isotropic(),
        base[2].isotropic()
    );
    assert!(
        (50.0..800.0).contains(&iso_o),
        "oxygen shielding {iso_o} ppm is outside any physical range"
    );
    assert!(para < 0.0, "the oxygen paramagnetic term must be negative");
}

/// The same gate on a split-valence basis, where the virtual space is big
/// enough for the paramagnetic term to be a real contribution rather than
/// the token one a minimal basis allows. More shells also means the
/// two-electron field derivative is exercised on many more quartets.
#[test]
fn gauge_independence_holds_in_a_split_valence_basis() {
    let mol = water();
    let (mb, func, scf) = hf_setup(&mol, "6-31g");
    let opts = tight();
    let a = shielding_tensors(&mol, &mb, &func, &scf, &opts, Gauge::Giao, Vec3::zero()).unwrap();
    let b = shielding_tensors(
        &mol,
        &mb,
        &func,
        &scf,
        &opts,
        Gauge::Giao,
        Vec3::new(3.7, -2.9, 1.6),
    )
    .unwrap();
    for (atom, (x, y)) in a.iter().zip(&b).enumerate() {
        let mut worst = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                worst = worst.max((x.tensor[i][j] - y.tensor[i][j]).abs());
            }
        }
        assert!(
            worst < 1e-5,
            "atom {atom}: shielding moved by {worst:e} ppm"
        );
    }
    let (dia, para) = a[0].isotropic_parts();
    println!(
        "H2O / HF / 6-31G   sigma(O) = {:.3} ppm (dia {dia:.3}, para {para:.3})",
        a[0].isotropic()
    );
    println!(
        "                   sigma(H) = {:.3} ppm, {:.3} ppm",
        a[1].isotropic(),
        a[2].isotropic()
    );
    // A split-valence basis must recover a substantial paramagnetic term.
    assert!(
        para < -50.0,
        "paramagnetic term {para} ppm is implausibly small"
    );
}

/// A custom exact-only RSH exercises both magnetic long-range exchange
/// pieces: the explicit GIAO `K'_erf[P]` derivative and the LR exchange
/// coupling of the first-order density. Origin independence requires their
/// coefficients and signs to cancel consistently with the full-range path.
#[test]
fn range_separated_exact_exchange_shielding_is_gauge_origin_independent() {
    let mol = water();
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let func = ResolvedFunctional::custom(
        "custom exact-only RSH",
        Vec::new(),
        HybridParams {
            cx: 0.21,
            clr: 0.67,
            omega: 0.33,
        },
        Family::Lda,
    );
    let opts = tight();
    let scf = run_scf(&mol, &mb, &opts, &func, None).unwrap();
    let origins = [
        Vec3::zero(),
        Vec3::new(0.73, -1.11, 0.46),
        Vec3::new(-3.8, 2.6, -1.9),
    ];
    let base = shielding_tensors(&mol, &mb, &func, &scf, &opts, Gauge::Giao, origins[0]).unwrap();
    for (idx, &origin) in origins.iter().enumerate().skip(1) {
        let shifted =
            shielding_tensors(&mol, &mb, &func, &scf, &opts, Gauge::Giao, origin).unwrap();
        for (atom, (a, b)) in base.iter().zip(&shifted).enumerate() {
            let mut worst = 0.0_f64;
            for i in 0..3 {
                for j in 0..3 {
                    worst = worst.max((a.tensor[i][j] - b.tensor[i][j]).abs());
                }
            }
            assert!(
                worst < 1e-5,
                "RSH origin {idx}, atom {atom}: shielding moved by {worst:e} ppm"
            );
        }
    }
    assert!(
        base[0].isotropic().abs() > 1.0,
        "RSH shielding is a numerical zero"
    );

    // The 1 MiB budget fits one STO-3G ERI tensor but not both the full and
    // LR kernels, so the LR engine must fall back to direct evaluation.
    let mut split_memory = opts.clone();
    split_memory.eri_mode = EriMode::InCore { limit_mb: 1 };
    let split = shielding_tensors(
        &mol,
        &mb,
        &func,
        &scf,
        &split_memory,
        Gauge::Giao,
        origins[0],
    )
    .unwrap();
    assert!(
        shielding_gap(&base, &split) < 1e-8,
        "LR-direct memory fallback changed the RSH shielding"
    );
    let mut direct = opts.clone();
    direct.eri_mode = EriMode::Direct {
        screen_threshold: 1e-12,
    };
    let all_direct =
        shielding_tensors(&mol, &mb, &func, &scf, &direct, Gauge::Giao, origins[0]).unwrap();
    assert!(
        shielding_gap(&base, &all_direct) < 1e-8,
        "direct J/K engines changed the RSH shielding"
    );
}

/// In the large-omega limit, `erf(omega r)/r` becomes the Coulomb kernel.
/// A 100% long-range exact-exchange reference must therefore reproduce HF,
/// including the separate diamagnetic and paramagnetic shielding pieces.
/// Unlike origin invariance, this catches a common wrong coefficient applied
/// consistently to both the explicit and response LR terms.
#[test]
fn large_omega_range_separated_shielding_reaches_the_hf_limit() {
    let mol = water();
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let opts = tight();
    let hf = ResolvedFunctional::hartree_fock();
    let lr_hf = ResolvedFunctional::custom(
        "long-range HF limit",
        Vec::new(),
        HybridParams {
            cx: 0.0,
            clr: 1.0,
            omega: 1.0e6,
        },
        Family::Lda,
    );
    let scf_hf = run_scf(&mol, &mb, &opts, &hf, None).unwrap();
    let scf_lr = run_scf(&mol, &mb, &opts, &lr_hf, None).unwrap();
    assert!(
        (scf_hf.energy - scf_lr.energy).abs() < 1e-8,
        "large-omega RSH energy {} does not reach HF {}",
        scf_lr.energy,
        scf_hf.energy
    );
    let origin = Vec3::new(0.37, -0.29, 0.41);
    let a = shielding_tensors(&mol, &mb, &hf, &scf_hf, &opts, Gauge::Giao, origin).unwrap();
    let b = shielding_tensors(&mol, &mb, &lr_hf, &scf_lr, &opts, Gauge::Giao, origin).unwrap();
    assert_eq!(a.len(), mol.len(), "HF shielding omitted an atom");
    assert_eq!(b.len(), a.len(), "large-omega RSH shielding length changed");
    let mut worst = 0.0_f64;
    for (hf_atom, lr_atom) in a.iter().zip(&b) {
        for (x, y) in [
            (&hf_atom.tensor, &lr_atom.tensor),
            (&hf_atom.diamagnetic, &lr_atom.diamagnetic),
            (&hf_atom.paramagnetic, &lr_atom.paramagnetic),
        ] {
            for i in 0..3 {
                for j in 0..3 {
                    let diff = (x[i][j] - y[i][j]).abs();
                    assert!(
                        diff.is_finite(),
                        "non-finite shielding difference at ({i},{j})"
                    );
                    worst = worst.max(diff);
                }
            }
        }
    }
    assert!(
        worst < 1e-4,
        "large-omega RSH and HF shielding pieces differ by {worst:e} ppm"
    );
}

/// At omega=0 the Erf kernel is identically zero. Adding any `clr` must
/// therefore leave the SCF and every shielding piece equal to the matching
/// global hybrid.
#[test]
fn zero_omega_range_separated_exchange_vanishes() {
    let mol = water();
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let opts = tight();
    let global = ResolvedFunctional::custom(
        "37% global exact exchange",
        Vec::new(),
        HybridParams {
            cx: 0.37,
            clr: 0.0,
            omega: 0.0,
        },
        Family::Lda,
    );
    let zero_lr = ResolvedFunctional::custom(
        "zero-omega RSH",
        Vec::new(),
        HybridParams {
            cx: 0.37,
            clr: 0.63,
            omega: 0.0,
        },
        Family::Lda,
    );
    let scf_global = run_scf(&mol, &mb, &opts, &global, None).unwrap();
    let scf_zero = run_scf(&mol, &mb, &opts, &zero_lr, None).unwrap();
    assert!(
        (scf_global.energy - scf_zero.energy).abs() < 1e-10,
        "zero-omega LR exchange changed the SCF energy"
    );
    let origin = Vec3::new(0.43, -0.36, 0.29);
    let a = shielding_tensors(&mol, &mb, &global, &scf_global, &opts, Gauge::Giao, origin).unwrap();
    let b = shielding_tensors(&mol, &mb, &zero_lr, &scf_zero, &opts, Gauge::Giao, origin).unwrap();
    assert!(
        shielding_gap(&a, &b) < 1e-9,
        "zero-omega LR exchange changed a shielding component"
    );
}

/// LDA adds a London-phase derivative of Vxc even though its magnetic
/// density kernel vanishes. Three unrelated origins gate that direct grid
/// contraction together with the rest of the magnetic response.
#[test]
fn svwn5_shielding_is_gauge_origin_independent() {
    let mol = water();
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let func = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 2,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let opts = tight();
    let scf = run_scf(&mol, &mb, &opts, &func, Some(&grid)).unwrap();
    let mut vv10_opts = opts.clone();
    vv10_opts.vv10 = Some(dft_rs::nlc::Vv10Params::ORIGINAL);
    assert!(
        shielding_tensors_with_grid(
            &mol,
            &mb,
            &func,
            Some(&grid),
            &scf,
            &vv10_opts,
            Gauge::Giao,
            Vec3::zero(),
        )
        .is_err(),
        "explicit SC-VV10 must be refused until its magnetic direct term exists"
    );
    assert!(
        shielding_tensors(&mol, &mb, &func, &scf, &opts, Gauge::Giao, Vec3::zero(),).is_err(),
        "the compatibility API must not silently rebuild or omit the LDA grid"
    );
    let origins = [
        Vec3::zero(),
        Vec3::new(0.91, -1.23, 0.57),
        Vec3::new(-4.1, 2.8, -2.2),
    ];
    let base = shielding_tensors_with_grid(
        &mol,
        &mb,
        &func,
        Some(&grid),
        &scf,
        &opts,
        Gauge::Giao,
        origins[0],
    )
    .unwrap();
    assert_eq!(base.len(), mol.len());
    for (idx, &origin) in origins.iter().enumerate().skip(1) {
        let shifted = shielding_tensors_with_grid(
            &mol,
            &mb,
            &func,
            Some(&grid),
            &scf,
            &opts,
            Gauge::Giao,
            origin,
        )
        .unwrap();
        assert_eq!(shifted.len(), base.len());
        let mut worst = 0.0_f64;
        for (a, b) in base.iter().zip(&shifted) {
            for i in 0..3 {
                for j in 0..3 {
                    let diff = (a.tensor[i][j] - b.tensor[i][j]).abs();
                    assert!(diff.is_finite(), "non-finite SVWN5 shielding difference");
                    worst = worst.max(diff);
                }
            }
        }
        assert!(
            worst < 1e-5,
            "SVWN5 origin {idx}: shielding moved by {worst:e} ppm"
        );
    }
    assert!(
        base[0].isotropic().abs() > 1.0,
        "SVWN5 shielding is a numerical zero"
    );

    // With a common gauge origin there are no London phases, so the LDA
    // direct term is exactly zero and the grid is not needed by the response.
    let common_origin = Vec3::new(0.31, -0.27, 0.18);
    let common_without_grid = shielding_tensors(
        &mol,
        &mb,
        &func,
        &scf,
        &opts,
        Gauge::CommonOrigin,
        common_origin,
    )
    .unwrap();
    let common_with_grid = shielding_tensors_with_grid(
        &mol,
        &mb,
        &func,
        Some(&grid),
        &scf,
        &opts,
        Gauge::CommonOrigin,
        common_origin,
    )
    .unwrap();
    assert_eq!(common_without_grid.len(), common_with_grid.len());
    for (a, b) in common_without_grid.iter().zip(&common_with_grid) {
        for (x, y) in [
            (&a.tensor, &b.tensor),
            (&a.diamagnetic, &b.diamagnetic),
            (&a.paramagnetic, &b.paramagnetic),
        ] {
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        (x[i][j] - y[i][j]).abs() < 1e-12,
                        "CommonOrigin LDA result depends on an unused grid"
                    );
                }
            }
        }
    }
}

/// Exercise the combined branch where LDA's direct London-phase Vxc term,
/// full-range exchange, and long-range Erf exchange are all nonzero.
#[test]
fn lda_range_separated_hybrid_shielding_is_gauge_origin_independent() {
    let mol = water();
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let lda = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    let func = ResolvedFunctional::custom(
        "SVWN5 plus global/LR exact exchange",
        lda.parts.clone(),
        HybridParams {
            cx: 0.18,
            clr: 0.42,
            omega: 0.33,
        },
        Family::Lda,
    );
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 2,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let opts = tight();
    let scf = run_scf(&mol, &mb, &opts, &func, Some(&grid)).unwrap();
    let origins = [
        Vec3::zero(),
        Vec3::new(0.68, -0.94, 0.52),
        Vec3::new(-3.2, 2.4, -1.7),
    ];
    let base = shielding_tensors_with_grid(
        &mol,
        &mb,
        &func,
        Some(&grid),
        &scf,
        &opts,
        Gauge::Giao,
        origins[0],
    )
    .unwrap();
    for (idx, &origin) in origins.iter().enumerate().skip(1) {
        let shifted = shielding_tensors_with_grid(
            &mol,
            &mb,
            &func,
            Some(&grid),
            &scf,
            &opts,
            Gauge::Giao,
            origin,
        )
        .unwrap();
        assert_eq!(shifted.len(), base.len());
        let mut worst = 0.0_f64;
        for (a, b) in base.iter().zip(&shifted) {
            for i in 0..3 {
                for j in 0..3 {
                    let diff = (a.tensor[i][j] - b.tensor[i][j]).abs();
                    assert!(diff.is_finite(), "non-finite hybrid-LDA shielding gap");
                    worst = worst.max(diff);
                }
            }
        }
        assert!(
            worst < 1e-5,
            "hybrid LDA origin {idx}: shielding moved by {worst:e} ppm"
        );
    }
}

/// GGA adds an origin-independent phase-gradient term in addition to the
/// coordinate moment already present in LDA. Exercise it as a pure GGA, a
/// global hybrid, and a range-separated hybrid on an asymmetric molecule so
/// no off-diagonal tensor component vanishes by point-group symmetry.
#[test]
fn gga_and_hybrid_shielding_are_gauge_origin_independent() {
    let mol = skew_water();
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 2,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let opts = tight();
    let origins = [
        Vec3::zero(),
        Vec3::new(0.91, -1.23, 0.57),
        Vec3::new(-4.1, 2.8, -2.2),
    ];

    for name in ["pbe", "b3lyp", "cam-b3lyp"] {
        let func = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let scf = run_scf(&mol, &mb, &opts, &func, Some(&grid)).unwrap();
        assert!(
            shielding_tensors(&mol, &mb, &func, &scf, &opts, Gauge::Giao, origins[0]).is_err(),
            "{name}: compatibility API silently omitted the GGA grid"
        );
        let base = shielding_tensors_with_grid(
            &mol,
            &mb,
            &func,
            Some(&grid),
            &scf,
            &opts,
            Gauge::Giao,
            origins[0],
        )
        .unwrap();
        assert_eq!(base.len(), mol.len(), "{name}: shielding omitted an atom");
        for (idx, &origin) in origins.iter().enumerate().skip(1) {
            let shifted = shielding_tensors_with_grid(
                &mol,
                &mb,
                &func,
                Some(&grid),
                &scf,
                &opts,
                Gauge::Giao,
                origin,
            )
            .unwrap();
            assert_eq!(
                shifted.len(),
                mol.len(),
                "{name}: shifted run omitted an atom"
            );
            let mut worst = 0.0_f64;
            for (a, b) in base.iter().zip(&shifted) {
                for i in 0..3 {
                    for j in 0..3 {
                        let diff = (a.tensor[i][j] - b.tensor[i][j]).abs();
                        assert!(diff.is_finite(), "{name}: non-finite GGA shielding gap");
                        worst = worst.max(diff);
                    }
                }
            }
            assert!(
                worst < 1e-5,
                "{name} origin {idx}: shielding moved by {worst:e} ppm"
            );
        }
        assert!(
            base.iter()
                .flat_map(|atom| atom.tensor)
                .flatten()
                .all(f64::is_finite),
            "{name}: non-finite base shielding"
        );

        // A common-origin calculation has no London basis derivative and
        // therefore must not depend on whether the otherwise unused grid is
        // supplied to the response API.
        let common_origin = Vec3::new(0.31, -0.27, 0.18);
        let common_without_grid = shielding_tensors(
            &mol,
            &mb,
            &func,
            &scf,
            &opts,
            Gauge::CommonOrigin,
            common_origin,
        )
        .unwrap();
        let common_with_grid = shielding_tensors_with_grid(
            &mol,
            &mb,
            &func,
            Some(&grid),
            &scf,
            &opts,
            Gauge::CommonOrigin,
            common_origin,
        )
        .unwrap();
        assert!(
            shielding_gap(&common_without_grid, &common_with_grid) < 1e-12,
            "{name}: CommonOrigin result depends on an unused grid"
        );
    }
}

#[test]
fn meta_gga_covariant_tau_shielding_is_gauge_origin_independent() {
    let mol = skew_water();
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 2,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let opts = tight();
    let meta = resolve(&FunctionalSpec::Combined("tpss".into())).unwrap();
    let scf = run_scf(&mol, &mb, &opts, &meta, Some(&grid)).unwrap();
    assert!(
        shielding_tensors(&mol, &mb, &meta, &scf, &opts, Gauge::Giao, Vec3::zero()).is_err(),
        "meta-GGA magnetic response must not omit its molecular grid"
    );
    let origins = [
        Vec3::zero(),
        Vec3::new(0.91, -1.23, 0.57),
        Vec3::new(-4.1, 2.8, -2.2),
    ];
    let base = shielding_tensors_with_grid(
        &mol,
        &mb,
        &meta,
        Some(&grid),
        &scf,
        &opts,
        Gauge::Giao,
        origins[0],
    )
    .unwrap();
    assert_eq!(base.len(), mol.len());
    let response = magnetic_response_with_grid(
        &mol,
        &mb,
        &meta,
        Some(&grid),
        &scf,
        &opts,
        Gauge::Giao,
        origins[0],
    )
    .unwrap();
    for axis in 0..3 {
        assert!(
            response.residual[axis].is_finite() && response.residual[axis] < 1e-9,
            "TPSS GIAO CPHF axis {axis} residual is {}",
            response.residual[axis]
        );
        assert!(response.d[axis]
            .as_slice()
            .iter()
            .all(|value| value.is_finite()));
    }
    for (idx, &origin) in origins.iter().enumerate().skip(1) {
        let shifted = shielding_tensors_with_grid(
            &mol,
            &mb,
            &meta,
            Some(&grid),
            &scf,
            &opts,
            Gauge::Giao,
            origin,
        )
        .unwrap();
        assert_eq!(shifted.len(), base.len());
        let mut worst = 0.0_f64;
        for (a, b) in base.iter().zip(&shifted) {
            for i in 0..3 {
                for j in 0..3 {
                    let diff = (a.tensor[i][j] - b.tensor[i][j]).abs();
                    assert!(diff.is_finite(), "non-finite TPSS shielding difference");
                    worst = worst.max(diff);
                }
            }
        }
        assert!(
            worst < 1e-5,
            "TPSS covariant-tau origin {idx}: shielding moved by {worst:e} ppm"
        );
    }
    assert!(base.iter().all(|s| s.isotropic().is_finite()));

    let common_origin = Vec3::new(-0.37, 0.62, -0.19);
    assert!(
        magnetic_response_with_grid(
            &mol,
            &mb,
            &meta,
            None,
            &scf,
            &opts,
            Gauge::CommonOrigin,
            common_origin,
        )
        .is_err(),
        "meta-GGA CommonOrigin response must not omit its current-kernel grid"
    );
    let common_response = magnetic_response_with_grid(
        &mol,
        &mb,
        &meta,
        Some(&grid),
        &scf,
        &opts,
        Gauge::CommonOrigin,
        common_origin,
    )
    .unwrap();
    for axis in 0..3 {
        assert!(
            common_response.residual[axis].is_finite() && common_response.residual[axis] < 1e-9,
            "TPSS CommonOrigin CPHF axis {axis} residual is {}",
            common_response.residual[axis]
        );
        assert!(common_response.d[axis]
            .as_slice()
            .iter()
            .all(|value| value.is_finite()));
    }
    let common_shielding = shielding_tensors_with_grid(
        &mol,
        &mb,
        &meta,
        Some(&grid),
        &scf,
        &opts,
        Gauge::CommonOrigin,
        common_origin,
    )
    .unwrap();
    assert_eq!(common_shielding.len(), mol.len());
    assert!(common_shielding.iter().all(|shielding| {
        shielding
            .tensor
            .iter()
            .flatten()
            .chain(shielding.diamagnetic.iter().flatten())
            .chain(shielding.paramagnetic.iter().flatten())
            .all(|value| value.is_finite())
    }));
}

/// A closed-shell atom whose occupied orbitals are all s has no orbital
/// angular momentum to respond with: the paramagnetic term vanishes
/// identically and the shielding is Lamb's diamagnetic formula, exactly
/// isotropic. The reference value comes from a nuclear-attraction integral,
/// i.e. a completely different code path.
#[test]
fn closed_shell_atom_reproduces_the_lamb_formula() {
    for (z, basis) in [(2u8, "6-31g"), (4u8, "6-31g")] {
        let at = Vec3::new(0.29, -0.41, 0.63);
        let mol = Molecule::new(vec![Atom { z, position: at }]);
        let (mb, func, scf) = hf_setup(&mol, basis);
        let opts = tight();
        for gauge in [Gauge::Giao, Gauge::CommonOrigin] {
            let s = shielding_tensors(&mol, &mb, &func, &scf, &opts, gauge, at).unwrap();
            let t = s[0].tensor;
            // Isotropy: every off-diagonal zero, all diagonals equal.
            let iso = s[0].isotropic();
            for i in 0..3 {
                for j in 0..3 {
                    let want = if i == j { iso } else { 0.0 };
                    assert!(
                        (t[i][j] - want).abs() < 1e-6 * iso.abs().max(1.0),
                        "Z={z} {gauge:?}: tensor is not isotropic at ({i},{j}): {}",
                        t[i][j]
                    );
                }
            }
            let (_, para) = s[0].isotropic_parts();
            assert!(
                para.abs() < 1e-6 * iso.abs(),
                "Z={z} {gauge:?}: paramagnetic term {para} should vanish for an s-only atom"
            );
            // Lamb: sigma = (alpha^2/3) <sum_i 1/r_i>.
            let probe = Molecule::new(vec![Atom { z: 1, position: at }]);
            let v = dft_rs::integrals::nuclear_matrix(&mb, &probe, Some(&[1.0]));
            let inv_r = -trace_product(&scf.p_total(), &v);
            let lamb = FINE_STRUCTURE * FINE_STRUCTURE * 1e6 * inv_r / 3.0;
            println!("Z={z} {gauge:?}: sigma = {iso:.6} ppm, Lamb = {lamb:.6} ppm");
            assert!(
                (iso - lamb).abs() < 1e-6 * lamb,
                "Z={z} {gauge:?}: sigma {iso} vs Lamb {lamb}"
            );
        }
    }
}

/// With every basis function on one centre and the gauge origin there, all
/// London phases are identically 1, so the two gauge treatments must agree
/// to machine precision — pinning them against each other.
#[test]
fn one_centre_giao_equals_the_common_gauge_origin() {
    let at = Vec3::new(-0.31, 0.52, 0.17);
    let mol = Molecule::new(vec![Atom {
        z: 10,
        position: at,
    }]);
    let (mb, func, scf) = hf_setup(&mol, "6-31g");
    let opts = tight();
    let a = shielding_tensors(&mol, &mb, &func, &scf, &opts, Gauge::Giao, at).unwrap();
    let b = shielding_tensors(&mol, &mb, &func, &scf, &opts, Gauge::CommonOrigin, at).unwrap();
    let mut worst = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            worst = worst.max((a[0].tensor[i][j] - b[0].tensor[i][j]).abs());
        }
    }
    println!(
        "Ne sigma = {:.6} ppm (GIAO) / {:.6} ppm (CGO)",
        a[0].isotropic(),
        b[0].isotropic()
    );
    assert!(
        worst < 1e-8,
        "one-centre GIAO and CGO differ by {worst:e} ppm"
    );
    assert!(a[0].isotropic() > 300.0, "neon shielding looks wrong");
}

/// C2v water: the two hydrogens are related by the sigma_v mirror `y -> -y`,
/// so their isotropic shieldings are equal and the oxygen tensor is
/// diagonal in the (x, y, z) symmetry frame. Neither is imposed anywhere in
/// the code — the symmetry has to come out of the integrals.
#[test]
fn shielding_respects_the_molecular_symmetry() {
    let mol = water();
    let (mb, func, scf) = hf_setup(&mol, "sto-3g");
    let opts = tight();
    let s = shielding_tensors(
        &mol,
        &mb,
        &func,
        &scf,
        &opts,
        Gauge::Giao,
        Vec3::new(0.4, -0.3, 0.9),
    )
    .unwrap();
    let (h1, h2) = (s[1].isotropic(), s[2].isotropic());
    assert!(
        (h1 - h2).abs() < 1e-6 * h1.abs().max(1.0),
        "the two hydrogens differ: {h1} vs {h2} ppm"
    );
    let o = s[0].tensor;
    let scale = o[0][0].abs().max(o[1][1].abs()).max(o[2][2].abs());
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                assert!(
                    o[i][j].abs() < 1e-6 * scale,
                    "oxygen tensor off-diagonal ({i},{j}) = {} is not zero",
                    o[i][j]
                );
            }
        }
    }
    // The two hydrogen tensors are mirror images: sigma_yz changes sign.
    let (a, b) = (s[1].tensor, s[2].tensor);
    for (i, j) in [(0, 0), (1, 1), (2, 2), (1, 2), (2, 1)] {
        let sign = if (i, j) == (1, 2) || (i, j) == (2, 1) {
            -1.0
        } else {
            1.0
        };
        assert!(
            (a[i][j] - sign * b[i][j]).abs() < 1e-6 * scale,
            "hydrogen mirror relation broken at ({i},{j}): {} vs {}",
            a[i][j],
            b[i][j]
        );
    }
}

/// The first-order density must satisfy the differentiated constraints:
/// `d/dB (P S P - 2P) = 0` and `d/dB Tr[P S] = 0`. These hold whatever the
/// CPHF solution is, so they gate the construction of `dP/dB` — in
/// particular the occupied-occupied block that the GIAO overlap derivative
/// forces and that is easy to forget.
#[test]
fn magnetic_density_response_satisfies_the_density_constraints() {
    let mol = water();
    let (mb, func, scf) = hf_setup(&mol, "sto-3g");
    let opts = tight();
    let o = Vec3::new(0.55, -0.21, 0.33);
    let r = magnetic_response(&mol, &mb, &func, &scf, &opts, Gauge::Giao, o).unwrap();
    let p = scf.p_total();
    let s = &scf.overlap;
    for comp in 0..3 {
        let d = &r.d[comp];
        let sd = &r.s[comp];
        // Antisymmetry.
        let mut asym = 0.0_f64;
        let mut scale = 0.0_f64;
        for i in 0..mb.nao_sph {
            for j in 0..mb.nao_sph {
                asym = asym.max((d[(i, j)] + d[(j, i)]).abs());
                scale = scale.max(d[(i, j)].abs());
            }
        }
        assert!(
            asym < 1e-10 * scale.max(1.0),
            "component {comp}: dP/dB not antisymmetric"
        );
        assert!(
            scale > 1e-8,
            "component {comp}: the response is identically zero"
        );
        // Idempotency: D S P + P S D + P Shat P = 2 D.
        let lhs = d
            .matmul(&s.matmul(&p))
            .plus(&p.matmul(&s.matmul(d)))
            .plus(&p.matmul(&sd.matmul(&p)));
        let mut worst = 0.0_f64;
        for i in 0..mb.nao_sph {
            for j in 0..mb.nao_sph {
                worst = worst.max((lhs[(i, j)] - 2.0 * d[(i, j)]).abs());
            }
        }
        assert!(
            worst < 1e-8 * scale,
            "component {comp}: idempotency residual {worst:e} (scale {scale:e})"
        );
        // Electron count: Tr[D S] + Tr[P Shat] = 0.
        let count = trace_product(d, s) + trace_product(&p, sd);
        assert!(
            count.abs() < 1e-9,
            "component {comp}: the electron count moves by {count:e}"
        );
        assert!(
            r.residual[comp] < 1e-9,
            "component {comp} CPHF residual too large"
        );
    }
}

/// `Tr[A B]` (not the Frobenius dot: the two differ by a sign when both
/// matrices are antisymmetric).
fn trace_product(a: &Matrix, b: &Matrix) -> f64 {
    let mut s = 0.0;
    for i in 0..a.rows {
        for j in 0..a.cols {
            s += a[(i, j)] * b[(j, i)];
        }
    }
    s
}

trait Plus {
    fn plus(&self, other: &Matrix) -> Matrix;
}

impl Plus for Matrix {
    fn plus(&self, other: &Matrix) -> Matrix {
        let mut out = self.clone();
        for (a, b) in out.as_mut_slice().iter_mut().zip(other.as_slice()) {
            *a += *b;
        }
        out
    }
}
