// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Continuum-solvation gates: the self-consistent C-PCM electrostatic
//! solvation free energy against PySCF, and the SMD non-electrostatic
//! term's structural behaviour.
//!
//! Tolerance note: our cavity uses a self-generated quasi-uniform sphere
//! tessellation while PySCF uses a Lebedev rule, so the two discretize the
//! same continuum model slightly differently. The comparison is therefore
//! made on the SOLVATION SHIFT with a tolerance reflecting that
//! discretization difference; the discretization-free Born limit is gated
//! exactly in the unit tests of `src/pcm.rs`.

use dft_rs::basis::{data, MolBasis};
use dft_rs::math::Vec3;
use dft_rs::pcm::{solvent_by_name, PcmModel, PcmOptions, SmdSolvent};
use dft_rs::scf::{run_scf, EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};
use dft_rs::DftError;

fn water() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.05, 0.2217),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.03, 1.4309, -0.8867),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, -1.4309, -0.8867),
        },
    ])
}

fn opts() -> ScfOptions {
    ScfOptions {
        e_tol: 1e-11,
        p_tol: 1e-9,
        max_iter: 200,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    }
}

/// The IEF family against the frozen oracle.
///
/// Comparing absolute shifts would prove nothing here: IEF-PCM and C-PCM
/// differ by 3.7e-5 Ha on this system, an order of magnitude BELOW the
/// discretization gap between our quasi-uniform tessellation and the
/// reference's Lebedev rule. A gate on the absolute value would pass with
/// the D matrix deleted. So the comparison is made on the model-to-model
/// DIFFERENCE, where the shared discretization error largely cancels and
/// the D matrix is the entire signal.
#[test]
fn ief_pcm_model_differences_match_oracle() {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/oracle_data/pcm_water_ief.csv"),
    )
    .unwrap();
    let mut want: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let (k, v) = line.split_once(',').unwrap();
        want.insert(k, v.parse().unwrap());
    }
    let set = data::load("sto-3g").unwrap();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let mol = water();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let gas = run_scf(&mol, &mb, &opts(), &func, None).unwrap().energy;

    let shift = |model: PcmModel| -> f64 {
        let mut o = opts();
        o.pcm = Some(PcmOptions {
            model,
            ..PcmOptions::default()
        });
        run_scf(&mol, &mb, &o, &func, None).unwrap().energy - gas
    };
    let cpcm = shift(PcmModel::CPcm);
    let ief = shift(PcmModel::IefPcm);
    let ssvpe = shift(PcmModel::Ssvpe);
    let cosmo = shift(PcmModel::Cosmo);

    // Each model must still land in the right ballpark overall.
    for (name, got) in [
        ("dg_C-PCM", cpcm),
        ("dg_IEF-PCM", ief),
        ("dg_SS(V)PE", ssvpe),
        ("dg_COSMO", cosmo),
    ] {
        let w = want[name];
        assert!(
            (got - w).abs() < 5e-4,
            "{name}: {got} vs oracle {w} (discretization-limited gate)"
        );
    }
    // The discriminating part: the D matrix's own signature.
    for (name, got) in [("dg_IEF-PCM", ief - cpcm), ("dg_SS(V)PE", ssvpe - cpcm)] {
        let w = want[name] - want["dg_C-PCM"];
        assert!(
            w.abs() > 1e-5,
            "the oracle's {name} - C-PCM gap is too small to gate on"
        );
        assert!(
            (got - w).abs() < 0.3 * w.abs(),
            "{name} minus C-PCM: {got} vs oracle {w} -- the D matrix is the \
             whole signal here, so this is what a missing or mis-signed D \
             would break"
        );
    }
}

/// The FULL solvated analytic Hessian against central differences of the
/// (complete) analytic solvation gradient.
///
/// This is the end-to-end gate: it exercises the skeleton second
/// derivatives, the reaction field's entry in the Fock derivative, and the
/// CPKS coupling through `pcm::fock_response` all at once. Nothing here
/// holds the density fixed, so the density-response chain has to be right.
#[test]
fn pcm_analytic_hessians_match_numerical() {
    let set = data::load("sto-3g").unwrap();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let mol = water();
    for model in [PcmModel::CPcm, PcmModel::IefPcm, PcmModel::Ssvpe] {
        let mut o = opts();
        o.pcm = Some(PcmOptions {
            model,
            n_points: 26,
            ..PcmOptions::default()
        });
        let mb = MolBasis::build(&mol, &set).unwrap();
        let scf = run_scf(&mol, &mb, &o, &func, None).unwrap();
        let an = dft_rs::hessian::analytic::analytic_hessian_full(&mol, &mb, &o, &func, &scf, None)
            .unwrap()
            .hessian;

        let shift = |at: usize, ax: usize, h: f64| -> Molecule {
            let mut m = water();
            match ax {
                0 => m.atoms[at].position.x += h,
                1 => m.atoms[at].position.y += h,
                _ => m.atoms[at].position.z += h,
            }
            m
        };
        let grad = |at: usize, ax: usize, h: f64| -> Vec<Vec3> {
            let m = shift(at, ax, h);
            let b = MolBasis::build(&m, &set).unwrap();
            dft_rs::gradient::analytic_gradient(&m, &b, &o, &func, None)
                .unwrap()
                .gradient
        };
        let h = 2e-4;
        for at in 0..3 {
            for ax in 0..3 {
                let gp = grad(at, ax, h);
                let gm = grad(at, ax, -h);
                for bt in 0..3 {
                    for by in 0..3 {
                        let fd = (gp[bt].get(by) - gm[bt].get(by)) / (2.0 * h);
                        let a = an[(3 * at + ax, 3 * bt + by)];
                        assert!(
                            (a - fd).abs() < 3e-5,
                            "{model:?} Hessian ({at},{ax})({bt},{by}): analytic {a} vs FD {fd}"
                        );
                    }
                }
            }
        }
    }
}

/// The continuum's own response kernel, gated where it is unambiguous:
/// the analytic polarizability in solvent against finite differences of
/// the dipole. Without the reaction-field term in `fxc_contraction` the
/// analytic value keeps the solvent frozen while the finite-difference one
/// lets it relax, so the two disagree by several percent -- this is the
/// test that distinguishes them.
#[test]
fn solvated_polarizability_matches_finite_field() {
    let set = data::load("sto-3g").unwrap();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let mut o = opts();
    o.pcm = Some(PcmOptions {
        n_points: 50,
        ..PcmOptions::default()
    });
    let mol = water();
    let mb = MolBasis::build(&mol, &set).unwrap();

    let scf0 = run_scf(&mol, &mb, &o, &func, None).unwrap();
    let alpha = dft_rs::response::polarizability::polarizability(&mol, &mb, &func, None, &scf0, &o)
        .unwrap();

    // mu_i(F) by central differences of the SCF dipole.
    let dipole = |ax: usize, f: f64| -> f64 {
        let mut oo = o.clone();
        let mut v = Vec3::zero();
        match ax {
            0 => v.x = f,
            1 => v.y = f,
            _ => v.z = f,
        }
        oo.external_field = Some(v);
        let scf = run_scf(&mol, &mb, &oo, &func, None).unwrap();
        dft_rs::properties::dipole_moment(&mol, &mb, &scf).get(ax)
    };
    let h = 1e-4;
    for ax in 0..3 {
        let fd = (dipole(ax, h) - dipole(ax, -h)) / (2.0 * h);
        assert!(
            (alpha[ax][ax] - fd).abs() < 1e-4 * (1.0 + fd.abs()),
            "solvated alpha_{ax}{ax}: analytic {} vs finite field {fd}",
            alpha[ax][ax]
        );
    }
}

/// The AQUEOUS G_CDS on the elements whose CDS radius differs from their
/// intrinsic Coulomb radius (C, F, Si, S, Cl, Br).
///
/// The original water gate used only H, O, N and P, whose two radius sets
/// coincide -- so it could not see areas built from the wrong set, and did
/// not. This closes that blind spot: every case here changes if either the
/// radius or the tension is off.
#[test]
fn smd_cds_water_wide_matches_oracle() {
    let ang = 1.0 / 0.52917721092;
    let m = |v: Vec<(u8, [f64; 3])>| -> Molecule {
        Molecule::new(
            v.into_iter()
                .map(|(z, r)| Atom {
                    z,
                    position: Vec3::new(r[0] * ang, r[1] * ang, r[2] * ang),
                })
                .collect(),
        )
    };
    let tet = |z: u8, zl: u8, d: f64| -> Molecule {
        m(vec![
            (z, [0.0, 0.0, 0.0]),
            (zl, [d, d, d]),
            (zl, [-d, -d, d]),
            (zl, [d, -d, -d]),
            (zl, [-d, d, -d]),
        ])
    };
    let geom: std::collections::HashMap<&str, Molecule> = [
        ("F2", m(vec![(9, [0.0; 3]), (9, [0.0, 0.0, 1.412])])),
        ("Cl2", m(vec![(17, [0.0; 3]), (17, [0.0, 0.0, 1.988])])),
        ("Br2", m(vec![(35, [0.0; 3]), (35, [0.0, 0.0, 2.281])])),
        ("S2", m(vec![(16, [0.0; 3]), (16, [0.0, 0.0, 1.889])])),
        ("ClF", m(vec![(17, [0.0; 3]), (9, [0.0, 0.0, 1.628])])),
        ("BrCl", m(vec![(35, [0.0; 3]), (17, [0.0, 0.0, 2.136])])),
        (
            "SF2",
            m(vec![
                (16, [0.0; 3]),
                (9, [0.0, 0.0, 1.587]),
                (9, [1.520, 0.0, -0.457]),
            ]),
        ),
        ("SiF4", tet(14, 9, 0.904)),
        ("CF4", tet(6, 9, 0.767)),
        ("CCl4", tet(6, 17, 1.021)),
    ]
    .into_iter()
    .collect();

    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/oracle_data/smd_cds_water_wide.csv"),
    )
    .unwrap();
    let mut checked = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let (label, value) = line.split_once(',').unwrap();
        let want: f64 = value.parse().unwrap();
        let got = dft_rs::pcm::smd_cds_energy(&geom[label], SmdSolvent::Water, 4096).unwrap();
        assert!(
            (got - want).abs() < 1e-6,
            "SMD CDS {label} in water: {got} vs oracle {want} ({:.4} kcal/mol off)",
            (got - want) * 627.509_474_063_1
        );
        checked += 1;
    }
    assert_eq!(checked, 10, "oracle rows lost");
}

/// The measured pairwise surface-tension terms, on real molecules.
///
/// The coefficients were fitted on diatomic scans, where the two atoms'
/// areas move together. Hydrocarbons put several partners on one atom at
/// once and at very different area ratios, so this is what decides whether
/// the fitted SPLIT between the two atoms is right -- reproducing the
/// diatomic curve alone would not.
#[test]
fn smd_cds_water_pair_terms_match_oracle() {
    let ang = 1.0 / 0.52917721092;
    let m = |v: Vec<(u8, [f64; 3])>| -> Molecule {
        Molecule::new(
            v.into_iter()
                .map(|(z, r)| Atom {
                    z,
                    position: Vec3::new(r[0] * ang, r[1] * ang, r[2] * ang),
                })
                .collect(),
        )
    };
    let mut benzene = Vec::new();
    for k in 0..6 {
        let a = k as f64 * std::f64::consts::PI / 3.0;
        benzene.push((6u8, [1.39 * a.cos(), 1.39 * a.sin(), 0.0]));
    }
    for k in 0..6 {
        let a = k as f64 * std::f64::consts::PI / 3.0;
        benzene.push((1u8, [2.48 * a.cos(), 2.48 * a.sin(), 0.0]));
    }
    let geom: std::collections::HashMap<&str, Molecule> = [
        (
            "CH4",
            m(vec![
                (6, [0.0; 3]),
                (1, [0.629, 0.629, 0.629]),
                (1, [-0.629, -0.629, 0.629]),
                (1, [0.629, -0.629, -0.629]),
                (1, [-0.629, 0.629, -0.629]),
            ]),
        ),
        (
            "C2H6",
            m(vec![
                (6, [0.0, 0.0, 0.7680]),
                (6, [0.0, 0.0, -0.7680]),
                (1, [1.0192, 0.0, 1.1573]),
                (1, [-0.5096, 0.8827, 1.1573]),
                (1, [-0.5096, -0.8827, 1.1573]),
                (1, [-1.0192, 0.0, -1.1573]),
                (1, [0.5096, -0.8827, -1.1573]),
                (1, [0.5096, 0.8827, -1.1573]),
            ]),
        ),
        (
            "C2H4",
            m(vec![
                (6, [0.0, 0.0, 0.6695]),
                (6, [0.0, 0.0, -0.6695]),
                (1, [0.0, 0.9289, 1.2321]),
                (1, [0.0, -0.9289, 1.2321]),
                (1, [0.0, 0.9289, -1.2321]),
                (1, [0.0, -0.9289, -1.2321]),
            ]),
        ),
        (
            "C2H2",
            m(vec![
                (6, [0.0, 0.0, 0.6015]),
                (6, [0.0, 0.0, -0.6015]),
                (1, [0.0, 0.0, 1.6635]),
                (1, [0.0, 0.0, -1.6635]),
            ]),
        ),
        (
            "C3H8",
            m(vec![
                (6, [0.0, 0.0, 0.5875]),
                (6, [0.0, 1.2681, -0.2600]),
                (6, [0.0, -1.2681, -0.2600]),
                (1, [0.8760, 0.0, 1.2400]),
                (1, [-0.8760, 0.0, 1.2400]),
                (1, [0.0, 2.1499, 0.3893]),
                (1, [0.0, -2.1499, 0.3893]),
                (1, [0.8800, 1.3106, -0.9100]),
                (1, [-0.8800, 1.3106, -0.9100]),
                (1, [0.8800, -1.3106, -0.9100]),
                (1, [-0.8800, -1.3106, -0.9100]),
            ]),
        ),
        ("benzene", m(benzene)),
        (
            "cyclopropane",
            m(vec![
                (6, [0.0, 0.8703, 0.0]),
                (6, [0.7537, -0.4351, 0.0]),
                (6, [-0.7537, -0.4351, 0.0]),
                (1, [0.0, 1.4595, 0.9126]),
                (1, [0.0, 1.4595, -0.9126]),
                (1, [1.2640, -0.7298, 0.9126]),
                (1, [1.2640, -0.7298, -0.9126]),
                (1, [-1.2640, -0.7298, 0.9126]),
                (1, [-1.2640, -0.7298, -0.9126]),
            ]),
        ),
        (
            "PH3O",
            m(vec![
                (15, [0.0; 3]),
                (8, [0.0, 0.0, 1.476]),
                (1, [1.199, 0.0, -0.513]),
                (1, [-0.600, 1.039, -0.513]),
                (1, [-0.600, -1.039, -0.513]),
            ]),
        ),
    ]
    .into_iter()
    .collect();

    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/oracle_data/smd_cds_water_pairs.csv"),
    )
    .unwrap();
    let mut checked = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let (label, value) = line.split_once(',').unwrap();
        let want: f64 = value.parse().unwrap();
        let got = dft_rs::pcm::smd_cds_energy(&geom[label], SmdSolvent::Water, 8192).unwrap();
        assert!(
            (got - want).abs() < 2e-5,
            "SMD CDS {label} in water: {got} vs oracle {want} ({:.4} kcal/mol off)",
            (got - want) * 627.509_474_063_1
        );
        checked += 1;
    }
    assert_eq!(checked, 8, "oracle rows lost");
}

/// The aqueous G_CDS on elements BEYOND the nine that carry an atomic
/// surface tension. Their tension is zero, but the shared molecular term
/// still multiplies their exposed area, so a wrong CDS radius is visible
/// here -- which is the whole point of the expanded radius table.
#[test]
fn smd_cds_water_elements_match_oracle() {
    let ang = 1.0 / 0.52917721092;
    let m = |v: Vec<(u8, [f64; 3])>| -> Molecule {
        Molecule::new(
            v.into_iter()
                .map(|(z, r)| Atom {
                    z,
                    position: Vec3::new(r[0] * ang, r[1] * ang, r[2] * ang),
                })
                .collect(),
        )
    };
    let tet = |z: u8, zl: u8, d: f64| -> Molecule {
        m(vec![
            (z, [0.0; 3]),
            (zl, [d, d, d]),
            (zl, [-d, -d, d]),
            (zl, [d, -d, -d]),
            (zl, [-d, d, -d]),
        ])
    };
    let geom: std::collections::HashMap<&str, Molecule> = [
        ("LiF", m(vec![(3, [0.0; 3]), (9, [0.0, 0.0, 1.564])])),
        ("NaCl", m(vec![(11, [0.0; 3]), (17, [0.0, 0.0, 2.361])])),
        ("KI", m(vec![(19, [0.0; 3]), (53, [0.0, 0.0, 3.048])])),
        (
            "BF3",
            m(vec![
                (5, [0.0; 3]),
                (9, [1.307, 0.0, 0.0]),
                (9, [-0.653, 1.132, 0.0]),
                (9, [-0.653, -1.132, 0.0]),
            ]),
        ),
        (
            "BeH2",
            m(vec![
                (4, [0.0; 3]),
                (1, [0.0, 0.0, 1.334]),
                (1, [0.0, 0.0, -1.334]),
            ]),
        ),
        (
            "AlH3",
            m(vec![
                (13, [0.0; 3]),
                (1, [1.584, 0.0, 0.0]),
                (1, [-0.792, 1.372, 0.0]),
                (1, [-0.792, -1.372, 0.0]),
            ]),
        ),
        ("GeH4", tet(32, 1, 0.883)),
        (
            "AsH3",
            m(vec![
                (33, [0.0; 3]),
                (1, [1.399, 0.0, 0.616]),
                (1, [-0.700, 1.212, 0.616]),
                (1, [-0.700, -1.212, 0.616]),
            ]),
        ),
        (
            "SeH2",
            m(vec![
                (34, [0.0; 3]),
                (1, [0.0, 0.0, 1.460]),
                (1, [1.420, 0.0, -0.337]),
            ]),
        ),
        ("HI", m(vec![(1, [0.0; 3]), (53, [0.0, 0.0, 1.609])])),
        ("ICl", m(vec![(53, [0.0; 3]), (17, [0.0, 0.0, 2.321])])),
        ("SnH4", tet(50, 1, 0.983)),
        (
            "SbF3",
            m(vec![
                (51, [0.0; 3]),
                (9, [1.240, 0.0, 1.240]),
                (9, [-1.240, 0.0, 1.240]),
                (9, [0.0, 1.240, -1.240]),
            ]),
        ),
        (
            "TeH2",
            m(vec![
                (52, [0.0; 3]),
                (1, [0.0, 0.0, 1.658]),
                (1, [1.610, 0.0, -0.400]),
            ]),
        ),
        (
            "ZnCl2",
            m(vec![
                (30, [0.0; 3]),
                (17, [0.0, 0.0, 2.072]),
                (17, [0.0, 0.0, -2.072]),
            ]),
        ),
        (
            "CaF2",
            m(vec![
                (20, [0.0; 3]),
                (9, [0.0, 0.0, 2.100]),
                (9, [0.0, 0.0, -2.100]),
            ]),
        ),
    ]
    .into_iter()
    .collect();

    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/oracle_data/smd_cds_water_elements.csv"),
    )
    .unwrap();
    let mut checked = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let (label, value) = line.split_once(',').unwrap();
        let want: f64 = value.parse().unwrap();
        // 16384 latitude points; the residual is the surface quadrature's
        // own convergence, not a parameter error (BF3, whose three fluorines
        // sit in a plane, is the slowest case here).
        let got = dft_rs::pcm::smd_cds_energy(&geom[label], SmdSolvent::Water, 16384).unwrap();
        assert!(
            (got - want).abs() < 5e-7,
            "SMD CDS {label} in water: {got} vs oracle {want} ({:.4} kcal/mol off)",
            (got - want) * 627.509_474_063_1
        );
        checked += 1;
    }
    assert_eq!(checked, 16, "oracle rows lost");
}

/// The NON-AQUEOUS SMD surface-tension term against the frozen oracle.
///
/// Every molecule here is built from element pairs measured to carry no
/// distance-dependent term, so the atomic sum is the exact answer: any
/// disagreement is a real defect, not a missing parameter. The cases span
/// eight solvents whose descriptors differ widely, so a mis-assigned
/// descriptor column cannot survive them all.
#[test]
fn smd_cds_nonaqueous_matches_oracle() {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/oracle_data/smd_cds_nonaqueous.csv"),
    )
    .unwrap();
    let geom: std::collections::HashMap<&str, Molecule> = [
        ("HF", vec![(1, [0.0, 0.0, 0.0]), (9, [0.0, 0.0, 0.917])]),
        (
            "H2S",
            vec![
                (16, [0.0, 0.0, 0.0]),
                (1, [0.0, 0.0, 1.336]),
                (1, [1.294, 0.0, -0.335]),
            ],
        ),
        (
            "PH3",
            vec![
                (15, [0.0, 0.0, 0.0]),
                (1, [1.192, 0.0, 0.772]),
                (1, [-0.596, 1.032, 0.772]),
                (1, [-0.596, -1.032, 0.772]),
            ],
        ),
        (
            "SiH4",
            vec![
                (14, [0.0, 0.0, 0.0]),
                (1, [0.856, 0.856, 0.856]),
                (1, [-0.856, -0.856, 0.856]),
                (1, [0.856, -0.856, -0.856]),
                (1, [-0.856, 0.856, -0.856]),
            ],
        ),
        ("HCl", vec![(1, [0.0, 0.0, 0.0]), (17, [0.0, 0.0, 1.275])]),
        ("Br2", vec![(35, [0.0, 0.0, 0.0]), (35, [0.0, 0.0, 2.281])]),
        (
            "NH3",
            vec![
                (7, [0.0, 0.0, 0.0]),
                (1, [0.937, 0.0, 0.381]),
                (1, [-0.469, 0.812, 0.381]),
                (1, [-0.469, -0.812, 0.381]),
            ],
        ),
    ]
    .into_iter()
    .map(|(k, v)| {
        let ang = 1.0 / 0.52917721092;
        (
            k,
            Molecule::new(
                v.into_iter()
                    .map(|(z, r)| Atom {
                        z,
                        position: Vec3::new(r[0] * ang, r[1] * ang, r[2] * ang),
                    })
                    .collect(),
            ),
        )
    })
    .collect();

    let mut checked = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        // Solvent names contain commas ("1,4-dioxane"), so peel the label
        // off the front and the value off the back.
        let (label, rest) = line.split_once(',').unwrap();
        let (solvent, value) = rest.rsplit_once(',').unwrap();
        let want: f64 = value.parse().unwrap();
        let d = dft_rs::pcm::solvent_by_name(solvent)
            .unwrap_or_else(|| panic!("solvent `{solvent}` missing from the table"));
        let got =
            dft_rs::pcm::smd_cds_energy(&geom[label], dft_rs::pcm::SmdSolvent::NonAqueous(d), 4096)
                .unwrap();
        // 1e-6 Hartree is 0.0006 kcal/mol -- well inside the surface
        // quadrature's own convergence at this point count.
        assert!(
            (got - want).abs() < 1e-6,
            "SMD CDS {label} in {solvent}: {got} vs oracle {want} \
             ({:.4} kcal/mol off)",
            (got - want) * 627.509_474_063_1
        );
        checked += 1;
    }
    assert_eq!(checked, 8, "oracle rows lost");
}

/// End-to-end gate on the C-PCM analytic gradient: the full SCF gradient
/// with a reaction field against central differences of the SCF energy.
/// This closes the chain the unit tests gate piecewise (dS/dR, dF/dR,
/// dv^nuc/dR, dv^el/dR) and additionally proves the density-response terms
/// really do cancel, since nothing here holds the density fixed.
#[test]
fn cpcm_analytic_gradient_matches_fd() {
    let shift = |at: usize, ax: usize, h: f64| -> Molecule {
        let mut m = water();
        match ax {
            0 => m.atoms[at].position.x += h,
            1 => m.atoms[at].position.y += h,
            _ => m.atoms[at].position.z += h,
        }
        m
    };
    let set = data::load("sto-3g").unwrap();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let mut o = opts();
    o.pcm = Some(PcmOptions {
        n_points: 50,
        ..PcmOptions::default()
    });

    let mol = water();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let an = dft_rs::gradient::analytic_gradient(&mol, &mb, &o, &func, None)
        .unwrap()
        .gradient;

    let energy = |at: usize, ax: usize, h: f64| -> f64 {
        let m = shift(at, ax, h);
        let b = MolBasis::build(&m, &set).unwrap();
        run_scf(&m, &b, &o, &func, None).unwrap().energy
    };
    let h = 1e-4;
    for at in 0..3 {
        for ax in 0..3 {
            let fd = (energy(at, ax, h) - energy(at, ax, -h)) / (2.0 * h);
            let a = an[at].get(ax);
            assert!(
                (a - fd).abs() < 5e-6,
                "PCM gradient (atom {at}, axis {ax}): analytic {a} vs FD {fd}"
            );
        }
    }
}

#[test]
fn cpcm_solvation_shift_matches_pyscf() {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/oracle_data/pcm_water_cpcm.csv"),
    )
    .unwrap();
    let mut dg_ref = f64::NAN;
    let mut e_gas_ref = f64::NAN;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let p: Vec<&str> = line.split(',').collect();
        match p[0] {
            "dg_elst" => dg_ref = p[1].parse().unwrap(),
            "e_gas" => e_gas_ref = p[1].parse().unwrap(),
            _ => {}
        }
    }
    let mol = water();
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let gas = run_scf(&mol, &mb, &opts(), &func, None).unwrap();
    assert!(
        (gas.energy - e_gas_ref).abs() < 1e-8,
        "gas-phase reference drifted: {} vs {e_gas_ref}",
        gas.energy
    );
    let mut o = opts();
    o.pcm = Some(PcmOptions {
        model: PcmModel::CPcm,
        eps: 78.3553,
        ..PcmOptions::default()
    });
    let sol = run_scf(&mol, &mb, &o, &func, None).unwrap();
    let dg = sol.energy - gas.energy;
    assert!(sol.e_pcm < 0.0, "polarization energy must be negative");
    assert!(
        (dg - dg_ref).abs() < 3e-4,
        "C-PCM solvation shift {dg} vs PySCF {dg_ref} (delta = {:.2e} Ha)",
        dg - dg_ref
    );
}

/// Raising the dielectric constant must deepen the solvation, and the
/// vacuum limit (eps -> 1) must switch the reaction field off exactly.
#[test]
fn cpcm_dielectric_limits() {
    let mol = water();
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let gas = run_scf(&mol, &mb, &opts(), &func, None).unwrap();
    let energy_at = |eps: f64| {
        let mut o = opts();
        o.pcm = Some(PcmOptions {
            eps,
            ..PcmOptions::default()
        });
        run_scf(&mol, &mb, &o, &func, None).unwrap().energy
    };
    let e_vac = energy_at(1.0);
    assert!(
        (e_vac - gas.energy).abs() < 1e-10,
        "eps = 1 must reproduce the gas phase: {e_vac} vs {}",
        gas.energy
    );
    let e_low = energy_at(4.0);
    let e_high = energy_at(78.3553);
    assert!(
        e_high < e_low && e_low < gas.energy,
        "solvation must deepen with eps: {} {} {}",
        gas.energy,
        e_low,
        e_high
    );
}

/// SMD's non-electrostatic term is deliberately disabled until the full
/// surface-tension parameterization (pairwise sigma_ZZ' + molecular
/// sigma_M) is in and gated against `tests/oracle_data/smd_cds.csv`: an
/// atomic-tensions-only version disagreed with the reference by several
/// kcal/mol INCLUDING ITS SIGN, so it must not be reported as a number.
#[test]
fn smd_cds_water_matches_oracle() {
    // Same geometry the oracle was generated on (Angstrom -> Bohr).
    let a = 1.0 / 0.52917721092;
    let mol = Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.757 * a, 0.587 * a),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, -0.757 * a, 0.587 * a),
        },
    ]);
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data/smd_cds.csv"),
    )
    .unwrap();
    let want = text
        .lines()
        .filter(|l| !l.starts_with('#'))
        .find_map(|l| {
            let p: Vec<&str> = l.split(',').collect();
            (p.len() == 3 && p[0] == "water" && p[1] == "water")
                .then(|| p[2].parse::<f64>().unwrap())
        })
        .expect("water/water oracle row");
    // The azimuthal sweep is analytic, so the only discretization left is
    // in cos(theta) -- and it is fully converged: doubling the latitude
    // count must not move the answer.
    // The latitude integrand has kinks where a neighbour circle starts and
    // stops cutting the ring, so convergence is steady rather than
    // spectral: refining must monotonically tighten the agreement.
    let mut prev = f64::MAX;
    for n in [128usize, 512, 2048] {
        let got = dft_rs::pcm::smd_cds_energy(&mol, SmdSolvent::Water, n).unwrap();
        let d = ((got - want) * 627.5094740631).abs();
        assert!(
            d < prev,
            "refining {n} latitudes did not improve agreement ({d:.6} vs {prev:.6} kcal/mol)"
        );
        prev = d;
    }
    assert!(
        prev < 2e-3,
        "converged SMD G_CDS still off by {prev:.6} kcal/mol"
    );
}

/// Every required pair lacking a measured coefficient must be refused rather
/// than answered with the wrong atomic sum.
///
/// Classification: **sanity**. This freezes the deliberate capability
/// boundary, including the error variant and enough context to identify both
/// the pair and the solvent class without parsing an untyped failure.
#[test]
fn smd_cds_refuses_all_ten_unmeasured_pair_solvent_cases() {
    let nonaqueous = SmdSolvent::NonAqueous(solvent_by_name("methanol").unwrap());
    let cases = [
        (6, 7, SmdSolvent::Water, "water"),
        (6, 8, SmdSolvent::Water, "water"),
        (7, 8, SmdSolvent::Water, "water"),
        (1, 6, nonaqueous, "non-aqueous"),
        (1, 8, nonaqueous, "non-aqueous"),
        (6, 6, nonaqueous, "non-aqueous"),
        (6, 7, nonaqueous, "non-aqueous"),
        (6, 8, nonaqueous, "non-aqueous"),
        (7, 8, nonaqueous, "non-aqueous"),
        (8, 8, nonaqueous, "non-aqueous"),
    ];

    for (za, zb, solvent, solvent_name) in cases {
        let mol = Molecule::new(vec![
            Atom {
                z: za,
                position: Vec3::zero(),
            },
            Atom {
                z: zb,
                position: Vec3::new(0.0, 0.0, 2.0),
            },
        ]);
        match dft_rs::pcm::smd_cds_energy(&mol, solvent, 64) {
            Err(DftError::Unsupported { feature, missing }) => {
                assert!(
                    feature.contains(&format!("Z={za}/Z={zb}")),
                    "pair missing from refusal: {feature}"
                );
                assert!(
                    feature.contains(solvent_name),
                    "solvent missing from refusal: {feature}"
                );
                assert!(missing.contains("provenance-bearing coefficient"));
                assert!(missing.contains("molecular oracle"));
            }
            other => {
                panic!("Z={za}/Z={zb} in {solvent_name}: expected typed Unsupported, got {other:?}")
            }
        }
    }
}

/// The frozen SMD oracle is present and self-consistent, so the upcoming
/// implementation has a target: G_CDS must be solvent dependent, and for
/// water/methanol it is positive and of order 1 kcal/mol.
#[test]
fn smd_cds_oracle_is_usable() {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data/smd_cds.csv"),
    )
    .unwrap();
    let mut rows = Vec::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let p: Vec<&str> = line.split(',').collect();
        if p.len() == 3 {
            rows.push((
                p[0].to_string(),
                p[1].to_string(),
                p[2].parse::<f64>().unwrap(),
            ));
        }
    }
    assert!(rows.len() >= 30, "only {} oracle rows", rows.len());
    let water_in_water = rows
        .iter()
        .find(|(m, s, _)| m == "water" && s == "water")
        .expect("water/water row");
    let kcal = water_in_water.2 * 627.5094740631;
    assert!(
        (0.5..5.0).contains(&kcal),
        "water G_CDS reference {kcal} kcal/mol is not in the published range"
    );
    // Solvent dependence is the whole point of the CDS term.
    let spread: f64 = {
        let vals: Vec<f64> = rows
            .iter()
            .filter(|(m, _, _)| m == "methanol")
            .map(|r| r.2)
            .collect();
        vals.iter().cloned().fold(f64::MIN, f64::max)
            - vals.iter().cloned().fold(f64::MAX, f64::min)
    };
    assert!(spread > 1e-3, "oracle shows no solvent dependence");
}

/// The IEF family must stay on the analytic path rather than regressing to
/// the typed numerical fallback that preceded `fock_derivative_ief`.
#[test]
fn ief_family_hessians_are_analytic() {
    let mol = water();
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();

    for model in [PcmModel::IefPcm, PcmModel::Ssvpe] {
        let mut o = opts();
        o.pcm = Some(PcmOptions {
            model,
            eps: 78.3553,
            n_points: 50,
            ..PcmOptions::default()
        });
        let scf = run_scf(&mol, &mb, &o, &functional, None).unwrap();
        dft_rs::hessian::analytic_hessian_full(&mol, &mb, &o, &functional, &scf, None)
            .unwrap_or_else(|e| panic!("{model:?} unexpectedly left the analytic path: {e}"));
    }
}
