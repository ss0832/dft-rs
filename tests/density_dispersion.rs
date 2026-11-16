// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use dft_rs::basis::{data, BasisSet, MolBasis};
use dft_rs::constants::ANGSTROM_TO_BOHR;
use dft_rs::dispersion::{d1, mbd, ts};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::grid::{eval::eval_aos_full, gauss_legendre, GridOptions, MolecularGrid};
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::nlc::vv09;
use dft_rs::scf::{run_scf, sad::sad_density, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

const POSITIONS: [[f64; 3]; 4] = [
    [0.13, -0.27, 0.19],
    [1.78, 0.34, -0.08],
    [-0.49, 1.65, 0.41],
    [0.31, -0.78, 1.91],
];
const NUMBERS: [u8; 4] = [7, 1, 1, 1];

fn ammonia() -> (Molecule, System) {
    let mol = Molecule::new(
        NUMBERS
            .into_iter()
            .zip(POSITIONS)
            .map(|(z, [x, y, w])| Atom {
                z,
                position: Vec3::new(x, y, w),
            })
            .collect(),
    );
    (mol, System::molecule(NUMBERS.to_vec(), POSITIONS.to_vec()))
}

fn references() -> [ts::FreeAtom; 4] {
    // The H/N C6 coefficients and radii are the individual values in the
    // 2009 TS paper. Static alpha is supplied independently. The hydrogen
    // 1s alpha is 9/2, and nitrogen alpha is taken from the neutral-atom
    // static polarizability compilation already used by the library.
    let hydrogen = ts::FreeAtom {
        polarizability: 4.5,
        c6: 6.5,
        radius: 1.64 * ANGSTROM_TO_BOHR,
    };
    let nitrogen = ts::FreeAtom {
        polarizability: 7.4,
        c6: 24.2,
        radius: 1.77 * ANGSTROM_TO_BOHR,
    };
    [nitrogen, hydrogen, hydrogen, hydrogen]
}

fn density_field(basis: &MolBasis, grid: &MolecularGrid, p: &Matrix) -> Vec<f64> {
    let ao = eval_aos_full(basis, &grid.points, false, false).phi;
    (0..grid.len())
        .map(|i| {
            let mut rho = 0.0;
            for a in 0..p.rows {
                for b in 0..p.cols {
                    rho += ao[(i, a)] * p[(a, b)] * ao[(i, b)];
                }
            }
            rho
        })
        .collect()
}

// Generate each proatom on its own neutral atomic basis and explicitly
// integrate the numerator and denominator of the volume ratio. No
// block-diagonal molecular SAD extraction is used by this oracle.
fn oracle_volumes(mol: &Molecule, basis: &BasisSet, grid: &MolecularGrid, rho: &[f64]) -> Vec<f64> {
    let proatoms: Vec<_> = mol
        .atoms
        .iter()
        .map(|atom| {
            let isolated = Molecule::new(vec![atom.clone()]);
            let atomic_basis = MolBasis::build(&isolated, basis).unwrap();
            let p = sad_density(&isolated, &atomic_basis).unwrap();
            density_field(&atomic_basis, grid, &p)
        })
        .collect();
    let pro_density: Vec<_> = (0..grid.len())
        .map(|i| proatoms.iter().map(|atom| atom[i]).sum::<f64>())
        .collect();
    mol.atoms
        .iter()
        .zip(proatoms)
        .map(|(atom, pro)| {
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for i in 0..grid.len() {
                if pro_density[i] > 1e-300 {
                    let r = (grid.points[i] - atom.position).norm();
                    let free = grid.weights[i] * pro[i] * r.powi(3);
                    denominator += free;
                    numerator += grid.weights[i] * rho[i] * (pro[i] / pro_density[i]) * r.powi(3);
                }
            }
            numerator / denominator
        })
        .collect()
}

fn oracle_parameters(ratios: &[f64], reference: &[ts::FreeAtom]) -> Vec<[f64; 3]> {
    ratios
        .iter()
        .zip(reference)
        .map(|(&ratio, r)| {
            [
                ratio * r.polarizability,
                ratio * ratio * r.c6,
                ratio.powf(1.0 / 3.0) * r.radius,
            ]
        })
        .collect()
}

fn oracle_ts(mol: &Molecule, atoms: &[[f64; 3]], sr: f64) -> f64 {
    let mut energy = 0.0;
    for (i, [alpha, c6, radius]) in atoms.iter().copied().enumerate() {
        for (j, [alpha2, c62, radius2]) in atoms[..i].iter().copied().enumerate() {
            let r = (mol.atoms[i].position - mol.atoms[j].position).norm();
            let c_ab = 2.0 * c6 * c62 / ((alpha2 / alpha) * c6 + (alpha / alpha2) * c62);
            let damping = 1.0 + f64::exp(20.0 - 20.0 * r / (sr * (radius + radius2)));
            energy -= c_ab / (r.powi(6) * damping);
        }
    }
    energy
}

// ACFDT log-det of the Coulomb-coupled oscillator susceptibility, with an
// independent scalar Cholesky and a transformed imaginary-frequency integral.
// Production uses the direct zero-point eigenvalue formula.
fn oracle_mbd(mol: &Molecule, atoms: &[[f64; 3]], beta: f64) -> f64 {
    let n = 3 * mol.len();
    let (nodes, weights) = gauss_legendre(96);
    let mut integral = 0.0;
    for (x, weight) in nodes.into_iter().zip(weights) {
        let s = 2.0 / (1.0 - x);
        let omega = s - 1.0;
        let response: Vec<_> = atoms
            .iter()
            .map(|&[alpha, c6, _]| {
                let single_mode = (4.0 / 3.0) * c6 / alpha.powi(2);
                alpha / (1.0 + (omega / single_mode).powi(2))
            })
            .collect();
        let mut ll = vec![vec![0.0_f64; n]; n];
        for (i, a) in mol.atoms.iter().enumerate() {
            for (j, b) in mol.atoms[..i].iter().enumerate() {
                let r = a.position - b.position;
                let radius_sum = atoms[i][2] + atoms[j][2];
                let f = 1.0 / (1.0 + (6.0 * (1.0 - r.norm() / (beta * radius_sum))).exp());
                let scale = (response[i] * response[j]).sqrt() * f / r.norm().powi(5);
                for a in 0..3 {
                    for b in 0..3 {
                        ll[3 * i + a][3 * j + b] =
                            scale * (r.norm2() * f64::from(a == b) - 3.0 * r.get(a) * r.get(b));
                    }
                }
            }
        }
        for i in 0..n {
            ll[i][i] = 1.0;
            for j in 0..=i {
                let dot: f64 = (0..j).map(|k| ll[i][k] * ll[j][k]).sum();
                ll[i][j] = if i == j {
                    (ll[i][j] - dot).sqrt()
                } else {
                    (ll[i][j] - dot) / ll[j][j]
                };
            }
        }
        integral += weight * (0.5 * s * s) * (0..n).map(|i| 2.0 * ll[i][i].ln()).sum::<f64>();
    }
    integral / (2.0 * std::f64::consts::PI)
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() < tolerance,
        "{actual:.15e} versus {expected:.15e}"
    );
}

#[test]
fn molecular_d1_pair_energy_uses_harmonic_mean_and_damping_23() {
    let (mol, system) = ammonia();
    let atoms = [
        d1::AtomParameters {
            c6: 18.2,
            radius: 2.6,
        },
        d1::AtomParameters {
            c6: 3.7,
            radius: 1.9,
        },
        d1::AtomParameters {
            c6: 4.1,
            radius: 1.7,
        },
        d1::AtomParameters {
            c6: 4.4,
            radius: 2.1,
        },
    ];
    let scale = 0.83;
    let mut oracle = 0.0;
    for (i, p) in atoms.iter().enumerate() {
        for (j, q) in atoms[..i].iter().enumerate() {
            let r = (mol.atoms[i].position - mol.atoms[j].position).norm();
            oracle -= scale * 2.0 * p.c6 * q.c6
                / (p.c6 + q.c6)
                / r.powi(6)
                / (1.0 + (23.0 * (1.0 - r / (p.radius + q.radius))).exp());
        }
    }
    let result = d1::energy(&mol, &atoms, scale).unwrap();
    close(result, oracle, 1e-15);
    assert!(result < 0.0 && d1::energy(&mol, &atoms[..3], scale).is_err());
    let single = Molecule::new(vec![mol.atoms[0].clone()]);
    assert_eq!(d1::energy(&single, &atoms[..1], scale).unwrap(), 0.0);
    let settings = Value::object()
        .with("xc", "blyp")
        .with("basis", "sto-3g")
        .with("grid_level", 0usize)
        .with("reference", "rks");
    let reference = run("energy", &system, &Controls::from_value(&settings).unwrap()).unwrap();
    let input = Controls::from_value(
        &settings
            .with("dispersion", "d1")
            .with("dispersion_energy_scale", scale)
            .with("dispersion_free_c6", atoms.map(|a| a.c6).to_vec())
            .with("dispersion_free_radii", atoms.map(|a| a.radius).to_vec()),
    )
    .unwrap();
    let report = run("energy", &system, &input).unwrap();
    let field = |out: &Value, key| out.get(key).and_then(Value::as_f64).unwrap();
    close(field(&report, "dispersion_energy"), oracle, 1e-15);
    close(
        field(&report, "energy") - field(&reference, "energy"),
        oracle,
        6e-15,
    );
    assert!(run("gradient", &system, &input)
        .unwrap_err()
        .to_string()
        .contains("D1 dispersion response"));
    println!("NH3 geometry D1 with explicit test parameters: energy={result:.15e} Eh independent={oracle:.15e} Eh");
}

#[test]
fn pbe_ammonia_ts_and_mbd_terms_match_independent_oracles() {
    let (mol, _) = ammonia();
    let basis_set = data::load("sto-3g").unwrap();
    let basis = MolBasis::build(&mol, &basis_set).unwrap();
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 1,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let pbe = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let scf = run_scf(&mol, &basis, &ScfOptions::default(), &pbe, Some(&grid)).unwrap();
    let p = scf.p_total();
    let expected_ratios =
        oracle_volumes(&mol, &basis_set, &grid, &density_field(&basis, &grid, &p));
    let refs = references();
    let effective = oracle_parameters(&expected_ratios, &refs);
    let ts = ts::energy_on_density(&mol, &basis, &grid, &p, &refs, 0.94).unwrap();
    let mbd = mbd::ts_energy_on_density(&mol, &basis, &grid, &p, &refs, 0.81).unwrap();
    let screened =
        mbd::rsscs_energy_on_density(&mol, &basis, &grid, &p, &refs, 0.81, 64, 0.8).unwrap();
    let ets = oracle_ts(&mol, &effective, 0.94);
    let embd = oracle_mbd(&mol, &effective, 0.81);
    close(ts.energy, ets, 2e-13);
    close(mbd.energy, embd, 2e-12);
    let screened_atoms: Vec<_> = screened
        .oscillators
        .atoms
        .iter()
        .map(|a| [a.polarizability, a.c6, a.radius])
        .collect();
    let screened_oracle = oracle_mbd(&mol, &screened_atoms, 0.81);
    close(screened.oscillators.energy, screened_oracle, 2e-12);
    assert!(screened.minimum_screening_eigenvalue > 0.0);
    assert!(screened.screening_residual < 1e-13);
    for (atom, reference) in ts.atoms.iter().zip(&effective) {
        close(atom.polarizability, reference[0], 2e-12);
        close(atom.c6, reference[1], 2e-11);
        close(atom.radius, reference[2], 2e-12);
    }
    for i in 0..mol.len() {
        close(ts.atoms[i].volume_ratio, expected_ratios[i], 2e-13);
        close(mbd.atoms[i].volume_ratio, expected_ratios[i], 2e-13);
        close(
            mbd.frequencies[i],
            4.0 * effective[i][1] / (3.0 * effective[i][0].powi(2)),
            2e-13,
        );
    }
    assert!(scf.converged && ts.energy < 0.0 && mbd.energy < 0.0);
    println!("PBE/STO-3G NH3: iterations={} points={} reference={:.15e} TS={:.15e}/{ets:.15e} MBD@TS={:.15e}/{embd:.15e} minimum mode squared={:.15e} ratios={expected_ratios:?}",scf.iterations,grid.len(),scf.energy,ts.energy,mbd.energy,mbd.minimum_squared_frequency);
    println!("PBE/STO-3G NH3 MBD@rsSCS: E={:.15e}/{screened_oracle:.15e} Hartree, min screening={:.12e}, screening residual={:.3e}, min mode squared={:.12e} Hartree^2",
        screened.oscillators.energy, screened.minimum_screening_eigenvalue,
        screened.screening_residual, screened.oscillators.minimum_squared_frequency);
}

#[test]
fn hydrogen_oscillator_dimer_matches_closed_form_and_london_limit() {
    let h = references()[1];
    let refs = [h; 2];
    let at = |distance| {
        Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::zero(),
            },
            Atom {
                z: 1,
                position: Vec3::new(distance, 0.0, 0.0),
            },
        ])
    };
    let beta = 0.81;
    let r = 7.0_f64;
    let mbd = mbd::ts_energy_from_volumes(&at(r), &[1.0; 2], &refs, beta).unwrap();
    let omega = 4.0 * h.c6 / (3.0 * h.polarizability.powi(2));
    let t =
        h.polarizability / (r.powi(3) * (1.0 + (6.0 * (1.0 - r / (2.0 * h.radius * beta))).exp()));
    let expected = [
        1.0 - 2.0 * t,
        1.0 - t,
        1.0 - t,
        1.0 + t,
        1.0 + t,
        1.0 + 2.0 * t,
    ]
    .map(|x| omega * x.sqrt());
    for (&value, expected) in mbd.coupled_frequencies.iter().zip(expected) {
        close(value, expected, 3e-14);
    }
    close(
        mbd.energy,
        0.5 * (expected.into_iter().sum::<f64>() - 6.0 * omega),
        3e-14,
    );
    let far = 30.0_f64;
    let tail = mbd::ts_energy_from_volumes(&at(far), &[1.0; 2], &refs, beta)
        .unwrap()
        .energy;
    close(-tail * far.powi(6), h.c6, 1e-5);
    let isolated = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }]);
    close(
        mbd::ts_energy_from_volumes(&isolated, &[1.0], &[h], beta)
            .unwrap()
            .energy,
        0.0,
        1e-15,
    );
    assert_eq!(
        ts::energy_from_volumes(&isolated, &[1.0], &[h], 0.94)
            .unwrap()
            .energy,
        0.0
    );
    assert!(
        mbd::ts_energy_from_volumes(&at(0.1), &[1.0; 2], &refs, beta)
            .unwrap_err()
            .to_string()
            .contains("unstable MBD@TS")
    );
    println!(
        "H/H oscillator r={r} Bohr omega={omega:.12e} MBD@TS={:.12e} far C6={:.12e}",
        mbd.energy,
        -tail * far.powi(6)
    );
}

fn controls(model: &str, reference: &str, open_shell: bool) -> Controls {
    let atoms = references();
    let scale = if model == "ts" { 0.94 } else { 0.81 };
    Controls::from_value(
        &Value::object()
            .with("xc", "pbe")
            .with("basis", "sto-3g")
            .with("grid_level", 1usize)
            .with("charge", i64::from(open_shell))
            .with("multiplicity", if open_shell { 2usize } else { 1usize })
            .with("reference", reference)
            .with("dispersion", model)
            .with("dispersion_radius_scale", scale)
            .with(
                "dispersion_free_polarizabilities",
                atoms.map(|a| a.polarizability).to_vec(),
            )
            .with("dispersion_free_c6", atoms.map(|a| a.c6).to_vec())
            .with("dispersion_free_radii", atoms.map(|a| a.radius).to_vec()),
    )
    .unwrap()
}

#[test]
fn molecule_energy_routes_report_one_post_scf_addition() {
    let (_, system) = ammonia();
    let value = |report: &Value, key| report.get(key).and_then(Value::as_f64).unwrap();
    for (reference, open_shell) in [
        ("rks", false),
        ("uks", true),
        ("roks", true),
        ("gks", false),
    ] {
        for model in ["ts", "mbd-ts", "mbd"] {
            let input = controls(model, reference, open_shell);
            let out = run("energy", &system, &input).unwrap();
            let mut bare = input.to_value();
            if let Value::Object(entries) = &mut bare {
                entries.retain(|(key, _)| {
                    key != "dispersion"
                        && !key.starts_with("dispersion_free_")
                        && key != "dispersion_radius_scale"
                });
            }
            let reference_out =
                run("energy", &system, &Controls::from_value(&bare).unwrap()).unwrap();
            let correction = value(&out, "dispersion_energy");
            close(
                value(&out, "energy") - value(&reference_out, "energy"),
                correction,
                1e-12,
            );
            close(
                value(out.get("energy_components").unwrap(), "dispersion"),
                correction,
                1e-15,
            );
            assert!(correction < 0.0);
            assert!(run("gradient", &system, &input)
                .unwrap_err()
                .to_string()
                .contains("density-dependent dispersion response"));
            if model == "mbd" {
                assert!(value(&out, "mbd_minimum_screening_eigenvalue") > 0.0);
                assert!(value(&out, "mbd_screening_residual") < 1e-13);
            }
            println!("{reference} {model}: reference={:.15e} correction={correction:.15e} total={:.15e} residual={:.15e}",value(&reference_out,"energy"),value(&out,"energy"),value(&out, "scf_orbital_residual"));
        }
    }
}

#[test]
fn density_dispersion_rejects_defining_range_separated_dispersion() {
    let (_, system) = ammonia();
    for xc in ["wb97x-d", "wb97x-v"] {
        let settings = controls("ts", "rks", false).to_value().with("xc", xc);
        let error = run("energy", &system, &Controls::from_value(&settings).unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("already requires"), "{xc}: {error}");
    }
}

fn vv09_oracle(grid: &MolecularGrid, rho: &[f64], sigma: &[f64], spin: &[f64]) -> f64 {
    let pi = std::f64::consts::PI;
    let (nodes, weights) = gauss_legendre(8);
    let mut energy = 0.0;
    let local: Vec<_> = (0..grid.len())
        .map(|i| {
            if rho[i] < 1e-8 {
                return (0.0, 1.0, 1.0);
            }
            let density = rho[i];
            let spin_scale =
                ((1.0 + spin[i]).cbrt().powi(2) + (1.0 - spin[i]).cbrt().powi(2)) / 2.0;
            let screening = (4.0 / pi) * (3.0 * pi.powi(2) * density).cbrt() * spin_scale.powi(2);
            let plasma_squared = 4.0 * pi * density;
            let gap_squared = 0.0089 * (sigma[i] / density.powi(2)).powi(2);
            (
                grid.weights[i] * plasma_squared,
                (plasma_squared / 3.0 + gap_squared).sqrt(),
                screening,
            )
        })
        .collect();
    for (i, &(plasma, omega, screening)) in local.iter().enumerate() {
        for (j, &(plasma2, omega2, screening2)) in local.iter().enumerate() {
            if plasma == 0.0 || plasma2 == 0.0 {
                continue;
            }
            let r = (grid.points[i] - grid.points[j]).norm();
            let inverse_length = 0.5 / (screening.recip() + screening2.recip()).sqrt();
            let k = inverse_length * r;
            let spatial = if k < 0.1 {
                // Independently integrate the Gaussian moment entering B in
                // the small-distance branch rather than using its series.
                let average = nodes
                    .iter()
                    .zip(&weights)
                    .map(|(x, w)| {
                        let t = 0.5 * (1.0 + x);
                        0.5 * w * t.powi(2) * (-k.powi(2) * t.powi(2)).exp()
                    })
                    .sum::<f64>();
                let a = 2.0 * (-k.powi(2)).exp() / pi.sqrt();
                let b = 4.0 * average / pi.sqrt();
                inverse_length.powi(6) * (4.0 / 3.0 * a * b - b * b)
            } else {
                let a = (2.0 * k / pi.sqrt()) * (-k * k).exp();
                let b = libm::erf(k) - a;
                ((4.0 / 3.0) * k * k * a * b - b * b) / r.powi(6)
            };
            energy += (3.0 / (64.0 * pi * pi)) * plasma * plasma2 * spatial
                / (omega * omega2 * (omega + omega2));
        }
    }
    energy
}

#[test]
fn vv09_matches_independent_quadrature_at_coincidence_and_long_range() {
    let grid = |positions: Vec<Vec3>, weights: Vec<f64>| MolecularGrid {
        points: positions,
        weights,
        raw_weights: Vec::new(),
        atom_of: Vec::new(),
    };
    let positions = vec![
        Vec3::zero(),
        Vec3::new(1e-7, 0.0, 0.0),
        Vec3::new(3.2, -2.7, 1.1),
    ];
    let probe = grid(positions, vec![0.7, 0.9, 1.1]);
    let rho = [0.8, 0.31, 0.54];
    let sigma = [0.028, 0.04, 0.07];
    let spin = [0.7, -0.3, 0.98];
    let result = vv09::energy_on_grid(&probe, &rho, &sigma, &spin).unwrap();
    close(result, vv09_oracle(&probe, &rho, &sigma, &spin), 2e-14);
    close(
        result,
        vv09::energy_on_grid(&probe, &rho, &sigma, &spin.map(|z| -z)).unwrap(),
        2e-15,
    );
    let point = grid(vec![Vec3::zero()], vec![1.0]);
    let uniform_rho = 0.64;
    let single = vv09::energy_on_grid(&point, &[uniform_rho], &[0.0], &[0.0]).unwrap();
    let pi = std::f64::consts::PI;
    let omega = (4.0 * pi * uniform_rho / 3.0).sqrt();
    let kappa = 4.0 * (3.0 * pi * pi * uniform_rho).cbrt() / pi;
    let diagonal = 0.75 * uniform_rho.powi(2) * kappa.powi(3) / (288.0 * pi * 2.0 * omega.powi(3));
    close(single, diagonal, 2e-15);
    let r = 25.0_f64;
    let distant = grid(vec![Vec3::zero(), Vec3::new(r, 0.0, 0.0)], vec![1.0; 2]);
    let whole = vv09::energy_on_grid(&distant, &[uniform_rho; 2], &[0.0; 2], &[0.0; 2]).unwrap();
    let c6 = -r.powi(6) * (whole - 2.0 * single);
    close(c6, 0.75 * uniform_rho.powi(2) / omega.powi(3), 1e-8);
    println!("VV09 asymmetric quadrature: E={result:.12e} Eh, coincidence={single:.12e} Eh, tail C6={c6:.12e} Eh Bohr^6");
}

#[test]
fn vv09_molecular_energy_keeps_the_pw92_reference_and_nonlocal_term_separate() {
    let (mol, system) = ammonia();
    let basis_set = data::load("sto-3g").unwrap();
    let basis = MolBasis::build(&mol, &basis_set).unwrap();
    let opts = GridOptions {
        level: 0,
        ..GridOptions::default()
    };
    let grid = MolecularGrid::build(&mol, &opts).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("spw92".into())).unwrap();
    let scf = run_scf(
        &mol,
        &basis,
        &ScfOptions::default(),
        &functional,
        Some(&grid),
    )
    .unwrap();
    let control = |xc| {
        Controls::from_value(
            &Value::object()
                .with("xc", xc)
                .with("basis", "sto-3g")
                .with("grid_level", 0usize)
                .with("dispersion", "vv09"),
        )
        .unwrap()
    };
    let report = run("energy", &system, &control("spw92")).unwrap();
    let basis_eval = eval_aos_full(&basis, &grid.points, true, false);
    let gradient = basis_eval.grad.as_ref().unwrap();
    let p = scf.p_total();
    let rho = density_field(&basis, &grid, &p);
    let sigma: Vec<_> = (0..grid.len())
        .map(|point| {
            gradient
                .iter()
                .map(|derivative| {
                    let mut g = 0.0;
                    for i in 0..p.rows {
                        for j in 0..p.cols {
                            g += 2.0
                                * p[(i, j)]
                                * basis_eval.phi[(point, i)]
                                * derivative[(point, j)];
                        }
                    }
                    g * g
                })
                .sum()
        })
        .collect();
    let oracle = vv09_oracle(&grid, &rho, &sigma, &vec![0.0; grid.len()]);
    let actual = report.get("dispersion_energy").unwrap().as_f64().unwrap();
    close(actual, oracle, 2e-11);
    close(
        report.get("energy").unwrap().as_f64().unwrap(),
        scf.energy + oracle,
        1e-10,
    );
    assert!(run("energy", &system, &control("pbe"))
        .unwrap_err()
        .to_string()
        .contains("requires unscaled original PW92"));
    println!("SPW92/STO-3G NH3: points={} iterations={} reference={:.12e} VV09={actual:.12e}/{oracle:.12e} Eh",grid.len(),scf.iterations,scf.energy);
}
