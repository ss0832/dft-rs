// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Spin-complete active-space state interaction in common DFT/HF orbitals.
//! The physical Coulomb Hamiltonian, nuclear one-electron SOC and Zeeman
//! operators are projected into the same determinant space. The functional
//! generates orbitals only: these excitation energies are CASCI, not TDDFT.
use super::complex::{invalid, rows};
use crate::dfpt::orbitals::{adjoint, generalized_orbitals, matmul};
use crate::driver::method::run_molecular_reference;
use crate::driver::molecular::{reference_report, setup};
use crate::driver::{Controls, System, Value};
use crate::integrals::{dipole_matrices, CoulombKernel};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::BlochMatrix;
use crate::properties::nmr::{angular_momentum_operator, paramagnetic_operator};
use crate::response::complex::Complex;
use crate::scf::fock::{EriMode, JkEngine};
use crate::{DftError, Result};
use std::collections::BTreeMap;
use std::f64::consts::PI;

const SPIN_G_FACTOR: f64 = 2.00231930436256;

#[path = "casci_london.rs"]
mod london;

fn z(m: &BlochMatrix, i: usize, j: usize) -> Complex {
    Complex::new(m.re[(i, j)], m.im[(i, j)])
}
fn put(m: &mut BlochMatrix, i: usize, j: usize, v: Complex) {
    m.re[(i, j)] = v.re;
    m.im[(i, j)] = v.im;
}
fn add(m: &mut BlochMatrix, i: usize, j: usize, v: Complex) {
    m.re[(i, j)] += v.re;
    m.im[(i, j)] += v.im;
}

fn annihilate(bits: u64, orbital: usize) -> Option<(u64, f64)> {
    let mask = 1_u64 << orbital;
    (bits & mask != 0).then(|| {
        (
            bits ^ mask,
            if (bits & (mask - 1)).count_ones() % 2 == 0 {
                1.0
            } else {
                -1.0
            },
        )
    })
}
fn create(bits: u64, orbital: usize) -> Option<(u64, f64)> {
    let mask = 1_u64 << orbital;
    (bits & mask == 0).then(|| {
        (
            bits | mask,
            if (bits & (mask - 1)).count_ones() % 2 == 0 {
                1.0
            } else {
                -1.0
            },
        )
    })
}
fn determinants(orbitals: usize, electrons: usize, limit: usize) -> Result<Vec<u64>> {
    if orbitals == 0 || orbitals > 32 || electrons > orbitals || limit == 0 || limit > 2048 {
        return Err(invalid(
            "active spin-orbitals require 1..32, a valid electron count and max_states 1..2048",
        ));
    }
    let mut count = 1_u128;
    for i in 0..electrons.min(orbitals - electrons) {
        count = count * (orbitals - i) as u128 / (i + 1) as u128;
    }
    if count > limit as u128 {
        return Err(DftError::Unsupported{feature:format!("spin-complete CASCI with {count} determinants"),missing:format!("explicitly choose a smaller active space or raise max_states (currently {limit}); no determinants are silently discarded")});
    }
    fn visit(start: usize, left: usize, n: usize, bits: u64, out: &mut Vec<u64>) {
        if left == 0 {
            out.push(bits);
            return;
        }
        for p in start..=n - left {
            visit(p + 1, left - 1, n, bits | (1_u64 << p), out);
        }
    }
    let mut out = Vec::with_capacity(count as usize);
    visit(0, electrons, orbitals, 0, &mut out);
    out.sort_unstable();
    Ok(out)
}

fn one_body(operator: &BlochMatrix, det: &[u64]) -> BlochMatrix {
    let index: BTreeMap<u64, usize> = det.iter().enumerate().map(|(i, d)| (*d, i)).collect();
    let mut out = BlochMatrix::zeros(det.len(), det.len());
    for (column, &bits) in det.iter().enumerate() {
        for q in 0..operator.rows() {
            if let Some((b, s)) = annihilate(bits, q) {
                for p in 0..operator.rows() {
                    if let Some((b, t)) = create(b, p) {
                        if let Some(&row) = index.get(&b) {
                            add(&mut out, row, column, z(operator, p, q) * (s * t));
                        }
                    }
                }
            }
        }
    }
    out
}

fn coulomb(eri: &[f64], n: usize, det: &[u64]) -> BlochMatrix {
    let index: BTreeMap<u64, usize> = det.iter().enumerate().map(|(i, d)| (*d, i)).collect();
    let mut out = BlochMatrix::zeros(det.len(), det.len());
    for (column, &bits) in det.iter().enumerate() {
        for q in 0..2 * n {
            if let Some((b, sq)) = annihilate(bits, q) {
                for s in 0..2 * n {
                    if let Some((b, ss)) = annihilate(b, s) {
                        for r in (s / n * n)..(s / n * n + n) {
                            if let Some((b, sr)) = create(b, r) {
                                for p in (q / n * n)..(q / n * n + n) {
                                    if let Some((b, sp)) = create(b, p) {
                                        if let Some(&row) = index.get(&b) {
                                            out.re[(row, column)] += 0.5
                                                * sq
                                                * ss
                                                * sr
                                                * sp
                                                * eri[(((p % n) * n + q % n) * n + r % n) * n
                                                    + s % n];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

fn transformed(operator: &BlochMatrix, vectors: &BlochMatrix) -> Result<BlochMatrix> {
    matmul(&adjoint(vectors)?, &matmul(operator, vectors)?)
}

#[derive(Clone, Copy)]
struct SpectrumConfig<'a> {
    temperature: f64,
    points: &'a [f64],
    width: f64,
}

fn spectra(
    h0: &BlochMatrix,
    magnetic: &[BlochMatrix; 3],
    electric: &[BlochMatrix; 3],
    spin_magnetic: &[BlochMatrix; 3],
    field: [f64; 3],
    config: &SpectrumConfig<'_>,
    direction: [f64; 3],
) -> Result<Value> {
    let SpectrumConfig {
        temperature,
        points,
        width,
    } = *config;
    let mut h = h0.clone();
    let origin = h.re[(0, 0)];
    for i in 0..h.rows() {
        h.re[(i, i)] -= origin;
    }
    for a in 0..3 {
        h.add_scaled(&magnetic[a], -field[a]);
    }
    let identity = BlochMatrix {
        re: Matrix::identity(h.rows()),
        im: Matrix::zeros(h.rows(), h.rows()),
    };
    let bands = generalized_orbitals(&h, &identity, 1e-12)?;
    if bands.eigen_residual > 1e-9 || bands.metric_defect > 1e-9 {
        return Err(DftError::ResponseNotConverged {
            iterations: 0,
            residual: bands.eigen_residual.max(bands.metric_defect),
        });
    }
    let kt = temperature * crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
    let mut pop: Vec<f64> = bands
        .energies
        .iter()
        .map(|e| (-(e - bands.energies[0]) / kt).exp())
        .collect();
    let partition: f64 = pop.iter().sum();
    for p in &mut pop {
        *p /= partition;
    }
    let norm = direction.iter().fold(0.0_f64, |n, v| n.hypot(*v));
    let d = direction.map(|v| v / norm);
    let helper = if d[2].abs() < 0.9 {
        [0., 0., 1.]
    } else {
        [1., 0., 0.]
    };
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let mut ex = cross(helper, d);
    let norm = ex.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut ex {
        *x /= norm;
    }
    let ey = cross(d, ex);
    let mu: Vec<BlochMatrix> = electric
        .iter()
        .map(|m| transformed(m, &bands.coefficients))
        .collect::<Result<_>>()?;
    let spin = spin_magnetic
        .iter()
        .map(|m| {
            let mut s = m.clone();
            for value in s.re.as_mut_slice().iter_mut().chain(s.im.as_mut_slice()) {
                *value *= -2. / SPIN_G_FACTOR;
            }
            s
        })
        .collect::<Vec<_>>();
    let mut spin_squared = BlochMatrix::zeros(h.rows(), h.rows());
    for s in &spin {
        spin_squared.add_scaled(&matmul(s, s)?, 1.);
    }
    let state_spin = spin
        .iter()
        .map(|s| transformed(s, &bands.coefficients))
        .collect::<Result<Vec<_>>>()?;
    let state_spin_squared = transformed(&spin_squared, &bands.coefficients)?;
    let mut left = vec![0.; points.len()];
    let mut right = left.clone();
    let mut lines = Vec::new();
    let mut strength_sum = 0.;
    for i in 0..h.rows() {
        for j in i + 1..h.rows() {
            let energy = bands.energies[j] - bands.energies[i];
            if energy <= 0. {
                continue;
            }
            let mut x = Complex::ZERO;
            let mut y = Complex::ZERO;
            let mut norm2 = 0.;
            for a in 0..3 {
                let v = z(&mu[a], j, i);
                x += v * ex[a];
                y += v * ey[a];
                norm2 += v.abs().powi(2);
            }
            let iy = Complex::new(-y.im, y.re);
            let plus = (x + iy).abs().powi(2) * 0.5;
            let minus = (x - iy).abs().powi(2) * 0.5;
            let population = pop[i] - pop[j];
            let oscillator = (2. / 3.) * energy * norm2 * population;
            strength_sum += oscillator;
            for (k, &w) in points.iter().enumerate() {
                let radius = (w - energy).hypot(width);
                let lorentz = (width / radius) / radius / PI;
                let factor =
                    4. * PI * PI * crate::constants::FINE_STRUCTURE * energy * population * lorentz;
                left[k] += factor * plus;
                right[k] += factor * minus;
            }
            lines.push(
                Value::object()
                    .with("initial_state", i)
                    .with("final_state", j)
                    .with("energy", energy)
                    .with("population_difference", population)
                    .with(
                        "dipole_real",
                        mu.iter().map(|m| m.re[(j, i)]).collect::<Vec<_>>(),
                    )
                    .with(
                        "dipole_imag",
                        mu.iter().map(|m| m.im[(j, i)]).collect::<Vec<_>>(),
                    )
                    .with("left_strength", plus)
                    .with("right_strength", minus)
                    .with("oscillator_strength", oscillator),
            );
        }
    }
    let average: Vec<f64> = left
        .iter()
        .zip(&right)
        .map(|(l, r)| 0.5 * (l + r))
        .collect();
    if left
        .iter()
        .chain(&right)
        .chain(&average)
        .any(|x| !x.is_finite())
        || !strength_sum.is_finite()
    {
        return Err(invalid(
            "spectral strength or linewidth is not representable",
        ));
    }
    let difference: Vec<f64> = left.iter().zip(&right).map(|(l, r)| l - r).collect();
    let ratio: Vec<Value> = difference
        .iter()
        .zip(&average)
        .map(|(d, a)| {
            if *a > 0. {
                Value::Float(d / a)
            } else {
                Value::Null
            }
        })
        .collect();
    Ok(Value::object()
        .with("state_energies_relative", bands.energies)
        .with("energy_origin", origin)
        .with("state_populations", pop)
        .with("state_spin_expectation", (0..h.rows()).map(|i|state_spin.iter().map(|s|s.re[(i,i)]).collect::<Vec<_>>()).collect::<Vec<_>>())
        .with("state_spin_squared", (0..h.rows()).map(|i|state_spin_squared.re[(i,i)]).collect::<Vec<_>>())
        .with("spin_state_convention", "Spin-complete determinant space. Returned <Sx,Sy,Sz> and <S^2> use hbar=1; SOC eigenstates need not have a pure total spin or integer multiplicity. State tables describe the representative orientation.")
        .with("eigen_residual", bands.eigen_residual)
        .with("metric_defect", bands.metric_defect)
        .with("transitions", lines)
        .with("absorption_left", left)
        .with("absorption_right", right)
        .with("absorption_average", average)
        .with("mcd_cross_section", difference)
        .with("mcd_normalized", ratio)
        .with("oscillator_strength_sum", strength_sum))
}

pub(super) fn run(system: &System, controls: &Controls) -> Result<Value> {
    let temperature = controls.f64("temperature", 0.)?;
    let width = controls.f64("spectrum_width", 0.001)?;
    let field = controls
        .opt_vec3("bfield")?
        .ok_or_else(|| invalid("molecular-casci-soc requires bfield, including zero"))?
        .to_array();
    let points = controls
        .opt_f64_list("frequencies")?
        .ok_or_else(|| invalid("positive photon energies in frequencies are required"))?;
    let direction = controls
        .opt_vec3("mcd_direction")?
        .map_or([0., 0., 1.], |v| v.to_array());
    let core = controls.usize("mcd_frozen_core", 0)?;
    let active = controls.opt_usize("mcd_active_orbitals")?;
    let limit = controls.usize("max_states", 512)?;
    let soc_scale = controls.f64("soc_scale", 1.)?;
    let origin_control = controls.opt_vec3("gauge_origin")?;
    let magnetic_basis = controls.str("mcd_magnetic_basis", "common-origin")?;
    if !matches!(
        magnetic_basis.as_str(),
        "common-origin" | "london-linear-frozen-soc"
    ) {
        return Err(invalid(
            "mcd_magnetic_basis is common-origin or london-linear-frozen-soc",
        ));
    }
    let orientation = controls.str("mcd_orientation", "fixed")?;
    let orientation_order = controls.usize("mcd_orientation_order", 6)?;
    if !matches!(orientation.as_str(), "fixed" | "isotropic")
        || !(1..=24).contains(&orientation_order)
    {
        return Err(invalid(
            "mcd_orientation is fixed or isotropic and mcd_orientation_order is 1..24",
        ));
    }
    let kt = temperature * crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
    if !(kt > 0. && kt.is_finite() && (1. / kt).is_finite() && width > 0. && width.is_finite())
        || points.is_empty()
        || points.iter().any(|w| !w.is_finite() || *w <= 0.)
        || points.windows(2).any(|p| p[1] <= p[0])
        || direction.iter().any(|v| !v.is_finite())
        || direction.iter().fold(0.0_f64, |n, v| n.hypot(*v)) == 0.
        || !direction
            .iter()
            .fold(0.0_f64, |n, v| n.hypot(*v))
            .is_finite()
        || field.iter().any(|v| !v.is_finite())
        || !soc_scale.is_finite()
    {
        return Err(invalid("finite positive temperature/width, ordered positive photon energies, finite SOC scale and nonzero light direction required"));
    }
    let setup = setup(system, controls)?;
    controls.finish("mcd molecular-casci-soc")?;
    if !setup.basis.ecps.is_empty()
        || setup.options.pcm.is_some()
        || setup.options.vv10.is_some()
        || setup.options.skala.is_some()
    {
        return Err(DftError::Unsupported {
            feature: "molecular CASCI/SOC with ECP, continuum or auxiliary nonlocal model".into(),
            missing: "a consistent active-space Hamiltonian and SOC for that extra model".into(),
        });
    }
    let reference = run_molecular_reference(&setup)?;
    let scf = &reference.scf;
    if !scf.converged || scf.electronic_temperature != 0. || !scf.has_integer_occupations() {
        return Err(invalid("orbital generation needs a converged integer-occupation reference; thermal populations are applied to many-electron states"));
    }
    let orbitals = &scf.mo_coeff[0];
    let n = active.unwrap_or(orbitals.cols.saturating_sub(core));
    let ne = scf.n_alpha + scf.n_beta;
    if core > ne / 2 || n == 0 || core + n > orbitals.cols || n > 16 || ne - 2 * core > 2 * n {
        return Err(invalid("frozen core and contiguous active spatial orbital count do not accommodate the molecular electron count"));
    }
    let det = determinants(2 * n, ne - 2 * core, limit)?;
    let m = core + n;
    let nao = orbitals.rows;
    let coefficients = Matrix::from_row_major(
        nao,
        m,
        (0..nao)
            .flat_map(|i| (0..m).map(move |j| orbitals[(i, j)]))
            .collect(),
    );
    let h = coefficients.transpose_matmul(&scf.h_core.matmul(&coefficients));
    let engine = JkEngine::new(
        &setup.basis,
        EriMode::InCore {
            limit_mb: setup.options.incore_mb,
        },
        CoulombKernel::Coulomb,
    )?;
    let zero = Matrix::zeros(nao, nao);
    let mut eri = vec![0.; m * m * m * m];
    for r in 0..m {
        for s in 0..=r {
            let density = Matrix::from_row_major(
                nao,
                nao,
                (0..nao)
                    .flat_map(|i| {
                        let c = &coefficients;
                        (0..nao).map(move |j| 0.5 * (c[(i, r)] * c[(j, s)] + c[(i, s)] * c[(j, r)]))
                    })
                    .collect(),
            );
            let (j, _) = engine.build(&setup.basis, &density, [&zero, &zero], false)?;
            let mo = coefficients.transpose_matmul(&j.matmul(&coefficients));
            for p in 0..m {
                for q in 0..m {
                    eri[((p * m + q) * m + r) * m + s] = mo[(p, q)];
                    eri[((p * m + q) * m + s) * m + r] = mo[(p, q)];
                }
            }
        }
    }
    let v = |p: usize, q: usize, r: usize, s: usize| eri[((p * m + q) * m + r) * m + s];
    let mut effective = BlochMatrix::zeros(2 * n, 2 * n);
    let mut active_eri = vec![0.; n * n * n * n];
    let mut ecore = 0.;
    for i in 0..core {
        ecore += 2. * h[(i, i)];
        for j in 0..core {
            ecore += 2. * v(i, i, j, j) - v(i, j, j, i);
        }
    }
    for p in 0..n {
        for q in 0..n {
            let mut hpq = h[(core + p, core + q)];
            for i in 0..core {
                hpq += 2. * v(core + p, core + q, i, i) - v(core + p, i, i, core + q);
            }
            effective.re[(p, q)] = hpq;
            effective.re[(n + p, n + q)] = hpq;
            for r in 0..n {
                for s in 0..n {
                    active_eri[((p * n + q) * n + r) * n + s] =
                        v(core + p, core + q, core + r, core + s);
                }
            }
        }
    }
    let mut charge = 0.;
    let mut center = Vec3::zero();
    for atom in &setup.molecule.atoms {
        charge += atom.z as f64;
        center += atom.position * (atom.z as f64);
    }
    let origin = origin_control.unwrap_or_else(|| {
        if charge > 0. {
            center * (1. / charge)
        } else {
            Vec3::zero()
        }
    });
    let project = |a: &Matrix| {
        let all = coefficients.transpose_matmul(&a.matmul(&coefficients));
        Matrix::from_row_major(
            n,
            n,
            (0..n)
                .flat_map(|i| {
                    let a = &all;
                    (0..n).map(move |j| a[(core + i, core + j)])
                })
                .collect(),
        )
    };
    let position = dipole_matrices(&setup.basis, origin);
    let angular = angular_momentum_operator(&setup.basis, origin);
    let mut orbital: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(nao, nao));
    let prefactor = 0.5 * crate::constants::FINE_STRUCTURE.powi(2) * soc_scale;
    for atom in &setup.molecule.atoms {
        let op = paramagnetic_operator(&setup.basis, atom.position);
        for a in 0..3 {
            for (x, y) in orbital[a].as_mut_slice().iter_mut().zip(op[a].as_slice()) {
                *x += prefactor * (atom.z as f64) * y;
            }
        }
    }
    let orbital = orbital.map(|a| project(&a));
    let angular = angular.map(|a| project(&a));
    let position = position.map(|a| project(&a));
    let mut soc = BlochMatrix::zeros(2 * n, 2 * n);
    let mut magnetic: [BlochMatrix; 3] = std::array::from_fn(|_| soc.clone());
    let mut electric = magnetic.clone();
    // Atomic magnetic dipole unit is 2*mu_B. The free-electron g value is
    // fixed explicitly; no measured molecular g factor is supplied or fitted.
    let ge = SPIN_G_FACTOR;
    for p in 0..n {
        for q in 0..n {
            soc.im[(p, q)] = -0.5 * orbital[2][(p, q)];
            soc.im[(n + p, n + q)] = 0.5 * orbital[2][(p, q)];
            let mixed = Complex::new(-0.5 * orbital[1][(p, q)], -0.5 * orbital[0][(p, q)]);
            put(&mut soc, p, n + q, mixed);
            put(&mut soc, n + q, p, mixed.conj());
            for a in 0..3 {
                for spin in 0..2 {
                    electric[a].re[(spin * n + p, spin * n + q)] = -position[a][(p, q)];
                    magnetic[a].im[(spin * n + p, spin * n + q)] = 0.5 * angular[a][(p, q)];
                }
            }
            if p == q {
                magnetic[0].re[(p, n + p)] = -ge / 4.;
                magnetic[0].re[(n + p, p)] = -ge / 4.;
                magnetic[1].im[(p, n + p)] = ge / 4.;
                magnetic[1].im[(n + p, p)] = -ge / 4.;
                magnetic[2].re[(p, p)] = -ge / 4.;
                magnetic[2].re[(n + p, n + p)] = ge / 4.;
            }
        }
    }
    effective.add_scaled(&soc, 1.);
    let mut h0 = one_body(&effective, &det);
    h0.add_scaled(&coulomb(&active_eri, n, &det), 1.);
    let mut magnetic = magnetic.map(|a| one_body(&a, &det));
    let electric = electric.map(|a| one_body(&a, &det));
    let mut spin: [BlochMatrix; 3] = std::array::from_fn(|_| BlochMatrix::zeros(2 * n, 2 * n));
    for p in 0..n {
        spin[0].re[(p, n + p)] = -ge / 4.;
        spin[0].re[(n + p, p)] = -ge / 4.;
        spin[1].im[(p, n + p)] = ge / 4.;
        spin[1].im[(n + p, p)] = -ge / 4.;
        spin[2].re[(p, p)] = -ge / 4.;
        spin[2].re[(n + p, n + p)] = ge / 4.;
    }
    let spin = spin.map(|a| one_body(&a, &det));
    let london = if magnetic_basis == "london-linear-frozen-soc" {
        let estimate = (nao as u128).pow(4) * 64 + (m as u128).pow(4) * 64;
        if estimate > (setup.options.incore_mb as u128) * 1024 * 1024 {
            return Err(DftError::Unsupported {
                feature: "London active-space magnetic response workspace".into(),
                missing: "a larger configured integral workspace or smaller basis; the dense four-index workspace is not silently allocated beyond the configured allowance".into(),
            });
        }
        let result = london::build(
            &setup.molecule,
            &setup.basis,
            &coefficients,
            core,
            &h,
            &eri,
            &det,
            origin.to_array(),
            &spin,
        )?;
        magnetic = result.magnetic.clone();
        Some(result)
    } else {
        None
    };
    let magnetic_hermiticity = magnetic.each_ref().map(BlochMatrix::hermiticity_error);
    let electric_hermiticity = electric.each_ref().map(BlochMatrix::hermiticity_error);
    if magnetic_hermiticity
        .iter()
        .chain(&electric_hermiticity)
        .any(|x| *x > 1e-10)
        || magnetic.iter().chain(&electric).any(|m| {
            m.re.as_slice()
                .iter()
                .chain(m.im.as_slice())
                .any(|x| !x.is_finite())
        })
    {
        return Err(invalid("generated magnetic/electric operators must be finite and Hermitian; no projection is applied"));
    }
    let spectrum_config = SpectrumConfig {
        temperature,
        points: &points,
        width,
    };
    let evaluate = |b: [f64; 3], axis| -> Result<Value> {
        let mut optical = electric.clone();
        if let Some(london) = &london {
            for a in 0..3 {
                for k in 0..3 {
                    optical[k].add_scaled(&london.electric_derivative[a][k], b[a]);
                }
            }
        }
        spectra(&h0, &magnetic, &optical, &spin, b, &spectrum_config, axis)
    };
    let mut result = evaluate(field, direction)?;
    if orientation == "isotropic" {
        let norm = direction.iter().fold(0.0_f64, |n, v| n.hypot(*v));
        let d = direction.map(|v| v / norm);
        let signed = field.iter().zip(d).map(|(b, k)| b * k).sum::<f64>();
        let transverse = field
            .iter()
            .zip(d)
            .map(|(b, k)| (b - signed * k).powi(2))
            .sum::<f64>()
            .sqrt();
        if transverse > 1e-12 * (1.0 + signed.abs()) {
            return Err(invalid(
                "isotropic MCD requires field parallel or antiparallel to the light direction",
            ));
        }
        let (nodes, weights) = crate::grid::gauss_legendre(orientation_order);
        let mut left = vec![0.; points.len()];
        let mut right = left.clone();
        let mut diagnostics = Vec::new();
        let mut sumrule = 0.;
        let nphi = 2 * orientation_order;
        for (&u, &weight) in nodes.iter().zip(&weights) {
            for k in 0..nphi {
                let phi = 2. * PI * (k as f64) / (nphi as f64);
                let radial = (1. - u * u).sqrt();
                let axis = [radial * phi.cos(), radial * phi.sin(), u];
                let b = axis.map(|v| v * signed);
                let report = evaluate(b, axis)?;
                let weight = weight / (2. * nphi as f64);
                for (key, target) in [
                    ("absorption_left", &mut left),
                    ("absorption_right", &mut right),
                ] {
                    let values = report
                        .get(key)
                        .and_then(Value::numeric_array)
                        .ok_or_else(|| invalid("missing orientation spectrum"))?
                        .1;
                    for (x, y) in target.iter_mut().zip(values) {
                        *x += weight * y;
                    }
                }
                sumrule += weight
                    * report
                        .get("oscillator_strength_sum")
                        .and_then(Value::as_f64)
                        .ok_or_else(|| invalid("missing orientation oscillator sum"))?;
                diagnostics.push(
                    Value::object()
                        .with("direction", axis)
                        .with("weight", weight)
                        .with(
                            "eigen_residual",
                            report.get("eigen_residual").cloned().unwrap_or(Value::Null),
                        )
                        .with(
                            "state_populations",
                            report
                                .get("state_populations")
                                .cloned()
                                .unwrap_or(Value::Null),
                        )
                        .with(
                            "state_energies_relative",
                            report
                                .get("state_energies_relative")
                                .cloned()
                                .unwrap_or(Value::Null),
                        ),
                );
            }
        }
        let average: Vec<f64> = left
            .iter()
            .zip(&right)
            .map(|(l, r)| 0.5 * (l + r))
            .collect();
        let difference: Vec<f64> = left.iter().zip(&right).map(|(l, r)| l - r).collect();
        let ratio: Vec<Value> = difference
            .iter()
            .zip(&average)
            .map(|(d, a)| {
                if *a > 0. {
                    Value::Float(d / a)
                } else {
                    Value::Null
                }
            })
            .collect();
        result.set("absorption_left", left);
        result.set("absorption_right", right);
        result.set("absorption_average", average);
        result.set("mcd_cross_section", difference);
        result.set("mcd_normalized", ratio);
        result.set("oscillator_strength_sum", sumrule);
        result.set("orientation_diagnostics", diagnostics);
        result.set("state_table_convention","State energies, populations and transition table are the representative input orientation only. Spectra and oscillator sum are the weighted isotropic average; per-orientation populations and residuals are retained in orientation_diagnostics. No averaged state or dipole phase is invented.");
    }
    let data=reference_report(&setup,&reference,"spin-complete CASCI in common alpha-channel DFT/HF orbitals with one-electron nuclear SOC and linear orbital/spin Zeeman terms; functional generates orbitals only, not TDDFT excitation energies");
    if let Value::Object(entries) = data {
        for (key, value) in entries {
            result.set(&key, value);
        }
    }
    result.set("mcd_model", "molecular-casci-soc");
    result.set("mcd_magnetic_basis", magnetic_basis.clone());
    result.set("magnetic_field_order", 1_usize);
    result.set("soc_field_treatment", "zero-field SOC frozen in the orthonormal active frame; moving SOC and two-electron SOC are absent");
    if let Some(london) = &london {
        result.set(
            "electric_magnetic_derivatives_real",
            london
                .electric_derivative
                .iter()
                .map(|axis| axis.iter().map(|m| rows(&m.re)).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        );
        result.set(
            "electric_magnetic_derivatives_imag",
            london
                .electric_derivative
                .iter()
                .map(|axis| axis.iter().map(|m| rows(&m.im)).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        );
        result.set("gauge_convention", "London orbital phases; analytic first-order overlap connection on every Hamiltonian and optical index. Gauge origin cancels algebraically. Linear-field model with frozen zero-field SOC, not finite-field relativistic self-consistency.");
    }
    result.set("geometry_used", true);
    result.set("temperature", temperature);
    result.set("bfield", field);
    result.set("frequencies", points);
    result.set("spectrum_width", width);
    result.set("mcd_direction", direction);
    result.set("mcd_orientation", orientation);
    result.set("mcd_orientation_order", orientation_order);
    result.set("gauge_origin", origin.to_array());
    result.set("soc_scale", soc_scale);
    result.set("spin_g_factor", ge);
    result.set("active_spatial_orbitals", n);
    result.set("active_electrons", ne - 2 * core);
    result.set("frozen_core_orbitals", core);
    result.set("determinant_count", det.len());
    result.set("frozen_core_electronic_energy", ecore);
    result.set("orbital_coefficients", rows(&coefficients));
    result.set("orbital_overlap", rows(&scf.overlap));
    result.set("hamiltonian_real", rows(&h0.re));
    result.set("hamiltonian_imag", rows(&h0.im));
    result.set(
        "magnetic_operators_real",
        magnetic.iter().map(|a| rows(&a.re)).collect::<Vec<_>>(),
    );
    result.set(
        "magnetic_operators_imag",
        magnetic.iter().map(|a| rows(&a.im)).collect::<Vec<_>>(),
    );
    result.set(
        "electric_operators_real",
        electric.iter().map(|a| rows(&a.re)).collect::<Vec<_>>(),
    );
    result.set(
        "electric_operators_imag",
        electric.iter().map(|a| rows(&a.im)).collect::<Vec<_>>(),
    );
    result.set("hermiticity_defect", h0.hermiticity_error());
    result.set("magnetic_hermiticity_defect", magnetic_hermiticity);
    result.set("electric_hermiticity_defect", electric_hermiticity);
    result.set("spectrum_convention","Fixed molecule and light direction. e_left=(e1+i e2)/sqrt(2), e_right=(e1-i e2)/sqrt(2), (e1,e2,k) right-handed; sigma_left-right is reported. Positive-frequency Lorentz lines with specified Hartree HWHM and analytic 4*pi^2*alpha*transition_energy*(p_i-p_j)*|e.d_ji|^2 area; no fitted scale or energy shift. Temperature populations apply to the complete retained many-electron manifold, not orbital Fermi occupations.");
    result.set("warnings", if london.is_some() {
        vec!["Active-space CASCI, not TDDFT. London first-order Coulomb, metric, orbital-Zeeman and optical response with frozen zero-field SOC. Moving SOC, two-electron SOC, quadratic field terms, host, solvent and vibronic models are absent. Field-strength, basis, active-space and orientation convergence remain necessary."]
    } else {
        vec!["Active-space CASCI, not TDDFT; functional only generates common orbitals. No two-electron SOC, London-orbital magnetic gauge completion, quadratic diamagnetic term, host, solvent or vibronic model is included. Fixed-origin linear-Zeeman results require active-space/basis/origin/field and orientation-quadrature convergence before experimental interpretation."]
    });
    let mut units = Value::object();
    units.set("energy", "hartree, orbital-generating reference energy");
    units.set(
        "orbital_coefficients",
        "dimensionless, common alpha-channel orbitals including frozen core",
    );
    units.set("orbital_overlap", "dimensionless");
    // Dotted paths let every surface convert heterogeneous nested entries
    // independently. A prose unit label on their parent would hide dimensions.
    for (key, label) in [
        ("transitions.energy", "hartree"),
        ("transitions.dipole_real", "e*bohr"),
        ("transitions.dipole_imag", "e*bohr"),
        ("transitions.population_difference", "dimensionless"),
        ("transitions.left_strength", "e^2*bohr^2"),
        ("transitions.right_strength", "e^2*bohr^2"),
        ("transitions.oscillator_strength", "dimensionless"),
        ("orientation_diagnostics.direction", "dimensionless"),
        ("orientation_diagnostics.weight", "dimensionless"),
        ("orientation_diagnostics.eigen_residual", "hartree"),
        ("orientation_diagnostics.state_populations", "dimensionless"),
        ("orientation_diagnostics.state_energies_relative", "hartree"),
    ] {
        units.set(key, label);
    }
    for (key, label) in [
        ("temperature", "K"),
        ("bfield", "atomic units of magnetic flux density"),
        ("frequencies", "hartree"),
        ("spectrum_width", "hartree HWHM"),
        ("state_energies_relative", "hartree"),
        ("energy_origin", "hartree"),
        ("state_populations", "dimensionless"),
        ("state_spin_expectation", "dimensionless, hbar=1"),
        ("state_spin_squared", "dimensionless, hbar^2=1"),
        ("absorption_left", "bohr^2 per molecule"),
        ("absorption_right", "bohr^2 per molecule"),
        ("absorption_average", "bohr^2 per molecule"),
        ("mcd_cross_section", "bohr^2 per molecule, left minus right"),
        ("mcd_normalized", "dimensionless"),
        ("eigen_residual", "hartree"),
        ("metric_defect", "dimensionless"),
        ("gauge_origin", "bohr"),
        ("hamiltonian_real", "hartree"),
        ("hamiltonian_imag", "hartree"),
        (
            "magnetic_operators_real",
            "atomic units of magnetic dipole moment",
        ),
        (
            "magnetic_operators_imag",
            "atomic units of magnetic dipole moment",
        ),
        ("electric_operators_real", "e*bohr"),
        ("electric_operators_imag", "e*bohr"),
        (
            "electric_magnetic_derivatives_real",
            "e*bohr per atomic unit of magnetic flux density",
        ),
        (
            "electric_magnetic_derivatives_imag",
            "e*bohr per atomic unit of magnetic flux density",
        ),
        ("magnetic_field_order", "dimensionless"),
        ("hermiticity_defect", "hartree"),
        (
            "magnetic_hermiticity_defect",
            "atomic units of magnetic dipole moment",
        ),
        ("electric_hermiticity_defect", "e*bohr"),
        ("oscillator_strength_sum", "dimensionless"),
        ("frozen_core_electronic_energy", "hartree"),
    ] {
        units.set(key, label);
    }
    for key in [
        "mcd_direction",
        "soc_scale",
        "spin_g_factor",
        "active_spatial_orbitals",
        "active_electrons",
        "frozen_core_orbitals",
        "determinant_count",
        "mcd_orientation_order",
    ] {
        units.set(key, "dimensionless");
    }
    result.set("units", units);
    Ok(result)
}
