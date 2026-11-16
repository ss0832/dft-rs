// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Retained complex projector and energy assembly. Rounding remainders are
//! propagated into the actual AO density and its variational energy, rather
//! than used only to print a smaller idempotency residual.

use super::*;
use crate::linalg::AccurateSum;
use crate::response::nuclear::precision::{coefficient, MatrixParts};

#[derive(Clone)]
struct Parts {
    re: MatrixParts,
    im: MatrixParts,
}
impl Parts {
    fn new(value: Cm) -> Self {
        Self {
            re: MatrixParts::new(value.re),
            im: MatrixParts::new(value.im),
        }
    }
    fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(Cm::zeros(rows, cols))
    }
    fn adjoint(&self) -> Self {
        let mut im = self.im.transpose();
        for part in [&mut im.high, &mut im.low] {
            for x in part.as_mut_slice() {
                *x = -*x;
            }
        }
        Self {
            re: self.re.transpose(),
            im,
        }
    }
    fn add_scaled(&mut self, rhs: &Self, scale: f64) {
        self.re.add_scaled(&rhs.re, scale);
        self.im.add_scaled(&rhs.im, scale);
    }
    fn max_abs(&self) -> f64 {
        self.re.max_abs().max(self.im.max_abs())
    }
    fn high(&self) -> Cm {
        Cm {
            re: self.re.high.clone(),
            im: self.im.high.clone(),
        }
    }
    fn low(&self) -> Cm {
        Cm {
            re: self.re.low.clone(),
            im: self.im.low.clone(),
        }
    }
    fn from_parts(high: Cm, low: Cm) -> Self {
        Self {
            re: MatrixParts {
                high: high.re,
                low: low.re,
            },
            im: MatrixParts {
                high: high.im,
                low: low.im,
            },
        }
    }
}
fn product(a: &[Parts], b: &[Parts], order: usize) -> Vec<Parts> {
    let ar: Vec<_> = a.iter().map(|x| x.re.clone()).collect();
    let ai: Vec<_> = a.iter().map(|x| x.im.clone()).collect();
    let br: Vec<_> = b.iter().map(|x| x.re.clone()).collect();
    let bi: Vec<_> = b.iter().map(|x| x.im.clone()).collect();
    (0..=order)
        .map(|n| {
            let mut re = coefficient(&ar, &br, n);
            re.add_scaled(&coefficient(&ai, &bi, n), -1.0);
            let mut im = coefficient(&ar, &bi, n);
            im.add_scaled(&coefficient(&ai, &br, n), 1.0);
            Parts { re, im }
        })
        .collect()
}

fn transport_retained(c: &Cm, overlap: &[Cm], order: usize) -> Vec<Parts> {
    let mut frame = vec![Parts::zeros(c.re.rows, c.re.cols); order + 1];
    frame[0] = Parts::new(c.clone());
    if largest(c) == 0.0 {
        return frame;
    }
    let adjoint: Vec<_> = frame.iter().map(Parts::adjoint).collect();
    let s: Vec<_> = overlap.iter().cloned().map(Parts::new).collect();
    let metric = product(&product(&adjoint, &s, order), &frame, order);
    product(&frame, &inverse_sqrt(metric, order), order)
}

fn inverse_sqrt(mut metric: Vec<Parts>, order: usize) -> Vec<Parts> {
    let dim = metric[0].re.high.rows;
    let identity = Parts::new(from_real(&Matrix::identity(dim)));
    metric[0].add_scaled(&identity, -1.0);
    let mut power = vec![Parts::zeros(dim, dim); order + 1];
    power[0] = identity;
    let mut inverse = power.clone();
    let mut binomial = 1.0;
    for k in 1..=order + 4 {
        power = product(&power, &metric, order);
        binomial *= (-0.5 - (k - 1) as f64) / k as f64;
        for n in 0..=order {
            inverse[n].add_scaled(&power[n], binomial);
        }
    }
    inverse
}

/// Transport the occupied/virtual frame with the same retained metric
/// algebra as the parent path. Preserve its low parts through the orbital
/// and Fock contractions, where fourth-order coefficients can cancel.
fn projector_frame(
    b: &[Parts],
    d: &[Parts],
    f: &[Parts],
    occupied: usize,
    order: usize,
) -> (Vec<Cm>, Vec<Cm>, f64) {
    let dim = d[0].re.high.rows;
    let identity = Parts::new(from_real(&Matrix::identity(dim)));
    let mut q = d.to_vec();
    q[0] = identity.clone();
    for value in q.iter_mut().skip(1) {
        for block in [&mut value.re, &mut value.im] {
            for i in 0..dim {
                for j in occupied..dim {
                    block.high[(i, j)] *= -1.0;
                    block.low[(i, j)] *= -1.0;
                }
            }
        }
    }
    let qt: Vec<_> = q.iter().map(Parts::adjoint).collect();
    let gram = product(&qt, &q, order);
    let frame = product(&q, &inverse_sqrt(gram, order), order);
    let ft: Vec<_> = frame.iter().map(Parts::adjoint).collect();
    let mut check = product(&ft, &frame, order);
    check[0].add_scaled(&identity, -1.0);
    let mut defect = check.iter().map(Parts::max_abs).fold(0.0, f64::max);
    let mut occupied_frame = frame.clone();
    for value in &mut occupied_frame {
        for block in [&mut value.re, &mut value.im] {
            for i in 0..dim {
                for j in occupied..dim {
                    block.high[(i, j)] = 0.0;
                    block.low[(i, j)] = 0.0;
                }
            }
        }
    }
    for (mut actual, expected) in product(&occupied_frame, &ft, order).into_iter().zip(d) {
        actual.add_scaled(expected, -1.0);
        defect = defect.max(actual.max_abs());
    }
    let orbitals = product(b, &frame, order).iter().map(Parts::high).collect();
    let fock = product(&product(&ft, f, order), &frame, order)
        .iter()
        .map(Parts::high)
        .collect();
    (orbitals, fock, defect)
}

type Channels = [Vec<Parts>; 2];
fn evaluate<B: RetainedComplexBackend>(
    backend: &B,
    b: &Channels,
    d: &Channels,
    order: usize,
) -> Result<(Channels, Channels)> {
    let p = std::array::from_fn(|s| {
        let bt: Vec<_> = b[s].iter().map(Parts::adjoint).collect();
        product(&product(&b[s], &d[s], order), &bt, order)
    });
    let high = std::array::from_fn(|s| p[s].iter().map(Parts::high).collect());
    let low = std::array::from_fn(|s| p[s].iter().map(Parts::low).collect());
    let mut f: Channels = std::array::from_fn(|_| Vec::with_capacity(order + 1));
    for n in 0..=order {
        let (h, l) = backend.retained_fock_coefficient(&high, &low, n)?;
        for s in 0..2 {
            f[s].push(Parts::from_parts(h[s].clone(), l[s].clone()));
        }
    }
    Ok((p, f))
}
fn mo_fock(b: &Channels, f: &Channels, order: usize) -> Channels {
    std::array::from_fn(|s| {
        let bt: Vec<_> = b[s].iter().map(Parts::adjoint).collect();
        product(&product(&bt, &f[s], order), &b[s], order)
    })
}
fn commutator(f: &Channels, d: &Channels, n: usize) -> [Parts; 2] {
    std::array::from_fn(|s| {
        let mut c = product(&f[s], &d[s], n).pop().expect("complete commutator");
        c.add_scaled(
            &product(&d[s], &f[s], n).pop().expect("complete commutator"),
            -1.0,
        );
        c
    })
}
fn source(reference: &ComplexPathReference, g: &[Parts; 2]) -> Vec<Complex> {
    let mut out = Vec::new();
    for s in 0..reference.spins() {
        for a in reference.occupations[s]..g[s].re.high.rows {
            for i in 0..reference.occupations[s] {
                out.push(Complex::from(-g[s].re.high[(a, i)]));
                out.push(Complex::from(-g[s].im.high[(a, i)]));
            }
        }
    }
    out
}
fn add_amplitudes(reference: &ComplexPathReference, d: &mut Channels, n: usize, x: &[Complex]) {
    let mut index = 0;
    for s in 0..reference.spins() {
        for a in reference.occupations[s]..d[s][n].re.high.rows {
            for i in 0..reference.occupations[s] {
                d[s][n].re.add_entry(a, i, x[index].re);
                d[s][n].im.add_entry(a, i, x[index + 1].re);
                d[s][n].re.high[(i, a)] = d[s][n].re.high[(a, i)];
                d[s][n].re.low[(i, a)] = d[s][n].re.low[(a, i)];
                d[s][n].im.high[(i, a)] = -d[s][n].im.high[(a, i)];
                d[s][n].im.low[(i, a)] = -d[s][n].im.low[(a, i)];
                index += 2;
            }
        }
    }
    if reference.spins() == 1 {
        d[1][n] = d[0][n].clone();
    }
}
fn rotate_reference(reference: &ComplexPathReference, b: &mut Channels, x: &[Complex]) {
    let mut index = 0;
    let mut rotations = Vec::new();
    for s in 0..reference.spins() {
        let dim = b[s][0].re.high.cols;
        let mut generator = Parts::zeros(dim, dim);
        for a in reference.occupations[s]..dim {
            for i in 0..reference.occupations[s] {
                generator.re.high[(a, i)] = x[index].re;
                generator.re.high[(i, a)] = -x[index].re;
                generator.im.high[(a, i)] = x[index + 1].re;
                generator.im.high[(i, a)] = x[index + 1].re;
                index += 2;
            }
        }
        let identity = Parts::new(from_real(&Matrix::identity(dim)));
        let mut rotation = identity.clone();
        let mut power = vec![identity];
        let mut factorial = 1.0;
        for k in 1..=4 {
            power = product(&power, std::slice::from_ref(&generator), 0);
            factorial /= k as f64;
            rotation.add_scaled(&power[0], factorial);
        }
        rotations.push(rotation);
    }
    if reference.spins() == 1 {
        rotations.push(rotations[0].clone());
    }
    for s in 0..2 {
        for coefficient in &mut b[s] {
            *coefficient = product(
                std::slice::from_ref(coefficient),
                std::slice::from_ref(&rotations[s]),
                0,
            )
            .pop()
            .expect("constant frame rotation");
        }
    }
}

pub(super) fn evaluate_path<B: RetainedComplexBackend>(
    backend: &B,
    reference: &ComplexPathReference,
    order: usize,
    full: bool,
    config: &ResponseSolverConfig,
) -> Result<(ComplexEnergyPath, Option<ComplexPathState>)> {
    use crate::response::nuclear::precision::RefinementHistory;
    let solved_order = if full { order } else { order / 2 };
    let mut b: Channels = std::array::from_fn(|s| {
        transport_retained(&reference.coefficients[s], backend.overlap(), order)
    });
    let mut d: Channels = std::array::from_fn(|s| {
        let dim = reference.coefficients[s].re.cols;
        let mut initial = Parts::zeros(dim, dim);
        for i in 0..reference.occupations[s] {
            initial.re.high[(i, i)] = 1.0;
        }
        vec![initial]
    });
    let start = backend.fock_coefficient(
        &std::array::from_fn(|s| vec![reference.density[s].clone()]),
        0,
    )?;
    let mut reference_defect = 0.0_f64;
    for s in 0..2 {
        let mut delta = start[s].clone();
        cadd(&mut delta, &reference.fock[s], -1.0);
        reference_defect = reference_defect.max(largest(&delta));
    }
    let action = ComplexOrbitalAction { backend, reference };
    let mut history = RefinementHistory::new();
    for step in 0..=12 {
        let (_, ao) = evaluate(backend, &b, &d, 0)?;
        let mo = mo_fock(&b, &ao, 0);
        let rhs = source(reference, &commutator(&mo, &d, 0));
        let norm = rhs.iter().fold(0.0_f64, |v, z| v.hypot(z.re));
        let stalled = history.observe(norm, &b)?;
        if norm <= config.tolerance * f64::EPSILON || stalled || step == 12 {
            break;
        }
        let solution = krylov::solve(&action, &rhs, None, &config.for_absolute_residual(norm)?)?;
        rotate_reference(reference, &mut b, &solution.amplitudes);
    }
    b = history.finish();
    let mut residuals = Vec::new();
    for n in 1..=order {
        for s in 0..2 {
            let dim = reference.coefficients[s].re.cols;
            let occupied = reference.occupations[s];
            d[s].push(Parts::zeros(dim, dim));
            let interior = product(&d[s], &d[s], n).pop().expect("complete product");
            for i in 0..dim {
                for j in 0..dim {
                    if (i < occupied) == (j < occupied) {
                        let sign = if i < occupied { -1.0 } else { 1.0 };
                        let dn = &mut d[s][n];
                        for (to, from) in [(&mut dn.re, &interior.re), (&mut dn.im, &interior.im)] {
                            to.high[(i, j)] = sign * from.high[(i, j)];
                            to.low[(i, j)] = sign * from.low[(i, j)];
                        }
                    }
                }
            }
        }
        if n <= solved_order {
            let mut history = RefinementHistory::new();
            let mut initial = None;
            for step in 0..=8 {
                let (_, ao) = evaluate(backend, &b, &d, n)?;
                let mo = mo_fock(&b, &ao, n);
                let rhs = source(reference, &commutator(&mo, &d, n));
                let norm = rhs.iter().fold(0.0_f64, |v, z| v.hypot(z.re));
                initial.get_or_insert(norm);
                let current: [Parts; 2] = std::array::from_fn(|s| d[s][n].clone());
                let stalled = history.observe(norm, &current)?;
                if norm <= config.tolerance * f64::EPSILON || stalled || step == 8 {
                    break;
                }
                let solution = if full {
                    // The first high-order source can be large. Correct the
                    // actual retained commutator in successive solves and
                    // apply the absolute bound to the final physical path.
                    krylov::solve(&action, &rhs, None, config)?
                } else {
                    krylov::solve(&action, &rhs, None, &config.for_absolute_residual(norm)?)?
                };
                add_amplitudes(reference, &mut d, n, &solution.amplitudes);
            }
            let initial = initial.expect("the physical complex source is evaluated");
            residuals.push(if initial == 0.0 {
                history.norm
            } else {
                history.norm / initial
            });
            let best = history.finish();
            for s in 0..2 {
                d[s][n] = best[s].clone();
            }
        }
    }
    let (p, ao) = evaluate(backend, &b, &d, order)?;
    let high: [Vec<Cm>; 2] = std::array::from_fn(|s| p[s].iter().map(Parts::high).collect());
    let mut energy = backend.energy(&high, order)?;
    // The same analytic energy's first variation accounts for retained low
    // densities. Low*low terms are second order in machine rounding, exactly
    // as in the real retained nuclear provider; no finite displacement occurs.
    for n in 0..=order {
        let mut value = AccurateSum::default();
        value.add_product(energy[n], 1.0);
        for k in 0..=n {
            for s in 0..2 {
                let f = &ao[s][n - k];
                for i in 0..f.re.high.rows {
                    for j in 0..f.re.high.cols {
                        value.add_product(p[s][k].re.low[(i, j)], f.re.high[(j, i)]);
                        value.add_product(-p[s][k].im.low[(i, j)], f.im.high[(j, i)]);
                    }
                }
            }
        }
        energy[n] = value.value();
    }
    let mut projector = vec![0.0_f64; order];
    let mut stationarity = vec![0.0_f64; solved_order];
    let mo = mo_fock(&b, &ao, solved_order);
    let g0 = commutator(&mo, &d, 0);
    reference_defect = reference_defect.max(g0.iter().map(Parts::max_abs).fold(0.0, f64::max));
    for s in 0..2 {
        let square = product(&d[s], &d[s], order);
        for n in 1..=order {
            let mut defect = square[n].clone();
            defect.add_scaled(&d[s][n], -1.0);
            projector[n - 1] = projector[n - 1].max(defect.max_abs());
        }
        let fd = product(&mo[s], &d[s], solved_order);
        let df = product(&d[s], &mo[s], solved_order);
        for n in 1..=solved_order {
            let mut defect = fd[n].clone();
            defect.add_scaled(&df[n], -1.0);
            stationarity[n - 1] = stationarity[n - 1].max(defect.max_abs());
        }
    }
    let hermiticity: Vec<f64> = (0..=solved_order)
        .map(|n| {
            (0..2)
                .map(|s| {
                    let mut defect = mo[s][n].clone();
                    defect.add_scaled(&mo[s][n].adjoint(), -1.0);
                    defect.max_abs()
                })
                .fold(0.0, f64::max)
        })
        .collect();
    crate::logger::debug(&format!(
        "[response] retained complex order={order} projector={projector:?} stationarity={stationarity:?} fock_hermiticity={hermiticity:?} reference={reference_defect:.12e}"
    ));
    let state = full.then(|| {
        let high_coefficients = |series: &Channels| {
            std::array::from_fn(|s| series[s].iter().map(Parts::high).collect())
        };
        ComplexPathState {
            b: high_coefficients(&b),
            bt: std::array::from_fn(|s| b[s].iter().map(|x| x.adjoint().high()).collect()),
            d: high_coefficients(&d),
            f: high_coefficients(&mo),
            p: high,
            retained_orbital_frames: Some(std::array::from_fn(|s| {
                projector_frame(&b[s], &d[s], &mo[s], reference.occupations[s], order)
            })),
            retained_number_defects: Some(std::array::from_fn(|s| {
                (0..=order)
                    .map(|n| {
                        // Measure Tr(P S) from the same retained AO density
                        // used for the physical Fock and energy. Tracing
                        // only the exported high parts loses the low parts
                        // precisely where different AO terms cancel.
                        let mut trace = AccurateSum::default();
                        if n == 0 {
                            trace.add_product(-1.0, reference.occupations[s] as f64);
                        }
                        for k in 0..=n {
                            let density = &p[s][k];
                            let overlap = &backend.overlap()[n - k];
                            for i in 0..overlap.re.rows {
                                for j in 0..overlap.re.cols {
                                    let real = overlap.re[(j, i)];
                                    let imag = overlap.im[(j, i)];
                                    trace.add_product(density.re.high[(i, j)], real);
                                    trace.add_product(density.re.low[(i, j)], real);
                                    trace.add_product(density.im.high[(i, j)], -imag);
                                    trace.add_product(density.im.low[(i, j)], -imag);
                                }
                            }
                        }
                        trace.value().abs()
                    })
                    .collect()
            })),
        }
    });
    Ok((
        ComplexEnergyPath {
            energy,
            solved_order,
            reference_defect,
            residuals,
            projector_defects: projector,
            stationarity_defects: stationarity,
        },
        state,
    ))
}
