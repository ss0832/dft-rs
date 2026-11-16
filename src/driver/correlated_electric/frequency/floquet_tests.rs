// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent finite-amplitude HF and PT2 Floquet quasi-energy oracle.

use super::super::add;
use super::*;
use crate::acceptance::{components, stencils::Stencil};
use crate::driver::double_hybrid::moment_tests::{correlated_controls, system, EXPONENT};
use crate::driver::{Controls, System};
use crate::linalg::{solve_linear, symmetric_eigen};
use crate::math::Vec3;
use crate::periodic::BlochMatrix;

const SIDEBANDS: i32 = 2;
const HARMONICS: usize = (2 * SIDEBANDS + 1) as usize;
const TIMES: usize = 17;

fn offset(h: i32) -> usize {
    (h + 2 * SIDEBANDS) as usize
}

fn phase(h: i32, t: usize) -> Complex {
    let angle = -std::f64::consts::TAU * f64::from(h) * t as f64 / TIMES as f64;
    Complex::new(angle.cos(), angle.sin())
}

fn extract(values: &[CMatrix], h: i32) -> CMatrix {
    let mut result = CMatrix::zero(values[0].re.rows);
    for (t, value) in values.iter().enumerate() {
        result.add(value, phase(-h, t) / TIMES as f64);
    }
    result
}

fn synthesize(values: &[CMatrix], t: usize) -> CMatrix {
    let mut result = CMatrix::zero(values[0].re.rows);
    for h in -2 * SIDEBANDS..=2 * SIDEBANDS {
        result.add(&values[offset(h)], phase(h, t));
    }
    result
}

fn adjoint(a: &CMatrix) -> CMatrix {
    let mut result = transpose(a);
    for v in result.im.as_mut_slice() {
        *v = -*v;
    }
    result
}

fn value(a: &CMatrix, row: usize, col: usize) -> Complex {
    Complex::new(a.re[(row, col)], a.im[(row, col)])
}

fn set(a: &mut CMatrix, row: usize, col: usize, z: Complex) {
    a.re[(row, col)] = z.re;
    a.im[(row, col)] = z.im;
}

type SpinLanes<T> = [Vec<T>; 2];

struct FiniteFloquet<'a> {
    prepared: &'a Prepared,
    core: [Matrix; 2],
    eri: [[Vec<f64>; 2]; 2],
    orbitals: &'a [Matrix; 2],
    occupied: [usize; 2],
    omega: f64,
    coordinates: Vec<(usize, i32, usize, usize)>,
    jacobian: Matrix,
}

impl<'a> FiniteFloquet<'a> {
    fn new(prepared: &'a Prepared, omega: f64) -> Self {
        let scf = &prepared.reference.scf;
        let orbitals = &scf.mo_coeff;
        let n = orbitals[0].cols;
        assert_eq!(n, orbitals[1].cols);
        let mut coordinates = Vec::new();
        for spin in 0..2 {
            for h in 0..=2 * SIDEBANDS {
                for i in 0..n {
                    for j in 0..n {
                        if h != 0 || i <= j {
                            coordinates.push((spin, h, i, j));
                        }
                    }
                }
            }
        }
        let mut model = Self {
            prepared,
            core: orbitals
                .each_ref()
                .map(|c| c.transpose_matmul_seq(&scf.h_core.matmul_seq(c))),
            eri: std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; n.pow(4)])),
            orbitals,
            occupied: [scf.n_alpha, scf.n_beta],
            omega,
            jacobian: Matrix::zeros(coordinates.len(), coordinates.len()),
            coordinates,
        };
        // Direct small four-index sum in each pair of spin orbital bases,
        // independent of the production coefficient path.
        for (spin, left) in orbitals.iter().enumerate() {
            for (other, right) in orbitals.iter().enumerate() {
                for p in 0..n {
                    for q in 0..n {
                        for r in 0..n {
                            for s in 0..n {
                                let mut integral = 0.0;
                                for a in 0..left.rows {
                                    for b in 0..left.rows {
                                        for c in 0..right.rows {
                                            for d in 0..right.rows {
                                                integral += left[(a, p)]
                                                    * left[(b, q)]
                                                    * right[(c, r)]
                                                    * right[(d, s)]
                                                    * prepared.eri.get(a, b, c, d);
                                            }
                                        }
                                    }
                                }
                                let index = model.index(p, q, r, s);
                                model.eri[spin][other][index] = integral;
                            }
                        }
                    }
                }
            }
        }
        model.jacobian = model.parent_jacobian();
        model
    }

    fn index(&self, p: usize, q: usize, r: usize, s: usize) -> usize {
        let n = self.orbitals[0].cols;
        ((p * n + q) * n + r) * n + s
    }

    fn two_electron_field(
        &self,
        spin: usize,
        other: usize,
        density: &Matrix,
        coulomb: f64,
        exchange: f64,
    ) -> Matrix {
        let n = density.rows;
        let mut fock = Matrix::zeros(n, n);
        let eri = &self.eri[spin][other];
        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        fock[(p, q)] += coulomb * eri[self.index(p, q, r, s)] * density[(s, r)]
                            + exchange * eri[self.index(p, r, q, s)] * density[(r, s)];
                    }
                }
            }
        }
        fock
    }

    fn mean_field(&self, spin: usize, density: [&Matrix; 2]) -> Matrix {
        let mut fock = self.two_electron_field(spin, spin, density[spin], 1.0, -1.0);
        let other = 1 - spin;
        add(
            &mut fock,
            &self.two_electron_field(spin, other, density[other], 1.0, 0.0),
            1.0,
        );
        fock
    }

    fn fock_lanes(&self, density: &SpinLanes<Matrix>, source: &[Matrix; 2]) -> SpinLanes<Matrix> {
        std::array::from_fn(|spin| {
            let mut fock: Vec<_> = density[0]
                .iter()
                .zip(&density[1])
                .map(|(alpha, beta)| self.mean_field(spin, [alpha, beta]))
                .collect();
            add(&mut fock[offset(0)], &self.core[spin], 1.0);
            add(&mut fock[offset(1)], &source[spin], 0.5);
            add(&mut fock[offset(-1)], &source[spin], 0.5);
            fock
        })
    }

    /// Differentiate the small oracle Sambe eigenprojector at zero
    /// drive to precondition its finite-amplitude density residual. No
    /// production response quantities enter this finite-field solve.
    fn parent_jacobian(&self) -> Matrix {
        let energies = &self.prepared.reference.scf.mo_energy;
        let n = energies[0].len();
        let mut jacobian = Matrix::zeros(self.coordinates.len(), self.coordinates.len());
        for (col, &(other, h, i, j)) in self.coordinates.iter().enumerate() {
            let mut density = Matrix::zeros(n, n);
            density[(i, j)] = 1.0;
            if h == 0 {
                density[(j, i)] = 1.0;
            }
            let fock: [Matrix; 2] = std::array::from_fn(|spin| {
                let exchange = if spin == other { -1.0 } else { 0.0 };
                self.two_electron_field(spin, other, &density, 1.0, exchange)
            });
            // A sideband eigenvector has zeroth-order energy eps_p-h*omega.
            // Only the occupied reference roots in the central block are
            // projected, with transpose-related real density harmonics.
            for (row, &(spin, lane, p, q)) in self.coordinates.iter().enumerate() {
                if lane == h && h <= SIDEBANDS {
                    let frequency = self.omega * f64::from(h);
                    let occupied = self.occupied[spin];
                    let eps = &energies[spin];
                    if q < occupied && (h != 0 || p >= occupied) {
                        jacobian[(row, col)] += fock[spin][(p, q)] / (eps[q] - eps[p] + frequency);
                    }
                    if p < occupied && (h != 0 || q >= occupied) {
                        jacobian[(row, col)] += fock[spin][(p, q)] / (eps[p] - eps[q] - frequency);
                    }
                }
            }
            jacobian[(col, col)] -= 1.0;
        }
        jacobian
    }

    fn project(&self, spin: usize, fock: &[Matrix]) -> (Vec<Matrix>, f64) {
        let n = self.orbitals[spin].cols;
        let mut floquet = Matrix::zeros(n * HARMONICS, n * HARMONICS);
        for h in -SIDEBANDS..=SIDEBANDS {
            for k in -SIDEBANDS..=SIDEBANDS {
                let (row, col) = ((h + SIDEBANDS) as usize * n, (k + SIDEBANDS) as usize * n);
                for i in 0..n {
                    for j in 0..n {
                        let shift = if h == k && i == j {
                            self.omega * f64::from(h)
                        } else {
                            0.0
                        };
                        floquet[(row + i, col + j)] = fock[offset(h - k)][(i, j)] - shift;
                    }
                }
            }
        }
        let (eigenvalues, states) = symmetric_eigen(&floquet).unwrap();
        let mut labels: Vec<_> = (0..eigenvalues.len()).collect();
        let occupied = self.occupied[spin];
        let reference_weight = |state| {
            (0..occupied)
                .map(|i| states[(SIDEBANDS as usize * n + i, state)].powi(2))
                .sum::<f64>()
        };
        labels.sort_by(|&a, &b| reference_weight(b).total_cmp(&reference_weight(a)));
        let roots = &labels[..occupied];
        assert!(
            roots.iter().all(|&root| reference_weight(root) > 0.5),
            "spin {spin}, Floquet root weights {:?}",
            roots
                .iter()
                .map(|&root| reference_weight(root))
                .collect::<Vec<_>>()
        );
        let mut density = vec![Matrix::zeros(n, n); 2 * HARMONICS - 1];
        for h in -SIDEBANDS..=SIDEBANDS {
            for k in -SIDEBANDS..=SIDEBANDS {
                let (row, col) = ((h + SIDEBANDS) as usize * n, (k + SIDEBANDS) as usize * n);
                for i in 0..n {
                    for j in 0..n {
                        density[offset(h - k)][(i, j)] += roots
                            .iter()
                            .map(|&root| states[(row + i, root)] * states[(col + j, root)])
                            .sum::<f64>();
                    }
                }
            }
        }
        (density, roots.iter().map(|&root| eigenvalues[root]).sum())
    }

    fn parent(&self, source: &[Matrix; 2]) -> (SpinLanes<CMatrix>, SpinLanes<CMatrix>, f64) {
        let n = self.orbitals[0].cols;
        let mut density: SpinLanes<Matrix> = self.occupied.map(|occupied| {
            let mut result = vec![Matrix::zeros(n, n); 2 * HARMONICS - 1];
            for i in 0..occupied {
                result[offset(0)][(i, i)] = 1.0;
            }
            result
        });
        for _ in 0..50 {
            let fock = self.fock_lanes(&density, source);
            let [(alpha, epsilon_alpha), (beta, epsilon_beta)] =
                std::array::from_fn(|spin| self.project(spin, &fock[spin]));
            let next = [alpha, beta];
            let defect = next
                .iter()
                .flatten()
                .zip(density.iter().flatten())
                .map(|(a, b)| a.max_abs_diff(b))
                .fold(0.0_f64, f64::max);
            if defect < 2e-13 {
                // The sideband eigenvalues contain the full mean field.
                // Remove half the interaction to count each pair once.
                let mut interaction = 0.0;
                for (spin, densities) in next.iter().enumerate() {
                    for (lane, p) in densities.iter().enumerate() {
                        let pair = [&next[0][lane], &next[1][lane]];
                        interaction += self.mean_field(spin, pair).frobenius_dot(p);
                    }
                }
                return (
                    next.map(|lanes| lanes.into_iter().map(CMatrix::real).collect()),
                    fock.map(|lanes| lanes.into_iter().map(CMatrix::real).collect()),
                    epsilon_alpha + epsilon_beta - 0.5 * interaction,
                );
            }
            let residual: Vec<_> = self
                .coordinates
                .iter()
                .map(|&(spin, h, i, j)| {
                    density[spin][offset(h)][(i, j)] - next[spin][offset(h)][(i, j)]
                })
                .collect();
            let correction = solve_linear(&self.jacobian, &residual).unwrap();
            for (&(spin, h, i, j), change) in self.coordinates.iter().zip(correction) {
                let updated = density[spin][offset(h)][(i, j)] + change;
                density[spin][offset(h)][(i, j)] = updated;
                density[spin][offset(-h)][(j, i)] = updated;
            }
        }
        panic!("independent periodic parent density did not converge");
    }

    fn target(&self, density: &SpinLanes<CMatrix>) -> f64 {
        let setup = &self.prepared.setup;
        let target = self.prepared.target.as_ref().unwrap();
        let transposed = self.orbitals.each_ref().map(Matrix::transpose);
        let mut sum = 0.0;
        for t in 0..TIMES {
            let p = density.each_ref().map(|lanes| synthesize(lanes, t));
            let ao: [CMatrix; 2] = std::array::from_fn(|spin| {
                p[spin].transform(&self.orbitals[spin], &transposed[spin])
            });
            sum += crate::ks::xc_contribution(
                target,
                &setup.basis,
                setup.grid.as_ref().unwrap(),
                [&ao[0].re, &ao[1].re],
            )
            .unwrap()
            .exc;
            // The target removes the parent's full HF exchange.
            for (spin, p) in p.iter().enumerate() {
                let exchange = CMatrix {
                    re: self.two_electron_field(spin, spin, &p.re, 0.0, 1.0),
                    im: self.two_electron_field(spin, spin, &p.im, 0.0, 1.0),
                };
                sum += 0.5 * trace_product(p, &exchange).re;
            }
        }
        sum / TIMES as f64
    }

    fn pair_coupling(
        &self,
        [spin, other]: [usize; 2],
        left: &CMatrix,
        right: &CMatrix,
        [i, a, j, b]: [usize; 4],
    ) -> Complex {
        let n = self.orbitals[0].cols;
        let mut coupling = Complex::ZERO;
        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        coupling += value(left, p, a).conj()
                            * value(left, q, i)
                            * value(right, r, b).conj()
                            * value(right, s, j)
                            * self.eri[spin][other][self.index(p, q, r, s)];
                    }
                }
            }
        }
        coupling
    }

    fn pair_element(
        left: &CMatrix,
        right: &CMatrix,
        [i, a, j, b]: [usize; 4],
        [k, c, l, d]: [usize; 4],
    ) -> Complex {
        let mut element = Complex::ZERO;
        if (a, j, b) == (c, l, d) {
            element += value(left, k, i);
        }
        if (i, a, b) == (k, c, d) {
            element += value(right, l, j);
        }
        if (i, j, b) == (k, l, d) {
            element -= value(left, a, c);
        }
        if (i, a, j) == (k, c, l) {
            element -= value(right, b, d);
        }
        element
    }

    fn moving_frame(
        &self,
        spin: usize,
        density: &[CMatrix],
        fock: &[CMatrix],
    ) -> (Vec<CMatrix>, Vec<CMatrix>) {
        let n = self.orbitals[spin].cols;
        let occupied = self.occupied[spin];
        let identity = CMatrix::real(Matrix::identity(n));
        let mut sign = Matrix::identity(n);
        for a in occupied..n {
            sign[(a, a)] = -1.0;
        }
        let frame: Vec<_> = (0..TIMES)
            .map(|t| {
                let mut difference = synthesize(density, t);
                for i in 0..occupied {
                    difference.re[(i, i)] -= 1.0;
                }
                let mut q = identity.clone();
                q.add(&difference.transform(&identity.re, &sign), Complex::ONE);
                // Form the finite polar frame directly. The binomial inverse
                // square root uses the full Gram matrix of the sampled orbitals.
                let mut error = identity.clone();
                error.add(&adjoint(&q).product(&q), Complex::from(-1.0));
                assert!(error.max_abs() < 0.1);
                let mut power = identity.clone();
                let mut normalizer = identity.clone();
                let mut weight = 1.0;
                for k in 1..=10 {
                    power = power.product(&error);
                    weight *= (2 * k - 1) as f64 / (2 * k) as f64;
                    normalizer.add(&power, Complex::from(weight));
                }
                q.product(&normalizer)
            })
            .collect();
        let derivative: Vec<_> = (-(TIMES as i32 / 2)..=TIMES as i32 / 2)
            .map(|h| (h, extract(&frame, h)))
            .collect();
        let generators: Vec<_> = frame
            .iter()
            .enumerate()
            .map(|(t, u)| {
                let mut du = CMatrix::zero(n);
                for (h, coefficient) in &derivative {
                    du.add(coefficient, phase(*h, t) * (self.omega * f64::from(*h)));
                }
                let dual = adjoint(u);
                let mut moving = dual.product(&synthesize(fock, t)).product(u);
                moving.add(&dual.product(&du), Complex::from(-1.0));
                moving
            })
            .collect();
        (frame, generators)
    }

    fn pair_energies(&self, density: &SpinLanes<CMatrix>, fock: &SpinLanes<CMatrix>) -> [f64; 3] {
        let frame: [(Vec<CMatrix>, Vec<CMatrix>); 2] =
            std::array::from_fn(|spin| self.moving_frame(spin, &density[spin], &fock[spin]));
        let n = self.orbitals[0].cols;
        [[0, 0], [0, 1], [1, 1]].map(|[spin, other]| {
            let same_spin = spin == other;
            let mut pairs = Vec::new();
            for i in 0..self.occupied[spin] {
                for a in self.occupied[spin]..n {
                    for j in 0..self.occupied[other] {
                        for b in self.occupied[other]..n {
                            if !same_spin || (i < j && a < b) {
                                pairs.push([i, a, j, b]);
                            }
                        }
                    }
                }
            }
            let size = pairs.len();
            let mut sampled_operator = Vec::new();
            let mut sampled_coupling = Vec::new();
            for t in 0..TIMES {
                let (left, right) = (&frame[spin].0[t], &frame[other].0[t]);
                let (fl, fr) = (&frame[spin].1[t], &frame[other].1[t]);
                let mut generator = CMatrix::zero(size);
                let mut coupling = Vec::new();
                for (row, &pair) in pairs.iter().enumerate() {
                    let [i, a, j, b] = pair;
                    let couple = |indices| self.pair_coupling([spin, other], left, right, indices);
                    let direct = couple(pair);
                    coupling.push(if same_spin {
                        direct - couple([i, b, j, a])
                    } else {
                        direct
                    });
                    for (col, &indices) in pairs.iter().enumerate() {
                        let element = |indices| Self::pair_element(fl, fr, pair, indices);
                        let mut total = element(indices);
                        if same_spin {
                            // The independent wedge determinants carry a
                            // minus sign for an occupied or virtual swap.
                            let [k, c, l, d] = indices;
                            total = total - element([l, c, k, d]) - element([k, d, l, c])
                                + element([l, d, k, c]);
                        }
                        set(&mut generator, row, col, total);
                    }
                }
                sampled_operator.push(generator);
                sampled_coupling.push(coupling);
            }
            let matrix_lanes: Vec<_> = (-2 * SIDEBANDS..=2 * SIDEBANDS)
                .map(|h| extract(&sampled_operator, h))
                .collect();
            let mut operator = BlochMatrix::zeros(size * HARMONICS, size * HARMONICS);
            let mut rhs = vec![Complex::ZERO; operator.rows()];
            for h in -SIDEBANDS..=SIDEBANDS {
                let row = (h + SIDEBANDS) as usize * size;
                for i in 0..size {
                    rhs[row + i] = sampled_coupling
                        .iter()
                        .enumerate()
                        .fold(Complex::ZERO, |sum, (t, g)| sum + g[i] * phase(-h, t))
                        / TIMES as f64;
                }
                for k in -SIDEBANDS..=SIDEBANDS {
                    let col = (k + SIDEBANDS) as usize * size;
                    for i in 0..size {
                        for j in 0..size {
                            let mut element = value(&matrix_lanes[offset(h - k)], i, j);
                            if h == k && i == j {
                                element += Complex::from(self.omega * f64::from(h));
                            }
                            operator.re[(row + i, col + j)] = element.re;
                            operator.im[(row + i, col + j)] = element.im;
                        }
                    }
                }
            }
            let real_rhs: Vec<_> = rhs
                .iter()
                .map(|z| z.re)
                .chain(rhs.iter().map(|z| z.im))
                .collect();
            let solution = solve_linear(&operator.real_embedding(), &real_rhs).unwrap();
            let energy = rhs.iter().enumerate().fold(Complex::ZERO, |sum, (i, g)| {
                sum + g.conj() * Complex::new(solution[i], solution[rhs.len() + i])
            });
            assert!(energy.im.abs() < 1e-11);
            energy.re
        })
    }

    fn energy(&self, source: &[Matrix; 2]) -> [f64; 7] {
        let (density, fock, parent) = self.parent(source);
        let target = self.target(&density);
        let [ss_alpha, opposite_spin, ss_beta] = self.pair_energies(&density, &fock);
        let coefficients = self.prepared.coefficients;
        let pt2 = coefficients.same_spin * (ss_alpha + ss_beta)
            + coefficients.opposite_spin * opposite_spin;
        [
            parent + target + pt2,
            parent,
            target,
            pt2,
            ss_alpha,
            opposite_spin,
            ss_beta,
        ]
    }
}

fn fit_floquet(unrestricted: bool) {
    let system = if unrestricted {
        system(true)
    } else {
        System::molecule(
            vec![1; 4],
            vec![
                [0.13, -0.21, 0.19],
                [1.57, 0.16, -0.08],
                [-0.19, 1.93, 0.37],
                [0.42, 0.29, 2.28],
            ],
        )
    };
    let basis = std::env::temp_dir().join(format!(
        "dft-rs-dynamic-pt2-floquet-oracle-{unrestricted}-{}.g94",
        std::process::id()
    ));
    std::fs::write(&basis, format!("H 0\nS 1 1.0\n{EXPONENT} 1.0\n****\n")).unwrap();
    let controls = correlated_controls("custom", unrestricted, basis.to_str().unwrap())
        .with("grid_level", 0usize);
    let input = Controls::from_value(&controls).unwrap();
    let mut prepared = crate::driver::double_hybrid::prepare(&system, &input, "energy").unwrap();
    let operators = crate::integrals::dipole_matrices(&prepared.setup.basis, Vec3::zero());
    let config = ResponseSolverConfig::default();
    // Keep the weak-drive fit away from close replicas of the occupied
    // spin orbitals. The open-shell reference has a 0.0554 Eh occupied
    // alpha spacing, so its two-sideband oracle converges better at 0.129.
    let omega = if unrestricted { 0.129 } else { 0.071 };
    let z = Complex::from(omega);
    let analytic = solve(&prepared, &operators, z, &config).unwrap();
    let exact = [
        &analytic.response.tensor,
        &analytic.parent,
        &analytic.target,
        &analytic.pt2,
        &analytic.channels[0],
        &analytic.channels[1],
        &analytic.channels[2],
    ];
    let pairs = (0..3)
        .flat_map(|a| (a..3).map(move |b| (a, b)))
        .collect::<Vec<_>>();
    let names: Vec<_> = pairs
        .iter()
        .flat_map(|&(a, b)| {
            [
                "total",
                "parent",
                "target",
                "weighted PT2",
                "SS-alpha",
                "OS",
                "SS-beta",
            ]
            .map(|part| format!("molecular Floquet {part} [{a},{b}]"))
        })
        .collect();
    for tensor in exact {
        for (a, b) in pairs.iter().copied() {
            assert!(tensor[a][b].im.abs() < 1e-9);
            assert!((tensor[a][b] - tensor[b][a]).abs() < 1e-8);
        }
    }
    let model = FiniteFloquet::new(&prepared, omega);
    assert_eq!(model.occupied, if unrestricted { [3, 2] } else { [2, 2] });
    let sources = operators.each_ref().map(|source| {
        model
            .orbitals
            .each_ref()
            .map(|c| c.transpose_matmul_seq(&source.matmul_seq(c)))
    });
    let sample = |a: usize, x: f64, b: usize, y: f64| {
        let source = std::array::from_fn(|spin| {
            let mut s = Matrix::zeros(sources[a][spin].rows, sources[a][spin].cols);
            add(&mut s, &sources[a][spin], x);
            add(&mut s, &sources[b][spin], y);
            s
        });
        model.energy(&source)
    };
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &[
            [0.0024, 0.0018, 0.0012, 0.00075],
            [0.0012, 0.0009, 0.0006, 0.00045],
        ],
        |steps| {
            let mut errors = vec![[0.0; 4]; names.len()];
            for (p, (a, b)) in pairs.iter().copied().enumerate() {
                for (step, &h) in steps.iter().enumerate() {
                    let pp = sample(a, h, b, h);
                    let pm = sample(a, h, b, -h);
                    let mp = sample(a, -h, b, h);
                    let mm = sample(a, -h, b, -h);
                    let first = |plus, minus| Stencil::Second.first(plus, minus, 0.0, 0.0, h);
                    for c in 0..7 {
                        // Each cosine field contains +omega and -omega
                        // amplitudes of one half. The mixed energy
                        // derivative is minus half the susceptibility.
                        let observed = -2.0 * first(first(pp[c], pm[c]), first(mp[c], mm[c]));
                        errors[7 * p + c][step] = (exact[c][a][b].re - observed).abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("molecular finite-amplitude Floquet response", &failures);
    println!(
        "Molecular Floquet correlated dipole response unrestricted={unrestricted} omega={omega}: {:?}",
        exact.map(|matrix| matrix[0][0])
    );

    // With zero pair weights and no distinct target the causal total
    // recovers the ordinary parent response for every ordered component.
    prepared.coefficients.same_spin = 0.0;
    prepared.coefficients.opposite_spin = 0.0;
    prepared.target = None;
    let parent_only = solve(&prepared, &operators, z, &config).unwrap();
    let setup = &prepared.setup;
    let paired = crate::response::dynamic::real_operator_polarizability(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &prepared.reference.scf,
        &setup.options,
        &operators,
        crate::response::Frequency::Retarded {
            omega,
            damping: 0.0,
        },
        &config,
    )
    .unwrap();
    for a in 0..3 {
        for b in 0..3 {
            assert!((parent_only.response.tensor[a][b] - analytic.parent[a][b]).abs() < 1e-11);
            assert!((parent_only.response.tensor[a][b] - paired.tensor[a][b]).abs() < 1e-9);
            assert_eq!(parent_only.pt2[a][b], Complex::ZERO);
            assert_eq!(parent_only.target[a][b], Complex::ZERO);
        }
    }
    std::fs::remove_file(basis).unwrap();
}

#[test]
fn assembled_collinear_pt2_floquet_fits_all_dipole_components_and_channels() {
    fit_floquet(false);
}

#[test]
fn assembled_unrestricted_pt2_floquet_fits_all_dipole_components_and_channels() {
    fit_floquet(true);
}
