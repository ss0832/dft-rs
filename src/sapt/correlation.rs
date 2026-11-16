// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Fixed-orbital mixed-order SRS coefficients for two RHF monomers.
//!
//! Independently correlated monomer states define ground and excited product
//! sectors. Expanding their projectors and the reduced resolvent keeps the
//! sector assignment valid as the Moller-Plesset fluctuations are switched on.

use super::antisymmetry::Interaction;
use super::correlation_kernels::{bare_actions, exchange_expectations, spin_bytes, vector_bytes};
use super::relaxed_induction::{density_fock, orbital_response, Monomer as ResponseMonomer};
use super::third_order::{self, Determinant, Operators, SaptThirdOrder};
use super::truncated_induction::{self, Corrections};
use crate::linalg::Matrix;
use crate::mcscf::ci::{ActiveHamiltonian, CiSpace};
use crate::{DftError, Result};

/// Mixed-order coefficients with all spatial orbitals correlated.
/// All energies are in Hartree.
#[derive(Clone, Copy, Debug)]
pub struct SaptCorrelationTerms {
    /// Polarization coefficients of V W, V W^2, and V W^3.
    pub electrostatics: [f64; 3],
    /// Full-antisymmetrizer exchange coefficients of V W, V W^2, and V W^3.
    pub exchange: [f64; 3],
    /// Orbital-relaxed E_elst(12) and E_elst(13), by monomer.
    pub relaxed_electrostatics: [[f64; 2]; 2],
    /// Truncated E_ind(22) with induction of A and B in Hartree.
    pub truncated_induction22: [f64; 2],
    /// Estimated exchange counterpart from the coupled second-order ratio.
    pub truncated_exchange_induction22: f64,
    /// Polarization coefficients of V^2 W and V^2 W^2 in each excited sector.
    /// Sector order is A only, B only, then both monomers.
    pub second_order_polarization: [[f64; 2]; 3],
    /// Exchange corrections to the same sector coefficients and orders.
    pub second_order_exchange: [[f64; 2]; 3],
}

struct Monomer {
    determinants: Vec<Determinant>,
    strings: usize,
    fluctuation: Matrix,
    energies: [f64; 4],
    ground: [Vec<f64>; 4],
    relaxed_electrostatics: [f64; 2],
    induction: Corrections,
    density_second: Matrix,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn add(target: &mut [f64], source: &[f64], factor: f64) {
    for (x, y) in target.iter_mut().zip(source) {
        *x += factor * y;
    }
}

fn outer(a: &[f64], b: &[f64]) -> Matrix {
    Matrix::from_row_major(
        a.len(),
        b.len(),
        a.iter()
            .flat_map(|x| b.iter().map(move |y| x * y))
            .collect(),
    )
}

fn apply(m: &Matrix, x: &[f64]) -> Vec<f64> {
    m.as_slice()
        .chunks_exact(m.cols)
        .map(|row| dot(row, x))
        .collect()
}

fn spatial_strings(nmo: usize, nocc: usize) -> usize {
    let mut strings = 1usize;
    for i in 0..nocc.min(nmo - nocc) {
        strings = strings.saturating_mul(nmo - i) / (i + 1);
    }
    strings
}

impl Monomer {
    fn new(
        response: &ResponseMonomer<'_>,
        h: &Matrix,
        offset: usize,
        operators: &Operators,
    ) -> Result<Self> {
        let scf = response.scf;
        let coefficients = &scf.mo_coeff[0];
        let n = coefficients.cols;
        let space = CiSpace::new(n, scf.n_alpha, scf.n_beta)?;
        let mut g = Vec::with_capacity(n.pow(4));
        for p in offset..offset + n {
            for q in offset..offset + n {
                for r in offset..offset + n {
                    for s in offset..offset + n {
                        g.push(operators.eri(p, q, r, s));
                    }
                }
            }
        }
        let ham = ActiveHamiltonian {
            n,
            constant: 0.0,
            h: coefficients.transpose_matmul(&h.matmul(coefficients)),
            g,
        };
        let d = space.dimension();
        let strings = spatial_strings(n, scf.n_alpha);
        let orbital_energies = &scf.mo_energy[0];
        let reference_energy = 2.0 * orbital_energies[..scf.n_alpha].iter().sum::<f64>();
        let determinants = (0..d)
            .map(|index| {
                let local = space
                    .occupations(index)
                    .map(|mask| (0..n).filter(|i| (mask >> i) & 1 != 0).collect::<Vec<_>>());
                let gap = local
                    .iter()
                    .flatten()
                    .map(|&p| orbital_energies[p])
                    .sum::<f64>()
                    - reference_energy;
                Determinant {
                    rank: local
                        .iter()
                        .flatten()
                        .filter(|&&p| p >= scf.n_alpha)
                        .count(),
                    orbitals: local.map(|list| list.into_iter().map(|p| offset + p).collect()),
                    gap,
                }
            })
            .collect::<Vec<_>>();
        let mut fluctuation = Matrix::zeros(d, d);
        let mut unit = vec![0.0; d];
        for (i, determinant) in determinants.iter().enumerate() {
            unit[i] = 1.0;
            let sigma = space.sigma(&ham, &unit)?;
            unit[i] = 0.0;
            for (j, &value) in sigma.iter().enumerate() {
                fluctuation[(j, i)] = value;
            }
            fluctuation[(i, i)] -= determinant.gap;
        }
        let constant = fluctuation[(0, 0)];
        for i in 0..d {
            fluctuation[(i, i)] -= constant;
        }

        // First solve the intermediate-normalized MP recurrence. Then
        // normalize the state to unit length for the product projectors.
        let mut ground: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0; d]);
        let mut energies = [0.0; 4];
        ground[0][0] = 1.0;
        for order in 1..=3 {
            let mut source = apply(&fluctuation, &ground[order - 1]);
            energies[order] = source[0];
            for lower in 1..order {
                add(&mut source, &ground[order - lower], -energies[lower]);
            }
            for i in 1..d {
                ground[order][i] = -source[i] / determinants[i].gap;
            }
        }
        let relaxed_electrostatics = electrostatic_response(
            response,
            &space,
            &ham,
            &fluctuation,
            &determinants,
            &ground[1],
            &ground[0],
        )?;
        let field = coefficients.transpose_matmul(&response.perturbation.matmul(coefficients));
        let gaps = determinants.iter().map(|s| s.gap).collect::<Vec<_>>();
        let induction = truncated_induction::monomer(
            &space,
            &ham.g,
            &field,
            orbital_energies,
            &gaps,
            &ground[1],
            &ground[0],
        )?;
        let norm2 = 0.5 * dot(&ground[1], &ground[1]);
        ground[3][0] = -dot(&ground[1], &ground[2]);
        let [_, first, _, third] = &mut ground;
        add(third, first, -norm2);
        ground[2][0] = -norm2;
        let [mut density_second, beta] = space.rdm1_spin(&ground[1], &ground[1])?;
        add(density_second.as_mut_slice(), beta.as_slice(), 1.0);
        for spin in space.rdm1_spin(&ground[0], &ground[2])? {
            add(density_second.as_mut_slice(), spin.as_slice(), 1.0);
            add(
                density_second.as_mut_slice(),
                spin.transpose().as_slice(),
                1.0,
            );
        }
        Ok(Self {
            determinants,
            strings,
            fluctuation,
            energies,
            ground,
            relaxed_electrostatics,
            induction,
            density_second,
        })
    }
}

/// Derivatives of MP2 and MP3 monomer correlation energies in the
/// electrostatic field. The orbital gauge follows only occupied-virtual
/// rotations; the resolvent applies the full occupied and virtual Fock
/// blocks, including their off-diagonal field derivatives.
fn electrostatic_response(
    response: &ResponseMonomer<'_>,
    space: &CiSpace,
    ham: &ActiveHamiltonian,
    fluctuation: &Matrix,
    determinants: &[Determinant],
    first: &[f64],
    reference: &[f64],
) -> Result<[f64; 2]> {
    let n = ham.n;
    let no = response.scf.n_alpha;
    let c = &response.scf.mo_coeff[0];
    let ov = orbital_response(response, 0);
    let mut rotation = Matrix::zeros(n, n);
    for a in no..n {
        for i in 0..no {
            rotation[(a, i)] = ov[(a - no, i)];
            rotation[(i, a)] = -ov[(a - no, i)];
        }
    }
    let to_mo = |m: &Matrix| c.transpose_matmul(&m.matmul(c));
    let field = to_mo(response.perturbation);
    let [density_response, _] = density_fock(response)?;
    let mut fock = to_mo(&density_response);
    add(fock.as_mut_slice(), field.as_slice(), 1.0);
    // Brillouin stationarity holds at each field in this moving frame.
    for a in no..n {
        for i in 0..no {
            fock[(a, i)] = 0.0;
            fock[(i, a)] = 0.0;
        }
    }
    let fock_ref = 2.0 * (0..no).map(|i| fock[(i, i)]).sum::<f64>();
    let df = ActiveHamiltonian {
        n,
        h: fock,
        g: vec![0.0; ham.g.len()],
        constant: -fock_ref,
    };
    let mut h = field;
    let left = ham.h.matmul(&rotation);
    let right = rotation.matmul(&ham.h);
    add(h.as_mut_slice(), left.as_slice(), 1.0);
    add(h.as_mut_slice(), right.as_slice(), -1.0);
    let index = |orb: [usize; 4]| orb.into_iter().fold(0, |offset, i| offset * n + i);
    let mut g = vec![0.0; ham.g.len()];
    for (k, value) in g.iter_mut().enumerate() {
        let orbitals = [k / n.pow(3), (k / n.pow(2)) % n, (k / n) % n, k % n];
        for (axis, &p) in orbitals.iter().enumerate() {
            let mut rotated = orbitals;
            for q in 0..n {
                rotated[axis] = q;
                *value += rotation[(q, p)] * ham.g[index(rotated)];
            }
        }
    }
    let dh = ActiveHamiltonian {
        n,
        h,
        g,
        constant: 0.0,
    };

    // E2 = -b^T G^-1 b and E3 = t^T (W - W00) t, with
    // t = -G^-1 b and b = (1 - |0><0|) W |0>.
    let b_first = space.sigma(&dh, reference)?;
    let f_first = space.sigma(&df, first)?;
    let second = 2.0 * dot(first, &b_first) + dot(first, &f_first);
    let mut t_first = vec![0.0; first.len()];
    for i in 1..first.len() {
        t_first[i] = -(b_first[i] + f_first[i]) / determinants[i].gap;
    }
    let first_h = space.sigma(&dh, first)?;
    let third = 2.0 * dot(&t_first, &apply(fluctuation, first)) + dot(first, &first_h)
        - dot(first, &f_first)
        - b_first[0] * dot(first, first);
    Ok([second, third])
}

fn divide<const N: usize>(numerator: [f64; N], denominator: &[f64; N]) -> [f64; N] {
    let mut quotient = [0.0; N];
    for n in 0..N {
        let rest: f64 = (1..=n).map(|i| denominator[i] * quotient[n - i]).sum();
        quotient[n] = (numerator[n] - rest) / denominator[0];
    }
    quotient
}

fn product_ground(a: &Monomer, b: &Monomer) -> [Matrix; 4] {
    std::array::from_fn(|order| {
        let mut result = Matrix::zeros(a.determinants.len(), b.determinants.len());
        for i in 0..=order {
            let term = outer(&a.ground[i], &b.ground[order - i]);
            add(result.as_mut_slice(), term.as_slice(), 1.0);
        }
        result
    })
}

fn project(monomer: &Monomer, order: usize, excited: bool, left: bool, field: &Matrix) -> Matrix {
    // The ground sector is the outer product of two monomer vectors.
    // Evaluate its Taylor coefficients as low-rank contractions.
    let mut result = if excited && order == 0 {
        field.clone()
    } else {
        Matrix::zeros(field.rows, field.cols)
    };
    let factor = if excited { -1.0 } else { 1.0 };
    for i in 0..=order {
        let g = &monomer.ground[i];
        let h = &monomer.ground[order - i];
        if left {
            let mut row = vec![0.0; field.cols];
            for (&c, source) in h.iter().zip(field.as_slice().chunks_exact(field.cols)) {
                if c != 0.0 {
                    add(&mut row, source, c);
                }
            }
            for (&c, target) in g
                .iter()
                .zip(result.as_mut_slice().chunks_exact_mut(field.cols))
            {
                if c != 0.0 {
                    add(target, &row, factor * c);
                }
            }
        } else {
            for (source, target) in field
                .as_slice()
                .chunks_exact(field.cols)
                .zip(result.as_mut_slice().chunks_exact_mut(field.cols))
            {
                let c = dot(source, g);
                if c != 0.0 {
                    add(target, h, factor * c);
                }
            }
        }
    }
    result
}

fn sector_source(field: &[Matrix; 4], a: &Monomer, b: &Monomer, excited: [bool; 2]) -> [Matrix; 3] {
    std::array::from_fn(|n| {
        let mut source = Matrix::zeros(field[0].rows, field[0].cols);
        for i in 0..=n {
            for (j, f) in field.iter().enumerate().take(n - i + 1) {
                let left = project(a, i, excited[0], true, f);
                let term = project(b, n - i - j, excited[1], false, &left);
                add(source.as_mut_slice(), term.as_slice(), 1.0);
            }
        }
        source
    })
}

fn solve(source: &[Matrix; 3], product: &[Matrix; 4], a: &Monomer, b: &Monomer) -> [Matrix; 3] {
    let [na, nb] = [a.determinants.len(), b.determinants.len()];
    let mut solution: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(na, nb));
    for n in 0..3 {
        let mut rhs = source[n].clone();
        if n > 0 {
            let prior = &solution[n - 1];
            add(
                rhs.as_mut_slice(),
                a.fluctuation.matmul(prior).as_slice(),
                1.0,
            );
            add(
                rhs.as_mut_slice(),
                prior.matmul(&b.fluctuation).as_slice(),
                1.0,
            );
        }
        for order in 1..=n {
            let prior = &solution[n - order];
            let energy = a.energies[order] + b.energies[order];
            add(rhs.as_mut_slice(), prior.as_slice(), -energy);
            // The rank-one ground projector removes the resolvent zero mode.
            for i in 0..=order {
                add(
                    rhs.as_mut_slice(),
                    product[i].as_slice(),
                    product[order - i].frobenius_dot(prior),
                );
            }
        }
        for i in 0..na {
            for j in 0..nb {
                let gap = a.determinants[i].gap + b.determinants[j].gap;
                solution[n][(i, j)] = -rhs[(i, j)] / if i == 0 && j == 0 { 1.0 } else { gap };
            }
        }
    }
    solution
}

pub(super) fn terms(
    interaction: &Interaction<'_>,
    responses: [&ResponseMonomer<'_>; 2],
    hcore: [&Matrix; 2],
    coupled_induction_exchange: [f64; 2],
    cubic: bool,
    limit_mb: usize,
) -> Result<(SaptCorrelationTerms, Option<SaptThirdOrder>)> {
    let monomers = responses.map(|r| r.scf);
    let nmos = monomers.map(|s| s.mo_coeff[0].cols);
    let dims = monomers.map(|s| spatial_strings(s.mo_coeff[0].cols, s.n_alpha).saturating_pow(2));
    let product_size = dims[0].saturating_mul(dims[1]);
    let eri_size = interaction
        .overlap
        .rows
        .saturating_pow(4)
        .saturating_add(nmos.iter().sum::<usize>().saturating_pow(4));
    let occupied = monomers.iter().map(|m| m.n_alpha).sum::<usize>();
    let strings = monomers
        .iter()
        .map(|m| spatial_strings(m.mo_coeff[0].cols, m.n_alpha))
        .fold(1, usize::saturating_mul);
    let spin_bytes = spin_bytes(strings, occupied);
    let monomer_bytes = dims
        .into_iter()
        .map(|d| d.saturating_pow(2).saturating_mul(8))
        .fold(0, usize::saturating_add);
    let work = spin_bytes
        .saturating_add(product_size.saturating_mul(vector_bytes()))
        .saturating_add(monomer_bytes)
        .saturating_add(eri_size.saturating_mul(16));
    if work > limit_mb.saturating_mul(1024 * 1024) {
        return Err(DftError::InvalidInput(format!(
            "fixed-orbital SAPT correlation spin workspace exceeds the {limit_mb} MiB integral limit"
        )));
    }
    let (operators, spectrum) = third_order::prepare(interaction, monomers, limit_mb)?;
    let third = cubic
        .then(|| third_order::prepared_terms(&operators, &spectrum).map(|t| t.third))
        .transpose()?;
    drop(spectrum);
    let a = Monomer::new(responses[0], hcore[0], 0, &operators)?;
    let b = Monomer::new(responses[1], hcore[1], nmos[0], &operators)?;
    let ground = product_ground(&a, &b);
    let dets = [a.determinants.as_slice(), b.determinants.as_slice()];
    let field = bare_actions(&operators, dets, &ground);
    let bare_first: [f64; 4] = std::array::from_fn(|n| {
        (0..=n)
            .map(|i| ground[n - i].frobenius_dot(&field[i]))
            .sum()
    });
    let mut second_order_polarization = [[0.0; 2]; 3];
    let sectors = [[true, false], [false, true], [true, true]];
    let responses = std::array::from_fn(|sector| {
        let source = sector_source(&field, &a, &b, sectors[sector]);
        let response = solve(&source, &ground, &a, &b);
        second_order_polarization[sector] = [1, 2].map(|n| {
            (0..=n)
                .map(|i| source[i].frobenius_dot(&response[n - i]))
                .sum()
        });
        response
    });
    drop(field);
    let exchange = exchange_expectations(
        &operators,
        dets,
        [a.strings, b.strings],
        nmos.iter().sum(),
        &ground,
        &responses,
    )?;
    let overlap = exchange.first.map(|row| row[1]);
    if !overlap[0].is_finite() || overlap[0] <= 1e-12 {
        return Err(DftError::InvalidInput(
            "fixed-orbital SAPT correlation has a singular antisymmetrizer overlap".into(),
        ));
    }
    let first = divide(exchange.first.map(|row| row[0]), &overlap);
    let second_order_exchange = std::array::from_fn(|sector| {
        let terms = &exchange.second[sector];
        let numerator: [f64; 3] = std::array::from_fn(|n| {
            terms[n][0] - (0..=n).map(|i| first[i] * terms[n - i][1]).sum::<f64>()
        });
        let full = divide(numerator, &[overlap[0], overlap[1], overlap[2]]);
        [1, 2].map(|n| full[n] - second_order_polarization[sector][n - 1])
    });
    // The correlated permanent density of the other monomer changes the
    // bare second-order induction by its cross contraction with the
    // uncoupled first-order response (Eq. 58).
    let monomer_data = [&a, &b];
    let offsets = [0, nmos[0]];
    let truncated_induction22 = [0, 1].map(|m| {
        let other = 1 - m;
        let rotation = &monomer_data[m].induction.rotation;
        let density = &monomer_data[other].density_second;
        let mut induced = monomer_data[m].induction.induction;
        for r in 0..rotation.rows {
            for i in 0..rotation.cols {
                let p = r + rotation.cols + offsets[m];
                let q = i + offsets[m];
                for k in 0..density.rows {
                    for l in 0..density.cols {
                        induced += 4.0
                            * rotation[(r, i)]
                            * density[(k, l)]
                            * operators.eri(p, q, k + offsets[other], l + offsets[other]);
                    }
                }
            }
        }
        induced
    });
    let [ind, exchange] = coupled_induction_exchange;
    let truncated_exchange_induction22 = if exchange == 0.0 {
        0.0
    } else if ind != 0.0 {
        truncated_induction22.iter().sum::<f64>() * exchange / ind
    } else {
        return Err(DftError::InvalidInput(
            "truncated exchange-induction ratio has zero coupled induction".into(),
        ));
    };
    let correlation = SaptCorrelationTerms {
        electrostatics: [bare_first[1], bare_first[2], bare_first[3]],
        exchange: std::array::from_fn(|i| first[i + 1] - bare_first[i + 1]),
        relaxed_electrostatics: [a.relaxed_electrostatics, b.relaxed_electrostatics],
        second_order_polarization,
        second_order_exchange,
        truncated_induction22,
        truncated_exchange_induction22,
    };
    Ok((correlation, third))
}
