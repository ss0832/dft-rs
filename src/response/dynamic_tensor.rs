// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Frequency-domain density-projector response. Distinct labelled field legs
//! form subset jets, so every coefficient is an actual mixed derivative.
//! i dD/dt=[F,D] and D^2=D determine all orders without finite fields.

use super::complex::Complex;
use super::krylov::{self, LinearOperator, OperatorStructure};
use super::{Frequency, ResponseReport, ResponseSolver, ResponseSolverConfig};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::integrals::dipole_matrices;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct DynamicElectricTensor {
    pub process: Option<super::ElectricProcess>,
    pub input_frequencies: Vec<Frequency>,
    /// Output Fourier frequency is the sum of the input complex frequencies.
    pub output_frequency: Complex,
    /// Row-major actual derivatives, observable axis followed by field axes.
    pub tensor: Vec<Complex>,
    pub projector_defect: f64,
    pub equation_defect: f64,
    pub particle_number_defect: f64,
    /// Intrinsic interchange is measured only between equal-frequency slots.
    pub intrinsic_permutation_defect: f64,
    pub intrinsic_permutation_comparisons: usize,
    pub subset_solves: usize,
    pub nonlinear_kernel_evaluations: usize,
}

#[derive(Clone)]
pub(crate) struct CMatrix {
    pub(crate) re: Matrix,
    pub(crate) im: Matrix,
}
impl CMatrix {
    pub(crate) fn zero(n: usize) -> Self {
        Self {
            re: Matrix::zeros(n, n),
            im: Matrix::zeros(n, n),
        }
    }
    pub(crate) fn real(re: Matrix) -> Self {
        let im = Matrix::zeros(re.rows, re.cols);
        Self { re, im }
    }
    fn get(&self, i: usize, j: usize) -> Complex {
        Complex::new(self.re[(i, j)], self.im[(i, j)])
    }
    fn set(&mut self, i: usize, j: usize, z: Complex) {
        self.re[(i, j)] = z.re;
        self.im[(i, j)] = z.im;
    }
    pub(crate) fn add(&mut self, b: &Self, weight: Complex) {
        for i in 0..self.re.rows {
            for j in 0..self.re.cols {
                self.set(i, j, self.get(i, j) + weight * b.get(i, j));
            }
        }
    }
    pub(crate) fn product(&self, b: &Self) -> Self {
        let mut re = self.re.matmul_seq(&b.re);
        let mut im = self.re.matmul_seq(&b.im);
        for (a, b) in re
            .as_mut_slice()
            .iter_mut()
            .zip(self.im.matmul_seq(&b.im).as_slice())
        {
            *a -= b;
        }
        for (a, b) in im
            .as_mut_slice()
            .iter_mut()
            .zip(self.im.matmul_seq(&b.re).as_slice())
        {
            *a += b;
        }
        Self { re, im }
    }
    fn commutator(&self, b: &Self) -> Self {
        let mut c = self.product(b);
        c.add(&b.product(self), (-1.0).into());
        c
    }
    pub(crate) fn transform(&self, left: &Matrix, right: &Matrix) -> Self {
        Self {
            re: left.matmul_seq(&self.re).matmul_seq(right),
            im: left.matmul_seq(&self.im).matmul_seq(right),
        }
    }
    pub(crate) fn max_abs(&self) -> f64 {
        self.re
            .as_slice()
            .iter()
            .zip(self.im.as_slice())
            .fold(0.0_f64, |m, (r, i)| {
                if r.is_finite() && i.is_finite() {
                    m.max(r.hypot(*i))
                } else {
                    f64::INFINITY
                }
            })
    }
    fn trace(&self) -> Complex {
        (0..self.re.rows).fold(Complex::ZERO, |v, i| v + self.get(i, i))
    }
    pub(crate) fn contract(&self, b: &Matrix) -> Complex {
        Complex::new(self.re.frobenius_dot(b), self.im.frobenius_dot(b))
    }
}
fn symmetric(m: &Matrix) -> Matrix {
    let mut s = m.clone();
    for i in 0..m.rows {
        for j in 0..m.cols {
            s[(i, j)] = 0.5 * (m[(i, j)] + m[(j, i)]);
        }
    }
    s
}

// Local density/gradient/tau, nonlocal density and PCM see the transpose-even
// part. Exchange also acts on the transpose-odd part; it must not be dropped.
fn real_linear(solver: &ResponseSolver<'_>, p: &[Matrix; 2]) -> Result<[Matrix; 2]> {
    if p.iter().all(|m| m.as_slice().iter().all(|x| *x == 0.0)) {
        let n = solver.mb.nao_sph;
        return Ok([Matrix::zeros(n, n), Matrix::zeros(n, n)]);
    }
    let sym: [Matrix; 2] = std::array::from_fn(|s| symmetric(&p[s]));
    let mut out = solver.fock_response(&sym)?;
    let anti: [Matrix; 2] = std::array::from_fn(|s| {
        let mut a = p[s].clone();
        for (v, w) in a.as_mut_slice().iter_mut().zip(sym[s].as_slice()) {
            *v -= w;
        }
        a
    });
    if anti.iter().all(|a| a.as_slice().iter().all(|x| *x == 0.0)) {
        return Ok(out);
    }
    for (engine, coefficient) in [
        (Some(&solver.jk), solver.functional.hybrid.cx),
        (solver.jk_lr.as_ref(), solver.functional.hybrid.clr),
    ] {
        if coefficient != 0.0 {
            let k = engine
                .expect("nonzero exchange has an engine")
                .build_k_only(solver.mb, [&anti[0], &anti[1]])?;
            for s in 0..2 {
                for (v, w) in out[s].as_mut_slice().iter_mut().zip(k[s].as_slice()) {
                    *v -= coefficient * w;
                }
            }
        }
    }
    Ok(out)
}
pub(crate) fn linear(solver: &ResponseSolver<'_>, p: &[CMatrix; 2]) -> Result<[CMatrix; 2]> {
    let re = real_linear(solver, &std::array::from_fn(|s| p[s].re.clone()))?;
    let im = real_linear(solver, &std::array::from_fn(|s| p[s].im.clone()))?;
    Ok(std::array::from_fn(|s| CMatrix {
        re: re[s].clone(),
        im: im[s].clone(),
    }))
}

struct PairedOperator<'a, 'b> {
    solver: &'b ResponseSolver<'a>,
    scf: &'a ScfResult,
    z: Complex,
}
impl PairedOperator<'_, '_> {
    fn pairs(&self) -> Vec<(usize, usize, usize)> {
        let mut pairs = Vec::new();
        for s in 0..2 {
            for a in [self.scf.n_alpha, self.scf.n_beta][s]..self.scf.mo_coeff[s].cols {
                for i in 0..[self.scf.n_alpha, self.scf.n_beta][s] {
                    pairs.push((s, a, i));
                }
            }
        }
        pairs
    }
    fn density(&self, x: &[Complex]) -> [CMatrix; 2] {
        let pairs = self.pairs();
        let n = pairs.len();
        let mut d = std::array::from_fn(|s| CMatrix::zero(self.scf.mo_coeff[s].cols));
        for (k, &(s, a, i)) in pairs.iter().enumerate() {
            d[s].set(a, i, x[k] + x[n + k]);
            d[s].set(i, a, x[k] - x[n + k]);
        }
        d
    }
}
impl LinearOperator for PairedOperator<'_, '_> {
    fn dimension(&self) -> usize {
        2 * self.pairs().len()
    }
    fn structure(&self) -> OperatorStructure {
        if self.z.im == 0.0 {
            OperatorStructure::Hermitian
        } else {
            OperatorStructure::General
        }
    }
    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        if x.len() != self.dimension() {
            return Err(DftError::InvalidInput(
                "invalid paired density amplitude dimension".into(),
            ));
        }
        let d = self.density(x);
        let ao = std::array::from_fn(|s| {
            d[s].transform(&self.scf.mo_coeff[s], &self.scf.mo_coeff[s].transpose())
        });
        let f = linear(self.solver, &ao)?;
        let f: [CMatrix; 2] = std::array::from_fn(|s| {
            f[s].transform(&self.scf.mo_coeff[s].transpose(), &self.scf.mo_coeff[s])
        });
        let pairs = self.pairs();
        let n = pairs.len();
        let mut out = vec![Complex::ZERO; 2 * n];
        for (k, &(s, a, i)) in pairs.iter().enumerate() {
            let gap = self.scf.mo_energy[s][a] - self.scf.mo_energy[s][i];
            out[k] = x[k] * gap + (f[s].get(a, i) + f[s].get(i, a)) / 2.0 - self.z * x[n + k];
            out[n + k] = x[n + k] * gap + (f[s].get(a, i) - f[s].get(i, a)) / 2.0 - self.z * x[k];
        }
        Ok(out)
    }
}

// Extend the existing real analytic Taylor kernels complex-multilinearly.
// Polarization is an exact coefficient identity, with no displacement h.
fn nonlinear_kernel(
    solver: &ResponseSolver<'_>,
    args: &[[CMatrix; 2]],
    calls: &mut usize,
) -> Result<[CMatrix; 2]> {
    let nao = solver.mb.nao_sph;
    let width = nao * nao;
    let mut out = [CMatrix::zero(nao), CMatrix::zero(nao)];
    if solver.functional.parts.is_empty() && solver.nlc.is_none() {
        return Ok(out);
    }
    for choice in 0..1usize << args.len() {
        let directions: Vec<Vec<f64>> = args
            .iter()
            .enumerate()
            .map(|(leg, a)| {
                a.iter()
                    .flat_map(|d| {
                        symmetric(if choice & (1 << leg) == 0 {
                            &d.re
                        } else {
                            &d.im
                        })
                        .as_slice()
                        .to_vec()
                    })
                    .collect()
            })
            .collect();
        if directions.iter().any(|d| d.iter().all(|x| *x == 0.0)) {
            continue;
        }
        let coefficient = super::mixed::polarized_mixed(2 * width, &directions, |d, order| {
            let dp: [Matrix; 2] = std::array::from_fn(|s| {
                Matrix::from_row_major(nao, nao, d[s * width..(s + 1) * width].to_vec())
            });
            let zero = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
            let mut densities = vec![solver.scf.density.clone(), dp.clone()];
            densities.resize(order + 1, zero.clone());
            let mut f = zero;
            if !solver.functional.parts.is_empty() {
                f = super::nonlinear::semilocal_fock_coefficient(
                    solver.mb,
                    solver.grid.expect("response grid checked"),
                    solver.functional,
                    &densities,
                    order,
                )?;
            }
            if let Some((params, grid, p0)) = &solver.nlc {
                let mut total = dp[0].clone();
                for (v, w) in total.as_mut_slice().iter_mut().zip(dp[1].as_slice()) {
                    *v += w;
                }
                let mut path = vec![p0.clone(), total];
                path.resize(order + 1, Matrix::zeros(nao, nao));
                let nlc = super::nonlocal::vv10_taylor(solver.mb, grid, &path, *params)?;
                for s in 0..2 {
                    for (v, w) in f[s]
                        .as_mut_slice()
                        .iter_mut()
                        .zip(nlc.fock_coefficients[order].as_slice())
                    {
                        *v += w;
                    }
                }
            }
            Ok(f.iter().flat_map(|m| m.as_slice().to_vec()).collect())
        })?;
        *calls += coefficient.directional_evaluations;
        let weight = match choice.count_ones() % 4 {
            0 => Complex::ONE,
            1 => Complex::new(0.0, 1.0),
            2 => (-1.0).into(),
            _ => Complex::new(0.0, -1.0),
        };
        for s in 0..2 {
            out[s].add(
                &CMatrix::real(Matrix::from_row_major(
                    nao,
                    nao,
                    coefficient.value[s * width..(s + 1) * width].to_vec(),
                )),
                weight,
            );
        }
    }
    Ok(out)
}

pub(crate) struct OneBodyFrequencyPath {
    /// Subset bit mask, then spin. Mask zero is the stationary reference.
    pub(crate) density_mo: Vec<[CMatrix; 2]>,
    pub(crate) density_ao: Vec<[CMatrix; 2]>,
    pub(crate) fock_mo: Vec<[CMatrix; 2]>,
    pub(crate) residuals: Vec<f64>,
    pub(crate) iterations: Vec<usize>,
    pub(crate) defects: Vec<OneBodyFrequencyDefects>,
    kernel_calls: usize,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct OneBodyFrequencyDefects {
    pub(crate) projector: f64,
    pub(crate) equation: f64,
    pub(crate) number: f64,
}

/// Complex density and Fock harmonics for labelled scalar AO operators.
/// Internal frequencies may include both z and -z for analytic continuation.
pub(crate) fn solve_one_body_path(
    solver: &ResponseSolver<'_>,
    sources: &[&Matrix],
    frequencies: &[Complex],
    config: &ResponseSolverConfig,
) -> Result<OneBodyFrequencyPath> {
    let nao = solver.mb.nao_sph;
    if !(1..=3).contains(&sources.len())
        || frequencies.len() != sources.len()
        || frequencies.iter().any(|z| !z.is_finite())
        || sources
            .iter()
            .any(|source| source.rows != nao || source.cols != nao)
    {
        return Err(DftError::InvalidInput(
            "frequency one-body path requires one to three AO source/frequency pairs".into(),
        ));
    }
    let scf = solver.scf;
    let count = 1usize << sources.len();
    let zero: [CMatrix; 2] = std::array::from_fn(|s| CMatrix::zero(scf.mo_coeff[s].cols));
    let mut densities = vec![zero.clone(); count];
    let mut focks = vec![zero.clone(); count];
    let mut aos = vec![
        [
            CMatrix::zero(solver.mb.nao_sph),
            CMatrix::zero(solver.mb.nao_sph)
        ];
        count
    ];
    for s in 0..2 {
        for i in 0..[scf.n_alpha, scf.n_beta][s] {
            densities[0][s].set(i, i, Complex::ONE);
        }
        focks[0][s] = CMatrix::real(
            scf.mo_coeff[s]
                .transpose()
                .matmul_seq(&scf.fock[s])
                .matmul_seq(&scf.mo_coeff[s]),
        );
        aos[0][s] = CMatrix::real(scf.density[s].clone());
    }
    let mut result = OneBodyFrequencyPath {
        density_mo: Vec::new(),
        density_ao: Vec::new(),
        fock_mo: Vec::new(),
        residuals: Vec::new(),
        iterations: Vec::new(),
        defects: vec![OneBodyFrequencyDefects::default(); count],
        kernel_calls: 0,
    };
    for order in 1..=sources.len() {
        for mask in 1..count {
            if mask.count_ones() as usize != order {
                continue;
            }
            let z = frequencies
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .fold(Complex::ZERO, |v, (_, z)| v + *z);
            let operator = PairedOperator { solver, scf, z };
            let mut products = zero.clone();
            let mut comm = zero.clone();
            for a in 1..mask {
                if a & mask != a {
                    continue;
                }
                let b = mask ^ a;
                for s in 0..2 {
                    products[s].add(&densities[a][s].product(&densities[b][s]), Complex::ONE);
                    comm[s].add(&focks[a][s].commutator(&densities[b][s]), Complex::ONE);
                }
            }
            let mut dn = zero.clone();
            for s in 0..2 {
                let no = [scf.n_alpha, scf.n_beta][s];
                for i in 0..dn[s].re.rows {
                    for j in 0..dn[s].re.cols {
                        if (i < no) == (j < no) {
                            dn[s].set(
                                i,
                                j,
                                products[s].get(i, j) * if i < no { -1.0 } else { 1.0 },
                            );
                        }
                    }
                }
            }
            let diag_ao = std::array::from_fn(|s| {
                dn[s].transform(&scf.mo_coeff[s], &scf.mo_coeff[s].transpose())
            });
            let mut fn_ao = linear(solver, &diag_ao)?;
            if order == 1 {
                let leg = mask.trailing_zeros() as usize;
                let source = CMatrix::real(sources[leg].clone());
                for fock in &mut fn_ao {
                    fock.add(&source, Complex::ONE);
                }
            }
            for a in 1..mask {
                if a & mask != a {
                    continue;
                }
                let b = mask ^ a;
                if a >= b {
                    continue;
                }
                let g = nonlinear_kernel(
                    solver,
                    &[aos[a].clone(), aos[b].clone()],
                    &mut result.kernel_calls,
                )?;
                for s in 0..2 {
                    fn_ao[s].add(&g[s], Complex::ONE);
                }
            }
            if order == 3 {
                let singles: Vec<_> = (0..sources.len())
                    .filter(|i| mask & (1 << i) != 0)
                    .map(|i| aos[1 << i].clone())
                    .collect();
                let h = nonlinear_kernel(solver, &singles, &mut result.kernel_calls)?;
                for s in 0..2 {
                    fn_ao[s].add(&h[s], Complex::ONE);
                }
            }
            let mut fn_mo: [CMatrix; 2] = std::array::from_fn(|s| {
                fn_ao[s].transform(&scf.mo_coeff[s].transpose(), &scf.mo_coeff[s])
            });
            let mut source = comm.clone();
            for s in 0..2 {
                source[s].add(&focks[0][s].commutator(&dn[s]), Complex::ONE);
                source[s].add(&fn_mo[s].commutator(&densities[0][s]), Complex::ONE);
                source[s].add(&dn[s], -z);
            }
            let pairs = operator.pairs();
            let mut rhs = vec![Complex::ZERO; 2 * pairs.len()];
            for (k, &(s, a, i)) in pairs.iter().enumerate() {
                rhs[k] = -(source[s].get(a, i) - source[s].get(i, a)) / 2.0;
                rhs[pairs.len() + k] = -(source[s].get(a, i) + source[s].get(i, a)) / 2.0;
            }
            let solution = krylov::solve(&operator, &rhs, None, config)?;
            let off = operator.density(&solution.amplitudes);
            let off_ao = std::array::from_fn(|s| {
                off[s].transform(&scf.mo_coeff[s], &scf.mo_coeff[s].transpose())
            });
            let df = linear(solver, &off_ao)?;
            for s in 0..2 {
                dn[s].add(&off[s], Complex::ONE);
                aos[mask][s] = diag_ao[s].clone();
                aos[mask][s].add(&off_ao[s], Complex::ONE);
                fn_mo[s].add(
                    &df[s].transform(&scf.mo_coeff[s].transpose(), &scf.mo_coeff[s]),
                    Complex::ONE,
                );
                let mut projector = products[s].clone();
                projector.add(&densities[0][s].product(&dn[s]), Complex::ONE);
                projector.add(&dn[s].product(&densities[0][s]), Complex::ONE);
                projector.add(&dn[s], (-1.0).into());
                let defects = &mut result.defects[mask];
                defects.projector = defects.projector.max(projector.max_abs());
                let mut equation = comm[s].clone();
                equation.add(&focks[0][s].commutator(&dn[s]), Complex::ONE);
                equation.add(&fn_mo[s].commutator(&densities[0][s]), Complex::ONE);
                equation.add(&dn[s], -z);
                defects.equation = defects.equation.max(equation.max_abs());
                defects.number = defects.number.max(dn[s].trace().abs());
            }
            densities[mask] = dn;
            focks[mask] = fn_mo;
            result.residuals.push(solution.relative_residual);
            result.iterations.push(solution.iterations);
        }
    }
    result.density_mo = densities;
    result.density_ao = aos;
    result.fock_mo = focks;
    Ok(result)
}

/// Complete alpha/beta/gamma at independently specified incoming frequencies.
/// Harmonic generation uses repeated frequencies; signed real frequencies
/// select sum/difference-frequency, rectification, Kerr and mixing processes.
#[allow(clippy::too_many_arguments)]
pub fn dynamic_electric_tensor(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    frequencies: &[Frequency],
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<DynamicElectricTensor>> {
    config.validate()?;
    if !(1..=3).contains(&frequencies.len()) {
        return Err(DftError::InvalidInput(
            "dynamic electric tensor requires one, two or three input frequencies".into(),
        ));
    }
    if !scf.converged {
        return Err(DftError::InvalidInput(
            "dynamic electric tensor requires a converged SCF reference".into(),
        ));
    }
    let zs: Vec<_> = frequencies
        .iter()
        .map(|f| f.energy())
        .collect::<Result<_>>()?;
    if functional.required_nonlocal_kind() == Some(crate::xc::RequiredNonlocalTerm::Vv10)
        && opts.vv10.is_none()
    {
        return Err(DftError::InvalidInput(
            "the functional requires self-consistent VV10 options for dynamic response".into(),
        ));
    }
    let solver = ResponseSolver::new(mol, mb, functional, grid, scf, opts)?;
    let dipoles = dipole_matrices(mb, Vec3::zero());
    let order = frequencies.len();
    let entries = 3usize.pow(order as u32);
    let mut tensor = vec![Complex::ZERO; 3 * entries];
    let mut residuals = Vec::new();
    let mut iterations = Vec::new();
    let mut projector = 0.0_f64;
    let mut equation = 0.0_f64;
    let mut number = 0.0_f64;
    let mut kernel_calls = 0;
    for flat in 0..entries {
        let mut code = flat;
        let mut sources = vec![&dipoles[0]; order];
        for leg in (0..order).rev() {
            sources[leg] = &dipoles[code % 3];
            code /= 3;
        }
        let d = solve_one_body_path(&solver, &sources, &zs, config)?;
        let density = d.density_ao.last().expect("nonempty frequency subset");
        for a in 0..3 {
            tensor[a * entries + flat] =
                -(density[0].contract(&dipoles[a]) + density[1].contract(&dipoles[a]));
        }
        residuals.extend(d.residuals);
        iterations.extend(d.iterations);
        for defects in d.defects {
            projector = projector.max(defects.projector);
            equation = equation.max(defects.equation);
            number = number.max(defects.number);
        }
        kernel_calls += d.kernel_calls;
    }
    let mut defect = 0.0_f64;
    let mut comparisons = 0;
    for flat in 0..tensor.len() {
        let mut code = flat;
        let mut axes = vec![0; order + 1];
        for a in axes.iter_mut().rev() {
            *a = code % 3;
            code /= 3;
        }
        for b in 0..order {
            for c in b + 1..order {
                if zs[b] != zs[c] || axes[b + 1] == axes[c + 1] {
                    continue;
                }
                let mut swapped = axes.clone();
                swapped.swap(b + 1, c + 1);
                let other = swapped.iter().fold(0, |v, a| 3 * v + a);
                if flat < other {
                    comparisons += 1;
                    defect = defect.max((tensor[flat] - tensor[other]).abs());
                }
            }
        }
    }
    if tensor.iter().any(|z| !z.is_finite()) {
        return Err(DftError::LinearAlgebra(
            "non-finite dynamic electric tensor".into(),
        ));
    }
    let scale = tensor.iter().fold(1.0_f64, |m, z| m.max(z.abs()));
    let reference = super::diagnostics::reference_diagnostics(scf)?;
    let mut warnings = Vec::new();
    if defect > config.tolerance * scale {
        warnings.push("raw equal-frequency intrinsic permutation defect exceeds tolerance".into());
    }
    if equation > config.tolerance * scale + reference.orbital_residual * scale {
        warnings.push("full density equation residual exceeds tolerance".into());
    }
    if projector > config.tolerance * scale || number > config.tolerance * scale {
        warnings.push("density projector or particle-number constraint exceeds tolerance".into());
    }
    Ok(ResponseReport {
        value: DynamicElectricTensor {
            process: None,
            input_frequencies: frequencies.to_vec(),
            output_frequency: zs.iter().fold(Complex::ZERO, |a, b| a + *b),
            tensor,
            projector_defect: projector,
            equation_defect: equation,
            particle_number_defect: number,
            intrinsic_permutation_defect: defect,
            intrinsic_permutation_comparisons: comparisons,
            subset_solves: residuals.len(),
            nonlinear_kernel_evaluations: kernel_calls,
        },
        shape: vec![3; order + 1],
        functional: functional.description.clone(),
        reference,
        units: match order {
            1 => "bohr^3",
            2 => "atomic units; e^3 Bohr^3/Hartree^2",
            _ => "atomic units; e^4 Bohr^4/Hartree^3",
        },
        model: "clamped-nuclei integer-occupation adiabatic KS/HF frequency-domain \
            density-projector response; labelled complex frequency subsets; \
            analytic nonlinear kernels",
        axis_convention: "raw row-major [dipole axis][ordered input field axes]; actual mixed \
            derivatives; Fourier exp(-i z t); output z=sum(input z); no Kleinman \
            symmetrization",
        converged: warnings.is_empty(),
        residuals,
        iterations,
        warnings,
    })
}

/// Full quadratic dipole response beta(-z1-z2; z1, z2).
#[allow(clippy::too_many_arguments)]
pub fn dynamic_beta(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    frequencies: [Frequency; 2],
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<DynamicElectricTensor>> {
    dynamic_electric_tensor(mol, mb, functional, grid, scf, opts, &frequencies, config)
}

/// Full cubic dipole response gamma(-z1-z2-z3; z1, z2, z3).
#[allow(clippy::too_many_arguments)]
pub fn dynamic_gamma(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    frequencies: [Frequency; 3],
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<DynamicElectricTensor>> {
    dynamic_electric_tensor(mol, mb, functional, grid, scf, opts, &frequencies, config)
}
