// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! Restricted/unrestricted frozen-scalar Hessian in the same analytic chart.
//! Explicit integral, grid, Hartree and XC derivatives use the shared
//! fixed-density Hessian. Taylor series differentiate the analytic AO chart
//! and frozen metric. Only active occupied/virtual rotations relax; frozen
//! neutral coefficient spans move and normalize in the declared AO metric.

use super::*;
use crate::hessian::fixed_density::{fixed_density_hessian_terms, FixedDensityHessianTerms};
use crate::integrals::analytic_basis::{overlap, AnalyticShell};
use crate::pseudo_sub::reduction_support::MatrixTaylor;
use crate::response::complex::Complex;
use crate::response::density_kernel::SemilocalDensityKernel;
use crate::response::krylov::{self, LinearOperator, OperatorStructure, ResponseSolverConfig};
use crate::response::nuclear::geometry::GeometryTaylor;
use crate::response::nuclear::metric_transport;
use crate::response::series::{evaluate_series_many_result, SeriesScalar};

pub struct AffineFrozenSecondDerivative {
    pub energy: f64,
    pub first_derivative: f64,
    pub second_derivative: f64,
    /// Fresh unpreconditioned active orbital equation, Hartree/Bohr.
    pub orbital_response_residual: f64,
    pub relative_response_residual: f64,
    pub response_iterations: usize,
    pub gradient_consistency_error: f64,
    /// Curvature with the active determinant fixed in its moving metric frame.
    pub fixed_second_derivative: f64,
    source: Vec<f64>,
    rotation: Vec<f64>,
}

pub struct AffineFrozenHessian {
    /// Cartesian derivatives of the declared raw coordinates, Hartree/Bohr^2.
    pub hessian: Matrix,
    /// Ordered columns before averaging independently solved relaxations.
    pub raw_hessian: Matrix,
    pub raw_asymmetry: f64,
    /// Diagonal directions followed by every i<j pair direction.
    pub directional_results: Vec<AffineFrozenSecondDerivative>,
    pub reference_orbital_residual: f64,
}

fn overlap_series(
    a: &MolBasis,
    b: &MolBasis,
    da: &[[f64; 3]],
    db: &[[f64; 3]],
) -> Result<MatrixTaylor> {
    let mut result = vec![Matrix::zeros(a.nao_sph, b.nao_sph); 3];
    for sa in &a.shells {
        for sb in &b.shells {
            let input = sa
                .center
                .to_array()
                .iter()
                .zip(da[sa.atom])
                .chain(sb.center.to_array().iter().zip(db[sb.atom]))
                .map(|(r, d)| vec![*r, d, 0.])
                .collect::<Vec<_>>();
            let values = evaluate_series_many_result(&input, |v| {
                let left = AnalyticShell::<SeriesScalar>::from_placed(sa, [v[0], v[1], v[2]])?;
                let right = AnalyticShell::<SeriesScalar>::from_placed(sb, [v[3], v[4], v[5]])?;
                Ok(overlap(&left, &right)?.values)
            })?;
            let width = crate::basis::n_sph(sb.l);
            for (index, values) in values.iter().enumerate() {
                for k in 0..3 {
                    result[k][(sa.sph_off + index / width, sb.sph_off + index % width)] = values[k];
                }
            }
        }
    }
    Ok(MatrixTaylor(result))
}

fn occupied_projector(size: usize, occupied: usize, rotation: Option<&[f64]>) -> MatrixTaylor {
    let mut d = vec![Matrix::zeros(size, size); 3];
    for i in 0..occupied {
        d[0][(i, i)] = 1.;
    }
    if let Some(u) = rotation {
        for i in 0..occupied {
            for a in occupied..size {
                d[1][(a, i)] = u[i * (size - occupied) + a - occupied];
                d[1][(i, a)] = d[1][(a, i)];
            }
        }
        let square = d[1].matmul_seq(&d[1]);
        for i in 0..size {
            for j in 0..size {
                if i < occupied && j < occupied {
                    d[2][(i, j)] = -square[(i, j)];
                } else if i >= occupied && j >= occupied {
                    d[2][(i, j)] = square[(i, j)];
                }
            }
        }
    }
    MatrixTaylor(d)
}

struct Frame {
    mapped: [MatrixTaylor; 2],
    frozen: [MatrixTaylor; 2],
    density: [Vec<Matrix>; 2],
}
fn frame(
    model: &CollinearAffineFrozenGeometry<'_>,
    fb: &MolBasis,
    rb: &MolBasis,
    base: &CollinearAffineFrozenGeometryResult,
    direction: &[[f64; 3]],
) -> Result<Frame> {
    let full_direction = model
        .full_atom_owners
        .iter()
        .map(|i| direction[*i])
        .collect::<Vec<_>>();
    let sf = overlap_series(fb, fb, &full_direction, &full_direction)?;
    let sr = overlap_series(rb, rb, direction, direction)?;
    let cross = overlap_series(fb, rb, &full_direction, direction)?;
    let channel = |s: usize| -> Result<(MatrixTaylor, MatrixTaylor, Vec<Matrix>)> {
        let saved = MatrixTaylor::constant(model.frozen_reference[s], 2);
        let frozen = if model.frozen_reference[s].cols == 0 {
            saved
        } else {
            let gram = saved.transpose().product(&sf.product(&saved));
            saved.product(&gram.power(true)?)
        };
        let chart =
            OrthogonalReducedChart::new(&sf.0[0], &cross.0[0], &sr.0[0], &frozen.0[0], 1e-10)?;
        let transform = MatrixTaylor(chart.taylor_transform([&sf.0, &cross.0, &sr.0, &frozen.0])?);
        let orbitals = MatrixTaylor(metric_transport(
            &base.active_scf.active_orbitals[s],
            &sr.0,
            2,
        ));
        let mapped = transform.product(&orbitals);
        let d = occupied_projector(mapped.0[0].cols, model.active_electrons[s], None);
        let density = channel_density(&mapped, &frozen, &d);
        Ok((mapped, frozen, density))
    };
    let (ma, ba, pa) = channel(0)?;
    let (mb, bb, pb) = channel(1)?;
    Ok(Frame {
        mapped: [ma, mb],
        frozen: [ba, bb],
        density: [pa, pb],
    })
}
fn channel_density(
    mapped: &MatrixTaylor,
    frozen: &MatrixTaylor,
    projector: &MatrixTaylor,
) -> Vec<Matrix> {
    let mut p = mapped.product(projector).product(&mapped.transpose());
    let bath = frozen.product(&frozen.transpose());
    for (a, b) in p.0.iter_mut().zip(&bath.0) {
        add(a, b, 1.);
    }
    p.0
}

/// One explicit physical Hessian and one static density kernel are shared by
/// every Cartesian direction. No converged all-atom ScfResult is fabricated.
struct Prepared<'a> {
    base: CollinearAffineFrozenGeometryResult,
    fb: MolBasis,
    rb: MolBasis,
    kernel: SemilocalDensityKernel<'a>,
    fock: [Matrix; 2],
    explicit: FixedDensityHessianTerms,
    explicit_gradient: Vec<Vec3>,
}
impl<'a> Prepared<'a> {
    fn new(
        model: &CollinearAffineFrozenGeometry<'a>,
        coordinates: &[Vec3],
        restricted: bool,
    ) -> Result<Self> {
        let base = model.evaluate_internal(coordinates, None, restricted)?;
        let mut raw = model.raw_reference.clone();
        for (a, r) in raw.atoms.iter_mut().zip(coordinates) {
            a.position = *r;
        }
        let fb = moved_basis(model.full_basis, &base.full_molecule)?;
        let rb = moved_basis(model.raw_basis, &raw)?;
        let grid = MolecularGrid::build(&base.full_molecule, model.grid_options)?;
        let value = frame(model, &fb, &rb, &base, &vec![[0.; 3]; coordinates.len()])?;
        let density = [
            vec![value.density[0][0].clone()],
            vec![value.density[1][0].clone()],
        ];
        let geometry = GeometryTaylor::new(
            &base.full_molecule,
            &fb,
            Some(&grid),
            model.functional,
            None,
            &vec![[0.; 3]; base.full_molecule.len()],
            [0.; 3],
            [0.; 3],
            0,
            crate::resources::component_limit_mb(1024),
        )?;
        let physical = geometry.evaluate(model.functional, &density, 0)?;
        let fock = [physical.fock[0][0].clone(), physical.fock[1][0].clone()];
        drop(geometry);
        let kernel = SemilocalDensityKernel::new(
            &base.full_molecule,
            &fb,
            &grid,
            model.functional,
            [&density[0][0], &density[1][0]],
            model.options,
        )?;
        let explicit = fixed_density_hessian_terms(
            &base.full_molecule,
            &fb,
            model.options,
            model.functional,
            [&density[0][0], &density[1][0]],
            Some(&grid),
        )?;
        let zero = Matrix::zeros(fb.nao_sph, fb.nao_sph);
        let explicit_gradient = crate::gradient::assemble_gradient_density(
            &base.full_molecule,
            &fb,
            model.functional,
            Some(&grid),
            [&density[0][0], &density[1][0]],
            model.options,
            &zero,
        )?;
        Ok(Self {
            base,
            fb,
            rb,
            kernel,
            fock,
            explicit,
            explicit_gradient,
        })
    }
    fn kernel(&self, density: [&Matrix; 2]) -> Result<[Matrix; 2]> {
        self.kernel.apply(&[density[0].clone(), density[1].clone()])
    }
    fn explicit_direction(
        &self,
        model: &CollinearAffineFrozenGeometry<'_>,
        direction: &[[f64; 3]],
    ) -> (f64, f64, [Matrix; 2]) {
        let d = model
            .full_atom_owners
            .iter()
            .flat_map(|i| direction[*i])
            .collect::<Vec<_>>();
        let first = self
            .explicit_gradient
            .iter()
            .flat_map(|g| g.to_array())
            .zip(&d)
            .map(|(g, d)| g * d)
            .sum();
        let mut second = 0.;
        for i in 0..d.len() {
            for j in 0..d.len() {
                second += d[i] * self.explicit.hessian[(i, j)] * d[j];
            }
        }
        let mut fock = [
            Matrix::zeros(self.fb.nao_sph, self.fb.nao_sph),
            Matrix::zeros(self.fb.nao_sph, self.fb.nao_sph),
        ];
        for (spin, f) in fock.iter_mut().enumerate() {
            for (atom, owner) in model.full_atom_owners.iter().enumerate() {
                for axis in 0..3 {
                    add(
                        f,
                        &self.explicit.fock_derivatives[spin][atom][axis],
                        direction[*owner][axis],
                    );
                }
            }
        }
        (first, second, fock)
    }
    /// Exact second chain rule: explicit E_RR + 2 F_R:P_R +
    /// P_R:K:P_R + F:P_RR, with Taylor P[2]=P_RR/2.
    fn curvature(
        &self,
        explicit: f64,
        fock_direction: &[Matrix; 2],
        p: &[Vec<Matrix>; 2],
    ) -> Result<f64> {
        let induced = self.kernel([&p[0][1], &p[1][1]])?;
        let mut sum = AccurateSum::default();
        sum.add_product(1., explicit);
        for spin in 0..2 {
            sum.add_product(2., fock_direction[spin].frobenius_dot(&p[spin][1]));
            sum.add_product(1., induced[spin].frobenius_dot(&p[spin][1]));
            sum.add_product(2., self.fock[spin].frobenius_dot(&p[spin][2]));
        }
        Ok(sum.value())
    }
}

struct ActiveKernel<'a> {
    prepared: &'a Prepared<'a>,
    model: &'a CollinearAffineFrozenGeometry<'a>,
    frame: &'a Frame,
    restricted: bool,
}
impl ActiveKernel<'_> {
    fn ranges(&self) -> [std::ops::Range<usize>; 2] {
        let size = |s: usize| {
            self.model.active_electrons[s]
                * (self.frame.mapped[s].0[0].cols - self.model.active_electrons[s])
        };
        let n = size(0);
        [
            0..n,
            if self.restricted {
                0..n
            } else {
                n..n + size(1)
            },
        ]
    }
    fn real_image(&self, u: &[f64]) -> Result<Vec<f64>> {
        let ranges = self.ranges();
        let q = std::array::from_fn::<_, 2, _>(|s| {
            let mapped = &self.frame.mapped[s].0[0];
            let d = occupied_projector(
                mapped.cols,
                self.model.active_electrons[s],
                Some(&u[ranges[s].clone()]),
            );
            mapped.matmul_seq(&d.0[1]).matmul_transpose_seq(mapped)
        });
        let induced = self.prepared.kernel([&q[0], &q[1]])?;
        let mut result = Vec::with_capacity(u.len());
        for spin in 0..if self.restricted { 1 } else { 2 } {
            let mapped = &self.frame.mapped[spin].0[0];
            let occupied = self.model.active_electrons[spin];
            let mo = mapped.transpose_matmul_seq(&induced[spin].matmul_seq(mapped));
            let energies = &self.prepared.base.active_scf.orbital_energies[spin];
            for i in 0..occupied {
                for a in occupied..mapped.cols {
                    let k = ranges[spin].start + i * (mapped.cols - occupied) + a - occupied;
                    result.push((energies[a] - energies[i]) * u[k] + mo[(a, i)]);
                }
            }
        }
        Ok(result)
    }
}
impl LinearOperator for ActiveKernel<'_> {
    fn dimension(&self) -> usize {
        self.ranges()[1].end
    }
    fn structure(&self) -> OperatorStructure {
        OperatorStructure::Hermitian
    }
    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        krylov::apply_real_linear(x, |u| self.real_image(u))
    }
}

fn second_from_reference(
    model: &CollinearAffineFrozenGeometry<'_>,
    direction: &[[f64; 3]],
    config: &ResponseSolverConfig,
    prepared: &Prepared,
    restricted: bool,
    cartesian_pair: Option<[&AffineFrozenSecondDerivative; 2]>,
) -> Result<AffineFrozenSecondDerivative> {
    if direction.len() != model.raw_reference.len()
        || direction.iter().flatten().any(|x| !x.is_finite())
    {
        return invalid("affine Hessian requires finite directions for every raw center");
    }
    let value = frame(model, &prepared.fb, &prepared.rb, &prepared.base, direction)?;
    let (first_explicit, second_explicit, fock_explicit) =
        prepared.explicit_direction(model, direction);
    let induced = prepared.kernel([&value.density[0][1], &value.density[1][1]])?;
    let mut source = Vec::new();
    let mut gaps = Vec::new();
    for spin in 0..if restricted { 1 } else { 2 } {
        let mapped = &value.mapped[spin];
        let mut derivative = fock_explicit[spin].clone();
        add(&mut derivative, &induced[spin], 1.);
        let mut mo = mapped.0[0].transpose_matmul_seq(&derivative.matmul_seq(&mapped.0[0]));
        let first = mapped.0[1].transpose_matmul_seq(&prepared.fock[spin].matmul_seq(&mapped.0[0]));
        add(&mut mo, &first, 1.);
        add(&mut mo, &first.transpose(), 1.);
        for i in 0..model.active_electrons[spin] {
            for a in model.active_electrons[spin]..mapped.0[0].cols {
                source.push(-mo[(a, i)]);
                gaps.push(
                    (prepared.base.active_scf.orbital_energies[spin][a]
                        - prepared.base.active_scf.orbital_energies[spin][i])
                        .abs(),
                );
            }
        }
    }
    if source.iter().chain(&gaps).any(|v| !v.is_finite()) {
        return invalid("affine Hessian requires finite orbital response sources and gaps");
    }
    let kernel = ActiveKernel {
        prepared,
        model,
        frame: &value,
        restricted,
    };
    let (rotation, solver_relative, iterations) = if source.is_empty() {
        (Vec::new(), Some(0.), 0)
    } else if let Some(pair) = cartesian_pair {
        // Only the first-order CPKS equation is linear. The pair's frame,
        // second-order frozen density and curvature above/below are still
        // constructed independently; no second derivative is added by hand.
        if pair
            .iter()
            .any(|p| p.source.len() != source.len() || p.rotation.len() != source.len())
        {
            return invalid("affine Hessian Cartesian response dimensions changed for a pair");
        }
        let mut defect = 0.0_f64;
        for (i, s) in source.iter().enumerate() {
            let difference = s - pair[0].source[i] - pair[1].source[i];
            if !difference.is_finite() {
                return invalid("affine Hessian pair source difference is nonfinite");
            }
            defect = defect.max(difference.abs());
        }
        if defect > config.acceptance() {
            return Err(DftError::LinearAlgebra(format!(
                "affine Hessian pair source linearity defect {defect:.6e} exceeds {:.6e}",
                config.acceptance()
            )));
        }
        (
            pair[0]
                .rotation
                .iter()
                .zip(&pair[1].rotation)
                .map(|(a, b)| a + b)
                .collect(),
            None,
            0,
        )
    } else {
        let norm = source.iter().map(|x| x * x).sum::<f64>().sqrt();
        let options = config.for_absolute_residual(norm)?;
        let rhs = source.iter().map(|x| Complex::from(*x)).collect::<Vec<_>>();
        let solved = krylov::solve(&kernel, &rhs, Some(&gaps), &options)?;
        (
            solved.amplitudes.iter().map(|x| x.re).collect::<Vec<_>>(),
            Some(solved.relative_residual),
            solved.iterations,
        )
    };
    if rotation.iter().any(|v| !v.is_finite()) {
        return invalid("affine Hessian orbital response contains a nonfinite entry");
    }
    let residual_vector = kernel
        .real_image(&rotation)?
        .iter()
        .zip(&source)
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    if residual_vector.iter().any(|v| !v.is_finite()) {
        return invalid("affine Hessian original orbital equation contains a nonfinite residual");
    }
    let residual = residual_vector.iter().map(|v| v.abs()).fold(0., f64::max);
    if cartesian_pair.is_some() && residual.is_finite() && residual > config.acceptance() {
        // The sum of two individually accepted solves can exceed the same
        // absolute gate. Refine only that pair through the original solver;
        // the failed reused residual is logged and no tolerance is relaxed.
        crate::logger::log(&format!(
            "affine hess: reused pair residual {residual:.6e} exceeds {:.6e}; solving that pair independently",
            config.acceptance()
        ));
        return second_from_reference(model, direction, config, prepared, restricted, None);
    }
    if !residual.is_finite() || residual > config.acceptance() {
        return Err(DftError::LinearAlgebra(format!(
            "affine frozen Hessian original orbital residual {residual:.6e} exceeds {:.6e}",
            config.acceptance()
        )));
    }
    // A reused pair has no Krylov stopping report. Return the relative norm
    // of its freshly evaluated, original equation rather than a copied value.
    let relative = solver_relative.unwrap_or_else(|| {
        let rhs_norm = source.iter().map(|v| v * v).sum::<f64>().sqrt();
        let residual_norm = residual_vector.iter().map(|v| v * v).sum::<f64>().sqrt();
        if rhs_norm == 0. {
            residual_norm
        } else {
            residual_norm / rhs_norm
        }
    });
    if !relative.is_finite() {
        return invalid("affine Hessian relative orbital residual is nonfinite");
    }
    let ranges = kernel.ranges();
    let relaxed = std::array::from_fn(|s| {
        channel_density(
            &value.mapped[s],
            &value.frozen[s],
            &occupied_projector(
                value.mapped[s].0[0].cols,
                model.active_electrons[s],
                Some(&rotation[ranges[s].clone()]),
            ),
        )
    });
    let mut first = first_explicit;
    for s in 0..2 {
        first += prepared.fock[s].frobenius_dot(&value.density[s][1]);
    }
    let gradient = prepared
        .base
        .gradient
        .iter()
        .zip(direction)
        .map(|(g, d)| g.x * d[0] + g.y * d[1] + g.z * d[2])
        .sum::<f64>();
    let defect = (first - gradient).abs();
    if !defect.is_finite() || defect > 1e-8 {
        return invalid(
            "affine Hessian first chart coefficient disagrees with its analytic gradient",
        );
    }
    Ok(AffineFrozenSecondDerivative {
        energy: prepared.base.energy,
        first_derivative: first,
        second_derivative: prepared.curvature(second_explicit, &fock_explicit, &relaxed)?,
        fixed_second_derivative: prepared.curvature(
            second_explicit,
            &fock_explicit,
            &value.density,
        )?,
        orbital_response_residual: residual,
        relative_response_residual: relative,
        response_iterations: iterations,
        gradient_consistency_error: defect,
        source,
        rotation,
    })
}

impl CollinearAffineFrozenGeometry<'_> {
    pub fn directional_second_derivative(
        &self,
        coordinates: &[Vec3],
        direction: &[[f64; 3]],
        config: &ResponseSolverConfig,
    ) -> Result<AffineFrozenSecondDerivative> {
        config.validate()?;
        let prepared = Prepared::new(self, coordinates, false)?;
        second_from_reference(self, direction, config, &prepared, false, None)
    }
    pub fn hessian(
        &self,
        coordinates: &[Vec3],
        config: &ResponseSolverConfig,
    ) -> Result<AffineFrozenHessian> {
        self.hessian_internal(coordinates, config, false)
    }
    fn hessian_internal(
        &self,
        coordinates: &[Vec3],
        config: &ResponseSolverConfig,
        restricted: bool,
    ) -> Result<AffineFrozenHessian> {
        config.validate()?;
        let preparation_started = std::time::Instant::now();
        let prepared = Prepared::new(self, coordinates, restricted)?;
        crate::logger::log(&format!(
            "affine hess: shared preparation: {:.3} s",
            preparation_started.elapsed().as_secs_f64()
        ));
        let factor = if restricted { 4. } else { 2. };
        let n = 3 * coordinates.len();
        let mut result = Matrix::zeros(n, n);
        let mut paths = Vec::with_capacity(n * (n + 1) / 2);
        let cartesian_started = std::time::Instant::now();
        for i in 0..n {
            let mut direction = vec![[0.; 3]; coordinates.len()];
            direction[i / 3][i % 3] = 1.;
            let value =
                second_from_reference(self, &direction, config, &prepared, restricted, None)?;
            result[(i, i)] = value.fixed_second_derivative
                - factor
                    * value
                        .source
                        .iter()
                        .zip(&value.rotation)
                        .map(|(a, b)| a * b)
                        .sum::<f64>();
            if (result[(i, i)] - value.second_derivative).abs() > 1e-8 {
                return invalid(
                    "ordered frozen Hessian relaxation disagrees with scalar curvature",
                );
            }
            paths.push(value);
        }
        crate::logger::log(&format!(
            "affine hess: {n} Cartesian responses: {:.3} s",
            cartesian_started.elapsed().as_secs_f64()
        ));
        let pairs_started = std::time::Instant::now();
        for i in 0..n {
            for j in i + 1..n {
                let mut direction = vec![[0.; 3]; coordinates.len()];
                direction[i / 3][i % 3] = 1.;
                direction[j / 3][j % 3] = 1.;
                let value = second_from_reference(
                    self,
                    &direction,
                    config,
                    &prepared,
                    restricted,
                    Some([&paths[i], &paths[j]]),
                )?;
                let fixed = 0.5
                    * (value.fixed_second_derivative
                        - paths[i].fixed_second_derivative
                        - paths[j].fixed_second_derivative);
                result[(i, j)] = fixed
                    - factor
                        * paths[i]
                            .source
                            .iter()
                            .zip(&paths[j].rotation)
                            .map(|(a, b)| a * b)
                            .sum::<f64>();
                result[(j, i)] = fixed
                    - factor
                        * paths[j]
                            .source
                            .iter()
                            .zip(&paths[i].rotation)
                            .map(|(a, b)| a * b)
                            .sum::<f64>();
                paths.push(value);
            }
        }
        crate::logger::log(&format!(
            "affine hess: {} checked pair curvatures: {:.3} s",
            n * (n - 1) / 2,
            pairs_started.elapsed().as_secs_f64()
        ));
        let raw_asymmetry = result.max_abs_diff(&result.transpose());
        let mut symmetric = result.clone();
        for i in 0..n {
            for j in 0..n {
                symmetric[(i, j)] = 0.5 * (result[(i, j)] + result[(j, i)]);
            }
        }
        Ok(AffineFrozenHessian {
            hessian: symmetric,
            raw_hessian: result,
            raw_asymmetry,
            directional_results: paths,
            reference_orbital_residual: prepared.base.active_scf.projected_orbital_residual,
        })
    }
}
impl AffineFrozenGeometry<'_> {
    pub fn directional_second_derivative(
        &self,
        coordinates: &[Vec3],
        direction: &[[f64; 3]],
        config: &ResponseSolverConfig,
    ) -> Result<AffineFrozenSecondDerivative> {
        config.validate()?;
        let model = self.collinear()?;
        let prepared = Prepared::new(&model, coordinates, true)?;
        second_from_reference(&model, direction, config, &prepared, true, None)
    }
    pub fn hessian(
        &self,
        coordinates: &[Vec3],
        config: &ResponseSolverConfig,
    ) -> Result<AffineFrozenHessian> {
        self.collinear()?
            .hessian_internal(coordinates, config, true)
    }
}
