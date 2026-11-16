// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Nuclear derivatives of the existing collinear second-order TDA propagator.
use super::{
    correlated_molecular_taylor_for_model, matrix_product, molecular_taylor_parts,
    MolecularDirection,
};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::Matrix;
use crate::periodic::BlochMatrix;
use crate::properties::pt2_response::projector_frames;
use crate::properties::pt2_transition::collinear_transition_path;
use crate::response::correlated_tda::{correlated_pole_taylor, PropagatorTaylor};
use crate::response::field_path::FieldEnergyModel;
use crate::response::spinor_tda::SpinorTdaConfig;
use crate::response::tddft::{TdaSolver, TdaSpin};
use crate::response::ResponseSolverConfig;
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::double_hybrid::Pt2Coefficients;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

/// Complete factorial-normalized absolute excited-state and pole coefficients.
pub struct CorrelatedExcitationPath {
    pub ground: Vec<f64>,
    pub parent_excitation: Vec<f64>,
    pub correlation_shift: Vec<f64>,
    pub excitation: Vec<f64>,
    pub total: Vec<f64>,
    pub parent_root_gap: f64,
    pub smallest_double_gap: f64,
    pub reference_operator_defect: f64,
    pub orbital_frame_defect: f64,
    pub coupling_duality_defect: f64,
    pub parent_residuals: Vec<f64>,
    pub double_residuals: Vec<f64>,
    pub amplitude_residuals: Vec<f64>,
    pub reference_residuals: Vec<f64>,
}

/// Complete Cartesian derivatives of the absolute correlated excited energy.
pub struct CorrelatedExcitedHessian {
    pub ground_energy: f64,
    pub parent_excitation_energy: f64,
    pub correlation_shift: f64,
    pub excitation_energy: f64,
    pub excited_state_energy: f64,
    pub gradient: Vec<f64>,
    pub hessian: Matrix,
    pub ground_hessian: Matrix,
    pub parent_excitation_hessian: Matrix,
    pub correlation_shift_hessian: Matrix,
    pub excitation_hessian: Matrix,
    pub minimum_parent_gap: f64,
    pub minimum_double_gap: f64,
    pub residual: f64,
    pub paths: usize,
}

/// Every diagonal and mixed Cartesian entry from analytic normalized paths.
#[allow(clippy::too_many_arguments)]
pub fn correlated_excited_state_hessian(
    molecule: &Molecule,
    basis: &MolBasis,
    functional: &ResolvedFunctional,
    target: Option<FieldEnergyModel<'_>>,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    options: &ScfOptions,
    coefficients: Pt2Coefficients,
    spin: TdaSpin,
    root: usize,
    tda_config: &SpinorTdaConfig,
    config: &ResponseSolverConfig,
) -> Result<CorrelatedExcitedHessian> {
    let n = 3 * molecule.len();
    let mut out = CorrelatedExcitedHessian {
        ground_energy: 0.0,
        parent_excitation_energy: 0.0,
        correlation_shift: 0.0,
        excitation_energy: 0.0,
        excited_state_energy: 0.0,
        gradient: vec![0.0; n],
        hessian: Matrix::zeros(n, n),
        ground_hessian: Matrix::zeros(n, n),
        parent_excitation_hessian: Matrix::zeros(n, n),
        correlation_shift_hessian: Matrix::zeros(n, n),
        excitation_hessian: Matrix::zeros(n, n),
        minimum_parent_gap: f64::INFINITY,
        minimum_double_gap: f64::INFINITY,
        residual: 0.0,
        paths: 0,
    };
    let mut second = vec![[0.0; 4]; n];
    let evaluate = |direction: &[[f64; 3]]| {
        correlated_excitation_path(
            molecule,
            basis,
            functional,
            target.clone(),
            grid,
            reference,
            options,
            coefficients,
            spin,
            root,
            direction,
            2,
            tda_config,
            config,
        )
    };
    let diagnostics = |out: &mut CorrelatedExcitedHessian, p: &CorrelatedExcitationPath| {
        out.paths += 1;
        out.minimum_parent_gap = out.minimum_parent_gap.min(p.parent_root_gap);
        out.minimum_double_gap = out.minimum_double_gap.min(p.smallest_double_gap);
        out.residual = out
            .residual
            .max(p.reference_operator_defect)
            .max(p.orbital_frame_defect)
            .max(p.coupling_duality_defect);
        for value in p
            .parent_residuals
            .iter()
            .chain(&p.double_residuals)
            .chain(&p.amplitude_residuals)
            .chain(&p.reference_residuals)
        {
            out.residual = out.residual.max(*value);
        }
    };
    for i in 0..n {
        let mut direction = vec![[0.0; 3]; molecule.len()];
        direction[i / 3][i % 3] = 1.0;
        let p = evaluate(&direction)?;
        diagnostics(&mut out, &p);
        out.ground_energy = p.ground[0];
        out.parent_excitation_energy = p.parent_excitation[0];
        out.correlation_shift = p.correlation_shift[0];
        out.excitation_energy = p.excitation[0];
        out.excited_state_energy = p.total[0];
        out.gradient[i] = p.total[1];
        second[i] = [
            p.ground[2],
            p.parent_excitation[2],
            p.correlation_shift[2],
            p.total[2],
        ];
        out.ground_hessian[(i, i)] = 2.0 * second[i][0];
        out.parent_excitation_hessian[(i, i)] = 2.0 * second[i][1];
        out.correlation_shift_hessian[(i, i)] = 2.0 * second[i][2];
        out.hessian[(i, i)] = 2.0 * second[i][3];
    }
    for i in 0..n {
        for j in 0..i {
            let mut direction = vec![[0.0; 3]; molecule.len()];
            direction[i / 3][i % 3] = 1.0;
            direction[j / 3][j % 3] = 1.0;
            let p = evaluate(&direction)?;
            diagnostics(&mut out, &p);
            let c = [
                p.ground[2],
                p.parent_excitation[2],
                p.correlation_shift[2],
                p.total[2],
            ];
            for (a, matrix) in [
                &mut out.ground_hessian,
                &mut out.parent_excitation_hessian,
                &mut out.correlation_shift_hessian,
                &mut out.hessian,
            ]
            .into_iter()
            .enumerate()
            {
                let value = c[a] - second[i][a] - second[j][a];
                matrix[(i, j)] = value;
                matrix[(j, i)] = value;
            }
        }
    }
    for i in 0..n {
        for j in 0..n {
            out.excitation_hessian[(i, j)] =
                out.parent_excitation_hessian[(i, j)] + out.correlation_shift_hessian[(i, j)];
        }
    }
    Ok(out)
}

// Four one-index coefficient convolutions. Every endpoint participates;
// dense row-major products use the existing sequential SIMD matrix kernel.
fn full_integrals(
    nao: usize,
    ao: &[Vec<f64>],
    orbitals: &[Vec<Matrix>; 2],
) -> Result<Vec<[Vec<f64>; 3]>> {
    let width = ao.len();
    let mut output: Vec<[Vec<f64>; 3]> = (0..width)
        .map(|_| std::array::from_fn(|_| Vec::new()))
        .collect();
    for (channel, (s, t)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
        let mut shape = [nao; 4];
        let mut tensor = ao.to_vec();
        for (axis, spin) in [s, s, t, t].into_iter().enumerate() {
            let rows = shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, n)| *n)
                .product();
            let coordinates: Vec<[usize; 4]> = (0..rows)
                .map(|mut row| {
                    let mut x = [0; 4];
                    for a in (0..4).rev() {
                        if a != axis {
                            x[a] = row % shape[a];
                            row /= shape[a];
                        }
                    }
                    x
                })
                .collect();
            let flat = |x: [usize; 4], shape: [usize; 4]| {
                ((x[0] * shape[1] + x[1]) * shape[2] + x[2]) * shape[3] + x[3]
            };
            let packed: Vec<_> = tensor
                .iter()
                .map(|plane| {
                    let mut m = Matrix::zeros(rows, nao);
                    for (row, x) in coordinates.iter().enumerate() {
                        for mu in 0..nao {
                            let mut xx = *x;
                            xx[axis] = mu;
                            m[(row, mu)] = plane[flat(xx, shape)];
                        }
                    }
                    m
                })
                .collect();
            let count = orbitals[spin][0].cols;
            let mut next = shape;
            next[axis] = count;
            tensor = (0..width)
                .map(|k| {
                    let mut m = Matrix::zeros(rows, count);
                    for a in 0..=k {
                        let value = packed[a].matmul_seq(&orbitals[spin][k - a]);
                        for (x, y) in m.as_mut_slice().iter_mut().zip(value.as_slice()) {
                            *x += y;
                        }
                    }
                    let mut out = vec![0.0; next.iter().product()];
                    for (row, x) in coordinates.iter().enumerate() {
                        for p in 0..count {
                            let mut xx = *x;
                            xx[axis] = p;
                            out[flat(xx, next)] = m[(row, p)];
                        }
                    }
                    out
                })
                .collect();
            shape = next;
        }
        for k in 0..width {
            output[k][channel] = std::mem::take(&mut tensor[k]);
        }
    }
    Ok(output)
}

fn block(c: &Matrix, start: usize, count: usize) -> Matrix {
    let mut out = Matrix::zeros(c.rows, count);
    for i in 0..c.rows {
        for j in 0..count {
            out[(i, j)] = c[(i, start + j)];
        }
    }
    out
}

/// The same full-Coulomb normal-ordered second-order propagator as
/// `correlated_tda_states`, on complete solved nuclear orbital/amplitude paths.
/// Positive collinear SS/OS coefficients are independently defined. The parent
/// TDA space is singlet/triplet or the complete spin-conserving UKS space.
/// `target` changes the explicit ground energy, not the parent propagator model.
#[allow(clippy::too_many_arguments)]
pub fn correlated_excitation_path(
    molecule: &Molecule,
    basis: &MolBasis,
    functional: &ResolvedFunctional,
    target: Option<FieldEnergyModel<'_>>,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    options: &ScfOptions,
    coefficients: Pt2Coefficients,
    spin: TdaSpin,
    root: usize,
    direction: &[[f64; 3]],
    order: usize,
    tda_config: &SpinorTdaConfig,
    config: &ResponseSolverConfig,
) -> Result<CorrelatedExcitationPath> {
    if !(1..=2).contains(&order)
        || reference.unrestricted && spin != TdaSpin::SpinResolved
        || !reference.unrestricted && spin == TdaSpin::SpinResolved
    {
        return Err(DftError::InvalidInput("correlated excitation nuclear paths require order one or two and a consistent collinear spin manifold".into()));
    }
    coefficients.validate()?;
    let size = reference
        .mo_coeff
        .iter()
        .map(|c| c.cols.max(c.rows))
        .max()
        .unwrap_or(0);
    let required_mb = size
        .checked_pow(4)
        .and_then(|n| n.checked_mul(8 * 48 * (order + 1)))
        .ok_or_else(|| {
            DftError::InvalidInput("correlated excitation integral workspace overflow".into())
        })?
        .div_ceil(1024 * 1024);
    if required_mb > tda_config.memory_mb {
        return Err(DftError::ResourceLimit {
            operation: "correlated excitation full integral paths",
            required_mb,
            limit_mb: tda_config.memory_mb,
        });
    }
    let direction = MolecularDirection::nuclear(direction.to_vec());
    let ground = correlated_molecular_taylor_for_model(
        molecule,
        basis,
        functional,
        target,
        grid,
        reference,
        options,
        coefficients,
        &direction,
        order,
        config,
    )?;
    let (report, path) = molecular_taylor_parts(
        molecule, basis, functional, grid, reference, options, &direction, order, config,
    )?;
    let o = path.n_occ;
    let n = [reference.mo_coeff[0].cols, reference.mo_coeff[1].cols];
    let v = [n[0] - o[0], n[1] - o[1]];
    let mut frame_defect = 0.0_f64;
    let frame: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
        let (f, e) = projector_frames(&path.d[s], o[s]);
        frame_defect = frame_defect.max(e);
        f
    });
    let moving: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
        let transpose: Vec<_> = frame[s].iter().map(Matrix::transpose).collect();
        matrix_product(
            &matrix_product(&transpose, &path.f[s], order),
            &frame[s],
            order,
        )
    });
    for s in 0..2 {
        let actual = matrix_product(&path.b[s], &frame[s], order);
        for (k, c) in actual.iter().enumerate() {
            frame_defect = frame_defect.max(c.max_abs_diff(&ground.orbitals[s][k]));
        }
    }
    if frame_defect > config.acceptance() {
        return Err(DftError::ResponseNotConverged {
            iterations: order,
            residual: frame_defect,
        });
    }
    let full = full_integrals(
        basis.nao_sph,
        path.geometry.coulomb_coefficients(),
        &ground.orbitals,
    )?;
    let tensors = collinear_transition_path(
        o,
        &moving,
        &full,
        &ground.amplitudes,
        coefficients,
        tda_config.memory_mb,
        tda_config.tolerance,
    )?;
    let singles: Vec<_> = (0..2)
        .flat_map(|s| (0..v[s]).flat_map(move |a| (0..o[s]).map(move |i| (s, a, i))))
        .collect();
    let dim = singles.len();
    let mut parent = vec![Matrix::zeros(dim, dim); order + 1];
    let co: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
        ground.orbitals[s]
            .iter()
            .map(|c| block(c, 0, o[s]))
            .collect()
    });
    let cv: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
        ground.orbitals[s]
            .iter()
            .map(|c| block(c, o[s], v[s]))
            .collect()
    });
    let co_dagger: [Vec<Matrix>; 2] =
        std::array::from_fn(|s| co[s].iter().map(Matrix::transpose).collect());
    let cv_dagger: [Vec<Matrix>; 2] =
        std::array::from_fn(|s| cv[s].iter().map(Matrix::transpose).collect());
    for (col, &(spin, a, i)) in singles.iter().enumerate() {
        let mut unit = vec![Matrix::zeros(v[spin], o[spin]); order + 1];
        unit[0][(a, i)] = 1.0;
        let transition = matrix_product(
            &matrix_product(&cv[spin], &unit, order),
            &co_dagger[spin],
            order,
        );
        let mut density: [Vec<Matrix>; 2] =
            std::array::from_fn(|_| vec![Matrix::zeros(basis.nao_sph, basis.nao_sph); order + 1]);
        density[spin] = transition;
        let induced = path
            .geometry
            .evaluate_linear(functional, &path.p, &density, order)?;
        let projected: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
            matrix_product(
                &matrix_product(&cv_dagger[s], &induced[s], order),
                &co[s],
                order,
            )
        });
        for (row, &(s, b, j)) in singles.iter().enumerate() {
            for k in 0..=order {
                let mut value = projected[s][k][(b, j)];
                if s == spin {
                    if i == j {
                        value += moving[s][k][(o[s] + b, o[s] + a)];
                    }
                    if a == b {
                        value -= moving[s][k][(i, j)];
                    }
                }
                parent[k][(row, col)] = value;
            }
        }
    }
    let adapted = spin != TdaSpin::SpinResolved;
    let reduced = if adapted { o[0] * v[0] } else { dim };
    let mut q = Matrix::zeros(dim, reduced);
    if adapted {
        if n[0] != n[1] || o[0] != o[1] {
            return Err(DftError::InvalidInput(
                "spin adaptation needs matching orbital spaces".into(),
            ));
        }
        for i in 0..reduced {
            q[(i, i)] = 1.0 / std::f64::consts::SQRT_2;
            q[(reduced + i, i)] =
                if spin == TdaSpin::Triplet { -1.0 } else { 1.0 } / std::f64::consts::SQRT_2;
        }
    } else {
        q = Matrix::identity(dim);
    }
    let qt = q.transpose();
    let reduce = |a: &Matrix| qt.matmul_seq(a).matmul_seq(&q);
    let complex = |a: Matrix| {
        let (rows, cols) = (a.rows, a.cols);
        BlochMatrix {
            re: a,
            im: Matrix::zeros(rows, cols),
        }
    };
    let parent: Vec<_> = parent.iter().map(|a| complex(reduce(a))).collect();
    let singles: Vec<_> = tensors
        .singles
        .iter()
        .map(|a| complex(reduce(&a.re)))
        .collect();
    let metric: Vec<_> = tensors
        .metric
        .iter()
        .map(|a| complex(reduce(&a.re)))
        .collect();
    let coupling: Vec<_> = tensors
        .coupling
        .iter()
        .map(|a| complex(qt.matmul_seq(&a.re)))
        .collect();
    let solver = TdaSolver::new(molecule, basis, functional, grid, reference, options, spin)?;
    let mut reference_operator_defect = 0.0_f64;
    for col in 0..reduced {
        let mut unit = vec![0.0; reduced];
        unit[col] = 1.0;
        let expected = solver.a_vec(&unit)?;
        for (row, &value) in expected.iter().enumerate() {
            reference_operator_defect =
                reference_operator_defect.max((parent[0].re[(row, col)] - value).abs());
        }
    }
    if reference_operator_defect > tda_config.tolerance {
        return Err(DftError::ResponseNotConverged {
            iterations: 0,
            residual: reference_operator_defect,
        });
    }
    let corrected = correlated_pole_taylor(
        &PropagatorTaylor {
            parent: &parent,
            singles: &singles,
            metric: &metric,
            doubles: &tensors.doubles,
            coupling: &coupling,
        },
        root,
        tda_config,
    )?;
    let root_gap =
        crate::response::spinor_tda::spinor_tda_eigenpair_taylor(&parent, root, tda_config)?
            .nearest_gap;
    let total = ground
        .energy
        .iter()
        .zip(&corrected.energies)
        .map(|(e, w)| e + w)
        .collect();
    if !report.converged {
        return Err(DftError::ResponseNotConverged {
            iterations: order,
            residual: ground
                .response_residuals
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
        });
    }
    Ok(CorrelatedExcitationPath {
        ground: ground.energy,
        parent_excitation: corrected.parent_energies,
        correlation_shift: corrected.corrections,
        excitation: corrected.energies,
        total,
        parent_root_gap: root_gap,
        smallest_double_gap: corrected.smallest_double_gap,
        reference_operator_defect,
        orbital_frame_defect: frame_defect,
        coupling_duality_defect: tensors.coupling_duality_defect,
        parent_residuals: corrected.parent_equation_residuals,
        double_residuals: corrected.double_equation_residuals,
        amplitude_residuals: ground.amplitude_residuals,
        reference_residuals: ground.response_residuals,
    })
}
