// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use super::{run_biased_collinear, run_collinear, CollinearConstraint, ConstraintOptions};
use crate::acceptance::{components, stencils};
use crate::driver::method::{prepare_molecular, MolecularMethod};
use crate::integrals::overlap_matrix;

#[test]
fn methyl_spin_and_charge_multiplier_response() {
    // Distorted pyramidal methyl with three unequal bonds. The geometry
    // was chosen directly without optimization.
    let setup = prepare_molecular(
        &[6, 1, 1, 1],
        &[
            [0.13, -0.27, 0.19],
            [1.78, 0.34, -0.08],
            [-0.49, 1.65, 0.41],
            [0.31, -0.78, 1.91],
        ],
        &MolecularMethod {
            xc: Some("wb97x-d".into()),
            unrestricted: true,
            multiplicity: 2,
            grid_level: 0,
            grid_weight_cutoff: 0.0,
            conv_tol: 1e-12,
            conv_tol_grad: Some(1e-10),
            conv_tol_dp: 1e-10,
            ..MolecularMethod::default()
        },
    )
    .unwrap();
    let overlap = overlap_matrix(&setup.basis);
    let owner = setup.basis.sph_ao_atoms();
    let constraints: [CollinearConstraint; 2] = std::array::from_fn(|a| {
        let mut matrix = overlap.clone();
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                matrix[(i, j)] *=
                    0.5 * (usize::from(owner[i] == a) + usize::from(owner[j] == a)) as f64;
            }
        }
        let mut beta = matrix.clone();
        if a == 1 {
            beta.as_mut_slice().iter_mut().for_each(|x| *x = -*x);
        }
        CollinearConstraint {
            operators: [matrix, beta],
            target: 0.0,
        }
    });
    let evaluate = |lambda: [f64; 2]| {
        let bias = std::array::from_fn(|s| {
            let mut m = constraints[0].operators[s].clone();
            for (x, y) in m
                .as_mut_slice()
                .iter_mut()
                .zip(constraints[1].operators[s].as_slice())
            {
                *x = lambda[0] * (*x) + lambda[1] * y;
            }
            m
        });
        let r = run_biased_collinear(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            setup.grid.as_ref(),
            &bias,
            None,
        )
        .unwrap();
        std::array::from_fn::<_, 2, _>(|a| {
            (0..2)
                .map(|s| r.density[s].frobenius_dot(&constraints[a].operators[s]))
                .sum::<f64>()
        })
    };
    let lambda = [0.033, -0.021];
    let targets = evaluate(lambda);
    let fitted_constraints = std::array::from_fn::<_, 2, _>(|a| {
        let mut constraint = constraints[a].clone();
        constraint.target = targets[a];
        constraint
    });
    let result = run_collinear(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        setup.grid.as_ref(),
        false,
        &fitted_constraints,
        &[0.0; 2],
        &ConstraintOptions::default(),
    )
    .unwrap();
    for (a, b) in result.multipliers.iter().zip(lambda) {
        assert!((a - b).abs() < 1e-7);
    }
    println!("CH3 constrained UKS charge and spin targets={targets:?} multipliers={:?} residuals={:?} response={:?}",
        result.multipliers,result.residuals,result.response_residuals);
    let names: Vec<_> = (0..2)
        .flat_map(|i| (0..2).map(move |j| format!("CH3 dQ{i}/dlambda{j}")))
        .collect();
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.010, 0.007, 0.005, 0.003], stencils::Stencil::Second),
            ([0.006, 0.0045, 0.003, 0.002], stencils::Stencil::Second),
            ([0.018, 0.014, 0.011, 0.008], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; 4];
            for (k, &h) in steps.iter().enumerate() {
                for j in 0..2 {
                    let sample = |d| {
                        let mut f = lambda;
                        f[j] += d;
                        evaluate(f)
                    };
                    let (p, m) = (sample(h), sample(-h));
                    let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                        (sample(2.0 * h), sample(-2.0 * h))
                    } else {
                        ([0.0; 2], [0.0; 2])
                    };
                    for i in 0..2 {
                        errors[2 * i + j][k] = (stencil.first(p[i], m[i], p2[i], m2[i], h)
                            - result.susceptibility[(i, j)])
                            .abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("constrained CH3 charge and spin", &failures);
}
