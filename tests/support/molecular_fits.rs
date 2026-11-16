// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Every-component four-step fits of the molecular response of one method
//! on a distorted, generically oriented water molecule: the analytic gradient
//! (9) and Hessian (81) against central differences of energies and
//! gradients, and the static polarizability (9) and first
//! hyperpolarizability (27) against central differences of the field
//! dipole and of the analytic polarizability at the field-polarized
//! reference. `controls(extra)` gives the method controls with `extra`
//! added; failed components are appended to `failures` by label.
//! Three nuclear centres are necessarily coplanar; these all-component water
//! fits are not a C1 point-group certificate. Separate noncoplanar fixtures
//! are required for acceptance that specifically claims C1 geometry.

use super::stencils::{fit_components_staged, Stencil};
use dft_rs::driver::{run, Controls, System, Value};
use std::collections::HashMap;

pub const WATER: [(u8, [f64; 3]); 3] = [
    (8, [0.0, 0.0, 0.2217]),
    (1, [0.0, 1.4309, -0.8867]),
    (1, [0.13, -1.4309, -0.8867]),
];

const AXES: [&str; 3] = ["x", "y", "z"];

pub fn system(atoms: &[(u8, [f64; 3])]) -> System {
    System::molecule(
        atoms.iter().map(|(z, _)| *z).collect(),
        atoms.iter().map(|(_, r)| *r).collect(),
    )
}

fn floats(v: &Value, key: &str) -> Vec<f64> {
    v.get(key)
        .unwrap_or_else(|| panic!("report has no `{key}`"))
        .numeric_array()
        .unwrap()
        .1
}

fn number(v: &Value, key: &str) -> f64 {
    v.get(key)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("report has no number `{key}`"))
}

fn vec3(v: [f64; 3]) -> Value {
    Value::Array(v.iter().map(|x| Value::Float(*x)).collect())
}

/// Every component of the analytic gradient and Hessian: one displaced
/// pair of gradient runs per Cartesian coordinate gives that gradient
/// component and the whole Hessian column.
///
/// The last stage uses the fourth-order stencil. PW91 H[3][1] has an h^2
/// coefficient near 6e-7 that cancels against its h^4 term between
/// h = 0.02 and 0.01: errors 1.98e-8, 1.23e-9, 6.14e-11, 4.25e-11,
/// 1.30e-11, 3.96e-12, 1.59e-12 from h = 0.08 down to 0.00125 (ratios 16,
/// 20, 1.4, 3.3, 3.3, 2.5), so no four-step interval of the second-order
/// differences shows a clean order 2. SCAN has large higher-order
/// derivatives on the grid: its Hessian differences reach the h^2 regime
/// only below h = 0.003 (H[0][8]: 2.26e-5, 1.52e-5, 6.85e-6, 1.72e-6,
/// 4.06e-7 from h = 0.02 to 0.00125, ratios 1.5, 2.2, 4.0, 4.2), so the
/// third stage halves the steps again.
pub fn gradient_and_hessian(
    label: &str,
    controls: &dyn Fn(&[(&str, Value)]) -> Controls,
    failures: &mut Vec<String>,
) {
    const STAGES: [([f64; 4], Stencil); 5] = [
        ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
        ([0.01, 0.005, 0.0025, 0.00125], Stencil::Second),
        ([0.005, 0.0025, 0.00125, 0.000625], Stencil::Second),
        ([0.08, 0.04, 0.02, 0.01], Stencil::Second),
        ([0.08, 0.04, 0.02, 0.01], Stencil::Fourth),
    ];
    let gradient = |atoms: &[(u8, [f64; 3])]| -> (f64, Vec<f64>) {
        let report = run("gradient", &system(atoms), &controls(&[])).unwrap();
        (number(&report, "energy"), floats(&report, "gradient"))
    };
    let (_, g0) = gradient(&WATER);
    let hessian = floats(
        &run("hessian", &system(&WATER), &controls(&[])).unwrap(),
        "hessian",
    );
    assert_eq!(hessian.len(), 81);
    for coord in 0..9 {
        let (atom, axis) = (coord / 3, coord % 3);
        let displaced = |h: f64| {
            let mut atoms = WATER;
            atoms[atom].1[axis] += h;
            gradient(&atoms)
        };
        let mut names = vec![format!("{label} gradient dE/d{}{atom}", AXES[axis])];
        names.extend((0..9).map(|r| format!("{label} Hessian H[{r}][{coord}]")));
        fit_components_staged(
            &names,
            &STAGES,
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; 10];
                for (s, &h) in steps.iter().enumerate() {
                    let (ep, gp) = displaced(h);
                    let (em, gm) = displaced(-h);
                    let ((ep2, gp2), (em2, gm2)) = if stencil == Stencil::Fourth {
                        (displaced(2.0 * h), displaced(-2.0 * h))
                    } else {
                        ((0.0, vec![0.0; 9]), (0.0, vec![0.0; 9]))
                    };
                    errors[0][s] = (stencil.first(ep, em, ep2, em2, h) - g0[coord]).abs();
                    for r in 0..9 {
                        errors[1 + r][s] = (stencil.first(gp[r], gm[r], gp2[r], gm2[r], h)
                            - hessian[9 * r + coord])
                            .abs();
                    }
                }
                errors
            },
            failures,
        );
    }
}

/// Every component of the static polarizability and of the first
/// hyperpolarizability: a uniform field along axis j gives alpha_ij from
/// central differences of the dipole and beta_ikj = d alpha_ik / d F_j for
/// all i, k from central differences of the analytic polarizability at the
/// field-polarized reference (`reference_field`), so the three field axes
/// fit all 9 + 27 components.
///
/// (A first version checked the contraction along one generic direction
/// only. Along the z axis alone the h^2 coefficient for RPBE is 26 times
/// smaller than for PBE, a = -0.012 against +0.32, so dipole round-off
/// dominated below h = 0.002 and the fit gave p = 3.76. A second version
/// took beta from second differences of the dipole, whose noise grows as
/// 1/h^2: PBEsol beta_xxx errors 1.56e-6, 3.82e-7, 6.81e-8, 1.00e-7,
/// 4.24e-7 from h = 0.016 to 0.001, and 24 components of nine functionals
/// failed on every interval. The polarizability route has noise growing
/// only as 1/h; along F_y its first differences agree with the dipole
/// second differences to five digits for SCAN.)
///
/// SCAN has large higher-order field derivatives: d alpha_yy / d F_y of
/// SCAN changes from +0.032 to -0.030 between F_y = -0.007 and +0.007, so
/// the h^4 term competes with the h^2 term and second-order fits give
/// p = 1.76 to 2.15 (SCAN alpha_xy errors 3.08e-8, 1.20e-7, 4.36e-7,
/// 1.16e-6 from h = 0.001 to 0.008). The fourth-order stencil gives p = 3.8
/// on the same steps (9.85e-10, 1.54e-8, 1.93e-7 from h = 0.001 to 0.004),
/// so the last stages use it.
pub fn polarizability_and_hyperpolarizability(
    label: &str,
    controls: &dyn Fn(&[(&str, Value)]) -> Controls,
    failures: &mut Vec<String>,
) {
    const STAGES: [([f64; 4], Stencil); 5] = [
        ([0.016, 0.008, 0.004, 0.002], Stencil::Second),
        ([0.008, 0.004, 0.002, 0.001], Stencil::Second),
        ([0.032, 0.016, 0.008, 0.004], Stencil::Second),
        ([0.008, 0.004, 0.002, 0.001], Stencil::Fourth),
        ([0.016, 0.008, 0.004, 0.002], Stencil::Fourth),
    ];
    let alpha = floats(
        &run("polarizability", &system(&WATER), &controls(&[])).unwrap(),
        "polarizability",
    );
    let beta = floats(
        &run("beta", &system(&WATER), &controls(&[])).unwrap(),
        "tensor",
    );
    assert_eq!((alpha.len(), beta.len()), (9, 27));
    // Dipole and analytic polarizability in a uniform field, cached by field
    // so that stages sharing a step reuse the SCF.
    let mut cache: HashMap<[u64; 3], (Vec<f64>, Vec<f64>)> = HashMap::new();
    let mut at = |field: [f64; 3]| -> (Vec<f64>, Vec<f64>) {
        let key = field.map(f64::to_bits);
        if let Some(hit) = cache.get(&key) {
            return hit.clone();
        }
        let tight = ("conv_tol_grad", Value::Float(1e-11));
        let dipole = floats(
            &run(
                "dipole",
                &system(&WATER),
                &controls(&[("field", vec3(field)), tight.clone()]),
            )
            .unwrap(),
            "dipole",
        );
        let polarizability = floats(
            &run(
                "polarizability",
                &system(&WATER),
                &controls(&[("reference_field", vec3(field)), tight]),
            )
            .unwrap(),
            "polarizability",
        );
        cache.insert(key, (dipole.clone(), polarizability.clone()));
        (dipole, polarizability)
    };
    for j in 0..3 {
        let mut names: Vec<String> = (0..3)
            .map(|i| format!("{label} alpha_{}{}", AXES[i], AXES[j]))
            .collect();
        for i in 0..3 {
            for k in 0..3 {
                names.push(format!("{label} beta_{}{}{}", AXES[i], AXES[k], AXES[j]));
            }
        }
        fit_components_staged(
            &names,
            &STAGES,
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; 12];
                for (s, &h) in steps.iter().enumerate() {
                    let mut along = |d: f64| {
                        let mut f = [0.0; 3];
                        f[j] = d;
                        at(f)
                    };
                    let (mp, ap) = along(h);
                    let (mm, am) = along(-h);
                    let ((mp2, ap2), (mm2, am2)) = if stencil == Stencil::Fourth {
                        (along(2.0 * h), along(-2.0 * h))
                    } else {
                        ((vec![0.0; 3], vec![0.0; 9]), (vec![0.0; 3], vec![0.0; 9]))
                    };
                    for i in 0..3 {
                        errors[i][s] = (stencil.first(mp[i], mm[i], mp2[i], mm2[i], h)
                            - alpha[3 * i + j])
                            .abs();
                        for k in 0..3 {
                            let ik = 3 * i + k;
                            errors[3 + ik][s] = (stencil
                                .first(ap[ik], am[ik], ap2[ik], am2[ik], h)
                                - beta[9 * i + 3 * k + j])
                                .abs();
                        }
                    }
                }
                errors
            },
            failures,
        );
    }
    println!("{label}: alpha {alpha:?}");
    println!("{label}: beta {beta:?}");
}
