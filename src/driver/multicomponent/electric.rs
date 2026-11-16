// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Common report conventions for nonrelativistic multicomponent electric moments.

use super::Value;
use crate::math::Vec3;
use crate::multicomponent::response::CoupledQuadrupoleResponse;
use crate::response::electric_tensor::{BUCKINGHAM_FACTOR, quadrupole_modes};

pub(super) fn quadrupole_report(
    result: &CoupledQuadrupoleResponse,
    origin: Vec3,
    convention: &str,
) -> (Value, bool) {
    let converged = result.responses.iter().all(|r| r.converged);
    let block = |a: usize, b: usize| result.tensor[a][b];
    let qq: [[f64; 5]; 5] =
        std::array::from_fn(|a| std::array::from_fn(|b| BUCKINGHAM_FACTOR * block(a + 3, b + 3)));
    let dq: [[f64; 5]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| block(a, b + 3)));
    let qd: [[f64; 3]; 5] = std::array::from_fn(|a| std::array::from_fn(|b| block(a + 3, b)));
    let raw_asymmetry = (0..5)
        .flat_map(|a| (0..5).map(move |b| (qq[a][b] - qq[b][a]).abs()))
        .fold(0.0_f64, f64::max);
    let cross_asymmetry = (0..3)
        .flat_map(|a| (0..5).map(move |b| (dq[a][b] - qd[b][a]).abs()))
        .fold(0.0_f64, f64::max);
    let value = Value::object()
        .with("moment", std::array::from_fn::<_, 5, _>(|a| BUCKINGHAM_FACTOR * result.moments[a + 3]))
        .with("dipole_moment", [result.moments[0], result.moments[1], result.moments[2]])
        .with("origin", origin.to_array())
        .with("origin_convention", convention)
        .with("mode_order", ["xx-yy", "xx+yy-2zz", "xy+yx", "xz+zx", "yz+zy"])
        .with("field_gradient_modes", quadrupole_modes())
        .with("polarizability", qq)
        .with("dipole_from_field_gradient", dq)
        .with("field_conjugate_moment_from_field", qd)
        .with("raw_asymmetry", raw_asymmetry)
        .with("cross_asymmetry", cross_asymmetry)
        .with("response_residuals", result.responses.iter().map(|r| r.relative_residual).collect::<Vec<_>>())
        .with("reference_residuals", result.responses.iter().map(|r| r.reference_residual).collect::<Vec<_>>())
        .with("dipole_position_response_residuals", result.responses[..3].iter().map(|r| r.position_residual).collect::<Vec<_>>())
        .with("quadrupole_position_response_residuals", result.responses[3..].iter().map(|r| r.position_residual).collect::<Vec<_>>())
        .with("axis_convention", "observed moment by driven field; electron H'(G)=G:rr/2; Buckingham moments are three times field-conjugate quadrupole moments")
        .with("converged", converged);
    (value, converged)
}
