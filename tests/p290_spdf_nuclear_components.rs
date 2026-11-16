// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete C1 s/p/d/f nuclear derivative acceptance, not a directional sample.
//! Every Hessian entry is fitted. Cubic/quartic response uses complete
//! unisolvent sets with 364/1365 independent generic directions. The oracle
//! is a fresh displaced SCF analytic gradient. Six reference/model cases
//! cover all-electron RHF/RKS and nonlocal-ECP RHF/RKS/UHF/UKS. A case is
//! accepted only after all its components pass the unchanged staged gates.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/quartic_oracle.rs"]
mod quartic_oracle;
#[path = "support/spdf_fixture.rs"]
mod spdf_fixture;
#[path = "support/stencils.rs"]
mod stencils;
#[path = "support/symmetric_cartesian_directions.rs"]
mod symmetric_directions;

#[path = "support/nuclear_components.rs"]
mod nuclear_components;
fn fit_case(ecp: bool, unrestricted: bool, xc: &str, order: usize) {
    let (system, controls) = spdf_fixture::fixture(ecp, unrestricted, xc);
    nuclear_components::fit_case(
        system,
        controls,
        nuclear_components::BasisContract {
            scalar_ecp: ecp,
            highest_l: 3,
            ao_count: 20,
            required_l: &[0, 1, 2, 3],
        },
        format!("C1 spdf ecp={ecp} unrestricted={unrestricted} xc={xc} order={order}"),
        order,
    );
}

macro_rules! cases {
    ($(($name:ident,$ecp:expr,$unrestricted:expr,$xc:expr,$order:expr)),* $(,)?)=>{$(
        #[test] fn $name(){fit_case($ecp,$unrestricted,$xc,$order);}
    )*};
}
cases! {
    (ae_rhf_hessian,false,false,"hf",2),(ae_rhf_cubic,false,false,"hf",3),(ae_rhf_quartic,false,false,"hf",4),
    (ae_rks_hessian,false,false,"pbe",2),(ae_rks_cubic,false,false,"pbe",3),(ae_rks_quartic,false,false,"pbe",4),
    (ecp_rhf_hessian,true,false,"hf",2),(ecp_rhf_cubic,true,false,"hf",3),(ecp_rhf_quartic,true,false,"hf",4),
    (ecp_rks_hessian,true,false,"pbe",2),(ecp_rks_cubic,true,false,"pbe",3),(ecp_rks_quartic,true,false,"pbe",4),
    (ecp_uhf_hessian,true,true,"hf",2),(ecp_uhf_cubic,true,true,"hf",3),(ecp_uhf_quartic,true,true,"hf",4),
    (ecp_uks_hessian,true,true,"pbe",2),(ecp_uks_cubic,true,true,"pbe",3),(ecp_uks_quartic,true,true,"pbe",4),
}
