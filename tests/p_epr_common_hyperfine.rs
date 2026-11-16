// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Reference-preserving static A checks on distorted C1 methyl. Minimal
//! basis values are implementation diagnostics, not a converged benchmark.

use dft_rs::constants;
use dft_rs::properties::epr;
#[path = "support/c1_geometry.rs"]
mod c1_geometry;

#[test]
fn constrained_hf_and_pbe_contact_match_independent_common_open_orbitals() {
    dft_rs::resources::ResourceLimits::new(Some(1),Some(1280)).unwrap().install(|| {
        let system=dft_rs::driver::System::molecule(vec![6,1,1,1],vec![
            [0.17,-0.11,0.23],[1.77,0.18,-0.53],[-0.31,1.74,-0.45],[-1.2,-1.31,-0.61],
        ]);
        c1_geometry::assert_c1(&system.numbers,&system.positions);
        for xc in ["hf","pbe"] {
            let controls=dft_rs::driver::Controls::from_value(&dft_rs::driver::Value::object()
                .with("xc",xc).with("basis","sto-3g").with("multiplicity",2usize)
                .with("grid_level",2usize).with("grid_weight_cutoff",0.)
                .with("conv_tol",1e-12).with("conv_tol_grad",1e-11).with("conv_tol_dp",1e-12)
                .with("dispersion","none").with("max_iter",500usize)).unwrap();
            let method=dft_rs::driver::method::MolecularMethod::from_controls(&controls).unwrap();
            let setup=dft_rs::driver::method::prepare_molecular(&system.numbers,&system.positions,&method).unwrap();
            let (reference,common)=dft_rs::scf::roks::run_roks_with_orbitals(
                &setup.molecule,&setup.basis,&setup.options,&setup.functional,setup.grid.as_ref()).unwrap();
            assert!(reference.converged);
            assert_eq!((reference.n_alpha,reference.n_beta),(5,4));
            let points=setup.molecule.atoms.iter().map(|a|a.position).collect::<Vec<_>>();
            let values=dft_rs::grid::eval::eval_aos(&setup.basis,&points,false);
            let scale=constants::FINE_STRUCTURE.powi(2)*constants::ELECTRON_G_FACTOR
                /(4.*constants::PROTON_ELECTRON_MASS_RATIO)
                *constants::HARTREE_TO_JOULE/(2.*std::f64::consts::PI*constants::HBAR_J_S)*1e-6;
            for atom in 0..4 {
                let result=epr::common_orbital_hyperfine_tensor(&setup.basis,&reference,&common,points[atom],1e-9).unwrap();
                let open=(reference.n_beta..reference.n_alpha).map(|p| {
                    (0..setup.basis.nao_sph).map(|mu|common[(mu,p)]*values.phi[(atom,mu)]).sum::<f64>().powi(2)
                }).sum::<f64>();
                let expected=scale*(8.*std::f64::consts::PI/3.)*open;
                assert!((result.contact-expected).abs()<1e-7,"{xc} nucleus{atom} independent contact density");
                assert!(result.symmetry_residual<1e-7);
                assert!(result.dipolar_trace.abs()<1e-7);
                println!("C1 methyl {xc} restricted-open-shell nucleus{atom}: contact={} MHz/nuclear_g, A={:?}",result.contact,result.total);
            }
        }
    }).unwrap();
}
