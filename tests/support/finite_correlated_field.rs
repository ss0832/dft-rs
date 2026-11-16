// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independently relaxed electric/magnetic SCF and direct complex PT2 sums.
//! Include together with `support/complex_pt2.rs`; no response recurrence is used.
use dft_rs::driver::method::MolecularSetup;
use dft_rs::integrals::london::{one_electron_series, repulsion_series, ComplexMatrix, FieldPath};
use dft_rs::response::complex::Complex;
use dft_rs::scf::magnetic::{magnetic_field_scf_with_core, MagneticFieldOptions};
use dft_rs::scf::ScfResult;
use dft_rs::xc::double_hybrid::Pt2Coefficients;
use dft_rs::Vec3;
use std::collections::BTreeMap;

/// Cache finite energies only, not orbital, integral or Taylor work arrays.
pub struct RelaxedFieldOracle<'a> {
    setup: &'a MolecularSetup,
    reference: &'a ScfResult,
    coefficients: Pt2Coefficients,
    london: bool,
    origin: [f64; 3],
    energies: BTreeMap<[u64; 6], [f64; 5]>,
}

impl<'a> RelaxedFieldOracle<'a> {
    pub fn new(
        setup: &'a MolecularSetup,
        reference: &'a ScfResult,
        coefficients: Pt2Coefficients,
        london: bool,
        origin: [f64; 3],
    ) -> Self {
        Self {
            setup,
            reference,
            coefficients,
            london,
            origin,
            energies: BTreeMap::new(),
        }
    }

    /// Channels: total, parent, SS-alpha, OS, SS-beta. Coordinates are E then B.
    pub fn at(&mut self, field: [f64; 6]) -> [f64; 5] {
        let key = field.map(|v| if v == 0.0 { 0 } else { v.to_bits() });
        if let Some(value) = self.energies.get(&key) {
            return *value;
        }
        let setup = self.setup;
        let magnetic = [field[3], field[4], field[5]];
        let electric = [field[0], field[1], field[2]];
        let options = MagneticFieldOptions {
            field: if self.london { magnetic } else { [0.0; 3] },
            spin_zeeman: false,
            series_order: Some(if self.london { 16 } else { 2 }),
            series_tolerance: 1e-15,
        };
        let n = setup.basis.nao_sph;
        let mut extra = [ComplexMatrix::zeros(n, n), ComplexMatrix::zeros(n, n)];
        if !self.london {
            // The fixed-AO magnetic Hamiltonian is exactly quadratic in B.
            // The electric operator is supplied separately through SCF options.
            let one = one_electron_series(
                &setup.molecule,
                &setup.basis,
                &FieldPath {
                    electric: [0.0; 3],
                    magnetic,
                    origin: self.origin,
                    london: false,
                    order: 2,
                },
            )
            .unwrap();
            for h in &mut extra {
                for i in 0..n {
                    for j in 0..n {
                        h.re[(i, j)] = one.core[1].re[(i, j)] + one.core[2].re[(i, j)];
                        h.im[(i, j)] = one.core[1].im[(i, j)] + one.core[2].im[(i, j)];
                    }
                }
            }
        }
        let mut scf_options = setup.options.clone();
        scf_options.external_field = Some(Vec3::new(electric[0], electric[1], electric[2]));
        let state = magnetic_field_scf_with_core(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            self.reference,
            &scf_options,
            &options,
            &extra,
        )
        .unwrap();
        assert!(state.converged);
        let integrals = repulsion_series(
            &setup.basis,
            &FieldPath {
                electric: [0.0; 3],
                magnetic,
                origin: self.origin,
                london: self.london,
                order: if self.london { 16 } else { 0 },
            },
            None,
        )
        .unwrap();
        let mut g = vec![Complex::ZERO; n.pow(4)];
        for (re, im) in integrals {
            for (p, v) in g.iter_mut().enumerate() {
                *v += Complex::new(re[p], im[p]);
            }
        }
        let e = super::complex_pt2::finite_channels(
            n,
            &g,
            &state.coefficients,
            &state.orbital_energies,
            [state.n_alpha, state.n_beta],
        );
        let value = [
            state.energy
                + self.coefficients.same_spin * (e[0] + e[2])
                + self.coefficients.opposite_spin * e[1],
            state.energy,
            e[0],
            e[1],
            e[2],
        ];
        self.energies.insert(key, value);
        value
    }
}
