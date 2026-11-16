// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The complete physical density evaluation shared by SCF and its line search.

use super::{JkEngine, ScfOptions};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::integrals::{kinetic_matrix, nuclear_matrix};
use crate::linalg::Matrix;
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::Result;

pub(super) fn one_electron(
    mol: &Molecule,
    basis: &MolBasis,
    opts: &ScfOptions,
    charges: &[f64],
) -> Result<(Matrix, f64)> {
    let has_ecp = !basis.ecps.is_empty();
    let mut core = kinetic_matrix(basis);
    let attraction = nuclear_matrix(basis, mol, has_ecp.then_some(charges));
    for (h, v) in core.as_mut_slice().iter_mut().zip(attraction.as_slice()) {
        *h += v;
    }
    if has_ecp {
        let potential = crate::integrals::ecp_matrix(basis)?;
        for (h, v) in core.as_mut_slice().iter_mut().zip(potential.as_slice()) {
            *h += v;
        }
    }
    let mut classical = if has_ecp {
        basis.nuclear_repulsion(mol)
    } else {
        mol.nuclear_repulsion()
    };
    if let Some(field) = opts.external_field {
        let dipoles = crate::integrals::dipole_matrices(basis, crate::math::Vec3::zero());
        for (axis, dipole) in dipoles.iter().enumerate() {
            let strength = field.get(axis);
            if strength != 0.0 {
                for (h, d) in core.as_mut_slice().iter_mut().zip(dipole.as_slice()) {
                    *h += strength * d;
                }
            }
        }
        let mut nuclear_field = 0.0;
        for (atom, z) in mol.atoms.iter().zip(charges) {
            nuclear_field -= z * atom.position.dot(field);
        }
        classical += nuclear_field;
    }
    if let Some(extra) = &opts.extra_hcore {
        for (h, v) in core.as_mut_slice().iter_mut().zip(extra.as_slice()) {
            *h += v;
        }
    }
    Ok((core, classical))
}

#[derive(Clone, Copy)]
pub(super) enum DensityExtension<'a> {
    SpinPotential(&'a [Matrix; 2]),
    Hubbard(&'a super::hubbard::HubbardModel),
}

impl DensityExtension<'_> {
    pub(super) fn add_response(
        self,
        variation: &[Matrix; 2],
        focks: &mut [Matrix; 2],
    ) -> Result<()> {
        // A frozen one-body operator has zero density derivative.
        if let Self::Hubbard(model) = self {
            add_spin_fock(focks, &model.response(variation)?);
        }
        Ok(())
    }
}

fn add_spin_fock(focks: &mut [Matrix; 2], addition: &[Matrix; 2]) {
    for (fock, potential) in focks.iter_mut().zip(addition) {
        for (value, extra) in fock.as_mut_slice().iter_mut().zip(potential.as_slice()) {
            *value += extra;
        }
    }
}

pub(super) struct MolecularDensityModel<'a> {
    pub mol: &'a Molecule,
    pub mb: &'a MolBasis,
    pub opts: &'a ScfOptions,
    pub functional: &'a ResolvedFunctional,
    pub grid: Option<&'a MolecularGrid>,
    pub h_core: &'a Matrix,
    pub extension: Option<DensityExtension<'a>>,
    pub jk: &'a JkEngine,
    pub jk_lr: Option<&'a JkEngine>,
    pub pcm_solver: Option<&'a crate::pcm::PcmSolver>,
    pub nlc_grid: Option<&'a MolecularGrid>,
    pub restricted: bool,
}

pub(super) struct MolecularDensityFields {
    pub fock: [Matrix; 2],
    pub electronic_energy: f64,
    pub xc_energy: f64,
    pub nonlocal_energy: f64,
    pub solvent_energy: f64,
    pub grid_electron_count: Option<f64>,
}

impl MolecularDensityModel<'_> {
    pub fn evaluate(&self, density: &[Matrix; 2]) -> Result<MolecularDensityFields> {
        let mol = self.mol;
        let mb = self.mb;
        let opts = self.opts;
        let functional = self.functional;
        let grid = self.grid;
        let h_core = self.h_core;
        let jk = self.jk;
        let jk_lr = self.jk_lr;
        let pcm_solver = self.pcm_solver;
        let nlc_grid = self.nlc_grid;
        let restricted = self.restricted;
        let cx = functional.hybrid.cx;
        let clr = functional.hybrid.clr;
        let has_xc = !functional.parts.is_empty();
        let want_k = cx != 0.0;
        let nspin = if restricted { 1 } else { 2 };
        let mut focks = [h_core.clone(), h_core.clone()];
        let p_tot = {
            let mut p = density[0].clone();
            for (t, b) in p.as_mut_slice().iter_mut().zip(density[1].as_slice()) {
                *t += *b;
            }
            p
        };
        let (j, k) = jk.build(mb, &p_tot, [&density[0], &density[1]], want_k)?;
        let k_lr = match &jk_lr {
            Some(engine) => Some(engine.build_k_only(mb, [&density[0], &density[1]])?),
            None => None,
        };
        let xc = if has_xc {
            Some(crate::ks::xc_contribution(
                functional,
                mb,
                grid.expect("grid presence validated"),
                [&density[0], &density[1]],
            )?)
        } else {
            None
        };
        let grac = match (opts.asymptotic_correction, grid) {
            (Some(shift), Some(g)) => Some(crate::ks::grac_correction(
                functional,
                mb,
                g,
                [&density[0], &density[1]],
                shift,
            )?),
            _ => None,
        };
        let nlc = match (&opts.vv10, &nlc_grid) {
            (Some(params), Some(ng)) => Some(crate::nlc::vv10_potential(mb, ng, &p_tot, *params)?),
            _ => None,
        };
        let learned = match (&opts.skala, grid) {
            (Some(model), Some(g)) => Some(crate::skala::skala_fock(
                model,
                mol,
                mb,
                g,
                [&density[0], &density[1]],
            )?),
            _ => None,
        };
        let pcm = match &pcm_solver {
            Some(s) => Some(s.energy_and_fock(&p_tot)?),
            None => None,
        };
        for spin in 0..nspin {
            let mut f = h_core.clone();
            for ((fv, jv), kv) in f
                .as_mut_slice()
                .iter_mut()
                .zip(j.as_slice())
                .zip(k[spin].as_slice())
            {
                *fv += *jv - cx * *kv;
            }
            if let Some(k_lr) = &k_lr {
                for (fv, kv) in f.as_mut_slice().iter_mut().zip(k_lr[spin].as_slice()) {
                    *fv -= clr * *kv;
                }
            }
            if let Some(xc) = &xc {
                for (fv, vv) in f.as_mut_slice().iter_mut().zip(xc.vxc[spin].as_slice()) {
                    *fv += *vv;
                }
            }
            if let Some(correction) = &grac {
                for (fv, vv) in f.as_mut_slice().iter_mut().zip(correction[spin].as_slice()) {
                    *fv += *vv;
                }
            }
            if let Some((_, v_nlc)) = &nlc {
                for (fv, vv) in f.as_mut_slice().iter_mut().zip(v_nlc.as_slice()) {
                    *fv += *vv;
                }
            }
            if let Some((_, v_learned)) = &learned {
                for (fv, vv) in f.as_mut_slice().iter_mut().zip(v_learned[spin].as_slice()) {
                    *fv += *vv;
                }
            }
            if let Some((_, v_pcm)) = &pcm {
                for (fv, vv) in f.as_mut_slice().iter_mut().zip(v_pcm.as_slice()) {
                    *fv += *vv;
                }
            }
            focks[spin] = f;
        }
        if restricted {
            focks[1] = focks[0].clone();
        }

        // Energy of the CURRENT density (explicit form, valid for HF, pure
        // DFT, global and range-separated hybrids):
        // Tr[P H] + ½Tr[P J] − ½ Σσ Tr[Pσ (cx Kσ + clr K_lr,σ)] + Exc.
        let mut e_elec = p_tot.frobenius_dot(h_core) + 0.5 * p_tot.frobenius_dot(&j);
        if want_k {
            for spin in 0..2 {
                e_elec -= 0.5 * cx * density[spin].frobenius_dot(&k[spin]);
            }
        }
        if let Some(k_lr) = &k_lr {
            for spin in 0..2 {
                e_elec -= 0.5 * clr * density[spin].frobenius_dot(&k_lr[spin]);
            }
        }
        if let Some(xc) = &xc {
            e_elec += xc.exc;
        }
        if let Some((e_nlc, _)) = &nlc {
            e_elec += e_nlc;
        }
        if let Some((e_learned, _)) = &learned {
            e_elec += e_learned;
        }
        if let Some((e_pcm, _)) = &pcm {
            e_elec += e_pcm;
        }
        if let Some(DensityExtension::SpinPotential(operators)) = self.extension {
            add_spin_fock(&mut focks, operators);
            e_elec += density
                .iter()
                .zip(operators)
                .map(|(p, v)| p.frobenius_dot(v))
                .sum::<f64>();
        }
        if let Some(DensityExtension::Hubbard(model)) = self.extension {
            let fields = model.evaluate(density)?;
            e_elec += fields.energy;
            add_spin_fock(&mut focks, &fields.potential);
        }

        Ok(MolecularDensityFields {
            fock: focks,
            electronic_energy: e_elec,
            xc_energy: xc.as_ref().map_or(0.0, |x| x.exc)
                + learned.as_ref().map_or(0.0, |(e, _)| *e),
            nonlocal_energy: nlc.as_ref().map_or(0.0, |(e, _)| *e),
            solvent_energy: pcm.as_ref().map_or(0.0, |(e, _)| *e),
            grid_electron_count: xc.as_ref().map(|x| x.n_elec),
        })
    }
}
