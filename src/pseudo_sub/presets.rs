// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! Explicit, idealized monovalent substituent seeds. No fitted operator is loaded.
//!
//! A preset is a discrete structure/attachment choice, not a point in a validated
//! continuous reduced-model chart. Geometry is fixed at construction and is never
//! regenerated during a parameter path. Ordinary atom labels remain integers.
//! Methyl-capped references are neutral closed-shell molecules; the uncapped
//! open-valence fragment is deliberately not offered as a singlet SCF input.

use crate::constants::ANGSTROM_TO_BOHR;
use crate::math::Vec3;
use crate::system::{Atom, Molecule};
use crate::{DftError, Result};
use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CatalogOrigin {
    RequestedBaseline,
    ExistingExtra,
    Additional,
}

#[derive(Clone, Copy, Debug)]
pub struct PresetDescriptor {
    pub id: &'static str,
    pub aliases: &'static [&'static str],
    pub attachment: &'static str,
    pub coverage: &'static str,
    pub origin: CatalogOrigin,
    /// Initial C(sp3)-X seed length, not an optimized or measured bond length.
    pub methyl_bond_angstrom: f64,
}

macro_rules! entry {
    ($id:literal, [$($alias:literal),*], $port:literal, $feature:literal, $origin:ident, $r:expr) => {
        PresetDescriptor { id:$id, aliases:&[$($alias),*], attachment:$port,
            coverage:$feature, origin:CatalogOrigin::$origin, methyl_bond_angstrom:$r }
    };
}

const CATALOG: &[PresetDescriptor] = &[
    entry!(
        "methyl",
        ["Me"],
        "C(sp3)",
        "compact alkyl",
        RequestedBaseline,
        1.52
    ),
    entry!(
        "ethyl",
        ["Et"],
        "terminal C(sp3)",
        "flexible alkyl",
        RequestedBaseline,
        1.52
    ),
    entry!(
        "tert_butyl",
        ["tBu", "tert-butyl"],
        "tertiary C(sp3)",
        "branched steric volume",
        RequestedBaseline,
        1.54
    ),
    entry!(
        "phenyl_c",
        ["Ph", "phenyl"],
        "aryl carbon",
        "planar aromatic",
        RequestedBaseline,
        1.50
    ),
    entry!(
        "trifluoromethyl",
        ["CF3"],
        "carbon",
        "withdrawing fluoroalkyl",
        RequestedBaseline,
        1.52
    ),
    entry!(
        "chloro",
        ["Cl"],
        "chlorine",
        "ordinary halogen identity",
        RequestedBaseline,
        1.78
    ),
    entry!(
        "methoxy_o",
        ["OMe", "methoxy"],
        "oxygen",
        "ether donor/acceptor",
        ExistingExtra,
        1.43
    ),
    entry!(
        "fluoro",
        ["F"],
        "fluorine",
        "compact electronegative halogen",
        Additional,
        1.38
    ),
    entry!(
        "bromo",
        ["Br"],
        "bromine",
        "larger polarizable halogen",
        Additional,
        1.94
    ),
    entry!(
        "iodo",
        ["I"],
        "iodine",
        "heavy polarizable halogen",
        Additional,
        2.14
    ),
    entry!(
        "hydroxy_o",
        ["OH", "hydroxy"],
        "oxygen; neutral O-H",
        "hydrogen-bond donor/acceptor",
        Additional,
        1.43
    ),
    entry!(
        "trifluoromethoxy_o",
        ["OCF3", "trifluoromethoxy"],
        "oxygen",
        "withdrawing flexible ether",
        Additional,
        1.43
    ),
    entry!(
        "phenoxy_o",
        ["OPh", "phenoxy"],
        "oxygen, not aryl carbon",
        "aromatic ether and torsion",
        Additional,
        1.43
    ),
    entry!(
        "amino_n",
        ["NH2", "amino"],
        "nitrogen; neutral NH2",
        "amine donor and hydrogen bonding",
        Additional,
        1.47
    ),
    entry!(
        "dimethylamino_n",
        ["NMe2", "dimethylamino"],
        "nitrogen",
        "donor without N-H",
        Additional,
        1.47
    ),
    entry!(
        "nitro_n",
        ["NO2", "nitro"],
        "N(+); one O(-), one N=O",
        "strongly polar withdrawing",
        Additional,
        1.48
    ),
    entry!(
        "cyano_c",
        ["CN", "cyano"],
        "carbon of C#N",
        "compact linear acceptor",
        Additional,
        1.47
    ),
    entry!(
        "acetyl_c",
        ["COMe", "acetyl"],
        "carbonyl carbon",
        "carbonyl acceptor",
        Additional,
        1.52
    ),
    entry!(
        "carboxy_c",
        ["COOH", "carboxy"],
        "carbonyl carbon; neutral COOH",
        "protonated acid donor/acceptor",
        Additional,
        1.52
    ),
    entry!(
        "methoxycarbonyl_c",
        ["COOMe", "methoxycarbonyl"],
        "carbonyl carbon",
        "ester through carbon",
        Additional,
        1.52
    ),
    entry!(
        "acetoxy_o",
        ["OAc", "acetoxy"],
        "oxygen of O-C(=O)-Me",
        "ester through oxygen",
        Additional,
        1.43
    ),
    entry!(
        "carbamoyl_c",
        ["CONH2", "carbamoyl"],
        "carbonyl carbon; C(=O)NH2",
        "amide donor/acceptor",
        Additional,
        1.52
    ),
    entry!(
        "vinyl_c",
        ["vinyl"],
        "CH of CH=CH2",
        "planar unsaturation",
        Additional,
        1.50
    ),
    entry!(
        "ethynyl_c",
        ["ethynyl"],
        "carbon of C#CH",
        "linear unsaturation",
        Additional,
        1.46
    ),
    entry!(
        "cyclopropyl_c",
        ["cyclopropyl"],
        "ring CH",
        "compact strained three-dimensional ring",
        Additional,
        1.52
    ),
    entry!(
        "pyridin_3_yl_c",
        ["3-pyridyl", "pyridin-3-yl"],
        "C3; unprotonated ring N",
        "low-symmetry aromatic heterocycle",
        Additional,
        1.50
    ),
    entry!(
        "methylsulfonyl_s",
        ["SO2Me", "methylsulfonyl"],
        "sulfur of S(=O)2-Me",
        "tetrahedral polar sulfur",
        Additional,
        1.78
    ),
];

pub fn catalog() -> &'static [PresetDescriptor] {
    CATALOG
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresetBond {
    pub atoms: [usize; 2],
    pub order: u8,
}

#[derive(Clone, Debug)]
pub struct PresetGeometry {
    pub descriptor: PresetDescriptor,
    /// Actual atoms in Bohr; no auxiliary geometry point is a nucleus.
    pub atoms: Vec<Atom>,
    pub formal_charges: Vec<i8>,
    pub bonds: Vec<PresetBond>,
    pub attachment_atom: usize,
    /// Fixed pose metadata, transported together with coordinates.
    pub outward: Vec3,
    pub transverse: Vec3,
}

#[derive(Clone, Debug)]
pub struct MethylReference {
    pub descriptor: PresetDescriptor,
    pub molecule: Molecule,
    /// Formal bonding annotations; the SCF uses the molecule's total charge.
    pub formal_charges: Vec<i8>,
    pub bonds: Vec<PresetBond>,
    pub fragment_atoms: Vec<usize>,
    pub retained_methyl_atoms: [usize; 4],
    /// Fragment atom, host carbon, host reference hydrogen; zero-based.
    pub port: [usize; 3],
}

fn invalid<T>(message: &str) -> Result<T> {
    Err(DftError::InvalidInput(message.into()))
}
fn finite(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}
fn unit(v: Vec3) -> Result<Vec3> {
    let norm = v.norm();
    if !norm.is_finite() || norm < 1e-12 {
        return invalid("degenerate preset seed direction");
    }
    Ok(v / norm)
}
fn seed_perpendicular(v: Vec3) -> Result<Vec3> {
    // Private canonical-seed construction only. No fallback or axis reselection
    // is used during a continuous chart or a posed-geometry transformation.
    unit(v.cross(Vec3::new(0.0, 0.0, 1.0)))
}
fn cone(v: Vec3) -> Result<[Vec3; 3]> {
    let v = unit(v)?;
    let e = seed_perpendicular(v)?;
    let f = v.cross(e);
    Ok(std::array::from_fn(|i| {
        let p = 2.0 * std::f64::consts::PI * i as f64 / 3.0;
        v * (-1.0 / 3.0) + (e * p.cos() + f * p.sin()) * (8.0_f64 / 9.0).sqrt()
    }))
}
fn valence(z: u8, q: i8) -> Result<u8> {
    match (z, q) {
        (1 | 9 | 17 | 35 | 53, 0) => Ok(1),
        (6, 0) => Ok(4),
        (7, 0) => Ok(3),
        (7, 1) => Ok(4),
        (8, 0) => Ok(2),
        (8, -1) => Ok(1),
        (16, 0) => Ok(6),
        _ => invalid("unsupported preset valence/charge"),
    }
}

impl PresetGeometry {
    pub fn validate(&self) -> Result<()> {
        let n = self.atoms.len();
        if n == 0
            || self.attachment_atom >= n
            || self.formal_charges.len() != n
            || self.atoms.iter().any(|a| a.z == 0 || !finite(a.position))
            || self
                .formal_charges
                .iter()
                .map(|&q| i32::from(q))
                .sum::<i32>()
                != 0
        {
            return invalid("invalid neutral preset atoms, charges or attachment");
        }
        if !finite(self.outward)
            || !finite(self.transverse)
            || (self.outward.norm() - 1.0).abs() > 1e-12
            || (self.transverse.norm() - 1.0).abs() > 1e-12
            || self.outward.dot(self.transverse).abs() > 1e-12
        {
            return invalid("invalid preset pose directions");
        }
        let mut totals = vec![0u8; n];
        totals[self.attachment_atom] = 1;
        let mut seen = HashSet::new();
        let mut adjacency = vec![Vec::new(); n];
        for bond in &self.bonds {
            let [a, b] = bond.atoms;
            if a >= n
                || b >= n
                || a == b
                || !(1..=3).contains(&bond.order)
                || !seen.insert((a.min(b), a.max(b)))
            {
                return invalid("invalid or duplicate preset bond");
            }
            totals[a] = totals[a]
                .checked_add(bond.order)
                .ok_or_else(|| DftError::InvalidInput("preset valence overflow".into()))?;
            totals[b] = totals[b]
                .checked_add(bond.order)
                .ok_or_else(|| DftError::InvalidInput("preset valence overflow".into()))?;
            adjacency[a].push(b);
            adjacency[b].push(a);
        }
        for (i, atom) in self.atoms.iter().enumerate() {
            if totals[i] != valence(atom.z, self.formal_charges[i])? {
                return invalid("preset bond valence mismatch");
            }
            for other in &self.atoms[..i] {
                if (atom.position - other.position).norm() < 0.8 {
                    return invalid("overlapping preset nuclei");
                }
            }
        }
        let mut visited = vec![false; n];
        let mut pending = vec![self.attachment_atom];
        while let Some(i) = pending.pop() {
            if visited[i] {
                continue;
            }
            visited[i] = true;
            pending.extend(&adjacency[i]);
        }
        if visited.iter().any(|v| !*v) {
            return invalid("disconnected preset structure");
        }
        Ok(())
    }
    /// Structure registration has no coefficient vector, frozen-electron ledger
    /// or compatible interpolation chart. A validated generator must supply them.
    pub fn reduced_model_available(&self) -> bool {
        false
    }
    pub fn element_counts(&self) -> BTreeMap<u8, usize> {
        let mut counts = BTreeMap::new();
        for a in &self.atoms {
            *counts.entry(a.z).or_insert(0) += 1;
        }
        counts
    }
    pub fn methyl_capped_reference(&self) -> Result<MethylReference> {
        self.validate()?;
        let mut atoms = self.atoms.clone();
        let mut bonds = self.bonds.clone();
        let c = atoms.len();
        let center = atoms[self.attachment_atom].position
            + self.outward * (self.descriptor.methyl_bond_angstrom * ANGSTROM_TO_BOHR);
        atoms.push(Atom {
            z: 6,
            position: center,
        });
        bonds.push(PresetBond {
            atoms: [self.attachment_atom, c],
            order: 1,
        });
        let v = self.outward * (-1.0);
        let e = self.transverse;
        let f = v.cross(e);
        for i in 0..3 {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / 3.0;
            let direction =
                v * (-1.0 / 3.0) + (e * angle.cos() + f * angle.sin()) * (8.0_f64 / 9.0).sqrt();
            atoms.push(Atom {
                z: 1,
                position: center + direction * (1.09 * ANGSTROM_TO_BOHR),
            });
            bonds.push(PresetBond {
                atoms: [c, c + i + 1],
                order: 1,
            });
        }
        let molecule = Molecule::new(atoms);
        let mut formal_charges = self.formal_charges.clone();
        formal_charges.extend([0; 4]);
        if molecule.n_electrons()? % 2 != 0 {
            return invalid("methyl reference is not closed-shell compatible");
        }
        for i in 0..molecule.len() {
            for j in 0..i {
                if (molecule.atoms[i].position - molecule.atoms[j].position).norm() < 0.8 {
                    return invalid("methyl cap overlaps preset nuclei");
                }
            }
        }
        Ok(MethylReference {
            descriptor: self.descriptor,
            molecule,
            formal_charges,
            bonds,
            fragment_atoms: (0..c).collect(),
            retained_methyl_atoms: [c, c + 1, c + 2, c + 3],
            port: [self.attachment_atom, c, c + 1],
        })
    }
    /// A caller-supplied proper rotation transports the existing pose; it does
    /// not regenerate hydrogens or choose new geometry-derived frame axes.
    pub fn transformed(&self, r: [[f64; 3]; 3], translation: Vec3) -> Result<Self> {
        if !finite(translation) || r.iter().flatten().any(|x| !x.is_finite()) {
            return invalid("nonfinite preset transform");
        }
        for i in 0..3 {
            for j in 0..3 {
                let dot = (0..3).map(|k| r[k][i] * r[k][j]).sum::<f64>();
                if (dot - if i == j { 1.0 } else { 0.0 }).abs() > 1e-12 {
                    return invalid("preset rotation is not orthogonal");
                }
            }
        }
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        if (det - 1.0).abs() > 1e-12 {
            return invalid("preset transform must be a proper rotation");
        }
        let rotate = |v: Vec3| {
            Vec3::new(
                r[0][0] * v.x + r[0][1] * v.y + r[0][2] * v.z,
                r[1][0] * v.x + r[1][1] * v.y + r[1][2] * v.z,
                r[2][0] * v.x + r[2][1] * v.y + r[2][2] * v.z,
            )
        };
        let mut out = self.clone();
        for a in &mut out.atoms {
            a.position = rotate(a.position) + translation;
        }
        out.outward = rotate(out.outward);
        out.transverse = rotate(out.transverse);
        out.validate()?;
        Ok(out)
    }
}

#[derive(Clone, Copy)]
enum Hybrid {
    Sp,
    Sp2,
    Sp3,
}
struct Builder {
    atoms: Vec<Atom>,
    charges: Vec<i8>,
    hybrid: Vec<Hybrid>,
    bonds: Vec<PresetBond>,
}
impl Builder {
    fn new(z: u8, h: Hybrid) -> Self {
        Self {
            atoms: vec![Atom {
                z,
                position: Vec3::zero(),
            }],
            charges: vec![0],
            hybrid: vec![h],
            bonds: Vec::new(),
        }
    }
    fn atom(&mut self, z: u8, p: Vec3, h: Hybrid, q: i8) -> usize {
        let i = self.atoms.len();
        self.atoms.push(Atom { z, position: p });
        self.charges.push(q);
        self.hybrid.push(h);
        i
    }
    fn bond(&mut self, a: usize, b: usize, order: u8) {
        self.bonds.push(PresetBond {
            atoms: [a, b],
            order,
        });
    }
    fn child(
        &mut self,
        parent: usize,
        z: u8,
        d: Vec3,
        length: f64,
        h: Hybrid,
        q: i8,
        order: u8,
    ) -> usize {
        let i = self.atom(
            z,
            self.atoms[parent].position + d * (length * ANGSTROM_TO_BOHR),
            h,
            q,
        );
        self.bond(parent, i, order);
        i
    }
    fn finish(mut self, d: PresetDescriptor) -> Result<PresetGeometry> {
        let heavy = self.atoms.len();
        for i in 0..heavy {
            let mut directions = Vec::new();
            let mut used = 0u8;
            if i == 0 {
                directions.push(Vec3::new(-1.0, 0.0, 0.0));
                used = 1;
            }
            for b in &self.bonds {
                if b.atoms.contains(&i) {
                    let j = if b.atoms[0] == i {
                        b.atoms[1]
                    } else {
                        b.atoms[0]
                    };
                    used += b.order;
                    directions.push(unit(self.atoms[j].position - self.atoms[i].position)?);
                }
            }
            let target = valence(self.atoms[i].z, self.charges[i])?;
            let count = target
                .checked_sub(used)
                .ok_or_else(|| DftError::InvalidInput("overbonded preset seed".into()))?;
            if count == 0 {
                continue;
            }
            let hs = match (self.hybrid[i], directions.len(), count) {
                (Hybrid::Sp, 1, 1) => vec![directions[0] * (-1.0)],
                (Hybrid::Sp2, 1, 1 | 2) => {
                    let v = directions[0];
                    let e = seed_perpendicular(v)?;
                    let a = v * (-0.5) + e * (3.0_f64.sqrt() / 2.0);
                    if count == 1 {
                        vec![a]
                    } else {
                        vec![a, v * (-0.5) - e * (3.0_f64.sqrt() / 2.0)]
                    }
                }
                (Hybrid::Sp2, 2, 1) => vec![unit((directions[0] + directions[1]) * (-1.0))?],
                (Hybrid::Sp3, 1, 1..=3) => cone(directions[0])?[..usize::from(count)].to_vec(),
                (Hybrid::Sp3, 2, 2) => {
                    let a = directions[0];
                    let b = directions[1];
                    let denominator = 3.0 * (1.0 + a.dot(b));
                    if denominator.abs() < 1e-12 {
                        return invalid("collinear tetrahedral seed");
                    }
                    let mean = (a + b) * (-1.0 / denominator);
                    let norm2 = mean.dot(mean);
                    if norm2 >= 1.0 {
                        return invalid("invalid tetrahedral seed angles");
                    }
                    let e = unit(a.cross(b))? * (1.0 - norm2).sqrt();
                    vec![mean + e, mean - e]
                }
                (Hybrid::Sp3, 3, 1) => vec![unit(
                    directions.iter().copied().fold(Vec3::zero(), |a, b| a + b) * (-1.0),
                )?],
                _ => return invalid("unsupported preset hydrogen completion"),
            };
            let length = match self.atoms[i].z {
                6 => {
                    if matches!(self.hybrid[i], Hybrid::Sp) {
                        1.06
                    } else {
                        1.09
                    }
                }
                7 => 1.01,
                8 => 0.97,
                _ => return invalid("unsupported hydrogen-bearing preset atom"),
            };
            for direction in hs {
                self.child(i, 1, direction, length, Hybrid::Sp3, 0, 1);
            }
        }
        let out = PresetGeometry {
            descriptor: d,
            atoms: self.atoms,
            formal_charges: self.charges,
            bonds: self.bonds,
            attachment_atom: 0,
            outward: Vec3::new(-1.0, 0.0, 0.0),
            transverse: Vec3::new(0.0, 1.0, 0.0),
        };
        out.validate()?;
        Ok(out)
    }
}

fn ring(b: &mut Builder, root: usize, u: Vec3, v: Vec3, nitrogen: bool) {
    b.hybrid[root] = Hybrid::Sp2;
    let origin = b.atoms[root].position;
    let h = 3.0_f64.sqrt() / 2.0;
    let coordinates = [
        (0.0, 0.0),
        (0.5, h),
        (1.5, h),
        (2.0, 0.0),
        (1.5, -h),
        (0.5, -h),
    ];
    let mut ids = [root; 6];
    for i in 1..6 {
        ids[i] = b.atom(
            if nitrogen && i == 4 { 7 } else { 6 },
            origin + (u * coordinates[i].0 + v * coordinates[i].1) * (1.39 * ANGSTROM_TO_BOHR),
            Hybrid::Sp2,
            0,
        );
    }
    for i in 0..6 {
        b.bond(ids[i], ids[(i + 1) % 6], if i % 2 == 0 { 2 } else { 1 });
    }
}

pub fn get(name: &str) -> Result<PresetGeometry> {
    let key = name.trim().to_ascii_lowercase();
    let d = *CATALOG
        .iter()
        .find(|d| d.id == key || d.aliases.iter().any(|a| a.eq_ignore_ascii_case(&key)))
        .ok_or_else(|| {
            DftError::InvalidInput(format!(
                "unknown or ambiguous explicit substituent preset `{name}`"
            ))
        })?;
    let ex = Vec3::new(-1.0, 0.0, 0.0);
    let plus = Vec3::new(0.5, 3.0_f64.sqrt() / 2.0, 0.0);
    let minus = Vec3::new(0.5, -3.0_f64.sqrt() / 2.0, 0.0);
    let tetra = cone(ex)?;
    let mut b = match d.id {
        "fluoro" => Builder::new(9, Hybrid::Sp3),
        "chloro" => Builder::new(17, Hybrid::Sp3),
        "bromo" => Builder::new(35, Hybrid::Sp3),
        "iodo" => Builder::new(53, Hybrid::Sp3),
        "hydroxy_o" | "methoxy_o" | "trifluoromethoxy_o" | "phenoxy_o" | "acetoxy_o" => {
            Builder::new(8, Hybrid::Sp3)
        }
        "amino_n" | "dimethylamino_n" => Builder::new(7, Hybrid::Sp3),
        "nitro_n" => Builder::new(7, Hybrid::Sp2),
        "cyano_c" | "ethynyl_c" => Builder::new(6, Hybrid::Sp),
        "methylsulfonyl_s" => Builder::new(16, Hybrid::Sp3),
        _ => Builder::new(6, Hybrid::Sp3),
    };
    match d.id {
        "methyl" | "fluoro" | "chloro" | "bromo" | "iodo" | "hydroxy_o" | "amino_n" => {}
        "ethyl" => {
            b.child(0, 6, tetra[0], 1.52, Hybrid::Sp3, 0, 1);
        }
        "tert_butyl" => {
            for direction in tetra {
                b.child(0, 6, direction, 1.54, Hybrid::Sp3, 0, 1);
            }
        }
        "trifluoromethyl" => {
            for direction in tetra {
                b.child(0, 9, direction, 1.33, Hybrid::Sp3, 0, 1);
            }
        }
        "methoxy_o" => {
            b.child(0, 6, tetra[0], 1.43, Hybrid::Sp3, 0, 1);
        }
        "trifluoromethoxy_o" => {
            let c = b.child(0, 6, tetra[0], 1.43, Hybrid::Sp3, 0, 1);
            for direction in cone(tetra[0] * (-1.0))? {
                b.child(c, 9, direction, 1.33, Hybrid::Sp3, 0, 1);
            }
        }
        "phenyl_c" => ring(
            &mut b,
            0,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            false,
        ),
        "pyridin_3_yl_c" => ring(
            &mut b,
            0,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            true,
        ),
        "phenoxy_o" => {
            let c = b.child(0, 6, tetra[0], 1.36, Hybrid::Sp2, 0, 1);
            ring(
                &mut b,
                c,
                tetra[0],
                Vec3::new(-tetra[0].y, tetra[0].x, 0.0),
                false,
            );
        }
        "dimethylamino_n" => {
            for direction in &tetra[..2] {
                b.child(0, 6, *direction, 1.47, Hybrid::Sp3, 0, 1);
            }
        }
        "nitro_n" => {
            b.charges[0] = 1;
            b.child(0, 8, plus, 1.22, Hybrid::Sp2, 0, 2);
            b.child(0, 8, minus, 1.24, Hybrid::Sp2, -1, 1);
        }
        "cyano_c" => {
            b.child(0, 7, Vec3::new(1.0, 0.0, 0.0), 1.16, Hybrid::Sp, 0, 3);
        }
        "ethynyl_c" => {
            b.child(0, 6, Vec3::new(1.0, 0.0, 0.0), 1.20, Hybrid::Sp, 0, 3);
        }
        "vinyl_c" => {
            b.hybrid[0] = Hybrid::Sp2;
            b.child(0, 6, plus, 1.34, Hybrid::Sp2, 0, 2);
        }
        "acetyl_c" | "carboxy_c" | "methoxycarbonyl_c" | "carbamoyl_c" => {
            b.hybrid[0] = Hybrid::Sp2;
            b.child(0, 8, plus, 1.21, Hybrid::Sp2, 0, 2);
            match d.id {
                "acetyl_c" => {
                    b.child(0, 6, minus, 1.51, Hybrid::Sp3, 0, 1);
                }
                "carboxy_c" => {
                    b.child(0, 8, minus, 1.35, Hybrid::Sp2, 0, 1);
                }
                "carbamoyl_c" => {
                    b.child(0, 7, minus, 1.35, Hybrid::Sp2, 0, 1);
                }
                _ => {
                    let o = b.child(0, 8, minus, 1.35, Hybrid::Sp3, 0, 1);
                    b.child(o, 6, cone(minus * (-1.0))?[0], 1.43, Hybrid::Sp3, 0, 1);
                }
            }
        }
        "acetoxy_o" => {
            let c = b.child(0, 6, tetra[0], 1.36, Hybrid::Sp2, 0, 1);
            let p =
                tetra[0] * 0.5 + seed_perpendicular(tetra[0] * (-1.0))? * (3.0_f64.sqrt() / 2.0);
            let m =
                tetra[0] * 0.5 - seed_perpendicular(tetra[0] * (-1.0))? * (3.0_f64.sqrt() / 2.0);
            b.child(c, 8, p, 1.21, Hybrid::Sp2, 0, 2);
            b.child(c, 6, m, 1.51, Hybrid::Sp3, 0, 1);
        }
        "cyclopropyl_c" => {
            let y = 23.0_f64.sqrt() / 6.0;
            let a = b.child(0, 6, Vec3::new(1.0 / 3.0, y, 0.5), 1.51, Hybrid::Sp3, 0, 1);
            let c = b.child(0, 6, Vec3::new(1.0 / 3.0, y, -0.5), 1.51, Hybrid::Sp3, 0, 1);
            b.bond(a, c, 1);
        }
        "methylsulfonyl_s" => {
            b.child(0, 8, tetra[0], 1.44, Hybrid::Sp2, 0, 2);
            b.child(0, 8, tetra[1], 1.44, Hybrid::Sp2, 0, 2);
            b.child(0, 6, tetra[2], 1.78, Hybrid::Sp3, 0, 1);
        }
        _ => return invalid("unimplemented explicit preset seed"),
    }
    b.finish(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn catalog_inventory_and_neutral_closed_shell_references() {
        assert_eq!(catalog().len(), 27);
        assert_eq!(
            catalog()
                .iter()
                .filter(|d| d.origin == CatalogOrigin::Additional)
                .count(),
            20
        );
        assert_eq!(
            catalog()
                .iter()
                .filter(|d| d.origin == CatalogOrigin::RequestedBaseline)
                .count(),
            6
        );
        let mut seen = HashSet::new();
        for d in catalog() {
            assert!(seen.insert(d.id));
            let seed = get(d.id).unwrap_or_else(|e| panic!("{}: {e}", d.id));
            seed.validate().unwrap();
            assert!(!seed.reduced_model_available());
            let capped = seed
                .methyl_capped_reference()
                .unwrap_or_else(|e| panic!("{}: {e}", d.id));
            assert_eq!(capped.molecule.charge, 0);
            assert_eq!(capped.molecule.multiplicity, 1);
            assert_eq!(capped.descriptor.id, d.id);
            assert_eq!(capped.formal_charges.len(), capped.molecule.len());
            assert_eq!(
                capped.formal_charges[..seed.atoms.len()],
                seed.formal_charges
            );
            assert_eq!(
                capped
                    .formal_charges
                    .iter()
                    .map(|&q| i32::from(q))
                    .sum::<i32>(),
                0
            );
            assert_eq!(capped.molecule.n_electrons().unwrap() % 2, 0);
            assert_eq!(capped.fragment_atoms.len() + 4, capped.molecule.len());
            assert_eq!(capped.port[0], 0);
            for alias in d.aliases {
                assert_eq!(get(alias).unwrap().descriptor.id, d.id);
            }
        }
    }
    #[test]
    fn catalog_attachment_and_charge_are_explicit() {
        for (name, z) in [
            ("acetyl", 6),
            ("OAc", 8),
            ("COOMe", 6),
            ("CONH2", 6),
            ("phenoxy", 8),
            ("phenyl", 6),
            ("3-pyridyl", 6),
            ("NO2", 7),
            ("SO2Me", 16),
        ] {
            assert_eq!(get(name).unwrap().atoms[0].z, z);
        }
        let n = get("nitro").unwrap();
        assert_eq!(n.formal_charges[..3], [1, 0, -1]);
        let pyr = get("3-pyridyl").unwrap();
        assert_eq!(pyr.atoms[4].z, 7);
        assert_eq!(pyr.element_counts().get(&1), Some(&4));
        for bad in ["", "amide", "pyridyl", "17", "unregistered"] {
            assert!(get(bad).is_err());
        }
    }
    #[test]
    fn catalog_formulas_match_independent_fragment_definitions() {
        let expected: &[(&str, &[(u8, usize)])] = &[
            ("methyl", &[(1, 3), (6, 1)]),
            ("ethyl", &[(1, 5), (6, 2)]),
            ("tert_butyl", &[(1, 9), (6, 4)]),
            ("phenyl_c", &[(1, 5), (6, 6)]),
            ("trifluoromethyl", &[(6, 1), (9, 3)]),
            ("chloro", &[(17, 1)]),
            ("methoxy_o", &[(1, 3), (6, 1), (8, 1)]),
            ("fluoro", &[(9, 1)]),
            ("bromo", &[(35, 1)]),
            ("iodo", &[(53, 1)]),
            ("hydroxy_o", &[(1, 1), (8, 1)]),
            ("trifluoromethoxy_o", &[(6, 1), (8, 1), (9, 3)]),
            ("phenoxy_o", &[(1, 5), (6, 6), (8, 1)]),
            ("amino_n", &[(1, 2), (7, 1)]),
            ("dimethylamino_n", &[(1, 6), (6, 2), (7, 1)]),
            ("nitro_n", &[(7, 1), (8, 2)]),
            ("cyano_c", &[(6, 1), (7, 1)]),
            ("acetyl_c", &[(1, 3), (6, 2), (8, 1)]),
            ("carboxy_c", &[(1, 1), (6, 1), (8, 2)]),
            ("methoxycarbonyl_c", &[(1, 3), (6, 2), (8, 2)]),
            ("acetoxy_o", &[(1, 3), (6, 2), (8, 2)]),
            ("carbamoyl_c", &[(1, 2), (6, 1), (7, 1), (8, 1)]),
            ("vinyl_c", &[(1, 3), (6, 2)]),
            ("ethynyl_c", &[(1, 1), (6, 2)]),
            ("cyclopropyl_c", &[(1, 5), (6, 3)]),
            ("pyridin_3_yl_c", &[(1, 4), (6, 5), (7, 1)]),
            ("methylsulfonyl_s", &[(1, 3), (6, 1), (8, 2), (16, 1)]),
        ];
        assert_eq!(expected.len(), catalog().len());
        for (name, counts) in expected {
            let actual = get(name).unwrap().element_counts();
            assert_eq!(actual, counts.iter().copied().collect(), "{name}");
        }
    }
    #[test]
    fn catalog_invalid_geometry_and_topology_are_rejected() {
        let seed = get("Me").unwrap();
        let mut bad = seed.clone();
        bad.atoms[0].position.x = f64::NAN;
        assert!(bad.validate().is_err());
        let mut bad = seed.clone();
        bad.bonds.push(bad.bonds[0]);
        assert!(bad.validate().is_err());
        let mut bad = seed.clone();
        bad.bonds[0].order = 0;
        assert!(bad.validate().is_err());
        let mut bad = seed;
        bad.transverse = bad.outward;
        assert!(bad.validate().is_err());
    }
    #[test]
    fn catalog_pose_transport_commutes_with_methyl_capping() {
        let angle = 0.37_f64;
        let r = [
            [angle.cos(), -angle.sin(), 0.0],
            [angle.sin(), angle.cos(), 0.0],
            [0.0, 0.0, 1.0],
        ];
        let t = Vec3::new(0.31, -0.27, 0.19);
        for name in ["OH", "OPh", "cyclopropyl", "3-pyridyl"] {
            let p = get(name).unwrap();
            let a = p.methyl_capped_reference().unwrap();
            let b = p
                .transformed(r, t)
                .unwrap()
                .methyl_capped_reference()
                .unwrap();
            for (u, v) in a.molecule.atoms.iter().zip(&b.molecule.atoms) {
                let expected = Vec3::new(
                    r[0][0] * u.position.x + r[0][1] * u.position.y,
                    r[1][0] * u.position.x + r[1][1] * u.position.y,
                    u.position.z,
                ) + t;
                assert!((expected - v.position).norm() < 1e-12);
            }
        }
    }
}
