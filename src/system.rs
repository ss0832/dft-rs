// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Molecular geometry and XYZ I/O.
//!
//! Positions are stored in **Bohr** internally. `from_xyz_*` reads Ångström
//! (the XYZ convention) and converts on input.

#[cfg(test)]
use crate::constants::ANGSTROM_TO_BOHR;
use crate::constants::BOHR_TO_ANGSTROM;
use crate::error::{DftError, Result};
use crate::math::Vec3;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Atom {
    pub z: u8,
    /// Position in Bohr.
    pub position: Vec3,
}

#[derive(Clone, Debug)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    /// Total molecular charge (electrons removed = positive).
    pub charge: i32,
    /// Spin multiplicity (2S+1). 1 = closed-shell singlet.
    pub multiplicity: usize,
}

impl Molecule {
    pub fn new(atoms: Vec<Atom>) -> Self {
        Self {
            atoms,
            charge: 0,
            multiplicity: 1,
        }
    }

    pub fn with_charge(mut self, charge: i32) -> Self {
        self.charge = charge;
        self
    }

    pub fn with_multiplicity(mut self, multiplicity: usize) -> Self {
        self.multiplicity = multiplicity.max(1);
        self
    }

    pub fn len(&self) -> usize {
        self.atoms.len()
    }
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    /// Sum of atomic numbers.
    pub fn total_nuclear_charge(&self) -> u32 {
        self.atoms.iter().map(|a| a.z as u32).sum()
    }

    /// Total electron count `Σ Z − charge`. Errors on a charge that strips
    /// more electrons than the molecule has.
    pub fn n_electrons(&self) -> Result<usize> {
        let n = self.total_nuclear_charge() as i64 - self.charge as i64;
        if n < 0 {
            return Err(DftError::InvalidInput(format!(
                "charge {} exceeds the total nuclear charge {}",
                self.charge,
                self.total_nuclear_charge()
            )));
        }
        Ok(n as usize)
    }

    /// Derive `(n_alpha, n_beta)` from the electron count and multiplicity,
    /// validating parity: `n_unpaired = multiplicity − 1` must not exceed the
    /// electron count and must share its parity.
    pub fn n_alpha_beta(&self) -> Result<(usize, usize)> {
        self.n_alpha_beta_for(self.n_electrons()?)
    }

    /// `(n_alpha, n_beta)` for an explicit electron count (ECP calculations
    /// remove `Σ n_core` electrons from the treatment).
    pub fn n_alpha_beta_for(&self, n_elec: usize) -> Result<(usize, usize)> {
        let n_unpaired = self.multiplicity - 1;
        if n_unpaired > n_elec {
            return Err(DftError::InvalidInput(format!(
                "multiplicity {} implies {} unpaired electrons but only {} electrons exist",
                self.multiplicity, n_unpaired, n_elec
            )));
        }
        if (n_elec - n_unpaired) % 2 != 0 {
            return Err(DftError::InvalidInput(format!(
                "multiplicity {} is inconsistent with {} electrons (parity mismatch); \
                 check the charge and multiplicity",
                self.multiplicity, n_elec
            )));
        }
        let n_beta = (n_elec - n_unpaired) / 2;
        let n_alpha = n_beta + n_unpaired;
        Ok((n_alpha, n_beta))
    }

    /// Nuclear–nuclear repulsion energy `Σ_{A<B} Z_A Z_B / R_AB` (Hartree).
    pub fn nuclear_repulsion(&self) -> f64 {
        let mut e = 0.0;
        for (i, a) in self.atoms.iter().enumerate() {
            for b in self.atoms.iter().skip(i + 1) {
                let r = (a.position - b.position).norm();
                e += (a.z as f64) * (b.z as f64) / r;
            }
        }
        e
    }

    pub fn from_xyz_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_xyz_str(&fs::read_to_string(path)?)
    }

    /// Parse a standard XYZ block. Coordinates are Ångström, converted to
    /// Bohr. Charge and multiplicity are *not* read from the file — set them
    /// with [`Molecule::with_charge`] / [`Molecule::with_multiplicity`].
    pub fn from_xyz_str(text: &str) -> Result<Self> {
        // Windows editors and PowerShell's `Out-File -Encoding utf8` write a
        // byte-order mark. It is invisible, and without this it lands inside
        // the atom count and the file is rejected for a reason the user
        // cannot see in their editor.
        let text = text.strip_prefix('\u{feff}').unwrap_or(text);
        let mut lines = text.lines();
        let natoms_line = lines
            .next()
            .ok_or_else(|| DftError::InvalidInput("empty XYZ".to_string()))?;
        let natoms = natoms_line.trim().parse::<usize>().map_err(|_| {
            DftError::InvalidInput(format!("invalid XYZ atom count: {natoms_line}"))
        })?;
        let _comment = lines.next().unwrap_or_default();
        let mut atoms = Vec::with_capacity(natoms);
        for idx in 0..natoms {
            let line_no = idx + 3;
            let line = lines
                .next()
                .ok_or_else(|| DftError::InvalidInput(format!("XYZ ended before atom {idx}")))?;
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() < 4 {
                return Err(DftError::InvalidInput(format!(
                    "XYZ line {line_no} has fewer than 4 fields"
                )));
            }
            let z = symbol_to_z(parts[0]).ok_or_else(|| {
                DftError::InvalidInput(format!("unknown element on line {line_no}: {}", parts[0]))
            })?;
            let input = crate::quantities::Position::<crate::quantities::Angstrom>::new([
                parse_f64(parts[1], line_no)?,
                parse_f64(parts[2], line_no)?,
                parse_f64(parts[3], line_no)?,
            ])?;
            let [x, y, z_coord] = input.into_bohr()?.into_array();
            let position = Vec3::new(x, y, z_coord);
            atoms.push(Atom { z, position });
        }
        Ok(Self {
            atoms,
            charge: 0,
            multiplicity: 1,
        })
    }

    /// Serialize to a standard XYZ block (Ångström).
    pub fn to_xyz_string(&self, comment: &str) -> String {
        let mut out = format!("{}\n{}\n", self.len(), comment);
        for atom in &self.atoms {
            let symbol = z_to_symbol(atom.z).unwrap_or("X");
            let pos = atom.position * BOHR_TO_ANGSTROM;
            out.push_str(&format!(
                "{symbol:2} {:+.10} {:+.10} {:+.10}\n",
                pos.x, pos.y, pos.z
            ));
        }
        out
    }
}

fn parse_f64(token: &str, line: usize) -> Result<f64> {
    token.parse::<f64>().map_err(|_| DftError::Parse {
        line,
        message: format!("invalid floating point value: {token}"),
    })
}

pub fn symbol_to_z(sym: &str) -> Option<u8> {
    if let Ok(z) = sym.parse::<u8>() {
        if (1..=118).contains(&z) {
            return Some(z);
        }
    }
    let s = normalize_symbol(sym);
    ELEMENTS
        .iter()
        .position(|&x| x == s.as_str())
        .map(|i| i as u8)
}

pub fn z_to_symbol(z: u8) -> Option<&'static str> {
    ELEMENTS.get(z as usize).copied().filter(|s| !s.is_empty())
}

fn normalize_symbol(sym: &str) -> String {
    let mut chars = sym.chars();
    match chars.next() {
        Some(first) => {
            let mut out = String::new();
            out.push(first.to_ascii_uppercase());
            for c in chars {
                out.push(c.to_ascii_lowercase());
            }
            out
        }
        None => String::new(),
    }
}

pub const ELEMENTS: [&str; 119] = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

#[cfg(test)]
mod tests {
    use super::*;

    const WATER: &str = "3\nwater\nO  0.0000  0.0000  0.0000\nH  0.9584  0.0000  0.0000\nH -0.2400  0.9278  0.0000\n";

    #[test]
    fn parses_water_and_counts_electrons() {
        let mol = Molecule::from_xyz_str(WATER).unwrap();
        assert_eq!(mol.len(), 3);
        assert_eq!(mol.total_nuclear_charge(), 10);
        assert_eq!(mol.n_electrons().unwrap(), 10);
        let (na, nb) = mol.n_alpha_beta().unwrap();
        assert_eq!((na, nb), (5, 5));
        // O–H bond length 0.9584 Å in Bohr.
        let r = (mol.atoms[0].position - mol.atoms[1].position).norm();
        assert!((r - 0.9584 * ANGSTROM_TO_BOHR).abs() < 1e-12);
    }

    #[test]
    fn open_shell_spin_partition() {
        // CH3 radical: 9 electrons, doublet -> (5, 4).
        let mol = Molecule::from_xyz_str(
            "4\nmethyl\nC 0 0 0\nH 1.079 0 0\nH -0.539 0.934 0\nH -0.539 -0.934 0\n",
        )
        .unwrap()
        .with_multiplicity(2);
        assert_eq!(mol.n_alpha_beta().unwrap(), (5, 4));
        // Cation singlet: 8 electrons -> (4, 4).
        let cation = Molecule::from_xyz_str(
            "4\nmethyl\nC 0 0 0\nH 1.079 0 0\nH -0.539 0.934 0\nH -0.539 -0.934 0\n",
        )
        .unwrap()
        .with_charge(1);
        assert_eq!(cation.n_alpha_beta().unwrap(), (4, 4));
    }

    #[test]
    fn parity_mismatch_is_rejected() {
        // Water (10 electrons) as a doublet is impossible.
        let mol = Molecule::from_xyz_str(WATER).unwrap().with_multiplicity(2);
        assert!(mol.n_alpha_beta().is_err());
    }

    #[test]
    fn nuclear_repulsion_h2() {
        // Two protons at 1.4 Bohr: E_nn = 1/1.4.
        let mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::zero(),
            },
            Atom {
                z: 1,
                position: Vec3::new(1.4, 0.0, 0.0),
            },
        ]);
        assert!((mol.nuclear_repulsion() - 1.0 / 1.4).abs() < 1e-15);
    }

    #[test]
    fn xyz_round_trip() {
        let mol = Molecule::from_xyz_str(WATER).unwrap();
        let text = mol.to_xyz_string("round trip");
        // A BOM-prefixed copy must parse identically: this is what Windows
        // editors hand us, and the failure it causes is invisible on screen.
        let with_bom = format!("\u{feff}{text}");
        assert_eq!(
            Molecule::from_xyz_str(&with_bom).unwrap().len(),
            mol.len(),
            "a UTF-8 BOM must not break XYZ parsing"
        );
        let re = Molecule::from_xyz_str(&text).unwrap();
        for (a, b) in mol.atoms.iter().zip(&re.atoms) {
            assert_eq!(a.z, b.z);
            assert!((a.position - b.position).norm() < 1e-9);
        }
    }
}
