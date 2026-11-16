// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Physical constants and unit conversions.
//!
//! Internal unit policy: energies in **Hartree**, distances in **Bohr**
//! (atomic units) throughout the computational core — the natural units of
//! the Gaussian-integral and DFT literature, and the units used by the
//! PySCF oracle data. XYZ files are Ångström and converted on input/output.
//!
//! Values are **2018 CODATA** (E. Tiesinga, P. J. Mohr, D. B. Newell,
//! B. N. Taylor, Rev. Mod. Phys. 93, 025010 (2021)). Do not silently swap in
//! a newer CODATA revision: oracle agreement tests pin these digits.

/// Bohr radius in Ångström (2018 CODATA).
pub const BOHR_TO_ANGSTROM: f64 = 0.529177210903;
pub const ANGSTROM_TO_BOHR: f64 = 1.0 / BOHR_TO_ANGSTROM;

/// One Hartree in eV (2018 CODATA).
pub const HARTREE_TO_EV: f64 = 27.211386245988;
pub const EV_TO_HARTREE: f64 = 1.0 / HARTREE_TO_EV;

/// One Hartree in kcal/mol (2018 CODATA derived).
pub const HARTREE_TO_KCAL_MOL: f64 = 627.509474063;
pub const KCAL_MOL_TO_HARTREE: f64 = 1.0 / HARTREE_TO_KCAL_MOL;

/// One Hartree in joule (2018 CODATA).
pub const HARTREE_TO_JOULE: f64 = 4.3597447222071e-18;

/// Bohr radius in metre (2018 CODATA).
pub const BOHR_TO_METER: f64 = 0.529177210903e-10;

/// Electron rest mass in kilograms (CODATA 2018).
pub const ELECTRON_MASS_KG: f64 = 9.1093837015e-31;

/// Elementary charge in coulomb (exact in the 2019 SI, CODATA 2018).
pub const ELEMENTARY_CHARGE_C: f64 = 1.602176634e-19;

/// Reduced Planck constant in J s (CODATA 2018).
pub const HBAR_J_S: f64 = 1.054571817e-34;

/// Boltzmann constant in J/K (exact in the 2019 SI, CODATA 2018).
pub const BOLTZMANN_J_PER_K: f64 = 1.380649e-23;

/// Avogadro constant in 1/mol (exact in the 2019 SI).
pub const AVOGADRO_PER_MOL: f64 = 6.02214076e23;

/// Magnitude of the free-electron g factor (CODATA 2018).
pub const ELECTRON_G_FACTOR: f64 = 2.00231930436256;

/// Proton mass in electron masses (CODATA 2018).
pub const PROTON_ELECTRON_MASS_RATIO: f64 = 1836.15267343;

/// Muon mass in electron masses (CODATA 2018).
pub const MUON_ELECTRON_MASS_RATIO: f64 = 206.7682830;

/// The atomic unit of time in seconds, hbar / E_h (CODATA 2018).
pub const AU_TIME_SECONDS: f64 = 2.4188843265857e-17;

/// One Hartree/Bohr^3 (stress, pressure) in GPa.
pub const HARTREE_PER_BOHR3_TO_GPA: f64 =
    HARTREE_TO_JOULE / (BOHR_TO_METER * BOHR_TO_METER * BOHR_TO_METER) * 1e-9;

/// One Hartree in kJ/mol.
pub const HARTREE_TO_KJ_MOL: f64 = 2625.4996394799;

/// One Hartree in cm⁻¹ (hc·1 Hartree; used in vibrational analysis).
pub const HARTREE_TO_CM1: f64 = 219474.63136320;

/// Dipole conversion: one e·Bohr in Debye.
pub const AU_DIPOLE_TO_DEBYE: f64 = 2.541746473;

/// Fine-structure constant α (2018 CODATA: 7.2973525693e-3).
///
/// In Hartree atomic units the magnetic vector potential enters the
/// Hamiltonian as `(1/2)(p + αA)²`, so a magnetic property of order n in
/// `A` carries `α^n`: NMR shieldings and magnetizabilities are `α²` times
/// the quantity the integrals produce.
pub const FINE_STRUCTURE: f64 = 7.2973525693e-3;

/// Electron mass in unified atomic mass units, inverted: 1 u in mₑ
/// (2018 CODATA: mₑ = 5.48579909065e-4 u). Used to mass-weight Hessians.
pub const AMU_TO_AU_MASS: f64 = 1.0 / 5.48579909065e-4;

/// Atomic masses (u) for Z = 1..=118; index 0 is unused and deliberately 0.0.
///
/// # Two different quantities live in this table, by necessity
///
/// Atomic weight is not uniformly defined across the periodic table, so the
/// rule applied to each element is recorded rather than blurred:
///
/// - **Standard atomic weight** for every element that has one. Where IUPAC
///   publishes an interval rather than a single number (H, Li, B, C, N, O,
///   Mg, Si, S, Cl, Ar, Br, Tl, Pb) this is IUPAC's own published
///   CONVENTIONAL value, not an invented interval midpoint — which is why
///   some entries carry only 4–5 significant figures while their neighbours
///   carry 9. That asymmetry is real and padding it would be fabrication.
/// - **Mass of the longest-lived isotope** for the 34 elements that have no
///   standard atomic weight, i.e. no stable isotope and no characteristic
///   terrestrial isotopic composition: Tc, Pm, and every Z ≥ 84 except Th,
///   Pa and U (those three do have characteristic compositions). Bi keeps a
///   standard weight despite Bi-209 being radioactive — its half-life is
///   ~2e19 yr.
///
/// Sources:
/// T. Prohaska, J. Irrgeher, J. Benefield, et al., "Standard atomic weights
/// of the elements 2021 (IUPAC Technical Report)", Pure Appl. Chem. 94(5),
/// 573–600 (2022), doi:10.1515/pac-2019-0603, cross-checked against CIAAW
/// (ciaaw.org). Longest-lived-isotope masses from AME2020/NUBASE2020, with
/// the recommended isotope taken from CIAAW's radioactive-elements table.
///
/// Two entries are genuinely unsettled in the literature and are called out
/// so nobody has to rediscover it: **Tc** uses Tc-97 (96.90636 u), CIAAW's
/// recommended isotope, though Tc-98 (97.90721 u) has a statistically
/// indistinguishable half-life and is what older references quote; and for
/// **Z ≥ 104** several masses are partly derived from mass-surface
/// systematics rather than measured. None of that matters for vibrational
/// analysis of any real molecule, but it should not be silently precise.
pub const ATOMIC_MASS_AMU: [f64; 119] = [
    0.0, // (index 0 unused)
    1.008,
    4.002602,
    6.94,
    9.0121831,
    10.81,
    12.011, // H He Li Be B C
    14.007,
    15.999,
    18.998403162,
    20.1797,
    22.98976928,
    24.305, // N O F Ne Na Mg
    26.9815384,
    28.085,
    30.973761998,
    32.06,
    35.45,
    39.95, // Al Si P S Cl Ar
    39.0983,
    40.078,
    44.955907,
    47.867,
    50.9415,
    51.9961, // K Ca Sc Ti V Cr
    54.938043,
    55.845,
    58.933194,
    58.6934,
    63.546,
    65.38, // Mn Fe Co Ni Cu Zn
    69.723,
    72.630,
    74.921595,
    78.971,
    79.904,
    83.798, // Ga Ge As Se Br Kr
    85.4678,
    87.62,
    88.905838,
    91.222,
    92.90637,
    95.95, // Rb Sr Y Zr Nb Mo
    96.90636,
    101.07,
    102.90549,
    106.42,
    107.8682,
    112.414, // Tc Ru Rh Pd Ag Cd
    114.818,
    118.710,
    121.760,
    127.60,
    126.90447,
    131.293, // In Sn Sb Te I Xe
    132.90545196,
    137.327,
    138.90547,
    140.116,
    140.90766,
    144.242, // Cs Ba La Ce Pr Nd
    144.91276,
    150.36,
    151.964,
    157.249,
    158.925354,
    162.500, // Pm Sm Eu Gd Tb Dy
    164.930329,
    167.259,
    168.934219,
    173.045,
    174.96669,
    178.486, // Ho Er Tm Yb Lu Hf
    180.94788,
    183.84,
    186.207,
    190.23,
    192.217,
    195.084, // Ta W Re Os Ir Pt
    196.966570,
    200.592,
    204.38,
    207.2,
    208.98040,
    208.98243, // Au Hg Tl Pb Bi Po
    209.98715,
    222.01758,
    223.01973,
    226.02541,
    227.02775,
    232.0377, // At Rn Fr Ra Ac Th
    231.03588,
    238.02891,
    237.04817,
    244.06420,
    243.06138,
    247.07035, // Pa U Np Pu Am Cm
    247.07031,
    251.07959,
    252.08298,
    257.09511,
    258.09843,
    259.10100, // Bk Cf Es Fm Md No
    266.11987,
    267.12179,
    268.12567,
    269.12850,
    270.13337,
    271.13708, // Lr Rf Db Sg Bh Hs
    278.15649,
    281.16455,
    282.16934,
    285.17723,
    286.18246,
    289.19052, // Mt Ds Rg Cn Nh Fl
    290.19624,
    293.20458,
    294.21084,
    294.21398, // Mc Lv Ts Og
];

/// Atomic mass for element `z` in electron masses (a.u.), or `0.0` if `z`
/// names no element.
///
/// # There is deliberately no fallback
///
/// This function used to cover Z = 1..=36 and return `2.5 * z` amu for
/// anything heavier — a fabricated number, returned silently. Because a
/// harmonic frequency scales as m^(−1/2), that mis-scaled every frequency,
/// ZPE and IR intensity for 82 of the 118 elements: Ru by +8.8 % in mass
/// (−4.2 % in frequency), Pd by +8.1 % (−3.8 %), I by +4.4 % (−2.1 %). Those
/// are precisely the heavy elements the bundled LANL and Stuttgart ECP sets
/// exist to treat, and the error is large enough to move a metal–ligand
/// stretch by tens of cm⁻¹ while looking entirely plausible.
///
/// It also happened to be nearly exact at Pt (−0.04 %) and Au (+0.27 %),
/// because A/Z ≈ 2.5 there — so spot-checking on gold would have confirmed
/// the bug rather than exposed it.
///
/// A caller that needs a mass must therefore check for zero and refuse:
/// [`crate::properties::vibrational::vibrational_analysis`] does, naming the
/// offending centre. Returning a plausible number for an unknown element is
/// exactly what this project forbids.
pub fn atomic_mass_au(z: u8) -> f64 {
    ATOMIC_MASS_AMU.get(z as usize).copied().unwrap_or(0.0) * AMU_TO_AU_MASS
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the 2018-CODATA conversion factors (transcription checkpoint).
    #[test]
    fn codata_2018_constants() {
        assert_eq!(BOHR_TO_ANGSTROM, 0.529177210903);
        assert_eq!(HARTREE_TO_EV, 27.211386245988);
        assert!((HARTREE_TO_KCAL_MOL - 627.509474063).abs() < 1e-9);
        // Cross-consistency: Hartree→kJ/mol vs Hartree→kcal/mol (1 kcal = 4.184 kJ).
        assert!((HARTREE_TO_KJ_MOL - HARTREE_TO_KCAL_MOL * 4.184).abs() < 1e-6);
        // Round-trip conversions.
        assert!((ANGSTROM_TO_BOHR * BOHR_TO_ANGSTROM - 1.0).abs() < 1e-15);
        assert!((HARTREE_TO_EV * EV_TO_HARTREE - 1.0).abs() < 1e-15);
    }

    /// Every element has a real mass and none is invented.
    ///
    /// The table previously stopped at Kr and returned `2.5 * Z` beyond it,
    /// which is why the alignment check below is worth more than a spot
    /// check: `2.5 * Z` passes any "is it roughly the right size?" test — it
    /// is within 0.3 % at Pt and Au — while being 8 % wrong at Ru and Pd.
    #[test]
    fn atomic_masses_cover_the_whole_table_with_no_fabricated_fallback() {
        assert_eq!(ATOMIC_MASS_AMU.len(), 119, "table must cover Z = 1..=118");
        assert_eq!(ATOMIC_MASS_AMU[0], 0.0, "index 0 is unused");

        for z in 1..=118usize {
            let m = ATOMIC_MASS_AMU[z];
            assert!(m.is_finite() && m > 0.0, "Z={z} has no mass");
            // A nucleus weighs at least its proton count and never more than
            // ~2.6x it anywhere on the chart of nuclides.
            assert!(m >= z as f64, "Z={z}: mass {m} is below the proton count");
            assert!(m <= 2.7 * z as f64, "Z={z}: mass {m} exceeds 2.7*Z");
        }

        // Row ALIGNMENT, which magnitude bounds cannot see: a table shifted
        // by one row still has plausible masses everywhere. Mass increases
        // with Z except at a handful of places, so demanding EXACTLY those
        // and no others pins every row to its element.
        //
        // Four are the classic standard-weight inversions, caused by isotopic
        // abundance rather than by nuclear mass:
        //   Ar>K, Co>Ni, Te>I, Th>Pa
        // Three more exist only because this table mixes two rules (see
        // ATOMIC_MASS_AMU): a standard atomic weight reflects the natural
        // abundance mixture, while a longest-lived-isotope mass does not.
        //   U>Np    238.029 (U-238 dominates natural U) > 237.048 (Np-237)
        //   Pu>Am   244.064 (Pu-244)                    > 243.061 (Am-243)
        //   Cm>Bk   247.07035 (Cm-247)                  > 247.07031 (Bk-247)
        // The Cm/Bk pair differs by 4e-5 u — the same mass number, so this
        // one is genuinely a near-tie and not a robust ordering fact.
        const KNOWN_INVERSIONS: [usize; 7] = [18, 27, 52, 90, 92, 94, 96];
        for z in 1..118usize {
            let decreasing = ATOMIC_MASS_AMU[z + 1] < ATOMIC_MASS_AMU[z];
            let expected = KNOWN_INVERSIONS.contains(&z);
            assert_eq!(
                decreasing,
                expected,
                "mass ordering at Z={z}->{}: got decreasing={decreasing}, expected {expected}. \
                 This table inverts at exactly seven places (Ar/K, Co/Ni, Te/I, Th/Pa, U/Np, \
                 Pu/Am, Cm/Bk); any other one means a row is misaligned with its element.",
                z + 1
            );
        }

        // Values that the fabricated fallback got badly wrong, pinned.
        let pins: [(usize, f64); 6] = [
            (44, 101.07),     // Ru  — fallback said 110.0 (+8.8 %)
            (46, 106.42),     // Pd  — fallback said 115.0 (+8.1 %)
            (53, 126.90447),  // I   — fallback said 132.5 (+4.4 %)
            (79, 196.966570), // Au  — fallback said 197.5, misleadingly close
            (92, 238.02891),  // U
            (43, 96.90636),   // Tc  — longest-lived isotope (Tc-97)
        ];
        for (z, want) in pins {
            assert!(
                (ATOMIC_MASS_AMU[z] - want).abs() < 5e-5,
                "Z={z}: {} vs expected {want}",
                ATOMIC_MASS_AMU[z]
            );
        }

        // No fabrication outside the table: an unknown centre gets zero, and
        // callers that need a mass must refuse it rather than receive a
        // plausible-looking number.
        assert_eq!(atomic_mass_au(0), 0.0);
        assert_eq!(atomic_mass_au(119), 0.0);
        assert_eq!(atomic_mass_au(255), 0.0);
        assert!(atomic_mass_au(46) > 0.0);
    }
}
