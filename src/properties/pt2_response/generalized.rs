// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Spin-complete pair response. A spinor is one orbital, rather than an
//! alpha/beta label. The antisymmetrized source and the factor 1/4 retain
//! all spin-flip substitutions, including transverse open-shell response.
use super::retained::RetainedComplex;
use super::{
    complex_transform_path, index, taylor_response_impl, AmplitudeNumber, FockCoefficient,
};
use crate::integrals::london::ComplexMatrix as Cm;
use crate::linalg::Matrix;
use crate::response::complex::Complex;
use crate::xc::double_hybrid::Pt2Coefficients;
use crate::{DftError, Result};

pub struct GeneralizedPt2Taylor {
    /// Unscaled, factorial-normalized energy coefficients, Hartree.
    pub energy: Vec<Complex>,
    /// [Taylor coefficient][i,a,j,b], antisymmetrized pair amplitudes.
    pub amplitudes: Vec<Vec<Complex>>,
    pub residuals: Vec<f64>,
    pub relative_residuals: Vec<f64>,
}

/// Spin-summed (a i|b j) coefficients in a full complex spinor MO frame.
/// The AO tensor remains spatial: no (2*nao)^4 tensor is allocated. One
/// existing three-channel transformation supplies the four spin traces;
/// its mixed block with the two electron coordinates exchanged supplies
/// the fourth. This uses electron-interchange symmetry, not spin symmetry.
pub fn spinor_transform_path(
    nao: usize,
    integrals: &[Vec<Complex>],
    orbitals: &[Cm],
    energies: &[f64],
    occupied: usize,
) -> Result<Vec<Vec<Complex>>> {
    let width = integrals.len();
    if width == 0
        || orbitals.len() != width
        || occupied > energies.len()
        || orbitals.iter().any(|c| {
            c.re.rows != 2 * nao
                || c.im.rows != 2 * nao
                || c.re.cols != energies.len()
                || c.im.cols != energies.len()
        })
    {
        return Err(DftError::InvalidInput("spinor PT2 needs matching spatial integrals and full two-component orbital coefficients".into()));
    }
    let components: [Vec<Cm>; 2] = std::array::from_fn(|spin| {
        orbitals
            .iter()
            .map(|c| {
                let mut part = Cm::zeros(nao, energies.len());
                for mu in 0..nao {
                    for p in 0..energies.len() {
                        part.re[(mu, p)] = c.re[(spin * nao + mu, p)];
                        part.im[(mu, p)] = c.im[(spin * nao + mu, p)];
                    }
                }
                part
            })
            .collect()
    });
    let transformed = complex_transform_path(
        nao,
        integrals,
        &components,
        &[energies.to_vec(), energies.to_vec()],
        [occupied; 2],
    )?;
    let virtuals = energies.len() - occupied;
    let shape = [occupied, virtuals, occupied, virtuals];
    Ok(transformed
        .into_iter()
        .map(|p| {
            let mut g = p.ovov[0].clone();
            for i in 0..occupied {
                for a in 0..virtuals {
                    for j in 0..occupied {
                        for b in 0..virtuals {
                            let k = index(shape, i, a, j, b);
                            g[k] +=
                                p.ovov[2][k] + p.ovov[1][k] + p.ovov[1][index(shape, j, b, i, a)];
                        }
                    }
                }
            }
            g
        })
        .collect())
}

/// Standard spin-orbital PT2 on a semicanonical complex reference. The
/// supplied full Fock series is in the same parallel occupied/virtual frame
/// as the source. Complex base integrals are retained; no real projection,
/// spin-diagonal approximation or denominator floor is made. At higher
/// orders the common noncanonical pair operator includes every spinor.
/// Spin-component scaling is deliberately outside this spin-free quantity:
/// callers can multiply it by a common SS=OS coefficient. Unequal scaling
/// needs an explicit covariant spin-pair model instead of orbital labels.
pub fn generalized_taylor_response(
    energies: &[f64],
    occupied: usize,
    integrals: &[Vec<Complex>],
    fock: &[Cm],
) -> Result<GeneralizedPt2Taylor> {
    let Some(base) = integrals.first() else {
        return Err(DftError::InvalidInput(
            "generalized PT2 path cannot be empty".into(),
        ));
    };
    if fock.len() != integrals.len()
        || fock.iter().any(|f| {
            f.re.rows != energies.len()
                || f.re.cols != energies.len()
                || f.im.rows != energies.len()
                || f.im.cols != energies.len()
                || f.re
                    .as_slice()
                    .iter()
                    .chain(f.im.as_slice())
                    .any(|v| !v.is_finite())
        })
    {
        return Err(DftError::InvalidInput(
            "generalized PT2 Fock/source coefficient dimensions disagree".into(),
        ));
    }
    let mut base_defect = 0.0_f64;
    for i in 0..energies.len() {
        for j in 0..energies.len() {
            base_defect = base_defect
                .max((fock[0].re[(i, j)] - if i == j { energies[i] } else { 0.0 }).abs())
                .max(fock[0].im[(i, j)].abs());
        }
    }
    if !base_defect.is_finite() || base_defect > 1e-10 {
        return Err(DftError::InvalidInput(
            "generalized PT2 base Fock must be semicanonical in the supplied frame".into(),
        ));
    }
    // Large third- and fourth-order coefficients can require much smaller
    // absolute residuals than their binary64 spacing. Keep the correction
    // in the solved amplitudes and in the fresh equation contraction.
    if integrals.len() >= 4 {
        spin_orbital_taylor::<RetainedComplex>(energies, occupied, base, integrals, fock)
    } else {
        spin_orbital_taylor::<Complex>(energies, occupied, base, integrals, fock)
    }
}

fn spin_orbital_taylor<S>(
    energies: &[f64],
    occupied: usize,
    base: &[Complex],
    integrals: &[Vec<Complex>],
    fock: &[Cm],
) -> Result<GeneralizedPt2Taylor>
where
    S: AmplitudeNumber + From<Complex> + Into<Complex>,
    Cm: FockCoefficient<S>,
{
    let convert = |values: &[Complex]| values.iter().copied().map(S::from).collect();
    let source = [convert(base), Vec::new(), Vec::new()];
    let higher = integrals
        .iter()
        .skip(1)
        .map(|g| [convert(g), Vec::new(), Vec::new()])
        .collect::<Vec<_>>();
    let fields = fock
        .iter()
        .skip(1)
        .map(|f| {
            [
                f.clone(),
                Cm {
                    re: Matrix::zeros(0, 0),
                    im: Matrix::zeros(0, 0),
                },
            ]
        })
        .collect::<Vec<_>>();
    let pairs = taylor_response_impl(
        &[energies.to_vec(), Vec::new()],
        [occupied, 0],
        &source,
        &higher,
        &fields,
        Pt2Coefficients {
            same_spin: 1.0,
            opposite_spin: 1.0,
        },
    )?;
    Ok(GeneralizedPt2Taylor {
        energy: pairs.correlation.into_iter().map(Into::into).collect(),
        amplitudes: pairs
            .amplitude_coefficients
            .into_iter()
            .map(|[g, _, _]| g.into_iter().map(Into::into).collect())
            .collect(),
        residuals: pairs.residuals,
        relative_residuals: pairs.relative_residuals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonfinite_base_fock_is_rejected_before_the_semicanonical_contraction() {
        let energies = [-1.0, 0.5];
        let mut f = Cm::zeros(2, 2);
        f.re[(0, 0)] = energies[0];
        f.re[(1, 1)] = energies[1];
        let integrals = [vec![Complex::from(0.1)]];
        assert!(generalized_taylor_response(&energies, 1, &integrals, &[f.clone()]).is_ok());
        for value in [f64::NAN, f64::INFINITY] {
            let mut invalid = f.clone();
            invalid.im[(0, 1)] = value;
            assert!(generalized_taylor_response(&energies, 1, &integrals, &[invalid]).is_err());
        }
    }

    #[test]
    fn retained_fourth_order_pair_satisfies_the_absolute_equation() {
        let energies = [-0.5, -0.25, 0.125, 0.375];
        let mut fock = vec![Cm::zeros(4, 4); 5];
        for (i, &energy) in energies.iter().enumerate() {
            fock[0].re[(i, i)] = energy;
        }
        let slope = 1001.3;
        fock[1].re[(0, 0)] = slope;
        let shape = [2, 2, 2, 2];
        let coupling = Complex::new(0.18, 0.13);
        let mut source = vec![vec![Complex::ZERO; 16]; 5];
        source[0][index(shape, 0, 0, 1, 1)] = coupling;
        source[0][index(shape, 1, 1, 0, 0)] = coupling;
        let pair = generalized_taylor_response(&energies, 2, &source, &fock).unwrap();

        // The nonzero double substitution has scalar denominator
        // -1.25 + slope*t. Expand the rational energy independently.
        let denominator = -1.25;
        for (n, energy) in pair.energy.iter().enumerate() {
            let expected =
                coupling.abs().powi(2) / denominator * (-slope / denominator).powi(n as i32);
            assert!((energy.re - expected).abs() < 2e-15 * expected.abs());
            assert!(energy.im.abs() < 1e-15);
        }
        assert!(pair.residuals.iter().all(|residual| *residual < 1e-12));
        println!(
            "retained fourth-order pair energy={:.15e}, absolute residuals={:?}",
            pair.energy[4].re, pair.residuals
        );
    }
}
