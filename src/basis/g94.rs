// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Gaussian94-format basis-set parser (the format served by the Basis Set
//! Exchange).
//!
//! Layout: `!`-prefixed comment lines, element blocks separated by `****`
//! lines. Each block starts with `<Symbol> 0`, followed by shell headers
//! `<TYPE> <nprim> <scale>` (`TYPE` ∈ S, P, D, F, G, H, I, or the combined
//! `SP`), each followed by `nprim` rows of exponent + coefficient(s).
//! Fortran `D` exponents (`1.2D+01`) are accepted. `SP` shells are split
//! into separate S and P shells sharing exponents. The scale factor
//! multiplies exponents by `scale²` (virtually always 1.00).

use super::{BasisSet, ElementBasis, Shell};
use crate::error::{DftError, Result};
use crate::system::symbol_to_z;

/// Parse a Gaussian94-format basis text into a named `BasisSet`.
pub fn parse(name: &str, text: &str) -> Result<BasisSet> {
    let mut set = BasisSet::new(name);
    // Strip comments and blank lines, keep 1-based line numbers for errors.
    let lines: Vec<(usize, &str)> = text
        .lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l.trim()))
        .filter(|(_, l)| !l.is_empty() && !l.starts_with('!'))
        .collect();
    let mut pos = 0usize;
    // Skip leading separator(s).
    while pos < lines.len() && lines[pos].1.starts_with("****") {
        pos += 1;
    }
    while pos < lines.len() {
        let (line_no, header) = lines[pos];
        let mut fields = header.split_whitespace();
        let symbol = fields.next().unwrap_or_default();
        let z = symbol_to_z(symbol).ok_or_else(|| DftError::Parse {
            line: line_no,
            message: format!("expected an element header, found `{header}`"),
        })?;
        pos += 1;
        // ECP section: the element header is immediately followed by an
        // `<SYM>-ECP <lmax> <ncore>` line. Parse the local and semilocal
        // channels into the ECP representation used by the integral engine.
        if pos < lines.len() && is_ecp_header(lines[pos].1) {
            pos = parse_ecp_block(&lines, pos, z, &mut set)?;
            continue;
        }
        let mut shells = Vec::new();
        while pos < lines.len() && !lines[pos].1.starts_with("****") {
            let (shell_line, shell_header) = lines[pos];
            let mut sf = shell_header.split_whitespace();
            let type_tok = sf
                .next()
                .ok_or_else(|| DftError::Parse {
                    line: shell_line,
                    message: "empty shell header".to_string(),
                })?
                .to_ascii_uppercase();
            let nprim: usize =
                sf.next()
                    .and_then(|t| t.parse().ok())
                    .ok_or_else(|| DftError::Parse {
                        line: shell_line,
                        message: format!("invalid primitive count in `{shell_header}`"),
                    })?;
            let scale: f64 = sf
                .next()
                .map(|t| parse_fortran_f64(t, shell_line))
                .transpose()?
                .unwrap_or(1.0);
            if nprim == 0 {
                return Err(DftError::Parse {
                    line: shell_line,
                    message: "a contracted shell must contain at least one primitive".to_string(),
                });
            }
            if !(scale.is_finite() && scale > 0.0) {
                return Err(DftError::Parse {
                    line: shell_line,
                    message: format!("shell scale must be finite and positive, found {scale}"),
                });
            }
            let scale2 = scale * scale;
            pos += 1;
            let ncols = if type_tok == "SP" { 2 } else { 1 };
            let mut exps = Vec::with_capacity(nprim);
            let mut cols: Vec<Vec<f64>> = vec![Vec::with_capacity(nprim); ncols];
            for _ in 0..nprim {
                let (row_line, row) = *lines.get(pos).ok_or_else(|| DftError::Parse {
                    line: shell_line,
                    message: "basis block ended inside a shell".to_string(),
                })?;
                let toks: Vec<&str> = row.split_whitespace().collect();
                if toks.len() < 1 + ncols {
                    return Err(DftError::Parse {
                        line: row_line,
                        message: format!(
                            "expected exponent + {ncols} coefficient(s), found `{row}`"
                        ),
                    });
                }
                let exponent = parse_fortran_f64(toks[0], row_line)? * scale2;
                if !(exponent.is_finite() && exponent > 0.0) {
                    return Err(DftError::Parse {
                        line: row_line,
                        message: format!(
                            "Gaussian exponent must be finite and positive, found {exponent}"
                        ),
                    });
                }
                exps.push(exponent);
                for (c, col) in cols.iter_mut().enumerate() {
                    let coefficient = parse_fortran_f64(toks[1 + c], row_line)?;
                    if !coefficient.is_finite() {
                        return Err(DftError::Parse {
                            line: row_line,
                            message: "contraction coefficient must be finite".to_string(),
                        });
                    }
                    col.push(coefficient);
                }
                pos += 1;
            }
            match type_tok.as_str() {
                "SP" => {
                    shells.push(Shell {
                        l: 0,
                        exps: exps.clone(),
                        coefs: cols[0].clone(),
                    });
                    shells.push(Shell {
                        l: 1,
                        exps,
                        coefs: cols[1].clone(),
                    });
                }
                simple => {
                    let l = match simple {
                        "S" => 0,
                        "P" => 1,
                        "D" => 2,
                        "F" => 3,
                        "G" => 4,
                        "H" => 5,
                        "I" => 6,
                        other => {
                            return Err(DftError::Parse {
                                line: shell_line,
                                message: format!("unsupported shell type `{other}`"),
                            })
                        }
                    };
                    shells.push(Shell {
                        l,
                        exps,
                        coefs: cols[0].clone(),
                    });
                }
            }
        }
        if shells.is_empty() {
            return Err(DftError::Parse {
                line: line_no,
                message: format!("element {symbol} has no shells"),
            });
        }
        set.insert(ElementBasis { z, shells });
        // Skip the trailing separator.
        while pos < lines.len() && lines[pos].1.starts_with("****") {
            pos += 1;
        }
    }
    Ok(set)
}

fn is_ecp_header(line: &str) -> bool {
    line.split_whitespace()
        .next()
        .map(|tok| tok.to_ascii_uppercase().ends_with("-ECP"))
        .unwrap_or(false)
}

/// Parse one element's ECP definition starting at the
/// `<SYM>-ECP <lmax> <ncore>` line; returns the position after the block.
///
/// Block layout: `lmax+1` channels, the **first** being the local channel
/// `U_L` (title like "f potential") followed by the projected channels
/// `U_l − U_L` for l = 0..L−1 ("s-f potential", …). Each channel is a title
/// line, a term-count line, and `<power> <exponent> <coefficient>` rows
/// describing `c · r^{power−2} · exp(−α r²)`.
fn parse_ecp_block(
    lines: &[(usize, &str)],
    mut pos: usize,
    z: u8,
    set: &mut BasisSet,
) -> Result<usize> {
    let (line_no, header) = lines[pos];
    let toks: Vec<&str> = header.split_whitespace().collect();
    if toks.len() < 3 {
        return Err(DftError::Parse {
            line: line_no,
            message: format!("malformed ECP header `{header}`"),
        });
    }
    let lmax: usize = toks[1].parse().map_err(|_| DftError::Parse {
        line: line_no,
        message: format!("invalid ECP lmax in `{header}`"),
    })?;
    let n_core: u32 = toks[2].parse().map_err(|_| DftError::Parse {
        line: line_no,
        message: format!("invalid ECP core count in `{header}`"),
    })?;
    let upper_header = toks[0].to_ascii_uppercase();
    let header_symbol = upper_header.strip_suffix("-ECP").unwrap_or_default();
    if crate::system::symbol_to_z(header_symbol) != Some(z) {
        return Err(DftError::Parse {
            line: line_no,
            message: format!("ECP header `{header}` does not match its element block"),
        });
    }
    if n_core > z as u32 {
        return Err(DftError::Parse {
            line: line_no,
            message: format!("ECP replaces {n_core} core electrons, more than element Z={z} has"),
        });
    }
    pos += 1;
    let mut channels: Vec<Vec<super::EcpTerm>> = Vec::with_capacity(lmax + 1);
    for _ in 0..=lmax {
        // Title line (e.g. "f potential", "s-f potential").
        pos += 1;
        let (count_line, count_text) = *lines.get(pos).ok_or_else(|| DftError::Parse {
            line: line_no,
            message: "ECP block ended before its potential terms".to_string(),
        })?;
        let n_terms: usize = count_text.trim().parse().map_err(|_| DftError::Parse {
            line: count_line,
            message: format!("invalid ECP term count `{count_text}`"),
        })?;
        pos += 1;
        let mut terms = Vec::with_capacity(n_terms);
        for _ in 0..n_terms {
            let (row_line, row) = *lines.get(pos).ok_or_else(|| DftError::Parse {
                line: count_line,
                message: "ECP block ended inside its potential terms".to_string(),
            })?;
            let rt: Vec<&str> = row.split_whitespace().collect();
            if rt.len() < 3 {
                return Err(DftError::Parse {
                    line: row_line,
                    message: format!("expected `power exponent coefficient`, found `{row}`"),
                });
            }
            let power: i32 = rt[0].parse().map_err(|_| DftError::Parse {
                line: row_line,
                message: format!("invalid ECP r-power `{}`", rt[0]),
            })?;
            let exp = parse_fortran_f64(rt[1], row_line)?;
            let coef = parse_fortran_f64(rt[2], row_line)?;
            if power < 0 {
                return Err(DftError::Parse {
                    line: row_line,
                    message: format!("ECP radial power must be non-negative, found {power}"),
                });
            }
            if !(exp.is_finite() && exp > 0.0) {
                return Err(DftError::Parse {
                    line: row_line,
                    message: format!("ECP exponent must be finite and positive, found {exp}"),
                });
            }
            if !coef.is_finite() {
                return Err(DftError::Parse {
                    line: row_line,
                    message: "ECP coefficient must be finite".to_string(),
                });
            }
            terms.push(super::EcpTerm { power, exp, coef });
            pos += 1;
        }
        channels.push(terms);
    }
    let local = channels.remove(0);
    set.set_ecp(
        z,
        super::EcpPotential {
            n_core,
            lmax,
            local,
            semilocal: channels,
        },
    );
    Ok(pos)
}

/// Parse an f64 accepting Fortran `D`/`d` exponent markers.
fn parse_fortran_f64(token: &str, line: usize) -> Result<f64> {
    let normalized: String = token
        .chars()
        .map(|c| match c {
            'D' | 'd' => 'E',
            other => other,
        })
        .collect();
    normalized.parse::<f64>().map_err(|_| DftError::Parse {
        line,
        message: format!("invalid number `{token}`"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const STO3G_SNIPPET: &str = r#"
! Comment line
****
H     0
S    3   1.00
      0.3425250914D+01       0.1543289673D+00
      0.6239137298D+00       0.5353281423D+00
      0.1688554040D+00       0.4446345422D+00
****
O     0
S    3   1.00
      0.1307093214D+03       0.1543289673D+00
      0.2380886605D+02       0.5353281423D+00
      0.6443608313D+01       0.4446345422D+00
SP   3   1.00
      0.5033151319D+01      -0.9996722919D-01       0.1559162750D+00
      0.1169596125D+01       0.3995128261D+00       0.6076837186D+00
      0.3803889600D+00       0.7001154689D+00       0.3919573931D+00
****
"#;

    #[test]
    fn parses_sto3g_h_and_o() {
        let set = parse("sto-3g", STO3G_SNIPPET).unwrap();
        let h = set.element(1).unwrap();
        assert_eq!(h.shells.len(), 1);
        assert_eq!(h.shells[0].l, 0);
        assert_eq!(h.shells[0].exps.len(), 3);
        assert!((h.shells[0].exps[0] - 3.425250914).abs() < 1e-9);
        assert!((h.shells[0].coefs[2] - 0.4446345422).abs() < 1e-10);

        // SP splits into S and P sharing exponents.
        let o = set.element(8).unwrap();
        assert_eq!(o.shells.len(), 3);
        assert_eq!(o.shells[1].l, 0);
        assert_eq!(o.shells[2].l, 1);
        assert_eq!(o.shells[1].exps, o.shells[2].exps);
        assert!((o.shells[1].coefs[0] - -0.09996722919).abs() < 1e-10);
        assert!((o.shells[2].coefs[0] - 0.1559162750).abs() < 1e-10);

        // Missing element is a clean error.
        assert!(set.element(6).is_err());
    }

    #[test]
    fn parses_ecp_section() {
        // Layout as served by BSE: orbital blocks with ****, then the ECP
        // section without separators between elements.
        let text = r#"
****
Rb     0
S    2   1.00
      5.0   0.3
      1.0   0.8
****

RB     0
RB-ECP     2     28
d potential
  1
2      3.8431140            -12.3169000
s-d potential
  2
2      5.0365510             89.5001980
2      1.9708490              0.4937610
p-d potential
  1
2      4.2583410             58.5689740
"#;
        let set = parse("ecp-test", text).unwrap();
        assert_eq!(set.element(37).unwrap().shells.len(), 1);
        let ecp = set.ecp(37).expect("Rb ECP must be recorded");
        assert_eq!(ecp.n_core, 28);
        assert_eq!(ecp.lmax, 2);
        assert_eq!(ecp.local.len(), 1);
        assert!((ecp.local[0].coef - -12.3169).abs() < 1e-9);
        assert_eq!(ecp.semilocal.len(), 2);
        assert_eq!(ecp.semilocal[0].len(), 2); // s channel
        assert_eq!(ecp.semilocal[1].len(), 1); // p channel
        assert!((ecp.semilocal[0][0].exp - 5.036551).abs() < 1e-9);
        assert_eq!(ecp.semilocal[0][0].power, 2);
        assert_eq!(set.ecp_core(37), Some(28));
        assert_eq!(set.ecp_core(1), None);
    }

    #[test]
    fn rejects_malformed_blocks() {
        assert!(parse("bad", "****\nXx 0\nS 1 1.00\n1.0 1.0\n****\n").is_err());
        assert!(parse("bad", "****\nH 0\nS 2 1.00\n1.0 1.0\n****\n").is_err());
        assert!(parse("bad", "****\nH 0\nQ 1 1.00\n1.0 1.0\n****\n").is_err());
        assert!(parse("bad", "****\nH 0\nS 1 1.00\n0.0 1.0\n****\n").is_err());
        assert!(parse("bad", "****\nH 0\nS 1 0.00\n1.0 1.0\n****\n").is_err());
    }

    /// An ECP is part of the Hamiltonian, so silently accepting a header
    /// for the wrong element or removing more electrons than Z would make
    /// every downstream energy finite but physically meaningless.
    #[test]
    fn rejects_unphysical_ecp_metadata() {
        let wrong_element =
            "****\nNa 0\nS 1 1.0\n1.0 1.0\n****\nNa 0\nMg-ECP 0 10\ns potential\n1\n2 1.0 1.0\n";
        assert!(parse("bad-ecp", wrong_element).is_err());

        let too_many_core =
            "****\nH 0\nS 1 1.0\n1.0 1.0\n****\nH 0\nH-ECP 0 2\ns potential\n1\n2 1.0 1.0\n";
        assert!(parse("bad-ecp", too_many_core).is_err());

        let bad_exponent =
            "****\nNa 0\nS 1 1.0\n1.0 1.0\n****\nNa 0\nNa-ECP 0 10\ns potential\n1\n2 -1.0 1.0\n";
        assert!(parse("bad-ecp", bad_exponent).is_err());
    }
}
