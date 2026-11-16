<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

# Implemented functionals, conventions, and capability matrix

Implemented providers use primary-literature definitions and numerical
reference fixtures. The roadmap below also lists unimplemented providers
and incomplete acceptance gates; implementation does not imply verification
of every derivative order or reference/provider combination.

Status labels use the [shared definitions](conventions.md#status-vocabulary).
In particular, `not_implemented` means the complete provider/route is absent;
an implemented component or a passed energy test does not establish its
unimplemented derivatives or reference combinations.

## Exchange / correlation components (`--x` / `--c`)

| Name | Type | Source |
|---|---|---|
| slater | LDA-X | Dirac 1930 / Slater 1951 |
| b88 | GGA-X | Becke, PRA 38, 3098 (1988) |
| pbe (X) | GGA-X | Perdew–Burke–Ernzerhof, PRL 77, 3865 (1996) |
| optx | GGA-X | Handy–Cohen, Mol. Phys. 99, 403 (2001) |
| revpbe (X) | GGA-X | Zhang–Yang, PRL 80, 890 (1998); κ = 1.245 |
| pbesol (X) | GGA-X | Perdew et al., PRL 100, 136406 (2008); μ = 10/81 |
| rpbe (X) | GGA-X | Hammer–Hansen–Nørskov, PRB 59, 7413 (1999); F = 1 + κ(1 − e^{−μs²/κ}) |
| pw91 (X) | GGA-X | Perdew–Wang 1991 (Perdew et al., PRB 46, 6671 (1992)) |
| scan (X) | mGGA-X | Sun–Ruzsinszky–Perdew, PRL 115, 036402 (2015) |
| tpss (X) | mGGA-X | Tao–Perdew–Staroverov–Scuseria, PRL 91, 146401 (2003) |
| r2scan (X) | mGGA-X | Furness et al., JPCL 11, 8208 (2020) |
| hjs-pbe | SR GGA-X | Henderson–Janesko–Scuseria ωPBE hole, JCP 128, 194105 (2008) |
| wpbeh | SR GGA-X | HSE screened hole (Heyd 2003; Henderson 2009 s-rescaling); ω=0 → PBE-hole |
| vwn5 / vwn3 | LDA-C | Vosko–Wilk–Nusair, Can. J. Phys. 58, 1200 (1980) |
| pw92 | LDA-C | Perdew–Wang, PRB 45, 13244 (1992) |
| lyp | GGA-C | Lee–Yang–Parr, PRB 37, 785 (1988) |
| pbe (C) | GGA-C | PBE 1996 (PW92 "MOD" digits) |
| tpss (C) | mGGA-C | revPKZB + d=2.8 SIC |
| r2scan (C) | mGGA-C | Furness 2020 (PW92 MOD base, regTPSS β(rs)) |
| pbesol (C) | GGA-C | PBE correlation with β = 0.046 |
| scan (C) | mGGA-C | SCAN 2015 (PW92 MOD base, β(rs), exponential f_c(α)) |
| p86 | GGA-C | Perdew, PRB 33, 8822 (1986) on the PZ81 local fit |
| pw91 / pw91c | GGA-C | Perdew–Wang 1991 (RG'86 Padé — see conventions) |

## Combined / hybrid / RSH / meta (`--xc`)

| Name | Class | HF exchange | Notes |
|---|---|---|---|
| svwn5, svwn3, spw92 | LDA | — | |
| pbe, blyp | GGA | — | |
| b3lyp / b3lyp5 | hybrid | 0.20 | VWN3 (Gaussian) / VWN5 convention |
| o3lyp | hybrid | 0.1161 | ΔOPTX gradient part |
| pbe0 | hybrid | 0.25 | |
| cam-b3lyp | RSH | α=0.19, β=0.46, ω=0.33 | ITYH SR-B88 |
| lc-blyp / lc-pbe | RSH | clr=1 | ITYH attenuation (not HJS/ωPBE) |
| b97, hcth/93/120/147/407 | XC-integrated GGA | b97: 0.1943 | power series; PW92 **paper** digits |
| wb97 / wb97x / wb97x-d | RSH GGA | 0 / 0.157706 / 0.222036 (SR) | SR-LSDA×B97 series; ωB97X-D tail = `--disp chg` |
| wb97x-v | RSH GGA | 0.167 (SR), ω=0.3 | + VV10 (b=6.0, C=0.01) |
| wb97m-v | RSH meta | 0.15 (SR), ω=0.3 | sparse 2D (w,u) series + VV10 |
| m06l / m06 / m06-2x | meta / hybrid meta | 0 / 0.27 / 0.54 | PW92 MOD digits, α_x=0.00186726; M06-L a11=−9.049762 (ref. impl.) |
| revm06l / revm06 | meta / hybrid meta | 0 / 0.4041 | Wang 2017/2018; same LSQ-verified machinery |
| m06-sx | screened hybrid meta | 0.335 (SR), ω=0.10 | Wang, PNAS 117, 2294 (2020); K−K_erf trick (cx=+0.335, clr=−0.335) |
| tpss / tpssh | meta / hybrid meta | 0 / 0.10 | nonempirical |
| r2scan | meta | — | re-regularized SCAN; FHC input bound (see conventions) |
| scan | meta | — | original SCAN; the α-interpolation is zero within 1e-3 of α = 1 (exact limit, value below 1e-280); FHC input bound |
| pbesol, rpbe, pw91 | GGA | — | PBEsol X+C; RPBE X + PBE C; PW91 X + PW91 C |
| lrc-wpbeh | RSH | SR 0.2, LR 1.0, ω=0.2 | Rohrdanz–Martins–Herbert 2009; 0.8 SR-ωPBE (HJS hole) + PBE-C |
| b97m-v | meta | — | Mardirossian–Head-Gordon 2015 sparse (w,u) series + VV10 |
| bp86 | GGA | — | B88 + P86 (Gaussian-legacy) |
| b3pw91 | hybrid | 0.20 | Becke 1993 three-parameter with PW91-C gradient part |
| hse06 / hse03 | screened hybrid | 0.25 (SR) | wPBEh(0) − 0.25·wPBEh(ω) + PBE-C; ω=0.11 / (0.1061, 0.1890) |
| hsesol | screened hybrid | 0.25 (SR), ω = 0.11 Bohr⁻¹ | Full PBEsol X+C minus 0.25 HJS short-range PBEsol exchange. [Schimka, Harl and Kresse (2011)](https://doi.org/10.1063/1.3524336) |
| lc-wpbe | RSH | clr=1, ω=0.4 | canonical Vydrov–Scuseria (wPBEh SR hole) |
| lc-wpbe-hjs | RSH | clr=1, ω=0.4 | HJS 2008 hole (libxc LC_WPBE_WHS) |
| hf | — | 1.0 | RHF/UHF |

`--omega <w>` re-tunes ω on any RSH (functional + K_lr kernel together).
XC-integrated names given component-wise: identical `--x`/`--c` continues,
mismatched is a hard pre-SCF error. `--list-functionals` prints the whole
registry with descriptions; names
match case-insensitively with `-`, `_` and spaces ignored.

### Screened and xDH energy checks

Select `xc = "hsesol"`, `"wb97x-2-lp"`, `"wb97x-2-tqz"` or `"xdh-pbe0"`
with the common driver. The same spelling is used in Rust `Controls`,
`dft-rs --xc`, native Python `xc=` and `DftRs(xc=...)`. The
functional catalogue reports the exact-exchange split, screening in
Bohr⁻¹, and the dimensionless same-spin and opposite-spin PT2 weights.

| Functional | Parent orbitals | Short-/long-range exact exchange | ω (Bohr⁻¹) | PT2 SS/OS | Defining reference |
|---|---|---|---:|---|---|
| `hsesol` | HSEsol | 0.25 / 0 | 0.11 | — | [Schimka, Harl and Kresse (2011), Eqs. (2) and (5)](https://doi.org/10.1063/1.3524336), [Henderson, Janesko and Scuseria (2008), Table II](https://doi.org/10.1063/1.2921797), [Perdew et al. (2008)](https://doi.org/10.1103/PhysRevLett.100.136406) |
| `wb97x-2-lp` | LP fit | 0.678792 / 1 | 0.30 | 0.581569 / 0.477992 | [Chai and Head-Gordon (2009), Table II](https://doi.org/10.1063/1.3244209) |
| `wb97x-2-tqz` | TQZ fit | 0.636158 / 1 | 0.30 | 0.529319 / 0.447105 | [Chai and Head-Gordon (2009), Table II](https://doi.org/10.1063/1.3244209) |
| `xdh-pbe0` | PBE0 | 0.8335 target, 0.25 parent | 0 | 0 / 0.5428 | [Su and Xu (2012)](https://doi.org/10.1063/1.3703893) |

The ωB97X-2 fits share the cubic screened B97 exchange, same-spin
correlation and opposite-spin correlation form. Their published local
exchange weights are independent of the exact-exchange fractions. PT2
uses the full Coulomb interaction. The TQZ registry entry uses the
supplied basis. For the paper's complete-basis estimate, extrapolate
the two PT2 components from separate triple- and quadruple-zeta calculations
using the paper's inverse-cube formula. The xDH-PBE0 target uses 0.1665
PBE exchange and 0.5292 PBE correlation on the stationary PBE0 orbitals.

`tests/screened_and_xdh_terms.rs` compares self-consistent driver outputs
with independent H₂/STO-3G totals and energy terms at 1.4 Bohr. Run
`python tools/oracle/gen_screened_named_checks.py` to reproduce the values. The
reference uses normalized two-center Gaussian integrals and 240-by-240
Gauss–Legendre quadrature in cylindrical coordinates. Inversion fixes
the bonding and antibonding orbitals. The independent OS PT2 term uses
the full Coulomb integral and the orbital eigenvalue denominator. The
SS term is structurally zero for one occupied orbital per spin channel
in this fixture. Nonzero SS acceptance requires a multielectron system.

| Functional | Computed total (Hartree) | Reference total (Hartree) | Computed OS PT2 (Hartree) | Raw orbital residual (Hartree) |
|---|---:|---:|---:|---:|
| `hsesol` | −1.14083881126366 | −1.140838811287166 | — | 1.2381278570e-16 |
| `wb97x-2-lp` | −1.14764804262653 | −1.147648039793244 | −0.013961958608569 | 1.3005189720e-16 |
| `wb97x-2-tqz` | −1.14385754339509 | −1.143857540697865 | −0.014063736762629 | 8.5865514685e-17 |
| `xdh-pbe0` | −1.14892606604642 | −1.148926066046099 | −0.018874640184623 | 1.0838950382e-16 |

All four total-energy comparisons pass the 5e-7 Hartree gate. The largest
energy-term error is 3.051e-9 Hartree, for the LP parent. All double-hybrid
PT2 amplitude residuals and both unscaled SS terms are zero on this fixture.
The HSEsol orbital residual comes from a dipole report reproducing the
energy exactly. The test initially looked for that field on the ordinary
energy report and was corrected to use the report that supplies it.
The standalone CLI agrees with all four Rust-driver total energies to
4.22e-15 Hartree. Each JSON result labels the total in Hartree and each
stderr stream contains the SCF iterations and convergence residual.

For nonzero spin terms, the same test file also uses neutral C1 ammonia
with STO-3G, grid level 0 and zero grid weight cutoff. It independently
transforms each bare AO quartet to the stationary parent orbitals and
directly sums the unordered same-spin pairs and ordered opposite-spin
pairs. The parent and AO integrals are shared with the production route.
All nine unscaled spin terms and three weighted PT2 energies agree
within 6.939e-17 Hartree against a 2e-10 Hartree gate.

| Functional | Unscaled SS α and SS β (each, Hartree) | Unscaled OS (Hartree) | Weighted PT2 (Hartree) |
|---|---:|---:|---:|
| `wb97x-2-lp` | −0.00185724510308602 | −0.0487954819869608 | −0.0254840823806246 |
| `wb97x-2-tqz` | −0.00187301856791701 | −0.0492146027901565 | −0.0239869436111955 |
| `xdh-pbe0` | −0.00249518956798801 | −0.0661123887323773 | −0.0358858046039344 |

The HJS short-range oracle additionally uses direct integration of the
published PBEsol exchange hole over the interelectronic distance. A first
integration ending at dimensionless `y = 48` missed a 1.4066e-7 tail at
`s = 0.2`, `ν = 0.045`. Extending to `y = 320` and refining from 16,384
to 131,072 panels resolved that tail. The largest error in the five
screened enhancement values was 5.452e-14. At the fixed-density HSEsol point
`s = 1.4`, `ν = 0.045` with one spin channel, the measured exchange,
screening correction and correlation densities are −0.17071222721720317,
0.038866724386425765 and −0.0006696571192611469 Hartree/Bohr³.
The frozen ωB97X-2 local oracle passes each of the three semilocal terms
in spin-polarized, single-spin and zero-gradient cases for both fits.
These checks cover the energy definitions. The
[molecular response ledger](response.md#molecular-response-across-the-registered-functional-catalogue)
tracks derivative coverage.

## Dispersion (`--disp`) — empirical and nonlocal under one flag

| Model | Kind | Source / notes |
|---|---|---|
| d2 | empirical | Grimme 2006 (paper table, H–Kr) |
| d3zero / d3bj | empirical | Grimme 2010 / BJ 2011 (data via MOPAC, Apache-2.0) |
| chg | empirical | Chai–Head-Gordon damping over the D2 C6 table (ωB97X-D tail, a=6) |
| xdm | density-dependent post-SCF | Becke–Johnson XDM; caller supplies `--xdm-a1`/`--xdm-a2`; energy only |
| vv10 | nonlocal (SC) | Vydrov–Van Voorhis 2010; V_nlc in the Fock — variational, exact analytic gradients |
| vv10-post | nonlocal (post-SCF) | fixed-density energy add-on; gradient response cross-term neglected |

-V functionals carry VV10 **by definition**: SC-VV10 (b=6.0, C=0.01) is
enabled automatically, and explicitly requesting VV10 again via
`--disp`/`--nlc` is a hard error (specification conflict); `--nlc none`
disables it for semilocal-only debugging. `--nlc` is otherwise a
compatibility alias. D4 is not implemented.

**XDM** (exchange-hole dipole moment; Becke & Johnson, JCP 122/123 (2005),
127, 154108 (2007); Johnson & Becke, JCP 124, 174104 (2006)) is implemented
in `src/dispersion/xdm.rs` as a density-dependent correction. C₆/C₈/C₁₀ come from the Becke–Roussel exchange hole of the
actual SCF density, Hirshfeld-partitioned per atom, scaled to an effective
polarizability by the atom's Hirshfeld volume relative to the free atom. The
only bundled external data is a small, openly published (and XDM-independent)
free-atom static-polarizability table
(`src/dispersion/data/xdm_free_atom_polarizability.csv`, Z = 1–36); the
free-atom reference VOLUME needed by the same volume-scaling equation is
computed on the fly from `crate::scf::sad::sad_density`'s spherically
averaged atomic density instead of a second external table — see the
module's doc comment for why that is the mathematically consistent choice
rather than a shortcut. Becke–Johnson damping is implemented as a general
mechanism; the caller supplies `a1`/`a2` rather than dft-rs asserting a
per-functional fit it cannot independently verify.

XDM is wired to `--disp xdm` for the `energy` and `charges` commands, with
explicit `--xdm-a1` and `--xdm-a2` parameters. It remains a separate library
entry point (`xdm_dispersion_energy`) rather than a geometry-only
`Dispersion` enum variant, and it has no analytic gradient (differentiating
the Becke–Roussel root solve and the Hirshfeld/Becke grid-weight response is
not implemented; derivative-requiring CLI commands refuse it). Gated
by the hydrogen-atom closed form of the Becke–Roussel hole (self-derived
from its defining equations: b_σ(r) = r exactly for any one-electron 1s-like
density), the free-atom limit (effective polarizability reduces exactly to
the tabulated free-atom value), the London R⁻⁶ asymptote, translation and
rotation invariance, and the classic noble-gas C₆ values — measured at
HF/aug-cc-pVTZ: He–He 1.73 a.u. (published 1.461, +18%), Ne–Ne 6.25
(published 6.38, −2%), Ar–Ar 56.0 (published 64.3, −13%). See
[scope.md §5](scope.md#5-dispersion) for the full layer-by-layer accounting.

A fixed, unrelaxed water dimer at PBE/STO-3G and grid level 3 gives an
XDM correction of `−1.149145694e-3` Hartree. The two fixed monomer
corrections are `−2.84653333e-4` and `−2.84158983e-4` Hartree, giving an
XDM interaction contribution of `−5.80333378e-4` Hartree. The three
SCFs converge in 9, 8 and 7 iterations, respectively. The check uses
the published PBE/aug-cc-pVTZ damping values `a1=0.4879` and
`a2=2.4683 Å` as explicit input to a different basis for a finite-energy
and sign check. The binding energy depends on geometry, basis convergence
and counterpoise treatment. [Published XDM parameters, Table S1](https://www.rsc.org/suppdata/cp/c3/c3cp51559a/c3cp51559a.pdf)

## Conventions that matter numerically

- **PW92 digits**: PBE-C and the M06/ωB97M families use the "MOD" A
  digits (0.0310907/0.01554535/0.0168869); the B97/HCTH/ωB97X families
  use the PAPER digits. Pinned pointwise vs libxc.
- **ζ=1 ferro branch**: same-spin PW92 channels use ec(rs,1) directly —
  the ζ-interpolation's (1−ζ)^{4/3} has a NaN second dual derivative at
  ζ=1 (THEORY.md §15).
- **libxc VWN_RPA** = RPA parameters + simple ζ-interpolation (no αc).
- B3LYP defaults to the VWN3/Gaussian convention (`b3lyp5` for VWN5).
- Grid: PySCF-style levels 0–9 (default 3), per-element/atom overrides.
- **PW91-C ≠ P86's C(rs)**: PW91 uses the Rasolt–Geldart **1986** Padé
  C_xc = (2.568 + 23.266rs + 7.389e-3·rs²)/(1000(1 + 8.723rs + 0.472rs²))
  — quadratic denominator, NO rs³ term — while P86 uses the RG'81 form
  with the 0.07389·rs³ term. PW91's local base is LDA_C_PW **paper**
  digits with the truncated f''(0) = 1.709921; β = ν·C_c0 with exact ν.
  Identified against libxc gga_c_pw91 by σ→0 probes (2026-08-04).
- **wPBEh numerics**: the published Fc1/Fc2/E floats of the HSE hole and
  the EG small-s quartic are used as-is (libxc 7.0.0 release
  convention). The **s-rescaling deliberately departs from libxc** — the
  one place in this project where that is true, so it is spelled out:

  libxc 7.0.0 (`maple/scaling.mpl`, its `s_scaling_2`, labelled there as
  the variant "apparently used by Gaussian") implements Henderson 2009
  Eq. (8) with two short-circuits, `s < 1 → s` and `s > 15 → smax`. Both
  are discontinuous: measured, the function jumps by **−5.141e-4 at
  s = 1** and **+1.616e-3 at s = 15**. s ≈ 1 is the valence-density
  region that carries most of the exchange integral, so these are not
  tail artefacts.

  dft-rs ships the smooth, branchless
  `f(s) = smax − ln(1+e^{smax−s}) + ln(1+e^{−smax})` instead. The offset
  matters: the bare Eq. (8) softplus maps s = 0 to **−1.892e-4**, i.e.
  negative, and is 19 % low at s = 0.001 — it violates the uniform-gas
  limit `s → 0 ⟹ s_scaled → 0` that libxc's `s < 1` short-circuit was
  enforcing. The offset restores that limit exactly while keeping C^∞.

  **Cost, measured:** HSE06 / HSE03 / LC-ωPBE totals shift by
  1.0–1.8e-4 Ha (≈ 0.1 kcal/mol), systematically, away from libxc.
  `lc-wpbe-hjs` is unaffected (HJS hole, not wPBEh) and still gated at
  full tolerance, which is itself the evidence that the deviation is
  confined to the rescaling.

  **What still gates it.** The frozen libxc points continue to pin the
  hole model pointwise at 1e-9 — the H(s) fit, the EG quartic, the
  nine-term screened kernel, erfcx/E1, spin scaling — by evaluating them
  through libxc's own rescaling, kept as a `#[cfg(test)]` reference. The
  SCF totals are gated at 5e-4 against libxc (so a real regression is
  still caught, just not the known offset) plus the reference-free
  identities: exact s → 0 limit, C^∞ continuity at s = 1 and s = 15,
  monotonicity, saturation, and the ω → 0 / ω → ∞ screening limits.
- **FHC input bound (r²SCAN)**: libxc 7's meta-GGA driver enforces
  σσσ ≤ 8ρστσ and |σαβ| ≤ (σαα+σββ)/2 on inputs for ENFORCE_FHC
  functionals (SCAN family). The MggaDriver mirrors this per-functional
  (`clamp_fhc`) — values clamped, derivative seeds left on the original
  variables, exactly like the reference driver.
- **HJS floors**: ζ(s), η(s) floored at 1e-10 (libxc 7 release value).

## Derivative capability matrix

| Method class | Energy | Analytic gradient | Analytic Hessian + APT | TDA/RPA | Polarizability/Raman |
|---|---|---|---|---|---|
| HF / LC-HF | ✅ | ✅ | ✅ | ✅ | ✅ |
| LDA / GGA / hybrids / RSH | ✅ | ✅ | ✅ | ✅ | ✅ |
| meta-GGA (M06 family) | ✅ | ✅ | ✅ | ✅ | ✅ |
| + SC-VV10 (-V functionals) | ✅ | ✅ (exact) | ✅ (nonlocal skeleton + f_nlc CPKS) | ✅ (exact f_nlc) | ✅ |
| + ECP (LANL/Stuttgart) | ✅ | ✅ | ✅ (analytic; ecp_second_contraction) | ✅ | ✅ |
| + D2/D3/CHG | ✅ | ✅ (dual-seeded) | ✅ (dual-seeded, one nesting level deeper) | n/a | n/a |

Numerical Hessians are central differences of the ANALYTIC gradients and
remain the permanent cross-check of the CPKS Hessian. Frequencies/IR come
from either path (Eckart-projected, APT intensities); Raman from the
response solver. Excited states: `tddft` command (TDA or full RPA,
singlet/triplet/spin-resolved, oscillator strengths).

Intermolecular analysis: the `sapt` command decomposes a dimer into
E_elst⁽¹⁾, E_exch⁽¹⁾ (Heitler-London — no S² truncation), coupled
E_ind⁽²⁾ (through the same CPKS solver, so every kernel above applies)
and uncoupled E_disp⁽²⁾; monomers may use any functional in the tables
above (THEORY.md §18).

## Oracles

Features are gated against frozen PySCF references in `tests/oracle_data/`
(energies, gradients, Hessians, excitations, ECP integral matrices for
Sc–Zn+Pd, organometallic Ni(CO)₄). Regeneration scripts live in
`tools/oracle/`; regenerated CSVs are reviewed like code.

Functional coverage is a census, not a habit:
`tests/p14_functional_coverage.rs` enumerates every registry entry (the
same list `--list-functionals` prints), cross-references it against the
`functional,<name>` keys of the frozen CSVs, and fails on any entry that
is neither gated nor listed in its `REFERENCE_FREE` table with a reason.
It also reads `src/xc/mod.rs` and fails if a table arm has no canonical
name, so a functional cannot be added and stay invisible to the census.

**Every registry entry now carries a frozen SCF energy**, so both excuse
tables in that file are empty. The last seven — HSE03, LC-BLYP, LC-PBE,
ωB97X-D, ωB97X-V, and bare `optx` / `hjs-pbe-x` at their registry ω — are
frozen by `tools/oracle/gen_dft_missing.py`, which refuses to write a CSV
whose resolved ω differs from the registry's.

Two of those cases needed the xc string spelled carefully, and both
mistakes would have frozen a *different functional* silently:

- **ωB97X-D**: `mf.xc = "wb97x_d"` raises `NotImplementedError` before the
  SCF starts, because PySCF's dispersion parser recognises the name as
  carrying a tail it cannot construct. `"wb97xd"` resolves to the same
  libxc entry (`HYB_GGA_XC_WB97X_D`, ω = 0.2, cx = 0.222036) without
  matching that lookup, so the semilocal part runs with no tail added —
  which is exactly the number wanted, since the CHG tail is `--disp chg`
  and gated separately.
- **HSE03**: `mf.omega` is deliberately NOT overridden. libxc carries the
  functional's two different screening constants (ω_HF = 0.15/√2,
  ω_PBE = 0.15·2^⅓) internally, and forcing a single ω collapses them.

The reference-free statements remain as independent checks rather than as
substitutes: spin scaling of exchange, the ω → 0 / ω → ∞ screening limits,
HSE03 retuned to ω = 0.11 against the **frozen HSE06** energy, the ωB97
`cx + clr = 1` / `cx₀ = 1 − cx` identities, the ω = 0 collapse of the ωB97
series onto its B97 form, and finite differences on every implemented
derivative order.

## Validation of new functional classes

Every new functional must satisfy physical checks and independent oracle
comparison before it is listed as validated. Agreement of total energies alone
can conceal compensating errors between contributions. These requirements
extend the [response validation program](response.md); implementation continues
in the order recorded in its coverage ledger. This document defines acceptance
requirements, rather than claiming that the new classes are implemented.

### Component comparisons

Record the oracle implementation/version or publication, source provenance,
geometry, basis/normalization, electronic space, charge/spin, integration grid,
occupation/temperature, correlation approximations, screening, and convergence
settings. Compare identical models and conventions. Separate fixed-density
and fixed-orbital component evaluations from fully self-consistent calculations
so that quadrature, kernel, and reference-state errors can be distinguished.
Converge numerical parameters; preserve the existing acceptance thresholds.

Compare the nuclear, one-electron kinetic/external, Hartree, semilocal exchange,
semilocal correlation, full-range exact exchange, short-/long-range exchange,
nonlocal correlation, and any added correction separately when present. Keep
the coefficient-weighted contribution and the underlying unweighted component
when the model permits both. Check that the reported components reconstruct
the total energy or free energy. Unsupported oracle models need a different
independent oracle or closed-form limit; a different model is not a benchmark.
Numerical data and provenance can be stored as frozen test fixtures. Oracle
libraries are not added to computational dependencies or required at runtime.

For each new class, compare potentials, kernel actions, and energy/parameter
derivatives independently at fixed density or orbitals. Then test relaxed
derivatives through the physical response equations. Use closed forms, small
dense systems, and independent multi-step differences/Richardson extrapolation
as complementary oracles. Every new or extended derivative acceptance requires
an independent error fit over at least four distinct steps: compare the oracle
with the analytic result and fit `log(abs(error)) = log(C) + p log(h)`. Record
the measured exponent, fit quality, units and resolved step interval. For an
O(h^2) central stencil, verify p near 2 rather than relying solely on pairwise
error ratios. Keep failed fine-step/cancellation measurements in the validation
record, use no error floors, and retain existing acceptance bounds. Closed-form
agreement supplements this fit requirement. Differences remain validation tools. Production derivatives
use analytic expressions, AD, or response equations.

### Double hybrids

The planned generalized PT2 model must report semilocal exchange/correlation,
the HF exchange contribution, and the same-spin and opposite-spin PT2 energies
separately. Compare spin channels and all functional coefficients independently
against an oracle using the same reference orbitals, orbital energies,
antisymmetrized-integral convention, correlation space, and frozen-core policy.
Distinguish restricted and unrestricted factors and test their closed-shell
limit. B2PLYP is the standard named validation example in the implementation
plan; a coefficient match alone does not constitute its validation.

Require the PT2 contribution and every correlation response leg to vanish when
its coefficient is zero. Check coefficient derivatives against the unweighted
components, spin-scaling linearity at fixed orbitals, particle number, energy
invariance under occupied/virtual rotations where physically applicable, and
independent small-system MP2 limits. Validate the orbital, amplitude, and
multiplier responses of the correlated Lagrangian. TD(A)-DH/CIS(D) corrections
and correlation-kernel response are distinct approximations: each requires
its own component comparisons, reference conventions, limiting checks,
and public report model identifier.

### Other added classes

| Class | Component and physical checks |
|---|---|
| Current-dependent DFT | Vorticity energy and current derivatives, paramagnetic/diamagnetic/contact terms, gauge covariance, gauge-consistent meta-GGA kinetic density |
| Noncollinear/SOC | Scalar and spin XC contributions, SOC matrix terms, collinear limit and spin-rotation identities when SOC is absent |
| Four-component DFT | Dirac kinetic/nuclear/Coulomb/exchange/XC terms, kinetic/magnetic balance, finite-nucleus and no-pair projector conventions, nonrelativistic limit, projector/basis response |
| Superconducting DFT | Electronic, electron-phonon, and screened-Coulomb pairing kernels, gap/anomalous density, normal-state limit, particle-hole symmetry, particle-number and phase constraints |

These checks accompany the full functional/space/q/frequency/temperature
combination tests. An interface, an isolated derivative kernel, a refusal,
or a single agreeing total energy does not complete a functional class.

## Skala (learned functional)

`xc = skala` with `skala_model = <model file>` evaluates a Skala model on
the molecular grid through the common driver. It is not a registered
semilocal functional: its weights and every hyperparameter come from the
model file. Scope, status and acceptance are in `docs/response.md`
("Skala learned functional").

## Double-hybrid composition: B2PLYP

B2PLYP is available through the existing molecular energy interfaces:
`dft-rs energy molecule.xyz --xc b2plyp --format json`,
`energy(numbers, positions, xc="b2plyp")`,
`calculate(numbers, positions, xc="b2plyp")`, and
`DftRs(xc="b2plyp").get_potential_energy(atoms)` or `get_energy_report`.
It uses 53 percent exact exchange, 47 percent B88 exchange, 73 percent
LYP correlation and 27 percent of both same-spin and opposite-spin PT2
on the resulting generalized Kohn-Sham orbitals. SS/OS energies and raw
amplitude residuals are reported separately. No extra calculation name
is introduced. See [the component theory and acceptance](response.md#double-hybrid-components-and-pt2-lagrangian).

Any existing parent can supply the orbitals by giving its usual `xc`
selection with `pt2_same_spin` and `pt2_opposite_spin` in the energy
controls. This includes GGA, meta-GGA, global and range-separated hybrids;
PT2 uses full Coulomb integrals. The zero-correlation limit recovers the
complete parent, including a learned-model XC contribution. Reports separate
`nonsemilocal_exchange_correlation_energy` in Hartree from semilocal
components and VV10. A model-provider zero-coefficient limit and nonzero
correlated dipole four-step fit are recorded in response.md. Integer zero-temperature occupations are required. The existing `dipole` API includes the analytic correlated electric
derivative with orbital and noncanonical PT2 amplitude response.
The common `polarizability` API includes relaxed static correlated response.
The common `gradient` and `hessian` APIs now include the complete parent
orbital and PT2 amplitude response, moving integral/metric/grid terms and
separate unscaled SS/OS tensors. They return all Cartesian components.
All 4368 restricted nuclear component/channel fits for HF/PBE/TPSS/PBE0
custom parents pass. B2PLYP, DSD-BLYP, DSD-PBEP86 and PWPB95 pass native
Python/ASE full-tensor comparisons. Canonical CLI and remaining reference
combinations retain their acceptance status in response.md. Distinct-target
molecular nonlocal nuclear derivatives, including omegaB97M(2), remain
pending. Relaxed density outputs and other dynamic/high-order correlated
properties are separate remaining combinations.

`xc="dsd-pbep86"` and `xc="dsd-blyp"` use the original 2013 D3(BJ)
parametrizations through these same energy APIs. Explicit `-d3bj` aliases
are accepted. The coefficients are:

| Recipe | HF exchange | DFT exchange | DFT correlation | OS PT2 | SS PT2 | BJ s6 | BJ s8 | BJ a1 | BJ a2 (Bohr) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DSD-PBEP86, 2013 BJ | 0.69 | 0.31 PBE | 0.44 P86 | 0.52 | 0.22 | 0.48 | 0 | 0 | 5.6 |
| DSD-BLYP, 2013 BJ | 0.71 | 0.29 B88 | 0.54 LYP | 0.47 | 0.40 | 0.57 | 0 | 0 | 5.4 |

These parameter sets are reproduced in the original-parameter rows of
the [authors' later comparison](https://doi.org/10.1021/acs.jpca.9b03157).
They differ from the original 2010 D2 DSD-BLYP fit and the revised D4
models. Pairwise BJ dispersion is included automatically and reported
separately; `dispersion="none"` explicitly removes it. PT2 currently
correlates all explicitly treated electrons, with no frozen-core option.
Independent electronic and dispersion term acceptance is tracked in
[response.md](response.md) and the coverage ledgers.

`xc="wb97m2"` (also `omega-b97m(2)`) selects omegaB97M(2).
It generates stationary omegaB97M-V orbitals, including the reference VV10
potential (b=6, C=0.01), then evaluates the distinct target of
[JCP 148, 241736 (2018)](https://doi.org/10.1063/1.5025226).
The target uses 0.62194 full-range plus 0.37806 additional long-range exact
exchange (omega=0.3 Bohr^-1), its own sparse meta-GGA exchange and separate
parallel/antiparallel correlation series, 0.34096 of both PT2 spin
components and 0.65904 VV10 with **b=10**, C=0.01. The b=10 target is
evaluated on the fixed reference density. It is never optimized as the
orbital-generating functional. `orbital_reference_energy` describes the
orbital functional; `reference_energy` describes the target before PT2.
`target_nonlocal_correlation`, `orbital_nonlocal_correlation`, the two
functional descriptions and `distinct_target_energy` distinguish them.

`xc="pwpb95"` selects 50% exact exchange, 50% reoptimized PW exchange,
0.731 reoptimized B95 correlation, zero same-spin and 0.269 opposite-spin
PT2, following [JCTC 7, 291 (2011)](https://doi.org/10.1021/ct100466k).
Exchange parameters (beta,c,exponent) are (0.00444,0.32620,3.7868);
B95 damping parameters (same-spin,opposite-spin) are (0.03241,0.00250).
Automatic pairwise D3(BJ) uses (s6,s8,a1,a2)=(0.82,0.2904,0,7.3141);
`dispersion="none"` explicitly selects the energy without that correction.
`pwpb95-d3bj` is an alias. The ingredients expose density and parameter
derivatives through the existing generic XC interface and arbitrary-order
Taylor paths. The fitted noninteger exchange power is nonanalytic in
higher gradient derivatives at an exactly zero gradient; no denominator
floor is introduced to disguise this mathematical restriction.

Both models use the existing Rust, CLI, native Python and ASE `energy`
interfaces. Component and full-energy acceptance is recorded in
[response.md](response.md); the same `dipole` API also includes the distinct-target
functional derivative for omegaB97M(2), including both VV10 potentials.
The remaining correlated properties are tracked separately. P86
retains the published rounded PZ81 branch join; local response
derivatives and finite-grid validation limitations are documented there.

## Functional catalogue

Status of every functional and method of the requested Jacob's-ladder
catalogue (all rungs except DFT-D4, which stays excluded). Additions
resumed after the global-hybrid batch with Minnesota functionals, further
hybrids and named double hybrids. Status meanings:

- *existing*: implemented before the catalogue request.
- *validated: pointwise, SCF, five molecules*: every component matches
  frozen libxc energy densities and first derivatives at the nine
  points of `tests/oracle_data/functional_points.json` (p225), the
  water/CH3 SCF energies of `tools/oracle/gen_dft.py` (p3), and the
  H2/LiH/water/phosphine/ferrocene PySCF energies (p237).
- *validated: ...; response fits (p225) pending in every component*: the
  response properties passed four-step fits along one component or
  direction; p225 now fits every component of the gradient, Hessian,
  polarizability, hyperpolarizability, periodic strain derivative and
  quantum-proton derivatives, and that run is not yet recorded.
- *implemented, validation pending*: in the registry; the oracle runs
  above have not all been recorded yet.
- *validated: pointwise, SCF; five molecules pending*: the libxc points
  (p225) and the water/CH3 SCF energies (p3), or the explicitly identified
  focused point/SCF suites in that row, pass; the complete standard
  five-molecule energies are not yet recorded.
- *implemented: term oracles of five molecules pending (p263)*: named
  double hybrids whose term-by-term oracles
  (`tools/oracle/gen_double_hybrids.py`) are being generated.

| Status | Count |
|---|---|
| excluded | 1 |
| existing | 59 |
| implemented: term oracles of five molecules pending (p263) | 15 |
| partial: pointwise, local first-third derivatives, ten molecular term cases and public energy routes; standard five pending (p306) | 2 |
| not implemented | 50 |
| partial: KCIS independent local and molecular term acceptance; public replay pending (p324) | 3 |
| partial: actual pointwise, local first-third and RKS/UKS term acceptance; standard-five/public and remaining molecular acceptance active | 3 |
| partial: actual pointwise, local first-third, RKS/UKS terms and C1 RKS nuclear acceptance; standard-five/public active | 1 |
| partial: actual-library pointwise, complete local first-third partials and RKS/UKS terms; molecular derivative/public/standard-five acceptance active (p308) | 1 |
| validated: pointwise, SCF, five molecules | 74 |
| validated: pointwise, SCF, five molecules; pointwise branch response by p280; 126 molecular branch-continuation fits validated; unfrozen crossings separate | 1 |
| validated: pointwise, SCF, five molecules; response fits (p225) pending in every component | 5 |
| validated: pointwise, SCF; five molecules except ferrocene (2.9e-5 against its PySCF energy, under investigation) | 1 |
| validated: pointwise, SCF; five molecules pending | 22 |

The TPSS-family total-density iso-orbital cap retains its value at one and
uses the physical inward derivative within the floating-point roundoff
enclosure of that endpoint. Genuinely super-bound inputs retain the constant
cap. This avoids inconsistent response branches when the same physical
one-orbital density is contracted by different AO providers. p283 checks
point potentials and every RKS/UKS water gradient, Hessian, alpha and beta
entry: all 266 fits pass, p=1.955170..2.013108 and minimum R2=0.99989964,
with four failed larger-step Hessian intervals retained. The TPSS coupled
directional Hessian also resolves its original failure; complete coupled
tensors and other combinations remain separate in response.md.

### LDA

| Name | Kind | Status | Note |
|---|---|---|---|
| `slater` | x/c/xc | existing |  |
| `vwn3` | x/c/xc | existing |  |
| `vwn5` | x/c/xc | existing |  |
| `pw92` | x/c/xc | existing |  |
| `svwn3` | x/c/xc | existing |  |
| `svwn5` | x/c/xc | existing |  |
| `spw92` | x/c/xc | existing |  |
| `xalpha(alpha=0.7)` | x/c/xc | validated: pointwise, SCF, five molecules | registry name xalpha07 (xalpha stays the Slater alias) |
| `pz81` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `wigner` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `hl` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `gl` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `cole-perdew` | c | not implemented | Local correlation. The published parameterization and independent checks are pending. [Cole and Perdew (1982)](https://doi.org/10.1103/PhysRevA.25.1265) |
| `spz81` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `spwlda` | x/c/xc | validated: pointwise, SCF, five molecules | alias of spw92 |

### GGA

| Name | Kind | Status | Note |
|---|---|---|---|
| `b88` | x/c/xc | existing |  |
| `pbe` | x/c/xc | existing |  |
| `revpbe` | x/c/xc | existing |  |
| `optx` | x/c/xc | existing |  |
| `lyp` | x/c/xc | existing |  |
| `p86` | x/c/xc | existing |  |
| `pw91c` | x/c/xc | existing |  |
| `blyp` | x/c/xc | existing |  |
| `bp86` | x/c/xc | existing |  |
| `hcth93` | x/c/xc | existing |  |
| `hcth120` | x/c/xc | existing |  |
| `hcth147` | x/c/xc | existing |  |
| `hcth407` | x/c/xc | existing |  |
| `pw91` | x/c/xc | validated: pointwise, SCF, five molecules; response fits (p225) pending in every component | [Perdew et al. (1992)](https://doi.org/10.1103/PhysRevB.46.6671) |
| `rpbe` | x/c/xc | validated: pointwise, SCF, five molecules; response fits (p225) pending in every component | [Hammer, Hansen and Nørskov (1999)](https://doi.org/10.1103/PhysRevB.59.7413) |
| `pbesol` | x/c/xc | validated: pointwise, SCF, five molecules; response fits (p225) pending in every component | [Perdew et al. (2008)](https://doi.org/10.1103/PhysRevLett.100.136406) |
| `pw86` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `mpw91` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `g96` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `lg93` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `ft97` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `xpbe` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `sogga` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `op` | x/c/xc | validated: pointwise, SCF, five molecules | OP correlation with B88 exchange (op/opb88); combined bop |
| `vpbe` | x/c/xc | validated: pointwise, SCF, five molecules | interpreted as PBE correlation on the VWN5 local base (libxc PBE_VWN) |
| `mpbe` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `apbe` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `pbeint` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `pbeloc` | x/c/xc | validated: pointwise, SCF, five molecules | PBEloc correlation; combined pbeloc = PBE exchange + PBEloc correlation |
| `wc` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `am05` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `bpw91` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `olyp` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `opbe` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `opw91` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `mpwlyp` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `mpwpw91` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `mpwpbe` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `g96lyp` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `xlyp` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `edf1` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `b97-d` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `sogga11` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `n12` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `gam` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `kt1` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `kt2` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `kt3` | x/c/xc | validated: pointwise, SCF, five molecules |  |

### meta-GGA

| Name | Kind | Status | Note |
|---|---|---|---|
| `tpss` | x/c/xc | existing |  |
| `r2scan` | x/c/xc | existing |  |
| `m06-l` | x/c/xc | existing | [Named molecular response](#named-m06-family-and-lc-bop-molecular-response). [Zhao and Truhlar (2006)](https://doi.org/10.1063/1.2370993) |
| `revm06-l` | x/c/xc | existing | [Named molecular response](#named-m06-family-and-lc-bop-molecular-response). [Wang et al. (2017)](https://doi.org/10.1073/pnas.1705670114) |
| `b97m-v` | x/c/xc | existing | [Mardirossian and Head-Gordon (2015)](https://doi.org/10.1063/1.4907719) |
| `scan` | x/c/xc | validated: pointwise, SCF, five molecules; response fits (p225) pending in every component | [Sun, Ruzsinszky and Perdew (2015)](https://doi.org/10.1103/PhysRevLett.115.036402) |
| `pkzb` | x/c/xc | validated: pointwise, SCF, five molecules | p300: complete first/second local components, frozen kernels, RKS/UKS energy terms,12 C1 RKS gradients and public routes pass; standard-five Fe defect3.660716174636036e-11 Hartree at unchanged1e-5 gate closes5/5; full higher tensors and provider/reference combinations remain separate |
| `revtpss` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `modtpss` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `tpssloc` | c/xc | validated: pointwise, SCF, five molecules | p301: 53 local partial and 12 C1 gradient fits, frozen kernels, RKS/UKS terms and public routes pass after explicitly approved four-entry physical inward reference correction; standard-five energy acceptance now5/5 at unchanged gates, Fe defect1.5916157e-12 Hartree; originals, failures and unchanged gates retained; higher responses remain separate |
| `kcis` | c | partial: KCIS independent local and molecular term acceptance; public replay pending (p324) | Original PW92 gas endpoints with local gap and spin-resolved self-correlation subtraction. [KCIS equations and acceptance](#kcis-correlation-and-global-hybrids). [Krieger et al. (1999)](https://doi.org/10.1007/978-1-4615-4715-0_28), [Toulouse, Savin and Adamo (2002)](https://doi.org/10.1063/1.1521432) |
| `tm` | x/c/xc | source registered | Tao–Mo density-matrix expansion and gradient-expansion exchange with TM correlation. Independent local energy and derivative tests are staged. [Tao and Mo (2016)](https://doi.org/10.1103/PhysRevLett.117.073001) |
| `revtm` | x/c/xc | source registered | Revised exchange gradient expansion and density-dependent correlation beta. Independent local energy and derivative tests are staged. [Jana, Sharma and Samal (2019)](https://doi.org/10.1021/acs.jpca.9b02921) |
| `rscan` | x/c/xc | validated: pointwise, SCF, five molecules; pointwise branch response by p280; 126 molecular branch-continuation fits validated; unfrozen crossings separate | switching interval parameters `switch_lower`, `switch_upper` |
| `task` | x/xc | validated: pointwise, SCF, five molecules | Original TASK exchange with explicit original-PW92 recipe; complete local partials through third order, all parameter partials, energy terms and public routes pass; all 12 C1 RKS nuclear fits pass; standard-five ferrocene energy defect6.59e-12 Hartree at unchanged1e-5 gate closes five-of-five energy acceptance |
| `mtask` | x/xc | validated: pointwise, SCF, five molecules | Modified TASK exchange with published coefficients and explicit original-PW92 recipe; same bounded p302 acceptance; all 12 C1 RKS nuclear fits pass; ferrocene energy defect3.1604940886609256e-11 Hartree at unchanged1e-5 gate |
| `lak` | x/c/xc | not implemented | [Lebeda, Aschebrock and Kümmel (2022)](https://doi.org/10.1103/PhysRevResearch.4.023061) |
| `b88c` | c | not implemented | The correlation-hole model requires the kinetic energy density. [Becke (1988)](https://doi.org/10.1063/1.454274) |
| `vsxc` | x/c/xc | validated: pointwise, SCF; five molecules pending | The code shares the VS98 terms with M06. The exchange fit residual is 1.9e-13. [Van Voorhis and Scuseria (1998)](https://doi.org/10.1063/1.476577) |
| `bb95` | x/c/xc | validated: pointwise, SCF, five molecules |  |
| `thcth` | x/c/xc | validated: pointwise, local derivatives, SCF, public gradients, five molecules | Pure tau-HCTH shared kinetic exchange with original-PW92 correlation refit; no HF. Actual84bd1e14:22 component points,126 first-third fits,98 interior fifth/490 zeros,2 RKS/UKS term cases,90 C1 RKS gradient/Hessian fits and2 public-gradient cases pass; standard5/5 after strict normalized-AH exact-ERI Fe reference recovery, energy defect5.68434e-12 Hartree; original timeouts retained |
| `m11-l` | x/c/xc | validated: pointwise, SCF; five molecules pending | M08 enhancement on the short- and long-range LSDA exchange. [Peverati and Truhlar (2012)](https://doi.org/10.1021/jz201525m) |
| `mn12-l` | x/c/xc | validated: pointwise, SCF; five molecules pending | [Peverati and Truhlar (2012)](https://doi.org/10.1039/C2CP42025B) |
| `mn15-l` | x/c/xc | validated: pointwise, SCF; five molecules pending | [Yu, He and Truhlar (2016)](https://doi.org/10.1021/acs.jctc.5b01082) |
| `b97m-rv` | xc | not implemented | B97M semilocal terms with the revised nonlocal kernel and b = 6.0. [Mardirossian et al. (2017)](https://doi.org/10.1021/acs.jpclett.6b02527) |

### global hybrid

| Name | Kind | Status | Note |
|---|---|---|---|
| `edf2` | xc | validated: pointwise, SCF, five molecules | p303: complete point/local/term checks and all standard H2/LiH/water/PH3/ferrocene energies pass; Fe40 iterations, error1.751e-11 Hartree at unchanged1e-5 gate; original approved boundary evidence retained; 12 C1 gradient+78 Hessian fits and public routes pass; broader combinations separate |
| `b3lyp` | xc | existing |  |
| `b3lyp5` | xc | existing |  |
| `o3lyp` | xc | existing |  |
| `b3pw91` | xc | existing |  |
| `pbe0` | xc | existing |  |
| `b97` | xc | existing |  |
| `tpssh` | xc | existing |  |
| `m06` | xc | existing | [Named molecular response](#named-m06-family-and-lc-bop-molecular-response). [Zhao and Truhlar (2008)](https://doi.org/10.1007/s00214-007-0310-x) |
| `m06-2x` | xc | existing | [Named molecular response](#named-m06-family-and-lc-bop-molecular-response). [Zhao and Truhlar (2008)](https://doi.org/10.1007/s00214-007-0310-x) |
| `revm06` | xc | existing |  |
| `b3lyp*` | xc | validated: pointwise, SCF, five molecules |  |
| `x3lyp` | xc | validated: pointwise, SCF, five molecules |  |
| `b3p86` | xc | validated: pointwise, SCF, five molecules |  |
| `b1lyp` | xc | validated: pointwise, SCF, five molecules |  |
| `b1pw91` | xc | validated: pointwise, SCF, five molecules |  |
| `bhandh` | xc | validated: pointwise, SCF, five molecules |  |
| `bhandhlyp` | xc | validated: pointwise, SCF, five molecules |  |
| `blyp35` | xc | validated: pointwise, SCF, five molecules | Global GGA hybrid: 0.35 HF + 0.65 B88 exchange + full LYP correlation; p294 point/kernel and 12 C1 gradients; p295 all five catalogue molecules |
| `pbe0-1/3` | xc | validated: pointwise, SCF, five molecules |  |
| `mpw1pw91` | xc | validated: pointwise, SCF, five molecules |  |
| `mpw1lyp` | xc | validated: pointwise, SCF, five molecules |  |
| `mpw1pbe` | xc | validated: pointwise, SCF, five molecules |  |
| `mpw3pbe` | xc | not implemented | The complete three-parameter recipe remains unidentified. The parent exchange and correlation are defined separately. [Adamo and Barone (1998)](https://doi.org/10.1063/1.475428), [Perdew, Burke and Ernzerhof (1996)](https://doi.org/10.1103/PhysRevLett.77.3865) |
| `mpw1k` | xc | validated: pointwise, SCF, five molecules |  |
| `b97-1` | xc | validated: pointwise, SCF, five molecules |  |
| `b97-2` | xc | validated: pointwise, SCF, five molecules |  |
| `b97-3` | xc | validated: pointwise, SCF, five molecules |  |
| `b97-k` | xc | validated: pointwise, SCF, five molecules |  |
| `b98` | xc | validated: pointwise, local derivatives, SCF, public gradients, five molecules | Schmider-Becke1998 fit2c (425), not Becke meta-GGA598; HF0.219847, original rounded PW92, fitted B97 terms unscaled. Actual84bd1e14:11 component points,34 first-third fits,56 interior fifth/70 zeros,2 RKS/UKS term cases and2 public-gradient cases pass; standard5/5 after strict normalized-AH exact-ERI Fe reference recovery, energy defect8.41283e-12 Hartree; no separate B98 nuclear-Hessian fit claim |
| `sogga11-x` | xc | validated: pointwise, SCF; five molecules pending | SOGGA11 form with the hybrid coefficients. [Peverati and Truhlar (2011)](https://doi.org/10.1063/1.3663871) |
| `apfd` | xc | not implemented | APF is registered. The spherical-atom dispersion model is pending. [Austin et al. (2012)](https://doi.org/10.1021/ct300778e) |
| `wc04` | xc | validated: pointwise, SCF, five molecules | Wiitala 2006 carbon-shift hybrid; explicit Libxc 7 VWN-RPA convention; full active point kernel, fixed-density components, all 12 C1 gradients and all public routes accepted by p295; final SCF2 artifact passes all five molecules after the explicitly approved one-entry Fe correction, defect1.159606e-11 Hartree; original failures retained |
| `wp04` | xc | validated: pointwise, SCF, five molecules | Wiitala 2006 proton-shift hybrid; explicit Libxc 7 VWN-RPA convention; full active point kernel, fixed-density components, all 12 C1 gradients and all public routes accepted by p295 |
| `tpss0` | xc | validated: pointwise, SCF, five molecules |  |
| `revtpssh` | xc | validated: pointwise, SCF, five molecules |  |
| `scan0` | xc | validated: pointwise, SCF; five molecules except ferrocene (2.9e-5 against its PySCF energy, under investigation) |  |
| `r2scan0` | xc | validated: pointwise, SCF, five molecules |  |
| `r2scanh` | xc | validated: pointwise, SCF, five molecules |  |
| `r2scan50` | xc | validated: pointwise, SCF, five molecules |  |
| `b1b95` | xc | validated: pointwise, SCF, five molecules |  |
| `b3b95` | xc | not implemented | The complete hybrid recipe still needs a defining reference. [B95 correlation, Becke (1996)](https://doi.org/10.1063/1.471637) |
| `mpw1b95` | xc | validated: pointwise, SCF, five molecules |  |
| `mpw1kcis` | xc | partial: KCIS independent local and molecular term acceptance; public replay pending (p324) | 0.15 full-range exact exchange, 0.85 mPW91 exchange and full KCIS correlation. [KCIS equations and acceptance](#kcis-correlation-and-global-hybrids). [Zhao, González-García and Truhlar (2005), Table 3](https://doi.org/10.1021/jp045141s), [Adamo and Barone (1998)](https://doi.org/10.1063/1.475428) |
| `pbe1kcis` | xc | partial: KCIS independent local and molecular term acceptance; public replay pending (p324) | 0.22 full-range exact exchange, 0.78 PBE exchange and full KCIS correlation. [KCIS equations and acceptance](#kcis-correlation-and-global-hybrids). [Zhao and Truhlar (2005)](https://doi.org/10.1021/ct049851d), [Perdew, Burke and Ernzerhof (1996)](https://doi.org/10.1103/PhysRevLett.77.3865) |
| `pw6b95` | xc | validated: pointwise, SCF, five molecules |  |
| `pwb6k` | xc | validated: pointwise, SCF, five molecules |  |
| `thcthhyb` | xc | validated: defining/local/SCF/public and standard-five energies | Hybrid tau-HCTH HF0.15, separately fitted local/kinetic/correlation coefficients, not a rescaled pure variant. Actual84bd1e14:22 component points,126 first-third fits,98 interior fifth/490 zeros,2 RKS/UKS term cases,90 C1 RKS gradient/Hessian fits and2 public-gradient cases pass; standard5/5 after fresh physical Fe check on3c1f7c96, energy error5.025e-11 Hartree |
| `bmk` | xc | validated: actual-build pointwise, first-third/fifth, RKS/UKS terms, C1 nuclear derivatives, public routes and standard-five energies | Published Boese-Martin fit: unscaled fitted local terms plus0.42 HF; raw references retained with separate high-precision oracle. Actual8fdc87c3:126 lower fits,98 fifth/490 zeros,90 C1 RKS gradient/Hessian fits,4 RKS/UKS public cases and5/5 molecular energies pass; no unmeasured ECP/ROKS/GKS/excited-state inference |
| `m05` | xc | validated: pointwise, SCF; five molecules pending | Shares the M06 form with the VS98 terms omitted. [Zhao, Schultz and Truhlar (2005)](https://doi.org/10.1063/1.2126975) |
| `m05-2x` | xc | validated: pointwise, SCF; five molecules pending | Shares the M06 form with the VS98 terms omitted. [Zhao, Schultz and Truhlar (2006)](https://doi.org/10.1021/ct0502763) |
| `m06-hf` | xc | validated: pointwise, SCF; five molecules pending | Full-range exact exchange coefficient of one. [Zhao and Truhlar (2006)](https://doi.org/10.1021/jp066479k) |
| `m08-hx` | xc | validated: pointwise, SCF; five molecules pending | PBE enhancement (μ = 0.21951) and RPBE-type enhancement (κ = 0.552, μ = 10/81). [Zhao and Truhlar (2008)](https://doi.org/10.1021/ct800246v) |
| `m08-so` | xc | validated: pointwise, SCF; five molecules pending | [Zhao and Truhlar (2008)](https://doi.org/10.1021/ct800246v) |
| `mn15` | xc | validated: pointwise, SCF; five molecules pending | [Yu et al. (2016)](https://doi.org/10.1039/C6SC00705H) |
| `dldf` | xc | partial: actual defining/local/SCF/public validation; standard-five Fe pending | Global hybrid meta-GGA HF0.6144129, fitted local fraction and corrected Fermi factor with shared FHC convention; no dispersion/PT2. Actual84bd1e14:24 component points,146 first-third fits,128 smooth-interior fifth/796 zeros,2 RKS/UKS term cases,90 C1 RKS gradient/Hessian fits and2 public-gradient cases pass; standard4/5, Fe finite queue; raw tau-zero continuation separate from driver floor |

### range-separated

| Name | Kind | Status | Note |
|---|---|---|---|
| `cam-b3lyp` | xc | existing |  |
| `lc-blyp` | xc | existing |  |
| `lc-pbe` | xc | existing |  |
| `lc-wpbe` | xc | existing |  |
| `lc-wpbe-hjs` | xc | existing |  |
| `wb97` | xc | existing |  |
| `wb97x` | xc | existing |  |
| `wb97x-d` | xc | existing | [Chai and Head-Gordon (2008)](https://doi.org/10.1039/B810189B) |
| `wb97x-v` | xc | existing | [Mardirossian and Head-Gordon (2014)](https://doi.org/10.1039/C3CP54374A) |
| `wb97m-v` | xc | existing | [Mardirossian and Head-Gordon (2016)](https://doi.org/10.1063/1.4952647) |
| `hse03` | xc | existing |  |
| `hse06` | xc | existing |  |
| `m06-sx` | xc | existing | [Named molecular response](#named-m06-family-and-lc-bop-molecular-response). [Wang et al. (2020)](https://doi.org/10.1073/pnas.1913699117) |
| `lrc-wpbeh` | xc | validated: pointwise, SCF, five molecules; response fits (p225) pending in every component | [Rohrdanz, Martins and Herbert (2009)](https://doi.org/10.1063/1.3073302) |
| `lc-whpbe` | xc/method | not implemented | This spelling lacks an unambiguous independent recipe. The HJS long-range hybrid has its own `lc-wpbe-hjs` entry. [Weintraub, Henderson and Scuseria (2009)](https://doi.org/10.1021/ct800530u) |
| `lc-bop` | xc/method | validated: pointwise, SCF; five molecules pending | ITYH SR-B88 + OP correlation. [Named molecular response](#named-m06-family-and-lc-bop-molecular-response). [Long-range exchange](https://doi.org/10.1063/1.1383587), [OP correlation](https://doi.org/10.1063/1.479012) |
| `lc-tpss` | xc/method | not implemented | The attenuated exchange hole and screening parameter need a defining source. [Parent TPSS, Tao et al. (2003)](https://doi.org/10.1103/PhysRevLett.91.146401) |
| `rcam-b3lyp` | xc/method | validated: pointwise, SCF; five molecules pending |  |
| `wb97x-d3` | xc | Validated on a frozen build for local derivatives through order four, NH₃ and NH₃⁺ energies and gradients, 1D/2D/3D D3 nuclear tensors through order four (p326), and six public route cases. | Original range-separated hybrid with pair-only D3 zero damping. [Recipe and measured checks](#original-wb97x-d3-range-separated-hybrid). [Lin et al. (2013)](https://doi.org/10.1021/ct300715s) |
| `wb97x-rv` | xc/method | not implemented | ωB97X semilocal terms with the revised nonlocal kernel and b = 6.2. [Mardirossian et al. (2017)](https://doi.org/10.1021/acs.jpclett.6b02527) |
| `wb97m-rv` | xc/method | not implemented | ωB97M semilocal terms with the revised nonlocal kernel and b = 6.2. [Mardirossian et al. (2017)](https://doi.org/10.1021/acs.jpclett.6b02527) |
| `m11` | xc/method | validated: pointwise, SCF; five molecules pending | [Peverati and Truhlar (2011)](https://doi.org/10.1021/jz201170d) |
| `revm11` | xc/method | validated: pointwise, SCF; five molecules pending | [Wang et al. (2019)](https://doi.org/10.1021/acs.jpca.8b11499) |
| `n12-sx` | xc/method | validated: local derivatives, RKS/UKS terms, molecular derivative/public and standard-five energies (p308) | Immutable library2a5b3ad9 and Python/CLI build pass; independent full kernels,86 local first-third fits, fourth45/fifth68 fits and two RKS/UKS term cases pass; all90 independent C1 RKS gradient/Hessian fits and all4 RKS/UKS gradient/Hessian public cases pass; standard5/5 after fresh physical Fe check on3c1f7c96, energy error4.980e-11 Hartree; original PW92 normalization, screened-HF.25/0, omega.11 |
| `mn12-sx` | xc/method | validated: pointwise, SCF; five molecules pending | [Peverati and Truhlar (2012)](https://doi.org/10.1039/C2CP42576A) |
| `hsesol` | xc | H₂ energy and C1 RKS/UKS/ROKS gradient and static polarizability validated | HJS screened PBEsol exchange with separate full PBEsol exchange and correlation. See [screened and xDH energy checks](#screened-and-xdh-energy-checks). [Schimka, Harl and Kresse (2011)](https://doi.org/10.1063/1.3524336) |
| `sx-lda` | xc/method | not implemented | Thomas–Fermi screened exact exchange with an LDA complement. [Bylander and Kleinman (1990)](https://doi.org/10.1103/PhysRevB.41.7868) |
| `ddh` | method | not implemented | The macroscopic electronic dielectric response determines the hybrid fraction. [Skone, Govoni and Galli (2014)](https://doi.org/10.1103/PhysRevB.89.195112) |
| `sc-hybrid` | method | not implemented | A self-consistent dielectric iteration updates that hybrid fraction. [Skone, Govoni and Galli (2014)](https://doi.org/10.1103/PhysRevB.89.195112) |
| `ot-rsh` | method | not implemented | Requires an outer optimization of the screening parameter using charged-system energies and orbital eigenvalues. [Stein et al. (2010)](https://doi.org/10.1103/PhysRevLett.105.266802) |

### double hybrid

| Name | Kind | Status | Note |
|---|---|---|---|
| `b2plyp` | xc | existing |  |
| `pwpb95` | xc | existing |  |
| `dsd-blyp` | xc | existing |  |
| `dsd-pbep86` | xc | existing |  |
| `wb97m(2)` | xc | existing |  |
| `b2gp-plyp` | xc | implemented: term oracles of five molecules pending (p263) | named double hybrid (`xc::double_hybrid`) |
| `b2k-plyp` | xc | implemented: term oracles of five molecules pending (p263) | named double hybrid (`xc::double_hybrid`) |
| `b2t-plyp` | xc | implemented: term oracles of five molecules pending (p263) | named double hybrid (`xc::double_hybrid`) |
| `mpw2plyp` | xc | implemented: term oracles of five molecules pending (p263) | named double hybrid (`xc::double_hybrid`) |
| `mpw2k-plyp` | xc | implemented: term oracles of five molecules pending (p263) | named double hybrid (`xc::double_hybrid`) |
| `ptpss` | xc | not implemented | Original refitted TPSS exchange/correlation and the original dispersion convention remain pending. [Goerigk and Grimme (2011)](https://doi.org/10.1021/ct100466k) |
| `dsd-pbeb95` | xc | validated: pointwise, SCF; five molecules pending | p304: original 2013 D3(BJ) recipe, .66 HF/.34 PBE-X/.55 B95-C/.09 SS/.46 OS; 46 local derivative fits, 72 structural zeros and 55 molecular term comparisons pass; all 24 C1 RKS/UKS full double-hybrid nuclear fits pass; standard five-molecule, Hessian and ECP/response combinations remain separate |
| `revdsd-pbep86-d3bj` | xc | implemented: term oracles of five molecules pending (p263) | named double hybrid (`xc::double_hybrid`) |
| `pbe0-dh` | xc | implemented: term oracles of five molecules pending (p263) | named double hybrid (`xc::double_hybrid`) |
| `pbe-qidh` | xc | implemented: term oracles of five molecules pending (p263) | named double hybrid (`xc::double_hybrid`) |
| `pbe0-2` | xc | implemented: term oracles of five molecules pending (p263) | named double hybrid (`xc::double_hybrid`) |
| `tpss-qidh` | xc | implemented: term oracles of five molecules pending (p263) | named double hybrid (`xc::double_hybrid`) |
| `scan0-dh` | xc | implemented: term oracles of five molecules pending (p263) | named double hybrid (`xc::double_hybrid`) |
| `scan-qidh` | xc | implemented: term oracles of five molecules pending (p263) | named double hybrid (`xc::double_hybrid`) |
| `r2scan0-dh` | xc | implemented: term oracles of five molecules pending (p263) | named double hybrid (`xc::double_hybrid`) |
| `r2scan-qidh` | xc | implemented: term oracles of five molecules pending (p263) | named double hybrid (`xc::double_hybrid`) |
| `wb97x-2-lp` | xc | local terms and H₂ energy terms validated | LP coefficients fit with 6-311++G(3df,3pd). See [screened and xDH energy checks](#screened-and-xdh-energy-checks). [Chai and Head-Gordon (2009), Table II](https://doi.org/10.1063/1.3244209) |
| `wb97x-2-tqz` | xc | local terms and fixed-basis H₂ energy terms validated | TQZ fit. The paper extrapolates PT2 between separate triple- and quadruple-zeta calculations. See [screened and xDH energy checks](#screened-and-xdh-energy-checks). [Chai and Head-Gordon (2009), Table II](https://doi.org/10.1063/1.3244209) |
| `rsx-qidh` | xc | partial: pointwise, local first-third derivatives, ten molecular term cases and public energy routes; standard five pending (p306) | Actual immutable RSX library8e512c3f passes p306 points/55 local fits/limits and five RKS/UKS molecular cases with11 terms each; public native/ASE/CLI H2 energy tests and all24 C1 RKS/UKS nuclear gradients pass; exact QIDH exchange3^(-1/3), full LR HF, omega.27, PT2 1/3; all four combined RSX public gradient cases pass, Fe/Hessian pending; further DH expansion deferred by user priority |
| `rsx-0dh` | xc | partial: pointwise, local first-third derivatives, ten molecular term cases and public energy routes; standard five pending (p306) | Same actual-library bounded acceptance plus all24 C1 RKS/UKS gradients, retaining one failed coarse interval and its resolved four-step fit; SR/full-LR HF.5/1, omega.33, .875 PBE-C and .125 full-Coulomb PT2; no dispersion; all four combined RSX public gradient cases pass, Fe/Hessian pending |
| `wb2plyp` | xc | validated: pointwise, derivatives, public routes; five molecules accepted | Original 2019 range-separated B2PLYP; .53 short-range/full long-range HF, omega .30, .47 short-range B88, .73 LYP and .27 full-Coulomb PT2; compiled local/term/limit tests, all 24 C1 RKS/UKS gradients and both spin public routes pass; all point kernels pass after the explicitly approved four-entry physical-limit correction; raw evidence retained; standard five-molecule acceptance passes, including the measured ferrocene total/term/SS/OS comparison |
| `wb2gp-plyp` | xc | validated: pointwise, SCF; four molecules accepted; ferrocene term comparison failed | Original 2019 range-separated B2GP-PLYP; .65 short-range/full long-range HF, omega .27, .35 short-range B88, .64 LYP and .36 full-Coulomb PT2; same complete local/term/24 C1 gradient/public-route tranche and all point kernels pass after the explicitly approved four-entry physical-limit correction; raw evidence retained; standard five-molecule acceptance pending |

### fifth rung

| Name | Kind | Status | Note |
|---|---|---|---|
| `drpa` | method | not implemented | Ground-state direct RPA correlation energy. [Furche (2001)](https://doi.org/10.1103/PhysRevB.64.195120) |
| `rpa+` | method | not implemented | Direct RPA with a short-range density correction. [Yan, Perdew and Kurth (2000)](https://doi.org/10.1103/PhysRevB.61.16430) |
| `sosex` | method | not implemented | Second-order screened exchange correction. [Grüneis et al. (2009)](https://doi.org/10.1063/1.3250347) |
| `rpt2` | method | not implemented | RPA, SOSEX and renormalized single excitations. [Ren et al. (2013)](https://doi.org/10.1103/PhysRevB.88.035120) |
| `xyg3` | method | implemented: term oracles of five molecules pending (p263) | xDH on B3LYP orbitals. [Zhang, Xu and Goddard (2009)](https://doi.org/10.1073/pnas.0901093106) |
| `xygj-os` | method | not implemented | The target VWN variant remains to be tied to an independent term check. [Zhang et al. (2011), Eq. (6)](https://doi.org/10.1073/pnas.1115123108) |
| `xdh-pbe0` | method | H₂ energy terms validated | Distinct PBE hybrid target and opposite-spin PT2 on PBE0 orbitals. See [screened and xDH energy checks](#screened-and-xdh-energy-checks). [Su and Xu (2012)](https://doi.org/10.1063/1.3703893) |

### dispersion/nonlocal

| Name | Kind | Status | Note |
|---|---|---|---|
| `vv10` | correction | existing | Variational total-density kernel; molecular and selected periodic routes. [Vydrov and Van Voorhis (2010)](https://doi.org/10.1063/1.3521275) |
| `d3(0)` | correction | existing | Geometry-only pair correction and optional ATM three-body term; molecular and periodic lattice sums. [Grimme et al. (2010)](https://doi.org/10.1063/1.3382344) |
| `d3(bj)` | correction | existing | Rational damping of the D3 pair correction with optional ATM; molecular and periodic lattice sums. [Grimme, Ehrlich and Goerigk (2011)](https://doi.org/10.1002/jcc.21759) |
| `xdm` | correction | existing | Molecular post-SCF energy with caller-supplied pair damping; density-dependent nuclear derivatives pending. [Becke and Johnson (2007)](https://doi.org/10.1063/1.2795701) |
| `d2` | correction | existing | Molecular energy and derivatives, and the 3D lattice sum through analytic first strain and fourth Gamma nuclear derivatives with published element and functional fits. [Periodic implementation and tests](response.md#periodic-d2-empirical-dispersion). [Grimme (2006)](https://doi.org/10.1002/jcc.20495) |
| `vdw-df` | XC | fixed molecular RKS density | Original Dion kernel and revPBE exchange. [Energy route and terms](response.md#fixed-density-molecular-vdw-df-family). [Dion et al. (2004)](https://doi.org/10.1103/PhysRevLett.92.246401) |
| `vdw-df2` | XC | fixed molecular RKS density | Revised internal inverse length and PW86r exchange. [Lee et al. (2010)](https://doi.org/10.1103/PhysRevB.82.081101), [Murray et al. (2009)](https://doi.org/10.1021/ct900365q) |
| `optpbe-vdw` | XC | fixed molecular RKS density | Dion kernel and the published PBE/RPBE exchange mixture. [Klimeš, Bowler and Michaelides (2010)](https://doi.org/10.1088/0953-8984/22/2/022201) |
| `optb88-vdw` | XC | fixed molecular RKS density | Dion kernel and optimized B88 exchange. [Klimeš, Bowler and Michaelides (2010)](https://doi.org/10.1088/0953-8984/22/2/022201) |
| `optb86b-vdw` | XC | fixed molecular RKS density | Dion kernel and optimized B86b exchange. [Klimeš, Bowler and Michaelides (2011)](https://doi.org/10.1103/PhysRevB.83.195131) |
| `vdw-df-cx` | XC | fixed molecular RKS density | Dion kernel and exchange interpolated between Langreth–Vosko and PW86r. [Berland and Hyldgaard (2014)](https://doi.org/10.1103/PhysRevB.89.035412) |
| `rev-vdw-df2` | XC | fixed molecular RKS density | Revised internal inverse length and revised B86b exchange. [Hamada (2014)](https://doi.org/10.1103/PhysRevB.89.121103), [Shukla et al. (2022), Eq. (10)](https://doi.org/10.1103/PhysRevX.12.041003) |
| `vv09` | correction | molecular post-SCF energy | Original nonlocal kernel with PW92 local correlation, evaluated on the total density and local spin polarization. [Common inputs and verification](response.md#r089-molecular-post-scf-dispersion-and-nonlocal-kernels). [Vydrov and Van Voorhis (2009)](https://doi.org/10.1103/PhysRevLett.103.063004) |
| `rvv10` | correction | partial: fixed molecular density | Direct real-space kernel and AO potential in `nlc::rvv10_potential`. Ammonia energy and all 36 AO potential components verified. Common driver, self-consistent reference and nuclear derivatives pending. [Verification](response.md#fixed-density-molecular-rvv10). [Sabatini, Gorni and de Gironcoli (2013)](https://doi.org/10.1103/PhysRevB.87.041108) |
| `d1` | correction | molecular energy with explicit inputs | Original harmonic C₆ mixing and damping. The caller supplies atomic references and the functional scale. [Verification](response.md#r089-molecular-post-scf-dispersion-and-nonlocal-kernels). [Grimme (2004)](https://doi.org/10.1002/jcc.20078) |
| `ts` | correction | molecular post-SCF energy | Hirshfeld-volume scaling using spherical neutral atomic densities in the molecular basis. The caller supplies the free-atom references and radius scale. [Common inputs and verification](response.md#r089-molecular-post-scf-dispersion-and-nonlocal-kernels). [Tkatchenko and Scheffler (2009)](https://doi.org/10.1103/PhysRevLett.102.073005) |
| `mbd-ts` | correction | molecular post-SCF energy | Coupled dipole oscillators using the unscreened TS atomic parameters. [Common inputs and verification](response.md#r089-molecular-post-scf-dispersion-and-nonlocal-kernels). [Ambrosetti et al. (2014)](https://doi.org/10.1063/1.4865104) |
| `mbd` | correction | molecular post-SCF MBD@rsSCS energy | Range-separated screening, frequency-dependent C₆ and screened radii with caller-supplied atom references and damping. [Verification](response.md#r089-molecular-post-scf-dispersion-and-nonlocal-kernels). [Tkatchenko et al. (2012)](https://doi.org/10.1103/PhysRevLett.108.236402), [Ambrosetti et al. (2014)](https://doi.org/10.1063/1.4865104), [Blood-Forsythe et al. (2016)](https://doi.org/10.1039/C5SC03234B) |
| `d4` | correction | excluded |  |

### strong correlation

| Name | Kind | Status | Note |
|---|---|---|---|
| `mc-pdft(t/ft)` | method | molecular translated and fully translated | State-specific and multistate on-top energies, with state-specific analytic gradient and Hessian subsets. [Model and coverage](response.md#r072-projected-multiconfigurational-nonorthogonal-dft). |
| `dft+u(dudarev)` | method | molecular fixed-geometry RKS/UKS/ROKS/GKS | Variational correction with explicit site AO indices and effective U. [Model and coverage](response.md#r087-molecular-dftu). |
| `dft+u(liechtenstein)` | method | molecular fixed-geometry RKS/UKS/ROKS/GKS | Anisotropic site Coulomb tensor and spin-covariant fully localized limit double counting. [Model and coverage](response.md#r087-molecular-dftu). |
| `pz-sic` | method | fixed-geometry real UKS | Orbital self-Hartree and self-XC subtraction. [Model and coverage](response.md#r088-molecular-pz-sic). |
| `flo-sic-fixed` | method | fixed-descriptor public source route | Real Fermi–Löwdin orbitals on a frozen integer UKS determinant, with the existing PZ orbital self-energy evaluation. Variational orbital and descriptor optimization remains open. [Model and coverage](response.md#fixed-descriptor-fermi-lowdin-sic). [Pederson, Ruzsinszky and Perdew (2014)](https://doi.org/10.1063/1.4869581) |
| `local-hybrid(lh07t-svwn)` | method | not implemented |  |
| `tmhf` | method | not implemented |  |
| `b05` | method | not implemented |  |
| `b13` | method | not implemented |  |
| `kp16` | method | not implemented |  |

### global hybrid

| Name | Kind | Status | Note |
|---|---|---|---|
| `bb1k` | xc | validated: pointwise, SCF, five molecules | added with the B95 family (not in the requested list) |
| `mpw3pw91` | xc | validated: pointwise, SCF, five molecules | B3-type mPW hybrid added with the B3 set (not in the requested list) |

## PKZB definition and acceptance scope (2026-10-30)

The original PKZB exchange and correlation are registered as `pkzb` in the
exchange, correlation and combined tables. They follow Perdew, Kurth,
Zupan and Blaha, [PRL 82, 2544 (1999)](https://doi.org/10.1103/PhysRevLett.82.2544)
and its [erratum, 82, 5179](https://doi.org/10.1103/PhysRevLett.82.5179),
with numerical reference identifiers 213 and 239. This is a pure meta-GGA
with no built-in exact exchange, dispersion or nonlocal term.

For each spin channel, exchange uses
`p = sigma/(4 (6 pi^2)^(2/3) rho^(8/3))`,
`q = (9/20)(tau/tau_uniform-1)-p/12`, and
`x = (10/81)p + (146/2025)q^2 - (73/405)pq + [D+(10/81)^2/kappa]p^2`,
where `D=0.113`, `kappa=0.804`, and
`Fx=1+kappa-kappa/(1+x/kappa)`. Kinetic densities include the factor one-half.
Correlation uses PBE/PW92-MOD with `C=0.53`:
`rho eps_PBE [1+C (sum tauW_spin/sum tau_spin)^2]`
`- (1+C) sum rho_spin eps_PBE_spin (tauW_spin/tau_spin)^2`,
where `tauW_spin=sigma_spin/(8 rho_spin)`.

The existing generic meta-GGA Fermi-hole input convention is used, including
unchanged derivative seeds at clamped values. Density cutoffs and input
boundaries limit smoothness; interior derivative fits must remain within the
physical domain. Dense derivatives are exposed through order three, with an
explicit error at order four; Taylor interfaces use the shared scalar body.
Neither interface availability nor a first/second local-kernel comparison
establishes complete higher-order or excited-state Hessian acceptance.

`tools/oracle/gen_pkzb.py` freezes 11 points per component, first potentials
and every active local kernel entry, plus spherical STO-3G/grid-9 RKS-water
and UKS-methyl energies and separate fixed-density exchange/correlation
terms. Their independent term reconstructions close to 1.422e-14 and
7.106e-15 Hartree. `tests/p300_pkzb.rs` owns production acceptance; its
five tests pass on the separate immutable production build. All 53
nonzero independent local first/second partials pass four-step fits:
`p=1.977406..2.015917`, minimum `R2=0.99996415`.
The 12 C1 RKS ammonia gradient entries pass at
`p=1.999408..2.001571`, minimum
`R2=0.99999984`, using
`h=[0.04,0.02,0.01,0.005]` Bohr, grid level 5, cutoff zero.
Local fits use the physical rho/sigma/tau interior; all failed intervals
and resolved stencils are retained without error floors or relaxed bounds.

Maximum frozen energy/potential/kernel absolute errors are
`[7.772e-16,4.441e-16,3.411e-13]` for exchange and
`[4.281e-12,4.949e-11,8.944e-8]` for correlation in their atomic-unit
channels. The unchanged per-entry comparison bounds are
`1e-11*(1+abs(reference))` for energy density,
`1e-10*(1+abs(reference))` for potentials and
`1e-9*(1+abs(reference))` for kernels. Low-tau clamped points have large
kernel entries; the absolute maximum must be interpreted with that scaled
bound. Every full active 7x7 kernel is compared; empty-spin singular mixed
channels are excluded explicitly and physical one-electron limits are
checked independently.

RKS-water and UKS-methyl SCF errors are
`+3.566925e-12` and `+3.453238e-12` Hartree.
Fixed-density core/Hartree errors are at most 1.848e-13 Hartree,
exchange 2.132e-14 and correlation 3.831e-12 Hartree; the complete energy
reconstruction also passes. Nine registry tests, the bounded registry
Taylor consistency test and both native-Python/ASE/canonical-CLI cases
pass. Maximum route differences are `1.582068e-15` Hartree/Bohr
and `2.842171e-14` Hartree. These route checks do not establish
independent UKS nuclear derivative fits.

The cooperating local-Taylor memory-repair library also passes the four
local/energy p300 tests, with exact equality of every XC source file to
its snapshot. The complete molecular FD fit sweep is retained from the
first immutable artifact rather than repeated for an unchanged XC body.
The remaining PKZB ferrocene catalogue case, full local fourth/fifth tensors,
general Hessians and excited-state/electric/magnetic response, ECP,
ROKS, complex GKS and periodic combinations remain unvalidated here.

### PKZB complete smooth local third tensor

All 104 independent active local third partials now pass four-step finite differences of the second-order kernel at the smooth seven-variable point (0.31, 0.22, 0.041, 0.017, 0.029, 0.37, 0.21): 20 exchange and 84 correlation entries. All 64 remaining exchange triples are exact structural zeros. Accepted exponents span 1.959029--2.005305, minimum R-squared 0.99987827. Of these, 103 pass steps [0.008, 0.004, 0.002, 0.001]; correlation (sigma-aa,sigma-aa,sigma-bb) fails that initial fit (p=1.808258, R-squared 0.99649033), retained in the log, and passes the predeclared second interval [0.004, 0.002, 0.001, 0.0005] with p=1.959029 and R-squared 0.99987827. No floors or gate changes are used. This is smooth local third-order acceptance; fourth/fifth orders, clamp or branch boundaries and molecular Hessian/response combinations remain separate.

### PKZB bounded standard-molecule gap closure

The p300 catalogue cases for H2, LiH, water and phosphine all pass against
independent level-9 spherical STO-3G RKS references, with errors
`+7.935874e-13`, `+2.129852e-12`, `+3.552714e-12` and `+6.195933e-12`
Hartree respectively. The original 5e-7 Hartree bounds and SCF convergence
gates are unchanged. Each reference/test process has an explicit wall-time
budget and runs serially with two threads. The peer's immutable local-Taylor
repair library supplies the production calculations; the PKZB scalar body
matches exactly. Previously frozen BLYP35/WC04/WP04 entries are preserved.
The subsequent one-shot ferrocene reference attempt reached its 180-second
wall-time budget without a terminal SCF result. No new reference value was
frozen and the Rust comparison was not run; the row remains
five-molecule-pending. The other four frozen values and all gates remain
unchanged. This timeout does not establish scientific nonconvergence.
The separate interrupted WC04 ferrocene case is unchanged.

## TASK and mTASK definition and acceptance scope (2026-10-30)

TASK follows Aschebrock and Kuemmel,
[PRResearch 1, 033082 (2019)](https://doi.org/10.1103/PhysRevResearch.1.033082),
equations (21), (23) and (30). mTASK follows Neupane et al.,
[PRMaterials 5, 063803 (2021)](https://doi.org/10.1103/PhysRevMaterials.5.063803),
including supplemental equations (10)--(11) and their coefficient table.
Independent numerical exchange identifiers are 707 and 724. Both are
exchange components; each named combined recipe explicitly adds original
PW92 correlation (`pw92`, independent identifier 12). There is no separately
invented TASK or mTASK correlation provider.

The enhancement factor is `h0*g + (1-f)*(h1-h0)*g^d`, with rational
Chebyshev series in `s^2` and alpha, `g=1-exp(-c/sqrt(s))`, `c=4.9479`
and `d=10`. The published rounded coefficients are retained, including
`h0=1.174` for TASK and `h0=1.29` for mTASK. Consequently, rounding errors
in the uniform and iso-orbital constraints are retained rather than
silently normalized away. The spin-scaling relation and half-weighted tau
convention match the shared meta-GGA interfaces. The numerical reference
convention first applies the shared FHC input-value clamp and then floors
the fractional Pauli factor `1-tau_W/tau` at 1e-10; it does not floor alpha
itself. Initial no-FHC arithmetic fails the low-tau reference point by
3.209e-7 and 3.520e-7 Hartree/Bohr^3. Applying the existing FHC input
convention reduces the independent arithmetic error across both sets of
11 points to 8.882e-16; the initial diagnostic remains recorded.
The smooth extension `g(0)=1` is explicit, avoiding singular powers at
exactly zero gradient. All eleven coefficients are scalar parameters,
including the power `d`.

The bodies use the generic meta-GGA derivative and Taylor dispatch without
name-based response branches. The dense derivative cap stays at order
three. On the shared immutable build, p302 passes all nine core tests:
36 first/second exchange partial fits, 40 complete independent third
exchange partial fits, 22 parameter partial fits and nine original-PW92
first/second/third partial fits. The 128 inactive exchange triples are
exact structural zeros. Complete frozen exchange and combined energy,
potential and kernel comparisons pass; maximum exchange errors are
7.772e-16, 5.108e-15 and 1.865e-11 in the corresponding atomic units.
All four RKS-water/UKS-methyl energy and fixed-density term cases pass,
maximum SCF error 2.416e-13 Hartree. Four native/ASE/CLI cases pass,
maximum route differences 5.685e-14 Hartree and 2.665e-15 Hartree/Bohr.
Nine registry tests and the bounded Taylor consistency test also pass.
All 24 C1 RKS gradient fits pass using steps [0.04,0.02,0.01,0.005] Bohr,
p=1.999556--2.001902, minimum R-squared
0.99999975; no nuclear interval fails. Five-molecule,
excited-state Hessian, ECP/ROKS/GKS and periodic combinations remain
unvalidated here. Higher-order provider acceptance must use its own
complete tensor evidence, rather than the scalar interface alone.

The strict combined-point check also exposed an existing original-PW92
normalization mismatch: its rounded fit parameters had been used with
the precise PW92-MOD spin normalization. Original PW92 now uses the
reference convention `f_zeta''(0)=1.709921`; the shared PW92-MOD helper
retains its precise expression. No input reference or gate is changed.
The initial combined-point failure (maximum energy 7.907e-11 and kernel
4.439e-8) is retained. Exact-source original-PW92 energy/potential/kernel
errors are at most 2.221e-16, 9.715e-17 and 6.662e-16; all nine local
partial fits pass, p=1.999924--2.000818, minimum R-squared 0.99999996.
The 165 local PW92-MOD bit patterns remain equal across the refactor, and
the complete built-library TPSS/revTPSS/PBEloc regression also matches
all 4554 finite values bit for bit through dense order three.

The cooperating complete fifth-tensor contract also passes on the same
shared artifact at the smooth interior point
`(0.14,0.085,0.5,0.1,0.38,1.05,0.92)`: 42 active exchange components per
model and six shared PW92 correlation components, 90 distinct partials
in total, plus 840 exact exchange structural zeros. The 21 failed
exploratory measurements are retained and resolved by the predeclared
second-/fourth-order stencils at unchanged gates; accepted exponents
span 1.928354--4.134520, minimum R-squared 0.99909330. The separate
PW92 six-component run passes its first interval with
p=1.968396--2.007210 and minimum R-squared 0.99992838; its repeated use
in both recipes does not create another independent correlation body.
These are local fifth-derivative tests, not molecular excited-state
Hessian or branch-boundary acceptance. The dense order-three cap is
retained; the higher test uses the generic bidirectional Taylor interface.

## TPSSloc definition and pending acceptance (2026-10-30)

TPSSloc correlation follows Constantin, Fabiano and Della Sala,
[PRB 86, 035130 (2012)](https://doi.org/10.1103/PhysRevB.86.035130),
with independent numerical identifier 247. Its localized PBE base uses
`beta=0.0375+0.08*t^2*(1-exp(-rs^2))`, and the TPSS correlation construction
uses `C(0,0)=0.35`, unchanged higher spin coefficients and `d=4.5`.
The `tpssloc` combined recipe explicitly pairs this correlation with TPSS
exchange. There is no separate TPSSloc exchange component. Shared TPSS,
revTPSS and PBEloc arithmetic/parameter layouts are retained, and their
order-zero-through-three outputs match the previous immutable library in all
4554 finite floating-point bit patterns. This regression is separate
from independent scientific oracles.

The new scalar body uses the generic meta-GGA derivative/Taylor machinery;
it adds no name-based response path. Dense order caps remain unchanged.
Max, iso-orbital, density and input-clamp boundaries constrain smoothness;
fifth-order availability required by full-spin excited-state kernels must
not be inferred from first/second local or molecular-gradient checks.
The warning-strict library build passes. In p301, all 53 nonzero local
first/second partial fits and all 12 C1 molecular-gradient fits pass; the
latter use steps 0.04, 0.02, 0.01 and 0.005 Bohr, with exponents
1.999630--2.000870 and minimum R-squared 0.99999995. RKS/UKS energy terms,
physical limits, nine registry tests and the selected Taylor contract pass.
Two native/ASE/CLI cases pass, with maximum energy/gradient route differences
5.685e-14 Hartree and 2.929e-15 Hartree/Bohr.

Full pointwise acceptance is **failed**, not complete: two p301 tests fail
on four zero-gradient exchange/combined kernel diagonal comparisons at
point 10, sigma-aa and sigma-bb. Raw reference magnitudes are about 1e22,
whereas production values are about -0.0282, -0.1095, -0.0237 and -0.1050.
Read-only independent inward-potential diagnostics support finite values:
all four four-step one-sided fits pass (exponents 2.020824--2.047667,
minimum R-squared 0.99996177), with analytic/independent-limit differences
below 1.152e-13. These diagnostics do not promote the failed comparisons.
The original fixture, gates and failure logs are retained. Automatic
approval review rejected replacing these reference entries because doing
so could mask a defect; a concrete unapplied correction awaits explicit
approval. Five-molecule, higher-response and general excited-state Hessian
acceptance remain pending; no ECP/ROKS/GKS combination is inferred.

## Added global-hybrid acceptance (2026-10-30)

### WC04/WP04 definition and bounded acceptance

The next catalogue batch uses the named numerical definitions
`HYB_GGA_XC_WC04` (611) and `HYB_GGA_XC_WP04` (612) of Libxc 7.0.0,
associated with Wiitala, Hoye and Cramer, JCTC 2, 1085 (2006),
[doi:10.1021/ct6001016](https://doi.org/10.1021/ct6001016).
The explicit coefficients multiplying complete component energies are:

| Name | HF exchange | Slater exchange | B88 exchange | VWN-RPA correlation | LYP correlation |
|---|---|---|---|---|---|
| WC04 | 0.7400 | -0.9998 | 0.9999 | 0.9998 | 0.0001 |
| WP04 | 0.1189 | 0.0385 | 0.9614 | 0.9998 | 0.0001 |

These are five independent fitted coefficients. Replacing Slater's weight
with `1-cx-cB88`, or forcing the correlation weights to sum to one, would
define a different functional. The uniform-density semilocal exchange
amplitudes are consequently 0.0001 and 0.9999 times Slater, respectively.
There is no implicit dispersion/nonlocal term.

Pointwise energies, potentials and density kernels here mean the semilocal
part. Nonlocal HF exchange is compared separately through the orbital
energy terms and the relaxed molecular gradients.

The [reference definition](https://gitlab.com/libxc/libxc/-/raw/7.0.0/src/hyb_gga_xc_b3lyp.c)
explicitly notes that the paper does not unambiguously identify the local
VWN fit and chooses VWN-RPA. This implementation adopts that named release
convention and records it in discovery descriptions and oracle provenance.
It does not claim that the original publication uniquely specifies the fit,
or that results using VWN5 are equivalent. A unique original-paper local-fit
interpretation remains an exact provenance limitation.

The next priorities are (1) complete this bounded hybrid batch and the
existing five-molecule gaps, (2) compositions whose component equations and
coefficients can be established independently, then (3) missing semilocal
components with full boundary and derivative acceptance. Nonlocal vdW
kernels, screened/meta components, double-hybrid recipes and fifth-rung or
strong-correlation methods require their own component, stationarity and
response work; a name alias is not an implementation. In particular,
`mpw3pbe` remains blocked on an exact local-correlation/mixing convention,
`cole-perdew` on a published parametrization and independent oracle, and
`apfd` on its added dispersion model rather than its already available APF
electronic part.

`tools/oracle/gen_nmr_hybrids.py` freezes nine point energies, first
potentials and every active 5x5 density-kernel element, separate component
energies, and fixed-density RKS/UKS energy terms. All orbital/density data
are numerical, in canonical spherical AO order. `tests/p295_nmr_hybrids.rs`
checks these values and every active density and C1 molecular-gradient
component; production continues through the generic drivers. Independent
five-molecule energies for BLYP35/WC04/WP04 use the existing H2, LiH, water,
phosphine and ferrocene geometries and spherical STO-3G basis at grid 9.
The dedicated named-hybrid fixtures are equivalent pointwise/SCF gates to
the older p225/p3 fixture sets: the nine points are shared unchanged, every
component is independently compared, and the four SCF CSVs are also discoverable
by the general energy-oracle loader. They do not imply that all older
functional response suites were repeated for these new names.

All five non-catalogue p295 tests pass against the actual shared immutable
library: 46 nonzero density-potential/kernel fits and all 24 C1 molecular
gradient fits, with the seven structural sigma zeros per model and all tau
zeros checked exactly. Frozen full-kernel maximum errors are 3.411e-13
(WC04) and 2.274e-13 (WP04) in their atomic-unit channels. Maximum pointwise
potential error is 4.441e-16 for both; maximum complete energy-density errors
are 4.718e-16 and 5.551e-16 Hartree/Bohr^3. Maximum unweighted component
energy-density error is 2.506e-15 Hartree/Bohr^3.

The RKS water and UKS methyl SCF errors are 2.629008e-12 and 2.870593e-12
Hartree for WC04, and 2.174261e-12 and 2.501110e-12 Hartree for WP04.
Maximum integrated electron-count defect is 1.210090e-10 electrons.
Every fixed-density core, Hartree, exact-exchange and separate Slater,
B88, VWN-RPA and LYP term passes; the largest component-energy difference
is 2.913225e-12 Hartree. Neither coefficients nor acceptance thresholds
are adjusted to obtain these comparisons. All nine registry coverage/limit
tests, the bounded registry-wide Taylor check and all six RKS/UKS
native-Python/ASE/canonical-CLI gradient comparisons pass. The portable
artifact audit, Python lint, API-reference consistency and coverage-ledger
checks also pass. BLYP35 and WP04 each pass all five catalogue molecules.
WP04 absolute energy errors are at most 6.435e-11 Hartree, including
ferrocene; its final fresh density-map defect is 8.939e-11 and orbital
residual is 4.712e-12. All original energy and stationarity gates remain
unchanged. WC04 passes H2, LiH, water and phosphine, while ferrocene
remains unaccepted: its execution was interrupted without a terminal status.
The last recorded state is iteration 52, energy -1617.516003688749 Hartree
and SCF residual 1.051e-4. The termination cause is uncertain; neither a
300-iteration convergence failure nor a watchdog signal is established.
A lower nonstationary energy does not establish a distinct physical root.
Independent diagnostic grid-4 minao and one-electron initial guesses both
reach their 200-iteration bound without meeting the requested 1e-9
orbital-gradient norm: 1.7294e-8 and 5.9382e-8, respectively. Their
energies agree near -1617.51486160477 Hartree. These failed diagnostics
are preserved and do not replace the grid-9 scientific reference.

### BLYP35 measurements

BLYP35 is the global GGA hybrid `0.35 Ex(HF) + 0.65 Ex(B88) + Ec(LYP)`
of Renz, Theilacker, Lambert and Kaupp, JACS 131, 16292 (2009),
[doi:10.1021/ja9070859](https://doi.org/10.1021/ja9070859). It uses the
existing generic semilocal and exact-exchange providers, with no implicit
dispersion/nonlocal correction or name-based response branch.

`tools/oracle/gen_catalogue_hybrids.py` freezes independent numerical values
from PySCF 2.14.0/libxc 7.0.0 in `tests/oracle_data/catalogue_hybrids.json`
and two `*_blyp35_dft.csv` fixtures. It evaluates B88 and LYP separately,
and stores spin density matrices in the canonical spherical AO convention.
The molecular model is all-electron spherical STO-3G, integer populations,
zero temperature, grid level 9, no frozen core, solvent or extra correction.
H2O is RKS and the methyl doublet is UKS. Fixed-density comparisons isolate
the component equations from independently relaxed SCF comparisons.

`tests/p294_catalogue_hybrids.rs` passes all five tests:

| Measurement | Recorded result |
|---|---|
| Nine fixed density/gradient points, exchange/correlation and first potentials | Passed at the unchanged 1e-11 energy-density and 1e-10 potential bounds |
| H2O / methyl SCF energy errors | +1.492e-12 / +1.720e-12 Hartree |
| Integrated electron-count defects | +1.205e-10 / -9.029e-11 electrons |
| Fixed-density core / Hartree / unweighted exact exchange | Maximum error 1.137e-13 Hartree |
| Fixed-density B88 / LYP | Maximum errors 2.702e-12 / 2.220e-16 Hartree |
| All five active potentials and 18 nonzero entries of the 5x5 density kernel | 23 four-step fits, p=1.991142..2.006570, minimum R2=0.99999464 |
| Seven structurally zero sigma-kernel entries and all inactive tau entries | Exact zeros checked independently of order fits |
| All 12 Cartesian gradient components of a noncoplanar C1 ammonia geometry | p=1.999584..2.001368, minimum R2=0.99999988 |
| Spin interchange, zero density and uniform-density exchange limits | Passed |

Density fits use h=[0.008,0.004,0.002,0.001] in the corresponding atomic-unit
density/sigma coordinates. LYP is linear in sigma: the original single-axis
fits for kernel [rho_b,sigma_aa] and [rho_a,sigma_bb] reached roundoff
(errors of roughly 1e-15..1e-13), and every failed interval is retained in
the local validation record. The initial coarsest fourth-order sigma_bb
stencil additionally crossed negative sigma_bb; that failed sample is
retained and is not a physical acceptance interval. The fallback now stays
inside the physical domain. Subtracting central differences along
`e_sigma + e_rho_a` and `e_rho_a` isolates each same partial derivative
with a measurable quadratic remainder. No error floor or threshold change
is used. Nuclear energy differences use h=[0.04,0.02,0.01,0.005] Bohr and
`grid_weight_cutoff=0`; gradients are Hartree/Bohr.

`python/tests/test_catalogue_hybrids.py` passes both neutral RKS and cationic
UKS C1 ammonia cases through native Python, ASE and the canonical CLI,
comparing every gradient entry at 1e-10 Hartree/Bohr and energy at 1e-10
Hartree. Measured maximum route differences are 4.441e-15 Hartree/Bohr
and 4.263e-14 Hartree; native/common/ASE agree exactly in these fixtures.
The CLI SCF stderr trace is also checked. Rust uses the same common
driver. Registry coverage, Python lint, the portable artifact audit and
coverage-ledger consistency pass.

The registry-wide directional Taylor check also exposed an existing revM11
cancellation failure, reproduced with the earlier library. Its original
central and fine fourth-order intervals remain in p225. A generic coarser
fourth-order interval h=[1.6,0.8,0.4,0.2] gives p=3.9890, R2=0.999997 at
unchanged bounds, so the aggregate passes. BLYP35's own directional cubic
check gives p=1.9908, R2=0.999984. This directional check does not certify a
complete third-/fourth-order density tensor.

The BLYP35 registration checkpoint left 69 unimplemented functional/method/correction
entries. With the two WC04/WP04 registrations pending acceptance above,
67 entries remain unimplemented; D4 remains separately excluded. BLYP35's
five-molecule catalogue now passes `p295_nmr_hybrids::catalogue_blyp35_five_molecules`
on the same independent H2/LiH/water/phosphine/ferrocene geometries and grid 9,
with errors +1.817e-12, +5.329e-12, +1.535e-12, +2.103e-12 and -9.777e-12
Hartree, respectively. The unchanged five-molecule bounds are 5e-7 Hartree
and 1e-5 for ferrocene. Complete molecular Hessian/electric/high-order response and periodic,
ROKS/GKS, relativistic, learned and nonlocal combinations are not certified
by this bounded acceptance. No whole-program completion or performance
improvement is inferred.


## EDF2 staged definition and validation limits (2026-10-30)

EDF2 follows Lin, George and Gill,
[Aust. J. Chem. 57, 365 (2004)](https://doi.org/10.1071/CH03263),
equation (2), with independent numerical identifier 476. It is a global
GGA hybrid, with exact-exchange coefficient 0.1695, rather than a pure
GGA. Its six semilocal weights are 0.2811 Slater, 0.6227 B88,
-0.0551 modified B88 (beta=0.0035), 0.3029 VWN5, 0.5998 LYP and
-0.0053 modified LYP (a=0.055, b=0.158, c=0.25, d=0.3505).
The modified B88 is a single exchange basis component, not the entire
EDF1 recipe. These published weights are independent: the semilocal
exchange sum is 0.8487, and neither exchange nor correlation is normalized.
The explicit VWN5 convention matches the named numerical identifier;
the paper itself specifies VWN without a numbered variant.

Source registration, optional reference generation and p303 acceptance
tests are staged. The warning-strict literal-recipe preflight uses the
already validated guard14 library, configurable existing components and
the corrected VWN5 parameter. It is not a build of the new registration.
It passes 16 local first/second partial fits, 17 independent third
partials, nine VWN5 density partials through third order and six B88/EDF2
inward-boundary fits: 48 fitted components in total. These results do
not certify the new production default or its molecular/public routes.

The preflight caught a pre-existing VWN5 stiffness-coefficient typo:
`-0.01688686894039` differs from the defining `-1/(6*pi*pi)`
(`-0.0168868639403896`). Only this source constant is corrected; no
reference or tolerance is changed. All six component energies and
potentials then agree with the frozen reference, maximum energy-density
and potential differences 3.064e-15 and 5.552e-16. The default affects
VWN5-dependent recipes; their new compiled integration remains pending.

The original frozen full-kernel test still fails six diagonal sigma
entries at point 10 (zero sigma): two each for B88, modified B88 and
the weighted EDF2 recipe. All original values, tolerances and failure
logs are retained. Independently differentiating the published B88
formula gives the inward limit `12*beta^2/rho_spin^4`; all six native
limits and four-step fits agree with that formula. This independent
check does not replace or promote the failed frozen comparisons.

The shared EDF2/VWN5 integration build launch was canceled by the user
before any artifact was confirmed. It was not retried. Thus compilation
of the new source, RKS/UKS SCF/term acceptance, native/ASE/CLI routes,
the prepared complete C1 gradient/Hessian suite, five standard molecules,
higher response and ECP/ROKS/GKS/excited-state combinations remain pending.
The earlier TPSSloc reference proposal remains unapplied pending approval.


### TPSSloc independent zero-gradient finding (2026-10-31)

The [independent boundary investigation](response.md#tpssloc-zero-gradient-reference-investigation-2026-10-31)
establishes finite physical inward kernels from paper-equation series,
80/120/160-digit arithmetic and two positive-side sequences per entry.
Direct native calls reproduce the original huge references and verify
their packed-index mapping. The failure mechanism is catastrophic
cancellation at the reference library's tiny positive sigma floor,
not a physical divergence at fixed strictly positive rho and tau.
All four original comparisons remain failed and unchanged; neither the
old finite proposal nor the newly derived values have been applied.
See the linked derivation for the physical-domain and provenance limits.


## EDF2 resumed compiled acceptance (2026-10-31)

After explicit user approval, the previously canceled shared build was
resumed once, using immutable guard14 plus only the bounded EDF2 registry,
the exact VWN5 stiffness-A correction and the cooperating eigenpair group.
Warning-strict library/Python/CLI compilation and portable audit pass.
Library SHA256 is3d8d7d44ae3965fa0f2d6bfb27e5c51584ac3ed9661014ffc1784c4d40caf5a0.
Later SCF recovery and correlated propagator changes are not in this
artifact. The earlier canceled-build record is historical; the current
catalogue status is partial acceptance because the original six raw
zero-sigma point-kernel comparisons still fail, unchanged.

On this actual artifact, all 48 local derivative/inward-limit fits pass
(p=1.959958--2.010538, minimum R-squared
0.99996571). All six component energy densities,
potentials and other frozen kernels pass; the failed entries remain
B88/modified-B88/combined sigma diagonals at point10. RKS water and UKS
methyl SCFs and all 18 fixed-density term comparisons pass, maximum
SCF difference 1.485034e-12 Hartree. These include core, Coulomb,
unscaled exact exchange and each of the six independent semilocal pieces.

All 12 C1 neutral RKS gradient components and all 78 independent Hessian
entries pass four-step fits using [0.02,0.01,0.005,0.0025] Bohr, grid5,
cutoff zero: p=1.999377--2.000534, minimum
R-squared 0.99999995. Every component resolves
on the first prescribed interval; no floors or gate changes are used.
The two native/common/ASE/CLI energy/gradient cases pass, with maximum
energy/gradient route differences 5.684342e-14 Hartree and
4.440892e-15 Hartree/Bohr. Registry9 and the bounded Taylor test1 pass.
Public Hessian routes are not certified by the gradient-only route suite.

The six parameter-distinct component fifth tensors pass 74 nonzero fits
and442 defining-equation zeros at one smooth interior point:
p=1.971098--2.025567, minimum R-squared
0.99994040. Labels distinguish part index and
parameters, including both B88 and both LYP variants. The initial attempt
to fit powers on affine-sigma LYP's exact zeros remains logged; the final
contract asserts those exact zeros and retains all nonzero fit gates.
Weighted-recipe independent dense-third contractions also pass. This is
factorized smooth-local coverage, not molecular excited-state Hessian
acceptance. The new bundle separately passes the peer's three eigenpair
contracts (24 complex-vector derivative fits); those tests do not establish
EDF2 excited-state physics.

Evidence is `.work/validation/edf2-20261031-final-results.json`, with
source/artifact identities and exact commands. Five standard molecules,
independent UKS nuclear fits, ECP/ROKS/GKS, periodic and excited-state
Hessian combinations remain separate. At this EDF2 checkpoint the TPSSloc
proposal was still pending; the subsequent explicit approval and narrowly
scoped application are recorded below. No docs Markdown/CSV file was added.

### Original DSD-PBEB95-D3(BJ) recipe (2026-10-31)

`dsd-pbeb95` implements the original 2013 BJ-damped fit, also accepted as
`dsd-pbeb95-d3bj` and `dsd-pbeb95-2013`. Its coefficients are 0.66 exact
exchange, 0.34 PBE exchange, 0.55 standard B95 correlation, 0.09 same-spin
PT2 and 0.46 opposite-spin PT2. Automatic pairwise D3(BJ) uses
`(s6,s8,a1,a2)=(0.61,0,0,6.2)` with no ATM term. These are independent
weights; their correlation sum is not normalized. The original authors'
[later comparison, original-parameter row](https://pmc.ncbi.nlm.nih.gov/articles/PMC9479158/)
identifies this fit separately from the dispersion-free, PBE-hole and
revised DSD variants. The original publication is
[JCC 34, 2327 (2013)](https://doi.org/10.1002/jcc.23391).

The typed double-hybrid provider uses existing PBE-X and standard B95-C
(`beta_ss=0.038`, `beta_os=0.0031`, PW92-MOD), and existing spin-resolved
PT2 and D3(BJ) implementations. Ordinary semilocal resolution deliberately
rejects this name rather than silently dropping PT2. No response equation
or provider callback was changed. PTPSS remains unimplemented until its
original refitted TPSS coefficients and dispersion convention are pinned
down; substituting ordinary TPSS would define a different functional.

The immutable `dsd-pbeb95-integration-20261031` Python/library/CLI bundle
passes the strict build and portable-artifact audit. `p304_dsd_pbeb95`
passes all four local/recipe/term tests: 46 complete active first, second
and third partial fits, 72 defining-equation zeros, frozen local kernels,
and 55 independent term comparisons over H2, LiH, water, PH3 and the LiH
cation. The largest molecular term difference is 1.950e-9 Hartree against
the unchanged 5e-7 bound. Registry9, bounded Taylor1 and the H2 public
energy case pass. All 24 C1 RKS/UKS nuclear fits pass on noncoplanar ammonia and its
cation, grid5 and cutoff zero, with steps [0.04,0.02,0.01,0.005] Bohr:
p=1.998258--2.001281, minimum R-squared 0.99999980. All components
resolve on the first interval without floors or changed gates. The
complete gradient test exits successfully in 1508.83 s. Local fits have
p=1.979985--2.011877, minimum R-squared 0.99997202; no failed local or
nuclear interval is omitted.

The native/common/ASE/CLI H4 route assertions complete for both RKS and
UKS (maximum gradient/energy differences 9.437e-16 Hartree/Bohr and
8.882e-16 Hartree). Following the timeout-ended two-case session, the
separate exact-node runs now both terminate successfully: RKS in 332.69 s
and UKS in 323.24 s. Both earlier 360 s
timeout trials, including the initial ammonia route trial, remain recorded
as not accepted. The compact H4 route geometry is a post-freeze test-only
change; the independent full ammonia nuclear test and production source
remain byte-identical to the built snapshot.

Evidence is `.work/validation/dsd-pbeb95-20261031-final-results.json`,
with exact commands, artifact identities, partial-fit metrics and every
terminal/timeout record. Ferrocene, molecular
double-hybrid Hessians, ECP/ROKS/GKS and excited-state combinations remain
unverified by this tranche.

### TPSSloc approved reference application (2026-10-31)

After the independent investigation above, the user explicitly approved
the original four-entry proposal while retaining old values, failed logs
and all acceptance thresholds. Exactly point10 exchange and combined
kernel diagonal indices 2 and 4 were updated. The raw fixture SHA256
`57fc992c7b937168309f1712b2a26b1cb5a59f570fde3e3a60fbd8127a084dac`
is archived byte for byte outside docs; the corrected fixture SHA256 is
`226f2ac9d9dcb8a514dc9404342a1f8a1759418c0a44d4ae60011f193806c390`.
The generator reproduces this correction only for the exact approved
input and original raw reference values; a changed raw result requires review.
Production TPSS equations, test source and thresholds are unchanged.
The approved values differ from the independent high-precision inward
limits by at most 1.153e-13. Earlier pending/failed records remain historical
evidence. Actual unchanged-test regressions are in
`.work/validation/tpssloc-approved-reference-20261031/regressions.json`;
all six unchanged p301 tests pass (440.23 s), including the 65 derivative fits,
and both public energy/gradient cases pass. The generator reproduces the
corrected fixture exactly. The catalogue now records bounded pointwise/SCF
acceptance; five standard molecules and higher responses remain separate.
This approval does not cover the separate six EDF2 raw boundary failures.

### Remaining implementation priorities after p304

Subsequent p305 implements the two original range-separated B2 recipes
with explicitly partial acceptance below; 59 entries remain unimplemented.

The p304 checkpoint had 61 unimplemented entries; this count includes
methods and nonlocal corrections as well as local functional bodies.
Next candidates for the existing point/Taylor/composer architecture are
TM/revTM and the remaining well-specified hybrid recipes, after their
exact spin conventions, published parameters and independent numerical
identifiers are verified. B3B95, mPW3PBE and B98 retain the explicit
definition blockers already listed in their rows; no alias-based formula
is inferred. PTPSS additionally needs its refitted components rather than
ordinary TPSS. Nonlocal vdW kernels, density-dependent local hybrids,
self-interaction corrections and RPA variants need dedicated providers
and independent response contracts; a weighted semilocal registration
would not implement them. These are priorities and blockers, not claims
that any of those providers has been implemented or validated.


### Original omegaB2PLYP and omegaB2GP-PLYP (p305, 2026-10-31)

The typed `wb2plyp` and `wb2gp-plyp` recipes follow Casanova-Paez,
Dardis and Goerigk, [JCTC 15, 4735 (2019)](https://doi.org/10.1021/acs.jctc.9b00013).
The authors' [2020 follow-up](https://arxiv.org/abs/2006.07821) confirms
unchanged global-parent components and the short-range B88 convention;
the [official parameter table](https://orca-manual.mpi-muelheim.mpg.de/contents/modelchemistries/DensityFunctionalTheory.html)
pins both full parameter sets. Their original, unscaled ground-state
definitions are:

| Recipe | SR exact exchange | LR exact exchange | omega / Bohr^-1 | SR B88 | LYP | SS PT2 | OS PT2 |
|---|---|---|---|---|---|---|---|
| `wb2plyp` | 0.53 | 1 | 0.30 | 0.47 | 0.73 | 0.27 | 0.27 |
| `wb2gp-plyp` | 0.65 | 1 | 0.27 | 0.35 | 0.64 | 0.36 | 0.36 |

SR B88 uses the existing ITYH attenuation. PT2 uses the full Coulomb
interaction on the stationary range-separated parent orbitals. No D3,
VV10, separate xDH target or additional spin scaling is implicit. These
are not the later SCS/SOS variants. The implementation only adds typed
recipes/aliases; shared density kernels and response equations are intact.

The immutable `wb2plyp-integration-20261031` bundle is based on accepted19
plus these owned recipes and the separately approved TPSSloc four-entry
fixture/generator. Warning-strict Python/library/CLI compilation and the
portable audit pass. Library SHA256 is
91d913f27140d74fa1f9d1f6dbef11f48259323cd6bd19027cea32fc8e3edd41.
The later active static spinor helper is outside this source manifest.

Actual p305 local contracts pass all 66 active first/second/third partial
fits and 44 defining-equation zeros across both parents. Spin exchange,
vacuum, unit long-range exchange and the zero-omega global-parent limit
pass. All 110 independent molecular term comparisons pass over five
cases per recipe: H2, LiH, water, PH3 and the LiH cation. The oracle
explicitly reconstructs full- and long-range exchange separately; an
initial helper call incorrectly retained full-range K under a context
manager, failed reconstruction and was corrected with explicit omega.
The failed diagnostic and unchanged numerical bounds remain recorded.

Acceptance is partial: eight raw zero-gradient sigma diagonal kernels
(exchange/parent, aa/aa and bb/bb, both recipes at point10) fail the frozen
comparison. All local energy densities, potentials and the other frozen
kernel entries pass. Raw references and gates remain unchanged; the
TPSSloc approval does not cover these entries. All 48 C1 H4 nuclear
gradient components (both recipes, RKS and UKS) pass at grid5/cutoff zero,
using [0.04,0.02,0.01,0.005] Bohr on the first interval:
p=1.958894--2.029668, minimum R-squared 0.99987741. The full nuclear
test terminates in 1304.21 s, without failed fit intervals or changed
gates. Public energy routes pass both recipes, and all four public gradient
cases terminate successfully, with native/common/ASE/CLI agreement.
Their measured route defects and retained initial failures are recorded below. Smooth-local
fifth tensors, molecular Hessians, standard ferrocene, ECP/ROKS/GKS,
periodic and excited-state combinations are not validated by this tranche.

The eight boundary comparisons have a separate read-only investigation,
not a reference correction. For one spin channel, the B88 zero-sigma
derivatives are `E1=-beta*rho^(-4/3)` and `E2=12*beta^2*rho^(-4)`.
Writing the screened energy as `G(E)=E*R(a)` with `a` proportional to
`sqrt(-E)` gives `G1=R+a*R1/2` and
`G2=a*(3*R1+a*R2)/(4*E)`. Thus the screened second sigma derivative
is `G1*E2+G2*E1^2`. This chain calculation and independent nine-point
positive-side energy stencils agree on the following exchange limits:

| omega | rho_spin | inward sigma diagonal second partial |
|---|---|---|
| 0.30 | 0.31 | 0.01516765225488927736 |
| 0.30 | 0.22 | 0.05666103617086564352 |
| 0.27 | 0.31 | 0.01584735898331864858 |
| 0.27 | 0.22 | 0.05955932544306251695 |

The four-step positive stencils use sigma steps
`[0.00025,0.000125,0.0000625,0.00003125]`; observed order is
6.833867--6.931799 with minimum R-squared 0.99995166. Independent
80/120-digit working-precision limits differ by at most 5.744e-44.
LYP is affine in the sigma invariants, so the two parent limits are the
published exchange weights times these values. They support the finite
production results, but do not reproduce the native generated operation
graph; cancellation there remains a mechanism to investigate rather than
an inferred permission to change data. The original eight failed raw
entries, fixture hash and unapplied review record remain in
`.work/validation/wb2plyp-boundary-diagnostic-20261031/`. No reference
value or acceptance threshold was modified by this investigation.

The next private RSX definition check is not an implementation milestone.
The originals are [RSX-QIDH](https://doi.org/10.1021/acs.jctc.8b00261)
and [RSX-0DH](https://doi.org/10.1063/1.5097164); the exact QIDH exchange
coefficient must retain `3^(-1/3)`, rather than a rounded table percentage.
Its first one-channel test `(rho_a,rho_b,sigma_aa)=(0.4,0,0.09)` stops at
an energy discrepancy above the existing 1e-11 gate. Component isolation
finds SR-PBE exchange agreeing to 5.274e-16, while raw PBE correlation
differs by 1.971e-11. An independent 80-digit fully polarized PBE/PW92-MOD
closed-form calculation supports the production correlation value to
2.448e-18. The exact native endpoint mechanism is not traced here, and
no reference value, gate, named registration or implementation count
changes on that basis. Evidence is
`.work/validation/rsx-private-points-20261031/endpoint-diagnostic.json`.


### omegaB2 terminal verification checkpoint (2026-10-31)

All four native/common/ASE/CLI C1 H4 gradient cases pass on the immutable
91d913f2 library: wb2plyp RKS/UKS in 385.56/474.80 s and wb2gp-plyp
RKS/UKS in 384.24/487.22 s. These are execution durations, not controlled
performance measurements. Maximum energy/gradient route differences are
4.440892098501e-15 Hartree and 7.259817746963e-16 Hartree/Bohr. Both separate H2 public
energy cases also pass, including the defining absence of dispersion.
The initial added energy test incorrectly inherited a D3 assumption;
that failed trial remains logged, and only the expectation for the two
original dispersion-free recipes was corrected. No energy gate changed.

All 48 independent nuclear fits, 66 local first/second/third partial fits,
44 structural zeros and 110 independent molecular term comparisons pass.
The nuclear test takes 1304.21 s and every fit resolves on its first
prescribed interval. A post-freeze Clippy clarity change makes the same
nonfinite/strict-kernel failure condition explicit and corrects a diagnostic
recipe label; the nuclear test and scientific gates are unchanged. The
current point test reproduces exactly the same eight failed raw entries.
Focused production-recipe/test Clippy and Rust formatting pass with the
repository's existing lint configuration. Full aggregate status, exact
commands, source/artifact hashes and retained unsuccessful attempts are
recorded in `.work/validation/wb2plyp-20261031-final-results.json`.

The catalogue therefore records 59 unimplemented entries and two partial
p305 entries. The eight frozen boundary comparisons remain failed; their
independent physical-limit diagnosis is not permission to replace them.
EDF2's six raw boundary failures are unchanged. Standard ferrocene, local
fifth screened-B88 tensors, molecular Hessians and ECP/ROKS/GKS/periodic/
excited-state combinations remain separate from this checkpoint.

### EDF2 and omega-B2 zero-gradient reference root cause (2026-10-31)

The fourteen remaining point10 discrepancies have a common reference-side
cause, traced through the actual Libxc 7.0.0 generated arithmetic. This is
a diagnosis and an unapplied correction proposal; it does not change the
partial catalogue status or promote the original failed comparisons.
The point is `(rho_a,rho_b,sigma_aa,sigma_ab,sigma_bb,tau_a,tau_b) =
(0.31,0.22,0,0,0,0.37,0.21)`. Kernels differentiate energy per volume.
Native packed `v2sigma2` positions 0 and 5 map to full kernel `(2,2)` and
`(4,4)`. The public native API reproduces all fourteen frozen values exactly.

The [B88 defining expression](https://gitlab.com/libxc/libxc/-/raw/7.0.0/maple/gga_exc/gga_x_b88.mpl)
contains `x*asinh(x)`, where `x=sqrt(sigma)/rho^(4/3)` for a spin channel.
Its generated C evaluates the asinh value as `log(x+sqrt(1+x*x))`.
At the reference's effective sigma floors (about `1e-40` for B88 and
`4.64159e-38` for ITYH), the addition rounds to one and the logarithm
becomes zero. Differentiating the resulting unsimplified expression loses
a finite curvature contribution. The raw B88 sigma diagonal second
partials are exactly 5/8 of the defining-equation limits, to double
precision. The EDF2 coefficients propagate that defect unchanged.

An isolated replay extracted only the two generated polarized FXC
functions, used the installed native parameter objects and thresholds,
and changed only the two log identities per function to `asinh(x)`.
The original replay reproduces the installed B88 values bit for bit;
ITYH agrees within several ulps. The stable replay restores every
primitive physical limit below within `8e-17`. The installed library,
production source and frozen fixtures were not modified. The initial
private replay compile missing an assert header is retained in the
diagnostic history; the corrected replay compiles with warnings as errors.

For fixed positive spin density, write `u=sigma/rho^(8/3)` and
`E_B88=-Cx*rho^(4/3)-beta*rho^(4/3)*u/(1+6*beta*sqrt(u)*asinh(sqrt(u)))`.
Then `E_B88'(0+)=-beta/rho^(4/3)` and
`E_B88''(0+)=12*beta^2/rho^4`. For
[ITYH screening](https://gitlab.com/libxc/libxc/-/raw/7.0.0/maple/gga_exc/gga_x_ityh.mpl),
`G(E)=E*R(a)` with `a` proportional to `sqrt(-E)`;
`G'=R+a*R'/2` and `G''=a*(3*R'+a*R'')/(4*E)`.
The screened curvature is `G'*E_B88''+G''*(E_B88')^2`.
Using the erroneous 5/8 B88 curvature in this chain independently
predicts all four raw screened values within `3e-16`.
The [attenuation definition](https://gitlab.com/libxc/libxc/-/raw/7.0.0/maple/attenuation.mpl)
is the Tawada/Toulouse expression; its large-a smoothing branch is
inactive at these points. LYP is affine in sigma and contributes zero
sigma second partials, including both EDF2 LYP parameterizations.

The proposed defining-equation values are listed for review, not applied:

| Component/recipe | `(2,2)` limit | `(4,4)` limit |
| --- | ---: | ---: |
| B88 beta 0.0042 | 0.022920973101856916 | 0.09036268014479885 |
| B88 beta 0.0035 | 0.01591734243184508 | 0.06275186121166587 |
| EDF2 parent | 0.013395844382531637 | 0.05281121337340346 |
| SR-B88 omega 0.30 | 0.015167652254889277 | 0.056661036170865646 |
| omega-B2PLYP parent | 0.00712879655979796 | 0.026630687000306852 |
| SR-B88 omega 0.27 | 0.015847358983318648 | 0.059559325443062514 |
| omega-B2GP-PLYP parent | 0.005546575644161527 | 0.020845763905071882 |

Independent 120-digit energy stencils approach these limits from positive
sigma, with four-step fits and no error floors. The physical inward limit
is unique at these fixed positive densities; no negative-sigma extension
or joint zero-density limit is asserted. Existing p303 and new p305
boundary regressions keep the raw reference checks separate. The p305
eight boundary/nearby-potential fits pass on the actual 91d913f2 library
with p=1.960058..1.983630 and minimum R2=0.99996589. The independent p303
closed B88 limits are additionally checked against finite-positive-sigma
production potentials and kernels.

Evidence, exact old/new fourteen JSON pointers, 120-digit values,
reference source hashes, native threshold/FHC sweeps and isolated replay
commands are under
`.work/validation/gga-zero-boundary-root-cause-20261031/`, especially
`unapplied-fourteen-entry-proposal.json`, `native-packed-thresholds.json`
and `private-replay.json`. The reference hashes remain
`c5bb8aa65353ea72c20a7af56518eb721634abac28e87843ca96d2d213ee8e39`
(EDF2) and `e7d57330b2db7c9a91a3d87faf36c939d2f39028f011d2a8281514b56f2d8aa6`
(omega-B2). TPSSloc's prior four-entry approval does not authorize these
fourteen replacements. Production changes are not indicated by this
diagnosis; reference correction and subsequent full frozen-point
acceptance remain pending scoped review.

### Approved fourteen-entry boundary correction closure (2026-10-31)

The six EDF2 and eight omega-B2 corrections received explicit scoped approval.
The approval and application records remain in the excluded validation evidence.
Exactly the saved fourteen JSON values were applied; no production formula,
derivative tolerance or other reference value changed. Original bytewise
fixtures, failed comparisons, 120-digit derivations, proposal hash and
machine-readable approval/application audits are retained under
`.work/validation/gga-approved-fourteen-reference-20261031/`.
The earlier unapplied proposal and failed snapshots remain historical evidence.

Both complete frozen point comparisons now pass on the unchanged actual
91d913f2 library. Focused p303 seven-test and p305 five-test regressions
pass, including local derivative fits, boundary fits, physical limits and
independent restricted/unrestricted energy terms. The generators reproduce
all 77 EDF2 and 66 omega-B2 point/component rows exactly. Their narrow
boundary helper rejects unexplained raw values and leaves other points
untouched. Python lint and API/ledger checks pass.

The existing independently accepted EDF2 12 C1 gradient plus78 Hessian fits,
omega-B2 48 C1 gradient fits and four native/common/ASE/CLI gradient cases
remain applicable: neither executable physics nor these molecular inputs
changed. These expensive molecular jobs were not repeated for a reference-
only edit. Public-route energy/gradient defects remain 4.440892098501e-15
Hartree and 7.259817746963e-16 Hartree/Bohr respectively. Registry9 and the
bounded Taylor target1 previously passed on the same artifact.

Catalogue status is now pointwise/SCF validated with standard five-molecule
acceptance pending for all three entries, not universal derivative/property
acceptance. There are still59 unimplemented entries, zero p303/p305 partial
entries and25 pointwise/SCF entries with five-molecule acceptance pending.
Standard ferrocene, screened-B88 fifth tensors, general molecular double-
hybrid Hessians and ECP/ROKS/GKS/periodic/excited combinations are separate.
The exact command/results record is `final-results.json` in the approval
directory. The diagnosis above remains the explanation of the old failures.

### EDF2 standard-five completion and parallel RSX staging (2026-10-31)

All five original standard molecular energies now pass on actual91d913f2:
H2, LiH, water, PH3 and ferrocene, spherical STO-3G, grid9, zero grid-weight
cutoff and the existing e_tol1e-12/p_tol1e-10 SCF criteria. Fe converged in
40 iterations (902.509s reported SCF time), E=-1632.123425614187 Hartree
against independent -1632.1234256142045, defect1.7507773e-11 below the
unchanged1e-5 standard Fe gate; the other four retain5e-7 gates and have
maximum defect4.681e-12. Initial small-case driver-default density-tolerance
attempts were archived and all four rerun with the existing stricter p_tol.
Exact controls/results and reference snapshot are in
`.work/validation/functional-standard-five-20261031/edf2-five-final.json`.
The other added providers' missing Fe cases remain actively under acceptance;
they are not promoted by EDF2's success.

The user's explicit instruction to implement further providers concurrently
is addressed by p306 RSX-QIDH/RSX-0DH staging. The defining recipes use the
existing ITYH SR-PBE body, PBE correlation, full LR HF and full-Coulomb SS/OS
PT2; there are no new scalar formulas or dispersion terms. Both exact-source
recipe point suites and all110 active first/second/third partial fits plus
spin/range/vacuum limits pass, before whole-crate integration. A separate
private threshold experiment explains the earlier polarized PBE-C reference
offset: explicitly lowering the native density threshold to1e-20 and1e-30
gives identical energies and approaches the independently derived one-spin
formula; the original failed private data remain unchanged. No existing
reference was changed for RSX. The reproducible generator records this point
threshold explicitly. Ground-state/new-artifact/standard-five/public and
higher-order/excited acceptance remain pending for the two staged recipes.
The next compiler window follows root's serialized scalar-response24 build;
existing molecule jobs continue independently. Stopped spinor helper work
remains owned by the paused response worker and is not included.


### Standard-five continuation and N12-SX staging (2026-10-31)

EDF2 and TPSSloc now both pass all five standard molecular energies with
the original gates. Across BLYP35/WC04/WP04/PKZB/TPSSloc/TASK/mTASK/EDF2/
DSD-PBEB95/omega-B2PLYP/omega-B2GP-PLYP, 49 of 55 cases are accepted.
TASK now passes all five, including ferrocene at energy defect6.59e-12 Hartree.
The remaining six are ferrocene, with CPU-progress calculations and a
separate WC04 solver diagnosis retained. This is molecular energy coverage,
not a claim about all derivatives, spin references or ECP combinations.

The RSX actual-library core now passes 110 local first-through-third fits,
110 molecular term comparisons and two native/ASE/CLI energy cases. Its
Fe remains unrun; all48 independent nuclear gradients now pass, retaining
one failed exploratory interval resolved at p2.087530/R-squared0.99966681
with unchanged gates. Four public-gradient cases are executing in the
existing bounded queue. New provider
work now prioritizes hybrid, GGA and meta-GGA, following the user's choice.

N12-SX is staged under p308, following [the original paper](https://doi.org/10.1039/C2CP42576A),
equations 1-9 and Table 2. Density/gradient coefficient indices are explicitly
resolved; the fitted local polynomial is included whole alongside 25% short-range
HF, zero long-range HF and omega0.11. Its B97-form correlation uses the
original rounded PW92 spin normalization1.709921. The inherited precise
normalization produced a maximum local energy discrepancy7.37e-11; an
independent 90-digit evaluation predicts that discrepancy at every point,
with unchanged frozen values and gates. Older B97 variants retain their
previous behavior. Complete new-library acceptance is pending, distinct
from the 86 literal-coefficient local derivative fits already passed.

The actual2a5b3ad9 library subsequently passes all86 local first-through-third
fits, both independent RKS/UKS term cases, all90 C1 RKS nuclear gradient and
independent Hessian fits, and both spin public-gradient cases. Separate local
fourth/fifth tests pass45/68 nonzero partial fits and95/184 defining-equation
zeros; the fifth test retains four failed exploratory measurements before
resolved intervals at unchanged gates. The expanded public-Hessian parity
test subsequently passes for both RKS and UKS, together with both spin
gradient tests (four public cases,3988.79s). These route checks are distinct
from independent RKS Hessian fits and add no independent UKS Hessian fits.
Standard-five N12-SX energy acceptance is four-of-five, with independent
ferrocene reference calculation active.

WC04 diagnosis also independently resolves an unstable direction of the
retained higher-energy reference at grid4: orbital Rayleigh curvature
-0.0456585363 Hartree, verified against four central energy curvatures with
p=1.99822680 and R-squared0.999999880. This establishes a descending direction,
not a converged lower state or grid9 acceptance. The original reference,
default-solver timeout remains preserved. The separate no-SOSCF calculation
now converges109 iterations at the original gates to -1617.516005986437 Hartree,
0.00114624321 Hartree below the frozen reference; its energy comparison
therefore remains failed. Independent lower-branch convergence is separate; no
reference replacement is made. See `wc04-negative-mode-energy-20261031/results.json`.

WC04 fixed-density diagnosis found an AO *representation* error in the
reference helper: angular component reordering had omitted source-shell
ordering for interleaved S/P shells. The corrected permutation independently
reduces the Fe overlap error from0.3309315588 to5.55e-16. With the same
density and identical473888-point grid, XC energy agrees to2.72e-12 Hartree,
Fock entries to6.73e-13 Hartree, and grid electrons to2.84e-14.
This invalidates the earlier mismapped matrix diagnostic, not the existing
scalar molecular reference. WC04 default SCF is still nonstationary and is
not accepted. An existing CDIIS solver alternative is tested separately at
identical energy/density gates. Original runs and references are retained.

N12-SX integration subsequently completed with warnings treated as errors and
portable artifact audit passed (rlib2a5b3ad9d177f11c3ef2cfdaf6194bb36caea3c8b61acca80be44e4412d49d1c).
Actual-library component kernels, all86 first-through-third local fits and
two RKS/UKS fixed-density term comparisons pass. The original failed inherited
correlation comparison remains retained. Full C1 molecular derivatives and
two public gradient cases are running; all five standard energy cases are
also executing in a separate bounded queue using the shared heavy-native cap.
The original55-case inventory remains48 accepted until its individual new
results complete. RSX's previously unexecuted48 gradient fits and four public
gradient cases are queued after current N12-SX acceptance, rather than left
as a documentation-only gap.

A separate original-author molecular convention check remains unresolved at
printed precision: [the author manual](https://comp.chem.umn.edu/mfm/221014_MFM_Manual_v7.0.pdf)
Table4 quotes water/6-31+G** with a pruned99/590 grid. Independent spherical5D
evaluation differs by1.886485e-3 Hartree; Cartesian6D reduces that to1.973025e-6
Hartree and maximum force difference7.689924e-6 Hartree/Bohr. The manual does
not fully specify pruning/radial and Cartesian conventions. Neither comparison
is claimed as a strict pass or used to change any fixture or tolerance.
Both raw outcomes are retained under `n12-sx-author-water-20261031` and
`n12-sx-author-water-cartesian-20261031`. The independently generated STO-3G
component/SCF fixtures have their own explicit conventions and unchanged gates.

### dlDF staged hybrid meta-GGA (2026-10-31)

The next non-double-hybrid provider is dlDF of Pernal, Podeszwa, Patkowski
and Szalewicz, [PRL103,263201 (2009)](https://doi.org/10.1103/PhysRevLett.103.263201).
Its canonical rounded XC definition has HF fraction0.6144129, PBE-form
exchange kappa4.8827323/mu0.3511128 multiplied by the fitted fifth-term
kinetic polynomial and its already included local fraction0.3855871.
The PW92-MOD same/opposite-spin correlation uses gamma0.06/0.0031 and the
corrected Fermi factor `(1-x^2/(8*t))*(1-exp(-4*t^2/eta^2))`, eta1e-10.
No intermolecular dispersion model or PT2 correction is silently included.
The primary implementation definitions and exact rounded coefficients are
independently traceable to the numerical specification for identifiers36/37.

The generic scalar body, constructors and narrow name registration are staged
in `xc::dldf`, with independent point/molecular reference generation and
`tests/dldf_contract.rs`. Exact-source preflight passes all24 component point
comparisons, including low-tau, zero-gradient and empty-spin cases, all40
independent first/second partial fits, all52 independent third partials
against independent kernel differences, and an independent80-digit zero-tau
continuation check. The latter retains seeded derivatives using an exponential
quotient series and fits the one-sided second derivative with p1.99572694,
R-squared0.999998770; it describes the raw mathematical continuation, not the
shared driver's finite tau-floor convention.

Expanded low-tau testing first failed because the raw body had not applied
the common input convention. A private native allocation reproduces both
failed raw energies with Fermi-hole enforcement disabled and both original
reference energies with it enabled. The new constructors now request the
existing shared value-only input bound, retaining independent derivative
seeds. The unchanged references and gates then pass all24 point comparisons;
the initial failed comparisons remain archived. This corrects production
configuration, not reference values. Evidence is under
`dldf-math-preflight-20261031/{results.json,unclamped-low-tau-failures.json,fhc-diagnosis.json,derivative-fits.json,tau-limit.json}`.
Third-partial evidence is in `third-fits.json` and `third-resolved-fits.json`:
one coarse interval failed and remains preserved; four smaller steps resolve
it at p1.94963786, R-squared0.999812325 with unchanged gates.
The standalone permanent test compiles with warnings denied. The frozen
whole-crate metadata-only typecheck also passes with warnings denied in131s;
it produces no executable library. Actual executable crate integration,
full higher-order provider acceptance, molecular derivatives,
public routes and standard-five acceptance remain pending. A frozen shared
build must follow the already reserved compiler windows.


### Bounded four-provider integration checkpoint (2026-10-31)

The current batch is fixed to dlDF, B98, tau-HCTH (`thcth`) and hybrid
tau-HCTH (`thcthhyb`). After this batch and existing outstanding validation,
new-functional additions are suspended at the user's request. No B3B95,
mPW3PBE, KCIS or TM/revTM implementation is started. BMK remains an
unregistered private prototype: its correlation absolute comparison at one
low-density point fails (1.1641532182693481e-9 versus2e-10). Its original
failed evidence is retained; it is not counted as a completed provider.

B98 reuses the B97-form body with the original PW92 convention, HF0.219847,
and Schmider/Becke fit2c coefficients (canonical425). This is distinct from
Becke's 1998 meta-GGA (598). The tau-HCTH pair shares one new scalar kinetic
enhancement kernel and the accepted B97-form correlation body; it is not
merely a GGA alias. Its fitted local coefficients already contain the
semilocal fractions; hybrid tau-HCTH declares HF0.15 separately.
Definitions are traced to DOI10.1063/1.476438 and
DOI10.1063/1.1476309, with independent parameterized component fixtures.

The immutable four-provider Python/rlib/CLI build passes with warnings denied
and portable artifact audit; rlib84bd1e14c427a9de819661e654b3d112fb4ef05c2ed0689e169b1bbf4c8619c8.
The compiler reservation is released. Actual dlDF point, all146 first-third
local derivative fits, and both RKS/UKS fixed-density term checks pass.
Molecular derivative, other component, public-route and all20 standard-five
cases are in bounded queues; none is promoted solely from build success.

All four providers pass the complete smooth-interior fifth-order contract:
380 nonzero fits,1846 defining-equation zeros, and weighted dense-third
composition parity. The initial dlDF exact-zero assertion formed A-A by
polarization and left -6.94e-18. Independent tau/opposite-spin mixed seeds
test the absent defining dependency directly and give exact zero. Production
and gates are unchanged, and both the original failure and a separate
missing-support harness error remain retained. Failed exploratory fit
intervals remain in the final log; the staged helper uses unchanged gates.
Evidence: `non-dh-batch-integration-20261031/build.json`,
`non-dh-batch-acceptance-20261031/results.json`,
`non-dh-batch-fifth-direct-zero-02-20261031/final-results.json`, and
`non-dh-standard-five-20261031/results.json`, all under `.work/validation`.

The original eleven-provider standard-five matrix is now50/55 accepted:
mTASK joins TASK with5/5, its ferrocene defect3.1604940886609256e-11 Hartree
at the unchanged1e-5 gate. WC04, PKZB, DSD-PBEB95 and both omega-B2
ferrocene cases remain unresolved. WC04's alternative native CDIIS solution
converges but differs from the old reference by0.00114624 Hartree;
independent lower-branch references are still not converged. No WC04
reference is replaced. N12-SX remains4/5 while its ferrocene reference runs;
all four RKS/UKS public gradient/Hessian route cases pass. RSX's all48
independent nuclear gradient fits and all four public gradient routes pass;
no independent Hessian or ferrocene acceptance is inferred.


The bounded four-provider core/API milestone subsequently passes on84bd1e14:
all432 first-third component/composition fits,270 complete representative
C1 RKS gradient/Hessian fits,8 RKS/UKS fixed-density term cases and8 public
gradient cases (native/common/ASE/CLI). The all-name Taylor contract and both
registry gates pass. The frozen batch initially omitted two existing N12-SX
CSV census fixtures; a separate fixture root restores the unchanged files,
with hashes recorded, without editing immutable source/artifacts or exempting
any name. The initial GGA absent-tau test access was also repaired; both raw
failed checks remain in the original `results.json`, whose all_pass=false is
not overwritten. Corrected evidence is in `bounded-contract-final-results.json`.
Production-library and three contract-target Clippy checks pass with warnings
denied. Direct-driver Cargo version/MSRV environment omissions were diagnosed
separately; the final library run uses the declared Rust1.82 compatibility
environment, with no lint suppression or unowned source change.

Each of the four new providers passes H2, LiH, water and phosphine:16/20
standard cases accepted at this checkpoint. Their four Fe cases remain in
finite reference/native queues and are not promoted. New-functional source
development is suspended and the compiler reservation is released. Original
five-provider Fe blockers and N12-SX/RSX limits above remain distinct.
B98 Fe subsequently converges in the original batch native artifact in 48
iterations, at -1632.072075033267 Hartree. Its independent reference phase
timed out, so the case remains incomplete and is not counted as a native SCF
failure or an accepted standard-five comparison.

### Molecular SCF recovery candidate (2026-10-31)

The user separately requests improving SCF convergence; the suspension of
new-functional additions remains in force. The real RKS/UKS globalization
strategy is being improved without changing an energy, grid, reference or
physical stopping gate. The retained WC04 Fe default run reached its 7200 s
supervisor timeout with no terminal native result; its logged quasi-Newton
iterates are nonstationary. An existing ordinary CDIIS alternative converges
in 109 iterations at the original 1e-12 energy, 1e-10 density and 1e-9 residual
controls. Its lower stationary branch remains separate from the old energy
reference, which is not replaced.

The candidate monitors cumulative improvement of the true physical residual
across a DIIS-history-sized window. A 30-percent improvement resets that
window; a stalled occupied-subspace minimizer is retired and a fresh
physical-Fock/(A)DIIS sequence resumes. The existing 100-fold local handback
and the accepted quadratic/HF Newton recovery are retained. No functional
name selects the strategy, and no new condition can declare convergence.
Fixed-population, smeared and tracked-state exclusions are preserved.
The independent progress-control tests demonstrate cumulative geometric
convergence and invariance under residual rescaling from 1e-200 to 1e200. Both pass;
whole-candidate metadata/Clippy also passes with warnings denied in 61.84 s.
The shared immutable continuous-species integration now compiles with warnings
denied and passes the portable artifact audit. Its library SHA-256 is
`25eb5c946a67d50266af17924854d12c19721f352b2b499c5fa8fee864f39892`;
the classical-energy hook and generator/retraction interface are preserved.
Actual-artifact HF oracle tests (three) and twenty related restricted/unrestricted
SCF/term cases pass. The default Fe test now passes: recovery executes at
iteration 26 with residual 1.41048e-3, and the original energy/density/residual
gates are satisfied at iteration 112 (SCF: 2699.846 s). The final energy is
-1617.516005986445 Hartree. Independently rebuilding J/K and XC gives both
spin metric-commutator maxima of 1.382078373248e-11; fresh Fock equality, particle
number, idempotency and energy-reconstruction checks all pass. The complete
physical test takes 2772.36 s. This validates the recovered stationary solution,
not the older higher-energy molecular reference: its 1.14624e-3 Hartree
discrepancy remains retained, and no reference or standard-five status is
changed. The remaining regressions now pass as well: all 24 independent C1
gradient fits (p = 1.994249..2.002354, minimum R2 = 0.999997740) and both RKS/UKS
native/common/ASE/CLI gradient cases. All 16 compile/run checks in this
first-artifact suite pass; the full steps/errors and independently recomputed
unfloored fit statistics are preserved in
`.work/validation/scf-progress-actual-20261031/final-results.json`.
An independent 80-digit contraction of all stored density/orbital entries
also passes: each spin has particle-number defect 1.141e-13, projector defect
1.365e-14 and orbital-metric defect 1.855e-14. This is a density-invariant
audit, not an independent XC or molecular-energy reference; its raw results
are in the same directory's `independent-density-audit-80digit.json`.
The Fe acceptance test rebuilds J/K and XC from the final density and checks
fresh metric commutators, particle number, idempotency and energy components.
Evidence: `scf-progress-preflight-20261031/results.json`,
`scf-globalizer-recovery-20261031-01/{prepared.json,typecheck.json}` and
`scf-functional-recovery-build-state.json` and
`scf-progress-actual-20261031/results.json` under `.work/validation`.

A second isolated revision corrects the entry detector as well. Its previous
per-step comparison incorrectly treats six individually smaller-than-30-percent
reductions as stagnation even when the cumulative reduction is large: the
retained Fe iterations 11..17 reduce the residual from 4.845e-4 to 9.836e-5.
The entry detector now reuses the cumulative monitor with the unchanged
six-step window. All three exact-source policy tests and whole-candidate
metadata/Clippy pass (warnings denied, 72.67 s). This later entry revision is
absent from the completed `25eb5c94` artifact. It now has its own actual shared
library, `3c1f7c960c22a4718e4daccd926df180897c7b196c56813e6dd5b1f8e43cef29`,
with library/CLI compilation, portable audit and native import all passing.
The owned iteration/controller source matches the isolated revision exactly.
All sixteen compile/run checks pass on this artifact: three HF tests, twenty
other RKS/UKS SCF/term cases, all twenty-four independent C1 gradient fits,
two public RKS/UKS cases, and the default Fe physical acceptance test.
The premature switch at iteration 17
is absent; the first switch occurs at iteration 33 after an actual residual
plateau, followed by recovery to physical Fock/DIIS at iteration 42. Fe converges
at iteration 134 (SCF: 3560.761 s), with returned energy
-1617.5160059864234 Hartree and independently rebuilt spin commutators
2.699877518718e-11. All original energy/density/residual and fresh physical
assertions pass. The final reconstruction changes the last logged energy by
2.4e-11 Hartree; the input energy tolerance alone is not an absolute energy
accuracy guarantee. Gradient exponents span 1.994249--2.002354, minimum
R-squared 0.99999773957, with no failed exploratory gradient intervals.
The incremental two-file patch is preserved in
`.work/validation/scf-globalizer-recovery-20261031-02`; actual results and
final evidence is in `scf-cumulative-actual-20261031/final-results.json` and
`scf-cumulative-current-checkpoint-20261031.json` under `.work/validation`.
One shared continuous-species integration avoids a duplicate library build.
No result from the first artifact is attributed to this later revision.
The different iteration counts and shared-machine timings are not a speed
benchmark or a universal convergence guarantee.

Independent PySCF 2.14.0/Libxc 7.0.0 calculations on the same WC04/STO-3G
Fe geometry and level-9 grid confirm the recovered lower stationary branch.
Exact packed ERIs with incremental J/K updates disabled give converged CDIIS
energy -1617.5160059864486 Hartree and fresh orbital-gradient norm
4.038599119683e-11 (energy tolerance 1e-13, gradient tolerance 1e-10).
An independent Newton solve using native orbitals only as its initial guess
gives -1617.5160059864481 Hartree and gradient norm 8.196710364659e-11.
The user subsequently approved exactly one molecular-reference correction:
the WC04 ferrocene entry is now -1617.5160059864486 Hartree. The original
higher value -1617.5148597432265 Hartree, full original fixture and all failed
logs are archived in `wc04-molecular-reference-proposal-20261031`, with
the specific approval and exact before/after hashes. No other fixture entry
or tolerance changes. The affected point/SCF and complete five-molecule
regressions now pass on the actual final SCF artifact, including Fe energy
defect 1.159606e-11 Hartree at the unchanged 1e-5 gate. The full five-molecule
test takes 4744.01 s on one worker; this is acceptance timing, not a benchmark.
Neither reference calculation
proves a global minimum.
Evidence is in `wc04-recovered-density-reference-20261031-{04,05}/results.json`
under `.work/validation`. Earlier incremental-J/K drift and failed attempts
remain recorded separately.

In the continuing standard-five queue, PKZB independently completes Fe:
energy defect 3.660716174636036e-11 Hartree at unchanged 1e-5, with strict
SCF convergence. The original eleven-provider matrix subsequently reaches 52/55
after the approved WC04 correction and terminal five-molecule rerun. External
WC04/N12-SX reference-solver stagnation is diagnosed separately from the
native iteration strategy and does not justify changing a reference or gate.

### Frozen functional batch handoff (2026-10-31)

New-functional implementation is suspended. The four registered providers have
completed the bounded defining/local/SCF/public contract on actual library
`84bd1e14`; their complete standard-five acceptance is still incomplete.

| Provider | Component points | First-third fits | Fifth fits / defining zeros | Independent C1 RKS gradient/Hessian fits | Public gradient cases | Standard molecules |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dlDF | 24 | 146 | 128 / 796 | 90 | 2 | 4/5; Fe reference and native phases timed out |
| B98 | 11 | 34 | 56 / 70 | not separately fitted | 2 | 5/5; strict exact-ERI reference recovery and native comparison pass |
| tau-HCTH | 22 | 126 | 98 / 490 | 90 | 2 | 5/5; strict exact-ERI reference recovery and native comparison pass |
| hybrid tau-HCTH | 22 | 126 | 98 / 490 | 90 | 2 | 4/5; Fe follows tau-HCTH in the existing queue |

All eight RKS/UKS fixed-density term cases and the registry/Taylor contract
pass. Initial test/registry-census failures and thirteen exploratory fifth
interval failures are retained with their explicit recoveries. This does not
establish new-provider UKS Hessians or ECP/ROKS/GKS/excited combinations.
The N12-SX implementation separately passes 86 local first-third fits,
45 fourth and 68 fifth fits, all 90 C1 RKS nuclear fits and all four public
RKS/UKS gradient/Hessian cases. Its molecular energies remain 4/5: the existing
Fe reference Newton solve has not met its unchanged gradient gate; the native
Fe phase follows that reference. BMK is an unregistered private prototype
with the previously recorded absolute-reference mismatch, not a delivered
provider. No additional candidate or compiler reservation is held.

Machine-readable per-model handoff and actual process inventory are stored
in `.work/validation/frozen-functional-batch-handoff-20261031.json`.
The remaining molecular validation is distinct from completed implementation
and local acceptance; it is not declared finished because it is documented.
The later strict reference recoveries bring this four-provider matrix to
18/20. Evidence is in `non-dh-standard-five-recovered-20261031.json`; the
original collector and timeout records remain unchanged.

### Explicit BMK and ferrocene continuation (2026-10-31)

The user subsequently requests completing BMK and the remaining ferrocene
acceptance. These are named exceptions to the pause; no other new functional
is being added. BMK exchange/correlation constructors now reuse the accepted
kinetic-exchange and original-PW92/Stoll bodies, with the published coefficients.
Recipe/component aliases are registered. The new library, Python extension and
CLI compile with warnings denied and pass the portable-artifact audit. This is
not yet complete molecular/public acceptance.
Published JCP 121, 3405 (2004), Table II gives kinetic exchange coefficient
4.73936; the initial arXiv preprint instead prints 0.47394. The published
coefficient and separate full-HF fraction 0.42 are used. Paper tau is twice
the engine's half-weighted tau, consistently with the shared exchange body.

At the retained low-density point [0.02,0.015,0.0001,0.00005,0.00006,0.004,0.003],
the failed correlation entry is the sigma-aa diagonal second partial of
energy per volume. Raw reference is 80718.64612830374; the native shared body
gives 80718.64612830257. Independent 80-/120-digit evaluation and a separate
100-digit closed sigma derivative give 80718.6461283025587184, agreeing to
at least sixty significant digits. A separate four-step potential limit has
p=2.00002697 and R-squared=0.999999999951. The native error is 1.4552e-11
against the unchanged absolute 2e-10 gate; the raw reference error is 1.1787e-9.
Coordinate round trips are exact at this point, both spin densities are
positive, and no density/FHC boundary is involved. The one-entry reference
proposal is saved unapplied; no earlier reference approval is extended to it.
The failed absolute 2e-10 comparison belongs to the additional private
preflight. The permanent component test instead uses its unchanged scaled
kernel gate, 1e-9*(1+abs(reference)), which this raw reference passes. The
relative discrepancy is about 1.46e-14; it is not evidence of a physically
significant product error. Raw component and composite values remain unchanged.
Independent high-precision evidence is a separate provenance-labelled oracle;
the extra absolute-preflight failure is retained, not recategorized as passed.

On the actual accepted shared-body artifact, the published BMK parameters pass
all 126 first-third component/composition fits and all 98 smooth-interior
fifth fits plus 490 defining zeros, with no failed exploratory intervals.
The weighted dense-third composition also passes. Two additional independently
generated molecular cases (water RKS and methyl-radical UKS) pass total energies
and every fixed-density local, one-electron, Coulomb and full-exchange term on
the same shared-body artifact. Their raw reference fixture is unchanged. This
validates the parameterized common bodies, not newly compiled constructors,
registration or named public molecular routes.
Evidence: `.work/validation/bmk-shared-body-acceptance-checkpoint-20261031.json`
and `bmk-reference-proposal-20261031/proposal.json`; the molecular checks are in
`bmk-shared-molecular-contract-20261031/results.json`.

The reference Newton stall has a separate reproduced setup cause: its initial
unnormalized gradient basis can have norm squared below the AH rank cutoff,
giving zero updates while the physical gradient gate remains unmet. An
independent positive-definite matrix reproduces this behavior. A normalized
trial basis and a sufficiently accurate inner solve pass all sixteen closed
AH step checks, with maximum component error 3.97e-16, keeping the original
rank and all physical stopping gates. Strict saved-density, exact-ERI
ferrocene recovery is in progress; earlier timeouts and unsuccessful methods
remain intact. No molecular acceptance is inferred solely from these algebra tests.

The actual BMK artifact `8fdc87c3e47127086457ff5084763a082a3d19e11f74fecbb2ca41a86f149a93`
passes the existing scaled raw-reference contract, the separate high-precision
kernel test, all 126 first-third fits and all 98 fifth partial fits plus 490
defining zeros. Independent refits of the latter give p=1.92813937..2.03982152
and minimum R-squared=0.9995743082, with no failed BMK exploratory interval.
Its two independent RKS/UKS energy and fixed-density term cases pass; the six
previous recipe cases, shared Taylor registry and three HF baselines pass too.
The standard H2/LiH/water/phosphine energy comparisons pass their unchanged
5e-7 Hartree gate, with maximum error 3.411e-13 Hartree. This is 4/5, not a
completed ferrocene or reference/property cross-product claim.
Evidence: `bmk-actual-acceptance-20261031/actual-local-milestone.json` and
`actual-milestone02.json` under `.work/validation`. An overbroad private pytest
`-k` selection and a subsequent relative-node path failure are retained as
harness failures; exact absolute BMK nodes are tested separately, with no
numerical gate or test-body change. Independent C1 RKS nuclear fits and the
remaining public/ferrocene acceptance continue on this immutable artifact.

### Completed BMK and two ferrocene recoveries (2026-11-01)

The completed immutable BMK artifact above passes all five standard molecular
energies, including ferrocene in68 ordinary SCF iterations. Its ferrocene
energy is -1631.0690666647397 Hartree versus the independently converged
-1631.0690666647313 Hartree (fresh orbital gradient2.496e-11); the error
8.413e-12 passes the unchanged1e-5 Hartree ferrocene gate. The four smaller
molecules pass the unchanged5e-7 gate. All90 independent C1 RKS gradient and
Hessian components pass four-step fits; independent raw-log refitting gives
p=1.99521826..2.00396422 and minimum R-squared0.9999985737, with no failed
interval. All four RKS/UKS gradient/Hessian common/native/ASE/CLI cases pass.
The historical primary runner retains its stopped overbroad pytest selection;
the exact-node recovery uses the same test body and unchanged gates. It
supersedes only that harness outcome, not any numerical comparison.

N12-SX and hybrid tau-HCTH also complete standard-five energy acceptance.
Fresh physical native ferrocene checks on artifact3c1f7c96 converge in58 and42
iterations, respectively; independent strict-reference energy errors are
4.980e-11 and5.025e-11 Hartree. No reference entry or tolerance changes.
Evidence outside the package is
`bmk-actual-acceptance-20261031/final-results.json`,
`bmk-public-selection-recovery-20261031-02/results.json`, and
`non-dh-ferrocene-paired-20261101.json` under `.work/validation`.

This closes those measured contracts, not arbitrary reference/functional/ECP
or excited-state combinations. Earlier progress sections remain historical.
The dlDF native10800-second and DSD native14400-second timeouts remain
unaccepted; their independent strict references have converged. The existing
serial omega/RSX native queue continues without duplicate launches. Further
functional additions remain paused apart from the completed named BMK task.

The recovered private DSD ferrocene reference runner had a separate setup
defect: semicolon-separated grid assignments were inside the non-DSD
conditional, so DSD alone used the reference engine's default grid3 while its
status incorrectly reported9. That private DSD result is invalidated, not
accepted as a grid9 oracle. The other four double-hybrid references executed
grid9. The original DSD generator's grid9 and1e-12/1e-9 energy/gradient gates
are restored in a new runner, with all old files retained and no distributed
reference or tolerance changes. Evidence is
`dsd-reference-grid-setup-diagnosis-20261101.json` outside the package.

The periodic execution owner has a runtime-resolved composition inventory
from artifactdd88d642:136 canonical combined XC cores and25 named double
hybrids, including component weights/parameters, dense derivative caps,
full/erf exchange fractions, and required nonlocal/PT2 ingredients. The raw
dump is `resolved-functional-inventory-20261101-02/resolved.json` outside the
package; `recent-functionals-periodic-capability-20261101.json` provides the
six recent recipes and inspected periodic paths. Registry resolution and
molecular acceptance do not establish periodic compatibility. Periodic
cutoffs, k-point meshes, tau/current conventions, exact exchange, nonlocal
terms and PT2 require their own measured convergence; no defaults are changed
by this inventory, and its dense caps do not imply higher Taylor acceptance.

### Remaining ferrocene convergence and reference stability

The corrected grid-9 DSD parent and unchanged post-SCF generator complete
their strict reference calculation with fresh orbital gradient 1.3672e-10,
below the original 1e-9 gate. Internal stability and the independent native
term comparison remain pending. The earlier grid-3 result remains invalid.

For dlDF, strict stationarity alone is insufficient: the old reference has
an internal orbital-Hessian eigenvalue of -0.0126412. A same-density native
calculation agrees with its energy and physical residual, so this is not
evidence of an XC-definition mismatch. An independent lower-branch recovery
was attempted with the original grid, energy and gradient gates. The
completed recovery outcomes below are failures; no unstable or
insufficiently converged branch is promoted to standard-five acceptance.

The SCF precision work separates two changes. Centered affine Fock assembly
has an actual immutable-library build, three HF regressions and eight
restricted/unrestricted recipe term checks passing. Eliminating the affine
error constraint before forming the DIIS Gram matrix is a later candidate.
It now passes six warning-strict source tests and actual library, CLI and
portable-artifact verification. Recorded molecular error histories
have independent difference directions despite near-zero eigenvalues of
the original bordered Gram matrix. A separate 100-digit least-squares audit
confirms those recorded directions; this diagnostic is not a convergence
or stability pass. Evidence is `molecular-diis-history-conditioning-20261101.json`
and `affine-diis-owned-delta-20261101-02/manifest.json` outside the package.

On its immutable artifact
`6723a67dc816132b7c3ce7e0c721162e80155ec6dc5791a021f085cec50f8d76`,
three HF tests, eight restricted/unrestricted recipe term cases and the
unchanged complete BMK C1 gradient/Hessian test pass. Independent refitting
of all 90 components gives p=1.99517721..2.00396459 and minimum
R-squared=0.9999986147, with no failed interval. This is a one-file SCF
algorithm regression, not completed ferrocene, periodic or arbitrary
functional/property acceptance. Evidence is
`affine-diis-baselines-20261101-02/final-results.json` outside the package.
The same four accepted RKS/UKS gradient/Hessian public tests also pass on
this actual artifact through common/native Python, ASE and the canonical
CLI, with warnings treated as errors. The unchanged test-body fingerprint
and exact node identities are recorded in
`affine-diis-public-20261101-02/results.json` outside the package.
Only future starts in the existing ferrocene queue select the new artifact;
running calculations retain their original immutable libraries. All raw
references and numerical stopping/comparison gates remain unchanged.

Elapsed-limit extensions preserve all physical stopping and comparison
gates. Earlier timeouts and failed methods remain recorded. BMK, N12-SX and
hybrid tau-HCTH retain their completed measured scopes above; six other
ferrocene comparisons remain unresolved. No further functional additions
are started.

The subsequent independent affine-SVD reference recovery is now terminal.
For dlDF, the strict stationary branch at -1636.424674767796 Hartree has
an internal Hessian eigenvalue -3.029898532212e-4 with measured eigenpair
residual 6.715150950692e-8 (original 1e-7 gate). Negative-mode descent
lowers the energy, but the following ordinary iteration returns near the
saddle and fails the unchanged 1e-13 energy convergence gate. Its fresh
gradient 2.322419675343e-11 alone cannot certify convergence or stability.
For DSD-PBEB95, the original strict branch is also internally unstable
(eigenvalue -1.890038089347e-3, measured residual 7.7214696e-8). A lower
branch reaches -1630.3541149309142 Hartree. Although the external solver
sets its convergence flag, a freshly rebuilt orbital gradient is
1.018660134766e-9, above the unchanged 1e-9 acceptance gate; it is rejected.
No post-SCF term or standard-five pass is inferred from that flag. Both
recovery workers are terminal and their CPU slot is returned. The next
reference recovery is prepared around the existing second-order orbital
minimizer and measured negative-mode descent, but has not run or passed.
Evidence is `stable-ferrocene-svd-reference-20261101-01` outside the
package. Every failed method and all raw reference values remain intact.


### Existing ferrocene terminal checkpoint (2026-11-01)

The original omegaB2PLYP ferrocene calculation converges in 118 iterations
without changing its SCF or reference gates. Its total energy is
-1630.6911183927762 Hartree, versus the independent reference
-1630.691118392629 Hartree (absolute difference 1.47111e-10 Hartree).
Every original total, parent, target, one-electron, Coulomb, exchange,
same-spin, opposite-spin and component comparison passes the unchanged
5e-7 Hartree tolerance; the largest term difference is 8.81119e-9 Hartree.
The original eleven-provider standard matrix therefore reaches 53/55,
and the bounded nineteen-model total reaches 90/95. This completes
omegaB2PLYP five-molecule acceptance only. The remaining queued cases
and the measured dlDF/DSD internal-instability blockers remain pending.
No reference values, thresholds or prior failed logs are replaced.


### Measured omegaB2GP-PLYP ferrocene comparison (2026-11-01)

On immutable artifact `6723a67d`, the grid-9 restricted parent converges in
122 iterations with the original energy and orbital stopping controls.
The post-SCF comparison nevertheless fails the unchanged 5e-7 Hartree
component gates. The total is -1630.5978934610835 Hartree versus the
independent -1630.6018216128434 Hartree, a difference of
0.003928151759964749 Hartree. The native parent is lower by
0.000363146715471885 Hartree, while its PT2 contribution is higher by
0.00429129846858145 Hartree; these differences close the total ledger
to 6.86e-12 Hartree. Dispersion agrees, but ten other term comparisons
fail. This is an actual failed acceptance case, not a timeout.

Different stationary branches are a hypothesis, not an established cause.
A matched-density operator/grid comparison and independent internal
stability check are required before choosing a product repair or proposing
any reference correction. The independent density is saved; this native
report did not retain its density, so no exact native warm restart is
claimed. No reference or tolerance is changed. Earlier named-recipe
results retain their measured artifact scopes, and the existing RSX queue
continues separately. Raw reports and the reconciled ledger are preserved
in `wb2gp-ferrocene-measured-failure-diagnosis-20261101.json` outside the
package.


### Measured RSX-QIDH ferrocene comparison (2026-11-02)

On the same immutable artifact `6723a67d`, the grid-9 restricted
parent converges in 112 iterations, but ten original component
comparisons fail the unchanged 5e-7 Hartree tolerance. The total
is -1630.632181795132 Hartree versus -1630.6354098634572 Hartree,
a difference of 0.0032280683251428854 Hartree. The parent is lower
by 0.00032820841556713276 Hartree. Dispersion agrees. This is a
measured failure, separate from the subsequently completed RSX-0DH case.
The latter exited successfully: native total -1630.871237168821 Hartree
versus independent -1630.8712371687998 Hartree, a 2.1146e-11 Hartree
defect. Its pass does not resolve either failed functional.

A shared cause with the earlier omegaB2GP-PLYP mismatch is not
established. Matched-density integral, range-exchange and grid
checks must precede any stationary-branch conclusion. The saved
independent density permits those checks; this native energy
report did not save its density. No reference value or tolerance
is changed. Raw reports and the causal-check ledger remain in
`rsx-qidh-ferrocene-measured-failure-diagnosis-20261102.json`
outside the package. No new functional or molecular acceptance
is inferred from the convergence flag.

### Ferrocene matched-density diagnosis (2026-11-02)

At the saved independent omegaB2GP-PLYP density, the source-level native
operator evaluation agrees with the independent calculation: one-electron
1.14e-11, Hartree 3.41e-12, semilocal XC 6.25e-13 and exact exchange
8.40e-12 Hartree maximum differences; the Fock matrix differs by at most
1.31e-12 Hartree. This uses the accepted core60 support library with the
original frozen density-field and one-electron source, rather than rerunning
the original failing executable. It performs no SCF or PT2 optimization.
That original 0.00392815 Hartree discrepancy is preserved historically.
The completed lower-branch comparisons below pass total and PT2 gates,
but remaining component failures still prevent reference correction. The original pipeline's missing-NumPy
postprocessing error is retained separately from its three successful
compile/reference/native subprocesses and the recovered artifact assessment.

The subsequent independent internal-stability calculation exited successfully.
Although the saved reference has an orbital gradient norm of 2.8942e-10,
its orbital Hessian has negative curvature -0.0242302637294592 Hartree,
with a fresh original-operator eigenvector residual of 5.1517e-13.
Four independent energy-curvature steps [0.08, 0.04, 0.02, 0.01] give
p=2.0013257034 and R2=0.9999986553; all eight signed trial energies are
below the original reference. A separate refit reproduces those fit values.
Thus this independent reference is an internally unstable stationary state,
rather than a certified minimum. Both signed mode-following restarts and
their unchanged PT2 evaluations have now completed; component closure
remains required before correcting a reference or declaring resolution.
An independent follow-up also finds a negative direction at the original
RSX-QIDH reference. A fresh-Fock audit at the reconstructed orbital state
gives an orbital gradient of 2.5583e-11 and directional Hessian curvature
-0.0226413040681822 Hartree. This direction is transferred between complete
orthonormal orbital bases and tested at RSX-QIDH; it is not claimed to be
an RSX-QIDH eigenvector. Reusing the original signed energy pairs at all four
steps with the matching fresh reference energy gives p=2.0057728547 and
R2=0.9999928969, reproduced by an independent refit. Both signed seeds lower
the original parent energy. The earlier 1.1163e-13 was a frozen-Fock
diagonalization residual, not fresh stationarity; its raw record is retained
separately. The original saved SCF itself reports fresh gradient 1.1091e-10.
Both signed augmented-Hessian restart stages and unchanged full-Coulomb
PT2 evaluations have actual native exit zero. All four fresh parent orbital
gradients are below the original 1e-9 gate. OmegaB2GP-PLYP passes 9/11
component comparisons: the new total is -1630.5978933631052 Hartree
(native defect 9.79783e-8), but one-electron and Hartree defects remain
3.08114e-6 and 3.43355e-6 Hartree. RSX-QIDH passes 8/11: its new total
is -1630.6321819833117 Hartree (defect 1.88180e-7), while one-electron,
Hartree and exchange defects remain 7.05545e-6, 7.85176e-6 and
6.94657e-7 Hartree. Both same-spin and opposite-spin PT2 comparisons pass
the unchanged 5e-7 gate. Re-evaluating the two new saved densities with
the source-local native operators agrees in every scalar to 3.01e-11
Hartree and in Fock entries to 7.39e-13; this isolates remaining native
state precision for further investigation, without certifying the original
binary's final density. Neither discrepancy is accepted as fully resolved.
Original failed comparisons and reference values remain intact.

The subsequent stricter native refinement is now terminal on actual support
library `22b155ea7113448d370fee0613061b199bd4d68027df00890b74cbe446658d14`.
It starts from the saved independently converged lower-branch densities and
reconverges the same all-electron STO-3G, grid-9, closed-shell Fe systems.
Numerical SCF residual and density tolerances are tightened to 1e-11; the
energy tolerance remains 1e-12 and every original 5e-7 Hartree component
comparison remains unchanged. OmegaB2GP-PLYP converges in 12 iterations and
passes all 11 comparisons, including separate same-spin/opposite-spin PT2;
its maximum component defect is 1.15415e-9 Hartree and its freshly rebuilt
orbital gradient is 7.09810e-12. RSX-QIDH converges in 7 iterations and
also passes all 11, with maximum defect 3.05363e-10 Hartree and fresh
gradient 6.48693e-12. Both PT2 amplitude residuals are 6.93889e-18.
Thus these two matched-state discrepancies close at the original gates;
the original artifact's failed comparisons are not retroactively relabelled.
The independent original references were unstable stationary states, and
the newly compared states are lower stationary branches, not a proof of
the global electronic minimum or experimental accuracy. No functional
coefficients, offsets, geometry, basis, grid or comparison gates were fitted
or relaxed. This linked-library saved-seed proof is separate from an
installed CLI/Python or arbitrary cold-start acceptance.

Native precision recovery also exposed a real memory defect: molecular XC
assembly retained two dense matrices for every grid batch. Ordered bounded
waves now retain at most 32 batch pairs while preserving every per-batch
expression and the global addition order. All 100 full-matrix parity cases
pass bitwise on the actual new library. Both 3,377,676-point Fe XC scalar
evaluations are bit-identical to the saved old native evaluations and use
about 166 MiB observed group RSS under a 512 MiB managed budget. The permanent
2,097,152-point regression fails on the old library's 1 MiB managed limit
and passes on the new one with 68,096 bytes peak managed reservations.
Managed reservations are distinct from whole-process RSS; GRAC is unchanged.
The Fe supervisor extended only elapsed time from 1800 to 3600 seconds on
verified descendant CPU progress. The earlier memory and build-environment
failures remain recorded. Evidence is
`fe-native-component-precision-20261102-06`,
`fe-original-component-gates-closure-20261102-01`, and
`ordered-xc-fe-density-validation-20261102-01` outside the package.

### KCIS correlation and global hybrids

`c="kcis"` selects the Krieger–Chen–Iafrate–Savin correlation provider.
It uses the original PW92 unpolarized and polarized uniform-gas endpoints,
spin interpolation and the spin-resolved von Weizsäcker subtraction.
Molecular and response callers use the same generic meta-GGA density
body. The scalar, dense derivatives and Taylor interfaces share its
parameters. The occupied single-spin branch recovers zero correlation
for a one-electron density when the spin kinetic density equals its
von Weizsäcker value.

The unpolarized gap model uses the published zero-gap slope and
curvature. Its rational function also satisfies
`lim(G -> infinity) G * epsilon_c(G) = -0.06483*k_F^2`. These three
constraints fix the sign of the quadratic denominator coefficient.
The literal sign in the Toulouse–Savin–Adamo Appendix Eq. (A6) gives
`c3=-0.276246996` at `r_s=2` Bohr and a pole at positive gap.
The matched coefficient is `+0.276246996`, consistent with the
general gap derivation in Fabiano et al. Eq. (A4). The polarized
coefficients retain the published 0.7, 1.5 and 2.59 factors.
The original gap fit was specified for `r_s <= 10` Bohr. The
molecular density tails use the published rational continuation.
[Original gas model](https://doi.org/10.1007/978-1-4615-4715-0_28),
[KCIS Appendix](https://doi.org/10.1063/1.1521432),
[Gap matching derivation](https://doi.org/10.1021/ct500073b).

The p324 test checks two interior spin/gradient points, the spin
permutation, uniform-gas, one-electron and large-gap limits, and each
exchange and correlation term. The independent scalar oracle
reconstructs the rational coefficients by matching the derivatives at
zero gap, and evaluates the exchange partners separately.
Three entries each have 7 first, 21 second and 47 third nonzero
independent derivative fits, plus 44 structural zeros. The 225
four-step fits used steps `[0.12, 0.06, 0.03, 0.015]` times the
coordinate-specific density-variable scales in p324. One retained
coarse interval for mPW1KCIS `D[rho_a,sigma_aa]` gave
`p=1.650208`, `R2=0.98452441` and `C=1.191e-3`. The next interval
`[0.06, 0.03, 0.015, 0.0075]` resolves it with `p=1.934853`,
`R2=0.99967751` and `C=3.400e-3`. Exponents fit absolute errors in
the dimensionless step; no error floor is used.

The linked-density harness uses the exact KCIS body and dense XC
driver with the frozen molecular SCF core. The C1 NH₃ RKS and
CH₃ UKS STO-3G tests use grid level 2, zero grid weight cutoff, and
energy, density and orbital convergence controls of `1e-12` Hartree,
`1e-10`, and `1e-9` Hartree. The independent AO and ERI contractions
reconstruct the total from nuclear repulsion, kinetic, attraction,
Coulomb, exact exchange and both semilocal terms within
`3.980e-13` Hartree.

| Molecule | Model | Local exchange (Hartree) | Correlation (Hartree) | Exact exchange (Hartree) | Total (Hartree) | Occupied–virtual residual (Hartree) |
|---|---|---:|---:|---:|---:|---:|
| NH₃ RKS | PBE1KCIS | -6.070906495646653 | -0.330467146087700 | -1.731789711288373 | -55.647404199570602 | 5.038e-12 |
| NH₃ RKS | mPW1KCIS | -6.652773119347850 | -0.330441356140165 | -1.180657167872709 | -55.678791104725676 | 6.285e-12 |
| CH₃ UKS | PBE1KCIS | -4.949811831142630 | -0.266707075167759 | -1.415285934934461 | -39.156463229428660 | 4.882e-12 |
| CH₃ UKS | mPW1KCIS | -5.425459045766001 | -0.266693310685790 | -0.964888072387934 | -39.182179388835600 | 2.602e-11 |

The named calculations use the existing energy and gradient routes.
The linked-density harness accepted all 48 nuclear gradient components
for the four C1 cases at `[0.02, 0.01, 0.005, 0.0025]` Bohr. The
exponents ranged from `1.999850` to `2.000726`, with minimum
R² `0.99999998` and prefactors from `0.005203` to `0.6685`.
The maximum unprojected translational defect was
`4.481843e-15` Hartree/Bohr.
The rebuilt package runtime and CLI/native Python/ASE route replay
are pending. The standard-five energy set and broader molecular,
periodic and high-order derivative combinations remain separate gates.

### TM and revTM independent local reference

At the spin-density, gradient and kinetic-energy-density point
`[0.31, 0.22, 0.041, 0.017, 0.029, 0.37, 0.21]`, the independent
reference gives the following exchange and correlation energy densities.

| Functional | Exchange (Hartree/Bohr³) | Correlation (Hartree/Bohr³) |
|---|---:|---:|
| TM | -0.32083890314213814 | -0.03363970000945447 |
| revTM | -0.32095441269366270 | -0.03369017961059882 |

The independent hydrogen 1s exchange integral for both variants is
`-0.3124999160504965` Hartree, against the closed value `-5/16`
Hartree. The one-electron correlation limit is zero. Independent
Taylor and central-stencil checks cover 314 nonzero first- through
third-order local components at the scaled steps
`[0.12, 0.06, 0.03, 0.015]`. Those checks give `p=1.9930..2.0164`
and minimum R² `0.9999824`. The production Rust test
`tests/p325_tao_mo.rs` compares the functional providers with the
reference and checks 162 structural zero entries. The compiled
provider and molecular public benchmarks are pending.
[TM](https://doi.org/10.1103/PhysRevLett.117.073001)
[revTM](https://doi.org/10.1021/acs.jpca.9b02921)

### Original ωB97X-D3 range-separated hybrid

Select `xc="wb97x-d3"` for the 2013 fit. The short-range exact
exchange fraction is 0.195728, the total long-range fraction is 1,
and ω is 0.25 Bohr⁻¹. The D3 pair correction uses `s6 = s8 = 1`,
`rs6 = 1.281`, `rs8 = 1.094`, and damping powers 14 and 16.
The geometry term uses the same D3 lattice sums and higher-order
response as other zero-damped D3 models. The common molecular and
dispersion-enabled periodic drivers select the correction automatically.
Conflicting dispersion or ATM requests are refused. A feature that
cannot carry its geometry term rejects the functional.
[Lin et al., Table 1 and Eqs. (21)–(25)](https://doi.org/10.1021/ct300715s).

The independent reference integrates the overlap of two Fermi spheres
for the short-range attenuation. The published spin-resolved PW92 gas
and B97 polynomials supply the two correlation channels. At
`[rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb] =
[0.31, 0.22, 0.041, 0.017, 0.029]`, the independent terms are
`-0.20217596299780535` (short-range exchange),
`-0.010701426212140007` (same-spin correlation) and
`-0.017980662850889333` (opposite-spin correlation) Hartree/Bohr³.
The six production term values at the two test points agree with the
reference values within `5.551e-17` Hartree/Bohr³.

| Geometry | Independent C₆ contribution (Hartree) | Independent C₈ contribution (Hartree) |
|---|---:|---:|
| Distorted NH₃ molecule | -6.905419944874021e-7 | -3.740114211755163e-6 |
| O–H, 1D lattice | -1.164056932663654e-4 | -3.944438957346698e-5 |
| O–H, 2D lattice | -1.525747090748414e-4 | -4.543196531626364e-5 |
| O–H, 3D lattice | -2.336382338464840e-4 | -6.176897750908122e-5 |

For those lattice sums the coordination and dispersion cutoffs are
10.41 and 12.98 Bohr, respectively. The derivative stencils stay
inside a fixed image chart with at least 0.24 Bohr of distance
between each reference image and either cutoff.

The frozen source manifest has SHA-256
`d98c4bd9f99cbe720269f198953a47e4b9ee79a092cac787abfe798b884478a9`.
The p326 executable has SHA-256
`24f94bbd16c7c02f3281d8fc4314d6d727b771cc0b1e733fc97fcb01c0835484`.
Strict Clippy passes for the library with the Python feature, the CLI
and the p326 acceptance target.

The production comparison resolves 194 local derivative components
through order four and four nonlinear parameter tangents on the
scaled interval `[0.08, 0.04, 0.02, 0.01]`. The fitted exponents are
`1.973713..2.044210`, the minimum R² is `0.99987858`, and the
prefactors are `6.738e-7..2.952e5`. The 306 structural zero
entries and 15 linear parameter tangents pass their exact checks.
The spin and attenuation limits and automatic defining correction
controls also pass.

The production periodic D3 tensor checks all 6, 21, 56 and 126
independent nuclear derivatives of orders one through four in each
of the three periodic dimensions. All 627 four-point absolute-error
fits resolve at `[0.08, 0.04, 0.02, 0.01]` Bohr, with
`p=2.004370..2.026850`, minimum `R²=0.99995313` and prefactors
`4.885e-6..2.440e-2`. The largest permutation defect is
`1.355e-18` in the order-four units of Hartree/Bohr⁴. The acoustic
defect is zero at all reported orders. Every local and periodic fit
resolves on its first interval without an error floor or failed interval.

The distorted NH₃ and NH₃⁺ calculations use STO-3G, grid level 1
and zero grid-weight cutoff at the same fixed geometry. The
energy, density and orbital controls are `1e-12` Hartree, `1e-11`
and `1e-9` Hartree. The two residuals are maximum occupied–virtual
Fock elements. Energy terms are in Hartree.

| State | Total energy | Local short-range X | Local same-spin C | Local opposite-spin C | Full exact X | Additional long-range exact X | Orbital residual | Reconstruction defect | SCF iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NH₃ RKS | -55.709938053687296 | -5.179194287611031 | -0.098902053249451 | -0.293354109949848 | -1.540321198594497 | -1.081795163558469 | 3.967e-12 | 4.974e-14 | 10 |
| NH₃⁺ UKS | -55.379715407850242 | -4.917345889459297 | -0.087896118214859 | -0.252354249221805 | -1.450737940970761 | -0.974817525453283 | 1.360e-12 | 4.263e-14 | 14 |

The shared D3 contribution is `-4.430656206242558e-6` Hartree.
All 24 molecular Cartesian gradient components pass their first
four-step interval `[0.02, 0.01, 0.005, 0.0025]` Bohr on the same
geometry and settings. NH₃ gives `p=1.999785..2.001445`, minimum
`R²=0.99999982` and prefactors `0.004471..0.6644`. NH₃⁺ gives
`p=1.999816..2.000184`, minimum R² of `1.00000000` at printed
precision and prefactors `0.01443..0.6924`. The maximum
smallest-step errors are `4.150274e-6` and `4.325815e-6`
Hartree/Bohr. Neither state has a failed fit interval.

The native replay at the p326 geometry returns the following Cartesian
energy gradients in Hartree/Bohr.

| Atom | NH₃ RKS (x, y, z) | NH₃⁺ UKS (x, y, z) |
|---|---|---|
| N | `(-0.067180454946, -0.148731985219, 0.099581697634)` | `(-0.072211008697, -0.158474882263, 0.102834546412)` |
| H₁ | `(-0.056646505606, -0.037404688464, 0.014147176627)` | `(-0.106595492401, -0.069564445250, 0.032491939791)` |
| H₂ | `(0.155054418575, -0.085880461653, -0.025923052047)` | `(0.216798480941, -0.117287025932, -0.030376948664)` |
| H₃ | `(-0.031227458023, 0.272017135336, -0.087805822214)` | `(-0.037991979843, 0.345326353445, -0.104949537539)` |

The raw translational residuals are `2.827742e-15` and
`1.514413e-15` Hartree/Bohr, respectively. The cation UKS
spin expectation is `0.752680275548014` ℏ².

The CLI, native Python and ASE pass the molecular and periodic replay
on the same frozen source. The CLI has SHA-256
`d6ee809f667aa4db4e6eac9182e1e3715bf887714040980872b3d070f638e91b`.
The molecular route fixtures use a second distorted NH₃ geometry,
STO-3G and grid level 3. The neutral and cation totals are
`-55.708060321683490` and `-55.362837878540134` Hartree.
Their shared defining D3 term is `-5.890539017903289e-6` Hartree.
The largest CLI–native energy and gradient differences are
`2.842171e-14` Hartree and `3.996803e-15` Hartree/Bohr.
The ASE and native results coincide to printed precision.

The periodic route fixtures use grid level 0, zero grid-weight
cutoff, and coordination and dispersion cutoffs of 5.4 Bohr.
The four stresses, forces, energies and orbital residuals agree across
the three routes to absolute tolerance `1e-10` in the reported units.
The native D3 energies agree with a separate directed image sum
within `8.674e-19` Hartree/cell. DOS retains the same D3 energy
and Γ ROKS lowers the objective in its fixed-cell relaxation step.
The canonical CLI writes SCF residuals to stderr.

| Periodic reference | k mesh | Total energy (Hartree/cell) | D3 (Hartree/cell) | Stress xx | Stress unit | Orbital residual (Hartree) |
|---|---|---:|---:|---:|---|---:|
| RKS, 1D | 2 × 1 × 1 | -8.945163736798934 | -2.744158945022959e-4 | -0.02657530433357035 | Hartree/Bohr | 2.678207e-15 |
| UKS, 2D | 2 × 1 × 1 | -4.726577945173450 | -1.332008962257307e-3 | -0.005799687890591475 | Hartree/Bohr² | 5.131395e-15 |
| ROKS, 3D | 1 × 1 × 1 | -7.011043353202490 | -5.076289932558375e-7 | -0.0002376458425672923 | Hartree/Bohr³ | 1.222336e-13 |
| GKS, 3D | 2 × 1 × 1 | -4.736694009378036 | -2.348260708593692e-3 | -0.001327247180650158 | Hartree/Bohr³ | 1.844247e-16 |

### Named M06-family and LC-BOP molecular response

T006 (gradient) and T019 (static polarizability) record bounded
class-level acceptance. The available pointwise, total-energy and
directional Taylor evidence for a named provider is supplemented by
component-wise molecular checks. The six selected models cover a
local meta-GGA, global hybrid meta-GGAs, a screened hybrid meta-GGA
and a long-range corrected GGA hybrid. The previously accepted
HSEsol RKS, UKS and ROKS cases are in [response.md](response.md).

| Provider | Acceptance test selection | Earlier provider-level evidence |
|---|---|---|
| M06, M06-L, M06-2X | RKS, UKS, ROKS, GKS gradients and static polarizabilities | Frozen molecular energies and local or directional response checks |
| M06-SX | RKS, UKS, ROKS, GKS gradients and static polarizabilities | Frozen water energy and directional Taylor response |
| revM06-L | ROKS and GKS gradients and static polarizabilities | Frozen water energy and directional Taylor response |
| LC-BOP | ROKS and GKS gradients and static polarizabilities | Pointwise energy and molecular SCF |

The `catalogue_molecular_response` executable has SHA-256
`ca766d76a446d2fd59c05858980c3a481ae0f3b3a0e8078c783b02afe73bacae`
and uses source manifest
`d98c4bd9f99cbe720269f198953a47e4b9ee79a092cac787abfe798b884478a9`.
Its `DFT_RS_TEST_FUNCTIONAL` filter uses canonical catalogue names
`m06l` and `revm06l` for M06-L and revM06-L.

The shared geometry is a C1 ammonia structure in Bohr with nitrogen
at `(0.13, -0.27, 0.19)` and the three hydrogen atoms at
`(1.78, 0.34, -0.08)`, `(-0.49, 1.65, 0.41)` and
`(0.31, -0.78, 1.91)`. RKS uses neutral NH₃ and UKS, ROKS and GKS
use doublet NH₃⁺. The runs use STO-3G, grid level zero, zero weight
cutoff, SCF energy tolerance `1e-13` Hartree, density tolerance
`1e-12`, orbital tolerance `1e-12` Hartree and a 300-iteration cap.
The analytic molecular gradient is compared against reconverged
energies for all twelve Cartesian coordinates. Static polarizability
is compared against reconverged dipoles for all nine ordered Cartesian
entries. The first central-difference intervals are
`[0.020, 0.014, 0.009, 0.006]` Bohr and
`[0.016, 0.011, 0.007, 0.004]` atomic units of electric field.

#### Measured C1 ammonia values

Gradients are listed in N, H₁, H₂, H₃ order in Hartree/Bohr.
Polarizability matrices are in Cartesian row order in e² Bohr²/Hartree.

| Provider | Reference | Energy (Hartree) | Cartesian gradient |
|---|---|---:|---|
| M06 | RKS | -55.67489617266337 | `(0.128805063, -0.039999609, 0.087090675), (-0.155705065, -0.025062588, 0.040768131), (0.027508399, 0.021801724, -0.000895713), (-0.000608396, 0.043260472, -0.126963093)` |
| M06 | UKS | -55.31325941244307 | `(0.108690985, -0.076952345, 0.044508346), (-0.181255240, -0.016716534, 0.075974185), (0.055475323, 0.001767626, 0.020018899), (0.017088932, 0.091901253, -0.140501430)` |
| M06 | ROKS | -55.31247674210914 | `(0.108698076, -0.077764594, 0.044159533), (-0.181232213, -0.016618644, 0.076270044), (0.055358014, 0.002206694, 0.020256763), (0.017176124, 0.092176545, -0.140686340)` |
| M06 | GKS | -55.31325941244312 | `(0.108690985, -0.076952345, 0.044508346), (-0.181255240, -0.016716534, 0.075974185), (0.055475323, 0.001767626, 0.020018899), (0.017088932, 0.091901253, -0.140501430)` |
| M06-L | RKS | -55.73797740027639 | `(0.131815571, -0.036045135, 0.094222321), (-0.155135701, -0.026327829, 0.038025500), (0.025812799, 0.022007494, -0.002724519), (-0.002492670, 0.040365470, -0.129523303)` |
| M06-L | UKS | -55.37846414711466 | `(0.105240605, -0.083623158, 0.041534786), (-0.177953443, -0.015279140, 0.075439864), (0.055164438, 0.007265748, 0.021657741), (0.017548399, 0.091636550, -0.138632392)` |
| M06-L | ROKS | -55.37765160397669 | `(0.105303148, -0.084431535, 0.041260939), (-0.177984193, -0.015201410, 0.075708210), (0.055072829, 0.007708828, 0.021917731), (0.017608216, 0.091924117, -0.138886881)` |

| Provider | Reference | Static polarizability | Largest response residual |
|---|---|---|---:|
| M06 | RKS | `(3.449781820, -0.708748577, -0.557749015), (-0.708748577, 4.485064140, -1.014925160), (-0.557749015, -1.014925160, 3.041767443)` | 5.1593e-12 |
| M06 | UKS | `(2.945720994, -0.394662896, -0.166435772), (-0.394662896, 3.831787369, -0.493367246), (-0.166435772, -0.493367246, 2.778327281)` | 9.3608e-11 |
| M06 | ROKS | `(2.938703564, -0.387084805, -0.158558455), (-0.387084805, 3.804172116, -0.493400196), (-0.158558455, -0.493400196, 2.765069885)` | 6.2187e-11 |
| M06-L | RKS | `(3.607949290, -0.742553136, -0.586256728), (-0.742553136, 4.665774908, -1.070047816), (-0.586256728, -1.070047816, 3.189554294)` | 1.0307e-11 |
| M06-L | UKS | `(2.977964755, -0.434669772, -0.229971834), (-0.434669772, 3.898794296, -0.543177277), (-0.229971834, -0.543177277, 2.731189363)` | 5.2449e-11 |
| M06-L | ROKS | `(2.974016645, -0.429196207, -0.226115812), (-0.429196207, 3.875061295, -0.545826206), (-0.226115812, -0.545826206, 2.720527269)` | 1.9183e-11 |

The collinear polarizability reports label the absolute response
residuals in e Bohr. Among the completed cases the raw ROKS orbital
residuals are `4.980391e-15` Hartree (M06) and `1.255801e-12`
Hartree (M06-L). The largest norm of the sum of the reported nuclear
gradients is `3.704643e-15` Hartree/Bohr. The largest polarizability
transpose defect is `2.052126e-10` e² Bohr²/Hartree.
For the M06 cation, the collinear UKS and GKS total energies differ
by `4.973799e-14` Hartree and their largest Cartesian gradient
difference is `2.842171e-14` Hartree/Bohr.

| Provider | Property | Fitted components | Exponent range | Minimum R² | Prefactor range |
|---|---|---:|---|---:|---|
| M06 | gradient | 48 | 1.998216–2.006738 | 0.99999882 | 3.133e-3–3.643e-1 |
| M06 | polarizability | 27 | 1.990489–2.152519 | 0.99943258 | 3.498e-2–4.707e0 |
| M06-L | gradient | 36 | 1.996511–2.012887 | 0.99999574 | 2.157e-3–3.086e-1 |
| M06-L | polarizability | 27 | 1.979052–2.012615 | 0.99998281 | 7.839e-2–4.402e0 |

All fits in this table resolve on the first interval. The remaining
selections in the test scope table are pending acceptance.

