<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

# dft-rs theory & conventions dossier

Frozen conventions, sign choices, index orders, and measured error budgets.
Grown at every phase; each phase close appends its license-audit checklist
result.

## 1. Global conventions (Phase 0)

- **Units**: Hartree / Bohr (atomic units) in every computational routine.
  XYZ I/O converts Å ↔ Bohr at the boundary (CODATA 2018,
  `src/constants.rs`, value-pinned by tests).
- **Matrices**: row-major dense `linalg::Matrix`; eigenvectors are columns;
  eigenvalues ascending (enforced, not assumed).
- **Non-orthogonal AO basis**: SCF works through the canonical
  orthogonalizer `X` (`Orthogonalizer::canonical`, eigenvalue cutoff default
  1e-7 with linear-dependence column dropping). Roothaan solve:
  `F' = Xᵀ F X`, `C = X C'`, so `Cᵀ S C = I` always holds on the retained
  subspace.
- **Electron counting**: `n_elec = Σ Z − charge`; `n_unpaired = 2S = mult − 1`;
  parity is validated before any SCF. `n_alpha ≥ n_beta` by construction.
- **Parallelism**: rayon global pool shared with faer via
  `ensure_parallelism()` (call once at every entry point;
  `DFT_RS_FAER_THREADS=1` forces sequential for reproducibility).

## 2. Basis sets and AO conventions (Phase 1)

- **Bundled sets** (`src/basis/data/*.g94`, refreshed by
  `tools/fetch_basis.py`): STO-nG, Pople, def2, Dunning cc/aug-cc, LANL,
  Stuttgart families — *all elements* BSE provides per set. ECP sections are
  fully parsed into `EcpPotential` (n_core, local channel U_L, projected
  channels U_l−U_L; radial terms `c·r^{power−2}·e^{−αr²}`); ECP *evaluation*
  is a later phase, so `MolBasis::build` rejects ECP elements loudly until
  then.
- **Normalization** (CCA): primitives scaled by the `(l,0,0)` Cartesian norm,
  contraction scaled to unit `(l,0,0)` self-overlap; all Cartesian
  components share the radial normalization.
- **Cartesian order**: x-major lexicographic (`xx, xy, xz, yy, yz, zz`).
- **Spherical order** (canonical, all l): m = 0, +1, −1, +2, −2, … .
  p shells in m terms are (z, x, y). The real-solid-harmonic coefficients
  follow the standard closed form and columns are normalized *exactly*
  against the analytic one-center metric; orthonormality is unit-tested for
  l ≤ 6. **Signs verified identical to PySCF** by the Phase 1 oracle gate.
- **Integrals** (McMurchie–Davidson): E-tables per primitive pair/axis;
  flat-array R-tensor (iterative, no recursion/hashing); kernel-adjusted
  Boys base (`Coulomb`/`Erf{ω}`/`Erfc{ω}`) is the single RSH switch point.
  ERI shell quartets contract ket-Hermite × R first (middle contraction),
  then bra. In-core ERI store has 8-fold-symmetric fill and a memory guard.
- **Oracle protocol**: geometries are exact **Bohr literals** shared verbatim
  between `tools/oracle/gen_integrals.py` and the Rust tests — PySCF pins
  CODATA-2010 Bohr, dft-rs CODATA-2018, so any Å→Bohr conversion inside the
  comparison would cost ~2e-11. PySCF matrices are permuted into the dft-rs
  AO order at generation time. Agreement gate: ≤ 1e-11 on all S/T/V/dipole
  elements and 250 ERI samples per case (H₂O/STO-3G, H₂O/def2-SVP with d
  shells, CH₄/STO-3G).

## 3. SCF (Phase 2)

- **Spin-resolved loop**: the driver always carries `P^α`, `P^β`; RHF is the
  closed-shell fast path (one Fock, cloned channels), UHF the general case.
  `F^σ = H_core + J(P_tot) − c_x·K(P^σ)` with `c_x = 1` for HF (hybrids will
  set 0 < c_x < 1, pure functionals 0, plus long-range `K_erf`).
- **Energy**: `E_elec = ½ Σ_σ Tr[P^σ (H_core + F^σ)]`.
- **J/K paths**: dense in-core contraction (reference) and screened direct
  recomputation, related tests force agreement ≤ 1e-10. Direct digestion
  enumerates the 8 slot permutations σ of each canonical shell quartet,
  deduplicated by permuted shell tuple (valid because coincident-shell
  images are equal by integral permutational symmetry). **Lesson**: an
  earlier version reconstructed the permutation from shell *values* — wrong
  whenever a shell index repeats in the quartet; permutations must be
  carried as slot mappings.
- **Convergence machinery**: CDIIS on `Xᵀ(FPS−SPF)X` (channels
  concatenated; bordered system solved by a pivot-guarded elimination that
  reports singularity for clean fallback), density damping while the DIIS
  error is large, optional static level shift applied in the orthonormal
  basis (`F_o += λ(I − P_o)`). Convergence on `|ΔE|` and `rms(ΔP)`.
- **⟨S²⟩** = `S_z(S_z+1) + N_β − Σ_ij |(C^αᵀ S C^β)_ij|²`.
- **Oracle gates** (all ≤ 1e-8 RHF / 1e-7 UHF vs frozen PySCF 2.14.0):
  10 closed-shell molecules × {STO-3G, def2-SVP} + O₂ triplet and CH₃
  doublet × both bases with ⟨S²⟩. Bohr-literal geometries embedded in the
  CSVs (single source of truth). Known behavior: bare-core guess on N₂
  converges to an excited SCF solution (guess quality, not a bug) — the
  guess-independence gate uses H₂O.
- **Wavefunction analysis**: `ScfResult` exposes MO coefficients/energies,
  spin densities, S, H_core, converged Focks; `--dump-wfn` writes the
  self-describing full-precision text checkpoint
  (`properties/checkpoint.rs`).

## 4. DFT: functionals, grid, RI-J (Phase 3)

- **Functional framework**: one `Scalar`-generic energy-density expression
  per functional (energy per volume; `vrho` = KS potential directly);
  derivatives up to order 3 via nested fixed-width duals (`dual.rs`);
  parameter gradients by seeding θ through the same path
  (`param_gradient`, FD-gated). Resolution rules (X/C/XC, the
  "XC-integrated given component-wise" early error, `hf` special case) are
  unit-tested in `xc/mod.rs`.
- **LDA set**: Slater (scale param), VWN5, VWN3(RPA), PW92 — parameters
  transcribed from the papers, verified end-to-end against the frozen
  PySCF oracle (9 cases, RKS+UKS, ≤ 5e-7 at near-converged grids).
- **Grid**: the standard atom-centred form — TA-M4 radial (Gauss–Chebyshev
  2nd kind, Treutler–Ahlrichs ξ for H–Kr) × Lebedev–Laikov angular rules
  (published orbit generators for degrees 3–131 stored in
  `grid/lebedev_data.rs`, every rule self-verified against closed-form
  monomial integrals), five-region angular pruning (boundaries
  α·R_Bragg with α = (0.25, 0.5, 1.0, 4.5) for H–He, (0.1667, 0.5, 0.9,
  3.5) for Li–Ne, (0.1, 0.4, 0.8, 2.5) beyond; degrees 11, 15, next lower,
  full, next lower), Becke partition (k=3) with Treutler's size adjustment
  χ = √(R_i/R_j). Level 0–9 tables of radial points and angular degree by
  period (level 3: 50/75/80/90 points, degree 29/29/35/35 for H–He, Li–Ne,
  Na–Ar, K–Kr) or the named presets `fine` (75, 302), `ultrafine` (99, 590)
  and `superfine` (150/225, 974); per-element/per-atom overrides. The
  former product rule (Gauss–Legendre cosθ × uniform φ) stays selectable
  with `grid_angular_scheme = product`. Partition of unity ⇒ ξ and
  Bragg–Slater tables affect convergence rate only. ∫ρ is reported every
  run (`grid_electron_count`). Acceptance: `tests/p238_standard_grid.rs`.
- **SCF driver**: single spin-resolved loop for HF/RKS/UKS/hybrids;
  energy assembled explicitly as
  `Tr[P H] + ½Tr[P J] − ½cx Σ Tr[Pσ Kσ] + Exc`. K work is skipped
  entirely for pure functionals. `DFT_RS_SCF_TRACE=1` prints per-iteration
  E / |FPS−SPF| / rms(ΔP).
- **Lesson (DIIS endgame)**: near convergence `B_ij ~ ‖e‖²` underflows an
  absolute pivot guard, silently disabling DIIS exactly when the plain
  LDA iteration is weakly unstable — observed as a commutator creeping up
  from 1e-7 while the energy sat at machine precision. Fix: normalize B by
  its largest element (coefficients are scale-invariant) and shrink the
  subspace from the oldest vector on residual singularity.
- **RI-J density fitting** (optional, off by default;
  `ScfOptions::density_fit` / `--ri-j`): 2c/3c Coulomb integrals via the
  zero-exponent unit-s-shell trick through the standard MD quartet path
  (closed-form-tested); `J = (μν|P) V⁻¹ (Q|λσ)Pλσ`, K always exact.
  Verified against PySCF `density_fit` with the *same* bundled
  def2-universal-JFIT: ≤ 5e-7 (grid-limited). RI is invariant to auxiliary
  normalization conventions (V⁻¹ compensates).
- **Molden output**: [5D]/[7F] spherical, p shells permuted to Molden's
  (x,y,z); verified by a PySCF `tools.molden.load` round-trip
  (max|CᵀSC−I| = 6e-11 for water/def2-SVP; `tools/oracle/check_molden.py`).
- **Deferred**: SAD initial guess (GWH suffices for the current gates);
  meta-GGA grid variables (τ, ∇²ρ) arrive with the GGA phase.

## 5. GGA, hybrids, range separation (Phase 4)

- **GGA driver**: 5-variable (ρα, ρβ, σαα, σαβ, σββ) dual tower, orders
  0–2 + analytic parameter gradients. Vxc gradient term
  `Σ w (2vσ_σσ∇ρσ + vσ_αβ∇ρσ̄)·∇(φμφν)` assembled as `AᵀΦ + ΦᵀA`.
- **Functionals**: B88, PBE-X, OPTX, LYP (Miehlich form), PBE-C; composites
  PBE, BLYP, B3LYP (VWN3 convention) / B3LYP5, O3LYP, PBE0. All gated
  against frozen PySCF ≤ 5e-7 (closed + open shell).
- **Lesson (VWN3)**: libxc's `VWN_RPA` (the B3LYP/Gaussian "VWN3") = RPA
  fit parameters + the SIMPLE ζ-interpolation ε_P + (ε_F−ε_P)f(ζ) — NOT
  the full VWN interpolation with the α_c spin-stiffness term. Closed-shell
  tests cannot distinguish (ζ = 0); the open-shell CH₃/B3LYP oracle caught
  a 2e-4 discrepancy. Diagnostic that settled it: feeding our density to
  PySCF (`energy_tot(dm)`) showed our energy expression, not our SCF, was
  off.
- **Lesson (PBE-C)**: PBE's local base is the "modified" PW92 digits
  (A = 0.0310907 / 0.01554535 / 0.0168869), not the PW92 paper table
  values — a uniform ~2e-6 Ha offset otherwise.
- **Level shift** is auto-released once the DIIS error < `level_shift_until`
  (1e-4): a statically shifted Fock can converge to non-aufbau states.
- **Range separation** (LC/CAM/ω): SR-GGA exchange via the ITYH effective-
  hole attenuation (large-a branch uses the asymptotic series F = u²/9 −
  u⁴/60, u = 1/(2a); the direct form loses to 4a³-scale cancellation).
  Pinned pointwise against libxc `gga_x_ityh` ≤ 1e-9. SCF adds a second
  `JkEngine` with the `Erf{ω}` kernel:
  `F −= cx·K + clr·K_lr`, `E −= ½Σσ Tr[Pσ(cx Kσ + clr K_lrσ)]`.
  CAM-B3LYP (α = 0.19, β = 0.46, ω = 0.33; libxc rsh_coeff (0.33, 0.65,
  −0.46) is the same thing in SR/full form) gated vs PySCF ≤ 5e-7 including
  open shell; LC-BLYP/LC-PBE gated by internal limits (ω→0 → pure GGA,
  ω→∞ LR-HF → HF). `with_omega` rebuilds ω everywhere (K_lr kernel + every
  functional part with an "omega" parameter) for ω-tuning workflows.
- **Analytic-derivative policy** (user directive): every production
  derivative — density, σ, functional parameters, later basis parameters —
  is analytic (dual towers / closed forms); finite differences appear only
  in test gates.

## 6. Analytic gradients & dispersion (Phase 5)

- **Integral derivatives** by the shifted-shell identity
  `∂/∂A φ(l) = 2a·φ(l+1) − l·φ(l−1)`: the raising partner folds 2a into the
  contraction coefficients, so every derivative reuses the untouched MD
  routines; ERIs keep the differentiated index Cartesian
  (`eri_quartet_masked`) for the recombination. Fourth-center/operator
  terms via translational invariance.
- **Lesson (ordered-pair factor)**: contracting `Σ m ∂S` over ordered shell
  pairs with bra-side derivatives counts exactly half — each ordered element
  has bra AND ket sides (equal under symmetric m). Factor 2 restored for
  S/T/V (operator-center term included).
- **Lesson (exchange weight symmetry)**: the per-element exchange weight
  `Pˢ(μλ)Pˢ(νσ)` is NOT invariant under individual index swaps; the
  canonical-quartet × degeneracy shortcut requires the σ-orbit-symmetrized
  `½[Pˢ(μλ)Pˢ(νσ) + Pˢ(μσ)Pˢ(νλ)]`.
- **XC gradient with FULL grid response**: orbital term (AO 1st+2nd
  derivatives), Becke weight derivatives (`becke_weight_gradient`,
  FD-gated), point-motion term `w∇_r e` — including ∇_r of the Becke weight
  itself (`∇_r w = −Σ_G ∂w/∂R_G`), whose omission left a 6e-7 sum-rule
  residual. Gates: translational sum rule ≤ 1e-8 (SVWN5/B3LYP/CAM-B3LYP),
  FD ≤ 5e-6, PySCF gradient oracle (7 cases incl. UHF and grid_response)
  ≤ 2e-6 (HF) / 1e-5 (DFT).
- **Dispersion**: D2 (paper table, H–Kr), D3 zero/BJ with MOPAC-provenance
  (Apache-2.0) reference data; `Scalar`-generic ⇒ analytic gradients by
  coordinate dual seeding (CN coupling exact; FD ≤ 1e-9, sum rule exact).
- **Optimizer**: L-BFGS (two-loop, step clip, energy backtracking).
- **RI note**: gradients always use exact-ERI derivatives; with RI-J SCF
  the residual inconsistency is the fitting error itself (documented).

## 7. Hessian, vibrations, IR, TS search (Phase 6)

- **Numerical Hessian**: central differences of the analytic gradient
  (parallel over 6N displacements, geometry-rebuilt basis/grid), APT
  (∂μ/∂R) harvested from the same displaced SCFs, symmetrized.
- **Vibrational analysis**: masses IUPAC-2021 (u → mₑ), Eckart projection
  by Gram–Schmidt TR vectors (near-null rotations dropped for linear
  molecules), ν̃ = √λ·219474.63 cm⁻¹, IR intensities
  `I[km/mol] = 974.8801·Σ|∂μ/∂Q|²` (Q in √amu·Bohr).
- **Gates**: frequencies vs the frozen PySCF **analytic**-Hessian oracle at
  PySCF-optimized geometries — HF/STO-3G ≤ 3 cm⁻¹, B3LYP/def2-SVP
  ≤ 12 cm⁻¹ (numerical-vs-analytic + grid + mass-table differences);
  exactly 6 projected TR modes; no spurious imaginary modes.
- **P-RFO**: uphill shift λ_p = ½λ₁+½√(λ₁²+4g₁²) on the lowest mode,
  downhill λ_n from the augmented-matrix eigenvalue on the rest; step
  clipping; per-step numerical Hessians. Unit-gated on a synthetic
  quadratic saddle.

## 8. External field, response, polarizability, Raman (Phase 7)

- **Field**: `H' = +E·r` (electronic; μ̂ = −r) + nuclear `−ΣZ R·E`;
  gate `dE/dE_ax = −μ_ax` ≤ 1e-6 (HF, B3LYP).
- **Response solver** (`response/`): matrix-free orbital-Hessian product
  `(εa−εi)U + [J(ΔP) − cx K(ΔPσ) − clr K_lr(ΔPσ) + f_xc ΔP]_ai`, Jacobi-PCG.
  The XC kernel contraction implements the full spin-resolved GGA form
  (v2rho2 / v2rhosigma / v2sigma2 + the ∇Δρ terms through vσ).
- **Gate**: analytic α = finite-field dμ/dE ≤ 1e-4 for HF / SVWN5 / B3LYP —
  an independent numerical route validating J/K response AND the GGA
  kernel; α symmetric ≤ 1e-6.
- **Raman**: activities 45ā′² + 7γ′² (Å⁴/amu) from central differences of
  the ANALYTIC α along normal modes (dα/dQ, Q in √amu·Bohr; conversion
  a.u.→Å²/√amu = a0²).

## 9. B97/HCTH family + run logging (Phase 9a)

- **B97 power series** (`xc/b97.rs`): one generic energy density serves
  B97 (hybrid, cx = 0.1943) and HCTH/93/120/147/407 (term count inferred
  from the parameter length; all γ's and coefficients differentiable).
  Same-spin/opposite-spin split on **PW92 with the PAPER digits** (unlike
  PBE-C's MOD digits) — pinned pointwise vs libxc (7e-11).
- **Lesson (coefficient recovery)**: the B97 energy is LINEAR in the 15
  series coefficients, so sampling libxc at ~400 random spin-polarized
  points and solving least squares recovers the reference tables to 1e-7 —
  this validated every table digit and caught a transcribed-value error in
  HCTH/147's c_ss (correct tail: 1.05747, 0.885429). Oracle gates: 6 new
  cases (incl. open shell) ≤ 5e-7.
- The XC-integrated resolution rule now has a real subject: `--x hcth407
  --c hcth407` continues, `--x hcth407 --c lyp` errors before any SCF.
- **Run logging** (`logger.rs`): SCF iteration lines + timings to console /
  file / both (CLI default: both, `<input>.dft.log`) / none; the library
  stays silent unless initialized (`DFT_RS_SCF_TRACE=1` remains as the raw
  debug fallback).

## 10. Meta-GGA, ωB97X-D/X-V/M-V, M06 family, VV10 (Phase 9b)

- **Meta-GGA infrastructure**: `GridVars.tau` (τσ = ½ Σ P^σ ∇φ·∇φ, the
  libxc convention), `XcOut.vtau`, a 7-variable dual `MggaDriver`
  (`xc/mgga.rs`), and the KS τ-term ½ Σ_ax gᵀ diag(w·vτ) g in `ks.rs`.
  Analytic nuclear gradients / response kernels for meta-GGA are **not yet
  implemented** and error out explicitly (no silent GGA fallback).
- **VV10 nonlocal correlation** (`nlc.rs`): E = ∫ρ(β + ½∫ρ′Φ),
  Φ = −3/(2gg′(g+g′)), g = ω₀R² + κ; post-SCF on a coarse grid; params
  (b, C) = (5.9, 0.0093) original / (6.0, 0.01) for the -V functionals.
  Gated against self-consistent PySCF ΔNLC ≤ 2e-5 (BLYP+VV10 and the full
  ωB97M-V total).
- **ωB97X-D / ωB97X-V** (`xc/b97.rs`): SR-LSDA-attenuated B97 series
  (F(ω/2kF) factor); coefficients least-squares-recovered from libxc.
  The ωB97X-D dispersion tail is the separate `--disp chg` model
  (Chai–Head-Gordon damping [1 + 6(R/Rr)⁻¹²]⁻¹ over the D2 C6 table).
- **M06 family** (`xc/m06.rs`): per-spin PBE-exchange × Σ aᵢwⁱ
  (w = (τ^LSDA−τ)/(τ^LSDA+τ)) + VS98 rational h(x², z) exchange;
  UEG-split correlation with the D-factor same-spin channel — all on
  **PW92 MOD digits** with α_x = 0.00186726 (5-digit α_x truncation costs
  ~1e-5!). LSQ recovery to ≤8e-13 confirmed every table digit and showed
  the reference implementation uses **a11 = −9.049762** for M06-L (the
  2006 paper prints −9.049841) and cab4 = −12.55699. SCF oracles (incl.
  open-shell M06-2X) ≤ 1e-6 at grid level 9.
- **ωB97M-V** (`xc/b97.rs::wb97m_v`): sparse 2D series in (w, u);
  hypothesis testing by LSQ residual pinned the form — **no D factor**,
  opposite-spin variables are the *averages* x̄² = ½(x²α+x²β),
  t̄ = ½(tα+tβ) (residual 1e-14; the wrong variants stall at 1e-3).
  cx = 0.15, clr = 0.85, ω = 0.3. Semilocal SCF ≤ 2.5e-6; total with
  VV10(6.0, 0.01) ≤ 2e-5.
- **Lesson (PySCF NLC quirk)**: `mf.xc = "wb97m_v"` silently auto-enables
  VV10 (libxc `is_nlc`); `mf.nlc = 0` yields the semilocal-only SCF needed
  for tight gating. The first wb97mv oracle was off by exactly −E_VV10.
- CLI: `--nlc auto` enables VV10 for names matching -V (`wb97x-v`,
  `wb97m-v`); M06 names never match.

## 11. Effective core potentials (Phase 4.5)

- **Semi-local ECP integrals** (`integrals/ecp.rs`), G94 convention
  `U = U_L + Σ_{l<L} Σ_m |lm⟩(U_l − U_L)⟨lm|`, radial terms
  c·r^{n−2}·e^{−ηr²} (`EcpTerm.power` stores **n**; the r² volume element
  folds with the potential to r^n in the radial integrand).
- Method: binomial shift of the Cartesian polynomial about the ECP
  center + the plane-wave-like expansion e^{2αs·d} = 4π Σ i_λ Y_λμ(ŝ)Y_λμ(d̂)
  (McMurchie–Davidson structure). Angular factors are **exact** closed-form
  monomial sphere integrals (4π(a−1)!!(b−1)!!(c−1)!!/(a+b+c+1)!!) through
  L²-normalized solid-harmonic tables; the projector Σ_m|lm⟩⟨lm| is
  invariant under the orthonormal m-basis choice, so the CCA solid
  harmonics serve directly. Radial integrals: 100-point Gauss–Legendre on
  the peak interval of e^{−pr²+kr+shift} with e^{−x}-scaled Bessel ĩ_λ
  (downward Miller recursion normalized by closed-form ĩ_0; upward when
  x ≫ λ) — all exponents non-positive, no overflow.
- Wiring: `MolBasis` carries `ecps`/`n_core_electrons`; H_core gains U_ecp;
  V and E_nn use Z_eff = Z − n_core; electron count and the nuclear dipole
  use the valence values. Analytic gradients with ECPs error out explicitly
  (derivative terms staged to the Phase 8 derivative pass).
- Gates: type-1 vs the exact three-Gaussian closed form (1e-12) and an
  independent spherical-coordinates quadrature reference (s/p/d shells,
  n = 0, 1, 2); type-2 vs an independent projector-quadrature reference
  (no Bessel/monomial machinery shared); Y-orthonormality and Bessel
  recursion self-tests; **LANL2DZ NaH/Cl₂ HF ≤ 1e-7, B3LYP/PBE ≤ 5e-7 vs
  PySCF** with basis + ECP numbers fed from our bundled g94 file.
- **Lesson (BSE g94 ECP layout)**: the ECP part of the file repeats the
  bare element header line before `<EL>-ECP`; a parser keyed only on
  headers clobbers the parsed shells with an empty block.

## 12. Analytic CPKS Hessian + analytic APT (Phase 8)

- **Skeleton second derivatives** (`integrals/deriv2.rs`): S/T need only
  the MIXED bra-ket derivative (translational invariance gives
  d²/dAdA = −d²/dAdB); V computes AA/AB/BB nested/mixed blocks with the
  operator-center rows from sum rules (AC = −AA−AB, CC = AA+AB+BA+BB);
  ERIs use the six mixed index pairs only (d_i d_i = −Σ_{j≠i} d_i d_j),
  full ordered quartet loops (no permutational shortcut — correctness
  first; the canonical-quartet optimization is a listed follow-up).
  Nested second-derivative combine:
  `rr[u+x̂+ŷ] − (u+ŷ)_x rl[u+ŷ−x̂] − u_y rl[u−ŷ+x̂] + u_y(u−ŷ)_x ll`.
- **Derivative matrices** (`integrals/deriv_mat.rs`): S/T/V (operator
  center included), derivative Fock J^{(a)}[P]/K^{(a)}[Pσ], and dipole
  integrals (fixed origin, same-atom pairs do NOT cancel). Ordered-pair
  loops with bra-derivative + transpose-scatter (no halving).
- **CPKS (gauge-invariant)**: only U_ai and S̃ ever appear —
  P⁽¹⁾ = Σ U_ai(C_aC_iᵀ+C_iC_aᵀ) − Σ S̃_ij C_iC_jᵀ;
  W⁽¹⁾ = Σ C_iC_jᵀ[F̃ᵗᵒᵗ_ij − (ε_i+ε_j)S̃_ij] + Σ ε_i U_ai(C_aC_iᵀ+C_iC_aᵀ),
  F̃ᵗᵒᵗ = MO(F^{skel} + G(P⁽¹⁾)); RHS_ai = [F^{skel}+G(ΔP_oo)]_ai − ε_i S̃_ai;
  H_ab = skel² + Σ_σ Tr[P^{(b)σ}F^{(a)skel,σ}] − Tr[W^{(b)}S^{(a)}]
  (restricted: α quantities ×2).
- **XC grid terms** (`hessian/xc_terms.rs`, LDA + GGA): total-derivative
  operator D_a = ∂/∂R_A + δ_{A,owner}∇_r with c^A_μ = δ_{A,own}−δ_{A,at(μ)}
  collapsing orbital terms into per-atom-restricted + full sums. GGA
  needs third AO derivatives on the grid (added to `grid/eval.rs`,
  FD-gated). **Becke weight second derivatives via nested forward duals**
  over a Scalar-generic weight (point seeded with its owner) — exact,
  in line with the project AD policy. V_xc^{(a)} derivative-potential
  matrices feed F^{(a)} and the CPKS RHS.
- **Analytic APT**: ∂μ/∂R = Z_eff δ − Tr[P⁽¹⁾D] − Tr[P D^{(a)}] from the
  same CPKS solutions → analytic IR intensities.
- Gates (`tests/p8_analytic_hessian.rs`): analytic vs numerical Hessian
  ≤ 2e-6 and APT ≤ 5e-6 for **RHF, UHF, SVWN5, BLYP, B3LYP** (water,
  STO-3G, default grid) + translational sum rules. The CLI
  `frequencies`/`hessian` commands auto-select the analytic path
  (HF/LC-HF/LDA/GGA, no ECP, no dispersion).
- Staged: meta-GGA τ terms, ECP derivative terms, dispersion Hessians
  (numerical path remains as the permanent cross-check), canonical-quartet
  speed for ≥15-atom systems.

## 13. Python bindings + ASE calculator (Phase 9)

- `src/python.rs` (feature `python`, pyo3/abi3-py39): explicit functions
  for molecular SCF, Hessians/vibrations/Raman, optimization and saddle
  search, magnetic/response properties, SAPT, and periodic calculations.
  Results are plain dictionaries with nested lists (`np.array(...)` on the
  Python side; no NumPy crate dependency). Atomic units throughout.
- `python/dft_rs/`: `DftRs` ASE calculator (energy/forces/dipole;
  eV/Å conversions via `ase.units`; `wavefunction=True` stores the dump in
  `results`). The compiled cdylib is imported as `dft_rs_core` (renamed
  from `libdft_rs.so` / built with maturin) — the pure-Python package name
  `dft_rs` stays distinct to avoid the import collision.
- Build/test host: WSL (rustc 1.94, CPython 3.12); Windows Python 3.14
  needs a pyo3 with 3.14 support or the abi3 forward-compat flag.
- Gates (`python/tests/test_ase.py`): ASE analytic forces vs central
  differences of the potential energy ≤ 5e-4 eV/Å (measured 2.7e-5) for
  HF and SVWN5; wavefunction dump orthonormality CᵀSC = I ≤ 1e-8.

## 14. Meta-GGA analytic gradients + TDDFT/TDA (post-plan)

- **Meta-GGA gradients** (`gradient.rs`): the vτ grid-response terms —
  orbital term ½vτ Σ_u h_{xu,μ} y_u,μ per AO (the −2w loop prefactor
  yields −w vτ Σ h y) and point motion (∇_r τ)_u = Σ_v Σ_μ h_{uv,μ} y_v,μ;
  the weight-response term follows automatically through e_i. Gates:
  M06-L (pure meta) and M06-2X (hybrid meta) analytic vs FD ≤5e-6 +
  translational sum rule. M06-family optimization and numerical
  frequencies are unlocked; the meta response kernel (∂²e/∂τ²) is still
  staged (no meta polarizability/TDDFT/analytic Hessian).
- **TDDFT (TDA)** (`response/tddft.rs`): spin-resolved Casida A-matrix
  product with the UNSYMMETRIZED transition density X = C_v U C_oᵀ —
  note `hess_vec`'s symmetrized ΔP = X + Xᵀ is the (A+B) product used by
  static CPKS, so TDA has its own product with a non-symmetric J/K
  contraction. That contraction respects the caller's explicit SCF
  direct/in-core mode rather than requiring a fixed-size in-core tensor.
  Closed-shell singlet/triplet emerge from
  X_β = ±X_α; fxc depends only on the symmetric part of dp, so the
  existing (symmetric-input) contraction is exact given ½(X+Xᵀ).
  Solvers: exact diagonalization (dim ≤ 512) and block Davidson (2-pass
  Gram–Schmidt, stagnation fallback, subspace collapse), cross-checked to
  1e-7. Transition dipoles/oscillator strengths: restricted-singlet spin
  factor √2, triplet f ≡ 0.
- Gates (`tests/p10_tddft.rs`): 6 frozen PySCF TDA oracles — HF-CIS
  singlet+triplet, SVWN5 singlet+triplet, B3LYP singlet (water), UKS
  SVWN5 spin-resolved (CH₃) — energies ≤2e-6 (HF) / 5e-6 (DFT),
  oscillator strengths ≤1e-4 on non-degenerate states. CLI: `tddft`
  command (`--nstates`, `--tda-spin`).
- **Lesson**: the ground-state response operator and the excitation
  A-matrix differ exactly by the X ↔ Xᵀ symmetrization of the exchange
  contraction; reusing hess_vec for TDA silently computes A+B.

## 15. RPA, VV10 gradients, meta response kernel, ωB97 derivatives

- **Full TDDFT/TDHF (RPA)** (`response/tddft.rs`): Casida reduced form
  M = (A−B)^{1/2}(A+B)(A−B)^{1/2}, MZ = ω²Z; (A−B) = (ε) − cx K[X−Xᵀ]
  (J and adiabatic fxc cancel), (A+B) doubles J/fxc with K[X+Xᵀ].
  Transition vector |X+Y⟩ = (A−B)^{1/2}Z/√ω feeds the oscillator
  strengths. 14 frozen PySCF TDA/RPA oracles (TDHF, RPA-SVWN/B3LYP,
  triplets, UKS, meta M06-L) — energies ≤2e-6/5e-6 (5e-5 meta: the M06
  w-series kernel is quadrature-sensitive vs PySCF's radial scheme).
- **PW92 ferro branch** (`lda::pw92_epsilon_ferro`): the ζ-interpolation
  evaluates (1−ζ)^{4/3} at ζ = 1 whose SECOND dual derivative is
  0^{−2/3}·0 = NaN — the B97/M06 same-spin channels now use the exact
  ec(rs,1) branch, and the opposite-spin block is skipped unless both
  spins are populated (E(ρ,0) ≡ E_ferro(ρ) makes it exactly zero).
  This unlocked the **ωB97X-D analytic Hessian** (RSH + GGA, gated vs
  numerical ≤2e-6) and ωB97 kernels generally; a finiteness regression
  test pins all second derivatives over the small-density range.
- **VV10 analytic gradient** (`nlc.rs::vv10_gradient`): orbital terms via
  the effective local potentials vρ = P1 + ρ(∂ω₀/∂ρ P2R + ∂κ/∂ρ P2K),
  vσ = ρ ∂ω₀/∂σ P2R (P1/P2R/P2K are the O(N²) kernel accumulators),
  Becke-weight response with ∂E/∂w_i = ρ_i P1_i, point motion, and the
  explicit kernel R_ij term (net factor 2 on the owner after the pair-½
  cancellation — FD-gated). ωB97X-V / ωB97M-V gradients are now exact
  (semilocal+RSH+VV10; wired into the CLI gradient command).
- **Meta-GGA response kernel** (dedicated modules per the project
  layout policy: `xc/mgga.rs` order-2 driver via nested 7-variable duals
  filling v2rhotau[4]/v2sigmatau[6]/v2tau2[3]; `response/mgga_kernel.rs`
  holds the τ-channel fxc contraction with the response image
  ½Σ_u g_uᵀdiag(w Δvτ)g_u). `fxc_contraction` dispatches by family, so
  meta CPKS/polarizability/TDA/RPA all work (M06-L TDA+RPA gated).
- **Meta-GGA analytic Hessian** (`hessian/mgga_terms.rs`, dedicated
  module): 7-variable kernel quadratic (v2rhotau/v2sigmatau/v2tau2),
  D_aτ = δ_Ao HY1_full − HY1_at (HY1[x] = Σ_u Σ_μ h_{xu,μ}(g_uP)_μ),
  D_aD_bτ = same-index third-derivative term (t3·gP) + hP-row cross term
  (both sides covered by ONE cross expression — unlike DD∇ρ, whose two
  factor functions differ), and the V_xc^{(a)} τ channel
  D_a[w ½vτ Σ_u g_{u,μ}g_{u,ν}]. Gates: FD at fixed P (all V^{(a)}
  entries; skeleton spot checks) and **M06-L / M06-2X analytic vs
  numerical Hessians ≤ 2e-6** — the CLI frequencies path now covers meta.
- **ECP analytic gradient** (`integrals/ecp.rs::ecp_gradient`): bra
  derivatives of the semi-local blocks via raised/lowered shells
  (ordered-pair loop, operator symmetry ⇒ overall factor 2), ECP-center
  term from translational invariance; wired into `assemble_gradient` with
  Z_eff charges. FD-gated (NaH/LANL2DZ ≤ 5e-7 + translational sum).
- **Organometallic gates** (`tests/p11_organometallic.rs`, user order):
  Ni(CO)₄ (Td, 9 atoms, LANL2DZ with the Ni ECP) — HF energy ≤1e-7,
  full analytic gradient (ECP derivative terms included) ≤1e-6 with the
  symmetry-zero Ni gradient, and three Hessian columns (central
  differences of our analytic gradient vs the frozen PySCF analytic ECP
  Hessian) ≤5e-5; B3LYP energy+gradient. Runtime ~10 min in release.
- **Transition-metal ECP sweep** (`gen_ecp_metals.py` + p45 gates, user
  order): the full ECP integral matrix ⟨μ|U_M|ν⟩ for EVERY 3d metal
  Sc..Zn plus Pd on M–H probes matches PySCF `ECPscalar` ≤1e-9 in the
  dft-rs AO ordering (state-free, exercises every semi-local channel per
  element), plus closed-shell SCF energies TiCl₄ (d⁰), CuH (d¹⁰), ZnH₂,
  and the Pd atom (d¹⁰ ¹S) ≤1e-7. PySCF quirk: `mf.Gradients()` /
  `mf.Hessian()` need `import pyscf.grad` / `import pyscf.hessian`.
- **Self-consistent VV10** (`nlc.rs::vv10_potential` + `ScfOptions::vv10`):
  the nonlocal potential (from the same P1/P2R/P2K kernel accumulators as
  the gradient) enters the Fock matrix every iteration, making the
  composite energy variational — SC totals match PySCF's self-consistent
  VV10 ≤5e-6 (vs the 2e-5 post-SCF delta), and the fixed-density VV10
  gradient becomes the EXACT derivative (FD-gated ≤5e-6). Both modes are
  user-selectable (`--nlc auto|vv10|vv10-post|none`); in post-SCF mode the
  density-response cross term of the gradient is neglected (documented).
  ωB97X-V/M-V frequencies now use the fully analytic SC-VV10 Hessian
  (§17); the numerical route remains the permanent cross-check.
- Still staged: the fully analytic VV10 Hessian (skeleton D_aD_b E_VV10 +
  V_nlc^{(a)} matrices + the nonlocal f_nlc response coupling), ECP second
  derivatives (numerical route available), D4, canonical-quartet speed,
  PCM/NMR/PBC.

## 16. Functional pack II: legacy correlation, screened/RSH holes, r²SCAN

- **P86 vs PW91 C(rs) conventions** (`xc/legacy_c.rs`): P86 rides on the
  PZ81 local fit and the Rasolt–Geldart **'81** coefficient
  C(rs) = 0.001667 + (0.002568 + 0.023266rs + 7.389e-6rs²)/(1 + 8.723rs
  + 0.472rs² + 0.07389rs³). PW91-C is NOT the same C: the reference
  implementation uses the RG **'86** Padé (quadratic/quadratic, no rs³
  term), the LDA_C_PW **paper** digits with truncated f''(0) = 1.709921
  as the local base, β = ν·C_c0 with exact ν = (16/π)(3π²)^{1/3}, and the
  H1 offset C_xc(0) = 0.002568. Identified by σ→0 probes against libxc
  (convention mismatches are probed numerically, never guessed).
  Ferro branches mirror §15. Pinned 3 points ≤1e-9; BP86/B3PW91 SCF
  gates (B3PW91 = 0.08 Slater + 0.72 B88 + 0.19 PW92 + 0.81 PW91-C,
  cx = 0.20).
- **SR exchange holes** (`xc/sr_x.rs`): HJS ωPBE (closed-form ζ/η/λ/χ
  machinery, PBE fit a[6]/b[9], floors 1e-10) and wPBEh (HSE hole:
  erfcx/E1-scaled special functions — added to the Scalar trait with
  exact-derivative lifts (d erfcx = 2x·erfcx − 2/√π; d e^xE₁ = e^xE₁ −
  1/x) and branch-continuity tests; EG small-s quartic; Henderson 2009
  s-rescaling; large-w branch at w > 14). Composites: HSE06/HSE03 =
  wPBEh(0) − 0.25·wPBEh(ω) + PBE-C with cx = +0.25, clr = −0.25 (the
  K_erfc = K − K_erf trick), LC-ωPBE (wPBEh, ω = 0.4, clr = 1),
  LC-ωPBE-HJS (= libxc LC_WPBE_WHS). ω is a live functional parameter
  (∂E/∂ω via the same dual machinery). Pinned 4–6 points per hole/ω
  ≤1e-9; SCF gates at 2.5e-6 for wPBEh composites (the s = 1 rescaling
  kink is C¹ and converges slowly in quadrature — level-8/9 totals
  straddle the oracle).
- **M06-SX** (`xc/m06.rs::m06_sx`): revM06 functional form, Wang 2020
  parameters, screened hybrid cx = +0.335 / clr = −0.335 / ω = 0.10.
  Pinned (5e-9, coefficient-transcription scale) + SCF gate.
- **r²SCAN** (`xc/r2scan.rs`): re-regularized SCAN. ᾱ = (τ − τ_W)/(τ_unif
  + η·τ_W), degree-7 α-polynomial on [0, 2.5] with exp branches outside,
  C_η/dp2-damped gradient expansion (exchange S10–S12) and the Δy term
  (correlation S34) with LSDA rs-derivatives obtained by nesting a
  single-slot dual over rs inside the S-generic body (no hand
  derivatives). PW92 MOD digits + exact f''(0) local base; SCAN LDA0 fit
  b1c/b2c/b3c; ferro branch at ζ = 1 (G_c(1) = 0 kills ec0). Pinned 5
  points (α branches + ferro) ≤1e-9 X and C; SCF gates h2o + CH3.
- **FHC input bound** (`xc/mgga.rs`): libxc 7's meta-GGA driver enforces
  the Fermi-hole-curvature bound σσσ ≤ 8ρστσ (τ floored 1e-20) and
  |σαβ| ≤ (σαα + σββ)/2 for ENFORCE_FHC functionals before evaluation,
  reporting derivatives w.r.t. the CLAMPED variables. `MggaDriver` gets a
  per-functional `clamp_fhc` (true for r²SCAN only): input values are
  clamped, dual seeds stay on the original slots — bit-compatible with
  the reference convention. Existing meta functionals keep clamp_fhc =
  false (their gates predate the bound and their pinned points satisfy
  it).

## 17. Fully analytic SC-VV10 Hessian (ωB97X-V / ωB97M-V)

The fixed-density VV10 energy is a closed form in the grid variables,
E = F({w_i}, {ρ_i, σ_i}, {R²_ij}) = βΣ_i w_iρ_i + ½Σ_ij w_iw_j e_ij with
e_ij = ρ_iρ_jΦ; every kernel partial (first and second, R² included)
comes from ONE nested 5-variable dual evaluation of the pair energy —
no hand-derived Φ curvature formulas anywhere.

- **f_nlc response** (`nlc::vv10_fxc_contraction`): directional second
  functional derivative — outer duals carry the (ρ_i, σ_i) rows, an
  inner single-slot dual carries the Δ direction seeded on BOTH pair
  sides, so same-point and cross-point curvature come out together.
  Lives in `ResponseSolver::fxc_contraction` (state: params + dedicated
  coarse grid + P_tot built from `ScfOptions::vv10`), so CPKS,
  polarizability AND TDDFT responses of the -V functionals now carry the
  exact nonlocal coupling. FD-gated against dV_nlc under P ± hΔP.
- **Skeleton + D_a V_nlc** (`hessian/nlc_terms.rs`): the local part rides
  the shared GGA machinery — `xc_terms::xc_hessian_terms_with` now takes
  an injected per-batch kernel, and VV10 supplies its EFFECTIVE kernel
  (ε^eff_i = ∂F/∂w_i = ρ_iP1_i; vρ_i, vσ_i; same-slot curvatures
  Σ_l w_l∂₁²e) spin-mapped onto the 5-channel GGA layout (vσ → (v,2v,v),
  v²σ² → (1,2,1,4,2,1)·c). The nonlocal Gram corrections are per-point
  M-contractions DEW/DVRHO/DVSIG[m×3N] (O(m²·3N)) built from the pair
  partials and the fixed-P first derivatives (Dw, Dρ, Dσ — same
  owner-riding convention as §"XC Hessian"), plus the sparse kernel-motion
  R² rows/columns and ½e_{R²}DDR² (only pairs with different owner
  atoms). The same M-vectors assemble the D_aV_nlc potential-derivative
  matrices (identical for both spins), completing F^{(a),skel}.
- Self-pair bookkeeping: the ordered-pair convention keeps the ∂₁∂₂
  self-terms in the CROSS loop (pair (i,i)) and strictly ∂₁² sums in the
  local curvature — together exact (the ζ=½ΣΣ formalism resolves the
  slot double-counting identically to the gradient's "net factor 2").
- Gates: skeleton vs FD of the analytic fixed-P VV10 gradient and
  D_aV_nlc vs FD of the potential under displaced geometries (rebuilt
  basis/grid) ≤5e-6; end-to-end ωB97X-V+SC-VV10 analytic vs numerical
  Hessian on water ≤2e-6 (tests/p8).

## 18. SAPT stage 1 (DFT-SAPT / SAPT(KS) style)

`src/sapt.rs` + CLI `sapt file.xyz --natoms-a N`. Closed- AND open-shell monomers (`--mult-a`/`--mult-b`; every stage is spin-resolved),
monomer-centered basis; monomer SCFs with any registered functional (HF
or KS — hybrid/RSH/meta/SC-VV10 all through the shared machinery).

- **E_elst^(1)**: exact static interaction
  Tr[P^A v^B] + Tr[P^B v^A] + Tr[P^A J[P^B]] + E_nn^{AB} in the dimer AO
  space. Gated against a frozen PySCF-integral oracle (independent
  stack) ≤1e-9.
- **E_exch^(1)**: NO S² truncation — Heitler-London through the Löwdin
  nonorthogonal-determinant density D = 2C(CᵀSC)⁻¹Cᵀ over the union of
  occupied monomer orbitals; E_exch = E^{HF-expr}[D] − E^{HF-expr}[A] −
  E^{HF-expr}[B] − E_elst (HF interaction operator on the KS/HF
  determinants — the standard SAPT(DFT) hybrid convention). Being a
  Slater-rule evaluation it carries no transcribed S² formula; gated by
  A↔B symmetry and exponential decay.
- **E_ind^(2)** (coupled): the partner's full electrostatic potential
  ω^Y = v^Y + J[P^Y] on the monomer AO block, solved with the shared
  CPKS `ResponseSolver` (so XC kernels and f_nlc apply);
  E = ½Tr[ΔP·ω]. Gated by the λ-scaled `extra_hcore` finite-difference
  identity (quadratic energy coefficient) ≤5e-7 — pinning both the
  factor and the solver sign conventions.
- **E_disp^(2)** (uncoupled): −4Σ_{ia∈A,jb∈B}(ia|jb)²/(ε_a−ε_i+ε_b−ε_j)
  from monomer orbitals with mixed-block dimer MO integrals. Gated by
  A↔B symmetry, attractiveness, and the ~R⁻⁶ decay of the C₆ estimate.
- **Supermolecular cross-check + delta** (stage 2, first term): setting
  `Atom.z = 0` keeps a centre's basis functions while removing its nucleus
  and electrons, so the dimer and both monomers are solved in ONE dimer
  basis and BSSE cancels exactly (counterpoise). `delta` = supermolecular
  minus the SAPT sum collects what the truncated expansion misses. For the
  STO-3G water dimer: SAPT sum -1.5637, supermolecular -1.3211, delta
  +0.2426 kcal/mol — POSITIVE, the correct sign, since exchange-induction
  quenches induction; 15 % of the total, i.e. a correction rather than the
  answer. No formula transcription: existing SCF and MolBasis only.
- **Coupled dispersion** (stage 2): a sum over the monomers' TDA spectra,
  `-4 sum_nm V_nm^2/(w_n+w_m)` with `V_nm` the Coulomb coupling of the two
  AO transition densities. Driving the SAME routine with the uncoupled
  excitations reproduces the closed-form `e_disp_unc` to 1e-8, which
  CALIBRATES the spin factor instead of guessing it. Both amplitude sets
  are orthonormal, so the transition weight is invariant and coupling acts
  only through the denominators; singlet CIS energies sit below the
  orbital gaps (the -J electron-hole attraction), so the coupled value is
  the larger one (1.40x for the STO-3G water dimer).
- **Exchange-induction and exchange-dispersion** (stage 2), both without
  transcribing an S² expression: exchange-induction re-runs the
  Heitler-London construction on the POLARIZED occupieds (recovered from
  the CPKS response as `U = C_v^T S dP S C_o / 2`); exchange-dispersion
  projects each monomer's transition density out of the partner's
  occupied space with `1 - rho_A S_ab rho_B S_ba` and re-runs the same sum
  over states. Both are therefore exact in the overlap rather than
  truncated at S². Gated by sign (both repulsive, quenching what they
  correct), magnitude, and collapse at long range.
- Staged: open-shell monomers. Policy: prefer formulations that AVOID
  transcribing published S² expressions, as all of the above do.
- `ScfOptions::extra_hcore` (general embedding hook) was added for the
  induction gate and doubles as the entry point for the upcoming
  C-PCM/SMD reaction-field work.

## 18b. Where the analytic Hessian actually spends its time

Stage timings on benzene/STO-3G/SVWN5 (grid level 2), read off the progress
log rather than guessed:

| stage | time |
| --- | --- |
| ERI second derivatives | 10.5 s |
| **XC skeleton + derivative potentials** | **318 s** |
| ERI first-derivative Fock | 2.6 s |
| CPKS block solve | 13.3 s |
| CPKS post-processing | 20 s |

This overturned the working hypothesis. Three earlier optimizations —
restricting the `yres` accumulation, shell-pair density screening in the
second-derivative contraction, and batching the CPKS right-hand sides —
each moved the total by a few percent, which was read as "the direct-mode
J/K rebuild inside CPKS dominates". Forcing in-core ERIs for the CPKS did
help (524 s -> 396 s), but the measurement above shows the CPKS solve is
13 s: the real cost was never there.

And the XC stage's own dominant cost turned out to be neither of the two
loops fixed below, but `becke_weight_second`: disabling it took benzene
from 356 s to 106 s, i.e. it was ~70 % of the whole analytic Hessian. It
was being called once per ATOM PAIR per grid point, each call re-evaluating
the entire `O(natoms^2)` weight expression in nested duals — quartic in the
atom count. Two facts fix it:

- with the grid point NOT seeded, a cell factor `s_bc` carries a derivative
  only when `b` or `c` is one of the two seeded centres — about `4 natoms`
  of the `natoms^2` factors. The rest are evaluated in plain f64 and folded
  in as constants, so the per-pair nested-dual work drops to `O(natoms)`;
- the point has to be seeded exactly when one of the two centres owns it,
  which is the case that would destroy that sparsity. Translational
  invariance removes it: `sum_A D_A w = 0` gives the owner row from the
  others, and summing that gives the owner-owner block.

Measured result (STO-3G/SVWN5, grid level 2, `tools/bench_hessian.ps1`):

| system | analytic | numerical | ratio |
| --- | --- | --- | --- |
| benzene (12 atoms) | 162 s | 145 s | 1.12 |
| naphthalene (18 atoms) | 934 s | 883 s | 1.06 |

That is 3.7x slower -> 1.06x over the course of this work, and the ratio
still improves with size. **The Phase 8 goal — analytic strictly faster
than numerical from about 15 atoms — is therefore NOT yet met**; the
crossover, extrapolating the two points above, sits somewhere past 25
atoms. Recorded as measured rather than declared reached.

The XC stage builds `D_a V_xc` for all `3N` displacements and both spins.
Written naively that is a full `nao^2` sweep per (spin, atom, axis) at
EVERY grid point — 3.4 billion inner iterations for twelve atoms. Two
structural facts collapse it:

- `(Dw) vrho phi phi + w (Dvrho) phi phi` is a SCALAR times the same
  rank-one `phi phi^T` for every (spin, atom, axis), so it becomes one
  weighted Gram matmul per batch instead of a per-point sweep;
- everything else carries `c^A_mu = delta_{A,owner} - delta_{A,atom(mu)}`,
  which vanishes unless `mu` sits on `at` (or, when `at` owns the point,
  unless it does not) — and the `nu`-derivative term is the exact transpose
  of the `mu`-derivative one, so one computed value fills both entries.

Also fixed: seven scratch buffers were heap-allocated per grid point, one
of them `natoms^2 x 72` doubles (~83 kB per point at twelve atoms) and
allocated even for LDA, where it is never read.

## 18c. The SCF chooses in-core ERIs on its own

Direct mode recomputes the whole screened ERI list every iteration. That is
the right default for large systems and the wrong one for everything that
would have fitted in memory anyway — benzene/6-31G/B3LYP is **21.6 s direct
against 5.5 s in-core**, a factor of four, and the same 4x had already shown
up inside the analytic Hessian's CPKS.

So `run_scf` now estimates the tensor and switches itself when it fits a
modest budget (2 GiB), leaving an explicit `--incore` or a density-fitted
run alone. The estimate budgets `n^4` doubles, not `n^4/8`: the eight-fold
permutational symmetry is exploited when FILLING the tensor, not in how it
is stored — a first attempt used `n^4/8` and the engine promptly refused its
own request. Measured after the change: 6.6 s, with the energy unchanged to
1e-12.

The crossover sits near 120 basis functions, above which direct takes over
again.

## 18d. SCF convergence: ADIIS -> CDIIS -> SOSCF

Three accelerators, each covering the region where the others are weak.

- **ADIIS** far from convergence. CDIIS drives the RESIDUAL to zero, which
  is right at the solution and unreliable away from it — the extrapolated
  Fock can be worse than any of its inputs. ADIIS instead minimizes a
  variational energy estimate over the CONVEX HULL of the stored densities,
  so every coefficient is non-negative and the result is an interpolation.
  The constraints `c_i >= 0`, `sum c_i = 1` are imposed by the substitution
  `c_i = t_i^2 / sum_j t_j^2`, which satisfies both identically — no active
  set, and no way to leave the simplex by accident.
- **CDIIS** once the error drops below `adiis_until_error` (0.1).
- **SOSCF** below `soscf_from_error` (1e-2): a Newton step in the orbital
  rotations, with the Hessian-vector product built from the SCF's OWN
  `JkEngine`. Going through `ResponseSolver` would rebuild the ERI tensor
  on every step.

Two bugs here were caught only by measuring iteration counts, and both
would have looked plausible in review:

1. **Placement.** The step was first taken AFTER the diagonalization,
   pairing the new orbitals with the previous density's Fock. That is not a
   Newton step at all, and it made benzene worse — HF 11 -> 17 iterations.
   Taken before the diagonalization, `focks` and `mo_coeff` belong to the
   same density and no extra Fock build is needed.
2. **Spin factor.** `delta_density` carried an occupation factor of two for
   the restricted case, which is right for the TOTAL density that J sees
   and wrong for the SPIN density that K sees — so the exchange half of the
   Hessian came out twice too large. Removing it (the caller sums the two
   channels, which supplies the factor for J automatically) took HF from 16
   back to 9.

Measured on benzene/6-31G, iterations to convergence:

| functional | CDIIS only | + ADIIS + SOSCF |
| --- | --- | --- |
| HF | 11 | **9** |
| B3LYP | 9 | 9 |
| M06-2X | 9 | 9 |

Easy cases were already fine; the value is in the hard ones, and in the
fact that neither accelerator can now make things worse.

Convergence thresholds follow PySCF: `conv_tol = 1e-9` on the energy,
`conv_tol_grad = sqrt(conv_tol)` on the orbital gradient (the same
commutator DIIS drives), plus a looser density-RMS guard. All three are
settable from the CLI (`--conv-tol`, `--conv-tol-grad`, `--conv-tol-dp`),
and `--conv-tol` moves the gradient threshold with it, which is what
"converge this more tightly" actually means.

## 18e. ECP second derivatives

The ECP analytic Hessian needed no new integral class. `ecp_second_contraction`
is `ecp_gradient` one order up: bra and ket derivatives from raised/lowered
shells, and the ECP CENTRE from translational invariance via the same sum
rules `deriv2` uses for the nuclear attraction —
`AC = -AA - AB`, `BC = -BA - BB`, `CC = AA + AB + BA + BB`.

One thing does differ from the nuclear-attraction case and is easy to miss:
the ECP operator is SEMI-LOCAL, so the bra and ket sides are NOT related by
a transpose. `AB` and `BA` are formed explicitly rather than one being
derived from the other. Gated on Ni(CO)4/LANL2DZ against central differences
of the (already FD-gated) ECP gradient — a metal centre with a real
semi-local potential, which is exactly where a transpose shortcut would have
been silently wrong.

### The ECP Hessian is NOT wired in, and here is exactly where it stands

Two pieces exist, each with its own passing gate:

- `ecp_second_contraction` — the skeleton term, FD-gated against
  `ecp_gradient` on Ni(CO)4/LANL2DZ;
- `ecp_derivative_matrices` — `dU_mn/dR_a`, the perturbed-core-Hamiltonian
  term CPKS needs, gated by `Tr[P dU] == ecp_gradient` component by
  component and by `sum_atoms dU/dR == 0` (translational invariance, which
  is what pins the ECP-centre term that has no integral class of its own).

Connecting both into `analytic_hessian_full` nevertheless produces a
Hessian that is WRONG BY TWO ORDERS OF MAGNITUDE. On H2S/LANL2DZ,
`H[0z,0z]` came out −126.8 against a finite-difference value of +0.43, and
the ratio is not constant across elements (−294 there, −165 elsewhere), so
it is neither a factor nor a sign.

What is now MEASURED rather than assumed, and what it rules out. Two
hypotheses were tested and both are dead:

1. *"`ecp_gradient`'s factor of two double-counts."* It sums bra-side
   derivatives over ALL ordered pairs and doubles, which looks like
   double-counting since the loop visits `(a,b)` and `(b,a)` and `U` is
   symmetric. `ecp_gradient_matches_fd_of_the_ecp_energy` settles it:
   central differences of `Tr[P U(R)]` at FIXED density — no SCF to absorb
   anything — agree to 1e-7. **The factor of two is correct**, and the
   algebra says why: the ket contribution `Tr[P_ij (B_ji)^T]` equals
   `Tr[P_ji B_ji]`, which relabels to the bra sum, so bra and ket are two
   EQUAL contributions rather than one covering the other.
2. *"`ecp_shell_pair(a,b)` is not the transpose of `ecp_shell_pair(b,a)`."*
   `ecp_shell_pair_transpose_symmetry` measures it directly: it is
   transpose-symmetric to 1e-12. (Worth having regardless — `ecp_matrix`
   only evaluates the upper triangle, so nothing before now depended on
   the `i>j` case being sane.)

With both of those established, the same algebra says `M + M^T` from the
bra blocks IS the right `dU/dR`, which is what `ecp_derivative_matrices`
builds. So the pieces look right and the assembly still does not.

**Resolved: it was `charges`.** `analytic_hessian_full` built its nuclear
terms from the bare atomic number, `mol.atoms.iter().map(|a| a.z)`, while
`assemble_gradient` had always used `mb.effective_charges(mol)`. With an
ECP those differ — sulphur enters as Z = 16 against the Z_eff = 6 the SCF
itself used — and the mismatch propagates through the nuclear-attraction
second derivative, its derivative matrices and the nuclear repulsion. That
is the ~150 Hartree/bohr². Nothing without an ECP was ever affected, since
there `Z_eff == Z`, which is why every other Hessian gate passed.

The rest of this section is the record of getting there, kept because the
route was longer than it should have been and the reasons are reusable.

A third hypothesis also died. `ecp_derivative_matrices` is now gated
ELEMENT BY ELEMENT against central differences of `ecp_matrix`
(`ecp_derivative_matrices_match_fd_of_the_ecp_matrix`) — the gate that
should have been written first, since a trace identity does not determine a
matrix, and the earlier trace-only gate is exactly what let this run so
long. It passes to 1e-6. `ecp_second_contraction` is likewise gated
element-wise against finite differences of `ecp_gradient`. **Both ECP
derivative objects are correct.**

The measurements, on one H2S geometry at HF/LANL2DZ where the
finite-difference reference is `H[0z,0z] = +0.436`:

| what is wired in | `H[0z,0z]` |
| --- | --- |
| nothing (ECP absent from the Hessian entirely) | −149.57 |
| skeleton only (`ecp_second_contraction`) | −149.39 |
| skeleton + perturbed Fock (`ecp_derivative_matrices`) | −129.14 |
| the same geometry in ALL-ELECTRON 6-31G | exact to 9.8e-9 |

That table is what solved it, and the decisive row is the last one. The
Hessian machinery is sound; the ECP integrals are each verified; yet the
assembled ECP result is wrong. The only thing left is an input that
CHANGES WHEN AN ECP IS PRESENT but is not itself an ECP integral — and
there is exactly one such quantity in the whole assembly: the nuclear
charge. Note also how little the two ECP terms move the number (0.18 and
20 out of ~150): they were never the problem, and reading that as "a term
is missing" cost several rounds.

The lesson generalizes past this bug: when a subsystem's own gates all
pass but the assembly is wrong, stop adding terms and look for a shared
INPUT that the subsystem's presence silently changes. A control run that
removes the subsystem (here: the all-electron basis) localizes it in one
measurement.

That was settled: the cause was `charges`, as recorded above, and the ECP
analytic Hessian ships. `analytic_hessian_full` calls
`ecp_second_contraction` and `ecp_derivative_matrices`, and
`tests/p11_organometallic.rs` gates it against central differences.

(This paragraph previously said the opposite — twice, duplicated verbatim —
and cited a test named `ecp_analytic_hessian_refuses_rather_than_guessing`
that does not exist anywhere in the tree. It described the state of the work
before the `charges` fix on the same page. Left here as a note because a
stale "we refuse this" is worse than no note at all: it tells a reader a
shipped, gated capability is missing.)

## 18f. GIAO NMR shielding: the derivation and its gates

`src/properties/nmr.rs` carries the one-electron operator layer,
`src/properties/giao_eri.rs` the two-electron field derivative, and
`src/properties/magnetic.rs` the CPHF and the shielding assembly. The
algebra is recorded here because it is easy to get subtly wrong and, in the
absence of a direct external formula-level oracle, a wrong formula produces a plausible number
rather than a visible failure.

### Convention

London orbitals about a gauge origin `O`, with `d_mu = R_mu - O` and
`r_O = r - O`:

```text
chi_mu^B(r) = exp(-(i/2) (B x d_mu) . r_O) chi_mu(r)
d chi_mu / dB_a |_{B=0} = -(i/2) (d_mu x r_O)_a chi_mu
h(B) = T + V + (1/2) B . L_O + (1/8) (B x r_O)^2,   L_O = r_O x p
```

Every first-order magnetic matrix is written as `i` times a REAL,
ANTISYMMETRIC matrix; the `i` is carried by the caller.

### The gauge-origin shift identity

Moving `O -> O + delta` multiplies each London orbital by the CONSTANT
phase `exp((i/2) B . (d_mu x delta))` times one common `r`-dependent phase
that the change of `A` exactly cancels. A diagonal unitary on the basis
changes no observable — that is the mechanism by which GIAOs remove the
gauge-origin dependence — and it forces an exact first-order identity on
every derivative matrix,

```text
X_a(O + delta) = X_a(O) + (gamma_n - gamma_m) X^(0),
gamma_m = (1/2) (d_m x delta)_a
```

with the four-index version `(gamma_n - gamma_m + gamma_s - gamma_l)` for
`d(mn|ls)/dB`. This is the sharpest oracle-free check available: it is
exact, cheap, and its proof consumes the kinetic commutator below, so it
fails for a dropped angular-momentum term or either phase sign.

### One-electron pieces

**Overlap.** Only the phases contribute:

```text
dS_mn/dB_a = (i/2) [ (R_m - R_n) x <m| r_O |n> ]_a
```

antisymmetric, and identically zero within one centre.

**Core Hamiltonian.** Bra phase, ket phase, and the operator's own field
dependence:

```text
dh_mn/dB_a = (i/2) [ <m| (d_m x r_O)_a h |n> - <m| h (d_n x r_O)_a |n> ]
           + (1/2) <m| L_{O,a} |n>
```

The tempting simplification — pulling `h` out and writing the first bracket
as `(R_m - R_n) x <m| r_O h |n>` — is WRONG: `h` contains the kinetic
operator, which does not commute with `r`. What their difference IS, is the
commutator

```text
<m| r_j h |n> - <m| h r_j |n> = <m| [r_j, T] |n> = <m| d_j |n>
```

which is gated directly against the (already gated) momentum integrals, and
is also the step that makes the gauge-shift identity above come out.

**Mixed `B`-`m_N` operator.** The non-response half of the shielding is
`Tr[P d^2 h/dB_a dm_{N,b}]`, and it has THREE terms, not one:

```text
(1/2) <m| (e_a x r_O) . A_N^(b) |n>                      operator x operator
+ (1/2) <m| (R_mn x r_O)_a (r_N x grad)_b / r_N^3 |n>    phase x dipole operator
- (1/2) <m| (d_n x A_N^(b))_a |n>                        momentum hitting the phase
```

Keeping only the first is the common way to get a plausible but wrong GIAO
shielding: the other two carry precisely the origin dependence that must
cancel against the response term.

### Two-electron

```text
d(mn|ls)/dB_a = (i/2) [ (R_m - R_n) x r_1O + (R_l - R_s) x r_2O ]_a (mn|ls)
```

evaluated without a new integral engine: a Cartesian factor on one index is
a shift of that index's angular momentum, `(r-O)_k chi_n = (r-R_n)_k chi_n
+ (R_n - O)_k chi_n`, so the moment-weighted quartet is the ordinary
quartet with one shell raised, left Cartesian, contracted with a transform
that maps each spherical component onto its raised Cartesian partner.

The **exchange index order** matters here in a way it never does for
ordinary ERIs. The Fock exchange is `K_mn = sum_ls P_sl (ml|sn)`; `(ml|sn)`
and `(ml|ns)` are the same integral but carry DIFFERENT phase weights, and
using the wrong one yields a symmetric `dF/dB` where an imaginary
perturbation demands an antisymmetric one.

For range-separated exchange the explicit two-electron term is
`J'_C - (cx/2)K'_C - (clr/2)K'_erf(omega)`. The weighted contraction keeps
the Erf pass exchange-only: adding a second `J'_erf` would be a different
Hamiltonian. The CPHF operator carries the matching `-(clr/2)K_erf[D]`.

### CPHF

The perturbation is imaginary and antisymmetric, with two consequences:

- the first-order density change `sum_mn D_mn chi_m chi_n` vanishes for
  antisymmetric `D`, and so do `grad rho` and conventional `tau`. `J[D] = 0`,
  and the LDA/GGA density kernels vanish. A meta-GGA is different: the
  gauge-covariant Dobson variable
  `tau_tilde_s = tau_s - |j_p,s|^2/(2 rho_s)` has a nonzero current-current
  second derivative, so even a pure meta-GGA remains coupled;
- the exchange coupling becomes `A - B` instead of `A + B`, and it does so
  BY ITSELF. Projecting the ordinary AO build `K[X]_mn = sum_ls X_ls (ml|ns)`
  into the ov block gives `(ab|ij) + (aj|ib)` for the symmetric response
  density and `(ab|ij) - (aj|ib)` for the antisymmetric one. The response
  term therefore keeps its usual `-(cx/2) K[D]`.

  Flipping that sign by hand — the tempting move, since the lore is
  "imaginary perturbations use A - B" — double-counts the change. It does
  not blow up: it produces a shielding whose residual gauge-origin
  dependence is a couple of ppm out of several hundred (0.6 % on HF/STO-3G
  water), which is exactly the size that reads as rounding. The gauge gate
  catches it; nothing else does. The consistency requirement is algebraic:
  under `O -> O + delta` the explicit two-electron derivative picks up
  `+(cx/2) sum_ls P_ls (gamma_s - gamma_l) (ml|sn)` beyond the `[K, G]`
  commutator, and that leftover is cancelled by the response to the
  density's own shift `D -> D + [P, G]` only with the unflipped sign.

Because the Coulomb term and the density kernels drop out and the density is
ANTIsymmetric, the magnetic CPHF carries its own conjugate-gradient solver
rather than going through `ResponseSolver`, whose `X + X^T` density
construction would silently give the `A + B` coupling. The equations:

```text
(eps_a - eps_i) Uhat_ai = eps_i Shat^MO_ai - Fhat^MO_ai
Uhat_ij = -(1/2) Shat^MO_ij
D = 2 (X - X^T) - 2 C_o Shat^oo C_o^T,   X = C_v Uhat C_o^T
```

### Units

In Hartree atomic units the vector potential enters as `(1/2)(p + alpha A)^2`,
so a mixed `B`-`m` second derivative carries `alpha^2`. Reported shieldings
are `alpha^2 x 1e6` times the raw integral quantity, i.e. ppm.

### Gates (tests/p15_nmr.rs and the module tests)

- gauge-origin independence of the full shielding tensor, three origins
  including one 5 Bohr off the molecule;
- the shift identity above, separately for `dS/dB`, `dh/dB`, the mixed
  `B`-`m` operator and `d(mn|ls)/dB`;
- `<m|[r_j, h]|n> = <m|d_j|n>`;
- the moment-weighted quartet computed by raising the BRA vs the KET;
- antisymmetry of every magnetic matrix, and of the assembled `dF/dB`;
- Erf bra/ket and pair symmetries, an exchange-only literal contraction,
  `Erf + Erfc = Coulomb`, finite-omega RSH origin independence, and the
  large-omega RSH-to-HF limit for energy and every dia/para tensor component;
- LDA/GGA `dVxc/dB` against a separate bra/ket complex-phase finite
  difference, exact origin-shift identities against the ordinary Vxc,
  antisymmetry, and SVWN5/PBE/B3LYP/CAM-B3LYP shieldings across three origins;
- unequal alpha/beta GGA densities, spin-swap covariance, and a direct nonzero
  gate on the origin-independent phase-gradient term that origin shifts cannot
  observe;
- free-atom limit: an s-only closed-shell atom has zero paramagnetic term
  and its shielding equals Lamb's `(alpha^2/3)<sum 1/r>` computed from an
  independent nuclear-attraction integral;
- one-centre GIAO = common gauge origin, exactly;
- C2v water: equal hydrogens, diagonal oxygen tensor, mirror-related
  hydrogen tensors;
- `dP/dB` satisfies the differentiated idempotency `D S P + P S D + P Shat P
  = 2D` and `Tr[D S] + Tr[P Shat] = 0`.

### Numbers

Gauge-origin independence: the full tensor moves by < 1e-5 ppm across
origins at (0,0,0), (0.83,-1.27,0.41) and (-4.5,3.1,-2.7) Bohr on HF/STO-3G
water, and across (0,0,0) / (3.7,-2.9,1.6) on HF/6-31G. The free-atom
shieldings match Lamb to all printed digits.

```text
He / HF / 6-31G     59.795879 ppm   (Lamb 59.795879, paramagnetic 0)
Be / HF / 6-31G    149.973082 ppm   (Lamb 149.973082, paramagnetic 0)
Ne / HF / 6-31G    552.209848 ppm   (GIAO = common origin to 1e-8 ppm)
H2O / HF / STO-3G  sigma(O) 365.678  (dia 377.714, para  -12.036)
                   sigma(H)  33.673  (both hydrogens identical)
H2O / HF / 6-31G   sigma(O) 326.836  (dia 390.076, para  -63.239)
                   sigma(H)  32.711
```

The minimal basis leaves water's oxygen with two virtual orbitals, so its
paramagnetic term is a token -12 ppm; the split-valence basis recovers a
realistic -63 ppm. Both are gauge-invariant, which is the point: the gate
tests the formalism, not the basis.

### Scope, and what is deliberately NOT implemented

**Semilocal XC.** With London orbitals `dV_xc/dB` is nonzero even though
`d rho/dB = 0`: the phase differentiates the matrix element itself,

```text
dV_xc,mn/dB_a = (i/2) [ moment-weighted XC matrix of (R_mn x r_O)_a chi_m chi_n ]
```

For LDA that moment-weighted contraction is implemented in bounded batches.
For spin-resolved GGA define
`b_alpha = 2 v_sigma_aa grad rho_alpha + v_sigma_ab grad rho_beta` and
`b_beta = 2 v_sigma_bb grad rho_beta + v_sigma_ab grad rho_alpha`. The
contraction then splits into
`M_j = int (r_O)_j [v_rho phi_m phi_n + b . grad(phi_m phi_n)]`
plus `L_j = int b_j phi_m phi_n`, with
`Vhat_a = 1/2 [(R_m-R_n) x (M+L)]_a`. `L_j` is origin-INDEPENDENT, so it is
gated by directly differentiating both the complex London AO values and their
gradients; an origin test alone would not see it. LDA and GGA are implemented.

Meta-GGA adds the conventional-tau value- and gradient-phase terms plus the
current-density correction. In this code's
`tau_s = (1/2) sum_i |grad psi_is|^2` convention the gauge-covariant variable
is `tau_tilde_s = tau_s - |j_p,s|^2/(2 rho_s)`. At a real ground state
`j_p,s=0`, so the SCF is unchanged, but magnetic response retains
`K_j[D] = int (v_tau_s/rho_s) j_s[D].J_AO`. Closed-shell total response
densities are halved before this spin kernel is applied. The London basis
current enters the explicit right-hand side through the same operator. Both
paths use bounded AO batches and cache `w v_tau/rho` once. A
conventional-tau-only trial failed the strict TPSS three-origin gate; that
gate was not relaxed and motivated the current kernel. The resulting path is
still experimental because the pinned external NMR oracle supports LDA/GGA
but not meta-GGA full tensors.

Range-separated exchange is implemented with an exchange-only Erf integral
pass. SC-VV10 remains refused because its nonlocal potential needs a separate
London-phase derivative.

**Magnetizability** `xi = -d^2E/dB_a dB_b` does NOT fall out of the first-order
response alone. Because the basis depends on `B` on BOTH sides, the second
derivative needs the explicit second-derivative integrals

```text
S^(ab) = -(1/4) <m| (R_mn x r_O)_a (R_mn x r_O)_b |n>
h^(ab): <m| r_i r_j h |n> (bra raised twice), <m| r_i h r_j |n> (bra and
        ket each raised once), <m| r_i L_a |n>, and the (1/8)(B x r_O)^2 term
G^(ab): double-moment four-centre integrals, including the r_1 r_2 cross
        moments, which do NOT vanish in the exchange contraction
```

plus the first-order energy-weighted density `W^(a)`. Every one of them is
mechanical with the raise-the-index trick above, but none of them exists
yet, so the magnetizability is not implemented rather than approximated.

**Cotton-Mouton** additionally needs the hypermagnetizability, a mixed
electric/magnetic fourth-order response; not implemented.

Electronic circular dichroism does NOT depend on any of this: its rotatory
strengths need only the electric and magnetic TRANSITION moments, which the
angular-momentum and momentum operators supply, and they are implemented
and gated (`src/response/tddft.rs`).

## 18g. Where the analytic GRADIENT spends its time, and what 1/10 needs

Benzene/6-31G/B3LYP, read off the progress log:

| stage | time |
| --- | --- |
| SCF (7 iterations, in-core, includes the grid build) | 4.75 s |
| gradient: one-electron (T, V, S) | 0.03 s |
| **gradient: ERI derivative** | **10.34 s** |
| gradient: XC grid, full response | 0.38 s |

So the ERI derivative is 96% of the gradient and 63% of the whole run.
Nothing else is worth touching until it moves.

Two cheap fixes were applied first, because they were obvious once
measured: the degeneracy factor was being counted by building a `HashSet`
per quartet (one heap allocation and eight hashes for each of ~10^5
quartets, for a number that is three comparisons), and the density
screening used the GLOBAL maximum density element rather than per-shell-pair
block maxima. Together: 10.34 → 9.96 s. **3.7%.** Benzene is compact and
conjugated, so almost nothing screens out — the lesson is that the
cost is the integral evaluation itself, not the bookkeeping around it.

A third idea was tried and REJECTED by measurement, which is worth
recording so it is not tried again: chunking the rayon loop over quartets
(`par_chunks` instead of `par_iter`) to save ~10^5 small per-quartet
allocations. It made things WORSE — 9.96 s to 10.83 s. Quartet cost varies
by orders of magnitude with the angular momenta involved, so work stealing
across many small tasks is worth more than the allocations cost. The rest
of the derivative code chunks because its per-pair cost is uniform; this
loop's is not.

**Sharing the R-tensor was then implemented, and it works but under-
delivers.** Raising or lowering a shell changes its angular momentum but
not its exponents or centre, so `p`, `q`, `alpha` and `P-Q` — hence the
Boys values and `R_tuv` — are identical across all six substituted
quartets; one build at `L+1` serves every variant. The E-tables collapse
the same way, since `ETable::build(i,j,..)` already covers every order up
to `(i,j)`: one bra table at `(la+1, lb+1)` and one ket table at
`(lc+1, ld)` replace thirty-six builds with six. Measured on the same
benzene case: **9.96 s → 8.82 s, 1.15x**, against an expectation of
1.6-1.8x.

That shortfall is itself the useful result: it says the Boys evaluation
and the R-tensor build are NOT where the time goes. What remains is the
contraction — the `G[t,u,v]` middle step and the bra dot product — which
is run six times over six different Cartesian component sets and cannot be
shared, only avoided.

What would actually move it, in order of expected return:

1. **Differentiate at the E-coefficient level.** Today each canonical
   quartet is evaluated up to SIX times: raised and lowered shells for each
   of three centres, each a full `eri_quartet_masked`. In McMurchie-Davidson
   the derivative can instead be folded into the Hermite expansion
   coefficients (`d/dA = 2a E_{i+1} - i E_{i-1}`), giving ONE quartet
   evaluation with modified E-tables. Expected 4-6x on this stage. The
   recursion is in the plan and already used for one-electron derivatives;
   it has never been applied to the ERI path.
2. **Share the R-tensor across the six evaluations.** Raising a shell
   changes the angular momentum but NOT the exponents or the centres, so
   `p`, `q`, `P`, `Q` and hence the Boys values and `R_tuv` are identical;
   only the maximum order differs by one. Building `R` once at `L+1` and
   contracting all six against it removes five Boys+R builds. Expected
   1.6-1.8x, and it is a much smaller change than (1) — a reasonable first
   step if (1) looks too invasive.

The 1/10 target is not reachable by tuning; it needs (1).

## 19. Canonical-quartet Hessian integrals (performance)

The SCF J/K build was canonical from the start, but the Hessian path kept
full ORDERED quartet loops ("correctness over speed"). Both hot routines
are now canonical (a<=b, c<=d, (a,b)<=(c,d)) with an explicit permutation
scatter, and the original ordered loops live on as `#[cfg(test)]`
references gated element-for-element against the fast paths (the project
policy of keeping a naive implementation in the tree).

- Mapping: with `T[k] = S[SIGMA[g][k]]` the canonical element indices
  satisfy `i[SIGMA[g][k]] = j[k]`, and the derivative block of T-slot
  `k` is the canonical block of slot `SIGMA[g][k]`. Enumerating the
  DISTINCT `T` tuples visits exactly the ordered loop's set once each, so
  the transformation is an identity, not an approximation. Screening is
  permutation invariant (Schwarz is symmetric), so filtering canonically
  selects the same quartets.
- `deriv_mat::eri_derivative_fock`: derivative blocks computed once per
  unique quartet. Measured 5.6x (98 ms vs 545 ms, water/STO-3G).
- `deriv2::eri_second_contraction`: additionally caches the SPHERICAL
  mixed second-derivative tensors per canonical quartet (six slot pairs x
  nine axis pairs), so each permutation only pays a weighted element sum
  — the Cartesian->spherical transformation is no longer repeated per
  ordered quartet. Measured 12.7x (188 ms vs 2.40 s). T-slot pair
  `(ti,tj)` maps to the sorted canonical pair with the axis roles swapped
  when the order flips.
- Gates: exact equivalence vs the ordered references (1e-12 / 1e-10) plus
  the unchanged end-to-end analytic-vs-numerical Hessian gates.
- **VV10 Hessian pair loop** (`hessian/nlc_terms.rs`): the Gram build ran
  over all 3N displacement slots per point pair, i.e. O(m^2 * 3N) with a
  per-pair cost that GREW with the molecule. Two identities fix that
  without changing a single number: (a) the fixed-P first derivatives
  `Dw/Drho/Dsigma` are exactly zero for atoms a grid point cannot feel, so
  a precomputed live-slot list per point makes the inner loop O(nnz) —
  bounded by locality, not by system size; (b) the R^2 row is INDEPENDENT
  of the axis (only its `dr2` prefactor is not), so it is built once per
  pair instead of three times, over the union of the two points' live
  slots plus the two owner atoms. FD-gated unchanged, and the end-to-end
  ωB97X-V analytic-vs-numerical Hessian still passes.
- **Measured effect on the analytic-Hessian gate suite** (10 tests, HF
  through ωB97M-V + SC-VV10): **10630 s -> 1240 s, 8.6x**, all passing.
  The canonical-quartet integrals, the O(nnz) VV10 pair loop and running
  the derivative-consistency gates on a coarse NLC grid each contribute;
  none of them changes a single gated number.
- **Phase 8 performance goal is NOT met yet** (`tools/bench_hessian.ps1`):
  benzene, 12 atoms, SVWN5/STO-3G — analytic CPKS Hessian **530 s** vs
  numerical (central differences of the analytic gradient) **144 s**, so
  the analytic path is still 3.7x SLOWER where the plan wants it faster
  from about 15 atoms. The numerical route needs 6N+1 gradients at ~2 s
  each, which is hard to beat while the analytic route pays the full
  skeleton second-derivative contraction plus 3N CPKS right-hand sides.
  Next targets, in the order the profile suggests: the ERI second
  contraction (still 24 integral blocks per canonical quartet), batching
  the 3N CPKS solves so the J/K builds are shared, and the XC grid terms.
- Also: the SC-VV10 Hessian gates now run their NLC quadrature at level 1
  — analytic and numerical sides share the grid, so derivative
  consistency is grid independent while the O(m^2) kernel (paid 18x by
  the numerical side) gets cheap. Absolute VV10 accuracy stays gated
  against PySCF in p9.

## 20. Continuum solvation: C-PCM and SMD

`src/pcm.rs`, self-consistent through `ScfOptions::pcm`; CLI
`--solvent water|smd-water|cosmo-water [--eps X]`.

- **Cavity**: one sphere per atom at `vdw_scale x` the (H-modified) Bondi
  radius, tessellated by a self-generated golden-spiral point set.
  A cavity discretization is a PARTITION into comparable patches, not a
  quadrature rule: each patch carries a Gaussian charge scaled to its own
  area, so two nearly coincident points make the charge-charge matrix
  singular. The Gauss-Legendre x uniform-phi product rule used elsewhere
  in the code clusters severely near the poles and DID produce a singular
  S — the quasi-uniform construction is what makes the model well posed
  (and it needs no tabulated Lebedev data).
- **Smooth burial**: the SWIG switching function of Lange & Herbert
  (JCP 133, 244111 (2010)), `h(d) = d^3(10 - 15d + 6d^2)` on
  `d = (r_iJ - R_in_J)/R_sw_J`, so the energy is C-infinity in the nuclear
  coordinates — a prerequisite for the analytic derivative chain.
- **Gaussian surface charges**: each point carries a unit-norm Gaussian of
  exponent `zeta_i^2`, hence `S_ij = erf(zeta_ij r)/r`,
  `S_ii = zeta_i (2/pi)^{1/2}/F_i`. The nuclear potential is the closed
  form `erf(zeta d)/d` (evaluating it through the ERI engine with two
  zero-exponent shells divides by zero — the first bug found), and the
  electronic one is the three-center integral `(mu nu | g_i)`, obtained by
  handing a directly-constructed `MolShell` to the existing RI-style
  three-center path (`eri_3c_shell`).
- **C-PCM**: `q = -f(eps) S^{-1} v`, `E = 1/2 q.v`, Fock contribution
  `-sum_i q_i (mu nu|g_i)`, rebuilt every SCF iteration so the total is
  variational.
- **SMD**: C-PCM electrostatics on SMD intrinsic Coulomb radii plus the
  non-electrostatic `G_CDS = sum_A sigma_A SASA_A` (Marenich, Cramer &
  Truhlar, JPC B 113, 6378 (2009)). The tensions are MEASURED off the
  reference implementation, not transcribed: a single isolated atom has
  no pairwise partner, so `sigma_Z = G_CDS/(4 pi (R_Z+0.4)^2)` reads
  straight off, and scanning a diatomic separation exposes the pairwise
  term as the excess over `sum_Z sigma_Z A_Z`. That mattered twice.
  A first draft carrying published-looking values (`sigma_O = -148.50`)
  produced aqueous G_CDS with the WRONG SIGN, whereas the reference has
  `sigma_O = 0` and no O-H pair term. And the first MEASUREMENT divided by
  areas built from the intrinsic Coulomb radii, so C, F, Si, S, Cl and Br
  came out scaled by the wrong factor — invisible to a gate covering only
  H, O, N and P, whose two radius sets coincide. Everything was re-measured
  against `smd_cds_radius`, and the gate now spans F2/Cl2/Br2/S2/ClF/BrCl/
  SF2/SiF4/CF4/CCl4 as well. Six aqueous pairs carry a distance-dependent
  term (H-C, C-C, C-N, C-O, N-O, O-P). The measured H-C, C-C and O-P
  coefficients are implemented; C-N, C-O and N-O remain typed refusals.
  The other 49, including every halogen, S, Si and P combination, are exact.
- **SASA**: latitude quadrature with an ANALYTIC azimuthal sweep — each
  neighbour cuts a latitude ring in a closed-form arc, so only the
  cos(theta) integration is discrete. Point sampling leaves an
  O(N^{-1/2}) boundary error that stalls around 0.02 kcal/mol; this
  reaches 0.001 kcal/mol against the reference.
- **Gates**: the discretization-FREE Born limit `-f Z^2/(2R)` and the
  Gauss-law sum `sum_i q_i = -f Z` (unit tests); the eps -> 1 vacuum limit
  reproducing the gas phase exactly and monotonic deepening with eps; the
  solvation shift vs PySCF's C-PCM (different tessellation, so gated at
  3e-4 Ha on a -3.85 kcal/mol shift); SASA sphere-area exactness and the
  burial monotonicity of `G_CDS`.
- **Solvent table**: `src/data/solvents.csv` bundles 179 solvents with
  their dielectric constant, refractive index, Abraham alpha/beta,
  macroscopic surface tension and the aromaticity/halogenicity fractions
  (Minnesota Solvent Descriptor Database, the descriptor set SMD is built
  on; routed via PySCF's Apache-2.0 copy exactly as the project already
  sources its D3 parameters via MOPAC). `--solvent NAME` looks a row up
  (case/separator insensitive, with the usual short names DMSO/THF/DCM
  /MeCN/...), and the descriptors can equally be supplied DIRECTLY
  (`--eps`, `--refractive-index`, `--solvent-alpha/-beta/-gamma`) with or
  without a name, so unlisted solvents need no table edit. `--eps` alone
  is enough for the electrostatic models.
- **Analytic C-PCM gradient**: differentiating `S q = -f v` and substituting
  `S^{-1} v = -q/f` collapses `dE/dR` to `q.(dv/dR) + 1/(2f) q.(dS/dR) q` —
  no second linear solve and no `S^{-1}` derivative. All four pieces are
  closed forms: `dS/dR` from `erf(zeta r)/r` plus the switching values
  `F_i` (an early FD check exposed the missing `F_i` term as a
  four-orders-of-magnitude discrepancy), `dv^nuc/dR` directly, and
  `dv^el/dR` from the raised/lowered shell identity applied to the bra with
  the surface Gaussian as the auxiliary shell — translational invariance
  then supplies the point motion for free. Only the bra is differentiated:
  the ordered pair loop visits every pair both ways and P is symmetric, so
  one factor of two covers the ket (FD caught this as an exact 2x).
  Gated piecewise against FD and end-to-end against central differences of
  the SCF energy.
- **Non-aqueous SMD**: determined by REGRESSION against the reference, not
  transcription. Two things had to be measured first. (1) The CDS surface
  uses BONDI radii, not the intrinsic Coulomb radii the electrostatics
  uses — read off by sweeping a hydrogen probe towards each element and
  bisecting the separation at which `G_CDS` leaves its asymptote, which is
  exactly `R_Z + R_H + 0.8`. Using the Coulomb set here is silently wrong
  for N, P, C, Si, S, Cl and Br. (2) With those radii, an isolated atom's
  `G_CDS/area` fitted against every descriptor column over all 179
  non-aqueous solvents leaves EXACTLY zero residual, which pins the
  coefficients, identifies the live refractive-index column, and shows
  `eps` does not enter. The molecular block came back identical for all ten
  elements — `0.35 gamma - 4.19 phi^2 - 6.68 psi^2` — which is what
  identifies it as shared rather than ten coincidences. Folding it into the
  atomic tensions is exact, since both multiply the same per-atom area.
  Seven pairs (H-C, H-O, C-C, C-N, C-O, N-O, O-O) carry a distance-dependent
  term and are refused; the other 48 agree to under a quarter of a cal/mol.
- **Analytic C-PCM Hessian**: writing the polarization energy as the
  stationary value of `G(q, R) = -(1/2f) q^T S q - q.v` makes the structure
  explicit. Stationarity kills the first-order response but not the second,
  so the fixed-density part is
  `(1/2f) q^T S_ab q + q.v_ab - f u_a^T S^{-1} u_b` with
  `u_a = v_a + (1/f) S_a q` — manifestly symmetric, and needing one solve
  per coordinate against the matrix that is already built. `S_ab` splits
  into the point-pair term (closed form) and the switching diagonal
  (nested duals over `1/F_i`, so no hand derivation); `v^nuc_ab` is closed
  form.
- **Surface charges as charge sites**: `v^el_ab` and `dV_pol/dR` need the
  second and first derivatives of the three-centre `(mu nu|g_i)` with
  respect to BOTH orbital centres and the moving surface point. Rather than
  a new integral class, the nuclear-attraction derivative machinery was
  generalized from "one point charge per atom" to arbitrary CHARGE SITES
  (`deriv2::charge_site_second_contraction`,
  `deriv_mat::charge_site_derivative_matrices`): a site supplies a Cartesian
  block and the atom it rides on, and the existing sum rules
  (`AC = -AA - AB`, `CC = AA + AB + BA + BB`) then give the site-centre
  derivatives for free. Point nuclei became the special case, so both paths
  share one implementation and one set of gates.
- **What the end-to-end gate caught**: with every piece individually
  FD-clean, the full solvated Hessian was still 1.5 % off. The cause was in
  the Hessian's own `G[dP]` helper, which applied the response kernel only
  `if has_xc` — correct until a reaction field existed without any XC part
  (HF in solvent). Piecewise gates cannot see that class of defect; only the
  end-to-end one can.
- **IEF-PCM / SS(V)PE**: `K = (I - (f/2pi) D A) S`, `R = -f (I - (1/2pi) D A)`,
  `q = K^{-1} R v`, with `f = (eps-1)/(eps+1)`. The D matrix is the normal
  derivative, at the SOURCE point, of the potential its Gaussian charge
  produces at the field point,
  `D_ij = -s'(r) (n_j . r_hat)`, which reduces to the textbook
  `(n_j . (p_i - p_j))/r^3` as `zeta -> infinity`; the diagonal is the
  curved-patch self term `-zeta_i (2/pi)^{1/2}/(2 R_i)`. SS(V)PE symmetrizes
  `D A S`, restoring the symmetry C-PCM has for free.
- **How the D matrix is gated**: comparing absolute solvation shifts would
  prove nothing — IEF-PCM and C-PCM differ by 3.7e-5 Ha on water, an order
  of magnitude BELOW our discretization gap against the reference, so such a
  gate passes with the D matrix deleted. The gate is therefore on the
  model-to-model DIFFERENCE, where the shared discretization error largely
  cancels and D is the entire signal.
- **IEF derivatives by the ADJOINT method**: `K = S` is what lets the
  C-PCM gradient collapse for free; the IEF family recovers the same
  structure explicitly. With `M = K^{-1} R` and `E = 1/2 v^T M v`,
  `dM = K^{-1}[dR - (dK) M]` gives
  `dE/dR_a = q_bar . v_a + 1/2 lambda^T (dR_a v - dK_a q)` with
  `q_bar = (q + R^T lambda)/2` and `lambda = K^{-T} v` — ONE transposed
  solve covers every coordinate. Differentiating once more gives the
  Hessian, needing `q_b` and `lambda_b` (one solve each) plus the second
  derivatives of `S`, `D` and the patch areas. Those are never materialized
  as `3N x 3N` families of `m x m` matrices: only their action on a vector
  is needed, and every one of them comes from nested duals over the closed
  forms, so `s''` and the `n_j . d_hat` chain were never written out by
  hand. `n_j` is invariant under the rigid motion of its owner, which is
  what reduces `d2D` to a 3x3 block per point pair.
- **What the D-Hessian gate caught**: two restriction bugs, both invisible
  to a coarser test. The switching factor `F_i` depends on EVERY atom, so
  the diagonal `S_ab` and `A_ab` contributions run over all points, not
  just those owned by the two differentiated atoms; and for a DIAGONAL
  block (`a == b`) the pair terms reach partners on any third atom, so the
  `{a, b}` index restriction that is right off-diagonal is wrong there.
- **SS(V)PE derivatives come for free from one identity**:
  `K_ssvpe = sym(K_ief)`, and every assembly above is LINEAR in `K`, so the
  whole model reduces to symmetrizing each application — `k_a`, `k_a_t` and
  the `K_ab` contraction. No second derivation was needed. The gate caught
  the one place the symmetrization had been missed (the `K_a q_b` crossing,
  which had been expanded inline instead of going through the helper), and
  nothing else.
- **SMD pairwise shape**: `T(R) = exp(d/(R - R0))` for `R < R0`, zero
  beyond. Identified, not transcribed: with the CDS radii and atomic
  tensions fixed, each pair's excess over the atomic sum was fitted by
  scanning `(R0, d)` with the two linear coefficients solved exactly at
  every node. Four pairs came back at 0.003 %-1 % residual and,
  independently, the SAME exponent `d = 0.300` — that agreement is what
  identifies the shape rather than merely fitting it. Three of them
  (C-C, H-C, O-P) also have an unambiguous SPLIT between the two atoms
  (C-C symmetric by construction; the other two returned one coefficient at
  zero to four decimals) and are implemented. N-O fits its own scan to about
  a percent but with two very different coefficient pairs depending on the
  exponent, so its individual coefficients are undetermined and it stays
  refused; C-N and C-O do not fit this shape at all yet (33 % residual).
  The implemented three unlock hydrocarbons, gated on
  CH4/C2H6/C2H4/C2H2/C3H8/benzene/cyclopropane/PH3O — molecules where one
  atom has several partners at very different area ratios, which is what
  tests the split rather than just the diatomic curve.
- **Element coverage**: the hydrogen-probe sweep was run over every element
  up to RADIUM, so the CDS radius table reaches far past the nine elements
  that carry an atomic tension — the rest still need a radius because the
  shared molecular term multiplies their exposed area too. Across 26
  elements, 345 of 351 aqueous pairs and 344 of 351 non-aqueous pairs carry
  no distance-dependent term at all.

## 21. TDA excited-state gradients (Z-vector)

`response/tddft_grad.rs`. Gradient of the TOTAL excited-state energy
`E_SCF + ω` (nuclear repulsion included), for a closed-shell SINGLET TDA
state over an HF/CIS or gas-phase pure-LDA reference. The grid-aware path
currently admits SVWN5/LDA; GGA, meta-GGA, hybrid/RSH, nonlocal correlation,
PCM, triplet, full-RPA, and open-shell gradients remain typed refusals.

- **Why a Z-vector at all.** Differentiating ω leaves the orbital-rotation
  derivatives `U^x_ai` standing. Solving for those directly is one CPHF
  system per nuclear coordinate — `3N` of them, each as expensive as the
  whole CPKS Hessian's per-perturbation cost. The Handy–Schaefer
  interchange moves the contraction so a SINGLE solve, with a right-hand
  side built from the excited-state densities, serves every coordinate.
  Without it the analytic route is not cheaper than finite differences.
- **The one operator, twice.** The Z-vector equation uses exactly the
  closed-shell orbital Hessian `A z = Δε z + 4(ai|bj)z − (ab|ij)z −
  (aj|ib)z` that `ResponseSolver::solve_ov` already applies for the
  analytic Hessian and the polarizability. No second CPHF implementation
  exists, so the two cannot drift apart.
- **Densities.** With the TDA amplitudes `X` (`Σ X² = 1` over spatial
  pairs), the spin-adapted singlet scale is `xs = X/√2`; then
  `dvv = 2 Σ_i xs_ai xs_bi`, `doo = −2 Σ_a xs_ai xs_aj`,
  `T = C_v xs C_oᵀ`, `D_zoo = C_o doo C_oᵀ + C_v dvv C_vᵀ`. Trace check:
  `Tr dvv = −Tr doo = 1`, one electron promoted. `D_zoo` is the change on
  excitation at FROZEN orbitals; `z1` adds the relaxation; `T` enters
  quadratically through the same kernel that built the A matrix. In TDA
  both `X+Y` and `X−Y` collapse to `X`, which is where the factor 2 in
  `dvv`/`doo` comes from — and, for the same reason, why `T` contributes
  through its symmetric part (full singlet kernel) AND its antisymmetric
  part (exchange only: `J` of an antisymmetric density vanishes).
- **The antisymmetric half is where index order becomes load-bearing.**
  `veff0mop` (from `T + Tᵀ`) is symmetric, so writing `v[k,i]` where the
  formula says `v[i,k]` is invisible. `veff0mom` (from `T − Tᵀ`) is
  ANTIsymmetric, so the same slip is a sign flip. Every `wvo` and
  Lagrangian term below carries both, and both index orders were fixed
  against the same formula rather than against whichever one "looked
  symmetric".
- **Derivative-integral bookkeeping — the one thing to get right when
  reading any published formulation.** This repo's
  `overlap/kinetic/nuclear_derivative_matrices` and `eri_derivative_fock`
  return the FULL derivative of the assembled matrix: every centre of
  every index, density held fixed (gated element-by-element against finite
  differences in `deriv_mat.rs`). So ONE Frobenius contraction
  `X · d(V[P])/dR` is already the complete derivative of the bilinear form
  `Σ (mn|ls) X_mn P_ls`. Formulations that differentiate only the first
  index need TWO contractions with the roles of `X` and `P` swapped to say
  the same thing. Transcribing such a formulation term-for-term therefore
  double-counts. Concretely, in the full-derivative convention:
  - ground-state 2e term: `oo0 · (2J' − cx K')[oo0]`, coefficient **1**,
    not 2 (the one-electron term next to it does carry 2, for `P = 2 oo0`);
  - difference-density term: `dmz1doo · (2J' − cx K')[oo0]`, a single
    contraction covering BOTH cross terms of that bilinear form;
  - transition-density terms: coefficient **½**, because the form is
    quadratic in one density and the full derivative differentiates both
    of its occurrences.
- **Amplitude normalisation is a second, independent factor-of-2 trap.**
  The A-matrix eigenvector is normalised to 1; the response densities want
  the spin-adapted amplitudes, smaller by `1/√2`. Every quantity quadratic
  in the amplitudes is then off by exactly 2 if the raw eigenvector is used
  — which is invisible in the translational sum rule and invisible in the
  DFT refusal, so only a finite-difference gate catches it.
- **Gates** (`tests/p13_tddft_gradient.rs`): HF/CIS analytic vs central
  differences of `E_SCF + ω` on distorted water, STO-3G and 6-31G, every
  atom and axis, `h = 5e-4` bohr, ≤1e-5 (measured residual ≤3e-7, and the
  error falls by exactly 4× per halving of `h` from 4e-3 to 5e-4 — that
  scaling is what identifies the residual as the difference formula's
  rather than the gradient's); the translational sum rule; the reduction to
  the RHF gradient when the amplitudes are zeroed. The LDA path adds a
  rebuilt-grid SVWN5 finite-difference gate over all nine components of
  distorted water, the LDA zero-amplitude reduction and translational sum
  rule, and an independent directional finite difference of the local
  third-kernel contraction. Displaced states are followed by AMPLITUDE OVERLAP, not by
  index — water's low TDA roots cross under a 5e-4 bohr step, and following
  the index differentiates a discontinuous function.
- **LDA XC Lagrangian.** The zero-order response includes `f_xc` acting on
  the unrelaxed, transition, and Z-vector densities plus the
  `f_xxc:T:T` potential. Its nuclear derivative evaluates
  `E_xc + v_xc.(D_zoo + Z/2) + T f_xc T` on the atom-centred grid and carries
  AO-centre motion, Becke-weight response, and owner-point motion. The
  `D_zoo + Z/2` scalar-density convention is essential: `D_zoo` is a total
  spin density while the nonsymmetric `z1ao` contributes locally through
  `(z1ao + z1ao^T)/2`. GGA/meta-GGA extensions still need the complete
  sigma/tau third-kernel and moving-grid Lagrangian and are refused until
  those terms are present.

## 22. Gates on the assembled periodic TOTAL energy (v0.1.0)

Every periodic operator was already pinned to an identity of its own. What
none of those can see is an error in the *assembly* — a background convention
counted twice, a Brillouin-zone weight correct for one mesh only, a Pulay
term that happens to vanish for the symmetric geometries the pieces were
tested on. `tests/p20_periodic_energy.rs` adds three gates on the total.

- **A neutral cell converges to the isolated molecule.** This is the sharpest
  single check of the electrostatic conventions, because the pieces it is
  assembled from individually have no molecular limit at all: `E_nuc` carries
  a neutralising background and stays far from the molecular nuclear
  repulsion, and `V(k)` carries a `G = 0` constant times the overlap. The
  test measures the residual over a *series* of cell sizes and requires it to
  decay, not merely to be small once. A mismatched convention leaves a
  size-INDEPENDENT offset, and no single-cell comparison — however tight its
  tolerance — can tell that apart from slow convergence.

- **Supercell extensivity under band folding.** A doubled cell on a `[2,1,1]`
  mesh must equal twice the primitive cell on `[4,1,1]`. The two runs take
  different code paths throughout (different k counts, image-block counts,
  matrix dimensions), so only correct Bloch phases, BZ weights and inverse
  transform make them agree; a mesh-dependent weighting error is invisible to
  any single-mesh test. Run in Hartree-Fock so that finite-k exact exchange
  is included, that being the piece most easily written right at Gamma and
  wrong elsewhere.

  The trap here is the cutoff. It must exceed the SUPERCELL lattice vector,
  not the primitive one. A pair the primitive reaches through its `a`
  translation sits in the supercell's `2a` translation, so a cutoff between
  the two silently keeps that interaction in one calculation and drops it
  from the other, and the test then fails for a reason that has nothing to do
  with the physics being checked.

- **Force sum rule and translation invariance.** Every term of the periodic
  force — Pulay, Ewald, Hartree, XC — contributes to the sum over atoms, so a
  term differentiated with respect to the wrong centre breaks it. That is
  strictly more than the existing finite-difference gates see, because those
  displace one atom at a time. The geometry is deliberately asymmetric: a
  symmetric cell satisfies the sum rule for the trivial reason that every
  force in it is separately zero.

All three are `#[ignore]`d, like the finite-difference force gates. A
periodic SCF costs minutes even for two hydrogens in a minimal basis, because
the Ewald-split Coulomb operator is rebuilt over its own real-space and
reciprocal shells at every k point and every iteration — the cost is set by
the lattice, not by the basis. Measured at roughly ten minutes per gate on a
16-thread desktop.

### Registry census: keep the table a table

`tests/p14_functional_coverage.rs` reads `src/xc/mod.rs` as source text and
treats every `=>` arm between `fn combined` and `pub fn resolve` as one
registry entry. When the nonlocal-term classification was written as a second
`match name` inside that span, the census read

```text
"wb97xv" | "wb97mv" | "b97mv" => Some(RequiredNonlocalTerm::Vv10)
```

as a single table entry naming three functionals at once, and failed. The fix
is not to loosen the census — its whole value is that the exported name lists
are provably a complete view of tables that are otherwise unenumerable — but
to move the classification into `required_nonlocal_for`, outside the span.
A helper that matches on the same names as the table belongs outside it.

### API wiring audit at v0.1.0

Three gaps between what the library computes and what a user can reach:

- `polarizability` and `raman` were implemented in the CLI but absent from
  the command allow-list, so both were unreachable. Added.
- The periodic SCF was reachable from Rust and Python but not from the CLI.
  Added as `periodic`, with `--cell` in Angstrom to match the XYZ convention
  and `--forces` for the analytic forces.
- `ScfOptions::external_field` had no CLI binding, and — more seriously — no
  nuclear derivative. Both are now present. See below.

### Complex finite-k ADIIS (v0.1.1)

Before geometry optimization, integer-occupation periodic SCF uses the same
ADIIS quadratic model as the molecular solver, but its inner product is the
full complex Bloch contraction

```text
<A,B> = sum_(spin,k) Re Tr[A(k)^dagger B(k)].
```

One convex coefficient vector is shared by every spin and k point, preserving
Hermiticity and time reversal while retaining imaginary finite-k blocks. The
stored density is paired with the unshifted physical Fock built from it; a
virtual level shift is applied only after interpolation. If the simplex solve
fails or cannot beat a pure stored endpoint, the newest physical Fock is used.
Above `adiis_until_error` this ADIIS stage is active, and CDIIS takes over at
or below the threshold. Fermi-Dirac runs use CDIIS throughout because this
zero-temperature quadratic model omits the Mermin entropy curvature.

### Fixed-cell periodic optimization (v0.1.1)

Atomic positions may now be relaxed against the analytic periodic force while
the three lattice vectors remain immutable. With Cartesian coordinates `R`,
the optimizer uses `g(R) = -F(R)` and an L-BFGS direction with Armijo
backtracking. Each trial constructs a new `Cell`, so overlap, Hamiltonian,
ECP, Ewald, XC-grid and finite-k SCF caches all belong to that geometry. The
accepted result retains the exact final `Cell` identity together with its SCF
result; an equal-looking separately rebuilt cell is deliberately not treated
as the differentiated object.

At zero temperature the objective is total energy. With Fermi-Dirac
occupations it is the Mermin free energy, matching the analytic-force
convention. A failed trial SCF may be retried at a shorter step, but if no
trial can be evaluated its error is propagated. An L-BFGS direction that
cannot lower the objective is retried once from the same geometry as steepest
descent instead of consuming an apparent accepted step.

This is deliberately not a cell optimizer. The available nuclear Ewald stress
omits electronic kinetic, Hartree, exchange, XC and basis/Pulay strain
response. Lattice-vector relaxation therefore remains unavailable until the
full electronic stress tensor and independent strain finite differences are
complete.

`--nbo` was added to `charges` so the NAO/NPA foundation is reachable from
the CLI as well as from Python.

### The nuclear gradient of a uniform external field

`ScfOptions::external_field` existed and was gated only at fixed geometry
(`dE/dE_ax = -mu_ax`, and the analytic polarizability against finite field).
`assemble_gradient` did not know about it at all, so before this change any
force computed in a field was silently missing a term.

The field contributes

```text
E_field = sum_c E_c Tr[P D^c]  -  sum_A Z_A R_A . E ,      D^c_uv = <u| r_c |v>
```

and both halves move with the nuclei:

```text
dE_field/dR_A = sum_c E_c Tr[P dD^c/dR_A]  -  Z_A E
```

Three things make this simpler than it looks, and one makes it easy to get
half right:

- The first-derivative dipole matrices already existed as
  `deriv_mat::dipole_derivative_matrices`, built for the analytic APT. The
  gradient needed no new integrals.
- There is **no operator-center term**. `dV/dR` has one because the
  nuclear-attraction operator sits on a nucleus; the dipole operator sits at
  a fixed origin, so only the two AO centers move. As a corollary the
  derivative matrices are not antisymmetric the way `dS/dR` is, and
  same-atom pairs do not cancel.
- The density `P` is already the field-perturbed one, and the energy is
  variational in it, so no response term is needed — the same argument that
  makes the ordinary Hellmann-Feynman/Pulay split work.
- The trap: the electronic and nuclear halves are comparable in magnitude
  and opposite in sign for a typical polar molecule. Implementing one and
  not the other leaves a gradient that still looks like a gradient. The gate
  is therefore a central difference of the field-perturbed SCF energy,
  component by component, plus the sum rule — and the test first asserts
  that the field actually moved the gradient, so it cannot pass with the
  field ignored on both sides.

The sum rule is worth stating because it is NOT zero in general. A rigid
translation changes the energy by `-Q_tot R.E`, so

```text
sum_A dE/dR_A = -Q_tot E
```

which vanishes only for a neutral system — and there it vanishes through a
cancellation between the electronic and nuclear halves, which is precisely
what a half-implemented term would break.

The analytic Hessian adds the direct skeleton contraction
`sum_c E_c Tr[P d^2 D^c/dR_a dR_b]`. The standard CPKS right-hand side already
contains the matching first dipole-integral derivative because it is part of
the perturbed core Hamiltonian. `dipole_second_contraction` evaluates the
nested bra and mixed bra-ket blocks directly, with ordered shell pairs
supplying the remaining centre derivatives. It is gated first against finite
differences of the dipole-integral contraction and then end to end against the
numerical Hessian of the field-dependent analytic gradient.

## Periodic and population-analysis foundations (post-10j)

- Finite-temperature occupations are solved over the entire weighted k mesh,
  separately for alpha and beta spin, from a single chemical potential per
  spin. The SCF convergence quantity is the Mermin free energy `F = E - T S`.
  Electron conservation, a uniform spectrum-shift identity, particle-hole
  symmetry, and the zero-temperature limit gate the occupation solver.
- The Pulay term uses the energy-weighted density `W = C f epsilon C^dagger`.
  Replacing it by `P F P` would introduce `f^2` and is wrong for fractional
  occupations; a noninteracting Mermin finite difference pins the implemented
  expression.
- Periodic ECP energies sum both translated AO pairs and translated ECP
  centers. A large-cell home block is required to reproduce the molecular ECP
  operator. Analytic periodic ECP forces are refused until AO-center and
  operator-center derivatives of every translated term are present.
- Partial periodicity, periodic VV10/CHG, and future ECP force support have
  explicit typed capability contracts. A 2D or 1D cell cannot accidentally
  fall through to the 3D Ewald kernel.
- The natural-orbital foundation diagonalizes each atom block of the density
  in the symmetric orthogonal metric. It reports NAO occupations, NPA charges
  and spins, and Wiberg indices. Tests pin electron conservation, metric
  orthonormality, bond-index symmetry, and invariance under atom-local AO
  rotations. NHO/Lewis/NBO localization remains a separately staged layer.

## 23. Boundary derivatives of the XC kernels (v0.1.1)

The domain of an XC functional has a boundary — a spin channel empties, the
density gradient vanishes at a point of exact symmetry, τ reaches the von
Weizsäcker limit — and the boundary is not a rare pathology. The centre of a
closed-shell atom has σ = 0 exactly. Any N_β = 0 system sits on ζ = 1 for
every grid point of the calculation. A hydrogen atom is on the boundary
everywhere.

Before this release the boundary was untested, and 182 of the 1,197
functional × point × order combinations in `tests/p22_xc_boundaries.rs`
failed on it. The convention below is what those fixes implement.

**Why a boundary value becomes NaN here, mechanically.** Derivatives come
from forward-mode AD (`src/dual.rs`), not from transcribed derivative
formulas. Under AD a term like `x^{2/3}` evaluated at `x = 0` computes its
tangent as `(2/3)·x^{-1/3}·ẋ = ∞·0 = NaN` — and because `DualN::lift` folds
that product into every slot, **one singular factor poisons the whole dual,
including slots that have nothing to do with the singular variable**. This is
why `√σ` at σ = 0 destroys `vrho`, not merely `vsigma`, and why the symptom
is so much larger than the cause. It is also why the failure is invisible to
an energy-only test: the *value* is finite, only the tangents are not.

**The convention.** Every functional declares, per boundary and per
derivative order, which of four things it returns:

| | meaning |
| --- | --- |
| `finite` | the derivative exists and is returned |
| `one-sided finite` | the two-sided limit does not exist; the one-sided limit from inside the domain does, and that is what is returned |
| `divergent` | the quantity genuinely has no finite limit; the code says so rather than returning a number |
| `unsupported` | the order is beyond `max_analytic_order()` for that family |

**NaN is not one of the four.** A NaN is never a correct answer to
"what is ∂e/∂ρ here" — it is the absence of an answer wearing the costume of
one, and it propagates silently into the Fock matrix.

**The trigger is a value cutoff, never an exact comparison.** Branching on
`ζ == 1.0` would be wrong: an SCF routinely produces ρ_β = 1e-300 at points
far outside the molecule, which is not exactly the boundary but is
numerically indistinguishable from it. Channels are treated as ABSENT below
`RHO_CUTOFF = 1e-13` (per channel) and gradients below
`SIGMA_CUTOFF = 1e-26`. Both are values, both are stated in the source next
to the branch, and `tests/p22_xc_boundaries.rs` walks ρ_β down through
1e-12, 1e-14, 1e-16 and 1e-300 to check the energy and the α-channel
potential are continuous across the switch.

**Empty channels: the ferromagnetic branch is exact, not an approximation.**
For a correlation functional, `E(ρ_α, 0) ≡ E_ferro(ρ_α)`; the opposite-spin
term is identically zero, not merely small. So the fix is to evaluate the
closed-form ζ = 1 expression rather than to evaluate the general expression
and hope it survives. The project already had `pw92_epsilon_ferro` and
`pbe_c_eps_ferro` — written precisely for this — but the LDA and PBE drivers
themselves never called them. B97's `have[s]` occupancy mask was the
existing precedent; A and B now follow it.

The β-channel potential `vrho[1]` as ρ_β → 0⁺ is the one quantity here that
is honestly `divergent`: `dφ/dζ = ⅓[(1+ζ)^{-1/3} − (1−ζ)^{-1/3}] → −∞`. It
is recorded as `divergent` in the expectation table, and below the cutoff the
ferro branch returns no contribution to a channel that holds no electrons —
which is the physically meaningful statement, since there is no β density for
that potential to act on.

**Zero gradient: rewrite the expression, do not clamp the value.** Several
functionals are written with `√σ` but are in fact analytic in σ. B88 is the
clearest case: it is built from `x = √σ/ρ^{4/3}`, but what actually appears
is `x²` and `x·asinh(x)`, and the second of those is an analytic function of
`x²`:

```
x·asinh(x) = u^{1/2}·asinh(u^{1/2})
           = u − u²/6 + 3u³/40 − 5u⁴/112 + 35u⁵/1152 − …      (u = x²)
```

So `x_asinh_x(u)` takes `u = σ/ρ^{8/3}` directly, sums the series below
u = 1e-2 and evaluates the closed form above it. The `√σ` never meets a dual
zero, and the derivative that comes out is the *correct* one — which a clamp
would not have given. Clamping replaces a NaN with a number; rewriting
recovers the number that was there all along. TPSS is handled the same way,
by factoring σ out of the term-3 square root so that what remains is a sum of
squares of σ-free coefficients.

Where a functional is genuinely non-analytic at σ = 0 — `√σ` standing alone,
with a divergent one-sided derivative — that is a *stated convention*, and
the list of functionals relying on it is pinned by name in
`the_absent_gradient_convention_is_in_force_exactly_where_declared`. Adding a
functional to that list is a deliberate edit to a test, not something that
can happen by accident.

**On deviating from libxc.** The rule for this project is that the published
paper is the specification; libxc decides only *which* published variant or
digit set to follow when a paper is ambiguous. Two cases in this release
tested that rule in opposite directions:

- The wPBEh s-rescaling short-circuits are discontinuous, in libxc and here
  identically (−5.141e-4 at s = 1, +1.616e-3 at s = 15, agreeing to the last
  digit). A discontinuous exchange enhancement in the valence region is a
  mathematical defect, so it was removed — and the *first* replacement, the
  bare Henderson softplus, was also wrong: it maps s = 0 to −1.892e-4 and
  violates the uniform-gas limit. The shipped form subtracts and re-adds the
  same quantity so that f(0) = 0 exactly. The measured cost of the deviation
  is recorded in `docs/FUNCTIONALS.md`.
- libxc's arithmetic-mean clamp on σ_αβ in the meta-GGA driver looked like a
  Cauchy–Schwarz bound stated wrongly. It is not: `|∇(n↑ ± n↓)|² ≥ 0` gives
  exactly the arithmetic mean. "Fixing" it to the geometric mean would have
  been a bug and would have broken the r²SCAN oracle. The rule cuts both
  ways; the check before the fix is what made the difference.

## 24. Anisotropic pseudo-substituent mechanisms and parameter boundary

A G site is deliberately not a new element. It is an ordinary
`Atom { z: 1 }` at `R_G`, an atom-indexed valence-basis override, and a
separate `PseudoSite` record. The hydrogen-like nucleus preserves all existing
grid, cavity, SAD, mass and file-format rules; the table is the only place
where a future anisotropic operator may distinguish G from H. With the
operator identically zero and the ordinary H basis, this representation is
gated by bit-identical HF and SVWN5 energy/gradient calculations on methane.

For a site bonded to X, the body frame is

```text
e_z = normalize(R_G - R_X)
d   = R_ref - R_X
e_x = normalize(d - e_z (d . e_z))
e_y = e_z x e_x
U   = [e_x e_y e_z]                 (columns)
```

X is the unique nearest atom to G. The model specification selects `R_ref`
deterministically from X's other bonded neighbours by highest atomic number
and then lowest atom index.
Equidistant X candidates, a nearly linear reference, and a missing reference
for a non-cylindrical site are errors, never arbitrary frame choices. A
cylindrical site may return no full frame because rotations about `e_z` are a
parameter gauge; a later parameter loader must enforce the corresponding
`m != 0` zeros rather than choosing an observable azimuth. The same generic
normalization/projection/cross-product arithmetic is evaluated with
`DualN<f64, 9>` to obtain `dU/d(R_X,R_G,R_ref)`, and an independent central
finite difference gates all nine coordinate columns.

`build_local_frame_with_connectivity` makes that topology a caller contract:
it receives X and a slice containing only X's other bonded neighbours,
validates the indices and duplicates, and applies the priority strictly inside
the slice. Adding an unrelated high-Z atom therefore leaves `R_ref` and U
bit-identical. The source-compatible `build_local_frame` still discovers a
unique nearest X, but refuses if more than one possible reference remains;
it never replaces missing connectivity with an empirical bond cutoff or an
all-atom priority search.

Real harmonic rotation is constructed from this project's own
`cart_to_sph(l)` tables. If C maps the real solid harmonics into Cartesian
monomials and T(U) is the exact degree-l monomial action, then

```text
D^(l)(U) = C^T T(U) C (C^T C)^-1,        l = 0..4
S_lm(U r) = sum_m' D^(l)_mm'(U) S_lm'(r).
```

The inverse is applied by a linear solve. Harmonic invariance makes this an
exact basis transformation, not a fitted least-squares rotation. Pointwise
rotation, composition, orthogonality, determinant, and the l=0/l=1 limits are
separate gates.

These mechanisms do not determine chemistry. The Phase-4 v1 programme for
Me/tBu requires a provenance-bearing fit of the valence basis, core density,
B operator, C6 and residual repulsion, followed by bond, torsion, structure
and analytic-gradient acceptance gates. Et/Ph additionally require the v2
torsion/torque and G-G/exclusion model and its transferability gates. None of
those fitted records exists in this tree. CF3 and OMe were not defined by the
Phase-4 v1/v2 specification at all, so they additionally lack a model form,
training set and acceptance thresholds. The input parser recognizes all six
names for deterministic configuration handling, but every physical build
returns `DftError::Unsupported` naming exactly this missing evidence. No
nonzero multipole, mass, repulsion or dispersion number is inferred from the
chemical name.

### Candidate fitter mechanics

For a fixed-density linearized row i with feature vector a_i, reference y_i,
unit scale s_i and explicit weight w_i, the implemented initialization solves

```text
min_p  1/2 sum_i w_i [(a_i.p - y_i)/s_i]^2
     + lambda/2 sum_k [(p_k-p0_k)/q_k]^2 .
```

The normal matrix and right-hand side are accumulated deterministically with
compensation. Its symmetric eigendecomposition supplies both the solution and
an explicit rank check. Molecule/geometry group IDs, rather than individual
observable rows, are hashed with a fixed recorded seed and split 80/20, so no
geometry leaks from train to validation.

Symmetric B blocks use Frobenius-preserving packing: diagonal entries are
stored directly and an off-diagonal is stored as sqrt(2) B_ij. Each nonlinear
trial is unpacked, diagonalized, eigenvalues below the configured floor are
clamped, and the block is reconstructed. Projected L-BFGS uses an Armijo line
search and convergence of the projected-gradient mapping. Failed
`aux_population`, `min_b_eigval`, `min_s_eigval`, or `lowest_mo` diagnostics
follow an explicit exclude-or-penalize policy, with a maximum exclusion
fraction. The candidate record embeds level, dataset fingerprint, seed, split,
regularization and diagnostics; level mismatch is an error.

The exact synthetic gate generates held-out observations from known c and B
values and recovers every parameter below 1e-6. This validates the fitting
machinery only. It neither supplies physical reference observations nor makes
a named substituent acceptable.

## License audits

Audit records of each implementation phase, newest first.

### v0.1.1 license audit — 2026-08-13

- [x] **XC boundary fixes.** All from the primary papers already cited in
      `docs/references.md`; the ferromagnetic branches are closed-form
      ζ = ±1 limits derived in-house, and the B88 rewrite is an ordinary
      series expansion of `x·asinh(x)` in `x²`. libxc (MPL) was READ ONLY to
      establish which published convention a value follows — and in the
      σ_αβ clamp case, reading it is what *prevented* a wrong change. No code
      or generated data copied. The one deliberate numerical deviation from
      libxc (the wPBEh s-rescaling) keeps libxc's own form as a
      `#[cfg(test)]` reference so the frozen pins still bind the hole model.
- [x] **Atomic masses** (`src/constants.rs`, Z = 1…118) from IUPAC 2021
      standard atomic weights, with the longest-lived-isotope rule applied
      and recorded per element where no standard weight exists. IUPAC's
      tables are published reference data, not an implementation.
- [x] **Grid atomic radii** unified on Slater, *J. Chem. Phys.* **41**, 3199
      (1964). Published table, transcribed from the paper's own values.
- [x] **Pseudo-substituent mechanisms** (`src/pseudo_sub/`) are derived
      in-house from vector normalization/projection and the crate's existing
      `cart_to_sph` tables. No external implementation or fitted parameter
      data was copied. In particular, no named physical parameter is bundled:
      the missing provenance and validation evidence is a typed refusal.
      The fitter uses only in-house weighted least squares, eigensolver-based
      projection and L-BFGS arithmetic over synthetic/generated inputs; it
      adds no dependency or third-party parameter data.
- [x] **XDM** (`src/dispersion/xdm.rs`) implemented from Becke & Johnson's
      primary papers. **No code or data from `postg`, Quantum ESPRESSO or
      psi4 was read or used**. Its coefficients come from the exchange
      hole of the actual density. One bundled table:
      `xdm_free_atom_polarizability.csv`, from Schwerdtfeger & Nagle,
      *Mol. Phys.* **117**, 1200 (2019), an independent atomic-physics
      compilation with no connection to any dispersion implementation. All
      36 rows were re-read from the openly published table
      (`ctcp.massey.ac.nz/2023Tablepol.pdf`, last updated 17 July 2023) and
      verified digit for digit, 2026-08-13. The companion free-atom *volume*
      is deliberately **not** a second external table — it is computed from
      this project's own `sad_density`, which is what makes the free-atom
      limit exact by construction rather than approximately consistent.
      Per-functional BJ damping parameters are NOT bundled: they are
      empirical fits this project cannot independently verify, so the caller
      supplies `a1`/`a2` and a request without them is refused by name.
- [x] **D4 remains outside implementation scope.**
- [x] Dependencies unchanged (`libm`, `rayon`, `faer`, `pyo3`). Headers and
      allowlist machine-checked; the new CSV carries a PROVENANCE header.

### GIAO NMR license audit — 2026-08-06

- [x] The GIAO formalism (London orbitals, the phase derivatives of S, h
      and the ERIs, the imaginary-perturbation CPHF, Ramsey's dia/para
      split) is standard textbook and primary-literature material, derived
      in-house and recorded in section 18f above. The moment-weighted
      integrals reuse this project's own McMurchie–Davidson layer; no
      external integral code was consulted or transcribed. A frozen
      PySCF/pyscf-properties full-tensor oracle is used for HF, LDA/GGA and
      their global/range-separated hybrids; pyscf-properties does not provide
      the meta-GGA NMR reference needed to release that experimental path.
      Independent internal identities additionally cover gauge-origin
      independence, the shift identity, the kinetic commutator, Lamb's
      free-atom formula and molecular symmetry. One new
      constant (the 2018-CODATA fine-structure constant) in
      `src/constants.rs`; no new bundled data; dependencies unchanged;
      headers/allowlist machine-checked.

### Functional-pack-II license audit — 2026-08-04

- [x] P86/PW91/HJS/wPBEh/M06-SX/r²SCAN implemented from the primary
      papers; all constants are published fit values (Perdew 1986/1991,
      HJS 2008, Heyd 2003 + Henderson 2009, Wang 2020, Furness 2020).
      libxc (MPL) sources were READ ONLY to identify which published
      convention/digit set the oracle uses (RG'86 Padé, 1e-10 floors,
      FHC flag, EG quartic constants as published); no code or generated
      data was copied — the Rust implementations are original and the
      numerical agreement is enforced by probes/pins committed under
      tools/oracle + unit tests. No new bundled data files. Dependencies
      unchanged; headers/allowlist machine-checked.

### Post-plan license audit — 2026-08-04

- [x] TDA/Davidson standard formalism implemented fresh; PySCF used as a
      numerical oracle only. No new data. Headers/allowlist
      machine-checked.

### Phase 9 license audit — 2026-08-04

- [x] Bindings and calculator written fresh (pyo3 is the allowlisted
      optional dependency; ASE is a runtime import on the Python side,
      not a crate dependency). No new data. Headers/allowlist
      machine-checked.

### Phase 8 license audit — 2026-08-04

- [x] CPKS/Hessian formalism is standard textbook material (Pople et al.
      CPHF; gauge-invariant W⁽¹⁾ derived in-house); no external code
      consulted, no new data files. Headers/allowlist machine-checked.

### Phase 4.5 license audit — 2026-08-03

- [x] ECP integrals implemented from the McMurchie–Davidson 1981 method
      description (papers only); LANL2DZ data was already bundled from BSE
      (CC0) with PROVENANCE. PySCF used strictly as a numerical oracle,
      with its built-in LANL2DZ data confirmed numerically identical to
      our BSE file before use. Headers/allowlist machine-checked.

### Phase 9b license audit — 2026-08-03

- [x] M06/ωB97M-V/VV10/CHG implemented from Zhao–Truhlar 2006/2008,
      Mardirossian–Head-Gordon 2016, Vydrov–Van Voorhis 2010, and
      Chai–Head-Gordon 2008 (papers only); libxc/PySCF used strictly as
      numerical oracles (pointwise values, least-squares recovery, SCF
      energies). No code or data copied; CHG reuses the D2 table already
      transcribed from Grimme 2006. Headers/allowlist machine-checked
      (license_audit tests green).

### Phase 9a license audit — 2026-08-03

- [x] B97/HCTH implemented from Becke 1997 / Hamprecht 1998 / Boese 2000;
      libxc used only as a numerical oracle (pointwise values and the
      least-squares parameter check). Headers/allowlist machine-checked.

### Phase 7 license audit — 2026-08-03

- [x] Standard response formalism implemented fresh; no new data;
      headers/allowlist machine-checked.

### Phase 6 license audit — 2026-08-03

- [x] All algorithms standard textbook material implemented fresh; no new
      data beyond the IUPAC mass table (facts). Headers/allowlist checked.

### Phase 5 license audit — 2026-08-03

- [x] Derivative machinery original (standard shifted-l identities); D3
      data via MOPAC v23.2.5 (Apache-2.0) with per-file PROVENANCE; D2
      table transcribed from Grimme 2006.
- [x] Headers/allowlist machine-checked; PySCF used as numerical oracle
      only.

### Phase 4 license audit — 2026-08-03

- [x] GGA/RSH functionals implemented from the primary papers (Becke 1988,
      PBE 1996, Handy–Cohen 2001, Lee–Yang–Parr/Miehlich, Iikura–Tsuneda–
      Yanai–Hirao 2001, Yanai–Tew–Handy 2004); libxc consulted only as a
      black-box numerical oracle (pointwise values, rsh coefficients).
- [x] No new bundled data. Headers/allowlist machine-checked.

### Phase 3 license audit — 2026-08-03

- [x] Functionals implemented from the primary papers (VWN 1980, PW92,
      Slater/Dirac); grid schemes from Becke 1988 / Treutler–Ahlrichs 1995;
      no libxc/libint code or data consulted.
- [x] New bundled data: def2-universal-JFIT/JKFIT auxiliary bases (BSE,
      CC0) with PROVENANCE headers.
- [x] SPDX + copyright headers machine-checked; dependency allowlist
      unchanged.
- [x] PySCF (libxc-backed) used strictly as a numerical oracle.

### Phase 2 license audit — 2026-08-03

- [x] SCF/DIIS/J-K code written fresh against the standard literature
      (Pulay CDIIS; Saunders–Hillier level shifting; Almlöf direct SCF);
      no third-party code consulted beyond the author's own pm6-rs.
- [x] No new bundled data.
- [x] SPDX + copyright headers machine-checked; dependency allowlist
      unchanged.
- [x] PySCF used strictly as a numerical oracle (frozen CSV energies).

### Phase 1 license audit — 2026-08-03

- [x] No verbatim/mechanically-translated code from copyleft sources; the
      MD integral engine is written from the standard literature
      (McMurchie–Davidson 1978; Helgaker/Jørgensen/Olsen for E/R/moment
      recursions and solid harmonics).
- [x] New bundled data: BSE basis sets (CC0) with per-file PROVENANCE
      headers and `third_party/bse/NOTICE.md`; no other data added.
- [x] SPDX + copyright headers machine-checked; dependency allowlist
      unchanged.
- [x] PySCF used strictly as a numerical oracle (frozen CSV values); no
      PySCF code consulted or translated.

### Phase 0 license audit — 2026-08-03

- [x] No verbatim/mechanically-translated code from copyleft sources
      (all Phase 0 code originates from the author's own projects or is new).
- [x] No third-party data bundled yet.
- [x] SPDX + copyright headers on all sources (machine-checked by
      `tests/license_audit.rs`).
- [x] Dependency allowlist unchanged: libm, rayon, faer, pyo3(optional).
- [x] External implementations referenced only as numerical oracles (none yet).
