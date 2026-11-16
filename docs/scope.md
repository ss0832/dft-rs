<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

# Scope

What dft-rs computes, what it refuses, and where the edges are. This is the
concrete version of the README's status table. A listed implementation has
the bounded tests described alongside it; it does not imply independent
physical acceptance of every functional, reference, basis or environment.
Unsupported requests are explicit errors rather than substitute methods.

The [implementation status](development-status.md) gives the current
v0.3.2 development checkpoint. Known failures and unresolved conditions
are retained in the response and molecular-property ledgers. An available
experimental route must not be interpreted as universal physical accuracy.

Implementation, support and validation labels are defined in the
[authoritative status vocabulary](conventions.md#status-vocabulary).

Contents: [1 Methods](#1-methods) · [2 Basis sets and ECPs](#2-basis-sets-and-ecps)
· [3 Grids](#3-grids) · [4 Properties](#4-properties) · [5 Dispersion](#5-dispersion)
· [6 Continuum solvation](#6-continuum-solvation) · [7 SAPT](#7-sapt)
· [8 Performance](#8-performance) · [9 Periodic boundary conditions](#9-periodic-boundary-conditions)
· [10 Surfaces](#10-surfaces-what-is-reachable-from-where)
· [11 Baseline exclusions and extended scope](#11-baseline-exclusions-and-extended-response-scope) · [12 Verification posture](#12-verification-posture)
· [13 Physical target registry](#13-physical-target-registry)
· [14 Extended DFT method variants](#14-extended-dft-method-variants)

---

## 1. Methods

| Method | Energy | Analytic gradient | Analytic Hessian |
| --- | --- | --- | --- |
| RHF / UHF | yes | yes | yes |
| LDA (SVWN5, SVWN3, SPW92) | yes | yes | yes |
| GGA (PBE, revPBE, BLYP, BP86, OPTX, …) | yes | yes | yes |
| Hybrids (B3LYP, B3LYP5, B3PW91, O3LYP, PBE0, …) | yes | yes | yes |
| Range-separated (CAM-B3LYP, LC-BLYP, LC-PBE, LC-ωPBE, HSE06/HSE03) | yes | yes | yes |
| B97 / HCTH power series | yes | yes | yes |
| meta-GGA (TPSS, TPSSh, r²SCAN, M06-L/M06/M06-2X, revM06, M06-SX) | yes | yes | yes |
| ωB97 family (ωB97, ωB97X, ωB97X-D) | yes | yes | yes |
| VV10 / SC-VV10 (-V functionals: ωB97X-V, ωB97M-V, B97M-V) | yes | yes | yes |
| ECP (LANL2DZ, Stuttgart RLC/RSC) | yes | yes | yes |
| TDDFT (TDA and full RPA) | yes | HF/CIS reference only (see below) | no |
| SAPT decomposition | yes | n/a | n/a |

Unrestricted (UHF/UKS) is the native formulation; the closed-shell path is a
special case of it, not a separate code. `--unrestricted` forces the
unrestricted reference even for a singlet — on its own it reproduces the
restricted energy to machine precision (which is how the unrestricted path
is gated), and with `--guess-mix <deg>` it reaches broken-symmetry solutions
for diradicals and homolytic dissociation. Stretched H₂ at 2.5 Å with
B3LYP/6-31G: −0.9535 restricted, −1.0049 broken-symmetry, ⟨S²⟩ = 0.94.

### SCF convergence

Starting point: **SAD** by default — a superposition of spherically averaged
atomic densities, each converged on the fly in the molecule's own basis, so
it needs no shipped data and follows any per-element basis mixture or ECP
assignment automatically. `--guess gwh|core` selects the one-electron
alternatives. Typical saving over GWH: water/B3LYP 11 → 7 iterations, water
dimer/PBE0 9 → 7. (A single open-shell ATOM is the one case where SAD is
worse, because a spherical guess is by construction the symmetry it has to
break.)

Iteration: ADIIS while the error is large → CDIIS → SOSCF near convergence.
Thresholds follow PySCF (`conv_tol` 1e-9, `conv_tol_grad` = √`conv_tol`) and
are individually adjustable.

### Excited-state gradients

The analytic TDA (CIS) excited-state gradient is a Handy–Schaefer Z-vector
assembly: one CPHF solve serves all `3N` coordinates instead of one per
coordinate. It returns the gradient of the TOTAL excited-state energy
`E_SCF + ω`, including nuclear repulsion, so it is the force on the excited
state. See [THEORY.md](THEORY.md#21-tda-excited-state-gradients-z-vector)
for the formulation.

Its gate is central differences of this program's own `E_SCF + ω`, component
by component (following the state by amplitude overlap rather than by index,
because roots cross under displacement), plus the translational sum rule and
the reduction to the RHF gradient at zero amplitude.

Scope: **closed-shell singlet TDA over an HF/CIS or gas-phase pure-LDA
reference**. The LDA path is grid-aware and currently gated with SVWN5.
Full RPA, triplet, open-shell, GGA/meta-GGA, hybrid/RSH, nonlocal-correlation,
and PCM gradients are not implemented.

### Deliberate refusals


- **Double hybrids** — five named energy recipes and generic typed parents
  are available, including independent SS/OS PT2 terms and the accepted static
  molecular correlated dipole/polarizability subset. Higher, dynamic,
  periodic and nuclear correlated combinations require the specific provider
  and acceptance named in [response.md](response.md) and the CSV ledgers.
  A parent-functional response alone is never the full double-hybrid result.
- **Excited-state gradient beyond pure LDA** — SVWN5/LDA includes the
  `f_xc`/`f_xxc` transition and response contractions and full moving-grid
  Lagrangian. GGA/meta-GGA sigma/tau terms and hybrid/RSH, nonlocal, and PCM
  response layers remain typed refusals rather than partial forces.

---

## 2. Basis sets and ECPs

48 sets bundled from the [Basis Set Exchange](https://www.basissetexchange.org/)
(CC0) in Gaussian94 format — 46 orbital sets plus two auxiliary fitting sets
for RI-J — including STO-3G, the Pople 6-31G/6-311G series,
def2-SV(P)/SVP/TZVP/QZVP with their diffuse variants, cc-pVXZ and
aug-cc-pVXZ up to 5Z, the LANL family and Stuttgart RLC/RSC. Any `.g94` file
can be passed by path.

**Per-element assignment.** `--basis` accepts a spec, not just a name:

```text
def2-tzvp; H=def2-svp            light atoms cheaper
def2-svp; Ni,Fe=lanl2dz          ECP set on the metals only
H=sto-3g; O=./my-oxygen.g94      no base; every element assigned
@basis.spec                      read the spec from a file
```

**Basis and ECP are independent.** `--ecp` takes the same grammar and is
applied after the orbital basis, so a core potential is never forced on you
because it happened to ship inside the set you wanted the orbitals from, and
it can come from a different set entirely:

```text
--basis "def2-tzvp; Ni=lanl2dz" --ecp none        orbitals only, no ECP
--basis def2-tzvp --ecp "Ni=stuttgart-rsc-1997"   def2 orbitals, Stuttgart core
```

An `--ecp` spec touches only the elements it names; `none` clears.

- Spherical harmonics throughout, in Molden's canonical `m` ordering
  (`0, +1, −1, +2, −2`). Cartesian (6d/10f) AOs exist only as the integral
  engine's internal intermediate and are **not** selectable — there is no
  `--cart`. Every oracle file was generated in the same spherical convention,
  so the comparison is like for like; a 6d run would not reproduce them and
  is not offered rather than being offered untested.
- ECPs: type-1 local plus type-2 semi-local integrals, with `Z_eff` and the
  electron count adjusted for the replaced core.
- Linear dependence is removed by canonical orthogonalization at a 1e-7
  threshold, with a warning when vectors are dropped.

---

## 3. Grids

Treutler–Ahlrichs M4 radial × Lebedev-free product angular, Becke
partitioning (Stratmann–SSF optional). PySCF-compatible one-knob
`--grid-level 0..9` (default 3), plus per-element and per-atom overrides and
three-zone pruning.

Weight derivatives are analytic to second order (nested duals over a
Scalar-generic weight function), so the gradient satisfies the translational
sum rule to 1e-9 and the Hessian matches finite differences.

---

## 4. Properties

| Property | Status |
| --- | --- |
| Mulliken / Löwdin charges | yes |
| NAO / NPA charges and Wiberg bond indices | yes; NHO/Lewis/NBO localization staged |
| Dipole moment | yes |
| MO energies / coefficients, density, overlap, Fock | yes (`--dump-wfn`, Python) |
| Molden output | yes |
| Harmonic frequencies (Eckart-projected) | yes |
| IR intensities (analytic APT) | yes |
| Raman activities | yes (∂α/∂R numerical) |
| Static polarizability | yes (analytic CPKS) |
| Uniform external field | yes, including analytic gradients and Hessians |
| Geometry optimization (L-BFGS) | yes |
| Transition-state search (P-RFO and Householder RS-I-RFO) | yes |
| Excitation energies + oscillator strengths | yes (TDA, RPA) |
| NMR shieldings (GIAO) | yes — closed-shell HF/global or range-separated exact exchange/LDA/GGA; meta-GGA experimental pending an external full-tensor oracle; see below |

Both walkers write the final geometry as XYZ next to the input, and
`--traj FILE` streams a multi-frame XYZ **as the walk proceeds** (appended
and flushed per accepted step, with the step number, energy and |g|max on
each frame's comment line) rather than buffering to the end — so an
interrupted run still leaves its path behind. `OptResult`/`SaddleResult`
also carry the frames in memory.

**External field.** `--field Ex,Ey,Ez` adds `+E·r` to the core Hamiltonian
and `−Z_A R_A·E` to the nuclear energy. Everything at a fixed geometry is
exact — energies, densities, dipoles, and the analytic polarizability, which
differentiates with respect to the *field* by CPKS on the already-perturbed
Fock operator. The gate there is the conjugacy identity `dE/dE_ax = −μ_ax`.

The **analytic gradient carries the field too**: the electronic
`Σ_c E_c Tr[P ∂D^c/∂R_A]`, from the geometric derivative of the dipole
integrals, plus the nuclear `−Z_A E`. Unlike `∂V/∂R` there is no
operator-center term, because the dipole operator sits at a fixed origin
rather than on a nucleus. `gradient`, `optimize` and `saddle` therefore all
run in a field, gated by central differences of the field-perturbed energy
and by the sum rule `Σ_A ∂E/∂R_A = −Q_tot E`.

The **analytic Hessian carries the field as well**. Its skeleton contains
`Σ_c E_c Tr[P ∂²D^c/∂R_A∂R_B]`; the ordinary CPKS right-hand side already
contains the first dipole-integral derivative through the perturbed core
Hamiltonian. The direct second-derivative contraction is gated independently
against displaced dipole integrals, and the complete field-dependent Hessian
against finite differences of the analytic gradient.

**NMR (GIAO).** Shielding tensors are computed and reported, for
**closed-shell HF, LDA, GGA, global hybrids, and range-separated hybrids**.
The current-corrected meta-GGA route is experimental until an independent
full-tensor meta-GGA NMR oracle is available.
The whole chain is implemented: the one-electron operator layer
(`src/properties/nmr.rs`), the field derivative of the two-electron
integrals that the London phase factors introduce (`giao_eri.rs`), the
imaginary-perturbation magnetic CPHF and the dia/paramagnetic assembly
(`magnetic.rs`). The derivation is in
[THEORY.md](THEORY.md#18f-giao-nmr-shielding-the-derivation-and-its-gates).

The external oracle freezes PySCF 2.14.0 plus the official pyscf-properties
commit `4eee5a430fb47eca5962f36fdcaf75c2b87e7ede`: full total/diamagnetic/
paramagnetic 3x3 tensors for HF, SVWN5, PBE, BLYP, B3LYP, and CAM-B3LYP on
asymmetric water. Grid levels 8/9 must agree within 0.005 ppm before the CSV is
written. Independent limiting checks additionally cover gauge-origin
independence across three origins including one 5
Bohr off the molecule, the first-order origin-shift identity on each
derivative matrix separately, the kinetic commutator `⟨m|[r_j,h]|n⟩ =
⟨m|d_j|n⟩`, Lamb's free-atom formula for a closed-shell atom, one-centre
GIAO equalling the common gauge origin exactly, C2v symmetry of water, and
the differentiated idempotency of `dP/dB`. RSH additionally gates
exchange-only Erf contraction, `Erf + Erfc = Coulomb`, finite-omega origin
independence and the large-omega HF limit. LDA/GGA gate their London-phase
matrices against a separate complex-phase finite difference and the
ordinary-Vxc origin-shift identity. The GGA gate uses unequal spin densities,
checks the origin-independent phase-gradient term directly, and is followed by
full PBE, B3LYP, and CAM-B3LYP three-origin shielding gates.

Three cases are refused rather than approximated:

Meta-GGA uses the spin-resolved Dobson variable
`tau_tilde = tau - |j_p|^2/(2 rho)`. Its conventional-tau London term and
the current-current magnetic Hessian are both retained; the latter keeps pure
meta-GGA CPHF coupled and requires the SCF molecular grid in either gauge.
A conventional-tau-only trial failed the strict TPSS three-origin gate and
was rejected before the current correction was implemented. The corrected
path has internal FD/origin gates, but the pinned external oracle does not
support meta-GGA NMR tensors.

- **SC-VV10** — the nonlocal-potential London-phase derivative is missing;
- **ECPs** — the core potential's moment-weighted integrals are missing;
- **open-shell references**.

Magnetizability `ξ = −d²E/dB_a dB_b` is also not implemented: because the
basis depends on `B` on both sides it needs explicit second-derivative
integrals, including double-moment four-centre ones. THEORY.md lists them.

---

## 5. Dispersion

| Model | Status |
| --- | --- |
| D2 | Molecular and 3D periodic, including analytic forces, periodic stress and Gamma-point Hessian, FC3 and FC4. [D2 verification](response.md#periodic-d2-empirical-dispersion) |
| D3(zero), D3(BJ) | yes, including the fully analytic Hessian |
| Chai–Head-Gordon tail (ωB97X-D) | yes, including the fully analytic Hessian |
| VV10 / SC-VV10 | yes, including the fully analytic Hessian |
| rVV10 | Rust fixed-density molecular energy and AO potential. [rVV10 verification](response.md#fixed-density-molecular-rvv10) |
| XDM (exchange-hole dipole moment) | energy only — library API and `--disp xdm` for `energy`/`charges`; no gradient |
| D4 / D4S / D4SL | Outside implementation scope |

The Hessian is a `DualN<DualN<f64,3>,3>` nested dual seeding, one level
deeper than the analytic gradient's `DualN<f64,3>` — see
`dispersion_hessian` in `src/dispersion/mod.rs`. `frequencies`/`hessian`
add it on top of the CPKS electronic Hessian whenever `--disp` selects one
of these models. RI-J remains the only supported SCF mode that forces the
numerical Hessian, because auxiliary-basis nuclear derivatives are staged.

**XDM** (`src/dispersion/xdm.rs`) calculates C₆/C₈/C₁₀ from the
Becke–Roussel exchange hole of the SCF density. It uses a small, openly published free-atom polarizability table (not
XDM-specific, not sourced from `postg`/Quantum ESPRESSO/psi4 — see the CSV's
PROVENANCE header). Implemented and gated: the Becke–Roussel hole solve,
Hirshfeld atomic partitioning of the exchange-hole moments, the free-atom
volume scaling to an effective polarizability, C₆/C₈/C₁₀ (Eq. 39-41 of the
formulation it follows), and Becke–Johnson damping with caller-supplied
`a1`/`a2` (this project does not have independently verified access to the
published per-functional fit values, so it asks the caller for them rather
than transcribe a possibly-wrong digit). Measured against the classic
noble-gas references at the HF/aug-cc-pVTZ level: He–He C₆ = 1.73 a.u. (18%
high vs. 1.461), Ne–Ne = 6.25 (2% low vs. 6.38), Ar–Ar = 56.0 (13% low vs.
64.3) — an honest "tens of percent" band for a modest, unfitted basis.
Refused: the analytic gradient (differentiating the BR root solve and the
Becke/Hirshfeld grid-weight response is not implemented) and the three-body
term (no established way to pair it with BJ damping — "canonical XDM" is the
two-body sum alone). XDM is a library function rather than a `Dispersion`
enum variant: D2/D3/CHG depend only on the geometry, whereas XDM's
coefficients are functionals of the converged density and can only be
evaluated after the SCF. Library callers use `xdm_dispersion_energy`;
`dft-rs energy|charges --disp xdm --xdm-a1 A1 --xdm-a2 A2` performs the
same post-SCF evaluation. Commands requiring nuclear derivatives refuse XDM
until its density/grid response is implemented.

---

## 6. Continuum solvation

| Model | Energy | Analytic gradient | Analytic Hessian |
| --- | --- | --- | --- |
| C-PCM | yes | yes | yes |
| COSMO | yes | yes | yes |
| IEF-PCM | yes | yes | yes |
| SS(V)PE | yes | yes | yes |

The C-PCM/COSMO pair has `K = S`, which is what lets its derivatives collapse
to closed forms. IEF-PCM brings in the D matrix and a non-symmetric `K`, and
its gradient and Hessian are assembled explicitly. SS(V)PE's `K` is the
symmetrization of the IEF one, so every `K_a`, `K_a^T` and `K_ab` application
is symmetrized before the same adjoint equations are solved. Both models are
gated end to end against central differences of the complete analytic
solvation gradient.

- Smooth (SWIG) cavity: switching function `h(d) = d³(10 − 15d + 6d²)`, so
  the energy is C^∞ in the nuclear coordinates and every derivative stays
  analytic.
- 179 bundled solvents by name, or descriptors supplied directly
  (`--eps`, `--refractive-index`, `--solvent-alpha/-beta/-gamma`).
- The reaction field participates in the CPKS/TDDFT response kernel, so
  polarizabilities and Hessians see the solvent relax with the density.

### SMD non-electrostatic term

- CDS radii measured for **every element up to radium (Z = 88)**.
- Atomic surface tensions: aqueous column measured directly; the non-aqueous
  column determined by regression over all 179 solvents with **exactly zero
  residual**, which is what pins the coefficients rather than merely fitting
  them.
- Pairwise terms `T(R) = exp(0.300/(R − R⁰))` implemented for **H-C, C-C and
  O-P** — the pairs whose split between the two atoms is unambiguous. This
  covers hydrocarbons.
- **Refused**: molecules containing C-N, C-O or N-O pairs in water (and
  additionally H-O, O-O in non-aqueous solvents). Their pair coefficients are
  not determined, and the atomic sum alone is wrong there by hundreds of
  cal/mol.

---

## 7. SAPT

**Which SAPT:** second order in the intermolecular operator, zeroth order in
the intramonomer correlation — i.e. SAPT0 with HF monomers (`--xc hf`) and
DFT-SAPT / SAPT(KS) with any other functional. Monomer-centred basis,
counterpoise-consistent throughout.

Every term is reported separately:

| Term | Closed shell | Open shell |
| --- | --- | --- |
| E_elst | yes | yes |
| E_exch (Heitler–London, no S² truncation) | yes | yes |
| E_ind (coupled, both directions) | yes | yes |
| E_exch-ind | yes | yes |
| E_disp (uncoupled) | yes | yes |
| E_disp (coupled, sum over TDA states) | yes | yes |
| E_exch-disp | yes | yes |
| E_int supermolecular (CP-corrected) + δ | yes | yes |

Two departures from textbook SAPT0, both toward *less* approximation: the
exchange terms are not S²-truncated (the Heitler–London and projected
constructions are evaluated exactly, carrying the overlap to all orders),
and induction is always the coupled/response value.

Two limitations: the monomer XC potential is **not** asymptotically
corrected (no LB94 / gradient-regulated shift), which production SAPT(DFT)
applies because KS induction and dispersion are sensitive to the potential's
long-range decay; and the coupled dispersion sums over **TDA** monomer
excitations rather than integrating the frequency-dependent density
susceptibility over imaginary frequency.

Open-shell monomers are selected with `--mult-a` / `--mult-b`. Every stage is
spin-resolved, and the gate on that is exact: forcing UKS on closed-shell
monomers reproduces the restricted numbers term by term.

**Refusal.** SAPT stops rather than reporting a number when a monomer's
first-order orbital rotation leaves the small-rotation regime (`|U|max ≥ 1`).
That happens when the orbital Hessian is nearly singular — typically a
partially filled degenerate shell whose symmetry-broken SCF solution is a
saddle point. The orbital-energy gaps look perfectly healthy in that case, so
a gap test does not catch it; a triplet oxygen atom beside a water reaches
|U| ≈ 19 (STO-3G) and 96 (6-31G), against ≈ 0.26 for a genuinely strongly
polarized but well-behaved lithium atom.

---

## 8. Performance

Direct and in-core ERI paths, RI-J density fitting (optional), Schwarz plus
density screening, incremental Fock, and rayon throughout the grid and
integral loops. SCF and TDA/full-RPA pick in-core automatically when the
tensor fits the budget (2 GiB by default, `--incore-mb N` to change), which
is worth about 4× up to roughly 120 basis functions. Excited states fall back
to direct non-symmetric transition-density contractions above that budget;
they do not impose a separate fixed in-core limit.

**Watch the in-core cliff.** The tensor is `n⁴` doubles, so the budget is
crossed abruptly and the fallback to recomputing every integral is far
slower. Ferrocene in 6-31G with an ECP on the metal is 132 AOs and needs
2317 MiB — 13% over the default — and that alone is the difference between
a 64-second SCF and a geometry optimization that completed no step at all
in 40 minutes. Either raise `--incore-mb`, or use `--ri-j`: for a pure GGA
there is no exact exchange, so density-fitting the Coulomb build removes
the two-electron cost entirely.

Limits: this is a from-scratch integral engine, not a tuned
production code. The working range is roughly def2-SVP on ~50 atoms and
def2-TZVP on ~25 atoms at interactive speed. Measured Hessian timings and
the reasoning behind each optimization are in [THEORY.md](THEORY.md).

---

## 9. Periodic boundary conditions

Finite-k periodic HF and generalized Kohn-Sham DFT are implemented for full
3D insulating cells. Every overlap, Hamiltonian, density, Fock, and XC matrix
remains a complex Hermitian matrix at its own k point. The inverse Bloch
transform is used only for the localized real-space J/K contractions.

| Piece | State |
|---|---|
| `Lattice` (triclinic, minimum image, shell enumeration, supercells) | yes |
| Ewald energy / forces / stress for point charges | yes |
| Charged cell: neutralising background, Makov–Payne monopole | yes, applied automatically when the net charge is non-zero |
| Electrostatic potential `phi(r)` from the periodic array | yes |
| `S(k)`, `T(k)` from image blocks, and their nuclear derivatives | yes |
| `V(k)` — nuclear attraction by Ewald splitting — and its nuclear derivatives | yes |
| `rho_mn(G) = <mu\|exp(iG.r)\|nu>` and the density's Fourier components | yes |
| Periodic Hartree `J(k)` from a given periodic density | yes |
| Periodic exact exchange `K(k)` | yes; explicit localized real-space cutoff |
| Periodic XC quadrature (LDA/GGA/meta-GGA) | yes; all-k rho, grad-rho, and tau |
| Gamma-point or finite-k HF/KS SCF | yes; integer or Fermi-Dirac occupations; CDIIS/damping and auto-released virtual level shift |
| Mermin free energy and chemical potential | yes; mesh-global electron conservation |
| Periodic ECP energy | yes; translated AO and ECP-center image sums |
| Periodic ECP analytic force | yes; derivative blocks are FD-gated, while the full multi-SCF total-energy FD comparison remains an ignored long-running gate (`PeriodicCapabilities::periodic_ecp_forces`) |
| Analytic electronic and total forces | yes; Pulay, Ewald J, K, XC, and nuclear terms for all-electron cells |
| Fixed-cell atomic optimization | yes; analytic-force L-BFGS, total energy or Mermin free energy |
| Cell-vector optimization / total electronic stress | yes for all-electron semilocal cells; analytic kinetic, attraction, Hartree, XC, metric and nuclear stress on the common driver; see [periodic.md](periodic.md#11-analytic-stress-and-cell-relaxation) for model limits |

The public Rust and Python APIs expose the uniform k mesh, boundary axes,
electronic temperature, overlap and exact-exchange cutoffs, XC grid,
convergence controls, wavefunction matrices, and analytic forces. The ASE
adapter forwards all three `Atoms.pbc` flags. Partial-PBC geometry and
prescribed point/Gaussian electrostatics now execute with true 1D/2D
boundaries, including explicitly compensated charged sources. The complete
electronic nuclear/Hartree and SCF/DFPT connections still require work;
their typed capability check does not substitute 3D periodicity.

Current electronic-driver boundaries include partial-PBC SCF integration,
empirical dispersion, and the periodic VV10/CHG nonlocal term. The
[dimensional source driver](periodic.md#1-boundary-conditions-and-electrostatics) is available separately.
[Masked Gaussian S/T operators, analytic atomic derivatives and open-direction
quadrature](periodic.md#2-electronic-basis-and-integration-grid) are also available on every public route.
Periodic Core, GWH and SAD initial Focks are implemented. SAD builds one
periodic HF Fock from on-the-fly spherical atomic densities and then uses the
ordinary finite-k occupation solver; Core remains the compatibility default.
Complex finite-k ADIIS is implemented for integer occupations and hands over
to CDIIS below a configurable commutator-error threshold. The same convex
coefficients act on all spin/k blocks, including their imaginary parts.
Fermi-Dirac runs use CDIIS because the present ADIIS objective has no entropy
curvature. Periodic SOSCF remains unavailable; damping and an auto-released
level shift are also implemented.
Named VV10/CHG functionals are rejected instead of silently returning only
their semilocal component. `PeriodicCapabilities` and `PeriodicEcpPlan` expose
these boundaries to callers without relying on error-message parsing.

Fixed-cell periodic optimization is available from Rust, CLI (`periodic
--relax`) and Python (`periodic_scf(relax=True)`). It rebuilds every
geometry-dependent cache at each line-search point and returns the exact final
`Cell`/SCF pair. Cell-vector relaxation remains an explicit refusal until the
electronic kinetic, Hartree, exchange, XC and Pulay stress terms are complete.

Verification here is by identity rather than by oracle. There is no periodic
reference implementation this project is permitted to diff against, so every
term is instead pinned to something the answer must satisfy whatever the
implementation: the rocksalt Madelung constant to ten digits, exact
independence of the Ewald splitting parameter, supercell extensivity, band
folding of a supercell Gamma spectrum onto the primitive k-points, the
molecular limit of a neutral cell, quadrature of the Ewald potential against
the analytic `V`, finite differences for every derivative at a generic `k`,
functional derivatives of the finite-k GGA/meta-GGA potentials, SCF
stationarity and number conservation, and analytic PBE0 forces against an
independent total-energy central difference.

Three of those gates are on the assembled TOTAL energy rather than on any one
operator (`tests/p20_periodic_energy.rs`), which is where a convention error
shared by several routines would otherwise hide: a neutral cell converges to
the isolated molecule and the residual falls steeply with cell size; a
supercell energy is exactly twice the primitive energy once the k mesh is
folded to match; and the analytic forces of an asymmetric cell sum to zero
while its energy is unchanged by translating the cell contents.

Those three, and the two finite-difference force gates in
`periodic::forces`, are `#[ignore]`d and run with
`cargo test --release -- --ignored`. A periodic SCF costs minutes even for
two hydrogens in a minimal basis, because the Ewald-split Coulomb operator is
rebuilt over its own lattice shells at every k point and every iteration —
that cost is set by the cell, not by the number of basis functions. Excluding
them from the default run is a statement about runtime, not about confidence.

The sharpest of them ties the three Coulomb terms together at once. Give the
electrons a density of tight normalised Gaussians and the whole
electrostatic energy has a closed form:

```text
E_nn + Tr[D V] + 1/2 Tr[D J]
    = E_Ewald({Z_A - q_A}) + 1/2 sum_A q_A^2 sqrt(2 gamma/pi)
                           - sum_A q_A Z_A 2 sqrt(gamma/pi)
```

The right-hand side comes from the point-charge code already pinned to the
Madelung constant; the left comes from three matrix routines carrying three
separately chosen `G = 0` conventions. Those conventions cancel only if all
three are right — and an error common to them shifts the total by a constant
that looks entirely physical, which is why no test of the individual terms
would catch it.

---

## 10. Surfaces: what is reachable from where

The [aligned response API tables](rust-api.md#common-calculation-driver) extend this baseline surface matrix
with dynamic and nonlinear molecular response, finite-q density and inverse
dielectric response, and uniform thermodynamic response. Their physical scope,
units, return diagnostics, and numerical acceptance are recorded separately in
[response.md](response.md) and the coverage ledger.

Three surfaces sit on one library: the Rust crate, the `dft-rs` binary,
and the Python extension (with an optional ASE adapter on top). They are not
equally wide, and the table below is the honest statement of that rather than
a promise that every capability is everywhere.

| Capability | Rust | CLI | `dft_rs.native` | ASE |
| --- | :-: | :-: | :-: | :-: |
| SCF energy, HF/DFT | yes | `energy` | `calculate` | yes |
| Analytic gradient | yes | `gradient` | `calculate(gradient=True)` | `get_forces` |
| Hessian, frequencies, IR | yes | `frequencies`, `hessian` | `hessian`, `vibrations` | `get_hessian`, `get_vibrations` |
| Geometry optimization / TS search | yes | `optimize`, `saddle` | `optimize`, `saddle` | ASE's own optimizers |
| Static polarizability | yes | `polarizability` | `polarizability` | — |
| Raman activities | yes | `raman` | `raman` | — |
| TDDFT (TDA/RPA), ECD | yes | `tddft` | `tddft` | — |
| Excited-state gradient (CIS) | yes | — | — | — |
| SAPT decomposition | yes | `sapt` | `sapt` | — |
| Continuum solvation (PCM/SMD) | yes | `--solvent` | `calculate(solvent=...)` | — |
| External electric field | yes | `--field` | `calculate(field=...)` | — |
| NAO/NPA + Wiberg | yes | `charges --nbo` | `calculate(nbo=True)` | `results["nbo"]` |
| Mulliken/Löwdin, dipole | yes | `charges` | `calculate` | `get_charges` |
| Molden / wavefunction dump | yes | `--molden`, `--dump-wfn` | `calculate(wavefunction=True)` | `results["wavefunction"]` |
| Periodic SCF energy | yes | `periodic` | `periodic_scf` | yes (`Atoms.pbc`) |
| Periodic analytic forces | yes | `periodic --forces` | `periodic_scf(forces=True)` | `get_forces` |
| Fixed-cell periodic optimization | yes | `periodic --relax` | `periodic_scf(relax=True)` | ASE optimizer over analytic forces |
| Periodic operators (S/T/V/J, derivatives) | yes | — | `periodic` | — |
| NMR shieldings (GIAO) | yes | — | `nmr` | — |

Every dash is a missing *binding*, not a missing capability: the physics is
in the crate and gated by the test suite in all of them. The Python surface
was deliberately narrow at v0.1.0 — only what an ASE-driven workflow needs —
and widened in v0.1.1 to polarizability, TDDFT/ECD, SAPT, continuum solvation
and the external field. It grows by hand: each entry point is a written
signature plus its own contract test, never a generated wrapper, which is why
it lags the CLI rather than tracking it.

The CLI is the widest surface and is the reference for what the library can
do; [cli.md](cli.md) documents it in full.

---

## 10b. Anisotropic pseudo-substituents — mechanisms and candidate fitter

A pseudo-substituent replaces a chemical group with one
monovalent site carrying an anisotropic operator. **v0.1.1 ships the wiring
and none of the physics**, which is a deliberate split: the plumbing can be
proved correct on its own, and mixing it with unfitted parameters would make
"is the model wrong or is the code wrong?" unanswerable later.

What exists: a per-atom basis override (`MolBasis::build_with_overrides`), the
`PseudoSite` / `PseudoSiteTable` / `SiteSymmetry` types, and a `SiteOperator`
that has no way to be nonzero. A site is represented as an ordinary
`Atom { z: 1 }` rather than a new pseudo-element, because `atom.z` drives the
quadrature grid in three places as well as the PCM radius, the atomic mass,
the SAD guess and the Molden writer — a Z-tag would fall to a default in every
one of them.

The gate is that "C + 3H + 1G" methane reproduces ordinary methane, in energy
and in the analytic gradient, under Hartree–Fock **and** under a grid-based
functional. The DFT half is the part a Z-tag representation would fail, so it
is not optional. Measured: identical to the last bit in both.

The reusable local-frame/Jacobian and real Wigner-D (`l = 0..4`) mechanisms
exist and have independent finite-difference, covariance and composition
gates. `parse_named_sites` also implements the specified zero-based
`"4=Me; 7=Ph"` grammar, comments, duplicate rejection and `@file` includes.
These are mechanism gates only: there is still no nonzero site operator,
Hamiltonian/gradient consumer, or accepted fitted parameter record.
`build_local_frame_with_connectivity` takes X and X's bonded reference
candidates explicitly; priority is applied only inside that list, so a remote
high-Z atom cannot select the body frame. The legacy nearest-X helper remains
source compatible but refuses when more than one possible reference remains,
rather than inventing a bond cutoff or searching every atom.

`NamedSubstituent::{Me, Et, TBu, Ph, Cf3, OMe}` therefore **refuse by name**.
Me/tBu report the missing v1 fit and acceptance gates; Et/Ph report the missing
v2 torsion/G-G data and acceptance gates. CF3 and OMe are accepted as input
names but were not defined by the Phase-4 v1/v2 model specification, so their
refusal additionally names the missing model definition, provenance and
validation programme. The CLI exposes `--pseudo-sub "4=Me"` and Python exposes
`calculate(..., pseudo_sub="4=Me")`. Both use the Rust parser and stop before
basis construction/SCF with the same typed refusal; even a future successfully
built table is guarded from becoming a silent ordinary-hydrogen calculation
until a Hamiltonian/gradient consumer exists.

The dependency-free `fit_substituent --synthetic-check` binary and
`pseudo_sub::fit` library core implement normalized weighted least squares,
Tikhonov regularization, a fixed-seed group-wise 80/20 split, diagnostic
exclusion/penalty policy, Frobenius-correct packed-symmetric B eigenvalue
projection and projected L-BFGS. A known synthetic c/B set is recovered below
1e-6 on train and held-out groups. Records embed the exact level, seed, split,
dataset fingerprint and diagnostics and are permanently labelled
`candidate-unaccepted`; they are not read by `NamedSubstituent::build`.
Physical reference evaluation (localized/masked density, ESP, repulsion
energies and forces, Casimir-Polder C6 provenance, torsion data) is not yet
implemented, so the fitter's ordinary `--input` route is a typed refusal.

---

## 11. Baseline exclusions and extended response scope

The original release excluded the first two categories below. The extended
[response implementation program](response.md) now includes double hybrids
and relativistic response, with physical backends tracked in
[the coverage ledger](response.md#response-coverage-ledger). They remain implementation work;
this historical baseline exclusion does not remove them from the objective.

- Baseline: post-HF correlation (MP2, CC, CI, CAS), including double hybrids.
- Baseline: relativistic Hamiltonians beyond ECPs.
- GPU offload — the design keeps the door open (batched SoA layouts) but no
  backend exists.
- Solvent models beyond the PCM family (no explicit solvent, no COSMO-RS).

---

## 12. Verification posture

The [implementation conventions](conventions.md#acceptance-tests) specify
the derivative and oracle gates. The [validation matrix](validation-matrix.md)
collects scoped results by provider, reference, ECP and target. The
[implementation status](development-status.md) identifies the current
release and scientific gates.

---

## 13. Physical target registry

*Physical property implementation targets.*

The [target registry](property_targets.csv) enumerates the original API
items and later additions, including continuum and ordinary Hall calculations.
Each target links to a physical family in the
[response coverage ledger](response.md#response-coverage-ledger). Its API
columns record the Rust, CLI, native Python and ASE entry points. The
availability and validation columns give their scope and evidence.

All physical observables in this registry are required calculations. A refusal
in a reference model is not a target implementation: London-orbital excited
rotatory strengths, Faraday A/B/C terms, core-level XANES and analytic periodic
strain/electric mixed derivatives require actual physical backends. Compatibility
checks still apply to mathematically defined combinations: Bloch flux
commensurability, polarization branches, isolated manifolds, thermodynamic
constraints and electronic symmetry must accompany the result.

The registry distinguishes observables, model controls, export/interface items,
and independent validation or performance algorithms. The original objective
also retains double-hybrid correlation and amplitude response, vorticity DFT,
balanced no-pair Dirac response, superconducting density and phase response,
lower-dimensional electrostatics, and finite-q/finite-temperature combinations.
The API rows are not the limit of the original objective.

### Milestone releases

`0.2.0-alpha.N` archives and `0.2.0` are verified subsets; none of them
completes the objective above. `0.2.0-alpha.1` contains:

- the common driver with its registry, named controls, JSON/text reports and
  `--save`, connected to Rust, the CLI, native Python and ASE;
- the molecular driver calculations listed in [rust-api.md#common-calculation-driver](rust-api.md#common-calculation-driver), molecular
  Fermi-Dirac references and their ensemble response, and conceptual DFT;
- the 3D finite-k periodic energy with analytic forces, electronic
  structure, bands and DOS, the range-separated screened Hartree build, the
  analytic stress tensor, fixed-cell and variable-cell relaxation, and
  Gamma-point phonons from analytic force constants for insulating LDA/GGA
  references ([periodic.md sections 9-12](periodic.md#9-range-separated-screened-hartree)).

`0.2.0-alpha.2` records the accepted source of the second alpha.
The comparison is against the actual frozen alpha.1 source archive, rather
than the older proposed milestone outline. The [changelog](../CHANGELOG.md)
lists additions and corrections; [response.md](response.md) retains measured
fits and failures. The aligned [API tables](rust-api.md#common-calculation-driver) contain 157 IDs, compared
with 151 in alpha.1. The target registry has 158 rows, compared with 149;
the response registry retains 68 families. Both registries now distinguish
28 combinations of HF/LDA/GGA/meta-GGA/hybrid/double-hybrid/learned functional
classes with restricted, unrestricted, restricted-open-shell and generalized
references. A `validated_subset` cell certifies only its documented subset.

The new milestone includes five named double-hybrid energy recipes and
static molecular correlated dipole/polarizability, higher periodic electric
and nuclear paths, finite-B/joint-field molecular and finite magnetic Gamma
nuclear response, periodic nonlocal and higher dispersion terms, magnetic
and optical-activity components, and balanced-Dirac/pairing/Nambu components.
The original dependencies remain unchanged. Physical limits, analytic
conventions and four-or-more-step error fits remain attached to each model.

The full objective is still open. The two CSV files and API scope paragraphs
remain authoritative for combinations not yet connected or accepted. The
Dirac components do not constitute interacting Dirac-Coulomb SCF; the
Born-charge lattice part does not constitute complete flexoelectricity;
supplied-space pairing/Nambu response does not constitute all SCDFT models.
Finite magnetic Gamma response does not certify arbitrary-q biased nuclear
DFPT, magnetic-BZ nuclear derivatives or the complete 1D/2D field hierarchy.

`0.2.0` packages the alpha.2 subset with the standard atom-centred grid as
the default, the functional corrections and the additions listed in the
[changelog](../CHANGELOG.md), which also states which derivative suites have
passed every-component fits on this source and which remain single-component
validation or have not been rerun.

### Required interpretation and validation

- Energy and free energy must name the ensemble and return every functional
  component. New functional classes require independent total and term-by-term
  oracles, including same-spin and opposite-spin correlation for double hybrids.
- Derivative reports carry raw stationarity, metric, number and interchange
  defects before symmetrization. A zero comparison count is not evidence of
  symmetry. All hyper-Raman field index orders must have their own evaluations.
- Production nuclear, field, strain, magnetic and mixed derivatives use analytic
  integral derivatives, automatic differentiation and physical response solves.
  Finite differences, multiple steps and Richardson extrapolation are independent
  validation routes, never the production substitute for missing kernels.
- Static beta/gamma and signed-frequency SHG, Pockels, rectification, sum/difference
  frequency, third-harmonic and Kerr processes retain intrinsic permutations.
  Finite-frequency Kleinman dispersion is not an error to symmetrize away.
- Periodic Born charges and dielectric response use physical DFT perturbations.
  Inverse dielectric matrices follow `epsilon_inverse = I + v chi_screened`.
  Proper piezoelectric and higher mixed derivatives retain convective and volume
  terms; energy elastic constants and stress-response constants include their
  different initial-stress corrections.
- Degenerate orbital, band, excited and spin manifolds retain their projected
  block operators. Band transport uses traces of velocity block products; scalar
  per-band masses are withheld when the block curvatures do not commute.
- Magnetic quantities retain paramagnetic, diamagnetic, contact, metric and
  gauge terms. Full indirect nuclear spin coupling reports FC/SD/PSO/DSO separately;
  a contact channel alone is not the total coupling. Spin susceptibility states
  whether the constraint fixes total number or each spin number separately.
- Wilson-loop flow, plaquette Chern number, gauge/translation covariance, phonon
  acoustic sum rules, particle-hole/Goldstone limits and causal frequency reversal
  require independent measurements appropriate to their physical model.
- Anharmonic intensities include mechanical and electrical contributions.
  The available static Watson VPT2 driver also includes equilibrium Coriolis
  and rotor-specific pseudopotential terms and transforms the properties with
  the same Hamiltonian. Frequency-dependent, finite-temperature and higher-order
  rotational combinations remain required targets.
  Franck-Condon factors require Eckart/Duschinsky/Doktorov and truncation diagnostics.
  EXAFS requires scattering provenance and relative-displacement Debye-Waller
  factors. XANES requires actual core initial states and transition/scattering
  theory rather than a refusal inherited from a valence-only approximation.

The registry is an implementation census, not a claim of complete support.
`pending` also covers existing standalone code that has not yet been connected
and verified against the target contract. Completion requires real calculations,
the three aligned API tables, physical acceptance tests and distribution audits.

The subsequent registry adds coherent conductivity (T144) and complex
periodic-field SCF (T145). The Chern kernel and zero-field occupied DFT slices
advance T066, while nonzero orbital-current periodic references advance R042.
These are model-specific calculations, not completion of the full magnetic,
spinor/SOC, uniform-flux, current-dependent or coherent finite-B program.
The same accepted molecular, periodic and high-order objective remains active.

T146 records the general complex spinor SCF provider and physical dimensional
HF/LDA adapter, total-number occupations, full spin-coherent exchange,
prescribed-potential and analytic nuclear Pauli SOC, Zeeman fields and their
variational external-amplitude derivative. The provider accepts arbitrary
Hermitian spin/SOC matrices and arbitrary spinor energy/Fock implementations.
This advances R048 without treating unconnected gradient/current XC,
nonlocal-core/two-electron SOC, nuclear vector-potential contacts, scalar
relativistic/Dirac terms, uniform flux or the complete spinor response hierarchy
as completed. The full accepted objective and its remaining combinations remain
tracked in both CSV registries.

T147 adds a distinct uniform-flux continuum London calculation on a 3D
finite magnetic torus. Ordered magnetic translation generators retain the
integer-flux cocycle, including odd flux. The one-Gamma finite-cell model
uses the same zero-average periodic Coulomb interaction in Hartree and
full/range exchange, and supplies uniform Zeeman and Pauli nuclear SOC with
the kinetic-momentum contact. It shares the general spinor SCF solver and
all public routes. Independent cell-size convergence and the infinite-crystal
k integral, open-axis uniform flux, core/two-electron SOC and gradient/current
XC remain required work. Acceptance status is recorded in both CSV files;
this does not complete the full accepted response objective.

The same T147 entry now includes magnetic Brillouin-zone integration with
independent k spaces, intersector full/range exchange, Gaussian auxiliary
singularity integration and XC evaluated from averaged spin densities.
Five Rust tests and HF/range/local-density public comparisons validate this
3D subset, including all seven nonzero-k crystal boundaries and four-step
derivative fits. Infinite-mesh convergence is measured separately from SCF;
open-axis uniform fields and the remaining functional/response combinations
are still required parts of the full accepted objective.

T145/T146 now include uniform fields along arbitrary wire directions and
in-plane slab fields, with analytic continuum orbital terms, spin Zeeman,
nuclear Pauli contact and independently fitted amplitude derivatives.
T147 also includes normal-flux magnetic slabs, their physical open-normal
Green operator, finite transverse charge compensation, finite Gamma and
in-plane crystal k integration with full/range intersector exchange.
Closed term oracles, four transfer meshes and four-step derivative fits
validate these HF/local-density/generic range-exchange subsets. All physical
calculations use the same driver on Rust, CLI, native Python and ASE.
This advances the original objective without declaring arbitrary functional,
ECP/current-XC or the entire molecular/periodic/high-order program complete.
The two coverage registries retain those remaining combinations.

The molecular hyperfine subset now includes contact, spin-dipolar and total
tensors, signed-isotope principal values/axes and unprojected tensor residuals
through one functional-independent kernel on every public route (T039/R036).
Closed Gaussian oracles, four-step derivative fits, seven native functional
classes and public HF/PBE0/TPSS comparisons cover this nonrelativistic collinear
subset. Degenerate electronic states need consistent state selection when
comparing frames; no degenerate-state average is implied. Relativistic
effective-spin corrections and the full magnetic response objective remain
required. Code-body overlap inspection and the resulting common-kernel
consolidations are documented in response.md; the CSVs are status ledgers,
not evidence that the implementations themselves are unique.


<!-- functional-reference-status-matrix:begin -->
### Functional and reference status cells

Both existing CSV ledgers now have 28 explicit functional/reference columns:
`hf`, `lda`, `gga`, `meta_gga`, `hybrid`, `double_hybrid`, `skala`, each crossed
with `restricted`, `unrestricted`, `restricted_open_shell`, `generalized`.
Restricted open shell means common-orbital ROHF/ROKS. Generalized means a
complex spinor reference with spin mixing; it does not imply a no-pair Dirac
or superconducting Nambu implementation. Hybrid includes range separation
only within the declared per-entry scope.

`validated_subset` certifies the specified acceptance cases only;
`implemented_subset` describes connected physical calculations with remaining
combinations; `implemented_unvalidated` has no completed acceptance for that
cell; `backend_validated` certifies kernel/oracle tests with physical/public
acceptance pending; `backend_only` is not an implemented public calculation.
`pending` denotes missing essential work, `not_audited` denotes an unresolved
class/reference audit, and `not_applicable` denotes a scalar/derived-data
entry or a reference that does not define that particular calculation.
`matrix_scope` and `matrix_evidence` give the restrictions and inspected
code/tests. Blank or unaudited evidence must never be interpreted as support.
These cells do not replace the required q/frequency/temperature/order/boundary
combination audit. They do not imply completion of the accepted objective.

Electrostriction and proper piezoelectric strain paths are implemented for
the stated static Gamma/Born-von Karman subset; independent strain acceptance
and the remaining functional/reference combinations are pending. The old
family-level `pending` label has been replaced by this actual partial scope.
<!-- functional-reference-status-matrix:end -->


## 14. Extended DFT method variants

The accepted method scope includes every family below, in addition to the
full molecular/periodic/high-order response program. These are physical
models, determinant constraints and numerical representations, rather than
a list of interchangeable functional names. Existing related components
do not certify a whole family or its response combinations.

| Family | Existing reusable components | Remaining required work |
|---|---|---|
| RKS, high-spin ROKS, UKS/BS | Common spin-resolved molecular SCF; shared-spatial high-spin ROKS; broken-symmetry seeds | Multiplet/singlet ROKS, the remaining functional/extra-term/field/periodic response combinations and independent acceptance |
| Noncollinear spinors, complex orbitals and current DFT | Spinor/SOC and London/periodic magnetic references; generic current components | Full reference/provider response combinations; complex orbital flexibility alone does not supply a current-dependent XC functional. Generalized KS nonlocal potentials and a generalized spinor determinant are distinct concepts |
| Slater/Xalpha, RS/LC and orbital-free density theory | Composable exchange/correlation and range-exchange operators | Orbital-free density minimization with its own kinetic functional and derivatives; an exchange-free Hartree orbital calculation is not orbital-free DFT |
| Local HF and OEP-EXX/DFT | Existing orbital-dependent exchange and density response | Local-potential variational optimization, potential-gauge/null-space constraints and a consistent OEP kernel; nonlocal hybrid exchange is not OEP |
| Approximate/exact spin projection | Broken-symmetry determinants and spin-exchange mapping | Explicit AP energy/forces and exact projected-density/energy prescriptions; a fitted Heisenberg J is not a projected electronic energy |
| SCDFT/Bogoliubov | Static LM2005 band gap/number/phase and supplied-space Nambu density components | Normal-density/orbital/phonon backreaction, general self-consistent pairing and causal dynamics, as already tracked in R050 |
| MC-PDFT/CAS-DFT | Ordinary electronic integrals and XC components | Multiconfigurational reduced densities, on-top pair-density functional/derivatives, double-counting definition and active-space response |
| Nonorthogonal/VB density methods | AO-metric and constrained-space linear algebra | Overlapping determinant transition densities, overlap Hamiltonian/functional prescription and variational orbital/configuration response |
| Thermal, fixed fractional occupation, FOD and ensemble/multiplet DFT | Mermin occupations/free energy and thermodynamic response; common real-collinear fixed-population energy, analytic gradient, static electric response and moving-metric Hessian/APT | Population-weighted higher/complex/periodic/correlated response, FOD spatial analysis, explicit ensemble weights, ensemble XC and multiplet stationarity. Neither thermal smearing nor fixed ordinary-XC populations establish all ensembles |
| LR/RT TDDFT and TDA | Existing TDA/RPA and response drivers | Complete functional/excited-derivative combinations, conserving real-time density propagation and field-dependent observables |
| Delta-SCF/MOM/IMOM, CVS and saddle methods | Common real collinear occupied-projector MOM/IMOM; CVS/TDA and shared core-hole relaxation | Complex/periodic and correlated state-selection response, SGM/STEP/sigma stationary objectives and independent excited-state derivatives |
| CAP/complex-scaled DFT | Complex matrix and response algebra | Biorthogonal non-Hermitian reference/metric, explicit XC analytic continuation, CAP/scale stationarity and resonance energy/width acceptance |
| DKS/DCB, X2C/DKH/ZORA | Generated finite-nucleus RKB/RMB operators and no-pair projector responses | Interacting self-consistent DKS, Breit terms, explicit two-component decoupling/picture-change providers and derivative acceptance. A projected supplied operator is not a converged DKS energy |
| NEO, positron-electron and QEDFT | Reusable electronic integrals, fields and metric response | Quantum nuclear/positron/photon spaces, species couplings, electron-proton/positron/photon XC providers, mass/charge statistics and the corresponding self-consistent responses |
| QM/MM, PCM and DFT/WF/DMFT embedding | Molecular continuum-solvent energy/gradient components | MM and polarizable coupling; subsystem kinetic/nonadditive XC, impurity/WF solvers, double-counting and bath response. PCM does not establish these other embedding models |
| Periodic/plane-wave, real-space/multiwavelet, FMO/DC, RI-J/RI-JK/COSX/Cholesky | Existing Gaussian periodic and molecular quadrature, DC subsystem and RI-J paths | Separate plane-wave/real-space/multiwavelet orbital representations, FMO inclusion/exclusion, linear-scaling acceptance, exchange fitting/seminumerical exchange and controlled Cholesky factorization/response |

The implementation separates an occupied-subspace selector from the common
SCF energy/Fock builder. Functional, spatial/particle representation,
occupation constraints and numerical acceleration must remain independently
composable. No method-name dispatcher or parent-energy substitution is used
to stand in for a missing physical functional. Arbitrary providers are
supported only when they supply the energy/action and required derivative
operators for that method. Kinetic, on-top, electron-proton, positron,
photon, current, ensemble and non-Hermitian functional prescriptions cannot
be inferred from an ordinary real ground-state XC provider.

The newly accepted real-collinear determinant subset is recorded in R069
and T159; R070-R078 and T160-T168 retain the additional required families.
All earlier response rows and unaccepted combinations remain in scope.


### Experimental non-Born-Oppenheimer multicomponent scope

The simultaneous quantum-particle model is an additional accepted method target,
not a replacement for the relativistic, generalized, positronic, FMO/DC and
other variants above. Electronic and selected nuclear orbitals are optimized
against one energy containing both kinetic operators and all Coulomb couplings.
The quantum-particle representation and interspecies functional are independent
of the ordinary electronic XC provider. EPC17 uses Eq. (13) and the original
2.35/2.4/3.2 density-fitted parameters of
[Yang et al. (2017)](https://arxiv.org/abs/1706.01836).

The first experimental model uses distinguishable normalized nuclear Hartree
orbitals. Unselected nuclei stay clamped; selecting every nucleus removes the
BO separation for the whole supplied molecule but retains its laboratory
center-of-mass kinetic energy and finite-basis localization. This is not an
internal-frame rotational/vibrational eigenstate solver. Coupled analytic
static electric response and moving-center gradients/Hessians are connected,
including expectation-position constraints and the internal stretch of an
all-quantum isotope pair at fixed nuclear COM. These accepted finite-basis
subsets and their four-step fits are recorded in `response.md`. General
internal-coordinate transformations, COM subtraction/mass polarization,
identical-nucleus exchange/spin statistics, general electron-nuclear
correlations, remaining response combinations and real-time vibronic
dynamics remain required. Electronic/nuclear orbital stationarity
does not establish all these features. Added positrons currently have
Coulomb/kinetic coupling; their correlation and annihilation are separate
required providers. EPC17 is applied only to the marked proton density.
The complete R076/T166 family remains unfinished.
