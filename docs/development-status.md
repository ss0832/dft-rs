<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

# Implementation status for v0.4.0

The development tree starts from the supplied v0.3.1 archive, SHA-256
`7d49b90f60b76d8e4b0408f7031d62a3c98878873e173b1c3a8ddc2746956727`.
The package metadata is v0.4.0 and the license is Apache-2.0. Evidence
below refers to the source and artifact snapshots identified in the
linked acceptance records.

- [Release verification snapshot](#release-verification-snapshot)
- [References and functionals](#references-and-functionals)
- [Molecular properties](#molecular-properties)
- [Periodic properties](#periodic-properties)
- [Relativistic and quantum-nuclear models](#relativistic-and-quantum-nuclear-models)
- [API, distribution and performance](#api-distribution-and-performance)
- [Release connection inventory](#release-connection-inventory)
- [Continuous chemical species](#continuous-chemical-species)

Use the [API guide](api-guide.md) for controls and examples, the
[validation matrix](validation-matrix.md) for scoped measurements, and the
[response ledger](response.md#response-coverage-ledger) for physical targets.
In the tables below, a connected route is callable under the stated
conditions. A verified subset has completed checks for the specified
model. Fixed-density operator fits, relaxed observables and installed
interface parity have separate acceptance gates.

The [target matrix](property_targets.csv) records availability, validation
state and 32 reference/class combinations per target. The scope and
evidence columns record ECP, field and dimensional restrictions. The
[status vocabulary](conventions.md#status-vocabulary) defines the labels.

## Release verification snapshot

v0.4.0 records the completed test results available on 2026-11-16.
The larger validation program is paused. The table scopes each result
to the tested model or fixture.

| Area | Result at the release snapshot |
|---|---|
| Molecular ROKS polarizability | The recorded M06-HF, SCAN, SCAN0 and r2SCAN replays pass all 72 ordered entries. The tighter Slater/P86 replay passes all nine entries. Its original branch-crossing and fit-quality failures remain in the record. BP86 and B3P86 tighter replays had no complete acceptance at the snapshot. The complete 250-recipe ROKS sweep retains failures and 28 unsupported common-orbital PT2 Hessian requests. [Catalogue results](response.md#molecular-response-across-the-registered-functional-catalogue) |
| Molecular GKS double-hybrid gradient | B2PLYP on C1 NH3+ passes all 12 total-gradient entries. The distinct-target and nonlocal-target termwise replays were incomplete after a test comparator indexing correction. The public comparison remains unverified for those replays. [Double-hybrid results](response.md#screened-and-double-hybrid-checkpoints) |
| Constrained DFT and VV10 | Corrected omegaB97X-V water passes all nine moving-fragment gradient fits. Both mandatory functionals pass the strict public water comparison. The corrected fixed-density VV10 kernel passes 110 entries. The independent constrained-water static polarizability acceptance remains open. [Constrained DFT](response.md#r083-constrained-density-functional-theory) |
| SAPT-KS and electric multipoles | All four strict KS-FDDS public tests pass after the two molecular VV10 energies are independently reconfirmed. HF Racah ranks four and five pass 808 ordered entries across all four references in the supplemental Rust replay. Integrated all-rank validation remains open. [SAPT-KS](response.md#ks-sapt-with-frequency-dependent-density-susceptibilities) · [Multipoles](response.md#r060-molecular-multipoles-and-derivatives) |
| X2C and positron DFT | The previously stalled X2C Legacy case converges in 36 iterations. Both stationary X2C branches pass their scoped 12-component electric derivative checks. M06/RKS ammonia plus a positron passes 18 ordered tensor comparisons under tight SCF and response tolerances. [X2C](response.md#r075-breit-and-two-component-relativistic-dft) · [Multicomponent](response.md#r076-multicomponent-nuclear-positron-photon-dft) |
| Periodic GKS stress | The explicit unpruned SVWN5 control passes all nine affine derivatives and 24 rigid rotations. The default-pruned fixture exceeds the original particle quadrature gate. The shared grid fallback has a complete geometry/AO selection probe, with the end-to-end production rerun incomplete. [Periodic stress](response.md#r026-stress-elastic-high-elastic) |
| Deferred or incomplete | PZ81 and charged-Li elasticity retries are deferred. The full DFT hyperfine derivative replay remains incomplete. The ledgers retain lower-dimensional, higher-order and reference/provider validation gaps. [Validation matrix](validation-matrix.md) |

## References and functionals

| Connected routes | Verification and remaining scope |
|---|---|
| RKS/RHF, UKS/UHF, common-orbital ROKS/ROHF and generalized spinor GKS/GHF. | Each has public energy routes. Response uses the stationary equations of the selected reference where available. Constrained, spinor and periodic property intersections have separate gates. [Reference ledger](response.md#r051-common-reference-request-backend) |
| HF, LDA, GGA, meta-GGA, global and range-separated hybrids, double hybrids and Skala. | The provider's derivative, exchange, nonlocal, learned and correlation capabilities determine applicability. Acceptance includes omegaB97X-D and omegaB97X-V with their defining CHG and VV10 terms. Unsupported combinations are recorded per property. Static spinor PT2 has a common-weight subset. ROKS doubles-only energy is connected, with separate SS/OS checks. Correlated ROKS response remains open. [ROKS energy and residuals](rust-api.md#restricted-open-shell-double-hybrid-energy) [Functional catalog](FUNCTIONALS.md) |
| Explicit weighted `xc_terms` on compatible molecular and periodic routes. | Each term retains the input weight and functional specification. Energy, gradient and selected response checks exist. Nonlocal, learned, range-separated and PT2 combinations depend on their provider contracts. Named global hybrids are available through the ordinary functional control. [Molecular controls](rust-api.md#weighted-molecular-xc) and [periodic controls](rust-api.md#weighted-periodic-xc) |

The catalogue-driven molecular suite enumerates the live canonical
exchange, correlation, combined and double-hybrid recipes for RKS, UKS,
ROKS and GKS. It checks Cartesian gradients and static dipole
polarizabilities, and retains capability refusals and unresolved fits.
The frozen preflight executable passes 21 component fits each for
omegaB97X-D on RKS, UKS and ROKS, omegaB97X-V on RKS and UKS, and
M06-L on RKS.
The updated harness adds a per-component absolute error gate and raw
orbital residuals, and passes warning-strict Clippy. The common staged
gate resolves the retained UKS/M06 and RKS/SCAN interval failures.
Full unfiltered acceptance remains pending.
[Catalogue scope and measurements](response.md#molecular-response-across-the-registered-functional-catalogue)

The installed v0.3.1 checkpoint passed 24 energy and dipole cases on
distorted water across the four references with omegaB97X-D, omegaB97X-V
and omegaB97M-V. The GKS results use STO-3G, main grid level 2, VV10 grid
level 0 and zero grid-weight cutoff.

| GKS functional | Energy (Hartree) | Dipole magnitude (D) | Orbital residual (Hartree) |
|---|---:|---:|---:|
| omegaB97X-D | -75.29899432715959 | 1.78891980260743 | 2.483e-11 |
| omegaB97X-V | -75.29140227665931 | 1.75651709872665 | 2.670e-12 |
| omegaB97M-V | -75.30271770483503 | 1.74436592286897 | 5.446e-12 |

## Molecular properties

| Area | Connected routes | Verified evidence and open work |
|---|---|---|
| Nuclear derivatives, IR and VPT2 | Analytic gradients, Cartesian Hessians, full FC3 and FC4, atomic polar tensors and IR. Watson VPT2 has resonance and Coriolis corrections and bounded mechanical and electrical intensity models. | Independent all-component tests exist in selected sectors. Other reference, provider, field and ECP combinations and full VPT2 intensity acceptance remain open. [Hessian and IR](response.md#r008-hessian-apt-ir), [FC3/FC4](response.md#r010-cubic-quartic-force-constants) and [VPT2](response.md#watson-vpt2-acceptance-checkpoint) |
| Electric moments and multipole response | Permanent moments through rank eight, Buckingham quadrupole polarizability and Racah rank-one through rank-five static, retarded and imaginary-frequency polarizabilities. Ranks three, four and five give octupole, hexadecapole and 32-pole response. | The default centre of mass moves with the geometry, so a rigid coordinate translation leaves its reported values unchanged. A change of physical expansion centre mixes ranks. Rank-three through rank-five complex state sums pass an independent one-electron oracle. All-rank derivative acceptance and RKS/UKS causal perturbative PT2 multipole acceptance remain open. [Multipoles and origin](response.md#r060-molecular-multipoles-and-derivatives) |
| Electric and vibrational response | Static and frequency-dependent dipole alpha, beta and gamma, plus Raman, hyper-Raman, Pockels and electric Kerr from selected tensors. | Static correlated RKS/UKS and common-weight GKS subsets are available for the documented tensors. The collinear causal perturbative PT2 route has seven scoped regressions and 42 independent RKS real-frequency dipole component/channel fits. Damped assembled and installed parity, additional UKS cases and general correlated nonlinear tensors remain open. [Linear](response.md#r001-dipole-polarizability), [nonlinear](response.md#r002-dipole-hyperpolarizability) and [Raman](response.md#r009-raman-hyperraman) |
| Magnetic response and EPR | Orbital and spin magnetic dipoles, electronic orbital/Pauli magnetic quadrupole, magnetizability and hypermagnetizability subsets, separate dia/paramagnetic channels where available, hyperfine and EPR g-tensor routes. | Zero-bias UKS NO2 passes strict units and spin-tensor origin covariance for omegaB97X-D, omegaB97X-V and omegaB97M-V. At the centre of mass omegaB97M-V gives Qxz = 0.0250793/0.0332706 and Qyz = 0.0564514/0.0746266 a.u. for STO-3G/6-31G. C1 ammonia omegaB97X-D/GKS passes the five finite-bias modes. General spin-orbit and correlated EPR derivatives remain open. A repaired PBE hyperfine batch stopped at 6,004 of 6,840 fits, with 5,974 passes and 30 failures. [Magnetic fields](response.md#r066-finite-molecular-magnetic-field-and-gradient), [quadrupole](response.md#r082-molecular-magnetic-quadrupole) and [hyperfine evidence](molecular-property-benchmarks.md#epr-continuation-retained-failures-and-independently-checked-spin-spin-derivatives) |
| NMR and magnetic coupling | London-orbital shielding on supported references, common-origin shielding and Ramsey FC, SD, PSO and DSO terms in bounded models, isotropic spin-Hamiltonian spectra and broken-symmetry HDVV exchange. | Fixed-conformer ethyl acetate HF/STO-3G passes on all four references. The eight-proton calculation yields 28 J pairs and 11,440 transitions. OCH2 geminal J is -23.394498932083, -23.394498931862, -23.394498932083 and -23.394498930694 Hz in RKS, UKS, ROKS and GKS order. The maximum dense-oracle line-position defect is 1.65e-12 ppm. The fixture uses a closed-shell state and common origin. Genuinely open-shell NMR, broader London/PT2 coverage and general multisite exchange remain open. [NMR evidence](response.md#ethyl-acetate-four-reference-nmr-checkpoint) and [exchange](response.md#r065-spin-susceptibility-and-exchange) |
| Chiroptics and medium optics | Bounded optical rotation, ECD, MCD, VCD, Cotton-Mouton and magnetic Kerr. The medium model supports Lorentz local fields and anisotropic-slab reflection, transmission, refractive indices and absorption. | Medium predictions use supplied density, orientation, linewidth, thickness, boundary media and incidence as applicable. The complete coupled electric, magnetic and multipole constitutive model and broader spectral validation remain open. [Chiroptics](response.md#r037-ecd-vcd-roa-mcd) and [medium optics](response.md#r080-molecular-medium-optics) |

| Additional method | Connected scope and evidence |
|---|---|
| Constrained DFT | Integer RKS, UKS, ROKS and GKS support fragment charge and spin constraints. Analytic moving-fragment forces, constrained relaxation and static electric response are connected. Distorted RKS omegaB97X-D water passes all nine Cartesian gradient and nine static polarizability fits. Constrained optimization gives -75.31564752370559 Hartree at a population of 8.959999999997505. Local CLI, native Python and ASE agree for the fragment gradient and static response on both mandatory functionals. The full VV10 and GKS independent fits and correlated PT2 response remain open. [R083](response.md#r083-constrained-density-functional-theory) |
| Car-Parrinello MD | Integer real RKS/UKS/ROKS and complex GKS trajectories use a Löwdin frame and RATTLE. The real-reference off-shell component tests and RKS water timestep checks pass for both required functionals. The earlier real-reference wheel and sdist checks also pass. The GKS water trajectory passes both required functionals, with maximum energy drift below 4.337e-8 Hartree. Complex off-shell acceptance, ECP and correlated PT2 dynamics remain open. [R084](response.md#r084-car-parrinello-molecular-dynamics) |
| NEXAFS | Finite-basis collinear CVS-TDA and per-hole relaxed-core-ion models. Eight RKS water runs give first lines of 19.0521962-19.1711375 Hartree (CVS) and 19.7283482-19.8603379 Hartree (relaxed ion). Continuum, spin-orbit edges, state-specific neutral relaxation, other references and installed parity remain open. [R085](response.md#r085-near-edge-xray-absorption-fine-structure) |
| Delta-SCF and CPKS solver | Real collinear MOM/IMOM has excited-state energy and bounded derivative routes with recorded derivative failures. Orbital-gap preconditioned CG and generic CG retain fresh residual checks. CPKS reference eligibility and performance remain open. [Delta-SCF](response.md#r069-occupied-projector-delta-scf) and [CPKS](response.md#r086-preconditioned-conjugate-gradient-cpks) |
| Superconducting DFT | Static pairing, gap/number equations and Nambu-density response have implemented subsets. General self-consistent pairing and causal dynamics remain open. [R050](response.md#r050-scdft-gap-anomalous-density) |

For PBE hyperfine nuclear derivatives, the fresh first-response oracle
passes all 76 ordered views of the previously failed Hessian direction 15
on the original four steps. The scalar second-difference failure record
and the complete tensor acceptance are tracked separately.
[H15 convergence and absolute errors](molecular-property-benchmarks.md#displaced-first-response-check-of-hyperfine-h15)

## Periodic properties

| Quantity | Connected routes | Verified evidence and open work |
|---|---|---|
| Stress and SOEC | `stress` gives analytic first strain derivatives in supported 1D/2D/3D sectors. `elastic` gives clamped- and relaxed-ion second-order Lagrangian and Birch elasticity on the conventional 3D route. Uniformly compensated charged 3D, scalar ECP, and geometry-only CHG/D3 dispersion SOEC terms are connected. The dispersion gate passes 648 ordered affine and mixed fits. | Charged C1 H-Na-H RKS/B2PLYP with scalar ECP gives E = -0.706952747595285 Hartree/cell, sigma_xx = 1.8890523668e-4 and sigma_xz = -2.3484584540e-4 Hartree/Bohr^3. Neutral one-site He/SVWN5 gives Lagrangian C11 = 0.0089947628 GPa and Birch C11 = 0.0084455568 GPa. At a 24-Bohr XC radius, eight of nine measured rows in the charged Li+ xx perturbation column meet the fit cutoffs. The zy/xx row fails and the other columns remain open. Scalar-ECP SOEC acceptance remains open. [Stress and elasticity](response.md#r026-stress-elastic-high-elastic) |
| FC3/FC4 and TOEC/FOEC | `periodic-cubic-fc` and `periodic-quartic-fc` return full repeated-primitive-cell Gamma tensors. `toec` and `foec` return clamped- and relaxed-ion Lagrangian strain tensors. Extensions include uniform compensation in 3D and scalar-ECP geometry terms. | The charged monopole strain operator passes 209 independent direction fits through fourth order. Existing dispersion and selected electronic/field fixtures have bounded checks. Full charged and scalar-ECP physical tests remain open. General phonon wave-vector tuples remain unimplemented. [High-order scope](response.md#charged-gamma-force-constants-and-higher-elastic-constants) |
| Gamma phonons and scalar ECP | Conventional collinear Gamma CPKS includes the scalar-ECP fixed-density Hessian and meta-GGA kinetic-energy-density response. Correlated Gamma Hessians use a separate analytic lattice Taylor route. | The scalar-ECP fixed-density gate passes all 81 ordered components. The correlated operator gate passes 1,026 fits with 49 failed earlier intervals retained. The fixed-density split locates the older PBE discrepancy in the one-electron/Pulay group for the measured column. The revised nuclear-attraction skeleton passes 162 operator fits. PBE and TPSS pass all 324 RKS/UKS relaxed scalar-ECP Hessian entries at electronic Gamma. The maximum Richardson defect is 4.280230e-9 Hartree/Bohr^2. Finite electronic k and current installed checks are pending. Relaxed correlated Hessian and installed NaH spectrum tests remain open. [Phonon ledger](response.md#r020-dynamical-matrix-phonons) |
| Born charge, dielectric, proper piezoelectricity and crystal IR | Insulating Berry/DDK and mixed strain/displacement routes provide Born charges, clamped- and relaxed-ion response and proper piezoelectricity in supported sectors. Crystal IR and directional LO-TO have model-specific bulk and lower-dimensional paths. | The conventional scalar-ECP RKS/PBE electronic-Gamma fixture passes 72 ordered clamped electric, Born and proper piezoelectric fits. Its epsilon_xx is 1.07657009 and Na Z*_xx is 0.18579181 e. Admissible references, long-wave conventions and dimensional normalization depend on the model. Complete relaxed-ion, reference/provider, ECP, metallic and finite-field intersections remain open. [Born charge](response.md#r023-born-charge-berry-polarization), [crystal IR](response.md#r024-crystal-ir-loto), [piezoelectricity](response.md#r028-proper-piezo-electrostriction) and [dimensional Coulomb models](response.md#r044-lower-dimensional-electrostatics) |
| Finite q, DOS and thermodynamics | Commensurate finite-q density and supercell phonon routes, phonon bands and DOS, harmonic thermodynamics, and 1D/2D/3D finite-k electronic DOS. | Supercell reconstruction uses the force constants and mesh of its model. General analytic incommensurate-q phonons and full dimensional/provider coverage remain open. [Density response](response.md#r022-inverse-dielectric) and [phonons](response.md#r020-dynamical-matrix-phonons) |
| Electric, magnetic and simultaneous bias | Finite Berry electric enthalpy on supported insulating branches, commensurate magnetic field/flux models and bounded joint-field nuclear responses. | Joint fixed-cell/flux Gamma nuclear tests pass in stated subsets. Uniform compensation for conventional charged-cell Ewald and the finite-cell Berry phase/compensation tests have separate applicability rules. Broader flux, strain, provider, ECP, dimension and q coverage remain open. [Electric enthalpy](rust-api.md#nuclear-response-of-a-finite-bias-periodic-electric-enthalpy) and [joint bias](rust-api.md#simultaneous-electric-and-magnetic-bias-on-a-finite-periodic-cell) |

Stress is normalized by periodic volume, area or length in 3D, 2D or 1D.
Conventional finite-k RKS omegaB97X-V NaH passes all 15 force and ordered
strain derivative fits, with `p=1.995310..1.999916`. Its xx stress is
3.682533893015e-4 Hartree/Bohr^3. The initial dimensional RKS/UKS VV10
tests stopped at the charge check. With a shared SCF and XC overlap
metric, the RKS omegaB97X-V wire and UKS omegaB97M-V slab each pass all
18 four-step fits and the local CLI, native Python and ASE comparisons.
Their xx stresses are -0.0248069481023 Hartree/Bohr and
-0.00370379063453 Hartree/Bohr^2, respectively.
[Periodic VV10 evidence](response.md#first-geometry-derivatives-of-variational-periodic-vv10)

The 3D strain tensors use the undeformed primitive-cell volume and Voigt
order `xx, yy, zz, yz, xz, xy`. Gamma FC3 and FC4 are per primitive cell
in Hartree/Bohr^3 and Hartree/Bohr^4, respectively. The
[CLI table](cli.md#table-2-cli) documents report units and controls.

## Relativistic and quantum-nuclear models

- Finite-nucleus restricted kinetic balance and a fixed-sector no-pair
  molecular Dirac–Coulomb KS route are connected for all-electron
  even-electron GKS, with electric multipoles and linear response.
  The DKS engine uses the stationary four-component density.
  Hydrogen fluoride STO-3G energies and individual terms agree with X2C
  within 3e-10 Hartree for omegaB97X-D and omegaB97X-V. The LiH runs for
  both functionals pass public DKS parity. The paired London-RMB
  model also connects magnetizability, NMR shielding, nuclear spin
  coupling and NMR spectra in the documented subsets. Gaunt and
  Breit have tested current-interaction magnetic cases. Open-shell
  hyperfine and EPR, ECP, periodic cells and the full derivative
  matrix remain pending.
  [Four-component and Breit scope](response.md#r075-breit-and-two-component-relativistic-dft)
- Isolated even-electron scalar and spin-orbit X2C, DKH2 and finite-basis
  matrix ZORA have picture-change and uniform-electric routes. The 240
  fixed-operator fits passed, with six failed earlier intervals retained.
  Spin-orbit X2C hydrogen fluoride with omegaB97X-D/6-31G gives
  E = -100.46875015651 Hartree and dipole z = 0.860348217124161 e Bohr.
  Frozen omegaB97X-D spin-orbit X2C and omegaB97X-V scalar DKH2
  hydrogen fluoride STO-3G replays pass CLI, native Python and ASE parity
  with SCF stderr and physical units.
  Full relaxed electric and public acceptance remain open after a
  displaced-field SCF failure.
  [Two-component scope](response.md#r075-breit-and-two-component-relativistic-dft)
- Multicomponent electron/quantum-nuclear SCF and bounded coupled
  derivative and response routes exist. Joint fixed no-pair HF has
  measured energy, dipole and static/retarded electric subsets.
  Complete nuclear Hessian, dynamic, correlated, Breit/ECP and external
  field combinations remain open.
  [Multicomponent scope](response.md#r076-multicomponent-nuclear-positron-photon-dft)
  and [joint response](response.md#r081-relativistic-quantum-nuclear-response)

## API, distribution and performance

- The [aligned Rust](rust-api.md#table-1-rust), [CLI](cli.md#table-2-cli)
  and [native Python/ASE](python-api.md#table-3-native-python-and-ase)
  tables identify common-driver routes and their controls. The pip
  distribution is `dft-rs-python` and its installed command is
  `dft-rs`. SCF diagnostics go to stderr by default.
- Separate clean wheel and sdist installations passed the expanded
  omegaB97X-D and omegaB97X-V CLI/native checks for the frozen
  constrained DFT and real-reference Car-Parrinello candidate. The
  earlier periodic meta-GGA candidate wheel passed 80 installed guard
  tests. Full warning-strict regression, installed parity on the integrated
  source and the final distribution audit remain open.
- A frozen development snapshot with v0.3.1 metadata built an sdist and a
  wheel from that sdist. The sdist's 1,840 source members agreed with the
  frozen inventory by SHA-256. All 17 public documentation files matched
  the frozen bytes in separate clean installations from the wheel and the sdist.
  Both installed `dft-rs` executables reported the snapshot version.
  The 1,841 source ZIP members, 1,841 sdist members and 36 wheel members,
  including the wheel's native extension, passed a scan for local host and
  account paths, build paths and common private key and credential signatures.
- The Linux x86-64 finite-q TPSS checkpoint with v0.3.1 metadata passes
  separate clean wheel and sdist installations. The 1,848 archived source
  files and 17 installed public documentation files match the frozen
  inventory. The installed `dft-rs` reports its version, registry, default
  SCF stderr records and error exit status. The PBE H2 checks pass for
  molecular energy, periodic energy and stress. With H–H distance 0.74
  Angstrom, STO-3G and grid level 0, the molecular energy is
  `-1.1511237143257742` Hartree. In the cubic 5 Angstrom cell, the
  energy is `-1.1510280956413386` Hartree per cell, `stress_zz` is
  `1.2689504842351027e-6` Hartree/Bohr^3 and the CLI orbital residual
  is `1.478e-16` Hartree. The skew H2 TPSS case
  agrees across the installed command, standalone Rust CLI, native Python
  and ASE. [Finite-q artifact evidence](response.md#meta-gga-current-response-at-finite-q)
- The current preflight ZIP and source distribution with v0.3.1 metadata
  match all 1,855 source files by SHA-256, including 17 public
  documentation files. The archives pass member, metadata, portability
  and privacy checks. With the version-matched frozen Rust executable and
  native extension, all 62 package and release-tool regressions pass.
- The repeated-spin-block eigensolver regression passes. Five alternating
  trials per functional show shorter wall time and higher process RSS on
  closed-shell GKS water for omegaB97X-D, omegaB97X-V and omegaB97M-V.
  The broader noncollinear and response workloads have separate performance
  gates. [Matched benchmark results](performance.md#closed-shell-molecular-gks)
- Molecular VV10 reuses the analytic density fields for the Fock
  kernel. The 110 independent density-Hessian entries pass. A
  five-pair omegaB97X-V/ROKS ammonia-cation test reduces median SCF
  time from 47.00 to 6.41 seconds with the same median memory,
  energy, iteration count and raw residual. The optimized ROKS
  polarizability also passes the nine-component fit and public replay.
  [Matched VV10 benchmark](performance.md#molecular-vv10-response-in-roks-scf)

## Release connection inventory

Run `python tools/ledger_check.py --release-gaps` from the source root.
The JSON includes the full CLI and API registry, each target's four public entries, all 32
functional/reference cells, the scope, and the evidence and tests from
the existing ledgers. The check exits nonzero if a ledger reference is
invalid. Use the linked response sections for conditions outside the
matrix, such as dimension, wave vector, frequency, temperature, nuclear
model and external fields.

The registry has 184 entries, including 157 common-driver entries and
27 dedicated entries. The standalone CLI itself has no ASE counterpart.
The inventory identifies each dedicated entry and its public wrappers.
The 194 physical targets include 188 with conditional support, one
excluded solvent parameterization, and five with their full public methods
still missing.

| Target | Missing public calculation |
|---|---|
| T160 | Orbital-free density functional based on a kinetic-energy density functional |
| T161 | Optimized local effective potential |
| T164 | Real-time saddle and non-Hermitian DFT |
| T167 | Subsystem wavefunction and environment embedding |
| T168 | The orbital representations, fragmentation and exchange acceleration method family recorded in the ledger |

Finite-wave-vector orbital susceptibility and uniform-flux SCF use
different reference and response paths.

| Target | Public calculation | Remaining connection |
|---|---|---|
| T148 | `orbital-susceptibility-q` | The real collinear Gamma supercell returns two diagonal transverse responses and has scoped SVWN5, TPSS and M06-L validation. The spinor solve returns the full transverse tensor. Noncollinear SVWN5 and closed-shell PBE and TPSS HeH+ each pass nine independent finite-field fits. M06-L, omegaB97X-D and GHF pass spinor public-route comparisons. Noncollinear GGA/meta-GGA, density-dependent nonlocal XC, SOC and ROKS have separate pending acceptance. [Finite-q evidence](response.md#commensurate-wave-vectors) |

The uniform-flux SCF driver T147 uses the shared covariant XC evaluator
in its finite-Gamma and magnetic Bloch models. The one-electron slab
calculations give 1.095635224295 Hartree with BLYP and 1.076537563921
Hartree with M06-L across CLI, native Python and ASE. The skew bulk
test verifies all four BLYP and M06-L spin-density Fock components.
On the skew H2 slab with a 1024 MiB managed limit, CLI, native Python
and ASE give 1.939727501367524 Hartree at finite Gamma and
1.9397275013680506 Hartree on a 2 x 1 x 1 magnetic Bloch mesh.
The spatial Fock defects are about 2.85e-11 Hartree. A shared evaluation
of the London AO values and kinetic momentum reduces the median
one-electron M06-L CLI time from 44.71 to 24.97 seconds in five
alternating trials per build. Wheel and sdist replay for the two-center
input remains open.
[Uniform-flux verification](response.md#covariant-semilocal-scf-on-uniform-flux-cells)

The molecular `spin-orbit` route T068 now uses the spin-covariant GKS
reference with the generated one-electron operator. The measured
subset covers BLYP, M06-L, omegaB97X-D and omegaB97X-V on water.
[Acceptance data](response.md#generated-one-electron-soc-in-molecular-gks)

Connecting these cells requires verifying the matching magnetic or
spinor provider for each calculation. The release inventory reports the
current pending and unaudited cell counts in `method_cell_counts`.
Each bounded supported cell retains its own scope. An unaudited cell
requires a source or runtime check before it is assigned a release status.

T145 now connects `periodic-field-scf` with an explicit GKS or GHF
reference to the existing complex spinor provider. BLYP and M06-L
helium-chain runs agree with the canonical T146 `periodic-spinor-scf`
route across the Rust CLI, native Python and ASE.
[Controls and measured checkpoints](response.md#explicit-generalized-reference-on-static-periodic-field-scf)

The previously missing entries for T165, T177 and T188 are present in the
source. The accepted subsets use `dirac-dks` and
`two-component-relativistic`, `multicomponent`, and `polarizability`
respectively. The last route already calls the orbital-gap preconditioned
block CG solver for integer collinear static dipole response.

## Continuous chemical species

This work is paused at the geometry checkpoint. The fixed-electron
reduced model has a neutral constrained affine minimum of
-172.09004009205933 Hartree with maximum gradient 3.271606e-6 Hartree/Bohr
and 27 positive constrained modes. The explicit UKS radical-cation
optimization completed 197 accepted steps at -171.8434973739514 Hartree
with maximum gradient 8.245932e-6 Hartree/Bohr. Its final Hessian and
optimized reaction thermochemistry remain open. The checkpoint retains
the model-domain and ranking limitations.
[Continuous-species ledger](response.md#r079-continuous-chemical-species-model)
