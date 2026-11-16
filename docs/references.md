<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

# References

The theory in dft-rs is implemented from these primary sources. Where a paper
and an existing implementation disagreed, the paper won and the disagreement
was resolved by measurement against a numerical oracle — the reasoning for
each such case is recorded in [THEORY.md](THEORY.md).

No source code or embedded data is taken from libxc, libint, dftd3/dftd4 or
any comparable package; see [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)
for the provenance of the bundled *data* and for the two deliberate
exceptions (published parameter tables routed through Apache-2.0 sources,
with the primary publication cited in each file's header).

## Method and formalism

- P. Hohenberg, W. Kohn, *Phys. Rev.* **136**, B864 (1964) — inhomogeneous electron gas.
- W. Kohn, L. J. Sham, *Phys. Rev.* **140**, A1133 (1965) — self-consistent equations.
- A. Szabo, N. S. Ostlund, *Modern Quantum Chemistry*, Dover (1996) — HF/SCF formalism.
- R. G. Parr, W. Yang, *Density-Functional Theory of Atoms and Molecules*, OUP (1989).

## Integrals

- L. E. McMurchie, E. R. Davidson, *J. Comput. Phys.* **26**, 218 (1978) — Gaussian integral recursions.
- S. Obara, A. Saika, *J. Chem. Phys.* **84**, 3963 (1986) — efficient recursion.
- L. E. McMurchie, E. R. Davidson, *J. Comput. Phys.* **44**, 289 (1981) — ECP integrals.
- R. A. Shaw, J. G. Hill, *J. Chem. Phys.* **147**, 074108 (2017) — ECP integral evaluation.

## Grids

- A. D. Becke, *J. Chem. Phys.* **88**, 2547 (1988) — multicenter numerical integration.
- O. Treutler, R. Ahlrichs, *J. Chem. Phys.* **102**, 346 (1995) — radial grids.
- R. E. Stratmann, G. E. Scuseria, M. J. Frisch, *Chem. Phys. Lett.* **257**, 213 (1996) — SSF partitioning.
- V. I. Lebedev, D. N. Laikov, *Dokl. Math.* **59**, 477 (1999) — angular quadrature.

## Exchange–correlation functionals

- S. H. Vosko, L. Wilk, M. Nusair, *Can. J. Phys.* **58**, 1200 (1980) — VWN.
- J. P. Perdew, Y. Wang, *Phys. Rev. B* **45**, 13244 (1992) — PW92.
- A. D. Becke, *Phys. Rev. A* **38**, 3098 (1988) — B88 exchange.
- C. Lee, W. Yang, R. G. Parr, *Phys. Rev. B* **37**, 785 (1988) — LYP.
- J. P. Perdew, *Phys. Rev. B* **33**, 8822 (1986) — P86.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE.
- A. D. Becke, *J. Chem. Phys.* **98**, 5648 (1993) — B3 hybrids.
- N. C. Handy, A. J. Cohen, *Mol. Phys.* **99**, 403 (2001) — OPTX.
- A. D. Becke, *J. Chem. Phys.* **107**, 8554 (1997) — B97 power series.
- F. A. Hamprecht, A. J. Cohen, D. J. Tozer, N. C. Handy, *J. Chem. Phys.* **109**, 6264 (1998) — HCTH.
- J. Tao, J. P. Perdew, V. N. Staroverov, G. E. Scuseria, *Phys. Rev. Lett.* **91**, 146401 (2003) — TPSS.
- J. W. Furness, A. D. Kaplan, J. Ning, J. P. Perdew, J. Sun, *J. Phys. Chem. Lett.* **11**, 8208 (2020) — r²SCAN.
- Y. Zhao, D. G. Truhlar, *Theor. Chem. Acc.* **120**, 215 (2008) — M06 family.
- J.-D. Chai, M. Head-Gordon, *J. Chem. Phys.* **128**, 084106 (2008) — ωB97/ωB97X.
- J.-D. Chai, M. Head-Gordon, *Phys. Chem. Chem. Phys.* **10**, 6615 (2008) — ωB97X-D.
- N. Mardirossian, M. Head-Gordon, *Phys. Chem. Chem. Phys.* **16**, 9904 (2014) — ωB97X-V.
- N. Mardirossian, M. Head-Gordon, *J. Chem. Phys.* **144**, 214110 (2016) — ωB97M-V.
- T. Yanai, D. P. Tew, N. C. Handy, *Chem. Phys. Lett.* **393**, 51 (2004) — CAM-B3LYP.
- J. Heyd, G. E. Scuseria, M. Ernzerhof, *J. Chem. Phys.* **118**, 8207 (2003) — HSE.
- T. M. Henderson, B. G. Janesko, G. E. Scuseria, *J. Chem. Phys.* **128**, 194105 (2008) — HJS exchange hole.
- O. A. Vydrov, T. Van Voorhis, *J. Chem. Phys.* **133**, 244103 (2010) — VV10 nonlocal correlation.

## Dispersion

- S. Grimme, *J. Comput. Chem.* **27**, 1787 (2006) — D2.
- S. Grimme, J. Antony, S. Ehrlich, H. Krieg, *J. Chem. Phys.* **132**, 154104 (2010) — D3.
- S. Grimme, S. Ehrlich, L. Goerigk, *J. Comput. Chem.* **32**, 1456 (2011) — D3(BJ).
- E. Caldeweyher *et al.*, *J. Chem. Phys.* **150**, 154122 (2019) — D4.

## Response, derivatives, and excited states

- P. Pulay, *Mol. Phys.* **17**, 197 (1969) — analytic gradients.
- P. Pulay, *Chem. Phys. Lett.* **73**, 393 (1980) — DIIS.
- Y. Yamaguchi, Y. Osamura, J. D. Goddard, H. F. Schaefer, *A New Dimension to Quantum Chemistry: Analytic Derivative Methods*, OUP (1994).
- B. G. Johnson, M. J. Frisch, *J. Chem. Phys.* **100**, 7429 (1994) — DFT second derivatives.
- M. E. Casida, in *Recent Advances in Density Functional Methods*, World Scientific (1995) — TDDFT.
- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — Tamm–Dancoff approximation.
- N. C. Handy, H. F. Schaefer, *J. Chem. Phys.* **81**, 5031 (1984) — Z-vector method for derivatives of non-variational energies.
- F. Furche, R. Ahlrichs, *J. Chem. Phys.* **117**, 7433 (2002) — excited-state analytic gradients from the TDDFT Lagrangian.
- E. R. Davidson, *J. Comput. Phys.* **17**, 87 (1975) — iterative diagonalization.

## Solvation

- A. Klamt, G. Schüürmann, *J. Chem. Soc. Perkin Trans. 2*, 799 (1993) — COSMO.
- V. Barone, M. Cossi, *J. Phys. Chem. A* **102**, 1995 (1998) — C-PCM.
- E. Cancès, B. Mennucci, J. Tomasi, *J. Chem. Phys.* **107**, 3032 (1997) — IEF-PCM.
- D. M. Chipman, *J. Chem. Phys.* **112**, 5558 (2000) — SS(V)PE.
- J. Tomasi, B. Mennucci, R. Cammi, *Chem. Rev.* **105**, 2999 (2005) — continuum models review.
- A. W. Lange, J. M. Herbert, *J. Chem. Phys.* **133**, 244111 (2010) — smooth (SWIG) cavity.
- A. V. Marenich, C. J. Cramer, D. G. Truhlar, *J. Phys. Chem. B* **113**, 6378 (2009) — SMD.

## Intermolecular interactions

- B. Jeziorski, R. Moszynski, K. Szalewicz, *Chem. Rev.* **94**, 1887 (1994) — SAPT.
- A. Heßelmann, G. Jansen, *Chem. Phys. Lett.* **367**, 778 (2003) — DFT-SAPT.
- S. F. Boys, F. Bernardi, *Mol. Phys.* **19**, 553 (1970) — counterpoise correction.

## Periodic boundary conditions

- P. P. Ewald, *Ann. Phys.* **369**, 253 (1921) — lattice sums of the Coulomb energy.
- H. J. Monkhorst, J. D. Pack, *Phys. Rev. B* **13**, 5188 (1976) — special points for Brillouin-zone integration.
- G. Makov, M. C. Payne, *Phys. Rev. B* **51**, 4014 (1995) — periodic boundary conditions for charged systems.
- N. D. Mermin, *Phys. Rev.* **137**, A1441 (1965) — thermal properties of the inhomogeneous electron gas.
- C. Pisani, R. Dovesi, *Int. J. Quantum Chem.* **17**, 501 (1980) — exact-exchange Hartree–Fock for periodic systems.

## Response framework reference material

- [OpenRSP](https://github.com/openrsp/openrsp) documents a recursive assembly
  approach to response properties. It is additional design reference material,
  rather than a computational dependency or a numerical oracle. Physical
  equations and derivative conventions are assessed independently; its
  advertised features do not establish this project's implementation coverage.

## Basis sets

- B. P. Pritchard, D. Altarawy, B. Didier, T. D. Gibson, T. L. Windus, *J. Chem. Inf. Model.* **59**, 4814 (2019) — Basis Set Exchange.
- F. Weigend, R. Ahlrichs, *Phys. Chem. Chem. Phys.* **7**, 3297 (2005) — def2 sets.
- T. H. Dunning Jr., *J. Chem. Phys.* **90**, 1007 (1989) — correlation-consistent sets.


## Development-philosophy comparison only

- M. F. Peintinger, *vibe-qc: An Open-Source, General-Purpose
  Electronic-Structure Code for Molecules and Solids Developed With
  Generative AI*, J. Comput. Chem. **47**, e70525 (2026),
  [doi:10.1002/jcc.70525](https://onlinelibrary.wiley.com/doi/10.1002/jcc.70525).
  This is a comparison of development philosophy only. It is not an
  implementation reference, computational dependency or numerical oracle.
  Its code is not consulted, copied or translated for this project; the
  citation does not imply common implementation or validated equivalence.

## Isotropic nuclear-spin spectra

- M. H. Levitt, *Spin Dynamics: Basics of Nuclear Magnetic Resonance*,
  2nd ed., Wiley (2008), chapters 8, 9, 14 and 18 — spin Hamiltonians,
  isotropic scalar coupling, transition operators and coupled-spin spectra.
  The finite-spin matrices and independent AB/product-space acceptance are
  derived from angular-momentum algebra, without external program code.

## Constrained spin response

- M. Royo, M. Stengel, *Dynamical Response of Noncollinear Spin Systems at
  Constrained Magnetic Moments*, Phys. Rev. X **16**, 011049 (2026),
  [doi:10.1103/c68v-8tkb](https://doi.org/10.1103/c68v-8tkb),
  [arXiv:2501.10188](https://arxiv.org/abs/2501.10188) — distinct fixed-moment
  and free-moment response ensembles and their Legendre-transform relations.
  The molecular provider here imposes linear total-spin expectation
  constraints with Lagrange multipliers; this citation does not establish
  implementation of the paper's full periodic dynamical spin-lattice model.
