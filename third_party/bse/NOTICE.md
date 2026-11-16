<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

# Basis Set Exchange data notice

The Gaussian94-format basis-set files under `src/basis/data/` were downloaded
from the **Basis Set Exchange** (https://www.basissetexchange.org, BSE
version 0.12, downloaded 2026-08-03) with the basis fetch script (now `tools/fetch_basis.py`). Each file
carries its own PROVENANCE header with the exact API URL.

BSE distributes its data under the **CC0-1.0 public-domain dedication**.
Attribution is not legally required but is requested; dft-rs cites:

- B. P. Pritchard, D. Altarawy, B. Didier, T. D. Gibson, T. L. Windus,
  *A New Basis Set Exchange: An Open, Up-to-date Resource for the Molecular
  Sciences Community*, J. Chem. Inf. Model. **59**, 4814–4820 (2019).
- D. Feller, J. Comput. Chem. **17**, 1571–1586 (1996).
- K. L. Schuchardt et al., J. Chem. Inf. Model. **47**, 1045–1052 (2007).

Users of specific basis sets should additionally cite the original
basis-set literature listed by BSE for each set (shown in the BSE web
interface per set and element).

Bundled families (all elements each set provides): STO-nG, Pople
(3-21G … 6-311++G(3df,3pd)), Karlsruhe def2 (SV(P) … QZVPPD), Dunning
cc-pVnZ / aug-cc-pVnZ (n = D…5), LANL2DZ/LANL2TZ/LANL08 (with ECPs), and
Stuttgart RLC / RSC 1997 (with ECPs). The composite "SDD" alias was not
retrievable from the BSE API at download time (server error) and may be
added later.


## Pinned expansion of 2026-10-29

Thirty additional orbital bases are included, with every element provided
by the selected BSE version:

| Family | Added names | Pinned data version |
| --- | --- | --- |
| Segmented polarization consistent | pcseg-0 through pcseg-4 | 1 |
| Diffuse segmented polarization consistent | aug-pcseg-0 through aug-pcseg-4 | 1 |
| Shielding optimized | pcSseg-0 through pcSseg-4 | 1 |
| Core-valence | cc-pCVDZ, cc-pCVTZ, cc-pCVQZ | 0 |
| Weighted core-valence | cc-pwCVDZ, cc-pwCVTZ, cc-pwCVQZ | 0 |
| Diffuse core-valence | aug-cc-pCVDZ, aug-cc-pCVTZ, aug-cc-pCVQZ | 0 |
| Douglas-Kroll optimized orbitals | cc-pVDZ-DK, cc-pVTZ-DK, cc-pVQZ-DK | 0 |
| J-coupling optimized | pcJ-0, pcJ-1, pcJ-2 | 1 |

The official BSE API identified its exporter as version 0.12. Each new
Gaussian94 file records the exact versioned URL, retrieval date and SHA-256
of the unmodified export. `catalogue.csv` records the complete-file hashes,
supported atomic numbers and upstream names. The separate public JSON
exports provide the numerical oracle in
`tests/oracle_data/bse_extended_bases.json`. All 860 element records and
49,901 nonzero exponent/coefficient pairs agree exactly with Gaussian94
in Decimal arithmetic before registration; p285 independently checks all
parsed numerical pairs without a network connection.

The upstream repository distributes its contents with the BSD 3-Clause
notice at [its LICENSE](https://github.com/MolSSI-BSE/basis_set_exchange/blob/master/LICENSE);
that notice is retained in `LICENSE` in this directory. No BSE Python
library source is included or needed. The historical download headers above
retain their original CC0 labels; the new files explicitly point to the
upstream redistribution notice rather than inferring an additional data
license. Original basis-set authors and references remain those recorded
by BSE; users should cite the original basis publications for their sets.

Element coverage is not filled in or interpolated: some core-valence sets
do not include H or He. Use the existing per-element basis selection when
needed. Selecting a DK-optimized orbital basis does not activate a
relativistic Hamiltonian or a kinetic/magnetic-balance transformation.
These are separate calculation controls and acceptance scopes.
