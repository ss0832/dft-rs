<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

# Solvent descriptors

`src/data/solvents.csv` holds 179 solvents with the descriptors the SMD
continuum-solvation model is parameterized against:

| Column | Meaning |
|---|---|
| `eps` | static dielectric constant |
| `refractive_index` | refractive index at 293 K |
| `abraham_alpha` | Abraham hydrogen-bond acidity |
| `abraham_beta` | Abraham hydrogen-bond basicity |
| `macroscopic_surface_tension_cal_per_mol_A2` | macroscopic surface tension γ |
| `phi2` | aromaticity (fraction of non-hydrogenic atoms that are aromatic C) |
| `psi2` | electronegative halogenicity (fraction that are F, Cl, Br) |

## Origin

These are the published values of the **Minnesota Solvent Descriptor
Database** (Winget, Dolney, Giesen, Cramer & Truhlar), the descriptor set
used by SMD:

> A. V. Marenich, C. J. Cramer, D. G. Truhlar,
> *J. Phys. Chem. B* **113**, 6378 (2009).

They are measured or empirically derived physical constants of the
solvents themselves — no implementation-specific content.

## Route

Transcribed via PySCF's Apache-2.0 distribution of the same table
(`pyscf.solvent.smd`), regenerable with `tools/oracle/gen_solvents.py`.
This mirrors how the project already sources its D3 parameters via MOPAC
(Apache-2.0): the primary publication is the authority, the intermediate
distribution is recorded for traceability, and both licenses are
compatible with Apache-2.0.

Retrieved 2026-08-04 from PySCF 2.14.0.

## Scope note

The descriptors are bundled for ALL listed solvents, so the electrostatic
models (C-PCM, COSMO) work for every one of them, and users may also pass
descriptors directly instead of naming a solvent. The SMD
non-electrostatic (CDS) surface-tension coefficients are currently
implemented for water only; other solvents raise an explicit error rather
than returning a silently wrong number.
