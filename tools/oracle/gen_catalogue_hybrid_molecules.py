# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Resume the standard-molecule numerical catalogue for added functionals.

The shared existing geometries and spherical STO-3G basis are used unchanged.
Only selected functional entries are replaced; other frozen values are kept.
"""

import argparse
import json
from pathlib import Path

import pyscf
from pyscf import dft, gto, lib

from gen_functional_molecules import SYMBOLS, molecules
from gen_integrals import DATA_DIR, OUT_DIR, parse_g94
from gen_nmr_hybrids import MODELS

FUNCTIONALS = {"blyp35": "0.35*HF + 0.65*GGA_X_B88, GGA_C_LYP",
               "edf2": "HYB_GGA_XC_EDF2",
               "tpssloc": "MGGA_X_TPSS,MGGA_C_TPSSLOC",
               "task": "MGGA_X_TASK,LDA_C_PW",
               "mtask": "MGGA_X_MTASK,LDA_C_PW",
               "pkzb": "MGGA_X_PKZB,MGGA_C_PKZB",
               **{name: row[0] for name, row in MODELS.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", choices=sorted(FUNCTIONALS), default=["blyp35"])
    parser.add_argument("--molecules", nargs="*")
    args = parser.parse_args()
    lib.num_threads(2)
    path = Path(OUT_DIR) / "catalogue_hybrid_molecules.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.update(provenance=f"PySCF {pyscf.__version__}, libxc {dft.libxc.__version__}",
                basis="sto-3g", grid_level=9, units="Hartree and Bohr", memory_mb=256, threads=2)
    data.setdefault("molecules", {})
    data.setdefault("functionals", {})
    for name in args.names:
        case = data["functionals"].setdefault(name, {"oracle_identifier": FUNCTIONALS[name], "energies": {}, "converged": {}})
        for label, (numbers, positions) in molecules().items():
            if args.molecules and label not in args.molecules:
                continue
            data["molecules"][label] = {"numbers": numbers, "positions": positions}
            elements = {SYMBOLS[z] for z in numbers}
            basis = parse_g94(str(Path(DATA_DIR) / "sto-3g.g94"), elements)
            atom = [(SYMBOLS[z], r) for z, r in zip(numbers, positions)]
            mol = gto.M(atom=atom, basis=basis, unit="Bohr", charge=0, spin=0, cart=False, verbose=0, max_memory=256)
            mf = dft.RKS(mol)
            mf._chkfile.close()
            mf.chkfile = None
            mf.max_memory = 256
            mf.xc = FUNCTIONALS[name]
            mf.grids.level = 9
            mf.conv_tol = 1e-11
            mf.max_cycle = 300
            if label == "ferrocene":
                mf.verbose = 4
            mf.kernel()
            attempts = [{"solver": "RKS", "converged": bool(mf.converged), "energy": float(mf.e_tot)}]
            if not mf.converged:
                mf = mf.newton()
                mf.kernel()
                attempts.append({"solver": "Newton", "converged": bool(mf.converged), "energy": float(mf.e_tot)})
            assert mf.converged, (name, label)
            case["energies"][label] = float(mf.e_tot)
            case["converged"][label] = True
            case.setdefault("reference_scf_attempts", {})[label] = attempts
            path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(name, label, f"{mf.e_tot:.14f}", flush=True)


if __name__ == "__main__":
    main()
