# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Download the bundled basis-set families from the Basis Set Exchange
(Gaussian94 format, all elements each set supports) and write them into
src/basis/data/ with PROVENANCE headers. Re-run manually to refresh; the
resulting diffs are reviewed like code changes. Standard library only.

Usage: python tools/fetch_basis.py [stem ...]
"""
import datetime
import hashlib
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "src" / "basis" / "data"

# file stem -> BSE basis name (as accepted by the API path)
SETS = {
    # Minimal / Pople family
    "sto-3g": "STO-3G",
    "sto-6g": "STO-6G",
    "3-21g": "3-21G",
    "4-31g": "4-31G",
    "6-31g": "6-31G",
    "6-31g_d": "6-31G*",
    "6-31g_d_p": "6-31G**",
    "6-31pg_d": "6-31+G*",
    "6-31pg_d_p": "6-31+G**",
    "6-31ppg_d_p": "6-31++G**",
    "6-311g": "6-311G",
    "6-311g_d_p": "6-311G**",
    "6-311pg_d_p": "6-311+G**",
    "6-311ppg_d_p": "6-311++G**",
    "6-311ppg_2d_2p": "6-311++G(2d,2p)",
    "6-311ppg_3df_3pd": "6-311++G(3df,3pd)",
    # Karlsruhe def2 family
    "def2-sv_p": "def2-SV(P)",
    "def2-svp": "def2-SVP",
    "def2-svpd": "def2-SVPD",
    "def2-tzvp": "def2-TZVP",
    "def2-tzvpp": "def2-TZVPP",
    "def2-tzvpd": "def2-TZVPD",
    "def2-tzvppd": "def2-TZVPPD",
    "def2-qzvp": "def2-QZVP",
    "def2-qzvpp": "def2-QZVPP",
    "def2-qzvpd": "def2-QZVPD",
    "def2-qzvppd": "def2-QZVPPD",
    # Dunning correlation-consistent family
    "cc-pvdz": "cc-pVDZ",
    "cc-pvtz": "cc-pVTZ",
    "cc-pvqz": "cc-pVQZ",
    "cc-pv5z": "cc-pV5Z",
    "aug-cc-pvdz": "aug-cc-pVDZ",
    "aug-cc-pvtz": "aug-cc-pVTZ",
    "aug-cc-pvqz": "aug-cc-pVQZ",
    "aug-cc-pv5z": "aug-cc-pV5Z",
    # ECP basis families
    "lanl2dz": "LANL2DZ",
    "lanl2dzdp": "LANL2DZdp",
    "lanl2tz": "LANL2TZ",
    "lanl2tz_f": "LANL2TZ(f)",
    "lanl2tzp": "LANL2TZ+",
    "lanl08": "LANL08",
    "lanl08_d": "LANL08(d)",
    "lanl08_f": "LANL08(f)",
    "lanl08p": "LANL08+",
    "sdd": "SDD",
    "stuttgart-rlc": "Stuttgart RLC",
    "stuttgart-rsc-1997": "Stuttgart RSC 1997",
    # Auxiliary (density-fitting) bases
    "def2-universal-jfit": "def2-universal-JFIT",
    "def2-universal-jkfit": "def2-universal-JKFIT",
    # Segmented polarization-consistent orbital and shielding bases.
    "pcseg-0": "pcseg-0",
    "pcseg-1": "pcseg-1",
    "pcseg-2": "pcseg-2",
    "pcseg-3": "pcseg-3",
    "pcseg-4": "pcseg-4",
    "aug-pcseg-0": "aug-pcseg-0",
    "aug-pcseg-1": "aug-pcseg-1",
    "aug-pcseg-2": "aug-pcseg-2",
    "aug-pcseg-3": "aug-pcseg-3",
    "aug-pcseg-4": "aug-pcseg-4",
    "pcsseg-0": "pcSseg-0",
    "pcsseg-1": "pcSseg-1",
    "pcsseg-2": "pcSseg-2",
    "pcsseg-3": "pcSseg-3",
    "pcsseg-4": "pcSseg-4",
    # Core-valence correlation, scalar-relativistic and J-coupling bases.
    "cc-pcvdz": "cc-pCVDZ",
    "cc-pcvtz": "cc-pCVTZ",
    "cc-pcvqz": "cc-pCVQZ",
    "cc-pwcvdz": "cc-pwCVDZ",
    "cc-pwcvtz": "cc-pwCVTZ",
    "cc-pwcvqz": "cc-pwCVQZ",
    "aug-cc-pcvdz": "aug-cc-pCVDZ",
    "aug-cc-pcvtz": "aug-cc-pCVTZ",
    "aug-cc-pcvqz": "aug-cc-pCVQZ",
    "cc-pvdz-dk": "cc-pVDZ-DK",
    "cc-pvtz-dk": "cc-pVTZ-DK",
    "cc-pvqz-dk": "cc-pVQZ-DK",
    "pcj-0": "pcJ-0",
    "pcj-1": "pcJ-1",
    "pcj-2": "pcJ-2",
}

# Pin the added data independently of future upstream updates.
PINNED_VERSIONS = {
    name: 1 if name.startswith(("pcseg-", "aug-pcseg-", "pcsseg-", "pcj-")) else 0
    for name in SETS
    if name.startswith(("pcseg-", "aug-pcseg-", "pcsseg-", "pcj-", "cc-pcv", "cc-pwcv", "aug-cc-pcv"))
    or name.endswith("-dk")
}

HEADER = """!----------------------------------------------------------------------
! PROVENANCE
!   Source:      Basis Set Exchange (https://www.basissetexchange.org)
!   Basis:       {bse}
!   API URL:     {url}
!   Downloaded:  {date}
!   License:     {license}
!   BSE version: {version}
!   Data SHA256: {digest}
!   Citation:    B. P. Pritchard, D. Altarawy, B. Didier, T. D. Gibson,
!                T. L. Windus, J. Chem. Inf. Model. 59, 4814 (2019).
!   Elements:    all elements provided by BSE for this set
!----------------------------------------------------------------------
"""


def fetch(stem, bse, date):
    version = PINNED_VERSIONS.get(stem)
    url = (
        "https://www.basissetexchange.org/api/basis/"
        + urllib.parse.quote(bse, safe="")
        + "/format/gaussian94/?header=true"
        + (f"&version={version}" if version is not None else "")
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        body = response.read().decode("ascii")
    if "****" not in body:
        raise ValueError("response does not look like Gaussian94 data")
    text = HEADER.format(
        bse=bse, url=url, date=date,
        license="BSD-3-Clause upstream notice; see third_party/bse/NOTICE.md",
        version=version if version is not None else "latest at retrieval",
        digest=hashlib.sha256(body.encode("ascii")).hexdigest(),
    ) + "\n" + body
    (DEST / f"{stem}.g94").write_bytes(text.encode("ascii"))
    return len(body)


def main(argv):
    wanted = argv or list(SETS)
    unknown = [stem for stem in wanted if stem not in SETS]
    if unknown:
        print("unknown basis stems:", ", ".join(unknown))
        return 2
    DEST.mkdir(parents=True, exist_ok=True)
    date = datetime.date.today().isoformat()
    failed = []
    for stem in wanted:
        try:
            size = fetch(stem, SETS[stem], date)
            print(f"{stem:<20} OK  ({size} bytes)")
        except Exception as error:  # report every failure, then exit non-zero
            failed.append(f"{stem} <- {SETS[stem]} : {error}")
            print(f"{stem:<20} FAILED: {error}")
        time.sleep(0.3)  # be polite to the BSE API
    if failed:
        print("\nFailed downloads:")
        for line in failed:
            print("  " + line)
        return 1
    print("\nAll basis sets downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
