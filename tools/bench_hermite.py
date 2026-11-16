# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Compare RTensor against a supplied baseline hermite.rs without a crate build.

Both implementations use the current math and Boys modules, the same compiler
and optimized flags. Compile/run their unit tests, compare every valid tensor
entry bitwise, then time alternating baseline/current batches in one process.
Results include raw samples, hashes, compiler and thread/environment metadata.
This is a kernel benchmark, not a whole-program speedup measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile


HARNESS = r'''
use std::{hint::black_box, time::Instant};
use math::Vec3;
use integrals::boys::boys;

fn verify() {
    let mut entries = 0usize;
    for l in 0..=24 {
        for (p, pc) in [
            (0.013, Vec3::zero()),
            (0.17, Vec3::new(0.31, -0.83, 0.57)),
            (1.7, Vec3::new(-0.21, 0.73, -0.46)),
            (3.4, Vec3::new(1.13, -0.37, 0.91)),
        ] {
            for synthetic in [false, true] {
                let f = if synthetic {
                    (0..=l).map(|n| (n as f64 + 0.3).sin()).collect()
                } else { boys(l, p * pc.norm2()) };
                let old = baseline::RTensor::build(l, p, pc, &f);
                let new = integrals::hermite::RTensor::build(l, p, pc, &f);
                for t in 0..=l { for u in 0..=l-t { for v in 0..=l-t-u {
                    assert_eq!(old.get(t,u,v).to_bits(), new.get(t,u,v).to_bits(),
                        "l={l} p={p} pc={pc:?} synthetic={synthetic} t={t} u={u} v={v}");
                    entries += 1;
                } } }
            }
        }
    }
    println!("verified {entries}");
}

fn measure(l: usize, count: usize, old: bool) -> (f64, f64) {
    let p = black_box(1.7);
    let pc = black_box(Vec3::new(0.31, -0.83, 0.57));
    let f = boys(l, p * pc.norm2());
    macro_rules! batch {
        ($tensor:ty) => {{
            let start = Instant::now();
            let mut checksum = 0.0;
            for _ in 0..count {
                let rt = <$tensor>::build(black_box(l), p, pc, black_box(&f));
                checksum += black_box(&rt).get(l, 0, 0);
            }
            (start.elapsed().as_secs_f64(), black_box(checksum))
        }};
    }
    if old { batch!(baseline::RTensor) } else { batch!(integrals::hermite::RTensor) }
}

fn main() {
    verify();
    let args: Vec<String> = std::env::args().collect();
    let samples: usize = args[1].parse().unwrap();
    let seconds: f64 = args[2].parse().unwrap();
    if samples == 0 { return; }
    for l in [0, 2, 4, 6, 8, 12, 16, 24] {
        let elapsed = measure(l, 5000, true).0;
        let count = ((5000.0 * seconds / elapsed) as usize).clamp(500, 10_000_000);
        // Warm both paths before recording samples.
        black_box(measure(l, count, true));
        black_box(measure(l, count, false));
        for sample in 0..samples {
            let order = if sample % 2 == 0 { [true, false] } else { [false, true] };
            for old in order {
                let (elapsed, checksum) = measure(l, count, old);
                println!("{l} {count} {old} {elapsed} {checksum}");
            }
        }
    }
}
'''


def run(command):
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True,
                              encoding="utf-8")
    except subprocess.CalledProcessError as error:
        print(error.stdout, file=sys.stderr)
        print(error.stderr, file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="preserved baseline src/integrals/hermite.rs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=0.15,
                        help="approximate duration of each measured batch")
    parser.add_argument("--contention-note", default="not recorded")
    parser.add_argument("--verify-only", action="store_true",
                        help="run regressions and bitwise checks without timing")
    args = parser.parse_args()
    if args.samples < 2 or not 0 < args.seconds < float("inf"):
        parser.error("use at least two samples and a positive finite batch duration")
    root = Path(__file__).resolve().parents[1]
    cpu_model = os.environ.get("PROCESSOR_IDENTIFIER", platform.processor())
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cpu_model = next((line.partition(":")[2].strip()
                          for line in cpuinfo.read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    sources = {"math": root / "src/math.rs", "boys": root / "src/integrals/boys.rs",
               "hermite": root / "src/integrals/hermite.rs",
               "baseline": args.baseline.resolve()}
    result = {"compiler": run(["rustc", "-Vv"]).stdout,
              "platform": platform.platform(), "processor": platform.processor(),
              "cpu_model": cpu_model,
              "logical_cpus": os.cpu_count(), "threads": 1,
              "benchmark": {"samples_per_version": 0 if args.verify_only else args.samples,
                            "target_batch_seconds": args.seconds,
                            "orders": [0, 2, 4, 6, 8, 12, 16, 24],
                            "p": 1.7, "pc": [0.31, -0.83, 0.57]},
              "environment": {k: os.environ.get(k) for k in
                              ["RAYON_NUM_THREADS", "DFT_RS_FAER_THREADS"]},
              "contention_note": args.contention_note,
              "sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest()
                         for k, p in sources.items()}, "cases": {}}
    with tempfile.TemporaryDirectory(prefix="dftrs-hermite-") as scratch:
        scratch = Path(scratch)
        # Preserve all input bytes during compilation, including concurrent edits.
        for name, source in list(sources.items()):
            copy = scratch / f"{name}.rs"
            content = source.read_bytes()
            if hashlib.sha256(content).hexdigest() != result["sha256"][name]:
                raise RuntimeError(f"source changed while snapshotting: {name}")
            copy.write_bytes(content)
            sources[name] = copy
        path = lambda k: json.dumps(sources[k].as_posix())
        header = (f'#[path={path("math")}] pub mod math;\n'
                  f'#[path={path("baseline")}] pub mod baseline;\n'
                  'pub mod integrals {\n'
                  f'#[path={path("boys")}] pub mod boys;\n'
                  f'#[path={path("hermite")}] pub mod hermite;\n' + '}\n')
        source = scratch / "benchmark.rs"
        source.write_text(header + HARNESS, encoding="utf-8")
        suffix = ".exe" if os.name == "nt" else ""
        exe = scratch / f"benchmark{suffix}"
        flags = ["rustc", "--edition=2021", "-Dwarnings", "-Copt-level=3",
                 "-Ccodegen-units=1", "-Clto=fat"]
        result["compiler_flags"] = flags[1:]
        tests = scratch / f"tests{suffix}"
        run([*flags, "--test", str(source), "-o", str(tests)])
        result["regression_output"] = run([str(tests), "--nocapture"]).stdout
        run([*flags, str(source), "-o", str(exe)])
        output = run([str(exe), str(0 if args.verify_only else args.samples),
                      str(args.seconds)]).stdout
        for line in output.splitlines():
            if line.startswith("verified "):
                result["bitwise_entries"] = int(line.split()[1])
                continue
            l, count, old, elapsed, checksum = line.split()
            case = result["cases"].setdefault(l, {"count": int(count),
                                                "before": [], "after": []})
            case["before" if old == "true" else "after"].append(float(elapsed))
            if "checksum" in case and checksum != case["checksum"]:
                raise RuntimeError(f"benchmark checksum changed at order {l}")
            case["checksum"] = checksum
        for l, case in result["cases"].items():
            case["before_median"] = statistics.median(case["before"])
            case["after_median"] = statistics.median(case["after"])
            case["speedup"] = case["before_median"] / case["after_median"]
            print(f'order={l:>2} speedup={case["speedup"]:.3f}x', flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f'bitwise-identical entries: {result["bitwise_entries"]}')
    print(result["regression_output"].strip().splitlines()[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
