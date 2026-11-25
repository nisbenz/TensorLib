#!/usr/bin/env python3
"""Build and validate blocking variants, then rotate fresh-process matmul runs."""

import argparse
import csv
import itertools
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mc", default="32,64,128")
    parser.add_argument("--kc", default="64,128,256")
    parser.add_argument("--nc", type=int, default=64)
    parser.add_argument("--threads", default="1,6")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("at least three fresh processes per variant are required")
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=root, text=True).strip()
    if dirty:
        parser.error("commit the source before building attributable variants")
    variants = list(itertools.product(map(int, args.mc.split(",")), map(int, args.kc.split(","))))
    for mc, kc in variants:
        build = output / f"mc{mc}-kc{kc}-nc{args.nc}"
        flags = (f"-O3 -g -DNDEBUG -fno-omit-frame-pointer -DTENSORLIB_MATMUL_CONFIG_MC={mc} "
                 f"-DTENSORLIB_MATMUL_CONFIG_KC={kc} -DTENSORLIB_MATMUL_CONFIG_NC={args.nc}")
        with (output / f"build-{mc}-{kc}.log").open("w") as log:
            command = ["cmake", "-S", str(root), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                       "-DTENSORLIB_BUILD_BENCHMARKS=ON", "-DTENSORLIB_ENABLE_OPENMP=ON",
                       "-DTENSORLIB_WARNINGS_AS_ERRORS=ON", "-DTENSORLIB_NATIVE_OPTIMIZATIONS=ON",
                       "-DCMAKE_C_FLAGS_RELEASE=" + flags]
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run(["cmake", "--build", str(build), "--parallel", "2", "--target",
                            "bench_tensorlib", "test_tensor_matmul"], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
            subprocess.run(["ctest", "--test-dir", str(build), "--output-on-failure", "-R",
                            "test_(tensor_matmul|matmul_policy_)"], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        print(f"Validated MC={mc} KC={kc} NC={args.nc}", flush=True)
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("OMP_", "TENSORLIB_"))}
    environment["OMP_DYNAMIC"] = "FALSE"
    measurements = {}
    for repeat in range(args.repeats):
        order = variants[repeat % len(variants):] + variants[:repeat % len(variants)]
        if repeat % 2:
            order.reverse()
        for mc, kc in order:
            build = output / f"mc{mc}-kc{kc}-nc{args.nc}"
            path = output / f"mc{mc}-kc{kc}-run{repeat}.csv"
            command = [sys.executable, str(root / "scripts/run_benchmarks.py"), "--executable",
                       str(build / "bench_tensorlib"), "--suite", "matmul", "--profile", "quick",
                       "--threads", args.threads, "--source-revision", revision, "--csv", str(path)]
            with path.with_suffix(".log").open("w") as log:
                subprocess.run(command, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            with path.open(newline="") as source:
                for row in csv.DictReader(source):
                    if row["status"] != "ok":
                        raise ValueError("A matmul measurement failed")
                    key = (mc, kc, row["case"], row["requested_threads"])
                    measurements.setdefault(key, []).append(float(row["median_seconds"]) * 1000)
            print(f"Measured MC={mc} KC={kc} repeat={repeat + 1}", flush=True)
    with (output / "summary.csv").open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["mc", "kc", "nc", "case", "threads", "median_ms", "min_ms", "max_ms"])
        for (mc, kc, case, threads), samples in sorted(measurements.items()):
            writer.writerow([mc, kc, args.nc, case, threads, statistics.median(samples),
                             min(samples), max(samples)])


if __name__ == "__main__":
    main()
