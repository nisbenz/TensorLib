#!/usr/bin/env python3
"""Run TensorLib benchmarks and enrich their CSV with host metadata."""

import argparse
import csv
import os
import platform
import re
import subprocess
import sys
import tempfile


def cpu_model():
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as source:
                for line in source:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return platform.processor() or platform.machine() or "unknown"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", default="build/bench_tensorlib")
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--suite", choices=("all", "kernels", "autograd", "nn", "scaling"),
                        default="all")
    parser.add_argument("--threads", help="comma-separated OpenMP thread ladder")
    parser.add_argument("--csv", default="benchmark-results.csv")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = {
        "os": platform.platform(),
        "cpu": cpu_model(),
        "logical_processors": str(os.cpu_count() or 1),
    }
    print("Host:")
    for key, value in metadata.items():
        print(f"  {key.replace('_', ' ')}: {value}")
    print()

    descriptor, raw_path = tempfile.mkstemp(prefix="tensorlib-bench-", suffix=".csv")
    os.close(descriptor)
    command = [args.executable, "--suite", args.suite, "--csv", raw_path]
    command += ["--smoke"] if args.smoke else ["--profile", args.profile]
    if args.threads:
        command += ["--threads", args.threads]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        compiler = re.search(r"^  compiler: (.+)$", result.stdout, re.MULTILINE)
        metadata["compiler"] = compiler.group(1) if compiler else "unknown"
        with open(raw_path, newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
            fields = list(reader.fieldnames or []) + list(metadata)
        with open(args.csv, "w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                row.update(metadata)
                writer.writerow(row)
        print(f"\nCSV results: {args.csv}")
        return result.returncode
    finally:
        try:
            os.unlink(raw_path)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
