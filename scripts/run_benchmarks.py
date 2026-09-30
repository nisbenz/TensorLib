#!/usr/bin/env python3
"""Run TensorLib benchmarks and enrich their CSV with host metadata."""

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except OSError:
        return "unavailable"


def build_metadata(executable, revision):
    path = Path(executable).resolve()
    cache = path.parent / "CMakeCache.txt"
    settings = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if re.match(r"(?:CMAKE_(?:BUILD_TYPE|C_COMPILER|C_FLAGS.*)|TENSORLIB_.*):", line):
                key, value = line.split("=", 1)
                settings[key] = value
    return {
        "source_revision": revision or "unverified",
        "checkout_revision": command_output(["git", "rev-parse", "HEAD"]),
        "checkout_status": command_output(["git", "status", "--porcelain", "--untracked-files=no"]),
        "executable_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "build_settings": json.dumps(settings, sort_keys=True),
        "cpu_topology": command_output(["lscpu", "-e=CPU,CORE,SOCKET,NODE,ONLINE"]),
        "process_affinity": str(sorted(os.sched_getaffinity(0)))
                            if hasattr(os, "sched_getaffinity") else "unavailable",
    }


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
    parser.add_argument("--profile", choices=("quick", "full", "fixed"), default="quick")
    parser.add_argument("--suite", choices=("all", "kernels", "autograd", "nn", "command", "training", "matmul", "scaling", "policy"),
                        default="all")
    parser.add_argument("--threads", help="comma-separated OpenMP thread ladder")
    parser.add_argument("--csv", default="benchmark-results.csv")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--source-revision", help="verified revision used to build the executable")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = {
        "os": platform.platform(),
        "cpu": cpu_model(),
        "logical_processors": str(os.cpu_count() or 1),
        "thread_environment": json.dumps({
            key: value for key, value in sorted(os.environ.items())
            if key.startswith(("TENSORLIB_", "OMP_"))
        }, sort_keys=True),
    }
    metadata.update(build_metadata(args.executable, args.source_revision))
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
    if args.no_diagnostics:
        command += ["--no-diagnostics"]
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
