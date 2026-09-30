#!/usr/bin/env python3
"""Alternate fresh-process training runs and summarize paired latency ratios."""

import argparse
import csv
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    result = {}
    for row in rows:
        if row["status"] != "ok":
            raise ValueError(f"Failed/skipped benchmark in {path}: {row['case']}")
        key = tuple(row[field] for field in
                    ("suite", "case", "shape", "layout", "profile", "requested_threads"))
        if key in result:
            raise ValueError(f"Duplicate benchmark in {path}: {key}")
        latency = float(row["median_seconds"])
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"Invalid latency in {path}: {key}")
        result[key] = row
    if not result:
        raise ValueError(f"No measurements in {path}")
    return result


def summarize(pairs):
    baseline, candidate = pairs[0]
    keys = baseline.keys()
    for left, right in pairs:
        if left.keys() != keys or right.keys() != keys:
            raise ValueError("Benchmark inventories differ")
    summaries = []
    for key in keys:
        ratios, before, after = [], [], []
        for left, right in pairs:
            a, b = left[key], right[key]
            if a["iterations"] != b["iterations"]:
                raise ValueError(f"Training sequences differ: {key}")
            if not math.isclose(float(a["checksum"]), float(b["checksum"]),
                                rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError(f"Training loss checksums differ: {key}")
            before.append(float(a["median_seconds"]))
            after.append(float(b["median_seconds"]))
            ratios.append(after[-1] / before[-1])
        median = statistics.median(ratios)
        summaries.append({
            "case": key[1], "shape": key[2], "threads": key[5],
            "baseline_ms": statistics.median(before) * 1000,
            "candidate_ms": statistics.median(after) * 1000,
            "paired_ratio": median, "ratio_min": min(ratios),
            "ratio_max": max(ratios), "change_percent": (median - 1) * 100,
            "baseline_spread_percent": (max(before) - min(before)) /
                                       statistics.median(before) * 100,
            "candidate_spread_percent": (max(after) - min(after)) /
                                        statistics.median(after) * 100,
            "regression_over_5_percent": median > 1.05,
        })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "baseline-revision", "candidate-revision", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--threads", default="1,6,16")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("at least three fresh-process pairs are required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("OMP_", "TENSORLIB_"))}
    environment["OMP_DYNAMIC"] = "FALSE"
    for setting in args.env:
        name, separator, value = setting.partition("=")
        if not separator or not name.startswith(("OMP_", "TENSORLIB_")):
            parser.error("--env requires an OMP_* or TENSORLIB_* setting")
        environment[name] = value
    runner = Path(__file__).with_name("run_benchmarks.py")
    pairs = []
    for repeat in range(args.repeats):
        paths = {}
        order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        for variant in order:
            path = output / f"{variant}-{repeat}.csv"
            command = [sys.executable, str(runner), "--executable", getattr(args, variant),
                       "--source-revision", getattr(args, variant + "_revision"),
                       "--suite", "training", "--profile", "fixed", "--no-diagnostics",
                       "--threads", args.threads, "--csv", str(path)]
            print(f"Pair {repeat + 1}: {variant}", flush=True)
            with path.with_suffix(".log").open("w", encoding="utf-8") as log:
                subprocess.run(command, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            paths[variant] = path
        pairs.append((read_rows(paths["baseline"]), read_rows(paths["candidate"])))
    rows = summarize(pairs)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['case']} threads={row['threads']}: {row['change_percent']:+.2f}% "
              f"(paired ratios {row['ratio_min']:.3f}..{row['ratio_max']:.3f})")
    return 1 if any(row["regression_over_5_percent"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
