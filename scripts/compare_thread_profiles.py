#!/usr/bin/env python3
"""Compare explicitly defined thread profiles in rotated fresh processes."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from compare_training import read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "source-revision", "profiles", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("at least three fresh processes per profile are required")
    profiles = json.loads(Path(args.profiles).read_text())
    names = [p["name"] for p in profiles]
    if not names or any(not n for n in names) or len(set(names)) != len(names):
        parser.error("profile names must be nonempty and unique")
    for profile in profiles:
        if profile["reference"] not in names or int(profile["threads"]) < 1:
            parser.error("each profile requires a valid reference and positive thread count")
        if any(not k.startswith(("OMP_", "TENSORLIB_")) for k in profile["environment"]):
            parser.error("profile settings must be OMP_* or TENSORLIB_*")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "profiles.json").write_text(json.dumps(profiles, indent=2) + "\n")
    inherited = {k: v for k, v in os.environ.items() if not k.startswith(("OMP_", "TENSORLIB_"))}
    results = {}
    for repeat in range(args.repeats):
        order = profiles[repeat % len(profiles):] + profiles[:repeat % len(profiles)]
        if repeat % 2:
            order.reverse()
        for profile in order:
            environment = dict(inherited, OMP_DYNAMIC="FALSE")
            environment.update(profile["environment"])
            path = output / f"profile{names.index(profile['name'])}-run{repeat}.csv"
            command = [sys.executable, str(Path(__file__).with_name("run_benchmarks.py")),
                       "--executable", args.executable, "--source-revision", args.source_revision,
                       "--suite", "training", "--profile", "fixed", "--no-diagnostics",
                       "--threads", str(profile["threads"]), "--csv", str(path)]
            print(f"Repeat {repeat + 1}: {profile['name']}", flush=True)
            with path.with_suffix(".log").open("w") as log:
                subprocess.run(command, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            results[(profile["name"], repeat)] = {k[:5]: r for k, r in read_rows(path).items()}
    summaries = []
    canonical = next(iter(results.values()))
    for rows in results.values():
        if rows.keys() != canonical.keys():
            raise ValueError("Training inventories differ")
        for key, row in rows.items():
            for field in ("cpu", "os", "executable_sha256", "source_revision", "process_affinity"):
                if row[field] != canonical[key][field]:
                    raise ValueError(f"Binary or host changed: {field}")
            if row["profile"] != "fixed" or row["iterations"] != canonical[key]["iterations"]:
                raise ValueError("Training sequences differ")
            if not math.isclose(float(row["checksum"]), float(canonical[key]["checksum"]),
                                rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError("Training loss checksums differ")
    for profile in profiles:
        for key in canonical:
            times, references, ratios = [], [], []
            for repeat in range(args.repeats):
                after = float(results[(profile["name"], repeat)][key]["median_seconds"])
                before = float(results[(profile["reference"], repeat)][key]["median_seconds"])
                times.append(after * 1000)
                references.append(before * 1000)
                ratios.append(after / before)
            median = statistics.median(ratios)
            summaries.append({"profile": profile["name"], "reference": profile["reference"],
                              "case": key[1], "threads": profile["threads"],
                              "median_ms": statistics.median(times),
                              "reference_ms": statistics.median(references),
                              "change_percent": (median - 1) * 100,
                              "ratio_min": min(ratios), "ratio_max": max(ratios),
                              "spread_percent": (max(times) - min(times)) /
                                                statistics.median(times) * 100,
                              "regression_over_5_percent": median > 1.05})
    with (output / "summary.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
