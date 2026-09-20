#!/usr/bin/env python3
"""Audit CommandLM records for duplication, leakage, and label noise."""
import sys
from collections import Counter, defaultdict
from pathlib import Path


def records(data):
    return [part for part in data.split(b"\n\n") if part.strip()]


def fields(record):
    parts = record.split(b"\nCOMMAND: ", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else None


def main(path):
    data = Path(path).read_bytes()
    rows = records(data)
    cut = len(data) * 9 // 10
    train, validation = records(data[:cut]), records(data[cut:])
    parsed = [fields(row) for row in rows]
    parsed = [row for row in parsed if row is not None]
    requests = defaultdict(Counter)
    for request, command in parsed:
        requests[request][command] += 1
    ambiguous = {key: value for key, value in requests.items()
                 if len(value) > 1}
    unique = len(set(rows))
    train_set = set(train)
    overlap = sum(row in train_set for row in validation)
    print(f"bytes={len(data)} records={len(rows)} unique_records={unique}")
    print(f"train_records={len(train)} validation_records={len(validation)}")
    print(f"duplicate_records={len(rows) - unique} validation_overlap={overlap}")
    print(f"unique_requests={len(requests)} ambiguous_requests={len(ambiguous)}")
    for request, commands in list(ambiguous.items())[:5]:
        print("ambiguous:", request.decode("utf-8", "replace"),
              "=>", len(commands), "commands")
    return 1 if not rows or len(parsed) != len(rows) or ambiguous or overlap else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} CORPUS", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
