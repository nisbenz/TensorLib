#!/usr/bin/env python3
"""Convert the pinned CLI-1M Bash/English Parquet shards to CommandLM text."""
import argparse
import glob
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import pyarrow.parquet as parquet
except ImportError as error:
    raise SystemExit("install pyarrow before importing CLI-1M") from error

sys.path.insert(0, str(Path(__file__).parent))
from import_nl2bash import safe_command

REVISION = "ec725c1269166e7f8ce8a20df1787ceecd1e421a"


def extract_messages(messages):
    return {item["role"]: item["content"] for item in (messages or [])
            if item and item["role"] in ("user", "assistant")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="local CLI-1M checkout")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--language", choices=("en", "all"), default="en")
    args = parser.parse_args()
    paths = sorted(glob.glob(str(args.source / "data/default/train/*.parquet")))
    if not paths:
        raise SystemExit("no CLI-1M default training shards found")
    rows = []
    stats = Counter()
    request_commands = defaultdict(set)
    for path in paths:
        table = parquet.read_table(path, columns=["messages", "shell", "language", "license_spdx"])
        for messages, shell, language, license_spdx in zip(
                table["messages"].to_pylist(), table["shell"].to_pylist(),
                table["language"].to_pylist(), table["license_spdx"].to_pylist()):
            stats["input_rows"] += 1
            stats[f"license:{license_spdx}"] += 1
            if shell != "bash" or (args.language == "en" and language != "en"):
                stats["non_english_bash"] += 1
                continue
            values = extract_messages(messages)
            request, command = values.get("user", "").strip(), values.get("assistant", "").strip()
            if args.language == "all":
                request = f"[language={language}] {request}"
            if not request or not safe_command(command):
                stats["unsafe_or_malformed"] += 1
                continue
            request_commands[request].add(command)
            rows.append((request, command))
    ambiguous = {request for request, commands in request_commands.items()
                 if len(commands) > 1}
    rows = [(request, command) for request, command in rows
            if request not in ambiguous]
    rows = list(dict.fromkeys(rows))
    random.Random(args.seed).shuffle(rows)
    stats["ambiguous_requests"] = len(ambiguous)
    stats["retained_rows"] = len(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        for request, command in rows:
            output.write(f"REQUEST: {request}\nCOMMAND: {command}\n\n".encode())
    digest = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as source_file:
            for block in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(block)
    manifest = {"dataset": "carosh/cli-1m", "revision": REVISION,
                "shards": len(paths), "source_sha256": digest.hexdigest(),
                "seed": args.seed, "language": args.language, "stats": dict(stats),
                "licenses": "per-row metadata retained upstream; dataset card is Apache-2.0"}
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
