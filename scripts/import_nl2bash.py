#!/usr/bin/env python3
"""Import a conservative, read-only subset of NL2Bash for CommandLM."""
import argparse
import hashlib
import json
import random
import shlex
from collections import defaultdict
from pathlib import Path

DENY = {
    "rm", "rmdir", "mv", "cp", "chmod", "chown", "chgrp", "ln", "touch",
    "mkdir", "mount", "umount", "dd", "mkfs", "fdisk", "shred", "kill",
    "pkill", "sudo", "su", "ssh", "scp", "curl", "wget", "nc", "ncat",
    "bash", "sh", "zsh", "fish", "python", "python3", "perl", "ruby",
    "php", "node", "eval", "exec", "source", "systemctl", "shutdown",
    "reboot", "init", "iptables", "route",
}
FORBIDDEN = set(";&$`<>()[\n\r")


def safe_command(command):
    if not command or any(char in command for char in FORBIDDEN):
        return False
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return False
    if not tokens:
        return False
    for token in tokens:
        if token == "|":
            continue
        name = token.rsplit("/", 1)[-1].lower()
        if name in DENY:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="NL2Bash data/bash directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    nl_path, command_path = args.source / "all.nl", args.source / "all.cm"
    descriptions = nl_path.read_text(encoding="utf-8").splitlines()
    commands = command_path.read_text(encoding="utf-8").splitlines()
    if len(descriptions) != len(commands):
        raise SystemExit("NL2Bash description/command counts differ")
    candidates = []
    rejected = defaultdict(int)
    for description, command in zip(descriptions, commands):
        description, command = description.strip(), command.strip()
        if not description or not safe_command(command):
            rejected["unsafe_or_malformed"] += 1
            continue
        candidates.append((description, command))
    grouped = defaultdict(set)
    for description, command in candidates:
        grouped[description].add(command)
    rows = [(description, next(iter(commands)))
            for description, commands in grouped.items() if len(commands) == 1]
    rejected["ambiguous_request"] = sum(
        len(commands) > 1 for commands in grouped.values())
    random.Random(args.seed).shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        for description, command in rows:
            output.write(f"REQUEST: {description}\nCOMMAND: {command}\n\n".encode())
    source_hash = hashlib.sha256(nl_path.read_bytes() + command_path.read_bytes()).hexdigest()
    metadata = {
        "source": "TellinaTool/nl2bash",
        "source_sha256": source_hash,
        "seed": args.seed,
        "input_pairs": len(descriptions),
        "retained_pairs": len(rows),
        "rejected": dict(rejected),
        "license": "MIT (data/bash/LICENSE)",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
