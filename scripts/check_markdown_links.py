#!/usr/bin/env python3
"""Fail when a repository Markdown file links to a missing local path."""

from pathlib import Path
import re
import sys
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK_PATTERN = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
SKIPPED_DIRECTORIES = {".git", "build", "bin"}


def markdown_files():
    for path in ROOT.rglob("*.md"):
        if not SKIPPED_DIRECTORIES.intersection(path.relative_to(ROOT).parts):
            yield path


def local_target(raw_target):
    target = raw_target.strip().strip("<>")
    if not target or target.startswith("#") or "://" in target:
        return None
    return unquote(target.split("#", 1)[0])


def main():
    missing = []
    checked = 0
    for document in markdown_files():
        checked += 1
        for match in LINK_PATTERN.finditer(document.read_text(encoding="utf-8")):
            target = local_target(match.group(1))
            if target and not (document.parent / target).exists():
                missing.append(
                    f"{document.relative_to(ROOT)}: missing target {target}"
                )

    if missing:
        print("\n".join(missing), file=sys.stderr)
        return 1

    print(f"Checked local links in {checked} Markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
