#!/usr/bin/env python3

from __future__ import annotations
import argparse
import pathlib
import re
import sys

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

DEFAULT_CODE_EXT = set()

SELF_PATH = pathlib.Path(__file__).resolve()


def looks_binary(data: bytes) -> bool:
    if b"\x00" in data:
        return True

    if len(data) < 4:
        return False

    text_ish = sum(32 <= b <= 126 or b in (9, 10, 13) for b in data)
    return (text_ish / max(1, len(data))) < 0.7


def should_check(path: pathlib.Path, include_ext: set[str] | None, exclude_ext: set[str]) -> bool:
    ext = path.suffix.lower()

    if include_ext is not None:
        return ext in include_ext

    return ext not in exclude_ext


def scan_file(path: pathlib.Path) -> list[tuple[int, str]]:
    try:
        raw = path.read_bytes()
    except Exception:
        return []
    if looks_binary(raw):
        return []
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        return []

    findings: list[tuple[int, str]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if CYRILLIC_RE.search(line):
            findings.append((i, line.strip()))
            if len(findings) >= 10:
                break
    return findings


def parse_ext_list(s: str | None) -> set[str] | None:
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return {("." + p if not p.startswith(".") else p).lower() for p in parts}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-ext", help='Comma-separeted list of extensions (e.g. "md,txt"')
    parser.add_argument("--exclude-ext", help='Comma-separated list of extensions (e.g. "md,txt"')
    parser.add_argument("files", nargs="*")
    args = parser.parse_args(argv)

    include_ext = parse_ext_list(args.include_ext)
    exclude_ext = parse_ext_list(args.exclude_ext) or set(DEFAULT_CODE_EXT)

    had_errors = False
    for name in args.files:
        path = pathlib.Path(name)
        if not path.is_file():
            continue
        try:
            if path.resolve() == SELF_PATH:
                continue
        except Exception:
            continue
        if not should_check(path, include_ext, exclude_ext):
            continue
        findings = scan_file(path)
        if findings:
            had_errors = True
            print(f"\n[cyrl] {path}")
            for ln, snippet in findings:
                print(f"  L{ln}: {snippet}")

    if had_errors:
        print("\nCyrillic symbols found. Fix & commit!")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
