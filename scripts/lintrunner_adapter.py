from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Callable
from pathlib import Path
from xml.etree import ElementTree

import yaml
from lintrunner_adapters import LintMessage, LintSeverity


def lint_message(
    path: Path,
    code: str,
    name: str,
    description: str,
    *,
    line: int | None = None,
    char: int | None = None,
    original: str | None = None,
    replacement: str | None = None,
) -> LintMessage:
    return LintMessage(
        path=str(path),
        line=line,
        char=char,
        code=code,
        severity=LintSeverity.ERROR,
        name=name,
        original=original,
        replacement=replacement,
        description=description,
    )


def check_python(path: Path) -> None:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def check_json(path: Path) -> None:
    with path.open(encoding="utf-8") as file:
        json.load(file)


def check_yaml(path: Path) -> None:
    with path.open(encoding="utf-8") as file:
        list(yaml.safe_load_all(file))


def check_xml(path: Path) -> None:
    ElementTree.parse(path)


SYNTAX_CHECKS: dict[str, Callable[[Path], None]] = {
    ".json": check_json,
    ".py": check_python,
    ".pyi": check_python,
    ".xml": check_xml,
    ".yaml": check_yaml,
    ".yml": check_yaml,
}


def syntax_error_location(error: Exception) -> tuple[int | None, int | None]:
    if isinstance(error, (SyntaxError, json.JSONDecodeError)):
        return error.lineno, error.offset if isinstance(error, SyntaxError) else error.colno
    if isinstance(error, yaml.MarkedYAMLError) and error.problem_mark is not None:
        return error.problem_mark.line + 1, error.problem_mark.column + 1
    if isinstance(error, ElementTree.ParseError):
        return error.position
    return None, None


def check_syntax(path: Path) -> LintMessage | None:
    check = SYNTAX_CHECKS.get(path.suffix.lower())
    if check is None:
        return None

    try:
        check(path)
    except (SyntaxError, json.JSONDecodeError, yaml.YAMLError, ElementTree.ParseError, UnicodeDecodeError) as error:
        line, char = syntax_error_location(error)
        return lint_message(
            path,
            "SYNTAX",
            f"invalid-{path.suffix.removeprefix('.').lower()}",
            str(error),
            line=line,
            char=char,
        )
    return None


def normalize_file(path: Path) -> LintMessage | None:
    original_bytes = path.read_bytes()
    crlf_count = original_bytes.count(b"\r\n")
    lf_count = original_bytes.count(b"\n") - crlf_count
    newline = b"\r\n" if crlf_count > lf_count else b"\n"

    normalized = original_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    lines = normalized.split(b"\n")

    for index, line in enumerate(lines):
        stripped = line.rstrip(b" \t")
        if path.suffix.lower() == ".md" and line.endswith(b"  "):
            stripped += b"  "
        lines[index] = stripped

    normalized = newline.join(lines)
    if normalized:
        normalized = normalized.rstrip(b"\r\n") + newline

    if normalized == original_bytes:
        return None

    try:
        original = original_bytes.decode("utf-8")
        replacement = normalized.decode("utf-8")
    except UnicodeDecodeError as error:
        return lint_message(path, "FILE-HYGIENE", "invalid-encoding", str(error))

    return lint_message(
        path,
        "FILE-HYGIENE",
        "file-hygiene",
        "Normalize line endings, trailing whitespace, and the final newline with `lintrunner -a`.",
        original=original,
        replacement=replacement,
    )


def main() -> None:
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("check", choices=("syntax", "file-hygiene"))
    parser.add_argument("filenames", nargs="+")
    args = parser.parse_args()

    check = check_syntax if args.check == "syntax" else normalize_file
    for filename in args.filenames:
        message = check(Path(filename))
        if message is not None:
            message.display()


if __name__ == "__main__":
    main()
