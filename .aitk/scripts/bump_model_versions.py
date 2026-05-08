import argparse
import re
import sys
from pathlib import Path
from typing import Optional

from sanitize.utils import iter_aitk_info_yml


VERSION_LINE_RE = re.compile(r"^(?P<indent>\s*)version:\s*(?P<value>\d+)\s*(?P<comment>#.*)?$")


def find_version_line(lines: list[str], version_value: int) -> Optional[int]:
    """Return the index of the `version:` line under top-level `aitk.modelInfo`.

    Only the top-level `aitk:` key (indent 0) is considered — recipe entries
    can contain their own nested `aitk:` blocks for per-recipe overrides.
    """
    in_aitk = False
    modelInfo_indent: Optional[int] = None
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(stripped)

        if not in_aitk:
            if indent == 0 and stripped.startswith("aitk:"):
                in_aitk = True
            continue

        # Left the top-level aitk: block.
        if indent == 0:
            return None

        if modelInfo_indent is None:
            if stripped.startswith("modelInfo:"):
                modelInfo_indent = indent
            continue

        if indent <= modelInfo_indent:
            # Left modelInfo: but still inside aitk: — keep scanning in case
            # version lives in a different subkey shape, though that's unusual.
            modelInfo_indent = None
            continue

        m = VERSION_LINE_RE.match(line.rstrip("\n"))
        if m and int(m.group("value")) == version_value:
            return idx
    return None


def bump_file(yml_file: Path, yaml_object: dict, delta: int, dry_run: bool) -> Optional[tuple[int, int]]:
    """Bump the aitk.modelInfo.version in one info.yml. Returns (old, new) or None."""
    modelInfo = yaml_object["aitk"].get("modelInfo")
    if not isinstance(modelInfo, dict) or "version" not in modelInfo:
        return None
    old = modelInfo["version"]
    if not isinstance(old, int):
        print(f"Skip {yml_file}: version is not an integer ({old!r})")
        return None
    new = old + delta
    if new < 1:
        print(f"Skip {yml_file}: version {old} + delta {delta:+d} would be < 1")
        return None

    with yml_file.open("r", encoding="utf-8", newline="") as f:
        text = f.read()
    lines = text.splitlines(keepends=True)
    idx = find_version_line(lines, old)
    if idx is None:
        print(f"Skip {yml_file}: could not locate 'version: {old}' line under aitk.modelInfo")
        return None
    m = VERSION_LINE_RE.match(lines[idx].rstrip("\n"))
    newline = lines[idx][len(lines[idx].rstrip("\r\n")):] or "\n"
    indent = m.group("indent")
    comment = m.group("comment")
    rebuilt = f"{indent}version: {new}"
    if comment:
        rebuilt += f"  {comment}"
    rebuilt += newline
    lines[idx] = rebuilt

    if not dry_run:
        with yml_file.open("w", encoding="utf-8", newline="") as f:
            f.writelines(lines)
    return old, new


def main():
    parser = argparse.ArgumentParser(
        description="Batch increase or decrease aitk.modelInfo.version across all info.yml files."
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=1,
        help="Integer delta to apply (e.g. +1, -1, 2). Must not reduce any version below 1.",
    )
    parser.add_argument(
        "--filter",
        dest="filter",
        default=None,
        help="Optional substring; only info.yml files whose modelInfo.id contains this string are updated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files.",
    )
    args = parser.parse_args()

    if args.delta == 0:
        parser.error("delta must be non-zero")

    root_dir = Path(__file__).parent.parent.parent
    changed = 0
    scanned = 0
    for yml_file, yaml_object in iter_aitk_info_yml(root_dir):
        scanned += 1
        if args.filter:
            mid = yaml_object["aitk"].get("modelInfo", {}).get("id", "")
            if args.filter not in mid:
                continue

        result = bump_file(yml_file, yaml_object, args.delta, args.dry_run)
        if result is None:
            continue
        old, new = result
        changed += 1
        rel = yml_file.relative_to(root_dir)
        tag = "[dry-run] " if args.dry_run else ""
        print(f"{tag}{rel}: version {old} -> {new}")

    action = "Would update" if args.dry_run else "Updated"
    print(f"\n{action} {changed} info.yml file(s) with aitk block (delta {args.delta:+d}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
