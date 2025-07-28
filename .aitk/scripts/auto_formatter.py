#!/usr/bin/env python3
"""
Auto-formatter module for Python scripts.
Provides comprehensive formatting capabilities including:
- Removing unused imports using autoflake
- Sorting and organizing imports using isort
- Formatting code using black with 120 character line length
- Checking that all imports are at the top of files
"""

import subprocess
import sys
from pathlib import Path

from sanitize.utils import printError, printInfo, printTip, printWarning


def install_formatter_tools():
    """
    Install required formatting tools if not available.
    """
    tools = [
        ("black", "black"),
        ("isort", "isort"),
        ("autoflake", "autoflake"),  # Added autoflake for removing unused imports
    ]

    for tool_name, package_name in tools:
        try:
            subprocess.run([tool_name, "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            printInfo(f"Installing {package_name} formatter...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package_name], check=True)
            except subprocess.CalledProcessError as e:
                printError(f"Failed to install {package_name}: {e}")
                return False
    return True


def check_imports_at_top(file_path):
    """
    Check if all imports are at the top of the file (after docstring and comments).
    Returns True if imports are properly placed, False otherwise.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Skip shebang, encoding declarations, and docstrings
        in_docstring = False
        docstring_quotes = None
        non_import_code_found = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith("#"):
                continue

            # Handle docstrings
            if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                docstring_quotes = stripped[:3]
                if stripped.count(docstring_quotes) >= 2:
                    # Single line docstring
                    continue
                else:
                    in_docstring = True
                    continue
            elif in_docstring and docstring_quotes and docstring_quotes in stripped:
                in_docstring = False
                continue
            elif in_docstring:
                continue

            # Check for imports and from statements (including multi-line imports)
            if (
                stripped.startswith("import ")
                or stripped.startswith("from ")
                or (not non_import_code_found and (stripped.endswith(",") or stripped.startswith(")")))
            ):
                if non_import_code_found:
                    printWarning(f"Import found after non-import code in {file_path}:{i+1}")
                    return False
            else:
                # Non-import code found (but ignore special variables and sys.path modifications)
                if (
                    stripped
                    and not stripped.startswith("__")
                    and not stripped.startswith("sys.path")
                    and not any(special in stripped for special in ["__all__", "__version__", "__author__"])
                ):
                    non_import_code_found = True

        return True
    except Exception as e:
        printError(f"Error checking imports in {file_path}: {e}")
        return True  # Don't fail the entire process


def auto_format_scripts(target_dir=None):
    """
    Auto-format all Python scripts in the target directory with comprehensive formatting:
    - Remove unused imports using autoflake
    - Sort and organize imports using isort
    - Format code using black with 120 character line length
    - Check that all imports are at the top of files

    Args:
        target_dir: Path to the directory to format. If None, uses the scripts directory.
    """
    if target_dir is None:
        target_dir = Path(__file__).parent
    else:
        target_dir = Path(target_dir)

    printTip(f"Auto-formatting Python scripts in {target_dir}...")

    # Find all Python files in the target directory
    python_files = []
    for py_file in target_dir.rglob("*.py"):
        if py_file.is_file():
            python_files.append(str(py_file))

    if not python_files:
        printInfo("No Python files found to format.")
        return

    # Install required tools
    if not install_formatter_tools():
        printError("Failed to install required formatting tools.")
        return

    # Step 1: Remove unused imports with autoflake
    printInfo("Step 1: Removing unused imports...")
    try:
        autoflake_cmd = [
            "autoflake",
            "--in-place",  # Modify files in place
            "--remove-all-unused-imports",  # Remove all unused imports
            "--remove-unused-variables",  # Remove unused variables
            "--remove-duplicate-keys",  # Remove duplicate keys in dictionaries
            "--ignore-init-module-imports",  # Don't remove imports in __init__.py files
        ] + python_files

        result = subprocess.run(autoflake_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            printInfo(f"Successfully removed unused imports from {len(python_files)} files.")
            if result.stdout:
                printInfo(result.stdout)
        else:
            printError(f"Autoflake failed: {result.stderr}")
    except Exception as e:
        printError(f"Error during unused import removal: {e}")

    # Step 2: Sort imports with isort
    printInfo("Step 2: Sorting and organizing imports...")
    try:
        isort_cmd = [
            "isort",
            "--line-length",
            "120",
            "--multi-line",
            "3",
            "--trailing-comma",
            "--force-grid-wrap",
            "0",
            "--combine-as",
            "--use-parentheses",
        ] + python_files

        result = subprocess.run(isort_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            printInfo(f"Successfully sorted imports in {len(python_files)} files.")
        else:
            printError(f"Import sorting failed: {result.stderr}")
    except Exception as e:
        printError(f"Error during import sorting: {e}")

    # Step 3: Check import placement
    printInfo("Step 3: Checking import placement...")
    for py_file in python_files:
        check_imports_at_top(py_file)

    # Step 4: Format with black
    printInfo("Step 4: Formatting code with black...")
    try:
        black_cmd = ["black", "--line-length", "120"] + python_files
        result = subprocess.run(black_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            printInfo(f"Successfully formatted {len(python_files)} Python files with 120 character line length.")
            if result.stdout:
                printInfo(result.stdout)
        else:
            printError(f"Black formatting failed: {result.stderr}")
    except Exception as e:
        printError(f"Error during black formatting: {e}")

    printInfo("Auto-formatting completed!")

    # Clear sanitize modules from cache
    modules_to_clear = [name for name in sys.modules.keys() if name.startswith("sanitize")]
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]


if __name__ == "__main__":
    # Allow running this module directly for testing
    auto_format_scripts()
