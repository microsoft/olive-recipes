#!/usr/bin/env python3
"""
Sanitize script - new modular version
This script maintains compatibility with the original sanitize.py while using the new modular structure.
Auto-formats all Python scripts in the scripts directory on every run.
"""

# Import main directly without going through __init__.py
import sys
from pathlib import Path

from auto_formatter import auto_format_scripts
from project_processor import project_processor
from sanitize.main import main
from sanitize.utils import GlobalVars

# Get the absolute path to the project root (parent of scripts)
project_root = Path(__file__).parent.parent
scripts_dir = Path(__file__).parent

# Add both the project root and scripts directory to Python path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))


# Import the main function from the new sanitize.main module
# Import here to avoid circular imports after formatting
def run_main():
    original_path = sys.path.copy()

    try:
        # Make sure the project root and scripts dir are in the path
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        main()
    finally:
        # Restore original sys.path
        sys.path = original_path


if __name__ == "__main__":
    # Check if verbose mode is requested
    if "-v" in sys.argv or "--verbose" in sys.argv:
        GlobalVars.verbose = True

    # Auto-format scripts before running sanitize
    auto_format_scripts()
    project_processor()
    run_main()
