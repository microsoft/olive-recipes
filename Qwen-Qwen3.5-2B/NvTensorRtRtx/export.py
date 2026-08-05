# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Run the shared Qwen NvTensorRtRtx exporter for this recipe."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qwen_nvtensorrtrtx_export import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parent))
