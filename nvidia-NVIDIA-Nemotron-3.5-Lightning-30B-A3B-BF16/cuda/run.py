"""Register and run the experimental CUDA export, requiring explicit opt-in."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-unvalidated-export",
        action="store_true",
        help="Acknowledge blocked full-model parity and permit a development-only weighted export.",
    )
    args = parser.parse_args()
    if not args.allow_unvalidated_export:
        parser.error("Full-model parity is blocked; development export requires --allow-unvalidated-export.")

    from olive import run

    directory = Path(__file__).parent
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    config["passes"]["export"]["allow_unvalidated_export"] = True
    output = run(
        config,
        package_config={
            "passes": {
                "NemotronLightningBuilder": {
                    "module_path": "lightning_builder.NemotronLightningBuilder",
                    "supported_providers": ["CUDAExecutionProvider"],
                    "supported_accelerators": ["gpu"],
                    "supported_precisions": ["bf16"],
                    "module_dependencies": ["mobius-onnx"],
                }
            }
        },
    )
    if not output.has_output_model():
        raise RuntimeError("Olive did not produce the paired export. Check the workflow logs for the failure.")


if __name__ == "__main__":
    main()
