"""Shared utilities for Gemma 4 recipes."""


def resolve_model_path(device: str, variant: str | None) -> str:
    """Resolve the model directory path from device/variant args."""
    if device == "cpu":
        variant = variant or "fp32"
        return f"cpu/{variant}/models"
    variant = variant or "int4"
    return f"cuda/{variant}/models"
