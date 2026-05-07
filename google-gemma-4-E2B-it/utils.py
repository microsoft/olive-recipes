"""Shared utilities for Gemma 4 recipes."""


def resolve_model_path(device: str, variant: str | None) -> str:
    """Resolve the model directory path from device/variant args."""
    if device == "cpu":
        return "cpu/models"
    variant = variant or "int4"
    return f"cuda/{variant}/models"
