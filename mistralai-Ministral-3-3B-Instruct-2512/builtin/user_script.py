"""Olive user script for Ministral-3-3B text decoder export.

Vision and embedding are exported via mobius (see optimize.py).
This script provides model config constants used by optimize.py.
Config loading is lazy to avoid import-time network calls.
"""

MODEL_NAME = "mistralai/Ministral-3-3B-Instruct-2512"

_CONFIG = None


def _get_config():
    """Lazily load HuggingFace config (avoids import-time network access)."""
    global _CONFIG
    if _CONFIG is None:
        from transformers import Mistral3Config

        _CONFIG = Mistral3Config.from_pretrained(MODEL_NAME)
    return _CONFIG


def __getattr__(name):
    """Lazily expose config-derived constants without import-time model loading."""
    config = _get_config()
    if name == "IMAGE_TOKEN_ID":
        return config.image_token_index  # 10
    if name == "PATCH_SIZE":
        return config.vision_config.patch_size  # 14
    if name == "SPATIAL_MERGE_SIZE":
        return config.spatial_merge_size  # 2
    if name == "TEXT_HIDDEN_SIZE":
        return config.text_config.hidden_size  # 3072
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [  # noqa: F822  (provided by __getattr__)
    "MODEL_NAME",
    "IMAGE_TOKEN_ID",
    "PATCH_SIZE",
    "SPATIAL_MERGE_SIZE",
    "TEXT_HIDDEN_SIZE",
]
