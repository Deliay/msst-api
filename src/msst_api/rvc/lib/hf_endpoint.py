"""Hugging Face endpoint resolution for the vendored RVC engine.

Honours ``MSST_HF_ENDPOINT`` (the setting used elsewhere in this project) and
falls back to the generic ``HF_ENDPOINT`` before the public Hugging Face host,
so the same mirror configuration drives both MSST and RVC downloads.
"""

from __future__ import annotations

import os

DEFAULT_HF_ENDPOINT = "https://huggingface.co"


def get_hf_endpoint() -> str:
    """Return the Hugging Face base URL, without a trailing slash."""

    endpoint = os.environ.get("MSST_HF_ENDPOINT") or os.environ.get("HF_ENDPOINT")
    return (endpoint or DEFAULT_HF_ENDPOINT).rstrip("/")
