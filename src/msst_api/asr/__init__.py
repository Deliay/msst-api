"""ASR capability: single-sentence (``asr``) and streaming (``streaming_asr``).

Supported engines:

* Qwen3-ASR -- single-sentence **and** streaming;
* PinyinASR -- single-sentence only;
* RomajiASR -- single-sentence only.
"""

from __future__ import annotations

from .catalog import ASR_CAPABILITY, BUNDLED_MODELS, build_store
from .engines import ASRResult, StreamingSession
from .manager import (
    ASRError,
    ASRManager,
    ASRModelNotFoundError,
    ASRModelSpec,
)

__all__ = [
    "ASR_CAPABILITY",
    "BUNDLED_MODELS",
    "build_store",
    "ASRResult",
    "StreamingSession",
    "ASRError",
    "ASRManager",
    "ASRModelNotFoundError",
    "ASRModelSpec",
]
