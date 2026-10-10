"""Vocal2Midi (``v2m``) capability: vocal audio -> lyric-aligned MIDI.

The heavy lifting is done by the runtimes vendored from the Vocal2Midi project
(``msst_api.v2m._vendor``).  This package adds the service layer: a model
catalog + store, a manager that routes every model load/inference through the
project-wide :class:`~msst_api.resources.SharedModelCache`, and the pipeline
orchestration used by the HTTP API.
"""

from __future__ import annotations

__all__ = ["V2MManager", "V2MError"]


def __getattr__(name: str):  # lazy to avoid importing onnxruntime at package import
    if name in {"V2MManager", "V2MError"}:
        from .manager import V2MError, V2MManager

        return {"V2MManager": V2MManager, "V2MError": V2MError}[name]
    raise AttributeError(name)
