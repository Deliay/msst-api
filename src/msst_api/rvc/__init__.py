"""Vendored Retrieval-based Voice Conversion (RVC) inference engine.

This is a trimmed port of the inference subset of
`Applio <https://github.com/IAHispano/Applio>`_ (MIT licensed), adapted to run
as a long-lived, path-configurable service inside ``msst-api``.  Only the code
needed to load an RVC ``.pth`` voice model and convert audio in memory is kept;
the training, realtime and Gradio UI layers are intentionally omitted.

The public entry point is :class:`msst_api.rvc.infer.convert.VoiceConverter`.
"""

from __future__ import annotations

__all__ = ["VoiceConverter"]


def __getattr__(name: str):  # pragma: no cover - lazy re-export
    if name == "VoiceConverter":
        from .infer.convert import VoiceConverter

        return VoiceConverter
    raise AttributeError(name)
