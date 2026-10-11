"""OpenUtau-compatible DiffSinger (SVS) inference.

This package ports the parts of `stakira/OpenUtau`'s C# DiffSinger renderer
that are needed to turn an OpenUtau ``.ustx`` score into a rendered ``.wav``:

* :mod:`msst_api.svs.voicebank` -- loads an OpenUtau DiffSinger voicebank
  (``dsconfig.yaml`` + phoneme set + g2p dictionaries + ONNX models + vocoder);
* :mod:`msst_api.svs.ustx` -- reads/writes the USTX score and its curves;
* :mod:`msst_api.svs.phonemizer` -- lyric -> phoneme + duration prediction
  (the ``dsdur`` linguistic + duration models) and note alignment;
* :mod:`msst_api.svs.render` -- the auto-pitch (``dspitch``) / variance
  (``dsvariance``) / acoustic / vocoder ONNX stages;
* :mod:`msst_api.svs.pipeline` -- ``ustx -> pitd+ustx -> wav`` orchestration.

The heavy ONNX runtime is imported lazily so the HTTP app and model listing
work even before a voicebank is installed.
"""

from __future__ import annotations

__all__ = [
    "SVSError",
    "SVSVoiceNotFoundError",
]


class SVSError(RuntimeError):
    """Raised when a DiffSinger voicebank cannot be used or rendering fails."""


class SVSVoiceNotFoundError(SVSError):
    """Raised when a requested DiffSinger voicebank id does not exist."""
