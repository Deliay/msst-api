"""RMVPE pitch extraction for v2m, reusing the RVC (Applio) predictor.

The v2m pipeline needs an F0 curve for USTX/VSQX pitch export.  Rather than
shipping a second RMVPE implementation (the ONNX ``rmvpe.onnx`` bundle), the
default backend reuses the **same** RMVPE torch predictor the RVC engine
already manages (``MSST_RVC_ASSET_DIR/predictors/rmvpe.pt``, downloaded on
demand from the Applio Hugging Face repo).  This keeps one copy of the weights
and one loader for both subsystems.

Set ``MSST_V2M_RMVPE_BACKEND=onnx`` to use the ONNX bundle from the v2m catalog
instead (useful if RVC is not deployed).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..config import Settings
from ._vendor.inference.API.rmvpe_api import RmvpeResult, RmvpeTranscriber

logger = logging.getLogger("msst_api.v2m.rmvpe")

_SAMPLE_RATE = 16000
_HOP_LENGTH = 160


def _torch_device(device: str) -> Any:
    import torch

    value = (device or "cpu").strip()
    if value.startswith("cuda") and torch.cuda.is_available():
        return value
    if value.startswith("cuda"):
        return "cpu"
    return value or "cpu"


class RvcRmvpeTranscriber:
    """Duck-typed stand-in for the ONNX :class:`RmvpeTranscriber`."""

    def __init__(self, device: str, settings: Settings) -> None:
        from ..rvc.lib.assets import set_assets_dir
        from ..rvc.lib.predictors.f0 import RMVPE

        set_assets_dir(settings.rvc_asset_dir)
        self.device = _torch_device(device)
        logger.info("Loading RMVPE via the RVC/Applio predictor on %s", self.device)
        self._rvc = RMVPE(self.device, sample_rate=_SAMPLE_RATE, hop_size=_HOP_LENGTH)
        self.time_step_seconds = _HOP_LENGTH / _SAMPLE_RATE

    def infer(self, waveform, sample_rate: int, cancel_checker=None) -> RmvpeResult:
        import librosa

        if cancel_checker and cancel_checker():
            raise InterruptedError("RMVPE task cancelled")

        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        if audio.size == 0:
            empty = np.zeros((0,), dtype=np.float32)
            return RmvpeResult(
                time_step_seconds=self.time_step_seconds,
                midi_pitch=empty,
                voiced_mask=empty.astype(bool),
            )
        if sample_rate != _SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=_SAMPLE_RATE)
        audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)

        f0 = np.asarray(self._rvc.get_f0(audio), dtype=np.float32).reshape(-1)
        voiced_mask = f0 > 0
        midi_pitch = RmvpeTranscriber._f0_to_interpolated_midi(f0)
        return RmvpeResult(
            time_step_seconds=self.time_step_seconds,
            midi_pitch=midi_pitch,
            voiced_mask=voiced_mask,
        )

    def shutdown(self) -> None:
        self._rvc = None

    release = shutdown
