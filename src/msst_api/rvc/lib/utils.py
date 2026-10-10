"""Audio/embedding helpers for the vendored RVC engine.

Trimmed from Applio's ``rvc.lib.utils``: only what the inference path needs is
kept, and everything works from in-memory arrays (paths are still accepted for
convenience).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn
from transformers import HubertModel

from .assets import ensure_embedder

logger = logging.getLogger("msst_api.rvc.utils")


class HubertModelWithFinalProj(HubertModel):
    """HuBERT with the projection head used by v1 RVC checkpoints."""

    def __init__(self, config):  # noqa: ANN001 - transformers config object
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


def format_title(title: str) -> str:
    """Normalise a model name the same way Applio does."""

    formatted = unicodedata.normalize("NFC", title)
    formatted = re.sub(r"[\u2500-\u257F]+", "", formatted)
    formatted = re.sub(r"[^\w\s.-]", "", formatted, flags=re.UNICODE)
    return re.sub(r"\s+", "_", formatted)


def _to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        data = librosa.to_mono(data.T)
    return np.asarray(data, dtype=np.float32).flatten()


def load_audio_infer(
    audio,
    sample_rate: int = 16000,
    formant_shifting: bool = False,
    formant_qfrency: float = 1.0,
    formant_timbre: float = 1.0,
) -> np.ndarray:
    """Return a mono ``float32`` waveform at ``sample_rate``.

    ``audio`` may be a filesystem path or an existing array (assumed to already
    be at ``sample_rate``).
    """

    if isinstance(audio, (str, Path)):
        data, source_rate = librosa.load(str(audio), sr=None, mono=False)
    else:
        data = np.asarray(audio, dtype=np.float32)
        source_rate = sample_rate

    data = _to_mono(data)
    if source_rate != sample_rate:
        data = librosa.resample(
            data, orig_sr=source_rate, target_sr=sample_rate, res_type="soxr_vhq"
        )

    if formant_shifting:
        try:
            from stftpitchshift import StftPitchShift

            shifter = StftPitchShift(1024, 32, sample_rate)
            data = shifter.shiftpitch(
                data,
                factors=1,
                quefrency=float(formant_qfrency) * 1e-3,
                distortion=float(formant_timbre),
            )
        except Exception as error:  # noqa: BLE001 - optional dependency
            logger.warning("Formant shifting failed, continuing without it: %s", error)

    return np.asarray(data, dtype=np.float32).flatten()


def load_embedding(embedder_model: str, custom_embedder: str | None = None):
    """Load the HuBERT/ContentVec embedder used for feature extraction."""

    if embedder_model == "custom" and custom_embedder and Path(custom_embedder).exists():
        model_path = custom_embedder
    else:
        if embedder_model == "custom":
            logger.warning("Custom embedder %s not found; using contentvec", custom_embedder)
            embedder_model = "contentvec"
        model_path = str(ensure_embedder(embedder_model))

    model = HubertModelWithFinalProj.from_pretrained(model_path)
    return model


def is_half_available(device: str) -> bool:
    """Whether the device supports fp16 (kept for parity with Applio callers)."""

    return str(device).startswith("cuda") and torch.cuda.is_available()
