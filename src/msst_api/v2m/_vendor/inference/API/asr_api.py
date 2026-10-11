"""ASR adapter for the vendored Vocal2Midi pipeline.

The upstream module drives Qwen3-ASR through a DirectML/GGUF subprocess.  This
service instead routes all ASR through :class:`msst_api.asr.manager.ASRManager`,
which loads models lazily and applies the project-wide model / concurrency
budgets.  Only the small, stable call surface used by the vendored pipeline is
kept:

* :func:`batch_transcribe_asr` -- Qwen3-ASR (text) over chunked audio;
* :func:`batch_transcribe_pinyin_asr` -- PinyinASR (toneless pinyin, CTC);
* :func:`batch_transcribe_romaji_asr` -- RomajiASR (romaji phonemes, CTC).

Each returns ``(results, chunk_indices)`` where ``results`` is a list of
dicts (``{"text", ...}`` / ``{"text", "phonemes"}``) aligned with the chunk
indices, matching what ``lfa_api.process_asr_to_phonemes`` expects.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

#: Sample rate every ASR engine assumes (Qwen3-ASR / PinyinASR / RomajiASR).
ASR_SAMPLE_RATE = 16000

#: Kept for import compatibility with the upstream module.
DEFAULT_QWEN_ASR_PROMPT = (
    "你是一位专业的歌词转录助手，专注于从音频中准确提取歌词文本。"
)
DEFAULT_QWEN_ASR_PROMPT_EN = (
    "You are a professional lyrics transcription assistant focused on accurately "
    "extracting lyric text from audio."
)


def _manager():
    from msst_api.v2m import runtime

    return runtime.asr_manager()


def _to_asr_rate(waveform, sr) -> np.ndarray:
    """Resample a pipeline chunk (44100 Hz) to the ASR engines' 16 kHz.

    The vendored pipeline slices at 44.1 kHz and, upstream, writes chunk WAVs
    that the ASR subprocess reads back at 16 kHz.  This adapter feeds the raw
    array instead, so it must resample explicitly -- otherwise a 30 s chunk is
    handed to the model as ~83 s (2.76x), wrecking the transcript and inflating
    the HuBERT/ONNX attention memory.
    """

    wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
    sr = int(sr) if sr else ASR_SAMPLE_RATE
    if sr == ASR_SAMPLE_RATE or wav.size == 0:
        return np.ascontiguousarray(wav, dtype=np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    factor = gcd(sr, ASR_SAMPLE_RATE)
    resampled = resample_poly(wav, ASR_SAMPLE_RATE // factor, sr // factor)
    return np.ascontiguousarray(resampled, dtype=np.float32)


def _write_chunks(chunks, sr, temp_dir_path) -> tuple[list[str], list[int]]:
    temp_dir_path = Path(temp_dir_path)
    audio_paths: list[str] = []
    chunk_indices: list[int] = []
    for chunk_idx, chunk in enumerate(chunks):
        chunk_path = temp_dir_path / f"chunk_{chunk_idx}.wav"
        sf.write(str(chunk_path), chunk["waveform"], sr)
        audio_paths.append(str(chunk_path))
        chunk_indices.append(chunk_idx)
    return audio_paths, chunk_indices


def batch_transcribe_asr(
    chunks,
    sr,
    asr_model=None,
    temp_dir_path=None,
    asr_batch_size=1,
    language="zh",
    cancel_checker=None,
    asr_model_path=None,
    device=None,
    force_subprocess=False,
    asr_timeout_sec=180,
    asr_prompt=None,
    session=None,
):
    """Transcribe chunks with Qwen3-ASR and return (results, chunk_indices)."""

    del asr_model, force_subprocess, asr_timeout_sec, asr_prompt, session
    audio_paths, chunk_indices = _write_chunks(chunks, sr, temp_dir_path)
    if not audio_paths:
        return [], []

    manager = _manager()
    model_id = _model_id_for_path(asr_model_path, default_kind="qwen3")
    results = []
    for idx, chunk in zip(chunk_indices, chunks):
        if cancel_checker and cancel_checker():
            raise InterruptedError("ASR task cancelled")
        waveform = _to_asr_rate(chunk["waveform"], sr)
        result = manager.transcribe_path(
            "qwen3",
            Path(asr_model_path) if asr_model_path else _path_for(model_id),
            waveform,
            language=language,
            device=device,
        )
        results.append({"text": result.text, "language": result.language})
    return results, chunk_indices


def batch_transcribe_pinyin_asr(
    chunks,
    sr,
    temp_dir_path,
    model_dir,
    device=None,
    asr_batch_size=1,
    cancel_checker=None,
):
    """Transcribe chunks with PinyinASR; returns (results, chunk_indices)."""

    del asr_batch_size
    audio_paths, chunk_indices = _write_chunks(chunks, sr, temp_dir_path)
    if not audio_paths:
        return [], []

    manager = _manager()
    path = Path(model_dir)
    results = []
    for chunk in chunks:
        if cancel_checker and cancel_checker():
            raise InterruptedError("ASR task cancelled")
        waveform = _to_asr_rate(chunk["waveform"], sr)
        result = manager.transcribe_path("pinyin", path, waveform, device=device)
        results.append({"text": result.text, "phonemes": result.phonemes or []})
    return results, chunk_indices


def batch_transcribe_romaji_asr(
    chunks,
    sr,
    temp_dir_path,
    model_dir,
    device=None,
    asr_batch_size=1,
    cancel_checker=None,
):
    """Transcribe chunks with RomajiASR; returns (results, chunk_indices)."""

    del asr_batch_size
    audio_paths, chunk_indices = _write_chunks(chunks, sr, temp_dir_path)
    if not audio_paths:
        return [], []

    manager = _manager()
    path = Path(model_dir)
    results = []
    for chunk in chunks:
        if cancel_checker and cancel_checker():
            raise InterruptedError("ASR task cancelled")
        waveform = _to_asr_rate(chunk["waveform"], sr)
        result = manager.transcribe_path("romaji", path, waveform, device=device)
        results.append({"text": result.text, "phonemes": result.phonemes or []})
    return results, chunk_indices


def _path_for(model_id: str | None) -> Path:
    manager = _manager()
    entry = manager.get_entry(model_id) if model_id else None
    if entry is None:
        return Path("")
    return manager.store.path_for(entry)


def _model_id_for_path(asr_model_path, *, default_kind: str) -> str | None:
    """Best-effort mapping from a model directory back to a catalog id."""

    if not asr_model_path:
        return None
    name = Path(asr_model_path).name.lower()
    if "qwen" in name:
        return "qwen3-asr-1.7b"
    return None


class AsrSubprocessSession:
    """Compatibility shim; the service runs ASR in-process under the cache."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self) -> None:  # pragma: no cover - trivial
        return None

    def invalidate(self) -> None:  # pragma: no cover - trivial
        return None
