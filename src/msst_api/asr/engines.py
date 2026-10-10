"""ASR engine implementations for the ``asr`` / ``streaming_asr`` capabilities.

Three engines are provided:

* :class:`Qwen3ASREngine` -- Qwen3-ASR via the official ``qwen-asr`` package
  (transformers backend).  Qwen3-ASR is the only model that can also be used
  for streaming.
* :class:`PinyinASREngine` -- Chinese toneless-pinyin singing ASR (HuBERT CTC).
* :class:`RomajiASREngine` -- Japanese romaji phoneme singing ASR (HuBERT CTC).

The CTC engines reuse the ONNX runtimes vendored under
:mod:`msst_api.v2m._vendor.inference` (the same ones Vocal2Midi ships).  Heavy
imports stay inside the engine so the HTTP app and model listing work before
the optional runtime dependencies are installed.
"""

from __future__ import annotations

import gc
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

from ..config import Settings

logger = logging.getLogger("msst_api.asr.engines")

QWEN_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    "ja": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    "en": "English",
    "english": "English",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
}


@dataclass
class ASRResult:
    """One transcription result."""

    text: str = ""
    language: Optional[str] = None
    phonemes: Optional[list[str]] = None
    segments: Optional[list[dict]] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload: dict[str, Any] = {"text": self.text, "language": self.language}
        if self.phonemes is not None:
            payload["phonemes"] = self.phonemes
        if self.segments is not None:
            payload["segments"] = self.segments
        if self.extra:
            payload.update(self.extra)
        return payload


class ASREngine(Protocol):
    """Common surface shared by every ASR engine."""

    kind: str
    sample_rate: int
    supports_streaming: bool

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        context: str | None = None,
    ) -> ASRResult: ...

    def close(self) -> None: ...


def normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    value = str(language).strip()
    if not value:
        return None
    return QWEN_LANGUAGE_NAMES.get(value.lower(), value)


# ---------------------------------------------------------------------------
# Qwen3-ASR
# ---------------------------------------------------------------------------
class Qwen3ASREngine:
    """Qwen3-ASR (offline transcription, transformers backend)."""

    kind = "qwen3"
    sample_rate = 16000
    supports_streaming = True

    def __init__(self, model_path: Path, device: str, settings: Settings) -> None:
        import torch

        from qwen_asr import Qwen3ASRModel

        self.settings = settings
        self.model_path = Path(model_path)
        self.device = device
        dtype_name = (settings.qwen_asr_dtype or "bfloat16").strip().lower()
        self.dtype = getattr(torch, dtype_name, torch.bfloat16)
        logger.info(
            "Loading Qwen3-ASR model=%s device=%s dtype=%s",
            self.model_path,
            device,
            dtype_name,
        )
        self._model = Qwen3ASRModel.from_pretrained(
            str(self.model_path),
            dtype=self.dtype,
            device_map=device,
            max_inference_batch_size=8,
            max_new_tokens=512,
        )

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        context: str | None = None,
    ) -> ASRResult:
        import torch

        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        with torch.inference_mode():
            results = self._model.transcribe(
                audio=(wav, self.sample_rate),
                context=context or "",
                language=normalize_language(language),
            )
        result = results[0]
        return ASRResult(
            text=str(getattr(result, "text", "") or ""),
            language=getattr(result, "language", None),
        )

    def new_chunked_session(self, *, language: str | None = None, context: str | None = None):
        """Fallback streaming session that re-decodes a growing audio buffer."""

        return Qwen3ASRChunkedStreamingSession(self, language=language, context=context)

    def close(self) -> None:
        self._model = None
        gc.collect()
        _empty_cuda_cache()


def _vllm_available() -> bool:
    try:
        import vllm  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Pinyin / Romaji CTC engines (vendored ONNX runtimes)
# ---------------------------------------------------------------------------
class _CTCASREngine:
    """Shared implementation for the pinyin / romaji HuBERT-CTC runtimes."""

    sample_rate = 16000
    supports_streaming = False

    runtime_attr = ""

    def __init__(self, model_path: Path, device: str, settings: Settings) -> None:
        from ..v2m._vendor.inference import pinyin_asr, romaji_asr

        self.settings = settings
        self.model_path = Path(model_path)
        self.device = device
        runtime = pinyin_asr.runtime if self.kind == "pinyin" else romaji_asr.runtime
        logger.info("Loading %s ASR model=%s device=%s", self.kind, self.model_path, device)
        self._model = runtime.PinyinASROnnxModel.from_model_path(
            self.model_path, device=device, verbose=True
        ) if self.kind == "pinyin" else runtime.RomajiASROnnxModel.from_model_path(
            self.model_path, device=device, verbose=True
        )
        self.sample_rate = int(getattr(self._model, "sample_rate", 16000))
        self.provider = getattr(self._model, "provider", "cpu")

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        context: str | None = None,
    ) -> ASRResult:
        del language, context
        import soundfile as sf

        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        temp_dir = Path(tempfile.mkdtemp(prefix="asr-", dir=str(self.settings.temp_dir)))
        try:
            wav_path = temp_dir / "input.wav"
            sf.write(wav_path, wav, self.sample_rate)
            results = self._model.transcribe([str(wav_path)], batch_size=1)
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

        if not results:
            return ASRResult(text="")
        result = results[0]
        phonemes = list(result.get("phonemes") or [])
        return ASRResult(text=str(result.get("text", "")), phonemes=phonemes)

    def close(self) -> None:
        session = getattr(self._model, "session", None)
        try:
            if session is not None:
                session = None
        finally:
            self._model = None
            gc.collect()
            _empty_cuda_cache()


class PinyinASREngine(_CTCASREngine):
    kind = "pinyin"


class RomajiASREngine(_CTCASREngine):
    kind = "romaji"


ENGINE_KINDS = {
    "qwen3": Qwen3ASREngine,
    "qwen": Qwen3ASREngine,
    "pinyin": PinyinASREngine,
    "romaji": RomajiASREngine,
}


def build_engine(kind: str, model_path: Path, device: str, settings: Settings) -> ASREngine:
    try:
        engine_cls = ENGINE_KINDS[kind]
    except KeyError as error:
        raise ValueError(f"Unsupported ASR engine kind: {kind!r}") from error
    return engine_cls(model_path, device, settings)


# ---------------------------------------------------------------------------
# Streaming sessions
# ---------------------------------------------------------------------------
class StreamingSession(Protocol):
    """A single live ASR stream."""

    def feed(self, pcm16k: np.ndarray) -> ASRResult: ...

    def finish(self) -> ASRResult: ...

    def close(self) -> None: ...


class Qwen3ASRVLLMStreamingModel:
    """Qwen3-ASR vLLM backend (the heavy, shareable model object).

    The vLLM ``LLM`` instance is cached; each live stream gets its own
    :class:`ASRStreamingState` so multiple clients never share decoder state.
    """

    kind = "qwen3_streaming"
    sample_rate = 16000
    supports_streaming = True

    def __init__(self, model_path: Path, device: str, settings: Settings) -> None:
        from qwen_asr import Qwen3ASRModel

        self.settings = settings
        self.model_path = Path(model_path)
        self.device = device
        logger.info("Loading Qwen3-ASR vLLM streaming model=%s", self.model_path)
        self._model = Qwen3ASRModel.LLM(
            model=str(self.model_path),
            gpu_memory_utilization=0.7,
            max_new_tokens=64,
        )

    def new_session(self, *, language: str | None = None, context: str | None = None) -> "StreamingSession":
        state = self._model.init_streaming_state(
            unfixed_chunk_num=4, unfixed_token_num=5, chunk_size_sec=1.0
        )
        return Qwen3ASRVLLMStreamingSession(
            self._model, state, language=normalize_language(language)
        )

    def close(self) -> None:
        self._model = None
        gc.collect()
        _empty_cuda_cache()


class Qwen3ASRVLLMStreamingSession:
    """One live stream on top of a shared :class:`Qwen3ASRVLLMStreamingModel`."""

    def __init__(self, model, state, *, language: str | None = None) -> None:
        self._model = model
        self._state = state
        self.language = language
        self._text = ""
        self._detected_language: str | None = None

    def feed(self, pcm16k: np.ndarray) -> ASRResult:
        wav = np.asarray(pcm16k, dtype=np.float32).reshape(-1)
        self._model.streaming_transcribe(wav, self._state)
        self._text = str(getattr(self._state, "text", "") or self._text)
        self._detected_language = getattr(self._state, "language", None) or self._detected_language
        return ASRResult(text=self._text, language=self._detected_language)

    def finish(self) -> ASRResult:
        self._model.finish_streaming_transcribe(self._state)
        self._text = str(getattr(self._state, "text", "") or self._text)
        self._detected_language = getattr(self._state, "language", None) or self._detected_language
        return ASRResult(text=self._text, language=self._detected_language)

    def close(self) -> None:
        self._state = None


class Qwen3ASRChunkedStreamingSession:
    """Fallback streaming: re-decode a growing audio buffer offline.

    Used when the vLLM backend is unavailable.  It is not a true incremental
    decoder, but it produces monotonically refined partial transcripts and
    keeps the ``streaming_asr`` API usable in a transformers-only deployment.
    """

    def __init__(
        self,
        engine: Qwen3ASREngine,
        *,
        language: str | None = None,
        context: str | None = None,
    ) -> None:
        self.engine = engine
        self.language = language
        self.context = context
        self._buffer = np.zeros(0, dtype=np.float32)
        self._text = ""
        self._language: str | None = None

    def feed(self, pcm16k: np.ndarray) -> ASRResult:
        wav = np.asarray(pcm16k, dtype=np.float32).reshape(-1)
        if wav.size:
            self._buffer = np.concatenate([self._buffer, wav])
        result = self.engine.transcribe(
            self._buffer, language=self.language, context=self.context
        )
        if result.text:
            self._text = result.text
        if result.language:
            self._language = result.language
        return ASRResult(text=self._text, language=self._language)

    def finish(self) -> ASRResult:
        return self.feed(np.zeros(0, dtype=np.float32))

    def close(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
