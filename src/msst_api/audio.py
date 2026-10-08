"""Audio decoding/encoding helpers used by the inference endpoint."""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger("msst_api.audio")

SUPPORTED_OUTPUT_FORMATS = {"wav", "flac", "mp3"}
DEFAULT_SUBTYPES = {"wav": "PCM_16", "flac": "PCM_24"}
VALID_SUBTYPES = {
    "wav": {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"},
    "flac": {"PCM_S8", "PCM_16", "PCM_24"},
}


def decode_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Decode an audio file to a channels-first ``float32`` array."""

    mix, _ = librosa.load(str(path), sr=sample_rate, mono=False)
    mix = np.asarray(mix, dtype=np.float32)
    if mix.ndim == 1:
        mix = mix[np.newaxis, :]
    return mix


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def encode_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    fmt: str,
    subtype: str | None = None,
) -> bytes:
    """Encode a ``(samples, channels)`` array to the requested container."""

    fmt = fmt.lower()
    if fmt not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format {fmt!r}")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    if fmt in {"wav", "flac"}:
        chosen = (subtype or DEFAULT_SUBTYPES[fmt]).upper()
        if chosen not in VALID_SUBTYPES[fmt]:
            raise ValueError(
                f"Invalid {fmt} subtype {chosen!r}; expected one of {sorted(VALID_SUBTYPES[fmt])}"
            )
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format=fmt.upper(), subtype=chosen)
        return buffer.getvalue()

    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg is required to encode MP3 output")
    channels = audio.shape[1]
    process = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "f32le", "-ar", str(sample_rate), "-ac", str(channels),
            "-i", "pipe:0", "-codec:a", "libmp3lame", "-b:a", subtype or "320k",
            "-f", "mp3", "pipe:1",
        ],
        input=audio.astype(np.float32).tobytes(),
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to encode MP3: {process.stderr.decode(errors='ignore')}"
        )
    return process.stdout


def build_zip(files: dict[str, bytes]) -> bytes:
    """Package ``{name: content}`` into an in-memory zip archive."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def save_upload(data: bytes, suffix: str, temp_dir: Path) -> Path:
    """Persist uploaded bytes to a temp file so librosa/ffmpeg can decode it."""

    temp_dir.mkdir(parents=True, exist_ok=True)
    handle, raw_path = tempfile.mkstemp(suffix=suffix or ".wav", dir=str(temp_dir))
    with open(handle, "wb") as file_handle:
        file_handle.write(data)
    return Path(raw_path)
