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

#: Output containers the RVC endpoint accepts (the reference plugin's set).
RVC_OUTPUT_FORMATS = {
    "wav",
    "mp3",
    "flac",
    "ogg",
    "opus",
    "m4a",
    "aac",
    "aiff",
    "ac3",
}

_AUDIO_MIME = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "opus": "audio/opus",
    "m4a": "audio/mp4",
    "aac": "audio/aac",
    "aiff": "audio/aiff",
    "ac3": "audio/ac3",
}

_FFMPEG_OUTPUT = {
    "mp3": ["-f", "mp3", "-codec:a", "libmp3lame", "-q:a", "2"],
    "ogg": ["-f", "ogg", "-codec:a", "libvorbis", "-q:a", "5"],
    "opus": ["-f", "opus", "-codec:a", "libopus", "-b:a", "192k"],
    "m4a": [
        "-f", "mp4", "-codec:a", "aac", "-b:a", "256k",
        "-movflags", "frag_keyframe+empty_moov",
    ],
    "aac": ["-f", "adts", "-codec:a", "aac", "-b:a", "256k"],
    "aiff": ["-f", "aiff", "-codec:a", "pcm_s16be"],
    "ac3": ["-f", "ac3", "-codec:a", "ac3", "-b:a", "192k"],
}


def decode_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Decode an audio file to a channels-first ``float32`` array."""

    mix, _ = librosa.load(str(path), sr=sample_rate, mono=False)
    mix = np.asarray(mix, dtype=np.float32)
    if mix.ndim == 1:
        mix = mix[np.newaxis, :]
    return mix


def decode_audio_mono(path: Path, sample_rate: int) -> np.ndarray:
    """Decode an audio file to a flat mono ``float32`` array (RVC input)."""

    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return np.asarray(audio, dtype=np.float32).flatten()


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


def encode_audio_dynamic(
    audio: np.ndarray, sample_rate: int, fmt: str
) -> tuple[bytes, str]:
    """Encode a flat/mono or ``(samples, channels)`` array for the RVC endpoint.

    Returns ``(content, media_type)``.  WAV/FLAC use libsndfile; the remaining
    containers are transcoded through ffmpeg, mirroring the reference plugin.
    """

    fmt = (fmt or "wav").lower().lstrip(".")
    if fmt not in RVC_OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported output format {fmt!r}; expected one of {sorted(RVC_OUTPUT_FORMATS)}"
        )

    audio = np.asarray(audio, dtype=np.float32)

    if fmt in {"wav", "flac"}:
        buffer = io.BytesIO()
        sf.write(buffer, audio, int(sample_rate), format=fmt.upper())
        return buffer.getvalue(), _AUDIO_MIME[fmt]

    # Everything else goes through an ffmpeg pipe (WAV is the interchange).
    if not _ffmpeg_available():
        raise RuntimeError(f"ffmpeg is required to encode {fmt.upper()} output")
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio, int(sample_rate), format="WAV")
    process = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", "pipe:0"]
        + _FFMPEG_OUTPUT[fmt]
        + ["pipe:1"],
        input=wav_buffer.getvalue(),
        capture_output=True,
        check=False,
    )
    if process.returncode != 0 or not process.stdout:
        raise RuntimeError(
            f"ffmpeg failed to encode {fmt.upper()}: "
            f"{process.stderr.decode(errors='ignore')[-400:]}"
        )
    return process.stdout, _AUDIO_MIME[fmt]


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
