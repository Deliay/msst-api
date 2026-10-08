from __future__ import annotations

import numpy as np
import pytest

from msst_api.audio import build_zip, encode_audio


def _sine(sample_rate: int = 44100, seconds: float = 0.25) -> np.ndarray:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    mono = 0.5 * np.sin(2 * np.pi * 440 * t)
    return np.stack([mono, mono], axis=1).astype(np.float32)


def test_encode_wav_roundtrip() -> None:
    import io

    import soundfile as sf

    audio = _sine()
    data = encode_audio(audio, 44100, fmt="wav", subtype="PCM_16")
    decoded, sr = sf.read(io.BytesIO(data), always_2d=True)
    assert sr == 44100
    assert decoded.shape == audio.shape


def test_encode_flac() -> None:
    data = encode_audio(_sine(), 44100, fmt="flac", subtype="PCM_24")
    assert data[:4] == b"fLaC"


def test_invalid_format_raises() -> None:
    with pytest.raises(ValueError):
        encode_audio(_sine(), 44100, fmt="ogg")


def test_zip_contains_files() -> None:
    archive = build_zip({"vocals.wav": b"abc", "manifest.json": b"{}"})
    assert archive[:2] == b"PK"
