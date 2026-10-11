from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from msst_api import main as main_module  # noqa: E402
from msst_api.v2m import pipeline as pipeline_module  # noqa: E402
from msst_api.v2m.manager import V2MManager  # noqa: E402


def _wav_bytes(sample_rate: int = 44100, seconds: float = 0.2) -> bytes:
    import wave

    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    mono = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(mono.tobytes())
    return buffer.getvalue()


@pytest.fixture()
def v2m_client(monkeypatch):
    def fake_pipeline(**kwargs):
        return {
            "vocal.mid": b"MIDI",
            "vocal.txt": b"onset\toffset\tpitch\n",
        }

    monkeypatch.setattr(pipeline_module, "run_v2m_pipeline", fake_pipeline)
    with TestClient(main_module.app) as client:
        yield client


def test_v2m_list_models(v2m_client):
    response = v2m_client.get("/api/v2m/models")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 5
    ids = {m["id"] for m in payload["models"]}
    assert {"game", "tifa", "hubertfa", "rmvpe", "kashi-g2p-onnx"} <= ids


def test_v2m_unknown_model_404(v2m_client):
    assert v2m_client.get("/api/v2m/models/nope").status_code == 404


def test_v2m_inference_returns_zip(v2m_client):
    response = v2m_client.post(
        "/api/v2m/inference",
        files={"audio": ("vocal.wav", _wav_bytes(), "audio/wav")},
        data={"language": "zh", "output_formats": "mid,txt"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = archive.namelist()
    assert "vocal.mid" in names
    assert "manifest.json" in names


def test_v2m_inference_rejects_bad_format(v2m_client):
    response = v2m_client.post(
        "/api/v2m/inference",
        files={"audio": ("vocal.wav", _wav_bytes(), "audio/wav")},
        data={"output_formats": "xyz"},
    )
    assert response.status_code == 400


def test_v2m_g2p_endpoint(v2m_client, monkeypatch):
    from msst_api.v2m import capabilities

    monkeypatch.setattr(
        capabilities,
        "g2p_reading",
        lambda manager, text, **kwargs: {"reading": "きょう", "edges": []},
    )
    response = v2m_client.post("/api/v2m/g2p", data={"text": "今日", "language": "ja"})
    assert response.status_code == 200, response.text
    assert response.json()["reading"] == "きょう"


def test_v2m_notes_endpoint(v2m_client, monkeypatch):
    from msst_api.v2m import capabilities

    monkeypatch.setattr(
        capabilities,
        "extract_notes",
        lambda manager, wav, sr, **kwargs: [
            {"onset": 0.0, "offset": 0.5, "pitch": 60.0, "lyric": "a"}
        ],
    )
    response = v2m_client.post(
        "/api/v2m/notes",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["count"] == 1


def test_v2m_models_report_downloaded_flag(v2m_client):
    payload = v2m_client.get("/api/v2m/models").json()
    assert all(isinstance(m["downloaded"], bool) for m in payload["models"])


def test_v2m_slice_returns_chunk_zip(v2m_client):
    # Exercises the vendored slicer end-to-end (no model required).
    response = v2m_client.post(
        "/api/v2m/slice",
        files={"audio": ("a.wav", _wav_bytes(seconds=1.0), "audio/wav")},
        data={"slice_min_sec": "0.2", "slice_max_sec": "0.5"},
    )
    assert response.status_code == 200, response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = archive.namelist()
    assert "manifest.json" in names
    assert any(name.endswith(".wav") for name in names)


def test_v2m_align_requires_text(v2m_client):
    response = v2m_client.post(
        "/api/v2m/align",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 400


def test_v2m_pipeline_no_lyrics_end_to_end(monkeypatch, tmp_path):
    """Runs the real pipeline (slicing + MIDI export) with the model stubbed."""

    import soundfile as sf

    from msst_api.v2m._vendor.inference.io.note_io import NoteInfo
    from msst_api.v2m._vendor.inference.pipeline import auto_lyric_hybrid as hybrid

    monkeypatch.setattr(
        V2MManager, "resolve_path", lambda self, model_id, source=None: tmp_path
    )
    monkeypatch.setattr(hybrid, "load_game_model", lambda *a, **k: object())

    def fake_pitches_only(chunks, sr, game_model, ts, seg_threshold, seg_radius, est_threshold, batch_size=4, cancel_checker=None, language=None):
        return [
            NoteInfo(onset=0.0, offset=0.4, pitch=60.0, lyric=""),
            NoteInfo(onset=0.4, offset=0.8, pitch=62.0, lyric=""),
        ]

    monkeypatch.setattr(hybrid, "extract_pitches_only", fake_pitches_only)

    wav_path = tmp_path / "vocal.wav"
    sf.write(str(wav_path), np.zeros(44100, dtype=np.float32), 44100)

    settings = main_module.get_settings()
    outputs = pipeline_module.run_v2m_pipeline(
        settings=settings,
        asr_manager=None,  # unused in no-lyrics mode
        v2m=V2MManager(settings),
        audio_path=wav_path,
        output_filename="vocal.wav",
        output_dir=tmp_path / "out",
        request=pipeline_module.V2MRequest(
            language="zh", output_lyrics=False, output_formats=["mid", "txt"]
        ),
    )
    assert "vocal.mid" in outputs
    assert outputs["vocal.mid"][:4] == b"MThd"




def test_onnx_cuda_provider_uses_memory_frugal_arena():
    from msst_api.v2m._vendor.inference import device_utils

    assert device_utils.resolve_onnx_providers("cpu") == ("cpu", ["CPUExecutionProvider"])
    if device_utils._cuda_provider_available():
        name, providers = device_utils.resolve_onnx_providers("cuda")
        assert name == "cuda"
        assert isinstance(providers[0], tuple)
        assert providers[0][0] == "CUDAExecutionProvider"
        # Avoid ORT's kNextPowerOfTwo arena rounding, which inflates VRAM.
        assert providers[0][1]["arena_extend_strategy"] == "kSameAsRequested"
        # DEFAULT conv search avoids cuDNN's multi-GB scratch on new shapes.
        assert providers[0][1]["cudnn_conv_algo_search"] == "DEFAULT"
        # Callers can opt into the heuristic search (e.g. the SVS vocoder).
        _, heuristic = device_utils.resolve_onnx_providers(
            "cuda", conv_algo_search="HEURISTIC"
        )
        assert heuristic[0][1]["cudnn_conv_algo_search"] == "HEURISTIC"


def test_to_asr_rate_resamples_to_16k():
    from msst_api.v2m._vendor.inference.API.asr_api import ASR_SAMPLE_RATE, _to_asr_rate

    one_sec_44k = np.sin(np.linspace(0.0, 2 * np.pi * 440, 44100)).astype(np.float32)
    out = _to_asr_rate(one_sec_44k, 44100)
    assert abs(out.size - ASR_SAMPLE_RATE) <= 1
    assert out.dtype == np.float32

    already_16k = np.zeros(1234, dtype=np.float32)
    assert _to_asr_rate(already_16k, 16000).size == 1234


def test_cap_chunk_length_splits_only_over_length_chunks():
    from msst_api.v2m._vendor.inference.pipeline.auto_lyric_hybrid import (
        _cap_chunk_length,
    )

    sr = 1000
    long_chunk = [{"offset": 2.0, "waveform": np.zeros(12500, dtype=np.float32)}]
    capped = _cap_chunk_length(long_chunk, sr, 5.0)
    assert len(capped) == 3
    assert [round(c["offset"], 2) for c in capped] == [2.0, 7.0, 12.0]
    assert [c["waveform"].shape[-1] for c in capped] == [5000, 5000, 2500]

    short = [{"offset": 1.0, "waveform": np.zeros(1000, dtype=np.float32)}]
    assert _cap_chunk_length(short, sr, 5.0) == short


def test_v2m_request_defaults_to_auto_language():
    assert pipeline_module.V2MRequest().language == "auto"


def test_map_detected_language():
    from msst_api.v2m.pipeline import _map_detected_language

    assert _map_detected_language("Chinese") == "zh"
    assert _map_detected_language("Mandarin") == "zh"
    assert _map_detected_language("Cantonese") == "yue"
    assert _map_detected_language("Japanese") == "ja"
    assert _map_detected_language("English") == "en"
    assert _map_detected_language("Chinese,English") == "zh"
    assert _map_detected_language("Korean") is None
    assert _map_detected_language("") is None
    assert _map_detected_language(None) is None


def test_resolve_lyric_output_mode_auto_prefers_romanised():
    from msst_api.v2m.pipeline import resolve_lyric_output_mode

    # auto (the default) renders lyrics the DiffSinger front-end can read.
    assert resolve_lyric_output_mode("zh", "auto") == "pinyin"
    assert resolve_lyric_output_mode("zh", "") == "pinyin"
    assert resolve_lyric_output_mode("ja", "auto") == "romaji"
    assert resolve_lyric_output_mode("yue", "auto") == "jyutping"
    assert resolve_lyric_output_mode("en", "auto") == "word"
    # An explicit mode is always respected.
    assert resolve_lyric_output_mode("zh", "hanzi") == "hanzi"
    assert resolve_lyric_output_mode("ja", "kana") == "kana"



def test_detect_language_uses_asr_and_falls_back(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from msst_api.v2m import pipeline as pl
    from msst_api.v2m._vendor.inference.io import audio_io

    monkeypatch.setattr(
        audio_io,
        "load_audio",
        lambda path, sr, mono=True: (np.zeros(sr * 5, dtype=np.float32), sr),
    )
    settings = SimpleNamespace(v2m_default_language="zh", v2m_language_detect_sec=30.0)

    calls = {}

    class FakeASR:
        def __init__(self, language):
            self.language = language

        def transcribe_path(self, kind, path, audio, language=None, device=None):
            calls["language"] = language
            return SimpleNamespace(language=self.language, text="x")

    code = pl.detect_language(
        asr_manager=FakeASR("Japanese"),
        qwen_path="/qwen",
        audio_path=Path("a.wav"),
        settings=settings,
    )
    assert code == "ja"
    assert calls["language"] is None  # unforced -> Qwen auto-detects

    fallback = pl.detect_language(
        asr_manager=FakeASR("Korean"),
        qwen_path="/qwen",
        audio_path=Path("a.wav"),
        settings=settings,
    )
    assert fallback == "zh"
