from __future__ import annotations

import io
import wave

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from msst_api import main as main_module  # noqa: E402
from msst_api.asr.engines import ASRResult  # noqa: E402
from msst_api.asr.manager import ASRManager  # noqa: E402


def _wav_bytes(sample_rate: int = 16000, seconds: float = 0.2) -> bytes:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    mono = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(mono.tobytes())
    return buffer.getvalue()


class _FakeStream:
    def __init__(self) -> None:
        self.feeds = 0

    def feed(self, pcm):
        self.feeds += 1
        return ASRResult(text="hello", language="English")

    def finish(self):
        return ASRResult(text="hello world", language="English")

    def close(self):
        pass


@pytest.fixture()
def asr_client(monkeypatch):
    monkeypatch.setattr(
        ASRManager,
        "transcribe",
        lambda self, model_id, audio, **kwargs: ASRResult(
            text="你好世界", language="Chinese"
        ),
    )
    monkeypatch.setattr(ASRManager, "open_stream", lambda self, model_id, **kwargs: _FakeStream())
    with TestClient(main_module.app) as client:
        yield client


def test_asr_list_models(asr_client):
    response = asr_client.get("/api/asr/models")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 4
    kinds = {m["kind"] for m in payload["models"]}
    assert kinds == {"qwen3", "pinyin", "romaji"}


def test_asr_models_filter_by_task(asr_client):
    streaming = asr_client.get("/api/asr/models", params={"task": "streaming"}).json()
    assert {m["id"] for m in streaming["models"]} == {"qwen3-asr-1.7b", "qwen3-asr-0.6b"}
    single = asr_client.get("/api/asr/models", params={"task": "transcribe"}).json()
    assert single["total"] == 4


def test_asr_unknown_model_404(asr_client):
    assert asr_client.get("/api/asr/models/nope").status_code == 404


def test_asr_get_model_detail(asr_client):
    detail = asr_client.get("/api/asr/models/pinyin-asr")
    assert detail.status_code == 200
    assert detail.json()["tasks"] == ["transcribe"]


def test_asr_transcribe(asr_client):
    response = asr_client.post(
        "/api/asr/transcribe",
        files={"audio": ("in.wav", _wav_bytes(), "audio/wav")},
        data={"model": "qwen3-asr-1.7b", "language": "zh"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["text"] == "你好世界"
    assert payload["model"] == "qwen3-asr-1.7b"
    assert "elapsed_seconds" in payload


def test_asr_streaming_websocket(asr_client):
    with asr_client.websocket_connect("/api/asr/streaming?model=qwen3-asr-1.7b") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        ws.send_bytes(np.zeros(1600, dtype="<f4").tobytes())
        partial = ws.receive_json()
        assert partial["type"] == "partial"
        assert partial["text"] == "hello"
        ws.send_text("finish")
        final = ws.receive_json()
        assert final["type"] == "final"
        assert final["text"] == "hello world"


def test_asr_populates_health(asr_client):
    health = asr_client.get("/health").json()
    assert health["asr_model_dir"]
    assert health["asr_streaming_backend"] == "auto"


def test_all_subsystems_share_resource_limits(asr_client):
    state = asr_client.app.state
    assert state.manager._cache is state.asr_manager._cache
    assert state.rvc_manager._cache is state.asr_manager._cache
    assert state.v2m_manager.cache is state.asr_manager._cache
