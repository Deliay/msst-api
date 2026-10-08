from __future__ import annotations

import io
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from msst_api import main as main_module  # noqa: E402
from msst_api.separator import ModelManager  # noqa: E402


def _wav_bytes(sample_rate: int = 44100, seconds: float = 0.1) -> bytes:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    mono = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(mono.tobytes())
    return buffer.getvalue()


class _FakeSeparator:
    sample_rate = 44100
    device = "cpu"

    def separate(self, mix):
        length = mix.shape[-1]
        vocals = np.zeros((length, 1), dtype=np.float32)
        return {"vocals": vocals, "instrumental": vocals}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(ModelManager, "get", lambda self, spec: _FakeSeparator())

    def fake_separate(self, spec, mix):
        return _FakeSeparator().separate(mix)

    monkeypatch.setattr(ModelManager, "separate", fake_separate)
    # Never hit the network during tests: pretend the checkpoint is present.
    monkeypatch.setattr(
        main_module, "download_model", lambda *args, **kwargs: Path("/tmp/fake.ckpt")
    )
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_root_and_health(client):
    assert client.get("/").status_code == 200
    health = client.get("/health")
    assert health.status_code == 200
    assert "cuda_available" in health.json()


def test_list_and_get_models(client):
    listing = client.get("/api/msst/models")
    assert listing.status_code == 200
    payload = listing.json()
    assert payload["total"] == 49

    detail = client.get("/api/msst/models/model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    assert detail.status_code == 200
    assert detail.json()["model_type"] == "bs_roformer"


def test_unknown_model_returns_404(client):
    assert client.get("/api/msst/models/nope.ckpt").status_code == 404


def test_inference_returns_zip(client):
    response = client.post(
        "/api/msst/inference",
        files={"audio": ("in.wav", _wav_bytes(), "audio/wav")},
        data={"model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt", "output_format": "wav"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = archive.namelist()
    assert "vocals.wav" in names
    assert "manifest.json" in names


def test_inference_json_and_stem_filter(client):
    response = client.post(
        "/api/msst/inference",
        files={"audio": ("in.wav", _wav_bytes(), "audio/wav")},
        data={
            "model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "response_format": "json",
            "stems": "vocals",
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert list(payload["stems"].keys()) == ["vocals"]


def test_inference_rejects_bad_format(client):
    response = client.post(
        "/api/msst/inference",
        files={"audio": ("in.wav", _wav_bytes(), "audio/wav")},
        data={"model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt", "output_format": "ogg"},
    )
    assert response.status_code == 400
