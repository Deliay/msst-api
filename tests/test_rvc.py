from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from msst_api import main as main_module  # noqa: E402
from msst_api.rvc_engine import (  # noqa: E402
    RVCManager,
    RVCVoiceLibrary,
    RVCVoiceNotFoundError,
)


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


def _populate(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "alice.pth").write_bytes(b"fake")
    (model_dir / "alice.index").write_bytes(b"idx")
    (model_dir / "bob_v2.pth").write_bytes(b"fake")
    (model_dir / "bob_v2.index").write_bytes(b"idx")


# ---------------------------------------------------------------------------
# Library resolution
# ---------------------------------------------------------------------------
def test_library_lists_voices(tmp_path):
    _populate(tmp_path)
    library = RVCVoiceLibrary(tmp_path)
    voices = library.all()
    assert [v.id for v in voices] == ["alice", "bob_v2"]
    assert voices[0].has_index is True
    assert voices[0].to_public_dict()["downloaded"] is True


def test_library_resolves_by_name_and_stem(tmp_path):
    _populate(tmp_path)
    library = RVCVoiceLibrary(tmp_path)
    assert library.get("alice.pth").id == "alice"
    assert library.get("alice").id == "alice"
    assert library.get("BOB_V2").id == "bob_v2"
    # Unique substring.
    assert library.get("bob").id == "bob_v2"


def test_library_ambiguous_and_missing(tmp_path):
    _populate(tmp_path)
    library = RVCVoiceLibrary(tmp_path)
    with pytest.raises(RVCVoiceNotFoundError):
        library.get("nope")
    # A substring that matches two voices (and no exact stem) is ambiguous.
    (tmp_path / "alice_clone.pth").write_bytes(b"fake")
    with pytest.raises(RVCVoiceNotFoundError):
        library.get("alic")


def test_library_resolve_index(tmp_path):
    _populate(tmp_path)
    library = RVCVoiceLibrary(tmp_path)
    assert library.resolve_index("alice").name == "alice.index"
    assert library.resolve_index("alice.index").name == "alice.index"
    assert library.resolve_index("") is None


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------
@pytest.fixture()
def rvc_client(monkeypatch, tmp_path):
    model_dir = tmp_path / "rvc_models"
    _populate(model_dir)

    def fake_convert(self, spec, audio, params):
        return np.zeros(2048, dtype=np.float32), 40000

    monkeypatch.setattr(RVCManager, "convert", fake_convert)
    # Tests run without a GPU: bypass the GPU-only device policy.
    monkeypatch.setattr(main_module, "resolve_device", lambda requested, settings: requested)

    with TestClient(main_module.app) as client:
        client.app.state.rvc_manager.library = RVCVoiceLibrary(model_dir)
        yield client


def test_rvc_list_and_get_models(rvc_client):
    listing = rvc_client.get("/api/rvc/models")
    assert listing.status_code == 200
    payload = listing.json()
    assert payload["total"] == 2
    assert {v["id"] for v in payload["voices"]} == {"alice", "bob_v2"}

    detail = rvc_client.get("/api/rvc/models/alice")
    assert detail.status_code == 200
    assert detail.json()["has_index"] is True


def test_rvc_unknown_model_returns_404(rvc_client):
    assert rvc_client.get("/api/rvc/models/nope").status_code == 404


def test_rvc_inference_returns_audio(rvc_client):
    response = rvc_client.post(
        "/api/rvc/inference",
        files={"audio": ("in.wav", _wav_bytes(), "audio/wav")},
        data={"model": "alice", "f0_method": "rmvpe", "output_format": "wav"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("audio/")
    assert response.headers["x-rvc-model"] == "alice.pth"


def test_rvc_inference_rejects_bad_format(rvc_client):
    response = rvc_client.post(
        "/api/rvc/inference",
        files={"audio": ("in.wav", _wav_bytes(), "audio/wav")},
        data={"model": "alice", "output_format": "xyz"},
    )
    assert response.status_code == 400


def test_rvc_inference_rejects_bad_f0_method(rvc_client):
    response = rvc_client.post(
        "/api/rvc/inference",
        files={"audio": ("in.wav", _wav_bytes(), "audio/wav")},
        data={"model": "alice", "f0_method": "swift"},
    )
    assert response.status_code == 400


def test_health_reports_rvc_dir(rvc_client):
    health = rvc_client.get("/health")
    assert health.status_code == 200
    assert health.json()["rvc_model_dir"]
