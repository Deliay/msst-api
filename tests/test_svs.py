from __future__ import annotations

import io

import numpy as np
import pytest
import yaml

from msst_api.svs.g2p import G2pDictionary
from msst_api.svs.phrase import durations_ms_to_frames
from msst_api.svs.render import fit_duration_sum
from msst_api.svs.romanize import kana_to_romaji, romanize_japanese
from msst_api.svs.time_axis import TimeAxis
from msst_api.svs.ustx import Curve, PitchPoint, UstxProject


# ---------------------------------------------------------------------------
# Time axis
# ---------------------------------------------------------------------------
def test_time_axis_single_tempo():
    axis = TimeAxis([(0, 120.0)], 480)
    # 480 ticks = 1 beat = 0.5 s at 120 bpm.
    assert axis.tick_to_ms(480) == pytest.approx(500.0)
    assert axis.ms_to_tick(500.0) == 480
    assert axis.ms_to_tick(axis.tick_to_ms(1234)) == 1234


def test_time_axis_tempo_change():
    axis = TimeAxis([(0, 120.0), (480, 60.0)], 480)
    # First beat at 120 bpm = 500 ms; the next beat at 60 bpm = 1000 ms.
    assert axis.tick_to_ms(480) == pytest.approx(500.0)
    assert axis.tick_to_ms(960) == pytest.approx(1500.0)
    assert axis.ms_to_tick(1500.0) == 960


# ---------------------------------------------------------------------------
# g2p dictionary
# ---------------------------------------------------------------------------
def _write_dict(tmp_path, name="dsdict-ja.yaml"):
    content = """
symbols:
- symbol: ja/a
  type: vowel
- symbol: ja/s
  type: fricative
- symbol: ja/y
  type: semivowel
entries:
- grapheme: a
  phonemes:
  - ja/a
- grapheme: so
  phonemes:
  - ja/s
  - ja/o
- grapheme: kya
  phonemes:
  - ja/ky
  - ja/a
"""
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_g2p_dictionary(tmp_path):
    g2p = G2pDictionary.from_yaml(_write_dict(tmp_path))
    assert g2p.is_valid_symbol("ja/a")
    assert g2p.is_vowel("ja/a")
    assert not g2p.is_vowel("ja/s")
    assert g2p.is_glide("ja/y")
    # "so" -> [ja/s, ja/o]; ja/o is not a declared symbol, so it is filtered out.
    assert g2p.query("so") == ["ja/s"]
    assert g2p.query("missing") is None
    assert g2p.query_ci("A") == ["ja/a"]


# ---------------------------------------------------------------------------
# Romanization
# ---------------------------------------------------------------------------
def test_kana_to_romaji():
    assert kana_to_romaji("そ") == "so"
    assert kana_to_romaji("きゃ") == "kya"
    assert kana_to_romaji("ん") == "N"
    assert kana_to_romaji("がっこう") == "gakkou"


def test_romanize_japanese_leaves_romaji():
    assert romanize_japanese(["so", "ん", "か"]) == ["so", "N", "ka"]


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------
def test_curve_sampling_linear_and_clamp():
    curve = Curve("pitd", [PitchPoint(0, 0, "lin"), PitchPoint(10, 100, "lin")])
    assert curve.sample(-5) == 0
    assert curve.sample(5) == pytest.approx(50)
    assert curve.sample(100) == 100
    assert Curve("pitd", []).sample(3) == 0


# ---------------------------------------------------------------------------
# USTX round trip + pitd injection
# ---------------------------------------------------------------------------
def _minimal_ustx() -> dict:
    return {
        "ustx_version": "0.7",
        "resolution": 480,
        "bpm": 120.0,
        "expressions": {
            "pitd": {
                "name": "pitch deviation",
                "abbr": "pitd",
                "type": "Curve",
                "min": -1200,
                "max": 1200,
                "default_value": 0,
            }
        },
        "tempos": [{"position": 0, "bpm": 120.0}],
        "tracks": [{"phonemizer": "OpenUtau.Core.DefaultPhonemizer"}],
        "voice_parts": [
            {
                "position": 0,
                "notes": [
                    {"position": 0, "duration": 480, "tone": 60, "lyric": "a"},
                    {"position": 480, "duration": 480, "tone": 62, "lyric": "+"},
                ],
                "curves": [],
            }
        ],
    }


def test_ustx_roundtrip_and_pitd(tmp_path):
    path = tmp_path / "song.ustx"
    path.write_text(yaml.safe_dump(_minimal_ustx(), allow_unicode=True), encoding="utf-8")
    project = UstxProject.load(path)
    assert len(project.notes) == 2
    assert project.notes[1].is_slur
    assert len(project.note_groups()) == 1

    project.set_pitd([PitchPoint(0, 0), PitchPoint(480, 120)])
    data = yaml.safe_load(project.to_bytes())
    curves = data["voice_parts"][0]["curves"]
    assert curves[0]["abbr"] == "pitd"
    assert curves[0]["xs"] == [0, 480]
    assert curves[0]["ys"] == [0, 120]

    # Reload sees the injected curve.
    again = UstxProject.load(path)
    again.set_pitd([PitchPoint(0, 0), PitchPoint(480, 120)])
    reloaded = UstxProject(
        yaml.safe_load(again.to_bytes())
    )
    assert not reloaded.curve("pitd").is_empty
    assert reloaded.curve("pitd").sample(480) == 120


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------
def test_durations_ms_to_frames():
    # 10 ms frames: cumulative 11, 24, 41 ms -> 2, 3, 5 frames -> 2, 1, 2.
    assert durations_ms_to_frames([11, 13, 17], 10.0) == [2, 1, 2]


def test_fit_duration_sum():
    assert fit_duration_sum([1, 2, 3], 10) == [1, 2, 7]
    assert fit_duration_sum([1, 2, 30], 10) == [1, 2, 7]


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
class _StubDur:
    def __init__(self, dictionaries):
        self._dicts = dictionaries

    def g2p(self, name=None):
        if name in self._dicts:
            return self._dicts[name]
        raise FileNotFoundError(name)


class _StubVoicebank:
    def __init__(self, dictionaries):
        self.dur = _StubDur(dictionaries)


def test_detect_language_prefers_covered_dictionary(tmp_path):
    ja = G2pDictionary.from_yaml(_write_dict(tmp_path, "dsdict-ja.yaml"))
    zh = G2pDictionary.from_yaml(_write_dict(tmp_path, "dsdict-zh.yaml"))
    from msst_api.svs.pipeline import detect_language

    class Note:
        def __init__(self, lyric):
            self.lyric = lyric

    groups = [[Note("so")], [Note("kya")]]
    voicebank = _StubVoicebank({"dsdict-ja.yaml": ja, "dsdict-zh.yaml": zh})
    assert detect_language(voicebank, groups) == "ja"


# ---------------------------------------------------------------------------
# Phrases with no phonemes
# ---------------------------------------------------------------------------
class _StubPhonemizer:
    def __init__(self, *args, **kwargs):
        pass

    def process_phrase_with_axis(self, phrase_groups, time_axis):
        # Only the sentence starting at tick 0 resolves to a phoneme.
        if phrase_groups[0][0].position == 0:
            return {0: [("ja/a", 0)]}
        return {}


class _StubRenderVoicebank:
    dur = object()
    has_pitch = False
    sample_rate = 44100


def _two_phrase_ustx() -> dict:
    data = _minimal_ustx()
    # Two non-contiguous notes => two sentences; the second has no phonemes.
    data["voice_parts"][0]["notes"] = [
        {"position": 0, "duration": 480, "tone": 60, "lyric": "a"},
        {"position": 960, "duration": 480, "tone": 62, "lyric": "i"},
    ]
    return data


def _write_two_phrase_ustx(tmp_path):
    path = tmp_path / "song.ustx"
    path.write_text(
        yaml.safe_dump(_two_phrase_ustx(), allow_unicode=True), encoding="utf-8"
    )
    return path


def test_svs_pipeline_skips_phonemeless_phrase(monkeypatch, tmp_path):
    from msst_api.svs import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "DiffSingerPhonemizer", _StubPhonemizer)
    result = pipeline_module.run_svs_pipeline(
        _StubRenderVoicebank(),
        str(_write_two_phrase_ustx(tmp_path)),
        pipeline_module.SVSRequest(language="ja", auto_pitch=False, render=False),
    )
    assert result.phrases == 2
    assert len(result.warnings) == 1
    assert "Skipped 1 phrase" in result.warnings[0]


def test_svs_pipeline_all_phonemeless_raises(monkeypatch, tmp_path):
    from msst_api.svs import SVSError
    from msst_api.svs import pipeline as pipeline_module

    class _EmptyPhonemizer(_StubPhonemizer):
        def process_phrase_with_axis(self, phrase_groups, time_axis):
            return {}

    monkeypatch.setattr(pipeline_module, "DiffSingerPhonemizer", _EmptyPhonemizer)
    with pytest.raises(SVSError):
        pipeline_module.run_svs_pipeline(
            _StubRenderVoicebank(),
            str(_write_two_phrase_ustx(tmp_path)),
            pipeline_module.SVSRequest(language="ja", auto_pitch=False, render=False),
        )


# ---------------------------------------------------------------------------
# Vocoder ONNX provider options
# ---------------------------------------------------------------------------
def test_providers_passes_conv_search_through(monkeypatch):
    from msst_api.svs import voicebank as vb

    calls = []

    def fake_resolve(device, *, label="ONNX", conv_algo_search=None):
        calls.append((device, label, conv_algo_search))
        return "cuda", [("CUDAExecutionProvider", {"cudnn_conv_algo_search": conv_algo_search})]

    monkeypatch.setattr(vb, "resolve_onnx_providers", fake_resolve)
    providers = vb._providers("cuda", "voc.onnx", vb._VOCODER_CONV_ALGO_SEARCH)
    assert providers[0][1]["cudnn_conv_algo_search"] == "HEURISTIC"
    assert calls == [("cuda", "voc.onnx", "HEURISTIC")]


def test_session_pool_keys_by_conv_search(monkeypatch):
    from msst_api.svs import voicebank as vb

    created = []

    class _FakeSession:
        def __init__(self, path, providers):
            self.path = path
            self.providers = providers

    class _FakeOrt:
        GraphOptimizationLevel = type("G", (), {"ORT_ENABLE_ALL": 0})

        class SessionOptions:
            graph_optimization_level = None

        @staticmethod
        def InferenceSession(path, sess_options=None, providers=None):
            created.append((path, providers))
            return _FakeSession(path, providers)

    import sys

    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrt)

    def fake_resolve(device, *, label="ONNX", conv_algo_search=None):
        return "cuda", [("CUDAExecutionProvider", {"cudnn_conv_algo_search": conv_algo_search})]

    monkeypatch.setattr(vb, "resolve_onnx_providers", fake_resolve)
    pool = vb._SessionPool("cuda")
    a = pool.get("voc.onnx", conv_algo_search="HEURISTIC")
    b = pool.get("voc.onnx")  # different option -> separate session
    c = pool.get("voc.onnx", conv_algo_search="HEURISTIC")  # cached
    assert a is c
    assert a is not b
    assert a.providers[0][1]["cudnn_conv_algo_search"] == "HEURISTIC"
    assert b.providers[0][1]["cudnn_conv_algo_search"] is None
    assert len(created) == 2


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------
def test_svs_auto_pitch_form_mapping():
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from msst_api import main as main_module

    assert main_module._build_svs_request({}).auto_pitch is True
    assert main_module._build_svs_request({"auto_pitch": "false"}).auto_pitch is False
    assert main_module._build_svs_request({"auto_pitch": "true"}).auto_pitch is True
    # ``generate_pitch`` stays accepted as a legacy alias.
    assert main_module._build_svs_request({"generate_pitch": "false"}).auto_pitch is False


def test_svs_models_empty(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MSST_SVS_MODEL_DIR", str(tmp_path / "svs"))
    from msst_api.config import get_settings

    get_settings.cache_clear()
    from msst_api import main as main_module

    try:
        with TestClient(main_module.app) as client:
            response = client.get("/api/diffsinger/models")
            assert response.status_code == 200, response.text
            assert response.json() == {"total": 0, "voices": []}
    finally:
        get_settings.cache_clear()


def test_svs_inference(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from msst_api.svs import engine as engine_module
    from msst_api.svs.pipeline import SVSResult

    # A voicebank folder so the library can resolve it.
    bank = tmp_path / "svs" / "TestBank"
    bank.mkdir(parents=True)
    (bank / "dsconfig.yaml").write_text("phonemes: p.json\nacoustic: a.onnx\n")
    (bank / "character.txt").write_text("name=TestBank\nauthor=x\n")
    monkeypatch.setenv("MSST_SVS_MODEL_DIR", str(tmp_path / "svs"))
    from msst_api.config import get_settings

    get_settings.cache_clear()

    def fake_render(self, spec, ustx_path, request):
        samples = (0.1 * np.sin(np.arange(4410) / 20)).astype(np.float32)
        return SVSResult(
            samples=samples,
            sample_rate=44100,
            ustx_bytes=b"pitd: []\n",
            language="ja",
            phonemizer="DiffSingerJaPhonemizer",
            phrases=1,
            notes=1,
            duration_seconds=0.1,
        )

    monkeypatch.setattr(engine_module.SVSManager, "render", fake_render)
    from msst_api import main as main_module

    try:
        with TestClient(main_module.app) as client:
            listing = client.get("/api/diffsinger/models")
            assert listing.status_code == 200, listing.text
            assert listing.json()["total"] == 1

            response = client.post(
                "/api/diffsinger/inference",
                data={"model": "TestBank"},
                files={"ustx": ("song.ustx", b"notes: []\n", "application/x-yaml")},
            )
            assert response.status_code == 200, response.text
            assert response.headers["content-type"].startswith("audio/wav")
            assert response.headers["X-SVS-Model"] == "TestBank"

            missing = client.post(
                "/api/diffsinger/inference",
                data={"model": "nope"},
                files={"ustx": ("song.ustx", b"notes: []\n", "application/x-yaml")},
            )
            assert missing.status_code == 404
    finally:
        get_settings.cache_clear()
