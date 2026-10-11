from __future__ import annotations

from pathlib import Path

import pytest

from msst_api.config import Settings
from msst_api.model_store import (
    ModelAssetNotFound,
    ModelEntry,
    ModelStore,
    register_prepare,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(model_dir=tmp_path / "models")


def test_is_downloaded_checks_required_files(tmp_path):
    settings = _settings(tmp_path)
    store = ModelStore(settings)
    entry = ModelEntry(
        id="bundle",
        capability="asr",
        kind="x",
        name="Bundle",
        dir_name="bundle",
        required_files=("config.json", "model.onnx"),
    )
    store.register([entry], "asr")
    assert store.is_downloaded(entry) is False

    dest = store.path_for(entry)
    dest.mkdir(parents=True)
    (dest / "config.json").write_text("{}")
    assert store.is_downloaded(entry) is False
    (dest / "model.onnx").write_bytes(b"x")
    assert store.is_downloaded(entry) is True


def test_ensure_downloads_snapshot_into_place(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    store = ModelStore(settings)
    entry = ModelEntry(
        id="bundle",
        capability="asr",
        kind="x",
        name="Bundle",
        dir_name="bundle",
        required_files=("config.json",),
        snapshots=({"backend": "huggingface", "repo_id": "org/repo"},),
    )
    store.register([entry], "asr")

    calls = {}

    def fake_snapshot(repo_id, dest, *, backend, **kwargs):
        calls["repo_id"] = repo_id
        calls["backend"] = backend
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "config.json").write_text("{}")
        return Path(dest)

    monkeypatch.setattr("msst_api.model_store.download_snapshot", fake_snapshot)
    path = store.ensure(entry)
    assert path.is_dir()
    assert (path / "config.json").is_file()
    assert calls == {"repo_id": "org/repo", "backend": "huggingface"}


def test_ensure_without_source_raises(tmp_path):
    settings = _settings(tmp_path)
    store = ModelStore(settings)
    entry = ModelEntry(id="none", capability="v2m", kind="x", name="None", dir_name="none")
    store.register([entry], "v2m")
    with pytest.raises(Exception):
        store.ensure(entry)


def test_unknown_model_raises(tmp_path):
    store = ModelStore(_settings(tmp_path))
    with pytest.raises(ModelAssetNotFound):
        store.get("does-not-exist")


def test_custom_catalog_override(tmp_path):
    settings = _settings(tmp_path)
    store = ModelStore(settings)
    store.register(
        [ModelEntry(id="a", capability="asr", kind="x", name="A")], "asr"
    )
    catalog = tmp_path / "cat.json"
    catalog.write_text(
        '{"models": [{"id": "a", "kind": "y", "name": "A2", "tasks": ["transcribe"]}]}',
        encoding="utf-8",
    )
    store.load_json(catalog, "asr", override=True)
    assert store.get("a").kind == "y"
    assert store.get("a").tasks == ("transcribe",)


def test_required_globs_presence(tmp_path):
    settings = _settings(tmp_path)
    store = ModelStore(settings)
    entry = ModelEntry(
        id="ctc",
        capability="asr",
        kind="pinyin",
        name="CTC",
        dir_name="ctc",
        required_globs=("*.onnx", "*vocab*.json"),
    )
    store.register([entry], "asr")
    dest = store.path_for(entry)
    dest.mkdir(parents=True)
    (dest / "model_fp16_dynb.onnx").write_bytes(b"onnx")
    assert store.is_downloaded(entry) is False  # vocab missing
    (dest / "pinyin_vocab.json").write_text("{}")
    assert store.is_downloaded(entry) is True


def test_archive_extraction_finds_nested_bundle(tmp_path, monkeypatch):
    import shutil
    import zipfile

    settings = _settings(tmp_path)
    store = ModelStore(settings)
    entry = ModelEntry(
        id="pinyin-asr",
        capability="asr",
        kind="pinyin",
        name="CTC",
        dir_name="pinyinASR",
        required_globs=("*.onnx", "*vocab*.json"),
        archives=({"url": "https://example.invalid/model_onnx.zip"},),
    )
    store.register([entry], "asr")

    archive = tmp_path / "model_onnx.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("pinyinASR/model_fp16_dynb.onnx", b"onnx")
        handle.writestr("pinyinASR/pinyin_vocab.json", "{}")

    def fake_download_file(url, dest, **kwargs):
        shutil.copy(archive, dest)
        return dest

    monkeypatch.setattr("msst_api.download.download_file", fake_download_file)
    path = store.ensure(entry)
    assert (path / "model_fp16_dynb.onnx").is_file()
    assert (path / "pinyin_vocab.json").is_file()


def test_prepare_step_runs_after_download(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    store = ModelStore(settings)
    entry = ModelEntry(
        id="tifa-like",
        capability="v2m",
        kind="x",
        name="Bundle",
        dir_name="bundle",
        required_files=("model.onnx", "config.json"),
        archives=({"url": "https://example.invalid/source.zip"},),
        prepare="test-prepare-after-download",
        prepare_inputs=("source.pt",),
    )
    store.register([entry], "v2m")

    def fake_archive(self, entry_, dest, archive, settings_):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "source.pt").write_bytes(b"pt")

    monkeypatch.setattr(ModelStore, "_download_archive", fake_archive)

    def handler(entry_, dest, settings_):
        (dest / "model.onnx").write_bytes(b"onnx")
        (dest / "config.json").write_text("{}")

    register_prepare("test-prepare-after-download", handler)

    path = store.ensure(entry)
    assert (path / "model.onnx").is_file()
    assert store.is_downloaded(entry) is True


def test_prepare_uses_existing_inputs_without_download(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    store = ModelStore(settings)
    entry = ModelEntry(
        id="tifa-like-2",
        capability="v2m",
        kind="x",
        name="Bundle",
        dir_name="bundle2",
        required_files=("model.onnx",),
        prepare="test-prepare-existing",
        prepare_inputs=("source.pt",),
    )
    store.register([entry], "v2m")
    dest = store.path_for(entry)
    dest.mkdir(parents=True)
    (dest / "source.pt").write_bytes(b"pt")

    def boom(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("download should be skipped when prepare inputs exist")

    monkeypatch.setattr(ModelStore, "_download", boom)
    register_prepare(
        "test-prepare-existing",
        lambda entry_, dest_, settings_: (dest_ / "model.onnx").write_bytes(b"onnx"),
    )

    path = store.ensure(entry)
    assert (path / "model.onnx").is_file()


def test_unknown_prepare_step_raises(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    store = ModelStore(settings)
    entry = ModelEntry(
        id="bad-prepare",
        capability="v2m",
        kind="x",
        name="Bundle",
        dir_name="bundle3",
        required_files=("model.onnx",),
        prepare="no-such-step",
        prepare_inputs=("source.pt",),
    )
    store.register([entry], "v2m")
    dest = store.path_for(entry)
    dest.mkdir(parents=True)
    (dest / "source.pt").write_bytes(b"pt")
    with pytest.raises(Exception):
        store.ensure(entry)

