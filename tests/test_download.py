from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from msst_api.config import Settings
from msst_api.download import (
    _apply_github_proxy,
    _candidates_for,
    _derive_modelscope_sources,
    download_snapshot,
)


def _settings(**kwargs) -> Settings:
    base = Settings(
        model_dir=Path("/tmp/does-not-matter"),
        config_dir=Path("/tmp/does-not-matter"),
        download_backend="modelscope",
    )
    return replace(base, **kwargs)


HF_SOURCE = {
    "backend": "huggingface",
    "repo_id": "Sucial/MSST-WebUI",
    "file": "All_Models/vocal_models/model.ckpt",
    "url": "https://huggingface.co/Sucial/MSST-WebUI/resolve/main/All_Models/vocal_models/model.ckpt",
}


def test_modelscope_derived_from_mirror() -> None:
    settings = _settings(modelscope_mirror="my-org/MSST-WebUI")
    candidates = _derive_modelscope_sources([HF_SOURCE], settings)
    assert candidates == [
        {
            "backend": "modelscope",
            "repo_id": "my-org/MSST-WebUI",
            "file": "All_Models/vocal_models/model.ckpt",
        }
    ]


def test_no_modelscope_candidate_without_mirror() -> None:
    settings = _settings(modelscope_mirror=None)
    assert _derive_modelscope_sources([HF_SOURCE], settings) == []
    # huggingface candidates are always available
    assert _candidates_for("huggingface", [HF_SOURCE], settings) == [HF_SOURCE]


def test_backend_order_defaults_to_modelscope_first() -> None:
    assert _settings(download_backend="modelscope").backend_order() == [
        "modelscope",
        "huggingface",
    ]
    assert _settings(download_backend="huggingface").backend_order() == [
        "huggingface",
        "modelscope",
    ]


def test_explicit_modelscope_source_is_kept() -> None:
    settings = _settings(modelscope_mirror=None)
    ms_source = {"backend": "modelscope", "repo_id": "org/repo", "file": "a.ckpt"}
    assert _derive_modelscope_sources([HF_SOURCE, ms_source], settings) == [ms_source]


def test_github_proxy_prefixes_github_urls() -> None:
    settings = _settings(github_proxy="https://ghfast.top/")
    url = "https://github.com/openvpi/GAME/releases/download/v1.0.3/GAME-1.0.3-medium-onnx.zip"
    assert _apply_github_proxy(url, settings) == f"https://ghfast.top/{url}"
    # Non-GitHub URLs are untouched.
    assert _apply_github_proxy("https://example.com/a.zip", settings) == "https://example.com/a.zip"


def test_github_proxy_disabled_when_empty() -> None:
    settings = _settings(github_proxy="")
    url = "https://github.com/openvpi/GAME/releases/download/v1.0.3/x.zip"
    assert _apply_github_proxy(url, settings) == url


def test_download_snapshot_populates_dest(tmp_path, monkeypatch) -> None:
    import huggingface_hub

    settings = Settings(model_dir=tmp_path / "models")

    def fake_snapshot_download(repo_id, *, repo_type, local_dir, **kwargs):
        dest = Path(local_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text("{}")
        (dest / "model.onnx").write_bytes(b"onnx")
        return local_dir

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    dest = tmp_path / "models" / "asr" / "bundle"
    path = download_snapshot("org/repo", dest, backend="huggingface", settings=settings)
    assert (path / "config.json").is_file()
    assert (path / "model.onnx").is_file()
