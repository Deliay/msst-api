from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from msst_api.config import Settings
from msst_api.download import _candidates_for, _derive_modelscope_sources


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
