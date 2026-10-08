#!/usr/bin/env python3
"""Generate ``src/msst_api/data/registry.json`` from the MSST-WebUI catalog.

The runtime model catalog is derived from MSST-WebUI's ``models_info.json``
(https://github.com/SUC-DriverOld/MSST-WebUI).  It lists the MSST pretrained
models that MSST-WebUI ships with, together with their architecture
(``model_type``), container category, weight file name, size, sha256 and the
Hugging Face download location.

The generator only keeps ``MSST`` checkpoints (it drops the UVR ``VR_Models``,
which are not supported by Music-Source-Separation-Training) and resolves the
per-checkpoint YAML config bundled under
``src/msst_api/data/configs/<category>/<model>.yaml``.

Usage::

    python scripts/build_registry.py

Run it again only when the upstream catalog or bundled configs change.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_INFO = REPO_ROOT / "scripts" / "models_info.json"
CONFIG_ROOT = REPO_ROOT / "src" / "msst_api" / "data" / "configs"
REGISTRY_OUT = REPO_ROOT / "src" / "msst_api" / "data" / "registry.json"

# Hugging Face model repository that hosts every official MSST-WebUI weight.
HF_REPO_ID = "Sucial/MSST-WebUI"
HF_RESOLVE_PREFIX = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/"

# Human friendly labels for the MSST-WebUI categories.
CATEGORY_LABELS = {
    "vocal_models": "Vocal models (vocals / instrumental)",
    "single_stem_models": "Single stem models",
    "multi_stem_models": "Multi stem models",
}


def _load_config(config_path: Path) -> dict:
    """Load a model YAML keeping ``!!python/tuple`` values intact."""

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def _extract_stems(config: dict) -> tuple[list[str], str | None]:
    training = config.get("training", {}) if isinstance(config, dict) else {}
    instruments = training.get("instruments") or []
    if not isinstance(instruments, list):
        instruments = [instruments]
    target = training.get("target_instrument")
    return [str(item) for item in instruments], (str(target) if target else None)


def _hf_source(link: str) -> dict | None:
    if link.startswith(HF_RESOLVE_PREFIX):
        return {
            "backend": "huggingface",
            "repo_id": HF_REPO_ID,
            "file": link[len(HF_RESOLVE_PREFIX):],
            "url": link,
        }
    if link.startswith("https://huggingface.co/"):
        return {"backend": "huggingface", "url": link}
    return None


def build() -> dict:
    with MODELS_INFO.open("r", encoding="utf-8") as handle:
        models_info = json.load(handle)

    entries: dict[str, dict] = {}
    for model_name, info in sorted(models_info.items()):
        category = info.get("model_class")
        if category == "VR_Models":
            # MSST inference does not support the UVR/VR architecture.
            continue

        model_type = info.get("model_type")
        if not model_type:
            continue

        config_rel = Path("configs") / category / f"{model_name}.yaml"
        config_path = CONFIG_ROOT / category / f"{model_name}.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Missing bundled config for {model_name!r}: {config_path}"
            )

        instruments, target = _extract_stems(_load_config(config_path))

        link = info.get("link", "")
        sources = []
        hf = _hf_source(link)
        if hf:
            sources.append(hf)

        entry = {
            "id": model_name,
            "name": model_name,
            "category": category,
            "category_label": CATEGORY_LABELS.get(category, category),
            "model_type": model_type,
            "instruments": instruments,
            "target_instrument": target,
            "config": config_rel.as_posix(),
            "size": info.get("model_size"),
            "sha256": info.get("sha256"),
            "sources": sources,
        }
        entries[model_name] = entry

    registry = {
        "version": 1,
        "generated_from": "SUC-DriverOld/MSST-WebUI data_backup/models_info.json",
        "categories": CATEGORY_LABELS,
        "models": entries,
    }
    return registry


def main() -> None:
    registry = build()
    REGISTRY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_OUT.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote {len(registry['models'])} models to {REGISTRY_OUT}")


if __name__ == "__main__":
    main()
