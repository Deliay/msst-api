"""Load and query the bundled MSST model catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .config import BUNDLED_REGISTRY, get_settings


class ModelNotFoundError(KeyError):
    """Raised when a requested model id is not part of the catalog."""


@dataclass(frozen=True)
class ModelEntry:
    """A single catalog model with resolved filesystem paths."""

    id: str
    name: str
    category: str
    category_label: str
    model_type: str
    instruments: tuple[str, ...]
    target_instrument: str | None
    config_rel: str
    config_path: Path
    size: int | None
    sha256: str | None
    sources: tuple[dict, ...] = field(default_factory=tuple)

    @property
    def weight_filename(self) -> str:
        return self.name

    @property
    def default_weight_path(self) -> Path:
        return get_settings().model_dir / self.category / self.name

    def to_public_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "category_label": self.category_label,
            "model_type": self.model_type,
            "instruments": list(self.instruments),
            "target_instrument": self.target_instrument,
            "size": self.size,
            "sha256": self.sha256,
        }


class ModelRegistry:
    """In-memory view of ``data/registry.json``."""

    def __init__(self, registry_path: Path, config_dir: Path) -> None:
        with registry_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.version: int = payload.get("version", 1)
        self.categories: dict = payload.get("categories", {})
        self._config_dir = config_dir
        self._models: dict[str, ModelEntry] = {}
        for model_id, raw in payload.get("models", {}).items():
            self._models[model_id] = ModelEntry(
                id=model_id,
                name=raw.get("name", model_id),
                category=raw["category"],
                category_label=raw.get("category_label", raw["category"]),
                model_type=raw["model_type"],
                instruments=tuple(raw.get("instruments") or ()),
                target_instrument=raw.get("target_instrument"),
                config_rel=raw["config"],
                config_path=config_dir / raw["config"].split("/", 1)[-1],
                size=raw.get("size"),
                sha256=raw.get("sha256"),
                sources=tuple(raw.get("sources") or ()),
            )

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._models

    def ids(self) -> list[str]:
        return sorted(self._models)

    def all(self) -> list[ModelEntry]:
        return [self._models[model_id] for model_id in self.ids()]

    def get(self, model_id: str) -> ModelEntry:
        try:
            return self._models[model_id]
        except KeyError as error:
            raise ModelNotFoundError(model_id) from error

    def resolve_config(self, entry: ModelEntry) -> Path:
        """Return the local path of a model's YAML config.

        The catalog configs are bundled with the package, but operators can
        override them by pointing ``MSST_CONFIG_DIR`` at a directory that
        mirrors the same ``<category>/<name>.yaml`` layout.
        """

        candidate = self._config_dir / entry.category / f"{entry.name}.yaml"
        if candidate.is_file():
            return candidate
        if entry.config_path.is_file():
            return entry.config_path
        raise FileNotFoundError(
            f"Config for model {entry.id!r} not found at {candidate}"
        )


@lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    settings = get_settings()
    return ModelRegistry(BUNDLED_REGISTRY, settings.config_dir)
