"""High-level DiffSinger service layer (mirrors :mod:`msst_api.rvc_engine`).

:class:`SVSVoiceLibrary` discovers voicebank folders under ``MSST_SVS_MODEL_DIR``;
:class:`SVSManager` keeps loaded voicebanks (ONNX sessions) resident with the
same global LRU + concurrency budget as MSST/RVC/ASR/v2m.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from . import SVSError, SVSVoiceNotFoundError
from ..config import Settings, get_settings
from .pipeline import SVSRequest, SVSResult, run_svs_pipeline
from ..resources import SharedModelCache
from .voicebank import DiffSingerVoicebank

logger = logging.getLogger("msst_api.svs")


@dataclass(frozen=True)
class SVSVoice:
    id: str
    name: str
    root: Path

    def to_public_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "path": str(self.root),
            "downloaded": (self.root / "dsconfig.yaml").is_file(),
        }


class SVSVoiceLibrary:
    """Scans a directory for OpenUtau DiffSinger voicebanks."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)

    def _scan(self) -> list[SVSVoice]:
        if not self.model_dir.is_dir():
            return []
        voices: list[SVSVoice] = []
        for entry in sorted(self.model_dir.iterdir()):
            if not entry.is_dir():
                continue
            if not (entry / "dsconfig.yaml").is_file():
                continue
            voices.append(SVSVoice(id=entry.name, name=entry.name, root=entry))
        return voices

    def all(self) -> list[SVSVoice]:
        return self._scan()

    def get(self, voice: str) -> SVSVoice:
        value = (voice or "").strip().strip('"').strip("'")
        if not value:
            raise SVSVoiceNotFoundError("Voicebank id is required")
        voices = self._scan()
        if not voices:
            raise SVSVoiceNotFoundError(
                f"No DiffSinger voicebanks installed in {self.model_dir}"
            )
        for candidate in voices:
            if candidate.id == value or candidate.name == value:
                return candidate
        lowered = value.lower()
        matches = [v for v in voices if lowered in v.id.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            names = ", ".join(v.id for v in matches[:10])
            raise SVSVoiceNotFoundError(
                f"Ambiguous voicebank {value!r}. Matches: {names}"
            )
        available = ", ".join(v.id for v in voices[:20]) or "(none)"
        raise SVSVoiceNotFoundError(
            f"Voicebank {value!r} not found. Available: {available}"
        )


@dataclass(frozen=True)
class SVSSpec:
    root: Path
    device: str

    @property
    def key(self) -> tuple:
        return (str(self.root), self.device)


class SVSManager:
    def __init__(
        self, settings: Settings | None = None, cache: SharedModelCache | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.library = SVSVoiceLibrary(self.settings.svs_model_dir)
        self._cache = cache or SharedModelCache(
            self.settings.max_loaded_models, self.settings.max_concurrency
        )
        self._lock = threading.RLock()

    def get(self, spec: SVSSpec) -> DiffSingerVoicebank:
        cached = self._cache.get(spec.key)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get(spec.key)
            if cached is not None:
                return cached
            logger.info("Loading DiffSinger voicebank %s device=%s", spec.root, spec.device)
            try:
                voicebank = DiffSingerVoicebank(spec.root, spec.device)
            except SVSVoiceNotFoundError:
                raise
            except Exception as error:  # noqa: BLE001
                raise SVSError(
                    f"Could not load DiffSinger voicebank {spec.root.name}: {error}"
                ) from error
            return self._cache.put(
                spec.key,
                voicebank,
                namespace="svs",
                label=spec.root.name,
                closer=lambda vb: vb.close(),
            )

    def render(self, spec: SVSSpec, ustx_path: str, request: SVSRequest) -> SVSResult:
        voicebank = self.get(spec)
        with self._cache.inference_slot():
            return run_svs_pipeline(voicebank, ustx_path, request)

    def unload(self) -> None:
        self._cache.clear("svs")

    @property
    def loaded(self) -> list[str]:
        return self._cache.loaded("svs")
