"""ONNX session management and model configuration for the TiFA aligner."""
from __future__ import annotations

import json
import logging
import pathlib

import onnxruntime as ort

from msst_api.v2m._vendor.inference.TiFA.lib.vocabulary import Vocabulary
from msst_api.v2m._vendor.inference.device_utils import resolve_onnx_providers

logger = logging.getLogger(__name__)

GRAPH_NAMES = ("spectrogram", "prepare", "score", "select", "model")


class TifaModel:
    """The five TiFA ONNX graphs plus timing config and phone vocabulary."""

    def __init__(self, model_dir: str | pathlib.Path, device: str | None = None):
        self.model_dir = pathlib.Path(model_dir)
        self.config: dict = json.loads(
            (self.model_dir / "config.json").read_text(encoding="utf-8")
        )
        self.vocabulary = Vocabulary.from_file(self.model_dir / "vocabulary.json")

        provider_name, providers = resolve_onnx_providers(device, label="TiFA ONNX")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.sessions: dict[str, ort.InferenceSession] = {
            name: ort.InferenceSession(
                str(self.model_dir / f"{name}.onnx"), options, providers=providers
            )
            for name in GRAPH_NAMES
        }
        self.provider_name = provider_name

    @property
    def samplerate(self) -> int:
        return int(self.config["samplerate"])

    @property
    def timestep(self) -> float:
        return float(self.config["timestep"])

    @property
    def hop_size(self) -> int:
        return int(self.config["hop_size"])

    @property
    def fft_size(self) -> int:
        return int(self.config["fft_size"])

    @property
    def win_size(self) -> int:
        return int(self.config["win_size"])

    def run(self, name: str, feed: dict) -> dict:
        session = self.sessions[name]
        names = [output.name for output in session.get_outputs()]
        return dict(zip(names, session.run(None, feed)))

    def shutdown(self) -> None:
        """Drop the ONNX sessions so the CUDA arena is released on eviction."""

        self.sessions.clear()

    release = shutdown


_TIFA_MODEL_CACHE: dict[tuple[str, str], TifaModel] = {}


def load_tifa_model(model_dir: str | pathlib.Path, device: str | None = None, use_cache: bool = True) -> TifaModel:
    """Load the TiFA ONNX bundle via the shared model cache.

    Sessions are cached in the project-wide SharedModelCache and reused across
    pipeline runs, honouring ``MSST_MAX_LOADED_MODELS`` / ``MSST_MAX_CONCURRENCY``.
    """
    del use_cache

    resolved = str(pathlib.Path(model_dir))
    key = ("tifa", resolved, str(device or "").strip().lower() or "auto")

    from msst_api.v2m import runtime

    return runtime.get_or_load(
        key,
        namespace="v2m",
        label=f"tifa:{pathlib.Path(model_dir).name}",
        builder=lambda: TifaModel(model_dir, device=device),
        closer=_shutdown_model,
    )


def _shutdown_model(model) -> None:  # pragma: no cover - sessions drop with the object
    shutdown = getattr(model, "shutdown", None)
    if callable(shutdown):
        shutdown()

