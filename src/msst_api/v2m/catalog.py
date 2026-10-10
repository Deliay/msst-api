"""Bundled model catalog for the v2m (Vocal2Midi) capability.

Each entry describes one ONNX bundle used by the pipeline:

* ``game``   -- GAME note extraction (encoder/segmenter/estimator/...);
* ``tifa``   -- TiFA forced aligner (five ONNX graphs + vocabulary);
* ``hfa``    -- HubertFA phoneme forced aligner;
* ``rmvpe``  -- RMVPE pitch estimator (optional; USTX/VSQX pitch curves).

Most upstream weights are distributed as release assets rather than Hub
snapshots, so ``snapshots`` is often empty: operators either drop the files in
``MSST_V2M_MODEL_DIR`` or point ``MSST_V2M_CATALOG`` at a JSON catalog with the
mirror of their choice.
"""

from __future__ import annotations

from ..config import Settings, get_settings
from ..model_store import ModelEntry, ModelStore

V2M_CAPABILITY = "v2m"

BUNDLED_MODELS: list[ModelEntry] = [
    ModelEntry(
        id="game",
        capability=V2M_CAPABILITY,
        kind="game",
        name="GAME-1.0.3-medium-onnx",
        description="GAME singing-note extractor (ONNX: encoder/segmenter/estimator).",
        tasks=("notes",),
        languages=("zh", "ja", "yue", "en"),
        dir_name="GAME-1.0.3-medium-onnx",
        required_files=(
            "config.json",
            "encoder.onnx",
            "segmenter.onnx",
            "estimator.onnx",
            "dur2bd.onnx",
            "bd2dur.onnx",
        ),
        snapshots=(),
        extra={"repo": "https://github.com/openvpi/GAME"},
    ),
    ModelEntry(
        id="tifa",
        capability=V2M_CAPABILITY,
        kind="tifa",
        name="tifa-1.0-onnx",
        description="TiFA token-imputing forced aligner (ONNX, 5 graphs).",
        tasks=("align",),
        languages=("zh", "ja", "yue", "en"),
        dir_name="tifa-1.0-onnx",
        required_files=(
            "config.json",
            "vocabulary.json",
            "spectrogram.onnx",
            "prepare.onnx",
            "score.onnx",
            "select.onnx",
            "model.onnx",
        ),
        snapshots=(),
        extra={"repo": "https://github.com/openvpi/TIFA"},
    ),
    ModelEntry(
        id="hubertfa",
        capability=V2M_CAPABILITY,
        kind="hfa",
        name="HubertFA",
        description="HuBERT phoneme forced aligner (ONNX).",
        tasks=("align",),
        languages=("zh", "ja", "en"),
        dir_name="1218_hfa_model_new_dict",
        required_files=("model.onnx", "vocab.json", "config.json", "VERSION"),
        snapshots=(),
        extra={"repo": "https://github.com/wolfgitpr/HubertFA"},
    ),
    ModelEntry(
        id="rmvpe",
        capability=V2M_CAPABILITY,
        kind="rmvpe",
        name="RMVPE",
        description="RMVPE vocal pitch estimator (ONNX), for USTX/VSQX curves.",
        tasks=("pitch",),
        languages=(),
        dir_name="RMVPE",
        required_files=("rmvpe.onnx",),
        snapshots=(),
        extra={"repo": "https://github.com/Dream-High/RMVPE"},
    ),
    ModelEntry(
        id="kashi-g2p-onnx",
        capability=V2M_CAPABILITY,
        kind="kashi_g2p",
        name="kashi-g2p-onnx",
        description="Japanese G2P (reading) ONNX runtime + lexicon for TiFA/lyrics.",
        tasks=("g2p",),
        languages=("ja",),
        dir_name="kashi-g2p-onnx",
        required_files=("model.onnx",),
        snapshots=(),
        extra={"repo": "https://github.com/Xiantaidu/kashi-g2p"},
    ),
]


def build_store(settings: Settings | None = None) -> ModelStore:
    settings = settings or get_settings()
    store = ModelStore(settings)
    store.register(BUNDLED_MODELS, V2M_CAPABILITY)
    if settings.v2m_catalog:
        store.load_json(settings.v2m_catalog, V2M_CAPABILITY, override=True)
    return store
