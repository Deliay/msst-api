"""Bundled model catalog for the v2m (Vocal2Midi) capability.

Each entry describes one ONNX bundle used by the pipeline:

* ``game``   -- GAME note extraction (encoder/segmenter/estimator/...);
* ``tifa``   -- TiFA forced aligner (five ONNX graphs + vocabulary);
* ``hfa``    -- HubertFA phoneme forced aligner;
* ``rmvpe``  -- RMVPE pitch estimator (optional; USTX/VSQX pitch curves).

Most weights are distributed as GitHub release assets rather than Hub
snapshots.  GAME, HubertFA and kashi-g2p are downloaded (and unpacked) from
their release archives on first use; RMVPE is not downloaded at all -- the
v2m pipeline reuses the RVC/Applio RMVPE predictor by default
(``MSST_V2M_RMVPE_BACKEND=rvc``).  TiFA's upstream release only ships the
PyTorch model, so the ONNX aligner graphs must be supplied out of band (place
them in ``MSST_V2M_MODEL_DIR/tifa-1.0-onnx`` or point ``MSST_V2M_CATALOG`` at an
ONNX bundle).

GitHub downloads go through ``MSST_GITHUB_PROXY`` (default ``https://ghfast.top/``).
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
        archives=(
            {
                "url": "https://github.com/openvpi/GAME/releases/download/v1.0.3/GAME-1.0.3-medium-onnx.zip",
                "size": 179775226,
                "sha256": "5b7a21e64c6310efac399f5d12838fffa70565be162436b5a4a65f290721e7d8",
            },
        ),
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
        extra={
            "repo": "https://github.com/openvpi/TIFA",
            "pytorch_url": "https://github.com/openvpi/TIFA/releases/download/v1.0.0/TIFA-1.0-ST.zip",
            "note": "The openvpi release ships the PyTorch model; this runtime needs "
            "the exported ONNX graphs (config.json + 5 .onnx files). Place them here "
            "or point a custom catalog at an ONNX bundle.",
        },
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
        archives=(
            {
                "url": "https://github.com/wolfgitpr/HubertFA/releases/download/v0.0.7/1218_hfa_model_new_dict.zip",
                "size": 256589553,
                "sha256": "48bd6dbcc293e47cc6cbcc556baf575e3d91d61d0ac88ddeccd6098f8f346fa2",
            },
        ),
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
        archives=(
            {
                "url": "https://github.com/Xiantaidu/kashi-g2p/releases/download/v1.0.0/models_onnx.zip",
                "size": 109095482,
                "sha256": "47b46e8028cfae5b908c08845796f661c8b8f8c7e5a8bb65a64d4bbc61dd35b3",
            },
        ),
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
