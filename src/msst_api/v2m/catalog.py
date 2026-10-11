"""Bundled model catalog for the v2m (Vocal2Midi) capability.

Each entry describes one ONNX bundle used by the pipeline:

* ``game``   -- GAME note extraction (encoder/segmenter/estimator/...);
* ``tifa``   -- TiFA forced aligner (five ONNX graphs + vocabulary);
* ``hfa``    -- HubertFA phoneme forced aligner;
* ``rmvpe``  -- RMVPE pitch estimator (optional; USTX/VSQX pitch curves).

Most weights are distributed as GitHub release assets rather than Hub
snapshots.  All of them download on first use: GAME, HubertFA and kashi-g2p are
unpacked from their release archives, and the TiFA ONNX graphs are exported from
openvpi's PyTorch release via the vendored TIFA ``deploy`` code (see
:mod:`msst_api.v2m.tifa_export`).  RMVPE is not downloaded at all -- the v2m
pipeline reuses the RVC/Applio RMVPE predictor by default
(``MSST_V2M_RMVPE_BACKEND=rvc``).

GitHub downloads go through ``MSST_GITHUB_PROXY`` (default ``https://ghfast.top/``).
"""

from __future__ import annotations

from pathlib import Path

from ..config import Settings, get_settings
from ..model_store import ModelEntry, ModelStore, register_prepare

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
            # The vendored TiFA G2P pipeline is rooted at the model directory and
            # reads these alongside the graphs (see g2p_pipeline.py).
            "dictionaries/ds-zh-pinyin-lite.txt",
            "dictionaries/japanese_dict_full.txt",
            "dictionaries/ds_cmudict-07b.txt",
            "assets/LstmG2p-Eng/encoder.onnx",
            "assets/LstmG2p-Eng/decoder.onnx",
            "assets/LstmG2p-Eng/char.json",
            "assets/LstmG2p-Eng/phonemes.json",
        ),
        snapshots=(),
        archives=(
            {
                "url": "https://github.com/openvpi/TIFA/releases/download/v1.0.0/TIFA-1.0-ST.zip",
                "size": 150789526,
                "sha256": "6da6832cd2cb981aae1ccc2d9330f8f7c085ce7977d1eb13bff63531d038a88b",
            },
        ),
        prepare="tifa_export",
        prepare_inputs=("model.pt", "config.yaml", "vocabulary.json"),
        extra={
            "repo": "https://github.com/openvpi/TIFA",
            "note": "The openvpi release ships the PyTorch model; the ONNX graphs "
            "needed by this runtime are exported on first use via the vendored "
            "TIFA deploy code (see MSST_V2M_TIFA_OPSET).",
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


def _prepare_tifa_export(entry: ModelEntry, dest: Path, settings: Settings) -> None:
    """Convert a downloaded TiFA PyTorch release into the ONNX bundle in place."""

    from .tifa_export import export_tifa_onnx

    export_tifa_onnx(
        dest,
        opset_version=settings.v2m_tifa_opset,
        keep_source=settings.v2m_tifa_keep_source,
    )


register_prepare("tifa_export", _prepare_tifa_export)


def build_store(settings: Settings | None = None) -> ModelStore:
    settings = settings or get_settings()
    store = ModelStore(settings)
    store.register(BUNDLED_MODELS, V2M_CAPABILITY)
    if settings.v2m_catalog:
        store.load_json(settings.v2m_catalog, V2M_CAPABILITY, override=True)
    return store
