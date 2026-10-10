"""Bundled model catalog for the ASR capability.

Models are catalogued by *kind* (engine family) and *tasks*:

* ``qwen3``  -- Qwen3-ASR, supports ``transcribe`` **and** ``streaming``;
* ``pinyin`` -- Chinese toneless-pinyin singing ASR, ``transcribe`` only;
* ``romaji`` -- Japanese romaji singing ASR, ``transcribe`` only.

The catalog is intentionally data-driven: operators can add or override
entries by pointing ``MSST_ASR_CATALOG`` at a JSON file (same shape as
``{"models": [...]}``).  Qwen3-ASR bundles are fetched from the Hub on first
use; the Pinyin/Romaji ONNX bundles are typically placed in
``MSST_ASR_MODEL_DIR`` (or fetched from a mirror configured in a custom
catalog), because their official weights are distributed separately from the
training repositories.
"""

from __future__ import annotations

from ..config import Settings, get_settings
from ..model_store import ModelEntry, ModelStore

ASR_CAPABILITY = "asr"

#: Qwen3-ASR checkpoints are published under the ``Qwen`` org on both hubs.
_QWEN_SNAPSHOTS = (
    {"backend": "huggingface", "repo_id": "Qwen/Qwen3-ASR-1.7B"},
    {"backend": "modelscope", "repo_id": "Qwen/Qwen3-ASR-1.7B"},
)
_QWEN_SNAPSHOTS_SMALL = (
    {"backend": "huggingface", "repo_id": "Qwen/Qwen3-ASR-0.6B"},
    {"backend": "modelscope", "repo_id": "Qwen/Qwen3-ASR-0.6B"},
)

BUNDLED_MODELS: list[ModelEntry] = [
    ModelEntry(
        id="qwen3-asr-1.7b",
        capability=ASR_CAPABILITY,
        kind="qwen3",
        name="Qwen3-ASR-1.7B",
        description="Multilingual (52 languages/dialects) speech & singing ASR. "
        "Single-sentence and streaming.",
        tasks=("transcribe", "streaming"),
        languages=("zh", "yue", "ja", "en"),
        dir_name="qwen3-asr-1.7b",
        required_files=("config.json",),
        snapshots=_QWEN_SNAPSHOTS,
        extra={"repo": "https://github.com/QwenLM/Qwen3-ASR"},
    ),
    ModelEntry(
        id="qwen3-asr-0.6b",
        capability=ASR_CAPABILITY,
        kind="qwen3",
        name="Qwen3-ASR-0.6B",
        description="Efficient multilingual speech & singing ASR. "
        "Single-sentence and streaming.",
        tasks=("transcribe", "streaming"),
        languages=("zh", "yue", "ja", "en"),
        dir_name="qwen3-asr-0.6b",
        required_files=("config.json",),
        snapshots=_QWEN_SNAPSHOTS_SMALL,
        extra={"repo": "https://github.com/QwenLM/Qwen3-ASR"},
    ),
    ModelEntry(
        id="pinyin-asr",
        capability=ASR_CAPABILITY,
        kind="pinyin",
        name="PinyinASR",
        description="Chinese toneless-pinyin singing ASR (HuBERT CTC). "
        "Single-sentence only.",
        tasks=("transcribe",),
        languages=("zh",),
        dir_name="pinyinASR",
        required_files=(),
        extra={"repo": "https://github.com/Xiantaidu/PinyinASR"},
    ),
    ModelEntry(
        id="romaji-asr",
        capability=ASR_CAPABILITY,
        kind="romaji",
        name="RomajiASR",
        description="Japanese romaji phoneme singing ASR (HuBERT CTC). "
        "Single-sentence only.",
        tasks=("transcribe",),
        languages=("ja",),
        dir_name="romajiASR",
        required_files=(),
        extra={"repo": "https://github.com/Xiantaidu/RomajiASR"},
    ),
]


def build_store(settings: Settings | None = None) -> ModelStore:
    settings = settings or get_settings()
    store = ModelStore(settings)
    store.register(BUNDLED_MODELS, ASR_CAPABILITY)
    if settings.asr_catalog:
        store.load_json(settings.asr_catalog, ASR_CAPABILITY, override=True)
    return store
