"""High-level v2m pipeline orchestration.

Mirrors Vocal2Midi's ``auto_lyric_hybrid_pipeline`` but resolves every model
through the service catalog and runs under the shared model / concurrency
budget.  Outputs are collected in-memory so the HTTP layer can return a zip.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..asr.manager import ASRError, ASRManager
from ..config import Settings
from .manager import V2MManager, V2MError

logger = logging.getLogger("msst_api.v2m.pipeline")

#: Output artifacts the pipeline can produce.
SUPPORTED_OUTPUT_FORMATS = {
    "mid",
    "txt",
    "csv",
    "ustx",
    "vsqx",
    "textgrid",
    "chunks",
    "asr_match_log",
}


@dataclass
class V2MRequest:
    """Validated inputs for one pipeline run."""

    language: str = "zh"
    lyric_output_mode: str = "auto"
    original_lyrics: str = ""
    output_formats: list[str] = field(default_factory=lambda: ["mid"])
    slicing_method: str = "auto"
    slice_min_sec: float = 5.0
    slice_max_sec: float = 10.0
    tempo: float = 120.0
    quantization_step: int = 0
    quantization_mode: str = "smart"
    quant_simplicity: float = 0.0
    pitch_format: str = "name"
    round_pitch: bool = True
    seg_threshold: float = 0.2
    seg_radius: float = 0.02
    est_threshold: float = 0.2
    batch_size: int = 1
    asr_batch_size: int = 2
    output_lyrics: bool = True
    output_pitch_curve: bool = False
    # engine selection
    chinese_asr_engine: str = "qwen"
    japanese_asr_engine: str = "romaji"
    alignment_engine: str = "tifa"
    japanese_g2p_engine: str = "kashi-g2p-onnx"
    # optional explicit model ids (defaults come from the catalogs)
    asr_model: str | None = None
    pinyin_asr_model: str = "pinyin-asr"
    romaji_asr_model: str = "romaji-asr"
    game_model: str = "game"
    tifa_model: str = "tifa"
    hfa_model: str = "hubertfa"
    rmvpe_model: str = "rmvpe"
    kashi_g2p_model: str = "kashi-g2p-onnx"
    device: str | None = None
    source: str | None = None


def _uses_pinyin(req: V2MRequest) -> bool:
    return req.language.strip().lower() in {"zh", "zh-pinyin"} and (
        req.chinese_asr_engine.strip().lower() == "pinyin"
    )


def _uses_romaji(req: V2MRequest) -> bool:
    return req.language.strip().lower() == "ja" and (
        req.japanese_asr_engine.strip().lower() != "qwen"
    )


def _resolve_asr_paths(
    asr_manager: ASRManager, req: V2MRequest
) -> tuple[str, str, str]:
    """Return (qwen_path, pinyin_path, romaji_path) as strings ('' when unused)."""

    qwen_path = ""
    pinyin_path = ""
    romaji_path = ""

    if req.output_lyrics:
        if _uses_pinyin(req):
            pinyin_path = str(_ensure_asr(asr_manager, req.pinyin_asr_model, req.source))
        else:
            qwen_path = str(_ensure_asr(asr_manager, req.asr_model, req.source))
        if _uses_romaji(req):
            romaji_path = str(_ensure_asr(asr_manager, req.romaji_asr_model, req.source))
    return qwen_path, pinyin_path, romaji_path


def _ensure_asr(asr_manager: ASRManager, model_id: str | None, source: str | None) -> Path:
    entry, path = asr_manager.ensure_path(model_id, source=source)
    return path


def _resolve_aligner(v2m: V2MManager, req: V2MRequest, language: str) -> tuple[str, str]:
    """Return (tifa_path, hfa_path); the unused one is ''."""

    use_tifa = req.alignment_engine.strip().lower() == "tifa" or language == "yue"
    if use_tifa:
        return str(v2m.resolve_path(req.tifa_model, req.source)), ""
    return "", str(v2m.resolve_path(req.hfa_model, req.source))


def run_v2m_pipeline(
    *,
    settings: Settings,
    asr_manager: ASRManager,
    v2m: V2MManager,
    audio_path: Path,
    output_filename: str,
    output_dir: Path,
    request: V2MRequest,
) -> dict[str, bytes]:
    """Run the full pipeline and return ``{relative_name: bytes}``."""

    from ._vendor.inference.pipeline.auto_lyric_hybrid import auto_lyric_hybrid_pipeline

    language = (request.language or "zh").strip().lower() or "zh"
    device = request.device or settings.device

    game_dir = str(v2m.resolve_path(request.game_model, request.source))
    if request.output_lyrics:
        tifa_path, hfa_path = _resolve_aligner(v2m, request, language)
    else:
        tifa_path, hfa_path = "", ""

    rmvpe_path = ""
    wants_pitch_curve = request.output_pitch_curve and any(
        fmt in {"ustx", "vsqx"} for fmt in request.output_formats
    )
    if wants_pitch_curve:
        rmvpe_path = str(v2m.resolve_path(request.rmvpe_model, request.source))

    kashi_g2p_path = ""
    if language == "ja" and (
        request.japanese_g2p_engine.strip().lower() == "kashi-g2p-onnx"
    ):
        kashi_g2p_path = str(_resolve_optional(v2m, request.kashi_g2p_model))

    qwen_path, pinyin_path, romaji_path = _resolve_asr_paths(asr_manager, request)

    output_dir.mkdir(parents=True, exist_ok=True)

    auto_lyric_hybrid_pipeline(
        audio_path=str(audio_path),
        output_filename=output_filename,
        game_model_dir=game_dir,
        device=device,
        hfa_model_dir=hfa_path,
        asr_model_path=qwen_path,
        ts=[2.0, 4.0, 6.0],
        language=language,
        lyric_output_mode=request.lyric_output_mode,
        original_lyrics=request.original_lyrics,
        output_dir=output_dir,
        output_formats=request.output_formats,
        slicing_method=request.slicing_method,
        tempo=request.tempo,
        quantization_step=request.quantization_step,
        pitch_format=request.pitch_format,
        round_pitch=request.round_pitch,
        quantization_mode=request.quantization_mode,
        seg_threshold=request.seg_threshold,
        seg_radius=request.seg_radius,
        est_threshold=request.est_threshold,
        batch_size=request.batch_size,
        asr_batch_size=request.asr_batch_size,
        quant_simplicity=request.quant_simplicity,
        slice_min_sec=request.slice_min_sec,
        slice_max_sec=request.slice_max_sec,
        output_lyrics=request.output_lyrics,
        output_pitch_curve=request.output_pitch_curve,
        rmvpe_model_path=rmvpe_path,
        phoneme_asr_model_path=romaji_path,
        pinyin_asr_model_path=pinyin_path,
        chinese_asr_engine=request.chinese_asr_engine,
        japanese_asr_engine=request.japanese_asr_engine,
        alignment_engine=request.alignment_engine,
        tifa_model_path=tifa_path,
        japanese_g2p_engine=request.japanese_g2p_engine,
        kashi_g2p_model_path=kashi_g2p_path,
    )

    return _collect_outputs(output_dir)


def _resolve_optional(v2m: V2MManager, model_id: str) -> Path | None:
    path = v2m.path_or_none(model_id)
    return path


def _collect_outputs(output_dir: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(output_dir))] = path.read_bytes()
    return files


def cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


__all__ = ["V2MRequest", "SUPPORTED_OUTPUT_FORMATS", "run_v2m_pipeline", "cleanup", "V2MError", "ASRError"]
