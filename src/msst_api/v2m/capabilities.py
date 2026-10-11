"""Individual v2m capability helpers exposed as their own API endpoints.

Each function is a thin, in-memory wrapper over the vendored runtimes so the
service can offer per-capability APIs (slicing, pitch, notes, alignment, G2P)
in addition to the end-to-end ``/api/v2m/inference`` pipeline.  Every model
inference runs under the shared concurrency semaphore.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from .manager import V2MManager, V2MError
from . import runtime

logger = logging.getLogger("msst_api.v2m.capabilities")

RMVPE_SAMPLE_RATE = 16000


def slice_audio(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    method: str = "auto",
    min_len_sec: float | None = None,
    max_len_sec: float | None = None,
) -> list[dict]:
    from ._vendor.inference.API.slicer_api import (
        normalize_slicing_method,
        slice_audio_with_custom_bounds,
    )

    method = normalize_slicing_method(method)
    chunks = slice_audio_with_custom_bounds(
        waveform,
        sample_rate,
        method,
        min_len_sec=min_len_sec,
        max_len_sec=max_len_sec,
    )
    return [
        {"offset": float(chunk["offset"]), "num_samples": int(np.asarray(chunk["waveform"]).shape[-1])}
        for chunk in chunks
    ]


def extract_pitch(
    v2m: V2MManager,
    waveform: np.ndarray,
    sample_rate: int,
    *,
    device: str | None = None,
    source: str | None = None,
    model_id: str = "rmvpe",
) -> dict:
    device = device or v2m.settings.device
    model_path = ""
    if v2m.settings.v2m_rmvpe_backend == "onnx":
        model_path = str(v2m.resolve_path(model_id, source))

    transcriber = runtime.build_rmvpe_transcriber(model_path, device)
    with runtime.inference_slot():
        result = transcriber.infer(np.asarray(waveform, dtype=np.float32), sample_rate)
    return {
        "time_step_seconds": float(result.time_step_seconds),
        "midi_pitch": [None if not np.isfinite(v) else float(v) for v in np.asarray(result.midi_pitch).tolist()],
        "voiced": (
            [bool(v) for v in np.asarray(result.voiced_mask).tolist()]
            if result.voiced_mask is not None
            else None
        ),
    }


def extract_notes(
    v2m: V2MManager,
    waveform: np.ndarray,
    sample_rate: int,
    *,
    device: str | None = None,
    source: str | None = None,
    model_id: str = "game",
    seg_threshold: float = 0.2,
    seg_radius: float = 0.02,
    est_threshold: float = 0.2,
) -> list[dict]:
    from ._vendor.inference.API.game_api import extract_pitches_only, load_game_model

    model_dir = str(v2m.resolve_path(model_id, source))
    device = device or v2m.settings.device
    game_model = load_game_model(model_dir, device=device)
    chunks = [
        {
            "offset": 0.0,
            "waveform": np.asarray(waveform, dtype=np.float32).reshape(-1),
        }
    ]
    with runtime.inference_slot():
        notes = extract_pitches_only(
            chunks,
            sample_rate,
            game_model,
            ts=[2.0, 4.0, 6.0],
            seg_threshold=seg_threshold,
            seg_radius=seg_radius,
            est_threshold=est_threshold,
        )
    return [
        {"onset": float(n.onset), "offset": float(n.offset), "pitch": float(n.pitch), "lyric": n.lyric}
        for n in notes
    ]


def align_text(
    v2m: V2MManager,
    waveform: np.ndarray,
    sample_rate: int,
    text: str,
    *,
    language: str = "zh",
    device: str | None = None,
    source: str | None = None,
    model_id: str = "tifa",
) -> dict:
    from ._vendor.inference.TiFA.aligner import run_tifa_fa
    from ._vendor.inference.TiFA.runtime import load_tifa_model

    model_dir = str(v2m.resolve_path(model_id, source))
    device = device or v2m.settings.device
    tifa_model = load_tifa_model(model_dir, device=device)

    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        import soundfile as sf

        sf.write(str(temp_path / "chunk_0.wav"), np.asarray(waveform, dtype=np.float32).reshape(-1), sample_rate)
        (temp_path / "chunk_0.txt").write_text(text, encoding="utf-8")
        with runtime.inference_slot():
            pred_dict, display = run_tifa_fa(
                tifa_model, temp_path, language=language
            )

    words = []
    for _stem, pred in pred_dict.items():
        word_list = pred[2]
        for word in word_list:
            words.append(
                {
                    "start": float(word.start),
                    "end": float(word.end),
                    "text": word.text,
                    "phonemes": [
                        {"start": float(p.start), "end": float(p.end), "text": p.text}
                        for p in word.phonemes
                    ],
                }
            )
    return {"words": words, "display": {k: list(v) for k, v in (display or {}).items()}}


def g2p_reading(
    v2m: V2MManager,
    text: str,
    *,
    language: str = "ja",
    engine: str = "kashi-g2p-onnx",
    source: str | None = None,
    model_id: str = "kashi-g2p-onnx",
) -> dict:
    language = (language or "ja").strip().lower()
    if language == "ja" and engine.strip().lower() == "kashi-g2p-onnx":
        model_path = v2m.path_or_none(model_id)
        if model_path is None:
            raise V2MError(
                f"Japanese G2P model {model_id!r} is not available; "
                "place it under MSST_V2M_MODEL_DIR or configure a catalog source."
            )
        from ._vendor.inference.kashi_g2p_ja.g2p_onnx_runtime import G2POnnxRuntime

        runtime_model = runtime.get_or_load(
            ("kashi_g2p", str(model_path), "cpu"),
            namespace="v2m",
            label=f"kashi-g2p:{Path(model_path).name}",
            builder=lambda: G2POnnxRuntime(model_path, prefer_dml=False),
            closer=lambda model: None,
        )
        result = runtime_model.predict(text)
        return {"reading": result.get("reading", ""), "edges": result.get("edges", [])}

    from ._vendor.inference.API.lfa_api import get_ja_g2p, get_zh_g2p, get_yue_g2p

    if language == "ja":
        g2p = get_ja_g2p(engine=engine, model_dir=None)
        return {"reading": g2p.convert(text, include_tone=False, convert_number=True)}
    if language == "yue":
        g2p = get_yue_g2p()
        return {"reading": g2p.convert(text, include_tone=False, convert_number=True)}
    g2p = get_zh_g2p()
    return {"reading": g2p.convert(text, include_tone=False, convert_number=True)}
