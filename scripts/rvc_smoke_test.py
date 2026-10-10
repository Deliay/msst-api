#!/usr/bin/env python3
"""End-to-end RVC smoke test: load a voice model and convert synthetic audio.

Example::

    uv run python scripts/rvc_smoke_test.py --model alice --device cuda:0

The voice model must already be present in ``MSST_RVC_MODEL_DIR``
(``models/rvc_models`` by default).  Auxiliary assets (ContentVec + RMVPE) are
downloaded on first use.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import soundfile as sf

from msst_api.config import get_settings
from msst_api.rvc_engine import RVCManager, RVCParams, RVCVoiceSpec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Voice id (defaults to the first installed).")
    parser.add_argument("--device", default=None, help="Torch device (defaults to MSST_DEVICE).")
    parser.add_argument("--f0-method", default="rmvpe", choices=["rmvpe", "crepe", "crepe-tiny", "fcpe"])
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--output", default="rvc_output.wav")
    args = parser.parse_args()

    settings = get_settings()
    manager = RVCManager(settings)
    library = manager.library

    voices = library.all()
    if not voices:
        raise SystemExit(f"No RVC voices found in {settings.rvc_model_dir}")
    voice = library.get(args.model) if args.model else voices[0]
    print(f"Voice: {voice.name} (index={voice.index_path.name if voice.index_path else '-'})")

    device = args.device or settings.device
    spec = RVCVoiceSpec(
        model_path=voice.path,
        index_path=voice.index_path,
        device=device,
        embedder=settings.rvc_embedder,
    )
    params = RVCParams(pitch=args.pitch, f0_method=args.f0_method)

    sample_rate = 16000
    t = np.linspace(0, args.seconds, int(sample_rate * args.seconds), endpoint=False)
    mix = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    started = time.time()
    converted, out_sr = manager.convert(spec, mix, params)
    elapsed = time.time() - started
    print(f"Converted {args.seconds}s in {elapsed:.2f}s on {device} -> {out_sr} Hz")

    sf.write(args.output, converted, out_sr)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
