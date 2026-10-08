#!/usr/bin/env python3
"""End-to-end smoke test: download a catalog model and separate synthetic audio.

Example::

    uv run python scripts/smoke_test.py \
        --model model_vocals_mel_band_roformer_sdr_8.42.ckpt \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from msst_api.download import download_model
from msst_api.registry import get_registry
from msst_api.separator import MSSeparator, ModelSpec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="model_vocals_mel_band_roformer_sdr_8.42.ckpt",
        help="Catalog model id (see GET /api/msst/models).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()

    registry = get_registry()
    entry = registry.get(args.model)
    print(f"Model: {entry.id} ({entry.model_type}) instruments={entry.instruments}")

    start = time.time()
    checkpoint = download_model(
        entry.sources,
        entry.default_weight_path,
        expected_size=entry.size,
        expected_sha=entry.sha256,
    )
    print(f"Checkpoint ready in {time.time() - start:.1f}s: {checkpoint}")

    config_path = registry.resolve_config(entry)
    spec = ModelSpec(
        model_type=entry.model_type,
        config_path=config_path,
        checkpoint_path=checkpoint,
        device=args.device,
    )
    separator = MSSeparator(spec)
    sample_rate = separator.sample_rate

    t = np.linspace(0, args.seconds, int(sample_rate * args.seconds), endpoint=False)
    left = 0.3 * np.sin(2 * np.pi * 220 * t)
    right = 0.3 * np.sin(2 * np.pi * 330 * t)
    mix = np.stack([left, right]).astype(np.float32)

    started = time.time()
    results = separator.separate(mix)
    elapsed = time.time() - started
    print(f"Separated {args.seconds}s in {elapsed:.2f}s on {separator.device}")
    for name, array in results.items():
        peak = float(np.abs(array).max()) if array.size else 0.0
        print(f"  {name:16} shape={array.shape} peak={peak:.3f}")


if __name__ == "__main__":
    main()
