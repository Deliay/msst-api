# Third-party notices

This project is an HTTP wrapper around third-party source separation software
and model artifacts. It does not redistribute model weights.

## Music-Source-Separation-Training (MSST)

- Source: https://github.com/ZFTurbo/Music-Source-Separation-Training
- License: MIT
- Used as the inference engine, installed from PyPI as the `msst` package.

## MSST-WebUI

- Source: https://github.com/SUC-DriverOld/MSST-WebUI
- License: AGPL-3.0
- Used as a **reference** for the service design and for the model catalog
  (`scripts/models_info.json`) and per-checkpoint YAML configuration files
  bundled under `src/msst_api/data/configs/`. No MSST-WebUI source code is
  copied; only data files (configuration and catalog metadata) are reused.

## Applio (RVC engine)

- Source: https://github.com/IAHispano/Applio
- License: MIT
- The RVC voice-conversion engine under `src/msst_api/rvc/` is a trimmed port
  of Applio's inference code (`rvc/infer`, `rvc/lib/algorithm`,
  `rvc/lib/predictors`, `rvc/configs`), adapted to run headless inside this
  service. The service design of the `/api/rvc/inference` endpoint follows
  [applio-api-plugin](https://github.com/Deliay/applio-api-plugin).
- Auxiliary assets (ContentVec embedder, RMVPE F0 predictor) are downloaded at
  runtime from the Hugging Face repository `IAHispano/Applio`.

## Vocal2Midi (v2m pipeline)

- Source: https://github.com/Xiantaidu/Vocal2Midi
- License: Apache-2.0
- The Vocal2Midi inference pipeline (slicing, GAME note extraction, TiFA /
  HubertFA forced alignment, G2P, quantization and MIDI/USTX/VSQX/TextGrid
  export) is vendored under `src/msst_api/v2m/_vendor/` (`inference/` and
  `application/`), adapted to run headless on CUDA/CPU onnxruntime instead of
  DirectML. The Qwen3-ASR GGUF/llama.cpp runtime is **not** vendored; Qwen3-ASR
  runs through the official `qwen-asr` package (see below).
- The vendor tree also embeds the following upstream components:
  - TiFA — https://github.com/openvpi/TIFA (MIT)
  - GAME — https://github.com/openvpi/GAME (MIT)
  - HubertFA — https://github.com/wolfgitpr/HubertFA
  - LyricFA — https://github.com/wolfgitpr/LyricFA
  - kashi-g2p — https://github.com/Xiantaidu/kashi-g2p (Apache-2.0; the
    compiled lexicon embeds JMdict/KANJIDIC2 (CC BY-SA 4.0), UniDic (BSD) and
    other dictionary data)
  - RMVPE, cc-cedict / cc-canto dictionary data (see the license files shipped
    inside the vendor tree)
- Vendored G2P dictionaries under `src/msst_api/v2m/_vendor/inference/**/dicts`
  retain their own licenses (CC-CEDICT / CC-CANTO / UniDic); see the
  `License.txt` files beside them.

## ASR models

- Qwen3-ASR — https://github.com/QwenLM/Qwen3-ASR (Apache-2.0). Used via the
  official `qwen-asr` package; weights are downloaded on first use.
- PinyinASR — https://github.com/Xiantaidu/PinyinASR (Apache-2.0).
- RomajiASR — https://github.com/Xiantaidu/RomajiASR (Apache-2.0).

## Vocal2Midi model weights (downloaded at runtime)

- GAME ONNX — https://github.com/openvpi/GAME (model files MIT; see the release).
- TiFA ONNX — exported from https://github.com/openvpi/TIFA (MIT code). The
  TIFA **model files** are released under **CC BY-NC-SA 4.0** (non-commercial);
  respect that license when deploying the aligner.
- HubertFA ONNX — https://github.com/wolfgitpr/HubertFA.
- kashi-g2p ONNX + lexicon — https://github.com/Xiantaidu/kashi-g2p
  (Apache-2.0; the lexicon bundle is CC BY-SA 4.0).
- RMVPE (v2m pitch) reuses the RVC/Applio `rmvpe.pt` asset (see above).

## Pretrained weights

Checkpoints are downloaded at runtime from the Hugging Face repository
`Sucial/MSST-WebUI` (see `THIRD_PARTY_NOTICES` in that repository and the
upstream projects listed in `docs/pretrained_models.md` of MSST for the
individual model licenses). Each model remains under the license chosen by its
original author.

## PyTorch

- Source: https://pytorch.org
- License: BSD-3-Clause
