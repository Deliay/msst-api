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

## Pretrained weights

Checkpoints are downloaded at runtime from the Hugging Face repository
`Sucial/MSST-WebUI` (see `THIRD_PARTY_NOTICES` in that repository and the
upstream projects listed in `docs/pretrained_models.md` of MSST for the
individual model licenses). Each model remains under the license chosen by its
original author.

## PyTorch

- Source: https://pytorch.org
- License: BSD-3-Clause
