"""F0 (pitch) predictors for the vendored RVC engine.

Only the three estimators exposed by the API are kept:

* ``rmvpe`` (default) — weights downloaded from the Applio HF repo;
* ``crepe`` / ``crepe-tiny`` — via ``torchcrepe`` (downloads its own weights);
* ``fcpe`` — via ``torchfcpe`` (bundled model, no download).

The upstream "swift" ONNX predictor is intentionally omitted to avoid pulling
``onnxruntime`` and ``swift-f0`` into the service image.
"""

from __future__ import annotations

import librosa
import numpy as np
import torch
import torchcrepe

from ..assets import ensure_predictor
from .RMVPE import RMVPE0Predictor


class RMVPE:
    def __init__(
        self,
        device,
        model_name: str = "rmvpe.pt",
        sample_rate: int = 16000,
        hop_size: int = 160,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.model = RMVPE0Predictor(
            str(ensure_predictor(model_name)), device=self.device
        )

    def get_f0(self, x, filter_radius: float = 0.03) -> np.ndarray:
        return self.model.infer_from_audio(x, thred=filter_radius)


class CREPE:
    def __init__(self, device, sample_rate: int = 16000, hop_size: int = 160) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size

    def get_f0(
        self, x, f0_min: float = 50, f0_max: float = 1100, p_len=None, model: str = "full"
    ) -> np.ndarray:
        if p_len is None:
            p_len = x.shape[0] // self.hop_size
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.asarray(x))

        f0, pd = torchcrepe.predict(
            x.float().to(self.device).unsqueeze(dim=0),
            self.sample_rate,
            self.hop_size,
            f0_min,
            f0_max,
            model=model,
            batch_size=512,
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        return f0[0].cpu().numpy()


class FCPE:
    def __init__(self, device, sample_rate: int = 16000, hop_size: int = 160) -> None:
        from torchfcpe import spawn_bundled_infer_model

        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.model = spawn_bundled_infer_model(device=device)

    def get_f0(self, x, p_len=None, filter_radius: float = 0.006) -> np.ndarray:
        if p_len is None:
            p_len = x.shape[0] // self.hop_size
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.asarray(x))
        f0 = (
            self.model.infer(
                x.float().to(self.device).unsqueeze(0),
                sr=self.sample_rate,
                decoder_mode="local_argmax",
                threshold=filter_radius,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        return f0
