"""Runtime configuration for the vendored RVC engine.

The upstream Applio ``Config`` reads its JSON presets relative to the current
working directory and is a process-wide singleton.  This port instead resolves
the presets relative to this package and takes an explicit device, so several
conversions with different devices can coexist in one process.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

_CONFIG_DIR = Path(__file__).resolve().parent

#: Presets keyed by the model's output sample rate.  The first entry whose
#: sample rate matches the checkpoint is used by the synthesizer.
VERSION_CONFIG_PATHS = ["48000.json", "40000.json", "32000.json", "24000.json"]


class Config:
    """Device + preset holder passed to :class:`~msst_api.rvc.infer.pipeline.Pipeline`."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gpu_name = None
        self.gpu_mem = None
        self.json_config = self.load_config_json()
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def load_config_json(self) -> dict:
        configs: dict = {}
        for config_file in VERSION_CONFIG_PATHS:
            path = _CONFIG_DIR / config_file
            with path.open("r", encoding="utf-8") as handle:
                configs[config_file] = json.load(handle)
        return configs

    def device_config(self) -> tuple[int, int, int, int]:
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            self.set_cuda_config()
        else:
            self.device = "cpu"

        # Default tuning for a 6 GB GPU.
        x_pad, x_query, x_center, x_max = (1, 6, 38, 41)
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            # Tighter tuning for a 5 GB GPU.
            x_pad, x_query, x_center, x_max = (1, 5, 30, 32)
        return x_pad, x_query, x_center, x_max

    def set_cuda_config(self) -> None:
        index = int(str(self.device).split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(index)
        self.gpu_mem = torch.cuda.get_device_properties(index).total_memory // (1024**3)


def max_vram_gpu(gpu: int) -> int | str:
    if torch.cuda.is_available():
        return round(torch.cuda.get_device_properties(gpu).total_memory / 1024 / 1024 / 1024)
    return "8"


def get_gpu_info() -> str:
    ngpu = torch.cuda.device_count()
    gpu_infos = []
    if torch.cuda.is_available() or ngpu != 0:
        for i in range(ngpu):
            name = torch.cuda.get_device_name(i)
            mem = int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4)
            gpu_infos.append(f"{i}: {name} ({mem} GB)")
    return "\n".join(gpu_infos) if gpu_infos else "No compatible GPU available."


def get_number_of_gpus() -> str:
    if torch.cuda.is_available():
        return "-".join(map(str, range(torch.cuda.device_count())))
    return "-"
