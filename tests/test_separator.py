from __future__ import annotations

import numpy as np
import torch

from msst_api.separator import MSSeparator, _state_matches, resolve_device
from msst_api.config import Settings
from pathlib import Path
import pytest


def test_state_matches_exact():
    model = torch.nn.Linear(4, 3)
    checkpoint = {k: v.clone() for k, v in model.state_dict().items()}
    assert _state_matches(model, checkpoint)


def test_state_matches_detects_shape_change():
    model = torch.nn.Linear(4, 3)
    other = torch.nn.Linear(4, 5)
    assert not _state_matches(model, other.state_dict())


def test_state_matches_ignores_non_dict():
    assert not _state_matches(torch.nn.Linear(1, 1), [1, 2, 3])


def _cpu_settings() -> Settings:
    return Settings(model_dir=Path("/tmp/models"), config_dir=Path("/tmp/configs"))


def test_resolve_device_rejects_cpu_by_default():
    with pytest.raises(Exception):
        resolve_device("cpu", _cpu_settings())


def test_resolve_device_allows_cpu_when_enabled():
    settings = Settings(
        model_dir=Path("/tmp/models"),
        config_dir=Path("/tmp/configs"),
        allow_cpu=True,
    )
    assert resolve_device("cpu", settings) == "cpu"


class _FakeModel:
    """Minimal stand-in exposing just the ``stereo`` attribute."""

    def __init__(self, stereo: bool) -> None:
        self.stereo = stereo


def _coerce_channels(model, mix: np.ndarray) -> np.ndarray:
    separator = object.__new__(MSSeparator)
    separator.model = model
    return separator._coerce_channels(mix)


def test_coerce_channels_mono_model_keeps_mono_input():
    mix = np.arange(8, dtype=np.float32).reshape(1, 8)
    out = _coerce_channels(_FakeModel(False), mix)
    assert out.shape == (1, 8)


def test_coerce_channels_mono_model_downmixes_stereo_input():
    mix = np.arange(16, dtype=np.float32).reshape(2, 8)
    out = _coerce_channels(_FakeModel(False), mix)
    assert out.shape == (1, 8)


def test_coerce_channels_stereo_model_duplicates_mono_input():
    mix = np.arange(8, dtype=np.float32).reshape(1, 8)
    out = _coerce_channels(_FakeModel(True), mix)
    assert out.shape == (2, 8)


def test_coerce_channels_defaults_to_stereo_without_attribute():
    mix = np.arange(8, dtype=np.float32).reshape(1, 8)
    out = _coerce_channels(object(), mix)
    assert out.shape == (2, 8)
