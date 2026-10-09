from __future__ import annotations

import numpy as np
import torch

from msst_api.separator import (
    MSSeparator,
    ModelSpec,
    _MonoInputAdapter,
    _state_matches,
    resolve_device,
)
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


class _MonoAssertModel(torch.nn.Module):
    """Mono model that rejects anything but a single input channel."""

    def __init__(self) -> None:
        super().__init__()
        self.stereo = False
        self.audio_channels = 1
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert not self.stereo and x.shape[1] == 1, "mono model requires 1 channel"
        return x.unsqueeze(1)  # (batch, stems=1, channels, time)


def _mono_config():
    from ml_collections import ConfigDict

    return ConfigDict(
        {
            "audio": {"chunk_size": 16},
            "inference": {"num_overlap": 1, "batch_size": 1},
            "training": {
                "instruments": ["noreverb"],
                "target_instrument": "noreverb",
                "use_amp": False,
            },
        }
    )


def test_bigshifts_wrapper_rejects_mono_model_without_adapter():
    """Documents the msst bug the adapter works around (demix expands mono)."""

    from msst.utils.model_utils import bigshifts_wrapper

    mix = np.random.default_rng(0).standard_normal((1, 64)).astype(np.float32)
    with pytest.raises(AssertionError):
        bigshifts_wrapper(
            _mono_config(), _MonoAssertModel(), mix, "cpu", model_type="bs_roformer"
        )


def test_mono_input_adapter_lets_bigshifts_run():
    from msst.utils.model_utils import bigshifts_wrapper

    mix = np.random.default_rng(0).standard_normal((1, 64)).astype(np.float32)
    adapter = _MonoInputAdapter(_MonoAssertModel())
    out = bigshifts_wrapper(
        _mono_config(), adapter, mix, "cpu", model_type="bs_roformer"
    )
    assert set(out) == {"noreverb"}
    assert out["noreverb"].shape == (1, 64)


def test_mono_input_adapter_drops_duplicated_channel():
    adapter = _MonoInputAdapter(_MonoAssertModel())
    stereo = torch.zeros(2, 2, 32)
    out = adapter(stereo)
    assert out.shape == (2, 1, 1, 32)
    assert adapter.stereo is False


def test_mono_input_adapter_preserves_stereo_flag():
    model = _MonoAssertModel()
    model.stereo = True
    adapter = _MonoInputAdapter(model)
    assert adapter.stereo is True


def test_load_wraps_mono_model(tmp_path, monkeypatch):
    import msst_api.separator as separator_module

    fake = _MonoAssertModel()
    config = {"training": {"model_type": "bs_roformer"}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text("training:\n  model_type: bs_roformer\n")
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_bytes(b"")

    monkeypatch.setattr(
        separator_module, "get_model_from_config", lambda *_: (fake, config)
    )
    monkeypatch.setattr(
        MSSeparator,
        "_read_state_dict",
        lambda self, _: {k: v.clone() for k, v in fake.state_dict().items()},
    )

    spec = ModelSpec(
        model_type="bs_roformer",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device="cpu",
    )
    separator = MSSeparator(
        spec, Settings(model_dir=tmp_path, config_dir=tmp_path, allow_cpu=True)
    )

    assert isinstance(separator.model, _MonoInputAdapter)
    # _coerce_channels must still see the underlying model's channel layout.
    assert separator.model.stereo is False
    mix = np.zeros((1, 8), dtype=np.float32)
    assert separator._coerce_channels(mix).shape == (1, 8)
