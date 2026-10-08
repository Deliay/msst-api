from __future__ import annotations

from pathlib import Path

from msst_api.config import BUNDLED_CONFIG_DIR
from msst_api.registry import ModelRegistry, get_registry

REGISTRY_JSON = BUNDLED_CONFIG_DIR.parent / "registry.json"


def test_registry_loads_all_catalog_models() -> None:
    registry = get_registry()
    assert len(registry) == 51
    assert "model_bs_roformer_ep_317_sdr_12.9755.ckpt" in registry


def test_every_model_has_a_bundled_config() -> None:
    registry = ModelRegistry(REGISTRY_JSON, BUNDLED_CONFIG_DIR)
    for entry in registry.all():
        config_path = registry.resolve_config(entry)
        assert config_path.is_file(), f"missing config for {entry.id}"
        assert entry.model_type, f"missing model_type for {entry.id}"


def test_public_dict_shape() -> None:
    entry = get_registry().get("scnet_checkpoint_musdb18.ckpt")
    payload = entry.to_public_dict()
    assert payload["model_type"] == "scnet"
    assert "vocals" in payload["instruments"]
    assert payload["category"] == "multi_stem_models"


def test_unknown_model_raises() -> None:
    import pytest

    from msst_api.registry import ModelNotFoundError

    with pytest.raises(ModelNotFoundError):
        get_registry().get("does-not-exist.ckpt")
