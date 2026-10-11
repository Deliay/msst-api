import json
import pathlib

from deployment.exporter import Exporter
from inference.api import load_config_for_inference
from inference.backend import InferenceBackend
from lib import logging
from lib.config.schema import ConfigurationScope
from lib.vocabulary import Vocabulary


def deploy_model(
    model: InferenceBackend,
    vocabulary: Vocabulary,
    save_dir: str | pathlib.Path,
    opset_version: int = 18,
    *,
    config_path: str | pathlib.Path,
):
    model_config, inference_config = load_config_for_inference(
        pathlib.Path(config_path), scope=ConfigurationScope.FA,
    )
    features = inference_config.features
    save_dir = pathlib.Path(save_dir)
    Exporter(model, save_dir, opset_version=opset_version).export()
    config = {
        "samplerate": features.audio_sample_rate,
        "timestep": features.timestep,
        "hop_size": features.hop_size,
        "fft_size": features.fft_size,
        "win_size": features.win_size,
        "num_mels": features.spectrogram.num_bins,
        "vocab_size": model_config.max_vocab_size,
    }
    with (save_dir / "config.json").open("w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    vocabulary.dump(save_dir / "vocabulary.json")
    logging.success(f"Deployment completed: '{save_dir.as_posix()}'.")
