import pathlib

import click

from deployment.api import deploy_model
from inference.api import load_inference_model
from lib.config.schema import ConfigurationScope


@click.command(help="Export supervised forced alignment as separate ONNX graphs.")
@click.option(
    "-m", "--model", required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Checkpoint with config.yaml and vocabulary.json beside it.",
)
@click.option(
    "-o", "--save-dir", required=True,
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    help="Directory for ONNX graphs, config.json and vocabulary.json.",
)
@click.option(
    "--opset-version", type=click.IntRange(min=18, max=20), default=18, show_default=True,
    help="ONNX opset; 18 supports scoring reductions and DirectML.",
)
def main(model: pathlib.Path, save_dir: pathlib.Path, opset_version: int):
    backend, vocabulary, _ = load_inference_model(model, scope=ConfigurationScope.FA)
    deploy_model(
        backend, vocabulary, save_dir, opset_version=opset_version,
        config_path=model.parent / "config.yaml",
    )


if __name__ == "__main__":
    main()
