"""One-time TiFA ONNX export from the upstream PyTorch release.

openvpi publishes TiFA as a PyTorch checkpoint (``model.pt`` + ``config.yaml``
+ ``vocabulary.json`` beside it), while this runtime consumes the five ONNX
graphs documented in TIFA's ``ONNX.md`` (``spectrogram``/``prepare``/``score``/
``select``/``model``).  Instead of reimplementing that conversion, the upstream
``deploy.py`` / ``deployment`` / ``inference`` code is vendored under
``_vendor/TIFA`` and executed in a **subprocess** so the torch/lightning-heavy
export never loads into the serving process.

The produced bundle is moved next to the source files, after which
:class:`~msst_api.model_store.ModelStore` sees ``tifa`` as ready.  The PyTorch
checkpoint is removed afterwards unless ``MSST_V2M_TIFA_KEEP_SOURCE`` is set.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger("msst_api.v2m.tifa_export")

VENDOR_DIR = Path(__file__).resolve().parent / "_vendor" / "TIFA"
_DEPLOY = VENDOR_DIR / "deploy.py"

#: Files the ONNX aligner runtime expects (see ``_vendor/inference/TiFA``).
ONNX_ARTIFACTS: tuple[str, ...] = (
    "config.json",
    "vocabulary.json",
    "spectrogram.onnx",
    "prepare.onnx",
    "score.onnx",
    "select.onnx",
    "model.onnx",
)

_EXPORT_LOCK = threading.Lock()


class TifaExportError(RuntimeError):
    """Raised when the TiFA PyTorch -> ONNX export fails."""


def onnx_bundle_present(model_dir: str | Path) -> bool:
    model_dir = Path(model_dir)
    return all((model_dir / name).is_file() for name in ONNX_ARTIFACTS)


def export_tifa_onnx(
    model_dir: str | Path,
    *,
    opset_version: int = 18,
    keep_source: bool = False,
    timeout: float = 3600.0,
) -> Path:
    """Export ``model_dir`` (a TiFA PyTorch release) into an ONNX bundle in place.

    Returns ``model_dir`` unchanged when the ONNX graphs are already present.
    """

    model_dir = Path(model_dir)
    model_pt = model_dir / "model.pt"
    if onnx_bundle_present(model_dir):
        return model_dir
    if not model_pt.is_file():
        raise TifaExportError(
            f"TiFA source checkpoint not found at {model_pt}; cannot export ONNX."
        )
    if not _DEPLOY.is_file():
        raise TifaExportError(f"Vendored TiFA exporter is missing at {_DEPLOY}.")

    with _EXPORT_LOCK:
        if onnx_bundle_present(model_dir):
            return model_dir
        staging = Path(tempfile.mkdtemp(prefix=".tifa-export-", dir=str(model_dir.parent)))
        try:
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                filter(None, [str(VENDOR_DIR), env.get("PYTHONPATH", "")])
            )
            env["PYTHONUNBUFFERED"] = "1"
            command = [
                sys.executable,
                str(_DEPLOY),
                "-m",
                str(model_pt),
                "-o",
                str(staging),
                "--opset-version",
                str(int(opset_version)),
            ]
            logger.info("Exporting TiFA ONNX graphs from %s (opset %s)", model_pt, opset_version)
            try:
                proc = subprocess.run(
                    command,
                    cwd=str(VENDOR_DIR),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as error:  # pragma: no cover - defensive
                raise TifaExportError(
                    f"TiFA ONNX export timed out after {timeout:.0f}s."
                ) from error
            if proc.returncode != 0:
                tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-25:])
                raise TifaExportError(
                    f"TiFA ONNX export failed (exit {proc.returncode}).\n{tail}"
                )
            missing = [name for name in ONNX_ARTIFACTS if not (staging / name).is_file()]
            if missing:
                raise TifaExportError(
                    f"TiFA export did not produce: {', '.join(missing)}."
                )
            for name in ONNX_ARTIFACTS:
                os.replace(staging / name, model_dir / name)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        if not keep_source:
            model_pt.unlink(missing_ok=True)
        logger.info("TiFA ONNX bundle ready at %s", model_dir)
        return model_dir
