"""Small helpers around onnxruntime sessions."""

from __future__ import annotations

import numpy as np


def run_session(session, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run ``session`` and return outputs keyed by their graph names."""

    names = [output.name for output in session.get_outputs()]
    outputs = session.run(names, feed)
    return dict(zip(names, outputs))


def input_names(session) -> set[str]:
    return {i.name for i in session.get_inputs()}
