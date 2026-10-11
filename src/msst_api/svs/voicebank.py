"""Loads an OpenUtau-compatible DiffSinger voicebank folder.

A voicebank directory looks like::

    <voicebank>/
      character.txt            # name / author metadata
      dsconfig.yaml            # acoustic config
      aco.phonemes.json        # acoustic phoneme -> id
      aco.<name>.onnx          # acoustic model
      dictionary-*.txt         # (training dictionaries; unused here)
      dsdict.yaml              # symbol types (+ lang-prefixed entries)
      dsdur/     {dsconfig.yaml, <dur>.*.onnx, dsdict-<lang>.yaml, ...}
      dspitch/   {dsconfig.yaml, <pitch>.*.onnx, dsdict.yaml, ...}
      dsvariance/{dsconfig.yaml, <var>.*.onnx, dsdict.yaml, ...}
      dsvocoder/ {vocoder.yaml, <vocoder>.onnx}

Only the ONNX Runtime sessions the pipeline needs are created; identical model
files (the three sub-models share one linguistic encoder in practice) are
loaded once via :class:`_SessionPool`.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from .dsconfig import MAX_MEL_BINS, DsConfig, VocoderConfig
from .g2p import G2pDictionary

logger = logging.getLogger("msst_api.svs")

try:  # reuse the service's tuned CUDA provider options
    from ..v2m._vendor.inference.device_utils import resolve_onnx_providers
except Exception:  # noqa: BLE001 - vendored module missing in some installs
    resolve_onnx_providers = None  # type: ignore[assignment]


#: The DiffSinger vocoder is a fixed-shape, convolution-heavy NSF-HiFiGAN.  Its
#: ``Conv`` nodes hit ORT's cudnn-frontend FALLBACK mode (and the "May be
#: extremely slow" warning) under the service-wide ``DEFAULT`` conv search, so
#: let cuDNN use its lightweight heuristic search here instead.
_VOCODER_CONV_ALGO_SEARCH = "HEURISTIC"


def _providers(device: str, label: str, conv_algo_search: str | None = None):
    if resolve_onnx_providers is not None:
        _, providers = resolve_onnx_providers(
            device, label=label, conv_algo_search=conv_algo_search
        )
        return providers
    import onnxruntime as ort

    if device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _load_phonemes(path: str | Path) -> dict[str, int]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in data.items()}
    tokens: dict[str, int] = {}
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line:
            tokens[line] = index
    return tokens


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_g2p(root: Path, dict_name: str | None = None) -> G2pDictionary:
    """Load ``dsdict-<lang>.yaml`` (or ``dsdict.yaml``) from ``root``."""

    candidates: list[str] = []
    if dict_name:
        candidates.append(dict_name)
    candidates.append("dsdict.yaml")
    for name in candidates:
        path = root / name
        if path.is_file():
            g2p = G2pDictionary.from_yaml(path)
            # SP and AP are always vowels (OpenUtau does this after loading).
            g2p.add_symbol("SP", "vowel")
            g2p.add_symbol("AP", "vowel")
            return g2p
    raise FileNotFoundError(
        f"No g2p dictionary in {root} (tried {', '.join(candidates)})"
    )


class _SessionPool:
    """Cache of ONNX sessions keyed by resolved model path."""

    def __init__(self, device: str) -> None:
        self.device = device
        self._lock = threading.Lock()
        # Keyed by (path, conv_algo_search) so the same file can be loaded with
        # different cudnn conv-search options without clobbering each other.
        self._sessions: dict[tuple[Path, str | None], object] = {}

    def get(self, path: str | Path, conv_algo_search: str | None = None) -> object:
        path = Path(path).resolve()
        key = (path, conv_algo_search)
        with self._lock:
            cached = self._sessions.get(key)
            if cached is not None:
                return cached
        import onnxruntime as ort

        logger.info("Loading SVS ONNX %s (%s)", path.name, self.device)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=_providers(self.device, path.name, conv_algo_search),
        )
        with self._lock:
            self._sessions.setdefault(key, session)
            return self._sessions[key]

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            try:
                session.run  # noqa: B018 - touch before dispose
            except Exception:  # noqa: BLE001
                pass


class SubModel:
    """A dsdur / dspitch / dsvariance sub-model (linguistic + head)."""

    def __init__(self, pool: _SessionPool, root: Path, head_key: str) -> None:
        self.root = root
        self.config = DsConfig.load(root / "dsconfig.yaml")
        self.phoneme_tokens = _load_phonemes(root / self.config.phonemes)
        self.language_ids: dict = {}
        if self.config.use_lang_id and self.config.languages:
            self.language_ids = _load_json(root / self.config.languages)
        self.frame_ms = self.config.frame_ms
        head = getattr(self.config, head_key)
        if head is None:
            raise ValueError(f"{head_key!r} is not set in {root / 'dsconfig.yaml'}")
        self.linguistic = pool.get(root / self.config.linguistic)
        self.model = pool.get(root / head)
        self._g2p_cache: dict[str, G2pDictionary] = {}

    def g2p(self, dict_name: str | None = None) -> G2pDictionary:
        key = dict_name or "dsdict.yaml"
        cached = self._g2p_cache.get(key)
        if cached is None:
            cached = load_g2p(self.root, dict_name)
            self._g2p_cache[key] = cached
        return cached


def _load_vocoder(root: Path, config: DsConfig) -> tuple[object, VocoderConfig, Path]:
    bundled = root / "dsvocoder"
    location: Path | None = None
    if (bundled / "vocoder.yaml").is_file():
        location = bundled
    elif config.vocoder:
        named = root.parent / config.vocoder
        if (named / "vocoder.yaml").is_file():
            location = named
    if location is None:
        raise FileNotFoundError(
            f"No vocoder found for {root.name} (expected dsvocoder/vocoder.yaml)"
        )
    vocoder_config = VocoderConfig.load(location / "vocoder.yaml")
    return location, vocoder_config, location / vocoder_config.model


def probe_voicebank(root: str | Path) -> dict:
    """Read a voicebank's metadata without creating any ONNX session."""

    root = Path(root)
    config = DsConfig.load(root / "dsconfig.yaml")
    name, author = _read_character(root)
    languages: list[str] = []
    for folder in ("dsdur", "dspitch", "dsvariance"):
        sub = root / folder
        if (sub / "dsconfig.yaml").is_file():
            sub_config = DsConfig.load(sub / "dsconfig.yaml")
            if sub_config.use_lang_id and sub_config.languages:
                try:
                    languages = sorted(_load_json(sub / sub_config.languages))
                except Exception:  # noqa: BLE001
                    languages = []
            break
    vocoder_name = ""
    vocoder_root = root / "dsvocoder"
    if (vocoder_root / "vocoder.yaml").is_file():
        try:
            vocoder_name = VocoderConfig.load(vocoder_root / "vocoder.yaml").name
        except Exception:  # noqa: BLE001
            vocoder_name = ""
    return {
        "id": root.name,
        "name": name,
        "author": author,
        "sample_rate": config.sample_rate,
        "hop_size": config.hop_size,
        "languages": languages,
        "has_pitch": (root / "dspitch" / "dsconfig.yaml").is_file(),
        "has_variance": (root / "dsvariance" / "dsconfig.yaml").is_file(),
        "has_duration": (root / "dsdur" / "dsconfig.yaml").is_file(),
        "vocoder": vocoder_name,
        "downloaded": True,
    }


def _read_character(root: Path) -> tuple[str, str]:
    name = root.name
    author = ""
    char_file = root / "character.txt"
    if char_file.is_file():
        for line in char_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("name="):
                name = line[5:].strip() or name
            elif line.startswith("author="):
                author = line[7:].strip()
    return name, author


class DiffSingerVoicebank:
    def __init__(self, root: str | Path, device: str = "cuda") -> None:
        self.root = Path(root).resolve()
        self.device = device
        config_path = self.root / "dsconfig.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Not a DiffSinger voicebank: {self.root}")
        self.config = DsConfig.load(config_path)
        self.phoneme_tokens = _load_phonemes(self.root / (self.config.phonemes or ""))
        self.language_ids: dict = {}
        if self.config.use_lang_id and self.config.languages:
            self.language_ids = _load_json(self.root / self.config.languages)
        self.name, self.author = self._read_character()

        self._pool = _SessionPool(device)
        self.acoustic = self._pool.get(self.root / (self.config.acoustic or ""))
        vocoder_root, self.vocoder_config, vocoder_model = _load_vocoder(
            self.root, self.config
        )
        self.vocoder_root = vocoder_root
        self.vocoder = self._pool.get(
            vocoder_model, conv_algo_search=_VOCODER_CONV_ALGO_SEARCH
        )
        self._validate_vocoder()

        self.dur = self._submodel("dsdur", "dur")
        self.pitch = self._submodel("dspitch", "pitch")
        self.variance = self._submodel("dsvariance", "variance")

    # -- construction helpers ---------------------------------------------
    def _submodel(self, folder: str, head_key: str) -> SubModel | None:
        root = self.root / folder
        if (root / "dsconfig.yaml").is_file():
            return SubModel(self._pool, root, head_key)
        return None

    def _read_character(self) -> tuple[str, str]:
        return _read_character(self.root)

    def _validate_vocoder(self) -> None:
        cfg, voc = self.config, self.vocoder_config
        if not (1 <= voc.num_mel_bins <= MAX_MEL_BINS):
            raise ValueError(f"Vocoder num_mel_bins out of range: {voc.num_mel_bins}")

    # -- convenience -------------------------------------------------------
    @property
    def sample_rate(self) -> int:
        return self.vocoder_config.sample_rate

    @property
    def hop_size(self) -> int:
        return self.vocoder_config.hop_size

    @property
    def frame_ms(self) -> float:
        return self.vocoder_config.frame_ms

    @property
    def has_pitch(self) -> bool:
        return self.pitch is not None

    @property
    def has_variance(self) -> bool:
        return self.variance is not None

    def tokenize(self, phoneme: str) -> int:
        try:
            return self.phoneme_tokens[phoneme]
        except KeyError:
            raise KeyError(
                f"Phoneme {phoneme!r} is not in the acoustic phoneme set"
            ) from None

    def close(self) -> None:
        self._pool.close_all()

    def info(self) -> dict:
        return {
            "id": self.root.name,
            "name": self.name,
            "author": self.author,
            "sample_rate": self.sample_rate,
            "hop_size": self.hop_size,
            "languages": sorted(self.language_ids) if self.language_ids else [],
            "has_pitch": self.has_pitch,
            "has_variance": self.has_variance,
            "has_duration": self.dur is not None,
            "vocoder": self.vocoder_config.name,
        }
