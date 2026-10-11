"""``dsconfig.yaml`` / ``vocoder.yaml`` data classes.

Mirrors ``OpenUtau.Core.DiffSinger.DsConfig`` and ``DsVocoderConfig`` (the yaml
keys are snake_case, as written by the OpenUtau voicebank tooling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

MAX_MEL_BINS = 255


@dataclass
class RandomPitchShifting:
    range: list[float] = field(default_factory=lambda: [-10.0, 8.0])


@dataclass
class DsConfig:
    phonemes: str = "phonemes.txt"
    languages: str | None = None
    acoustic: str | None = None
    vocoder: str | None = None
    speakers: list[str] | None = None
    hidden_size: int = 256
    use_key_shift_embed: bool = False
    use_speed_embed: bool = False
    use_energy_embed: bool = False
    use_breathiness_embed: bool = False
    use_voicing_embed: bool = False
    use_tension_embed: bool = False
    use_continuous_acceleration: bool = False
    use_lang_id: bool = False
    use_variable_depth: bool = False
    max_depth: float = 1.0
    # duration / pitch / variance sub-models
    linguistic: str | None = None
    dur: str | None = None
    pitch: str | None = None
    variance: str | None = None
    predict_dur: bool = True
    predict_energy: bool = True
    predict_breathiness: bool = True
    predict_voicing: bool = False
    predict_tension: bool = False
    use_expr: bool = False
    use_note_rest: bool = False
    # mel / audio
    sample_rate: int = 44100
    hop_size: int = 512
    win_size: int = 2048
    fft_size: int = 2048
    num_mel_bins: int = 128
    mel_fmin: float = 40.0
    mel_fmax: float = 16000.0
    mel_base: str = "10"
    mel_scale: str = "slaney"
    augmentation: RandomPitchShifting = field(default_factory=RandomPitchShifting)
    speakers_present: bool = False

    @property
    def frame_ms(self) -> float:
        return 1000.0 * self.hop_size / self.sample_rate

    @classmethod
    def load(cls, path: str | Path) -> "DsConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "DsConfig":
        aug_raw = (data.get("augmentation_args") or {}).get(
            "random_pitch_shifting"
        ) or {}
        augmentation = RandomPitchShifting(
            range=[float(x) for x in (aug_raw.get("range") or [-10.0, 8.0])]
        )

        variable_depth = data.get("use_variable_depth")
        if variable_depth is None:
            variable_depth = data.get("use_shallow_diffusion")
        continuous = bool(data.get("use_continuous_acceleration", False))
        raw_max_depth = float(data.get("max_depth", 1.0))
        max_depth = raw_max_depth if continuous else raw_max_depth / 1000.0

        speakers = data.get("speakers")

        return cls(
            phonemes=str(data.get("phonemes", "phonemes.txt")),
            languages=_opt_str(data.get("languages")),
            acoustic=_opt_str(data.get("acoustic")),
            vocoder=_opt_str(data.get("vocoder")),
            speakers=list(speakers) if speakers else None,
            hidden_size=int(data.get("hidden_size", 256)),
            use_key_shift_embed=bool(data.get("use_key_shift_embed", False)),
            use_speed_embed=bool(data.get("use_speed_embed", False)),
            use_energy_embed=bool(data.get("use_energy_embed", False)),
            use_breathiness_embed=bool(data.get("use_breathiness_embed", False)),
            use_voicing_embed=bool(data.get("use_voicing_embed", False)),
            use_tension_embed=bool(data.get("use_tension_embed", False)),
            use_continuous_acceleration=continuous,
            use_lang_id=bool(data.get("use_lang_id", False)),
            use_variable_depth=bool(variable_depth) if variable_depth is not None else False,
            max_depth=max_depth,
            linguistic=_opt_str(data.get("linguistic")),
            dur=_opt_str(data.get("dur")),
            pitch=_opt_str(data.get("pitch")),
            variance=_opt_str(data.get("variance")),
            predict_dur=bool(data.get("predict_dur", True)),
            predict_energy=bool(data.get("predict_energy", True)),
            predict_breathiness=bool(data.get("predict_breathiness", True)),
            predict_voicing=bool(data.get("predict_voicing", False)),
            predict_tension=bool(data.get("predict_tension", False)),
            use_expr=bool(data.get("use_expr", False)),
            use_note_rest=bool(data.get("use_note_rest", False)),
            sample_rate=int(data.get("sample_rate", 44100)),
            hop_size=int(data.get("hop_size", 512)),
            win_size=int(data.get("win_size", 2048)),
            fft_size=int(data.get("fft_size", 2048)),
            num_mel_bins=int(data.get("num_mel_bins", 128)),
            mel_fmin=float(data.get("mel_fmin", 40.0)),
            mel_fmax=float(data.get("mel_fmax", 16000.0)),
            mel_base=str(data.get("mel_base", "10")),
            mel_scale=str(data.get("mel_scale", "slaney")),
            augmentation=augmentation,
            speakers_present=bool(speakers),
        )


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


@dataclass
class VocoderConfig:
    name: str = "vocoder"
    model: str = "model.onnx"
    sample_rate: int = 44100
    hop_size: int = 512
    win_size: int = 2048
    fft_size: int = 2048
    num_mel_bins: int = 128
    mel_fmin: float = 40.0
    mel_fmax: float = 16000.0
    mel_base: str = "10"
    mel_scale: str = "slaney"
    pitch_controllable: bool = False
    force_on_cpu: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "VocoderConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "VocoderConfig":
        return cls(
            name=str(data.get("name", "vocoder")),
            model=str(data.get("model", "model.onnx")),
            sample_rate=int(data.get("sample_rate", 44100)),
            hop_size=int(data.get("hop_size", 512)),
            win_size=int(data.get("win_size", 2048)),
            fft_size=int(data.get("fft_size", 2048)),
            num_mel_bins=int(data.get("num_mel_bins", 128)),
            mel_fmin=float(data.get("mel_fmin", 40.0)),
            mel_fmax=float(data.get("mel_fmax", 16000.0)),
            mel_base=str(data.get("mel_base", "10")),
            mel_scale=str(data.get("mel_scale", "slaney")),
            pitch_controllable=bool(data.get("pitch_controllable", False)),
            force_on_cpu=bool(data.get("force_on_cpu", False)),
        )

    @property
    def frame_ms(self) -> float:
        return 1000.0 * self.hop_size / self.sample_rate
