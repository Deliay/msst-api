"""In-memory RVC voice conversion.

Ported from Applio's ``rvc.infer.infer.VoiceConverter`` but stripped of file
I/O: the caller supplies a mono 16 kHz ``float32`` array and receives the
converted array plus its sample rate.  This is the same shape as the
``applio-api-plugin`` ``convert_in_memory`` helper.
"""

from __future__ import annotations

import logging

import librosa
import numpy as np
import torch

from ..configs.config import Config
from ..lib.algorithm.synthesizers import Synthesizer
from ..lib.tools.split_audio import merge_audio, process_audio
from ..lib.utils import load_embedding

logger = logging.getLogger("msst_api.rvc.convert")


class RVCConversionError(RuntimeError):
    """Raised when a voice model cannot be loaded or a conversion fails."""


class VoiceConverter:
    """Loads RVC checkpoints and converts audio entirely in memory."""

    def __init__(self, device: str | None = None) -> None:
        self.config = Config(device)
        self.hubert_model = None
        self.last_embedder_model = None
        self.tgt_sr: int | None = None
        self.net_g = None
        self.vc = None
        self.cpt = None
        self.version = None
        self.n_spk = None
        self.use_f0 = None
        self.loaded_model = None

    # -- model loading -----------------------------------------------------
    def load_hubert(self, embedder_model: str, embedder_model_custom: str | None = None) -> None:
        self.hubert_model = load_embedding(embedder_model, embedder_model_custom)
        self.hubert_model = self.hubert_model.to(self.config.device).float()
        self.hubert_model.eval()

    def get_vc(self, weight_root: str, sid) -> None:
        if sid == "" or sid == []:
            self.cleanup_model()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not self.loaded_model or self.loaded_model != weight_root:
            self.load_model(weight_root)
            if self.cpt is not None:
                self.setup_network()
                self.setup_vc_instance()
                self.loaded_model = weight_root
            else:
                self.vc = None
                self.loaded_model = None

    def cleanup_model(self) -> None:
        if self.hubert_model is not None:
            del self.net_g, self.n_spk, self.vc, self.hubert_model, self.tgt_sr
            self.hubert_model = self.net_g = self.n_spk = self.vc = self.tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del self.net_g, self.cpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.cpt = None

    def load_model(self, weight_root: str) -> None:
        self.cpt = torch.load(weight_root, map_location="cpu", weights_only=True)

    def setup_network(self) -> None:
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
        self.use_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")
        text_enc_hidden_dim = 768 if self.version == "v2" else 256
        vocoder = self.cpt.get("vocoder", "HiFi-GAN")
        self.net_g = Synthesizer(
            *self.cpt["config"],
            use_f0=self.use_f0,
            text_enc_hidden_dim=text_enc_hidden_dim,
            vocoder=vocoder,
        )
        del self.net_g.enc_q
        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g = self.net_g.to(self.config.device).float()
        self.net_g.eval()

    def setup_vc_instance(self) -> None:
        from ..infer.pipeline import Pipeline as VC

        self.vc = VC(self.tgt_sr, self.config)
        self.n_spk = self.cpt["config"][-3]

    # -- optional post-processing -----------------------------------------
    @staticmethod
    def remove_audio_noise(data, sr, reduction_strength: float = 0.7):
        try:
            import noisereduce as nr

            return nr.reduce_noise(y=data, sr=sr, prop_decrease=reduction_strength)
        except Exception as error:  # noqa: BLE001 - optional dependency
            logger.warning("Noise reduction failed, continuing without it: %s", error)
            return None

    @staticmethod
    def post_process_audio(audio_input, sample_rate, **kwargs):
        from pedalboard import (
            Bitcrush,
            Chorus,
            Clipping,
            Compressor,
            Delay,
            Distortion,
            Gain,
            Limiter,
            Pedalboard,
            PitchShift,
            Reverb,
        )

        board = Pedalboard()
        if kwargs.get("reverb", False):
            board.append(
                Reverb(
                    room_size=kwargs.get("reverb_room_size", 0.5),
                    damping=kwargs.get("reverb_damping", 0.5),
                    wet_level=kwargs.get("reverb_wet_level", 0.33),
                    dry_level=kwargs.get("reverb_dry_level", 0.4),
                    width=kwargs.get("reverb_width", 1.0),
                    freeze_mode=kwargs.get("reverb_freeze_mode", 0),
                )
            )
        if kwargs.get("pitch_shift", False):
            board.append(PitchShift(semitones=kwargs.get("pitch_shift_semitones", 0)))
        if kwargs.get("limiter", False):
            board.append(
                Limiter(
                    threshold_db=kwargs.get("limiter_threshold", -6),
                    release_ms=kwargs.get("limiter_release", 0.05),
                )
            )
        if kwargs.get("gain", False):
            board.append(Gain(gain_db=kwargs.get("gain_db", 0)))
        if kwargs.get("distortion", False):
            board.append(Distortion(drive_db=kwargs.get("distortion_gain", 25)))
        if kwargs.get("chorus", False):
            board.append(
                Chorus(
                    rate_hz=kwargs.get("chorus_rate", 1.0),
                    depth=kwargs.get("chorus_depth", 0.25),
                    centre_delay_ms=kwargs.get("chorus_delay", 7),
                    feedback=kwargs.get("chorus_feedback", 0.0),
                    mix=kwargs.get("chorus_mix", 0.5),
                )
            )
        if kwargs.get("bitcrush", False):
            board.append(Bitcrush(bit_depth=kwargs.get("bitcrush_bit_depth", 8)))
        if kwargs.get("clipping", False):
            board.append(Clipping(threshold_db=kwargs.get("clipping_threshold", 0)))
        if kwargs.get("compressor", False):
            board.append(
                Compressor(
                    threshold_db=kwargs.get("compressor_threshold", 0),
                    ratio=kwargs.get("compressor_ratio", 1),
                    attack_ms=kwargs.get("compressor_attack", 1.0),
                    release_ms=kwargs.get("compressor_release", 100),
                )
            )
        if kwargs.get("delay", False):
            board.append(
                Delay(
                    delay_seconds=kwargs.get("delay_seconds", 0.5),
                    feedback=kwargs.get("delay_feedback", 0.0),
                    mix=kwargs.get("delay_mix", 0.5),
                )
            )
        return board(audio_input, sample_rate)

    # -- conversion --------------------------------------------------------
    def convert(
        self,
        audio: np.ndarray,
        model_path: str,
        index_path: str = "",
        pitch: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        volume_envelope: float = 1.0,
        protect: float = 0.5,
        split_audio: bool = False,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1.0,
        embedder_model: str = "contentvec",
        embedder_model_custom: str | None = None,
        clean_audio: bool = False,
        clean_strength: float = 0.5,
        resample_sr: int = 0,
        sid: int = 0,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
        post_process: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, int]:
        """Convert ``audio`` (mono 16 kHz float32) and return ``(audio, sr)``."""

        if not model_path:
            raise RVCConversionError("No RVC model path provided")

        self.get_vc(model_path, sid)
        if self.vc is None:
            raise RVCConversionError(f"Failed to load RVC model: {model_path}")

        audio = np.asarray(audio, dtype=np.float32).flatten()
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio = audio / audio_max

        if not self.hubert_model or embedder_model != self.last_embedder_model:
            self.load_hubert(embedder_model, embedder_model_custom)
            self.last_embedder_model = embedder_model

        file_index = (
            (index_path or "")
            .strip()
            .strip('"')
            .strip("\n")
            .strip()
            .replace("trained", "added")
        )

        if split_audio:
            chunks, intervals = process_audio(audio, 16000)
        else:
            chunks, intervals = [audio], None

        converted_chunks = []
        for chunk in chunks:
            converted_chunks.append(
                self.vc.pipeline(
                    model=self.hubert_model,
                    net_g=self.net_g,
                    sid=sid,
                    audio=chunk,
                    pitch=pitch,
                    f0_method=f0_method,
                    file_index=file_index,
                    index_rate=index_rate,
                    pitch_guidance=self.use_f0,
                    volume_envelope=volume_envelope,
                    version=self.version,
                    protect=protect,
                    f0_autotune=f0_autotune,
                    f0_autotune_strength=f0_autotune_strength,
                    proposed_pitch=proposed_pitch,
                    proposed_pitch_threshold=proposed_pitch_threshold,
                )
            )

        if split_audio:
            audio_opt = merge_audio(chunks, converted_chunks, intervals, 16000, self.tgt_sr)
        else:
            audio_opt = converted_chunks[0]

        if clean_audio:
            cleaned = self.remove_audio_noise(audio_opt, self.tgt_sr, clean_strength)
            if cleaned is not None:
                audio_opt = cleaned

        if post_process:
            audio_opt = self.post_process_audio(
                audio_input=audio_opt, sample_rate=self.tgt_sr, **kwargs
            )

        out_sr = int(self.tgt_sr)
        if resample_sr and resample_sr >= 16000 and resample_sr != out_sr:
            audio_opt = librosa.resample(
                np.asarray(audio_opt, dtype=np.float32),
                orig_sr=out_sr,
                target_sr=int(resample_sr),
                res_type="soxr_vhq",
            )
            out_sr = int(resample_sr)

        return np.asarray(audio_opt, dtype=np.float32).flatten(), out_sr
