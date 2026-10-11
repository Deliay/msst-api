"""The DiffSinger model stages: auto-pitch, variance, acoustic, vocoder.

Ports of ``OpenUtau.Core.DiffSinger.DsPitch.Process``,
``DsVariance.Process`` and ``DiffSingerRenderer.InvokeDiffsinger`` for the
inputs the bundled voicebank declares (no speakers, ``use_lang_id: false``,
continuous acceleration, variable depth).
"""

from __future__ import annotations

import numpy as np

from .dsconfig import DsConfig
from .onnx_util import input_names, run_session
from .phrase import (
    HEAD_FRAMES,
    TAIL_FRAMES,
    PitchResult,
    RenderPhrase,
    durations_ms_to_frames,
    tone_to_freq,
)
from .voicebank import DiffSingerVoicebank, SubModel

HEAD_MS_FRAMES = HEAD_FRAMES
TAIL_MS_FRAMES = TAIL_FRAMES


# ---------------------------------------------------------------------------
# Helpers ported from DiffSingerUtils
# ---------------------------------------------------------------------------
def fit_duration_sum(durations: list[int], total_frames: int) -> list[int]:
    if not durations:
        return durations
    result = list(durations)
    delta = total_frames - sum(result)
    result[-1] += delta
    if result[-1] < 0:
        deficit = -result[-1]
        result[-1] = 0
        for i in range(len(result) - 2, -1, -1):
            if deficit <= 0:
                break
            take = min(result[i], deficit)
            result[i] -= take
            deficit -= take
        if deficit > 0:
            raise ValueError(f"Cannot fit DiffSinger durations to {total_frames} frames")
    return result


def padded_word_div_and_dur(
    phrase: RenderPhrase, ph_dur: list[int], is_vowel, frame_ms: float
) -> tuple[list[int], list[int]]:
    segments = phrase.padded_segments(frame_ms)
    vowel_ids = [
        i
        for i in range(len(segments))
        if segments[i][2] >= 0 and is_vowel(segments[i][0])
    ]
    if not vowel_ids:
        vowel_ids = [len(segments) - 2]
    word_div = [vowel_ids[0]]
    word_div += [b - a for a, b in zip(vowel_ids, vowel_ids[1:])]
    word_div += [len(segments) - vowel_ids[-1]]

    word_dur: list[int] = []
    offset = 0
    for length in word_div:
        word_dur.append(sum(ph_dur[offset : offset + length]))
        offset += length
    return word_div, word_dur


def resample_padded_curve(
    curve: list[float] | None,
    length: int,
    source_head: int,
    source_tail: int,
    target_head: int,
    target_tail: int,
    source_frame_ms: float,
    target_frame_ms: float,
) -> list[float] | None:
    if not curve:
        return None
    if (
        length == len(curve)
        and source_head == target_head
        and source_tail == target_tail
        and abs(source_frame_ms - target_frame_ms) < 0.001
    ):
        return curve
    # Different frame rate: linear resample by frame time; fall back to nearest.
    result = []
    for i in range(length):
        x = i * target_frame_ms / source_frame_ms
        if x <= 0:
            result.append(curve[0])
        elif x >= len(curve) - 1:
            result.append(curve[-1])
        else:
            x0 = int(x)
            frac = x - x0
            result.append(curve[x0] + (curve[x0 + 1] - curve[x0]) * frac)
    return result


# ---------------------------------------------------------------------------
# Auto-pitch (dspitch)
# ---------------------------------------------------------------------------
def predict_pitch(
    voicebank: DiffSingerVoicebank, phrase: RenderPhrase, steps: float = 20.0
) -> PitchResult:
    sub = voicebank.pitch
    if sub is None:
        raise ValueError("This voicebank has no pitch predictor (dspitch/)")
    config = sub.config
    frame_ms = sub.frame_ms
    g2p = sub.g2p()

    for phone in phrase.phones:
        if not g2p.is_valid_symbol(phone.phoneme):
            raise ValueError(
                f"Type definition of symbol {phone.phoneme!r} not found in the pitch dsdict"
            )

    segments = phrase.padded_segments(frame_ms)
    tokens = [_tokenize(sub, seg[0]) for seg in segments]
    ph_dur = phrase.padded_phone_durations(frame_ms)
    total_frames = sum(ph_dur)
    voiced = phrase.padded_voiced_mask(segments, ph_dur)

    linguistic_feed = {"tokens": np.asarray([tokens], dtype=np.int64)}
    if config.predict_dur:
        word_div, word_dur = padded_word_div_and_dur(
            phrase, ph_dur, g2p.is_vowel, frame_ms
        )
        linguistic_feed["word_div"] = np.asarray([word_div], dtype=np.int64)
        linguistic_feed["word_dur"] = np.asarray([word_dur], dtype=np.int64)
    else:
        linguistic_feed["ph_dur"] = np.asarray([ph_dur], dtype=np.int64)
    if config.use_lang_id:
        linguistic_feed["languages"] = np.asarray(
            [[_lang_id(sub, seg[0]) for seg in segments]], dtype=np.int64
        )
    linguistic_out = run_session(sub.linguistic, linguistic_feed)
    encoder_out = linguistic_out["encoder_out"]

    note_dur_ms, note_midi, note_rest = _build_notes(
        phrase, frame_ms, g2p.is_vowel
    )
    note_dur = fit_duration_sum(
        durations_ms_to_frames(note_dur_ms, frame_ms), total_frames
    )
    note_midi = _fill_rest_midi(note_midi, note_rest)

    pitch_feed = {
        "encoder_out": encoder_out,
        "note_midi": np.asarray([note_midi], dtype=np.float32),
        "note_dur": np.asarray([note_dur], dtype=np.int64),
        "ph_dur": np.asarray([ph_dur], dtype=np.int64),
        "pitch": np.full((1, total_frames), 60.0, dtype=np.float32),
        "retake": np.ones((1, total_frames), dtype=bool),
    }
    _add_pitch_sampling(pitch_feed, sub, steps)
    if config.use_expr:
        pitch_feed["expr"] = np.ones((1, total_frames), dtype=np.float32)
    if config.use_note_rest:
        pitch_feed["note_rest"] = np.asarray([note_rest], dtype=bool)

    pitch_out = run_session(sub.model, pitch_feed)
    tones = np.asarray(next(iter(pitch_out.values())), dtype=np.float32).reshape(-1)

    return _render_pitch_result(phrase, tones, voiced, frame_ms, total_frames)


def _render_pitch_result(phrase, tones, voiced, frame_ms, total_frames) -> PitchResult:
    start_ms = phrase.phones[0].position_ms - HEAD_FRAMES * frame_ms
    axis = phrase.time_axis
    ticks = [
        float(axis.ms_to_tick(start_ms + i * frame_ms) - phrase.position_tick)
        for i in range(total_frames)
    ]
    tone_list = [float(t) for t in tones]
    voiced_list = [bool(v) for v in voiced]
    pitch_end = (
        axis.ms_to_tick(start_ms + (total_frames - 1) * frame_ms) - phrase.position_tick
    )
    if pitch_end <= phrase.duration_tick:
        return PitchResult(
            ticks=ticks + [float(phrase.duration_tick + 1)],
            tones=tone_list + [tone_list[-1]],
            voiced=voiced_list + [voiced_list[-1]],
        )
    return PitchResult(ticks=ticks, tones=tone_list, voiced=voiced_list)


def _add_pitch_sampling(feed: dict, sub: SubModel, steps: float) -> None:
    names = input_names(sub.model)
    config = sub.config
    if steps < 1.0 and "depth" in names:
        depth = min(max(steps, 0.01), 1.0)
        sampling_steps = 1
        if config.use_continuous_acceleration:
            feed["depth"] = np.asarray([depth], dtype=np.float32)
            feed["steps"] = np.asarray([sampling_steps], dtype=np.int64)
        else:
            int_depth = int(min(max(round(depth * 1000), 1), 1000))
            speedup = max(1, int_depth // sampling_steps)
            while int_depth % speedup != 0 and speedup > 1:
                speedup -= 1
            int_depth = int_depth // speedup * speedup
            feed["depth"] = np.asarray([int_depth], dtype=np.int64)
            feed["speedup"] = np.asarray([speedup], dtype=np.int64)
        return
    if config.use_continuous_acceleration:
        feed["steps"] = np.asarray([int(max(1, round(steps)))], dtype=np.int64)
        return
    speedup = int(max(1, round(1000 / max(0.5, steps))))
    while 1000 % speedup != 0 and speedup > 1:
        speedup -= 1
    feed["speedup"] = np.asarray([speedup], dtype=np.int64)


def _build_notes(phrase: RenderPhrase, frame_ms: float, is_vowel):
    start_ms = phrase.phones[0].position_ms - HEAD_FRAMES * frame_ms
    notes = phrase.notes
    note_dur_ms = [max(0.0, notes[0].position_ms - start_ms)]
    note_midi = [float(notes[0].adjusted_tone)]
    note_rest = [True]
    prev_end = notes[0].position_ms
    for real_idx, note in enumerate(notes):
        gap = note.position_ms - prev_end
        if gap > 0:
            note_dur_ms.append(gap)
            note_midi.append(float(note.adjusted_tone))
            note_rest.append(True)
        note_dur_ms.append(note.duration_ms)
        note_midi.append(float(note.adjusted_tone))
        if note.lyric.startswith("+"):
            note_rest.append(note_rest[-1])
        else:
            phones = [p for p in phrase.phones if p.note_index == real_idx]
            is_rest = not phones or all(
                p.phoneme in ("AP", "SP") or not is_vowel(p.phoneme) for p in phones
            )
            note_rest.append(is_rest)
        prev_end = note.position_ms + note.duration_ms
    note_dur_ms.append(TAIL_FRAMES * frame_ms)
    note_midi.append(float(notes[-1].adjusted_tone))
    note_rest.append(True)
    return note_dur_ms, note_midi, note_rest


def _fill_rest_midi(note_midi: list[float], note_rest: list[bool]) -> list[float]:
    midi = list(note_midi)
    if all(note_rest):
        return [60.0] * len(midi)
    # Fill each rest run from the nearest non-rest neighbours.
    groups: list[tuple[int, int]] = []
    i = 0
    while i < len(note_rest):
        if not note_rest[i]:
            i += 1
            continue
        j = i
        while j < len(note_rest) and note_rest[j]:
            j += 1
        groups.append((i, j))
        i = j
    for start, end in groups:
        if start == 0:
            for k in range(0, end):
                midi[k] = midi[end]
        elif end == len(note_rest):
            for k in range(start, len(note_rest)):
                midi[k] = midi[start - 1]
        else:
            mid = (start + end + 1) // 2
            for k in range(start, mid):
                midi[k] = midi[start - 1]
            for k in range(mid, end):
                midi[k] = midi[end]
    return midi


# ---------------------------------------------------------------------------
# Variance (dsvariance)
# ---------------------------------------------------------------------------
def predict_variance(
    voicebank: DiffSingerVoicebank, phrase: RenderPhrase, steps: float = 20.0
) -> dict[str, list[float]]:
    sub = voicebank.variance
    if sub is None:
        raise ValueError("This voicebank has no variance predictor (dsvariance/)")
    config = sub.config
    frame_ms = sub.frame_ms

    segments = phrase.padded_segments(frame_ms)
    tokens = [_tokenize(sub, seg[0]) for seg in segments]
    ph_dur = phrase.padded_phone_durations(frame_ms)
    total_frames = sum(ph_dur)

    linguistic_feed = {"tokens": np.asarray([tokens], dtype=np.int64)}
    if config.predict_dur:
        g2p = sub.g2p()
        word_div, word_dur = padded_word_div_and_dur(
            phrase, ph_dur, g2p.is_vowel, frame_ms
        )
        linguistic_feed["word_div"] = np.asarray([word_div], dtype=np.int64)
        linguistic_feed["word_dur"] = np.asarray([word_dur], dtype=np.int64)
    else:
        linguistic_feed["ph_dur"] = np.asarray([ph_dur], dtype=np.int64)
    if config.use_lang_id:
        linguistic_feed["languages"] = np.asarray(
            [[_lang_id(sub, seg[0]) for seg in segments]], dtype=np.int64
        )
    linguistic_out = run_session(sub.linguistic, linguistic_feed)
    encoder_out = linguistic_out["encoder_out"]

    pitch = phrase.sample_curve(
        phrase.pitches, 0.0, frame_ms, total_frames, HEAD_FRAMES, TAIL_FRAMES,
        lambda x: x * 0.01,
    )
    tone_shift = phrase.sample_curve(
        phrase.tone_shift, 0.0, frame_ms, total_frames, HEAD_FRAMES, TAIL_FRAMES,
        lambda x: x * 0.01,
    )
    pitch = [a + b for a, b in zip(pitch, tone_shift)]

    num_variances = sum(
        [
            config.predict_energy,
            config.predict_breathiness,
            config.predict_voicing,
            config.predict_tension,
        ]
    )
    variance_feed = {
        "encoder_out": encoder_out,
        "ph_dur": np.asarray([ph_dur], dtype=np.int64),
        "pitch": np.asarray([pitch], dtype=np.float32),
        "retake": np.ones((1, total_frames, max(1, num_variances)), dtype=bool),
    }
    if config.predict_energy:
        variance_feed["energy"] = np.zeros((1, total_frames), dtype=np.float32)
    if config.predict_breathiness:
        variance_feed["breathiness"] = np.zeros((1, total_frames), dtype=np.float32)
    if config.predict_voicing:
        variance_feed["voicing"] = np.zeros((1, total_frames), dtype=np.float32)
    if config.predict_tension:
        variance_feed["tension"] = np.zeros((1, total_frames), dtype=np.float32)
    if config.use_continuous_acceleration:
        variance_feed["steps"] = np.asarray([int(max(1, round(steps)))], dtype=np.int64)
    else:
        speedup = max(1, 1000 // max(1, int(steps)))
        while 1000 % speedup != 0 and speedup > 1:
            speedup -= 1
        variance_feed["speedup"] = np.asarray([speedup], dtype=np.int64)

    outputs = run_session(sub.model, variance_feed)
    result: dict[str, list[float]] = {}
    for key in ("energy", "breathiness", "voicing", "tension"):
        pred = outputs.get(f"{key}_pred")
        if pred is not None:
            result[key] = [float(x) for x in np.asarray(pred).reshape(-1)]
    return result


# ---------------------------------------------------------------------------
# Acoustic + vocoder
# ---------------------------------------------------------------------------
def render_phrase(
    voicebank: DiffSingerVoicebank,
    phrase: RenderPhrase,
    variance: dict[str, list[float]] | None = None,
    depth: float = 1.0,
    steps: int = 20,
    variance_steps: int = 20,
) -> np.ndarray:
    config = voicebank.config
    vocoder = voicebank.vocoder_config
    frame_ms = vocoder.frame_ms

    if variance is None and (
        config.use_breathiness_embed
        or config.use_energy_embed
        or config.use_voicing_embed
        or config.use_tension_embed
    ):
        variance = predict_variance(voicebank, phrase, variance_steps)

    segments = phrase.padded_segments(frame_ms)
    tokens = [voicebank.tokenize(seg[0]) for seg in segments]
    durations = phrase.padded_phone_durations(frame_ms)
    total_frames = sum(durations)

    f0 = phrase.sample_curve(
        phrase.pitches, 0.0, frame_ms, total_frames, HEAD_FRAMES, TAIL_FRAMES,
        lambda x: tone_to_freq(x * 0.01),
    )
    shifted = phrase.sample_curve(
        phrase.tone_shift, 0.0, frame_ms, total_frames, HEAD_FRAMES, TAIL_FRAMES,
        lambda x: x,
    )
    shifted_f0 = [f * (2.0 ** (d / 1200.0)) for f, d in zip(f0, shifted)]

    feed: dict[str, np.ndarray] = {
        "tokens": np.asarray([tokens], dtype=np.int64),
        "durations": np.asarray([durations], dtype=np.int64),
        "f0": np.asarray([shifted_f0 if vocoder.pitch_controllable else f0], dtype=np.float32),
    }

    if config.use_breathiness_embed:
        user = phrase.sample_curve(
            phrase.breathiness, 0.0, frame_ms, total_frames, HEAD_FRAMES, TAIL_FRAMES,
            lambda x: x,
        )
        feed["breathiness"] = np.asarray(
            [_combine(variance.get("breathiness") if variance else None, user, "brec", total_frames)],
            dtype=np.float32,
        )
    if config.use_voicing_embed:
        user = phrase.sample_curve(
            phrase.voicing, 0.0, frame_ms, total_frames, HEAD_FRAMES, TAIL_FRAMES,
            lambda x: x,
        )
        feed["voicing"] = np.asarray(
            [_combine(variance.get("voicing") if variance else None, user, "voic", total_frames)],
            dtype=np.float32,
        )
    if config.use_tension_embed:
        user = phrase.sample_curve(
            phrase.tension, 0.0, frame_ms, total_frames, HEAD_FRAMES, TAIL_FRAMES,
            lambda x: x,
        )
        feed["tension"] = np.asarray(
            [_combine(variance.get("tension") if variance else None, user, "tenc", total_frames)],
            dtype=np.float32,
        )
    if config.use_key_shift_embed:
        gender = phrase.sample_curve(
            phrase.gender, 0.0, frame_ms, total_frames, HEAD_FRAMES, TAIL_FRAMES,
            _gender_convert(config),
        )
        feed["gender"] = np.asarray([gender], dtype=np.float32)
    if config.use_speed_embed:
        feed["velocity"] = np.ones((1, total_frames), dtype=np.float32)

    if config.use_continuous_acceleration:
        if config.use_variable_depth:
            real_depth = min(depth, config.max_depth)
            feed["depth"] = np.asarray([real_depth], dtype=np.float32)
        feed["steps"] = np.asarray([int(steps)], dtype=np.int64)
    else:
        speedup = max(1, 1000 // max(1, int(steps)))
        while 1000 % speedup != 0 and speedup > 1:
            speedup -= 1
        feed["speedup"] = np.asarray([speedup], dtype=np.int64)

    acoustic_out = run_session(voicebank.acoustic, feed)
    mel = np.asarray(next(iter(acoustic_out.values())), dtype=np.float32)
    if mel.ndim == 2:
        mel = mel[None, ...]
    if vocoder.mel_base != config.mel_base:
        if vocoder.mel_base == "e" and config.mel_base == "10":
            mel = mel * 2.30259
        elif vocoder.mel_base == "10" and config.mel_base == "e":
            mel = mel * 0.434294

    vocoder_feed = {
        "mel": mel.astype(np.float32),
        "f0": np.asarray([f0], dtype=np.float32),
    }
    vocoder_out = run_session(voicebank.vocoder, vocoder_feed)
    waveform = np.asarray(next(iter(vocoder_out.values())), dtype=np.float32).reshape(-1)

    hop = vocoder.hop_size
    if waveform.size > (HEAD_FRAMES + TAIL_FRAMES) * hop:
        waveform = waveform[HEAD_FRAMES * hop : waveform.size - TAIL_FRAMES * hop]
    return waveform


def _combine(predicted, user, abbr: str, total_frames: int):
    if predicted is None:
        return np.zeros(total_frames, dtype=np.float32)
    pred = np.asarray(predicted, dtype=np.float32)
    usr = np.asarray(user, dtype=np.float32)
    if pred.shape[0] != total_frames:
        pred = np.interp(
            np.linspace(0, 1, total_frames),
            np.linspace(0, 1, pred.shape[0]),
            pred,
        ).astype(np.float32)
    if abbr == "voic":
        out = pred + (usr - 100.0) * 12.0 / 100.0
        return np.clip(out, -96.0, 0.0)
    if abbr == "tenc":
        out = pred + usr / 20.0
        return np.clip(out, -10.0, 10.0)
    out = pred + usr * 12.0 / 100.0
    return np.clip(out, -96.0, 0.0)


def _gender_convert(config: DsConfig):
    low, high = config.augmentation.range[0], config.augmentation.range[1]
    positive = 0.0 if high == 0 else (12.0 / high / 100.0)
    negative = 0.0 if low == 0 else (-12.0 / low / 100.0)

    def convert(value: float) -> float:
        return (-value * positive) if value < 0 else (-value * negative)

    return convert


def _tokenize(sub: SubModel, phoneme: str) -> int:
    try:
        return sub.phoneme_tokens[phoneme]
    except KeyError:
        raise KeyError(
            f"Phoneme {phoneme!r} is not in the sub-model phoneme set"
        ) from None


def _lang_id(sub: SubModel, phoneme: str) -> int:
    lang = phoneme.split("/", 1)[0] if "/" in phoneme else ""
    return int(sub.language_ids.get(lang, 0))
