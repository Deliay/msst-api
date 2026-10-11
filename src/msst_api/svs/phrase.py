"""The per-phrase data the DiffSinger models consume.

`RenderPhrase` is a faithful-enough port of ``OpenUtau.Core.Render.RenderPhrase``
for the fields the DiffSinger renderer reads:

* ``notes`` -- the phrase's notes in part-absolute ticks/ms;
* ``phones`` -- contiguous phonemes with part-absolute positions;
* ``pitches`` -- the pitch (in cents) on OpenUtau's 5-tick grid, built from the
  note tones, their pitch-bend points, vibrato, and finally the ``pitd`` curve;
* ``tone_shift`` / ``gender`` / ``breathiness`` / ``voicing`` / ``tension`` --
  expression curves on the same grid.

The 5-tick index of the grid maps to ``phrase.position_tick + i * 5``; the
DiffSinger ``SampleCurve`` helper resamples that grid onto the model's frames.
This module also ports ``DiffSingerUtils.SampleCurve`` as
:meth:`RenderPhrase.sample_curve`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .time_axis import TimeAxis
from .ustx import Curve, PitchPoint, UstxNote, UstxProject

PITCH_INTERVAL = 5
HEAD_FRAMES = 8
TAIL_FRAMES = 8


def tone_to_freq(tone: float) -> float:
    """MIDI/scale tone (semitones) -> Hz."""

    return 440.0 * (2.0 ** ((tone - 69.0) / 12.0))


@dataclass
class RenderNote:
    lyric: str
    tone: int
    adjusted_tone: float
    position: int
    duration: int
    end: int
    position_ms: float
    duration_ms: float
    end_ms: float
    pitch_points: list[PitchPoint] = field(default_factory=list)
    vibrato: dict = field(default_factory=dict)


@dataclass
class RenderPhone:
    phoneme: str
    note_index: int
    position: int
    duration: int
    end: int
    position_ms: float
    duration_ms: float
    end_ms: float


@dataclass
class PitchResult:
    """Port of ``RenderPitchResult`` (tones in semitones)."""

    ticks: list[float]
    tones: list[float]
    voiced: list[bool] | None = None
    retake_mask: list[bool] | None = None


class RenderPhrase:
    def __init__(
        self,
        project: UstxProject,
        notes: list[RenderNote],
        phones: list[RenderPhone],
        head_frames: int = HEAD_FRAMES,
        tail_frames: int = TAIL_FRAMES,
    ) -> None:
        self.project = project
        self.time_axis: TimeAxis = project.time_axis
        self.notes = notes
        self.phones = phones
        self.head_frames = head_frames
        self.tail_frames = tail_frames
        self.leading = 0

        self.position_tick = phones[0].position
        self.end_tick = phones[-1].end
        self.duration_tick = self.end_tick - self.position_tick
        self.position_ms = phones[0].position_ms
        self.end_ms = phones[-1].end_ms
        self.duration_ms = self.end_ms - self.position_ms
        self.leading_ms = phones[0].position_ms

        self.pitches, self.pitches_before_deviation = self._build_pitches()
        self.tone_shift = self._sampled_curve("shfc", 0.0)
        self.gender = self._sampled_curve("genc", 0.0)
        self.breathiness = self._sampled_curve("brec", self._curve_default("brec"))
        self.voicing = self._sampled_curve("voic", self._curve_default("voic", 100.0))
        self.tension = self._sampled_curve("tenc", self._curve_default("tenc"))
        self.dynamics = self._sampled_curve("dyn", self._curve_default("dyn"))

    # -- grid helpers ------------------------------------------------------
    def _curve_default(self, abbr: str, fallback: float = 0.0) -> float:
        descriptor = self.project.expressions.get(abbr)
        if descriptor is None:
            return fallback
        try:
            return float(descriptor.get("default_value", fallback))
        except (TypeError, ValueError):
            return fallback

    def _grid_length(self) -> int:
        span = self.end_tick - self.position_tick
        return max(1, span // PITCH_INTERVAL + 1)

    def _sampled_curve(self, abbr: str, default: float) -> list[float]:
        curve = self.project.curve(abbr)
        lo, hi = self.project.expression_range(abbr)
        length = self._grid_length()
        result: list[float] = []
        for i in range(length):
            tick = self.position_tick + i * PITCH_INTERVAL
            value = curve.sample(tick) if not curve.is_empty else default
            if value < lo:
                value = lo
            elif value > hi:
                value = hi
            result.append(value)
        return result

    def _note_at(self, tick: float) -> RenderNote | None:
        for note in self.notes:
            if note.position <= tick < note.end:
                return note
        if tick < self.notes[0].position:
            return self.notes[0]
        return self.notes[-1]

    def _build_pitches(self) -> tuple[list[float], list[float]]:
        length = self._grid_length()
        base = [0.0] * length
        for i in range(length):
            tick = self.position_tick + i * PITCH_INTERVAL
            note = self._note_at(tick)
            base[i] = note.adjusted_tone * 100.0 if note else 60 * 100.0

        values = list(base)
        for note in self.notes:
            points = self._note_contour(note)
            if points is None:
                continue
            contour = Curve("pitch", points)
            for i in range(length):
                tick = self.position_tick + i * PITCH_INTERVAL
                if note.position <= tick < note.end:
                    values[i] = contour.sample(tick)
            if note.vibrato.get("length", 0) and note.vibrato.get("length", 0) > 0:
                self._apply_vibrato(note, values, length)

        before_deviation = list(values)
        pitd = self.project.curve("pitd")
        if not pitd.is_empty:
            for i in range(length):
                tick = self.position_tick + i * PITCH_INTERVAL
                values[i] += pitd.sample(tick)
        return values, before_deviation

    def _note_contour(self, note: RenderNote) -> list[PitchPoint] | None:
        if not note.pitch_points:
            return None
        base = note.adjusted_tone * 100.0
        points: list[PitchPoint] = []
        for p in note.pitch_points:
            ms = note.position_ms + p.x
            tick = self.time_axis.ms_to_tick(ms)
            points.append(PitchPoint(float(tick), base + p.y * 10.0, p.shape))
        points.sort(key=lambda q: q.x)
        if points[0].x > note.position:
            points.insert(0, PitchPoint(float(note.position), points[0].y, "io"))
        if points[-1].x < note.end:
            points.append(PitchPoint(float(note.end), points[-1].y, "io"))
        return points

    def _apply_vibrato(
        self, note: RenderNote, values: list[float], length: int
    ) -> None:
        vib = note.vibrato
        length_ticks = max(1, int(vib.get("length", 0)))
        period_ms = float(vib.get("period", 175) or 175)
        depth = float(vib.get("depth", 0) or 0)
        if period_ms <= 0 or depth == 0:
            return
        start = note.end - length_ticks
        for i in range(length):
            tick = self.position_tick + i * PITCH_INTERVAL
            if start <= tick < note.end:
                pos_ms = self.time_axis.tick_to_ms(tick) - self.time_axis.tick_to_ms(start)
                values[i] += depth * math.sin(2.0 * math.pi * pos_ms / period_ms)

    # -- DiffSingerUtils.SampleCurve ---------------------------------------
    def sample_curve(
        self,
        array: list[float] | None,
        default: float,
        frame_ms: float,
        length: int,
        head_frames: int,
        tail_frames: int,
        convert=lambda x: x,
    ) -> list[float]:
        result: list[float] = []
        if not array:
            return [convert(default) for _ in range(length)]
        start_ms = self.position_ms - head_frames * frame_ms
        last = len(array) - 1
        for i in range(length):
            pos_ms = start_ms + i * frame_ms
            ticks = self.time_axis.ms_to_tick(pos_ms) - self.position_tick
            index = ticks / PITCH_INTERVAL
            if index <= 0:
                value = array[0]
            elif index >= last:
                value = array[last]
            else:
                i0 = int(math.floor(index))
                frac = index - i0
                value = array[i0] + (array[i0 + 1] - array[i0]) * frac
            result.append(convert(value))
        return result

    # -- PhraseSource/PhonemeSource surrogate ------------------------------
    def padded_segments(self, frame_ms: float) -> list[tuple[str, float, int]]:
        """Port of ``DiffSingerUtils.PaddedSegments``: (phoneme, ms, phone index)."""

        segments: list[tuple[str, float, int]] = [
            ("SP", self.head_frames * frame_ms, -1)
        ]
        for i, phone in enumerate(self.phones):
            if i > 0:
                gap_ms = phone.position_ms - self.phones[i - 1].end_ms
                if gap_ms > 0:
                    segments.append(("SP", gap_ms, -1))
            segments.append((phone.phoneme, phone.duration_ms, i))
        segments.append(("SP", self.tail_frames * frame_ms, -1))
        return segments

    def padded_phone_durations(self, frame_ms: float) -> list[int]:
        segments = self.padded_segments(frame_ms)
        return durations_ms_to_frames([s[1] for s in segments], frame_ms)

    def padded_voiced_mask(
        self, segments: list[tuple[str, float, int]], durations: list[int]
    ) -> list[bool]:
        total = sum(durations)
        mask = [False] * total
        frame = 0
        for i in range(min(len(segments), len(durations))):
            voiced = segments[i][2] >= 0
            for _ in range(durations[i]):
                if frame >= total:
                    break
                mask[frame] = voiced
                frame += 1
        return mask


def durations_ms_to_frames(durations_ms: list[float], frame_ms: float) -> list[int]:
    """Port of ``DiffSingerUtils.DurationsMsToFrames`` (round-half-even)."""

    result: list[int] = []
    accumulated = 0.0
    previous = 0
    for duration in durations_ms:
        if duration < 0:
            raise ValueError(f"Negative DiffSinger duration: {duration}")
        accumulated += duration
        # C# `Math.Round(x, MidpointRounding.ToEven)` == Python's banker's round.
        frame = int(round(accumulated / frame_ms + 0.5))
        result.append(frame - previous)
        previous = frame
    return result


def build_phrase(
    project: UstxProject,
    phrase_notes: list[UstxNote],
    phone_specs: list[tuple[str, int, int]],
    part_position: int = 0,
    head_frames: int = HEAD_FRAMES,
    tail_frames: int = TAIL_FRAMES,
) -> RenderPhrase:
    """Build a :class:`RenderPhrase`.

    ``phone_specs`` is a list of ``(phoneme, note_index, position_tick)`` with
    part-absolute ticks; ``note_index`` indexes ``phrase_notes``.
    """

    axis = project.time_axis
    notes: list[RenderNote] = []
    for note in phrase_notes:
        position = part_position + note.position
        end = part_position + note.end
        notes.append(
            RenderNote(
                lyric=note.lyric,
                tone=note.tone,
                adjusted_tone=float(note.tone),
                position=position,
                duration=note.duration,
                end=end,
                position_ms=axis.tick_to_ms(position),
                duration_ms=axis.tick_to_ms(end) - axis.tick_to_ms(position),
                end_ms=axis.tick_to_ms(end),
                pitch_points=list(note.pitch_points),
                vibrato=dict(note.vibrato),
            )
        )

    if not phone_specs:
        raise ValueError("Phrase has no phonemes")

    specs = sorted(phone_specs, key=lambda s: s[2])
    last_note_end = max(n.end for n in notes)
    phones: list[RenderPhone] = []
    for i, (phoneme, note_index, tick) in enumerate(specs):
        nxt = specs[i + 1][2] if i + 1 < len(specs) else max(last_note_end, tick + 1)
        end = max(tick + 1, nxt)
        phones.append(
            RenderPhone(
                phoneme=phoneme,
                note_index=note_index,
                position=tick,
                duration=end - tick,
                end=end,
                position_ms=axis.tick_to_ms(tick),
                duration_ms=axis.tick_to_ms(end) - axis.tick_to_ms(tick),
                end_ms=axis.tick_to_ms(end),
            )
        )
    return RenderPhrase(project, notes, phones, head_frames, tail_frames)
