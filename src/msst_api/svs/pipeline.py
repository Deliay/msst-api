"""``ustx -> pitd + ustx -> wav`` orchestration.

The flow mirrors OpenUtau:

1. romanize + phonemize the score and predict phoneme durations (``dsdur``);
2. run the auto-pitch model (``dspitch``) phrase by phrase and write the result
   back into the project as a ``pitd`` (pitch deviation) curve -- the
   "pitd+ustx" step;
3. rebuild the phrases (now carrying ``pitd``) and render each with variance +
   acoustic + vocoder, placing the phrase audio on the song timeline.

The intermediate ``.ustx`` (with ``pitd``) is returned alongside the wave so the
caller can inspect or re-edit the score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from . import SVSError
from .phrase import PITCH_INTERVAL, build_phrase
from .phonemizer import DiffSingerPhonemizer
from .render import predict_pitch, render_phrase
from .romanize import has_hanzi, romanize_chinese, romanize_japanese
from .ustx import PitchPoint, UstxNote, UstxProject
from .voicebank import DiffSingerVoicebank

logger = logging.getLogger("msst_api.svs")

SUPPORTED_LANGUAGES = ("ja", "zh")


@dataclass
class SVSRequest:
    language: str = "auto"
    #: Sampling steps for the auto-pitch model.
    pitch_steps: int = 20
    #: Sampling steps for the variance model.
    variance_steps: int = 20
    #: Diffusion depth for the acoustic model (clamped by ``dsconfig.max_depth``).
    depth: float = 1.0
    #: Sampling steps for the acoustic model.
    steps: int = 20
    #: When False the auto-pitch step is skipped and the score's own ``pitd``
    #: (if any) is used.
    auto_pitch: bool = True
    #: When False only the auto-pitch step runs (returns the ``pitd`` USTX with
    #: no audio).
    render: bool = True


@dataclass
class SVSResult:
    samples: np.ndarray
    sample_rate: int
    ustx_bytes: bytes
    language: str
    phonemizer: str
    phrases: int
    notes: int
    duration_seconds: float
    warnings: list[str] = field(default_factory=list)


def _romanize(lyrics: list[str], language: str) -> list[str]:
    if language == "ja":
        return romanize_japanese(lyrics)
    if language == "zh":
        return romanize_chinese(lyrics)
    return list(lyrics)


def detect_language(
    voicebank: DiffSingerVoicebank, note_groups: list[list[UstxNote]]
) -> str:
    """Pick the voicebank language whose g2p dictionary covers the lyrics best."""

    candidates = [lang for lang in SUPPORTED_LANGUAGES]
    main_lyrics = [group[0].lyric for group in note_groups if group[0].lyric]
    # Hanzi lyrics can only be Chinese.
    if any(has_hanzi(lyric) for lyric in main_lyrics):
        return "zh"
    scores: dict[str, int] = {}
    for lang in candidates:
        dict_name = f"dsdict-{lang}.yaml"
        try:
            g2p = (
                voicebank.dur.g2p(dict_name)
                if voicebank.dur is not None
                else None
            )
        except Exception:  # noqa: BLE001 - missing dictionary
            g2p = None
        if g2p is None:
            continue
        romanized = _romanize(main_lyrics, lang)
        hits = sum(
            1
            for lyric in romanized
            if g2p.query_ci(lyric) is not None and lyric
        )
        scores[lang] = hits
    if not scores:
        return "ja"
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "ja"
    return best


def _split_phrases(groups: list[list[UstxNote]]) -> list[list[list[UstxNote]]]:
    phrases: list[list[list[UstxNote]]] = []
    current: list[list[UstxNote]] = [groups[0]]
    for group in groups[1:]:
        prev_last = current[-1][-1]
        if prev_last.position + prev_last.duration == group[0].position:
            current.append(group)
        else:
            phrases.append(current)
            current = [group]
    phrases.append(current)
    return phrases


def _phone_specs(
    phonemizer: DiffSingerPhonemizer,
    project: UstxProject,
    phrase_groups: list[list[UstxNote]],
) -> tuple[list[tuple[str, int, int]], list[UstxNote]]:
    """Return ``(phone_specs, phrase_notes)``.

    ``phone_specs`` items are ``(phoneme, note_index, part-absolute tick)``.
    """

    placements = phonemizer.process_phrase_with_axis(phrase_groups, project.time_axis)
    phrase_notes: list[UstxNote] = []
    note_index_of: dict[int, int] = {}
    for group in phrase_groups:
        for note in group:
            note_index_of[note.position] = len(phrase_notes)
            phrase_notes.append(note)

    specs: list[tuple[str, int, int]] = []
    for group in phrase_groups:
        main = group[0]
        note_index = note_index_of.get(main.position, 0)
        for phoneme, offset in placements.get(main.position, []):
            specs.append(
                (phoneme, note_index, project.part_position + main.position + offset)
            )
    return specs, phrase_notes


def _pitd_points(
    phrase, result, min_pitd: float
) -> list[PitchPoint]:
    ticks = result.ticks
    tones = result.tones
    voiced = result.voiced
    points: list[PitchPoint] = []
    for i, (tick, tone) in enumerate(zip(ticks, tones)):
        if tone < 0:
            continue
        if voiced is not None and i < len(voiced) and not voiced[i]:
            continue
        if tick < 0 and (i + 1 >= len(ticks) or ticks[i + 1] <= 0):
            continue
        x = phrase.position_tick + int(tick)
        pitch_index = int((x - phrase.position_tick) / PITCH_INTERVAL)
        pitch_index = max(0, min(pitch_index, len(phrase.pitches_before_deviation) - 1))
        base = phrase.pitches_before_deviation[pitch_index]
        y = tone * 100.0 - base
        if y <= min_pitd:
            continue
        points.append(PitchPoint(float(x), y))
    return points


def run_svs_pipeline(
    voicebank: DiffSingerVoicebank,
    ustx_path: str,
    request: SVSRequest | None = None,
) -> SVSResult:
    request = request or SVSRequest()
    project = UstxProject.load(ustx_path)
    if not project.notes:
        raise SVSError("The USTX contains no notes")

    # --- romanize + group + split ----------------------------------------
    groups = project.note_groups()
    language = request.language.strip().lower()
    if language in ("", "auto"):
        language = detect_language(voicebank, groups)
    if language not in SUPPORTED_LANGUAGES:
        raise SVSError(
            f"Unsupported language {language!r}. Expected one of {SUPPORTED_LANGUAGES}"
        )

    romanized = _romanize([g[0].lyric for g in groups], language)
    for group, lyric in zip(groups, romanized):
        group[0].lyric = lyric or "a"
    if language == "zh":
        unresolved = [g[0].lyric for g in groups if has_hanzi(g[0].lyric)]
        if unresolved:
            raise SVSError(
                "Chinese lyrics contain hanzi but pypinyin is not installed; "
                "provide pinyin lyrics or install pypinyin."
            )

    if voicebank.dur is None:
        raise SVSError("This voicebank has no duration model (dsdur/)")
    phonemizer = DiffSingerPhonemizer(voicebank.dur, language)
    phrases = _split_phrases(groups)

    # Phonemize each sentence once: the auto-pitch and render stages consume the
    # same alignment.  A sentence whose notes resolve to no phonemes (all rests
    # / orphan slurs, or lyrics the voicebank's dictionary does not cover) cannot
    # be built into a phrase, so skip it instead of aborting the whole request.
    warnings: list[str] = []
    prepared: list[tuple[list[UstxNote], list[tuple[str, int, int]]]] = []
    skipped_phrases = 0
    for phrase_groups in phrases:
        specs, phrase_notes = _phone_specs(phonemizer, project, phrase_groups)
        if not specs:
            skipped_phrases += 1
            continue
        prepared.append((phrase_notes, specs))
    if skipped_phrases:
        warnings.append(
            f"Skipped {skipped_phrases} phrase(s) with no phonemes (rests, "
            "orphan slurs, or lyrics missing from the voicebank dictionary)."
        )
    if not prepared:
        raise SVSError(
            "No phonemes could be generated for any phrase; check the note "
            "lyrics and that the voicebank language matches them."
        )

    # --- auto pitch -------------------------------------------------------
    all_points: list[PitchPoint] = []
    min_pitd, _ = project.expression_range("pitd")
    if request.auto_pitch and voicebank.has_pitch:
        for phrase_notes, specs in prepared:
            render_phrase_obj = build_phrase(
                project, phrase_notes, specs, project.part_position
            )
            result = predict_pitch(voicebank, render_phrase_obj, request.pitch_steps)
            all_points.extend(_pitd_points(render_phrase_obj, result, min_pitd))
        all_points.sort(key=lambda p: p.x)
        project.set_pitd(all_points)

    if not request.render:
        return SVSResult(
            samples=np.zeros(0, dtype=np.float32),
            sample_rate=voicebank.sample_rate,
            ustx_bytes=project.to_bytes(),
            language=language,
            phonemizer=f"DiffSinger{language.capitalize()}Phonemizer",
            phrases=len(phrases),
            notes=len(project.notes),
            duration_seconds=0.0,
            warnings=warnings,
        )

    # --- render -----------------------------------------------------------
    rendered: list[tuple[float, np.ndarray]] = []
    for phrase_notes, specs in prepared:
        render_phrase_obj = build_phrase(
            project, phrase_notes, specs, project.part_position
        )
        samples = render_phrase(
            voicebank,
            render_phrase_obj,
            depth=request.depth,
            steps=request.steps,
            variance_steps=request.variance_steps,
        )
        rendered.append((render_phrase_obj.position_ms / 1000.0, samples))

    samples = _assemble(rendered, voicebank.sample_rate)
    duration = samples.size / voicebank.sample_rate
    return SVSResult(
        samples=samples,
        sample_rate=voicebank.sample_rate,
        ustx_bytes=project.to_bytes(),
        language=language,
        phonemizer=f"DiffSinger{language.capitalize()}Phonemizer",
        phrases=len(phrases),
        notes=len(project.notes),
        duration_seconds=duration,
        warnings=warnings,
    )


def _assemble(rendered: list[tuple[float, np.ndarray]], sample_rate: int) -> np.ndarray:
    if not rendered:
        return np.zeros(0, dtype=np.float32)
    starts = [start for start, _ in rendered]
    min_start = min(starts)
    end = max(start + clip.size / sample_rate for start, clip in rendered)
    total = int(round((end - min_start) * sample_rate)) + 1
    buffer = np.zeros(max(total, 1), dtype=np.float32)
    for start, clip in rendered:
        offset = int(round((start - min_start) * sample_rate))
        if offset < 0:
            clip = clip[-offset:]
            offset = 0
        end_index = min(buffer.size, offset + clip.size)
        if end_index > offset:
            buffer[offset:end_index] += clip[: end_index - offset]
    return buffer
