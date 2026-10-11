"""The DiffSinger front-end: lyric -> phoneme -> predicted phoneme durations.

This is the port of ``OpenUtau.Core.DiffSinger.DiffSingerBasePhonemizer`` (the
``dsdur`` models).  It does three things per sentence (a run of notes with no
rest between them):

1. **Phonemize** -- each note's lyric is looked up in the language g2p
   dictionary (``dsdict-<lang>.yaml``) to get its phoneme symbols, and the
   symbols are distributed to the notes of the syllable.
2. **Duration prediction** -- the linguistic encoder (``word_div``/``word_dur``
   mode) plus the duration model predict a frame count per phoneme.
3. **Alignment** -- the predicted phoneme durations are stretched so each word
   ends at its note boundary, yielding a tick offset per phoneme relative to its
   note start.

The result is ``{note_position_tick: [(phoneme, offset_tick), ...]}`` which the
pipeline turns into contiguous :class:`~msst_api.svs.phrase.RenderPhone`s.
"""

from __future__ import annotations

import numpy as np

from .onnx_util import run_session
from .voicebank import SubModel

DEFAULT_PAUSE = "SP"
PADDING_MS = 500.0


class DiffSingerPhonemizer:
    def __init__(self, submodel: SubModel, language: str = "") -> None:
        self.sm = submodel
        self.language = language
        self.config = submodel.config
        self.frame_ms = submodel.frame_ms
        self.phoneme_tokens = submodel.phoneme_tokens
        dict_name = f"dsdict-{language}.yaml" if language else None
        self.g2p = submodel.g2p(dict_name)
        self._lang_ids = submodel.language_ids

    # -- symbol handling ---------------------------------------------------
    def _validate_symbol(self, symbol: str) -> str:
        if self.g2p.is_valid_symbol(symbol) and symbol in self.phoneme_tokens:
            return symbol
        if self.language:
            prefixed = f"{self.language}/{symbol}"
            if self.g2p.is_valid_symbol(prefixed) and prefixed in self.phoneme_tokens:
                return prefixed
        return ""

    def _parse_hint(self, hint: str) -> list[str]:
        out: list[str] = []
        for token in hint.split():
            validated = self._validate_symbol(token)
            if validated:
                out.append(validated)
        return out

    def get_symbols(self, note) -> list[str]:
        symbols = self.g2p.query_ci(note.lyric)
        if symbols is not None:
            return [s for s in symbols if self.g2p.is_valid_symbol(s)]
        # Not in the dictionary: treat the lyric as a phonetic hint.
        return self._parse_hint(note.lyric)

    # -- word distribution -------------------------------------------------
    @staticmethod
    def _is_vowel_extension(note) -> bool:
        return note.lyric.startswith("+~") or note.lyric.startswith("+*")

    def _process_word(self, notes, symbols: list[str]) -> list[list]:
        for symbol in symbols:
            if not self.g2p.is_valid_symbol(symbol):
                raise ValueError(
                    f"Type definition of symbol {symbol!r} not found in the dictionary"
                )
        # Each entry is [position_tick, tone, [phoneme, ...]].
        word_phonemes: list[list] = [[-1, notes[0].tone, []]]
        is_vowel = [self.g2p.is_vowel(s) for s in symbols]
        is_glide = [self.g2p.is_glide(s) for s in symbols]
        non_extension = [n for n in notes if not self._is_vowel_extension(n)]
        is_start = [False] * len(symbols)
        if not any(is_vowel):
            is_start[0] = True
        for i in range(len(symbols)):
            if is_vowel[i]:
                if i >= 2 and is_glide[i - 1] and not is_vowel[i - 2]:
                    is_start[i - 1] = True
                else:
                    is_start[i] = True
        note_index = 0
        for i, symbol in enumerate(symbols):
            if is_start[i] and note_index < len(non_extension):
                note = non_extension[note_index]
                word_phonemes.append([note.position, note.tone, []])
                note_index += 1
            word_phonemes[-1][2].append(symbol)
        return word_phonemes

    # -- sentence processing ----------------------------------------------
    def process_phrase_with_axis(self, phrase_groups, time_axis):
        frame_ms = self.frame_ms
        last_note = phrase_groups[-1][-1]
        end_tick = last_note.position + last_note.duration

        groups: list[list] = [[-1, phrase_groups[0][0].tone, ["SP"]]]
        note_ph_index = [1]
        word_found: list[bool] = []
        for word in phrase_groups:
            raw_symbols = self.get_symbols(word[0])
            symbols = [s for s in raw_symbols if s in self.phoneme_tokens]
            if not symbols:
                symbols = [DEFAULT_PAUSE]
                word_found.append(False)
            else:
                word_found.append(True)
            word_phonemes = self._process_word(word, symbols)
            groups[-1][2].extend(word_phonemes[0][2])
            groups.extend(word_phonemes[1:])
            note_ph_index.append(
                note_ph_index[-1]
                + sum(len(group[2]) for group in word_phonemes)
            )
        groups.append([end_tick, last_note.tone, []])
        groups[0][0] = time_axis.ms_to_tick(
            time_axis.tick_to_ms(groups[1][0]) - PADDING_MS
        )

        # -- linguistic encoder -------------------------------------------
        tokens = [self.phoneme_tokens[p] for group in groups for p in group[2]]
        word_div = [len(group[2]) for group in groups[:-1]]
        word_dur = [
            self._frames_between(time_axis, a[0], b[0])
            for a, b in zip(groups, groups[1:])
        ]

        linguistic_feed = {
            "tokens": np.asarray([tokens], dtype=np.int64),
            "word_div": np.asarray([word_div], dtype=np.int64),
            "word_dur": np.asarray([word_dur], dtype=np.int64),
        }
        if self.config.use_lang_id:
            lang_ids = [
                int(self._lang_ids.get(_phoneme_language(p), 0))
                for group in groups
                for p in group[2]
            ]
            linguistic_feed["languages"] = np.asarray([lang_ids], dtype=np.int64)

        linguistic_out = run_session(self.sm.linguistic, linguistic_feed)
        encoder_out = linguistic_out["encoder_out"]
        x_masks = linguistic_out["x_masks"]

        ph_midi = [group[1] for group in groups for _ in group[2]]
        duration_feed = {
            "encoder_out": encoder_out,
            "x_masks": x_masks,
            "ph_midi": np.asarray([ph_midi], dtype=np.int64),
        }
        duration_out = run_session(self.sm.model, duration_feed)
        first_output = next(iter(duration_out.values()))
        duration_frames = [float(x) for x in np.asarray(first_output).reshape(-1)]

        positions = self._align(groups, duration_frames, frame_ms, time_axis)

        # -- map back to notes --------------------------------------------
        phs = [p for group in groups for p in group[2]]
        result: dict[int, list[tuple[str, int]]] = {}
        for word_index, word in enumerate(phrase_groups):
            if not word_found[word_index]:
                result[word[0].position] = []
                continue
            if word[0].lyric.startswith("+"):
                continue
            note_pos_ms = time_axis.tick_to_ms(word[0].position)
            note_result: list[tuple[str, int]] = []
            for ph_index in range(
                note_ph_index[word_index], note_ph_index[word_index + 1]
            ):
                symbol = phs[ph_index]
                if symbol:
                    offset = time_axis.ms_to_tick(positions[ph_index - 1]) - time_axis.ms_to_tick(note_pos_ms)
                    note_result.append((symbol, offset))
            result[word[0].position] = note_result
        return result

    # -- helpers -----------------------------------------------------------
    def _frames_between(self, time_axis, tick1: int, tick2: int) -> int:
        return int(time_axis.tick_to_ms(tick2) / self.frame_ms) - int(
            time_axis.tick_to_ms(tick1) / self.frame_ms
        )

    @staticmethod
    def _align(groups, duration_frames, frame_ms, time_axis):
        cumsum = 0
        counts = []
        for group in groups:
            cumsum += len(group[2])
            counts.append(cumsum)
        ph_align = list(
            zip(counts, [time_axis.tick_to_ms(group[0]) for group in groups[1:]])
        )
        positions: list[float] = []
        align_group = duration_frames[1 : ph_align[0][0]]
        positions.extend(_stretch(align_group, frame_ms, ph_align[0][1]))
        for current, following in zip(ph_align, ph_align[1:]):
            group = duration_frames[current[0] : following[0]]
            group_sum = sum(group)
            ratio = (following[1] - current[1]) / group_sum if group_sum else 0.0
            positions.extend(_stretch(group, ratio, following[1]))
        return positions


def _stretch(source: list[float], ratio: float, end_pos: float) -> list[float]:
    if not source:
        return []
    start = end_pos - sum(source) * ratio
    result = [start]
    acc = start
    for value in source:
        acc += value * ratio
        result.append(acc)
    result.pop()
    return result


def _phoneme_language(phoneme: str) -> str:
    return phoneme.split("/", 1)[0] if "/" in phoneme else ""
