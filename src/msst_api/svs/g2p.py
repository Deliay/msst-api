"""DiffSinger g2p dictionary (``dsdict*.yaml``).

Port of ``OpenUtau.Core.Api.G2pDictionary``.  A dictionary holds two things:

* ``symbols`` -- every phoneme symbol with its type (``vowel`` / ``semivowel``
  / ``liquid`` / ``fricative`` ...).  The type decides the vowel/glide flags the
  model alignment uses.
* ``entries`` -- grapheme (a pinyin/romaji syllable) -> list of phoneme symbols.
  For voicebanks like the bundled one the symbols are already language
  prefixed (``zh/a``, ``ja/o``).

Entries are filtered at load time so that a grapheme only maps to symbols that
are declared, mirroring OpenUtau's ``BuildTrie``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_GLIDE_TYPES = {"semivowel", "liquid"}


class G2pDictionary:
    """A diff-singer grapheme-to-phoneme dictionary."""

    def __init__(self) -> None:
        self._symbols: dict[str, bool] = {}
        self._glides: set[str] = set()
        self._entries: dict[str, list[str]] = {}

    # -- construction ------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "G2pDictionary":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        dictionary = cls()
        for symbol in data.get("symbols") or []:
            dictionary.add_symbol(str(symbol.get("symbol")), symbol.get("type"))
        for entry in data.get("entries") or []:
            dictionary.add_entry(
                str(entry.get("grapheme")), list(entry.get("phonemes") or [])
            )
        return dictionary

    def add_symbol(self, symbol: str, symbol_type: str | None) -> None:
        self._symbols[symbol] = symbol_type == "vowel"
        if symbol_type in _GLIDE_TYPES:
            self._glides.add(symbol)
        else:
            self._glides.discard(symbol)

    def add_entry(self, grapheme: str, phonemes: list[str]) -> None:
        self._entries[grapheme] = [p for p in phonemes if p in self._symbols]

    # -- queries -----------------------------------------------------------
    def is_valid_symbol(self, symbol: str) -> bool:
        return symbol in self._symbols

    def is_vowel(self, symbol: str) -> bool:
        return self._symbols.get(symbol, False)

    def is_glide(self, symbol: str) -> bool:
        return symbol in self._glides

    def query(self, grapheme: str) -> list[str] | None:
        return self._entries.get(grapheme)

    def query_ci(self, grapheme: str) -> list[str] | None:
        """Lookup the grapheme, then its lower-cased form (OpenUtau's fallback)."""

        result = self.query(grapheme)
        if result is not None:
            return result
        if grapheme != grapheme.lower():
            return self.query(grapheme.lower())
        return None

    def symbols(self) -> list[str]:
        return list(self._symbols)
