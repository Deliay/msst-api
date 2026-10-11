"""Lyric romanization for the DiffSinger phonemizers.

Ports the two front-end romanizers OpenUtau uses:

* :func:`romanize_japanese` -- kana -> romaji (``DiffSingerJapanesePhonemizer``
  uses ``Kana.KanaToRomaji``); the trailing ``ん`` becomes ``N``.
* :func:`romanize_chinese` -- hanzi -> pinyin (``BaseChinesePhonemizer`` uses
  the bundled Pinyin table); we use ``pypinyin`` when installed and otherwise
  leave the lyric untouched so already-romanized scores (e.g. from v2m) work
  without the extra dependency.
"""

from __future__ import annotations

# Hepburn-ish kana -> romaji, matching the ``dsdict-ja.yaml`` graphemes
# (sha/cha/tsu/shi/chi/ja...), digraphs first so the longest match wins.
_KANA: dict[str, str] = {}


def _add(romaji: str, *kana: str) -> None:
    for k in kana:
        _KANA[k] = romaji


# Digraphs (youon)
_add("kya", "きゃ", "キャ"); _add("kyu", "きゅ", "キュ"); _add("kyo", "きょ", "キョ")
_add("gya", "ぎゃ", "ギャ"); _add("gyu", "ぎゅ", "ギュ"); _add("gyo", "ぎょ", "ギョ")
_add("sha", "しゃ", "シャ"); _add("shu", "しゅ", "シュ"); _add("sho", "しょ", "ショ")
_add("ja", "じゃ", "ジャ", "ぢゃ", "ヂャ"); _add("ju", "じゅ", "ジュ", "ぢゅ", "ヂュ"); _add("jo", "じょ", "ジョ", "ぢょ", "ヂョ")
_add("cha", "ちゃ", "チャ"); _add("chu", "ちゅ", "チュ"); _add("cho", "ちょ", "チョ")
_add("nya", "にゃ", "ニャ"); _add("nyu", "にゅ", "ニュ"); _add("nyo", "にょ", "ニョ")
_add("hya", "ひゃ", "ヒャ"); _add("hyu", "ひゅ", "ヒュ"); _add("hyo", "ひょ", "ヒョ")
_add("bya", "びゃ", "ビャ"); _add("byu", "びゅ", "ビュ"); _add("byo", "びょ", "ビョ")
_add("pya", "ぴゃ", "ピャ"); _add("pyu", "ぴゅ", "ピュ"); _add("pyo", "ぴょ", "ピョ")
_add("mya", "みゃ", "ミャ"); _add("myu", "みゅ", "ミュ"); _add("myo", "みょ", "ミョ")
_add("rya", "りゃ", "リャ"); _add("ryu", "りゅ", "リュ"); _add("ryo", "りょ", "リョ")
_add("dya", "でゃ", "デャ"); _add("dyu", "でゅ", "デュ"); _add("dyo", "でょ", "デョ")
_add("tya", "てゃ", "テャ"); _add("tyu", "てゅ", "テュ"); _add("tyo", "てょ", "テョ")
# Monographs
_add("a", "あ", "ア"); _add("i", "い", "イ"); _add("u", "う", "ウ", "ゔ", "ヴ")
_add("e", "え", "エ"); _add("o", "お", "オ")
_add("ka", "か", "カ"); _add("ki", "き", "キ"); _add("ku", "く", "ク"); _add("ke", "け", "ケ"); _add("ko", "こ", "コ")
_add("ga", "が", "ガ"); _add("gi", "ぎ", "ギ"); _add("gu", "ぐ", "グ"); _add("ge", "げ", "ゲ"); _add("go", "ご", "ゴ")
_add("sa", "さ", "サ"); _add("shi", "し", "シ"); _add("su", "す", "ス"); _add("se", "せ", "セ"); _add("so", "そ", "ソ")
_add("za", "ざ", "ザ"); _add("ji", "じ", "ジ", "ぢ", "ヂ"); _add("zu", "ず", "ズ", "づ", "ヅ"); _add("ze", "ぜ", "ゼ"); _add("zo", "ぞ", "ゾ")
_add("ta", "た", "タ"); _add("chi", "ち", "チ"); _add("tsu", "つ", "ツ"); _add("te", "て", "テ"); _add("to", "と", "ト")
_add("da", "だ", "ダ"); _add("di", "でぃ", "ディ"); _add("du", "どぅ", "ドゥ"); _add("de", "で", "デ"); _add("do", "ど", "ド")
_add("na", "な", "ナ"); _add("ni", "に", "ニ"); _add("nu", "ぬ", "ヌ"); _add("ne", "ね", "ネ"); _add("no", "の", "ノ")
_add("ha", "は", "ハ"); _add("hi", "ひ", "ヒ"); _add("fu", "ふ", "フ"); _add("he", "へ", "ヘ"); _add("ho", "ほ", "ホ")
_add("ba", "ば", "バ"); _add("bi", "び", "ビ"); _add("bu", "ぶ", "ブ"); _add("be", "べ", "ベ"); _add("bo", "ぼ", "ボ")
_add("pa", "ぱ", "パ"); _add("pi", "ぴ", "ピ"); _add("pu", "ぷ", "プ"); _add("pe", "ぺ", "ペ"); _add("po", "ぽ", "ポ")
_add("ma", "ま", "マ"); _add("mi", "み", "ミ"); _add("mu", "む", "ム"); _add("me", "め", "メ"); _add("mo", "も", "モ")
_add("ya", "や", "ヤ"); _add("yu", "ゆ", "ユ"); _add("yo", "よ", "ヨ")
_add("ra", "ら", "ラ"); _add("ri", "り", "リ"); _add("ru", "る", "ル"); _add("re", "れ", "レ"); _add("ro", "ろ", "ロ")
_add("wa", "わ", "ワ"); _add("wo", "を", "ヲ")
_add("fa", "ふぁ", "ファ"); _add("fi", "ふぃ", "フィ"); _add("fe", "ふぇ", "フェ"); _add("fo", "ふぉ", "フォ")
_add("va", "ゔぁ", "ヴァ", "ば", "バ"); _add("vi", "ゔぃ", "ヴィ"); _add("ve", "ゔぇ", "ヴェ"); _add("vo", "ゔぉ", "ヴォ")
_add("ye", "いぇ", "イェ"); _add("wi", "うぃ", "ウィ"); _add("we", "うぇ", "ウェ")
_add("kwa", "くぁ", "クァ"); _add("kwi", "くぃ", "クィ"); _add("kwe", "くぇ", "クェ"); _add("kwo", "くぉ", "クォ")
_add("she", "しぇ", "シェ"); _add("che", "ちぇ", "チェ"); _add("je", "じぇ", "ジェ")
_add("tsa", "つぁ", "ツァ"); _add("tsi", "つぃ", "ツィ"); _add("tse", "つぇ", "ツェ"); _add("tso", "つぉ", "ツォ")
_add("ti", "てぃ", "ティ"); _add("tu", "とぅ", "トゥ")
_add("fyu", "ふゅ", "フュ"); _add("dyi", "でぃ", "ディ")

_SOKUON = {"っ", "ッ"}
_KANA_N = {"ん", "ン"}
_KANA_DOT = {"・", "･", "ー", "ｰ"}


def is_kana(text: str) -> bool:
    """True when ``text`` contains hiragana/katakana (ignoring long marks)."""

    return any(
        ("\u3040" <= ch <= "\u30ff") and ch not in _KANA_DOT for ch in text
    )


def kana_to_romaji(text: str) -> str:
    """Convert a kana string to Hepburn romaji (``ん`` -> ``N``).

    Unknown characters are dropped.  A small sokuon (``っ``) doubles the next
    consonant; a long mark repeats the previous vowel.
    """

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _SOKUON:
            nxt = None
            for length in (2, 1):
                if i + 1 + length <= n and text[i + 1 : i + 1 + length] in _KANA:
                    nxt = _KANA[text[i + 1 : i + 1 + length]]
                    break
            if nxt and nxt[0] not in "aeiou":
                out.append(nxt[0])
            elif nxt:
                out.append("t")
            i += 1
            continue
        if ch in _KANA_N:
            # ん is N, but before a vowel/y it is n' -> we emit N (OpenUtau does).
            out.append("N")
            i += 1
            continue
        if ch in _KANA_DOT:
            if out:
                out.append(out[-1][-1] if out[-1][-1] in "aeiou" else "")
            i += 1
            continue
        matched = False
        for length in (2, 1):
            piece = text[i : i + length]
            if piece in _KANA:
                out.append(_KANA[piece])
                i += length
                matched = True
                break
        if not matched:
            i += 1
    return "".join(out)


def romanize_japanese(lyrics: list[str]) -> list[str]:
    result = list(lyrics)
    for i, lyric in enumerate(result):
        if is_kana(lyric):
            if lyric == "ん":
                result[i] = "N"
            else:
                result[i] = kana_to_romaji(lyric)
    return result


def _has_hanzi(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def has_hanzi(text: str) -> bool:
    """True when ``text`` contains CJK ideographs."""

    return _has_hanzi(text)


def romanize_chinese(lyrics: list[str]) -> list[str]:
    """Hanzi -> toneless pinyin using ``pypinyin`` when available.

    Lyrics that are already ASCII (pinyin) are returned unchanged, so the
    pipeline works with or without the optional dependency.
    """

    try:
        from pypinyin import Style, lazy_pinyin  # type: ignore
    except Exception:  # noqa: BLE001 - optional dependency
        return list(lyrics)
    result: list[str] = []
    for lyric in lyrics:
        if _has_hanzi(lyric):
            syllables = lazy_pinyin(lyric, style=Style.NORMAL, errors="ignore")
            result.append("".join(syllables) if syllables else lyric)
        else:
            result.append(lyric)
    return result
