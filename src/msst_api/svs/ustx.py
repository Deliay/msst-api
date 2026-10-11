"""OpenUtau USTX score reading/writing.

The whole file is kept as a plain nested dict (so unknown/newer fields survive a
round-trip); this module only extracts the pieces the renderer needs -- tempo
map, notes, expression curves -- and can inject a generated ``pitd`` curve back
into the project.

Supports both the 0.7 layout (notes live in ``voice_parts``) and the older
layout (``tracks[].parts[]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .time_axis import TimeAxis

CURVE_INTERVAL = 5


@dataclass
class PitchPoint:
    x: float
    y: float
    shape: str = "io"


@dataclass
class UstxNote:
    position: int
    duration: int
    tone: int
    lyric: str
    pitch_points: list[PitchPoint] = field(default_factory=list)
    vibrato: dict = field(default_factory=dict)

    @property
    def end(self) -> int:
        return self.position + self.duration

    @property
    def is_slur(self) -> bool:
        return self.lyric.startswith("+")

    @classmethod
    def from_dict(cls, data: dict) -> "UstxNote":
        raw_points = ((data.get("pitch") or {}).get("data")) or []
        points = [
            PitchPoint(
                float(p.get("x", 0.0)),
                float(p.get("y", 0.0)),
                str(p.get("shape", "io")),
            )
            for p in raw_points
        ]
        return cls(
            position=int(data.get("position", 0)),
            duration=int(data.get("duration", 0)),
            tone=int(data.get("tone", 60)),
            lyric=str(data.get("lyric", "") or ""),
            pitch_points=points,
            vibrato=dict(data.get("vibrato") or {}),
        )


def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


class Curve:
    """A sampled expression curve, piecewise between its points."""

    def __init__(
        self,
        abbr: str,
        points: list[PitchPoint],
        default: float = 0.0,
    ) -> None:
        self.abbr = abbr
        self.points = points
        self.default = default

    @property
    def is_empty(self) -> bool:
        return not self.points

    def sample(self, tick: float) -> float:
        pts = self.points
        if not pts:
            return self.default
        if tick <= pts[0].x:
            return pts[0].y
        if tick >= pts[-1].x:
            return pts[-1].y
        lo, hi = 0, len(pts) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if pts[mid].x <= tick:
                lo = mid
            else:
                hi = mid
        p1, p2 = pts[lo], pts[hi]
        span = p2.x - p1.x
        if span <= 0:
            return p2.y
        t = (tick - p1.x) / span
        shape = p1.shape if p1.shape else "io"
        if shape in ("l", "lin", "linear"):
            return p1.y + (p2.y - p1.y) * t
        if shape == "i":
            return p1.y + (p2.y - p1.y) * (t * t)
        if shape == "o":
            return p1.y + (p2.y - p1.y) * (1.0 - (1.0 - t) * (1.0 - t))
        if shape == "sp":
            p0 = pts[lo - 1] if lo - 1 >= 0 else p1
            p3 = pts[hi + 1] if hi + 1 < len(pts) else p2
            t2, t3 = t * t, t * t * t
            return 0.5 * (
                (2 * p1.y)
                + (-p0.y + p2.y) * t
                + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2
                + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3
            )
        return p1.y + (p2.y - p1.y) * _smoothstep(t)


class UstxProject:
    def __init__(self, raw: dict, path: Path | None = None) -> None:
        self.raw = raw
        self.path = path
        self.resolution = int(raw.get("resolution", 480))
        self.bpm = float(raw.get("bpm", 120.0))
        tempos = raw.get("tempos") or [{"position": 0, "bpm": self.bpm}]
        self.time_axis = TimeAxis(
            [(int(t["position"]), float(t["bpm"])) for t in tempos],
            self.resolution,
        )
        self.expressions: dict = dict(raw.get("expressions") or {})

        self._voice_part = self._find_voice_part(raw)
        self.part_position = int(self._voice_part.get("position", 0) or 0)
        notes = self._voice_part.get("notes") or []
        self.notes: list[UstxNote] = [UstxNote.from_dict(n) for n in notes]

        self.track = (raw.get("tracks") or [{}])[0]
        self.phonemizer = str(self.track.get("phonemizer") or "")
        self.renderer_settings = dict(self.track.get("renderer_settings") or {})

        self.curves: dict[str, Curve] = self._parse_curves(self._voice_part)
        self.warnings: list[str] = []

    # -- parsing -----------------------------------------------------------
    @staticmethod
    def _find_voice_part(raw: dict) -> dict:
        parts = raw.get("voice_parts")
        if parts:
            return parts[0]
        for track in raw.get("tracks") or []:
            if track.get("parts"):
                return track["parts"][0]
        return {}

    def _parse_curves(self, part: dict) -> dict[str, Curve]:
        curves: dict[str, Curve] = {}
        for entry in part.get("curves") or []:
            abbr = str(entry.get("abbr", ""))
            if not abbr:
                continue
            xs = entry.get("xs") or []
            ys = entry.get("ys") or []
            points = [PitchPoint(float(x), float(y)) for x, y in zip(xs, ys)]
            curves[abbr] = Curve(abbr, points, self._curve_default(abbr))
        return curves

    def _curve_default(self, abbr: str) -> float:
        descriptor = self.expressions.get(abbr) or {}
        try:
            return float(descriptor.get("default_value", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def curve(self, abbr: str) -> Curve:
        existing = self.curves.get(abbr)
        if existing is not None:
            return existing
        return Curve(abbr, [], self._curve_default(abbr))

    def expression_range(self, abbr: str) -> tuple[float, float]:
        descriptor = self.expressions.get(abbr) or {}
        try:
            return float(descriptor.get("min", -1e9)), float(descriptor.get("max", 1e9))
        except (TypeError, ValueError):
            return -1e9, 1e9

    # -- writing -----------------------------------------------------------
    def set_pitd(self, points: list[PitchPoint]) -> None:
        """Replace the part's ``pitd`` curve (adding one if absent)."""

        curves = self._voice_part.setdefault("curves", [])
        curves[:] = [c for c in curves if str(c.get("abbr", "")) != "pitd"]
        if points:
            curves.append(
                {
                    "abbr": "pitd",
                    "xs": [int(round(p.x)) for p in points],
                    "ys": [int(round(p.y)) for p in points],
                }
            )
        # If the project has no pitd expression descriptor, add one so the
        # generated curve is understood by OpenUtau/UtaFormatix.
        if "pitd" not in self.expressions:
            self.expressions["pitd"] = {
                "name": "pitch deviation",
                "abbr": "pitd",
                "type": "Curve",
                "min": -1200,
                "max": 1200,
                "default_value": 0,
                "is_flag": False,
                "flag": "",
            }
        self.raw["expressions"] = self.expressions
        self.curves["pitd"] = Curve("pitd", list(points), 0.0)

    def to_bytes(self) -> bytes:
        return yaml.safe_dump(
            self.raw, allow_unicode=True, sort_keys=False
        ).encode("utf-8")

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_bytes())

    # -- helpers -----------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "UstxProject":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Not a USTX file (expected a YAML mapping)")
        return cls(raw, path)

    def note_groups(self) -> list[list[UstxNote]]:
        """Group notes into (syllable note + following slur notes)."""

        groups: list[list[UstxNote]] = []
        for note in self.notes:
            if note.is_slur and groups:
                groups[-1].append(note)
            else:
                groups.append([note])
        return groups
