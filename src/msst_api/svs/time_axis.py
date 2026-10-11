"""Piecewise-constant tempo map: tick <-> millisecond conversion.

OpenUtau stores time in ticks (``resolution`` ticks per beat) and tempos as a
list of ``{position, bpm}`` changes.  The DiffSinger models work in frames
(``hop_size / sample_rate``), so every stage converts ticks to ms and back.
"""

from __future__ import annotations

import bisect


class TimeAxis:
    def __init__(self, tempos: list[tuple[int, float]], resolution: int) -> None:
        if resolution <= 0:
            raise ValueError("USTX resolution must be positive")
        if not tempos:
            tempos = [(0, 120.0)]
        tempos = sorted((int(pos), float(bpm)) for pos, bpm in tempos)
        # Ensure a tempo exists at or before tick 0 so negative ticks resolve.
        if tempos[0][0] > 0:
            tempos.insert(0, (0, tempos[0][1]))
        self.resolution = int(resolution)
        self._positions = [pos for pos, _ in tempos]
        self._bpms = [bpm for _, bpm in tempos]
        # Cumulative milliseconds at each tempo position.
        self._ms = [0.0] * len(tempos)
        for i in range(1, len(tempos)):
            dt = self._positions[i] - self._positions[i - 1]
            self._ms[i] = self._ms[i - 1] + self._ms_per_tick(i - 1) * dt

    def _ms_per_tick(self, index: int) -> float:
        bpm = self._bpms[index]
        return 60000.0 / (bpm * self.resolution)

    def _segment(self, tick: float) -> int:
        i = bisect.bisect_right(self._positions, tick) - 1
        return max(0, min(i, len(self._positions) - 1))

    def tick_to_ms(self, tick: float) -> float:
        i = self._segment(tick)
        return self._ms[i] + (tick - self._positions[i]) * self._ms_per_tick(i)

    def ms_to_tick(self, ms: float) -> int:
        # Find the segment whose cumulative ms range contains ``ms``.  A linear
        # scan is fine: tempo changes are rare.
        i = 0
        for j in range(1, len(self._ms)):
            if ms >= self._ms[j]:
                i = j
            else:
                break
        return int(round(self._positions[i] + (ms - self._ms[i]) / self._ms_per_tick(i)))

    def ms_per_tick_at(self, tick: float) -> float:
        return self._ms_per_tick(self._segment(tick))
