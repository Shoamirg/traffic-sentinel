"""Time-segment helpers: boolean flags -> segments, merging, and final cleanup."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass(frozen=True)
class Event:
    start: float
    end: float
    label: str
    score: float = 1.0
    tids: tuple[int, ...] = ()          # tracks involved (for review / rendering only)

    def as_list(self) -> list:
        return [round(float(self.start), 2), round(float(self.end), 2), self.label]


def runs(times: np.ndarray, flags: np.ndarray, max_gap: float = config.MERGE_GAP_SEC) -> list[tuple[float, float]]:
    """Maximal runs of True in `flags` sampled at `times`, bridging gaps shorter than max_gap."""
    out: list[list[float]] = []
    for t in times[np.asarray(flags, bool)]:
        if out and t - out[-1][1] <= max_gap:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [(s, e) for s, e in out]


def merge_segments(segs: list[tuple[float, float]], max_gap: float) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for s, e in sorted(segs):
        if merged and s - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def finalize(events: list[Event], duration: float, merge_gap: dict[str, float] | None = None,
             min_len: dict[str, float] | None = None) -> list[Event]:
    """Per class: merge overlaps and small gaps, drop blips, clip to the video."""
    merge_gap = merge_gap or {}
    min_len = min_len or {}
    out: list[Event] = []
    for label in sorted({e.label for e in events}):
        mine = [e for e in events if e.label == label]
        segs = [(max(0.0, e.start), min(duration, e.end)) for e in mine]
        for s, e in merge_segments(segs, merge_gap.get(label, config.MERGE_GAP_SEC)):
            if e - s >= min_len.get(label, config.MIN_EVENT_SEC):
                tids = sorted({t for ev in mine if ev.start <= e and ev.end >= s for t in ev.tids})
                out.append(Event(s, e, label, tids=tuple(tids)))
    return sorted(out, key=lambda ev: (ev.start, ev.label))
