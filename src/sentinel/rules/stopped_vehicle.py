"""stopped_vehicle: stationary on the carriageway >= 10 s, not queued at a signal.

Stationary intervals are collected from all vehicle tracks and chained across
tracker ID switches by location (a parked car that is briefly occluded keeps its
place). An interval counts if it lasts >= 10 s, is on observed road, and the
vehicle is not part of a queue (other stopped vehicles right next to it in a
habitual queue zone).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config
from ..segments import Event, runs
from . import Context

MIN_STOP_SEC = 10.0
STILL_SPEED_REL = 0.08     # speed / box size (sizes per second) below this = stationary
MOVING_REL = 0.4           # sizes/s: clearly driving
MOVING_BEFORE_SEC = 1.0    # it must have been seen driving this long before stopping (not parked)
CHAIN_GAP_SEC = 6.0
CHAIN_DIST_REL = 0.5       # same place = within half a box size
QUEUE_RADIUS_REL = 2.5
QUEUE_MIN_NEIGHBOURS = 1


@dataclass
class _Still:
    start: float
    end: float
    xy: np.ndarray
    size: float
    moved_before: bool
    tids: tuple[int, ...] = ()


def _still_intervals(ctx: Context) -> list[_Still]:
    out = []
    for tr in ctx.tracks.vehicles():
        rel = tr.speed / np.maximum(tr.size, 1.0)
        for s, e in runs(tr.ts, rel < STILL_SPEED_REL, max_gap=1.0):
            i, j = tr.at(s), tr.at(e)
            moved = float((rel[: i + 1] >= MOVING_REL).sum()) / config.PART_A_TARGET_FPS >= MOVING_BEFORE_SEC
            out.append(_Still(s, e, tr.xy[i:j + 1].mean(axis=0), float(np.median(tr.size[i:j + 1])), moved,
                              (tr.tid,)))
    return sorted(out, key=lambda x: x.start)


def _chain(items: list[_Still]) -> list[_Still]:
    chains: list[_Still] = []
    for it in items:
        for ch in chains:
            near = np.linalg.norm(ch.xy - it.xy) < CHAIN_DIST_REL * max(ch.size, it.size)
            if near and -1.0 <= it.start - ch.end <= CHAIN_GAP_SEC:
                ch.end = max(ch.end, it.end)
                ch.tids = tuple(sorted(set(ch.tids) | set(it.tids)))
                break
        else:
            chains.append(_Still(it.start, it.end, it.xy, it.size, it.moved_before, it.tids))
    return chains


def _queued(ch: _Still, all_still: list[_Still], ctx: Context) -> bool:
    if not ctx.scene.queue_zone(ch.xy[None])[0]:
        return False
    neighbours = [o for o in all_still if o is not ch
                  and np.linalg.norm(o.xy - ch.xy) < QUEUE_RADIUS_REL * ch.size
                  and min(o.end, ch.end) - max(o.start, ch.start) > 0.5 * (ch.end - ch.start)]
    return len(neighbours) >= QUEUE_MIN_NEIGHBOURS


def detect(ctx: Context) -> list[Event]:
    chains = _chain(_still_intervals(ctx))
    events = []
    for ch in chains:
        if ch.end - ch.start < MIN_STOP_SEC or not ch.moved_before:
            continue
        if not ctx.scene.road(ch.xy[None])[0] or _queued(ch, chains, ctx):
            continue
        end = ctx.duration if ctx.duration - ch.end < 1.5 else ch.end
        events.append(Event(ch.start, end, "stopped_vehicle", tids=ch.tids))
    return events
