"""jaywalking: a pedestrian on the carriageway outside a marked crossing.

The pedestrian's ground point must be well inside the drawn road area (see
``interaction.on_road``: people waiting on the kerb do not count), outside
every crossing and refuge island, and the person must actually move across the
road while there - standing at a kerb for a minute is not jaywalking. Riders
are rejected: a "person" that overlaps a two-wheeler or moves at vehicle speed
is not a pedestrian.

Duration and distance are judged per *crossing*, not per track ID: small
pedestrians walking in groups swap IDs, which splits one crossing into
sub-second pieces that pass no threshold (and whether they split depends on
detector noise, i.e. on the machine). On-road pieces that continue where the
previous one ended - within CHAIN_GAP_SEC and CHAIN_DIST_REL body heights - are
chained before the gates are applied.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..segments import Event, runs
from . import Context
from .interaction import in_vehicle_mask, near_polygon, on_road, rider_mask

MIN_DURATION = 1.0
MIN_TRAVEL_REL = 1.5        # body heights walked while on the road
MAX_WALK_REL = 2.5          # body heights per second; faster = rider
CROSSING_MARGIN_PX = 15.0   # walking just beside the painted stripes still counts as using the crossing
MIN_HEIGHT_PX = 18.0        # smaller (far-away) person detections are too unreliable to judge
CHAIN_GAP_SEC = 0.8         # a piece may start this long after the previous one ended (ID switch)
CHAIN_DIST_REL = 1.5        # ...and this many body heights from where it ended


@dataclass
class _Piece:
    start: float
    end: float
    xy_start: np.ndarray
    xy_end: np.ndarray
    height: float
    tids: tuple[int, ...]


def _pieces(ctx: Context) -> list[_Piece]:
    out = []
    for tr in ctx.tracks.people():
        road = on_road(ctx, tr.xy, track_order=True)
        if not road.any():
            continue
        for poly in ctx.layout.crossings + ctx.layout.refuges:
            road &= ~near_polygon(poly, tr.xy, CROSSING_MARGIN_PX)
        heights = np.asarray([b[3] - b[1] for b in tr.box])
        walking = tr.speed / np.maximum(heights, 1.0) <= MAX_WALK_REL
        big_enough = heights >= MIN_HEIGHT_PX
        flags = road & walking & big_enough & ~rider_mask(ctx, tr) & ~in_vehicle_mask(ctx, tr)
        for s, e in runs(tr.ts, flags, max_gap=1.0):
            i, j = tr.at(s), tr.at(e)
            out.append(_Piece(s, e, tr.xy[i].copy(), tr.xy[j].copy(), float(np.median(heights[i:j + 1])), (tr.tid,)))
    return sorted(out, key=lambda p: p.start)


def _chain(pieces: list[_Piece]) -> list[_Piece]:
    chains: list[_Piece] = []
    for p in pieces:
        best = None
        for ch in chains:
            gap = p.start - ch.end
            dist = float(np.linalg.norm(p.xy_start - ch.xy_end)) / max(ch.height, p.height, 1.0)
            if -0.3 <= gap <= CHAIN_GAP_SEC and dist <= CHAIN_DIST_REL and (best is None or dist < best[0]):
                best = (dist, ch)
        if best is None:
            chains.append(_Piece(p.start, p.end, p.xy_start, p.xy_end, p.height, p.tids))
            continue
        ch = best[1]
        if p.end > ch.end:
            ch.end, ch.xy_end = p.end, p.xy_end
        ch.height = max(ch.height, p.height)
        ch.tids = tuple(sorted(set(ch.tids) | set(p.tids)))
    return chains


def detect(ctx: Context) -> list[Event]:
    events = []
    for ch in _chain(_pieces(ctx)):
        travel = float(np.linalg.norm(ch.xy_end - ch.xy_start)) / max(ch.height, 1.0)
        if ch.end - ch.start >= MIN_DURATION and travel >= MIN_TRAVEL_REL:
            events.append(Event(ch.start, ch.end, "jaywalking", tids=ch.tids))
    return events
