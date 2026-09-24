"""illegal_u_turn: a vehicle leaves one carriageway and drives back along the opposite one.

A pure heading test does not work from this viewpoint: a left turn into the
road that runs towards the camera looks like a hairpin in image space. So the
U-turn is defined on the layout's one-way carriageways: the vehicle is seen
driving along carriageway A, later along carriageway B, and its direction of
travel on B is opposite to its direction on A. Start = it leaves A (the turn
begins), end = it is travelling steadily along B. Turns inside a layout
``u_turn_allowed`` polygon are legal and skipped.
"""
from __future__ import annotations

import numpy as np

from ..geometry import enter_leave, inside
from ..segments import Event
from . import Context

MIN_ON_SEC = 1.0            # time driving along each carriageway
MIN_SPEED_REL = 0.3         # sizes/s: moving, not creeping
OPPOSITE_COS = -0.5         # mean directions on A and B at > 120 degrees
MAX_TURN_SEC = 25.0
SETTLE_SEC = 1.0


def _mean_dir(tr, sel: np.ndarray) -> np.ndarray | None:
    moving = sel & (tr.speed / np.maximum(tr.size, 1.0) >= MIN_SPEED_REL)
    if moving.sum() < 3:
        return None
    v = tr.vel[moving].mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def _passes(tr, name: str, poly: np.ndarray) -> list[tuple[str, float, float, np.ndarray]]:
    """(carriageway, enter, leave, unit direction) for every sustained drive along it."""
    out = []
    for s, e in enter_leave(tr.ts, inside(poly, tr.xy)):
        if e - s < MIN_ON_SEC:
            continue
        d = _mean_dir(tr, (tr.ts >= s) & (tr.ts <= e))
        if d is not None:
            out.append((name, s, e, d))
    return out


def _turn(tr, carriageways: list[dict]) -> tuple[float, float] | None:
    passes = sorted((p for c in carriageways for p in _passes(tr, c["name"], c["poly"])), key=lambda p: p[1])
    for i, (name_a, _sa, ea, da) in enumerate(passes):
        for name_b, sb, _eb, db in passes[i + 1:]:
            if name_b != name_a and 0 <= sb - ea <= MAX_TURN_SEC and float(da @ db) <= OPPOSITE_COS:
                return ea, sb + SETTLE_SEC
    return None


def detect(ctx: Context) -> list[Event]:
    if len(ctx.layout.carriageways) < 2:
        return []
    events = []
    for tr in ctx.tracks.vehicles():
        span = _turn(tr, ctx.layout.carriageways)
        if span is None:
            continue
        mid = tr.xy[tr.at((span[0] + span[1]) / 2)][None]
        if any(inside(poly, mid)[0] for poly in ctx.layout.u_turn_allowed):
            continue
        events.append(Event(span[0], min(span[1], tr.end), "illegal_u_turn", tids=(tr.tid,)))
    return events
