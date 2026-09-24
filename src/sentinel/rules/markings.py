"""solid_line_crossing and illegal_turn: manoeuvres against road markings.

* solid_line_crossing: the vehicle's ground point changes side of a solid
  marking polyline. Start = the wheel crosses (ground point minus half the box
  width before the centre crossing), end = fully in the new lane (centre at
  least half a box width past the line).
* illegal_turn: a vehicle passes through a layout ``prohibited_turns`` pair
  (from-zone then to-zone). Start = it leaves the from-zone, end = it is
  settled inside the to-zone.
"""
from __future__ import annotations

import numpy as np

from ..geometry import crossings, enter_leave, inside, polyline_side
from ..segments import Event
from . import Context

WHEEL_REL = 0.25            # half box width, as a fraction of the box diagonal
MAX_MANOEUVRE_SEC = 8.0
MAX_TURN_SEC = 20.0


def detect_solid_line(ctx: Context) -> list[Event]:
    events = []
    for line in ctx.layout.solid_lines:
        for tr in ctx.tracks.vehicles():
            side, on = polyline_side(line, tr.xy)
            if not on.any() or np.abs(side[on]).min() > tr.size.max():
                continue
            for t_c, new_sign in crossings(tr.ts, side, on):
                i = tr.at(t_c)
                half = WHEEL_REL * tr.size[i]
                speed = max(tr.speed[i], 1.0)
                start = t_c - half / speed
                past = np.flatnonzero((tr.ts > t_c) & (np.sign(side) == new_sign) & (np.abs(side) >= half) & on)
                end = tr.ts[past[0]] if past.size else min(tr.end, t_c + half / speed)
                end = min(end, start + MAX_MANOEUVRE_SEC)
                events.append(Event(max(0.0, start), end, "solid_line_crossing", tids=(tr.tid,)))
    return events


def detect_illegal_turn(ctx: Context) -> list[Event]:
    events = []
    for move in ctx.layout.prohibited_turns:
        for tr in ctx.tracks.vehicles():
            in_from = enter_leave(tr.ts, inside(move["from"], tr.xy))
            in_to = enter_leave(tr.ts, inside(move["to"], tr.xy))
            for _fs, fe in in_from:
                later = [(ts, te) for ts, te in in_to if fe <= ts <= fe + MAX_TURN_SEC]
                if later:
                    ts, te = later[0]
                    events.append(Event(fe, min(te, ts + 2.0), "illegal_turn", tids=(tr.tid,)))
                    break
    return events
