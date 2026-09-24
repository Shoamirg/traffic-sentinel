"""red_light and stop_line: behaviour at a signalised stop line.

Each stop line in the layout names the signal that controls it and which side
is the approach (``"approach_sign": +1|-1`` of the signed distance).

* red_light: the vehicle crosses from the approach side while the signal has
  been red for at least RED_GRACE_SEC and stays red for AMBER_GUARD_SEC more
  (amber-to-red runs and red+amber early starts are not flagged).
  Start = the front crosses the line (bottom-point crossing shifted by half the
  box length at current speed), end = the vehicle leaves the junction box or
  the frame.
* stop_line: the vehicle comes to a stop already past the line (but not inside
  the junction) during red. Start = it stops, end = the signal turns green.
"""
from __future__ import annotations

import numpy as np

from ..geometry import crossings, inside, signed_side, within_extent
from ..segments import Event, runs
from . import Context

RED_GRACE_SEC = 1.0
AMBER_GUARD_SEC = 2.0       # starts during red+amber (just before green) are not red-light running
MIN_CROSS_SPEED = 20.0      # px/s
STOP_REL = 0.08             # speed / size below this = stopped
MIN_STOP_SEC = 2.0
MAX_OVERSHOOT_REL = 1.5     # stop_line: at most this many box sizes past the line
MAX_EVENT_SEC = 20.0
MIN_CROSSINGS_FOR_CHECK = 8   # enough traffic to judge the lamp reading
MAX_RED_SHARE = 0.25          # more crossings 'on red' than this = the lamp was misread


def _exit_time(ctx: Context, tr, t0: float) -> float:
    after = tr.ts > t0
    if ctx.layout.intersection is not None and after.any():
        inside_box = inside(ctx.layout.intersection, tr.xy[after])
        idx = np.flatnonzero(after)
        entered = np.flatnonzero(inside_box)
        if entered.size:
            left = np.flatnonzero(~inside_box[entered[0]:])
            if left.size:
                return float(tr.ts[idx[entered[0] + left[0]]])
    return float(min(tr.end, t0 + MAX_EVENT_SEC))


def _line_crossings(ctx: Context, sl: dict) -> list[tuple[object, float, float]]:
    """(track, time the ground point crosses, time the front crosses) for every forward crossing."""
    line = np.asarray(sl["line"], float)
    approach = float(sl.get("approach_sign", 1))
    out = []
    for tr in ctx.tracks.vehicles():
        side = signed_side(line, tr.xy) * approach          # > 0 on the approach side
        valid = within_extent(line, tr.xy, margin=10.0)
        for t_c, new_sign in crossings(tr.ts, side, valid):
            i = tr.at(t_c)
            if new_sign > 0 or tr.speed[i] < MIN_CROSS_SPEED:
                continue                                     # backwards, or creeping over the line
            out.append((tr, t_c, t_c - 0.5 * tr.size[i] * 0.7 / max(tr.speed[i], 1.0)))
    return out


def _trusted_signal(ctx: Context, sl: dict, passes: list | None = None):
    """The stop line's signal series, or None when it cannot be trusted.

    Sanity check from the traffic itself: almost everyone crosses on green. If
    the lamp reading claims a large share of crossings happened on red, the
    lamp was misread (glare, occlusion, camera shift) and the rules stay silent.
    """
    sig = ctx.signals.get(sl.get("signal", ""))
    if sig is None or not sig.known:
        return None
    passes = _line_crossings(ctx, sl) if passes is None else passes
    if len(passes) >= MIN_CROSSINGS_FOR_CHECK:
        on_red = sum(sig.red_since(front) >= RED_GRACE_SEC for _, _, front in passes)
        if on_red / len(passes) > MAX_RED_SHARE:
            return None
    return sig


def detect_red_light(ctx: Context) -> list[Event]:
    events = []
    for sl in ctx.layout.stop_lines:
        passes = _line_crossings(ctx, sl)
        sig = _trusted_signal(ctx, sl, passes)
        if sig is None:
            continue
        for tr, t_c, front in passes:
            green = sig.next_green(front)
            still_red = green is None or green - front >= AMBER_GUARD_SEC
            if sig.red_since(front) >= RED_GRACE_SEC and still_red:
                events.append(Event(max(0.0, front), _exit_time(ctx, tr, t_c), "red_light", tids=(tr.tid,)))
    return events


def detect_stop_line(ctx: Context) -> list[Event]:
    events = []
    for sl in ctx.layout.stop_lines:
        sig = _trusted_signal(ctx, sl)
        if sig is None:
            continue
        line = np.asarray(sl["line"], float)
        approach = float(sl.get("approach_sign", 1))
        for tr in ctx.tracks.vehicles():
            past = -signed_side(line, tr.xy) * approach       # > 0 past the line
            valid = within_extent(line, tr.xy, margin=10.0) & (past > 0) & (past < MAX_OVERSHOOT_REL * tr.size)
            if ctx.layout.intersection is not None:
                valid &= ~inside(ctx.layout.intersection, tr.xy)
            stopped = valid & (tr.speed / np.maximum(tr.size, 1.0) < STOP_REL)
            for s, e in runs(tr.ts, stopped, max_gap=0.5):
                if e - s < MIN_STOP_SEC or not sig.red_at(s):
                    continue
                green = sig.next_green(s)
                end = green if green is not None else ctx.duration
                events.append(Event(s, min(end, s + 120.0), "stop_line", tids=(tr.tid,)))
    return events
