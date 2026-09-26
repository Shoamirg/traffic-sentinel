"""road_obstacle: debris, an animal, or a fallen object on the carriageway.

Two sources:
1. animals detected by YOLO whose ground point is on the road;
2. a background-change detector on the 1-fps thumbnails: a short-term median
   (the scene as it has looked for the last SHORT_SEC) that differs from the
   long-term background, on the carriageway, and not explained by any tracked
   road user, is a new static object. Start = it appears, end = it is gone.

The long-term background is the median of the video itself. A stored
"empty road" image does not work for this camera: the samples span daylight to
dusk (mean luma 94 vs 43), and shadows, markings and islands differ locally in
ways global normalisation cannot remove - a stored background flagged a dusk
video's whole road as a static change for its entire length. An object present
for the entire video is indistinguishable from the scene and is not reported.
Moments whose mean brightness is more than LIGHT_CHANGE_MAX away from the
video's typical level (a cloud, auto-exposure: C3905 jumps from luma 37 to 51
for a minute) are not judged, since every painted island and shadow edge
changes then.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..segments import Event, runs
from . import Context

SHORT_SEC = 8.0
DIFF_THR = 28              # grey-level difference (on normalised images)
MIN_BLOB_FRAC = 0.0006     # of the thumbnail area
MAX_BLOB_FRAC = 0.05       # bigger changes are lighting/shadows, not objects
MIN_DURATION = 5.0
ANIMAL_MIN_SEC = 1.0
LIGHT_CHANGE_MAX = 0.15     # relative mean-brightness change beyond which static change is not judged


def _norm(gray: np.ndarray) -> np.ndarray:
    """Remove global brightness changes (clouds, auto-exposure)."""
    g = gray.astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6) * 40 + 128


def _reference(ctx: Context) -> np.ndarray:
    grays = np.stack([cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) for t in ctx.thumbs])
    return _norm(np.median(grays, axis=0))


def _road_mask(ctx: Context, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    fw, fh = ctx.tracks.frame_size
    pts = np.stack([xs.ravel() * fw / w, ys.ravel() * fh / h], axis=1)
    return ctx.scene.road(pts).reshape(h, w)


def _user_mask(ctx: Context, t: float, shape: tuple[int, int]) -> np.ndarray:
    """Pixels covered by any tracked road user around time t."""
    h, w = shape
    fw, fh = ctx.tracks.frame_size
    mask = np.zeros((h, w), np.uint8)
    for tr in ctx.tracks.active_at(t, tol=SHORT_SEC):
        for k in np.flatnonzero(np.abs(tr.ts - t) <= SHORT_SEC):
            x1, y1, x2, y2 = tr.box[k]
            cv2.rectangle(mask, (int(x1 * w / fw), int(y1 * h / fh)), (int(x2 * w / fw), int(y2 * h / fh)), 1, -1)
    return mask.astype(bool)


def _static_change(ctx: Context) -> list[Event]:
    if len(ctx.thumbs) < SHORT_SEC * 2:
        return []
    shape = ctx.thumbs.shape[1:3]
    ref = _reference(ctx)
    road = _road_mask(ctx, shape)
    grays = np.stack([cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) for t in ctx.thumbs])
    luma = grays.reshape(len(grays), -1).mean(axis=1)
    typical = float(np.median(luma))
    k = int(SHORT_SEC)
    area = shape[0] * shape[1]
    flags = np.zeros(len(grays), bool)
    for i in range(k, len(grays)):
        if abs(float(luma[i - k:i + 1].mean()) - typical) > LIGHT_CHANGE_MAX * typical:
            continue
        short = _norm(np.median(grays[i - k:i + 1], axis=0))
        diff = (np.abs(short - ref) > DIFF_THR) & road & ~_user_mask(ctx, ctx.thumb_times[i], shape)
        diff = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(diff)
        blob_areas = stats[1:, cv2.CC_STAT_AREA] / area if n > 1 else np.zeros(0)
        flags[i] = bool(np.any((blob_areas >= MIN_BLOB_FRAC) & (blob_areas <= MAX_BLOB_FRAC)))
    return [Event(max(0.0, s - SHORT_SEC / 2), e, "road_obstacle")
            for s, e in runs(ctx.thumb_times, flags, max_gap=3.0) if e - s >= MIN_DURATION]


def _animals(ctx: Context) -> list[Event]:
    events = []
    for tr in ctx.tracks.usable():
        if tr.kind != "animal":
            continue
        for s, e in runs(tr.ts, ctx.scene.road(tr.xy), max_gap=1.0):
            if e - s >= ANIMAL_MIN_SEC:
                events.append(Event(s, e, "road_obstacle", tids=(tr.tid,)))
    return events


def detect(ctx: Context) -> list[Event]:
    return _animals(ctx) + _static_change(ctx)
