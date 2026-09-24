"""Traffic-signal state from the lamp regions drawn in the layout.

Per analysed frame we measure how lit each lamp ROI is (colour chroma of the
lamp, read from the full-resolution frame). Over the whole video the lit/unlit values form two
clusters, so the on/off threshold is chosen per video with Otsu's method
(robust to day/night and exposure changes). Output: a boolean "red" series per
signal, sampled at the analysed frame times.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

LIT_PERCENTILE = 97       # the lamp covers only a few % of its ROI
MIN_SEPARATION = 8.0      # lit and unlit clusters must differ by this much to trust the lamp
MIN_PHASE_SEC = 5.0       # real phases last tens of seconds; shorter flips are glare/noise


def lamp_score(frame: np.ndarray, roi: list[int], lamp: str) -> float:
    """How lit the lamp colour is inside roi = [x1, y1, x2, y2].

    Chroma, not brightness: red = R - max(G, B), green = G - R. Grey road, white
    cars and sky score ~0 on both, so sunlight does not look like a lit lamp.
    """
    x1, y1, x2, y2 = (int(v) for v in roi)
    crop = frame[max(0, y1):y2, max(0, x1):x2].astype(np.int16)
    if crop.size == 0:
        return 0.0
    b, g, r = crop[..., 0], crop[..., 1], crop[..., 2]
    chroma = r - np.maximum(g, b) if lamp == "red" else g - r
    return float(np.percentile(np.clip(chroma, 0, None), LIT_PERCENTILE))


SEARCH_PAD_FULL = 90          # full-res px searched around the predicted signal head
MIN_MATCH = 0.45


def template_path(name: str):
    from . import config
    return config.SCENE_DIR / f"signal_{name}.png"


def locate_lamps(full: np.ndarray, rois: dict[str, list[int]], scale: float, name: str) -> dict[str, list[int]]:
    """Full-resolution lamp ROIs for one signal.

    `rois` are working-frame boxes already mapped into this video (incl. "head").
    If a head template exists, the head is re-found by template matching near its
    predicted position and every lamp ROI is shifted by the same correction.
    """
    to_full = lambda r: [int(round(v / scale)) for v in r]  # noqa: E731
    lamps = {k: to_full(v) for k, v in rois.items() if k != "head"}
    tpl_path = template_path(name)
    if "head" not in rois or not tpl_path.exists():
        return lamps
    tpl = cv2.imread(str(tpl_path), cv2.IMREAD_GRAYSCALE)
    x1, y1, x2, y2 = to_full(rois["head"])
    h, w = full.shape[:2]
    sx1, sy1 = max(0, x1 - SEARCH_PAD_FULL), max(0, y1 - SEARCH_PAD_FULL)
    sx2, sy2 = min(w, x2 + SEARCH_PAD_FULL), min(h, y2 + SEARCH_PAD_FULL)
    window = cv2.cvtColor(full[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2GRAY)
    if window.shape[0] < tpl.shape[0] or window.shape[1] < tpl.shape[1]:
        return lamps
    res = cv2.matchTemplate(window, tpl, cv2.TM_CCOEFF_NORMED)
    _, best, _, (mx, my) = cv2.minMaxLoc(res)
    if best < MIN_MATCH:
        return lamps
    dx, dy = sx1 + mx - x1, sy1 + my - y1
    return {k: [r[0] + dx, r[1] + dy, r[2] + dx, r[3] + dy] for k, r in lamps.items()}


def _otsu_on(values: np.ndarray) -> np.ndarray | None:
    if len(values) < 10:
        return None
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < MIN_SEPARATION:
        return None
    norm = np.clip((values - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return norm > thr


@dataclass(frozen=True)
class SignalSeries:
    times: np.ndarray
    red: np.ndarray           # bool per time
    known: bool               # False if the lamp could not be read reliably

    def red_at(self, t: float) -> bool:
        if not self.known or len(self.times) == 0:
            return False
        return bool(self.red[min(np.searchsorted(self.times, t), len(self.times) - 1)])

    def red_since(self, t: float) -> float:
        """How long the signal has been red at time t (0 if not red)."""
        if not self.red_at(t):
            return 0.0
        k = min(np.searchsorted(self.times, t), len(self.times) - 1)
        off = np.flatnonzero(~self.red[:k + 1])
        return float(t - (self.times[off[-1]] if off.size else self.times[0]))

    def next_green(self, t: float) -> float | None:
        k = np.searchsorted(self.times, t)
        off = np.flatnonzero(~self.red[k:])
        return float(self.times[k + off[0]]) if off.size else None


def classify(times: np.ndarray, red_scores: np.ndarray, green_scores: np.ndarray | None = None) -> SignalSeries:
    red_on = _otsu_on(red_scores)
    if red_on is None:
        return SignalSeries(times, np.zeros(len(times), bool), False)
    if green_scores is not None:
        green_on = _otsu_on(green_scores)
        if green_on is not None:
            red_on &= ~green_on
    return SignalSeries(times, _min_phase(times, red_on, MIN_PHASE_SEC), True)


def _min_phase(times: np.ndarray, state: np.ndarray, min_sec: float) -> np.ndarray:
    """Remove phases shorter than min_sec by absorbing them into their neighbours.

    Real signal phases last tens of seconds; sub-second to few-second flips come
    from glare, a passing bus or sensor noise. Shortest runs are removed first.
    """
    out = state.copy()
    while True:
        edges = np.flatnonzero(np.diff(out.astype(int)) != 0) + 1
        starts = np.r_[0, edges]
        ends = np.r_[edges, len(out)]
        lengths = [times[e - 1] - times[s] for s, e in zip(starts, ends)]
        # runs touching the video start/end may legitimately be short (clip starts mid-phase)
        short = [k for k in range(1, len(starts) - 1) if lengths[k] < min_sec]
        if not short:
            return out
        k = min(short, key=lambda i: lengths[i])
        out[starts[k]:ends[k]] = ~out[starts[k]]
