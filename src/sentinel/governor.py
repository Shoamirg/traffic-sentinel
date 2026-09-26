"""Time-budget governor for Part A.

The harness gives each video TIME_FACTOR x its duration for Part A and Part B
together, and Part B's cost is not ours to choose: the harness decodes every
frame of a 140 Mbit/s 10-bit 4:2:2 stream on the CPU, which alone ranges from
~0.4x duration (20 cores) to over 3x (2 slow vCPUs). The governor therefore:

1. probes: decodes the first PROBE_SEC of the video with OpenCV, the harness's
   own decoder, and projects the harness's Part B time from it;
2. allots Part A what is left of TARGET_FRACTION of the budget after Part B and
   a reserve for the rules (the planning deadline); if the probe says full
   quality cannot even meet the hard deadline, it starts directly in `keyframes`;
3. watches Part A's actual progress (video seconds analysed per wall second over
   the last WINDOW_SEC of video). Over the planning deadline it steps down the
   mild modes one at a time; only when the projection misses the *hard* deadline
   does it go to `keyframes`, the one mode that cuts the CPU decode (~3x) - and
   costs most events, since tracking at 2 fps loses fast road users:

       full        10 fps, 1280 px, fire/smoke every 1 s
       no_fire     fire/smoke every 5 s
       half_fps    5 fps, detector at 960 px
       keyframes   I-frames only (2 fps here): ~3x cheaper to decode

4. stops reading at the hard deadline (HARD_FRACTION of the budget, minus Part B
   and the rules reserve), so the rules run on the part analysed so far -
   partial events score, a video over budget scores empty.

Progress is measured rather than modelled because the CPU/GPU balance differs a
lot between machines. On a machine with room to spare the governor never
leaves `full`, and the output is identical to running without it.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import cv2

from .video import ReadControl, VideoMeta

log = logging.getLogger(__name__)

TIME_FACTOR = float(os.environ.get("SENTINEL_TIME_FACTOR", 3.0))   # the harness's budget multiplier
TARGET_FRACTION = 0.8        # plan (choose modes) to finish within this share of the budget
HARD_FRACTION = 0.92         # ...but only stop reading when this share would be exceeded
PROBE_SKIP_SEC = 0.5         # decoder start-up, not timed
PROBE_SEC = 2.0              # video seconds decoded and timed to measure the machine
PART_B_OVERHEAD = 1.12       # our RiskEstimator.step on top of the harness's decode (measured: T4 1.10)
FULL_COST_REL = 1.1          # full-quality Part A ~ this x the harness decode (Spark 2 cores 1.10, T4 0.89)
RULES_RESERVE_FRAC = 0.2     # of the duration, kept for scene model + rules (measured <= 0.11 on one core)
RULES_RESERVE_MIN = 5.0      # s
WARMUP_SEC = 4.0             # video seconds before the first projection (model load, cuDNN warm-up)
WINDOW_SEC = 6.0             # video seconds the progress rate is measured over
DEGRADE_MARGIN = 1.05        # step down when the projection exceeds the allotment by this factor


@dataclass(frozen=True)
class Mode:
    name: str
    fire_every_sec: float
    stride_mult: int
    imgsz: int | None           # None = detector default
    keyframes_only: bool


MODES = (
    Mode("full", 1.0, 1, None, False),
    Mode("no_fire", 5.0, 1, None, False),
    Mode("half_fps", 5.0, 2, 960, False),
    Mode("keyframes", 5.0, 1, 960, True),
)
KEYFRAMES = len(MODES) - 1


def probe_decode_rate(path: str, seconds: float = PROBE_SEC) -> float:
    """Video seconds per wall second when OpenCV decodes every frame (the harness's Part B loop).

    The first PROBE_SKIP_SEC are decoded but not timed: decoder and thread start-up
    made an untimed-warm-up probe read ~15% slow."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    skip, frames = int(PROBE_SKIP_SEC * fps), max(1, int(seconds * fps))
    n = 0
    try:
        while n < skip and cap.grab():
            n += 1
        t0 = time.perf_counter()
        n = 0
        while n < frames and cap.read()[0]:
            n += 1
        wall = max(time.perf_counter() - t0, 1e-3)
    finally:
        cap.release()
    return (n / fps) / wall if n else 0.0


@dataclass
class Governor:
    meta: VideoMeta
    started: float                      # perf_counter() when detect_events began
    enabled: bool = True
    control: ReadControl = field(default_factory=ReadControl)
    level: int = 0
    part_b_est: float = 0.0
    allot: float = float("inf")         # planning deadline for Part A, s since `started`
    hard: float = float("inf")          # hard deadline: stop reading, s since `started`
    stopped_at: float | None = None
    history: list[tuple[float, float]] = field(default_factory=list)   # (video t, wall elapsed)
    log_lines: list[str] = field(default_factory=list)

    @property
    def mode(self) -> Mode:
        return MODES[self.level]

    def plan(self) -> None:
        """Probe the machine and fix Part A's allotment."""
        duration = self.meta.duration
        if not self.enabled or duration <= 0:
            return
        rate = probe_decode_rate(self.meta.path)
        budget = TIME_FACTOR * duration
        self.part_b_est = PART_B_OVERHEAD * duration / rate if rate > 0 else budget
        reserve = max(RULES_RESERVE_MIN, RULES_RESERVE_FRAC * duration)
        self.allot = TARGET_FRACTION * budget - self.part_b_est - reserve
        self.hard = HARD_FRACTION * budget - self.part_b_est - reserve
        spent = time.perf_counter() - self.started
        full_est = spent + FULL_COST_REL * duration / rate if rate > 0 else float("inf")
        if full_est > self.hard:
            self._set_level(KEYFRAMES)
        self._note(f"probe: harness decode {rate:.2f}x realtime -> Part B ~{self.part_b_est:.0f}s; Part A "
                   f"planned {self.allot:.0f}s, hard stop {self.hard:.0f}s of budget {budget:.0f}s; "
                   f"full quality ~{full_est:.0f}s -> start in {self.mode.name}")

    def update(self, t_video: float) -> bool:
        """Record progress at video time t_video; step modes down if behind. False = stop reading now."""
        if not self.enabled:
            return True
        elapsed = time.perf_counter() - self.started
        if elapsed >= self.hard:
            self.stopped_at = t_video
            self._note(f"hard stop at {t_video:.1f}s of {self.meta.duration:.1f}s (elapsed {elapsed:.0f}s)")
            return False
        if t_video < WARMUP_SEC:                                # model / cuDNN warm-up is not representative
            return True
        self.history.append((t_video, elapsed))
        while t_video - self.history[0][0] > WINDOW_SEC:
            self.history.pop(0)
        t0, e0 = self.history[0]
        if t_video - t0 < 0.5 * WINDOW_SEC or elapsed <= e0:
            return True
        rate = (t_video - t0) / (elapsed - e0)                 # video s per wall s in the current mode
        projected = elapsed + (self.meta.duration - t_video) / rate
        if projected > self.hard and self.level < KEYFRAMES:
            self._set_level(KEYFRAMES)
            limit = f"hard stop {self.hard:.0f}s"
        elif projected > DEGRADE_MARGIN * self.allot and self.level < KEYFRAMES - 1:
            self._set_level(self.level + 1)
            limit = f"plan {self.allot:.0f}s"
        else:
            return True
        self._note(f"t={t_video:.1f}s: projected Part A {projected:.0f}s > {limit} -> mode {self.mode.name}")
        return True

    def _set_level(self, level: int) -> None:
        self.level = level
        self.control.keyframes_only = self.mode.keyframes_only
        self.control.stride_mult = self.mode.stride_mult
        self.history.clear()                                    # measure the new mode afresh

    def summary(self) -> str:
        end = f", stopped at {self.stopped_at:.1f}s" if self.stopped_at is not None else ""
        return f"governor: final mode {self.mode.name}{end}"

    def _note(self, line: str) -> None:
        self.log_lines.append(line)
        log.info(line)
        print(f"[governor] {line}", flush=True)
