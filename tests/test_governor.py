"""Time-budget governor: stays at full quality with room to spare, degrades step by step, stops in time."""
from __future__ import annotations

import pytest

from sentinel import governor as G
from sentinel.video import VideoMeta

META = VideoMeta(path="unused.mp4", fps=30.0, width=3840, height=2160, n_frames=30 * 100)   # 100 s video


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(G.time, "perf_counter", c)
    return c


def run(gov: G.Governor, clock: Clock, wall_per_video_sec) -> list[str]:
    """Feed progress at 10 fps; wall_per_video_sec(mode) says how long a video second takes in that mode."""
    modes = []
    t = 0.0
    while t < META.duration:
        if not gov.update(t):
            break
        modes.append(gov.mode.name)
        clock.now += 0.1 * wall_per_video_sec(gov.mode)
        t += 0.1
    return modes


@pytest.fixture
def planned(clock, monkeypatch):
    def make(part_b_rate: float) -> G.Governor:
        monkeypatch.setattr(G, "probe_decode_rate", lambda path, seconds=G.PROBE_SEC: part_b_rate)
        gov = G.Governor(META, started=0.0)
        gov.plan()
        return gov
    return make


def test_fast_machine_stays_full(clock, planned):
    gov = planned(3.0)            # harness decodes 3x realtime
    modes = run(gov, clock, lambda m: 0.6)            # Part A at 0.6 s per video second
    assert set(modes) == {"full"} and gov.stopped_at is None


def changes(modes: list[str]) -> list[str]:
    return [m for k, m in enumerate(modes) if k == 0 or modes[k - 1] != m]


def test_mild_overrun_steps_down_one_level(clock, planned):
    gov = planned(1.0)                               # Part B ~112 s; Part A planned ~108 s of a 300 s budget
    cost = {"full": 1.2, "no_fire": 1.0, "half_fps": 0.9, "keyframes": 0.4}
    modes = run(gov, clock, lambda m: cost[m.name])
    assert changes(modes) == ["full", "no_fire"]
    assert gov.stopped_at is None and clock.now <= gov.hard


def test_large_overrun_jumps_to_keyframes(clock, planned):
    gov = planned(1.0)
    cost = {"full": 2.5, "no_fire": 2.4, "half_fps": 2.2, "keyframes": 0.6}
    modes = run(gov, clock, lambda m: cost[m.name])
    assert changes(modes) == ["full", "keyframes"]
    assert gov.stopped_at is None and clock.now <= gov.hard


def test_small_machine_starts_in_keyframes_and_stops_before_hard_deadline(clock, planned):
    gov = planned(0.45)                              # Part B alone ~250 s of 300 s
    assert gov.mode.name == "keyframes"
    modes = run(gov, clock, lambda m: 1.0)
    assert set(modes) == {"keyframes"}
    assert gov.stopped_at is not None and 0 < gov.stopped_at < META.duration
    assert clock.now <= gov.hard + 0.2


def test_disabled_governor_never_interferes(clock):
    gov = G.Governor(META, started=0.0, enabled=False)
    gov.plan()
    modes = run(gov, clock, lambda m: 10.0)
    assert len(modes) >= 1000 and set(modes) == {"full"} and gov.stopped_at is None
