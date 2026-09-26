"""solution.py - entry point imported by the organizers' harness (run_submission.py).

Part A: detector + tracker -> trajectories -> learned scene model -> rules.
Part B: an independent causal tracker -> time-to-collision risk.
All logic lives in src/sentinel; this file only adapts it to the interface.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sentinel.pipeline import detect_video  # noqa: E402
from sentinel.risk import CausalRisk  # noqa: E402

# Score A averages F1 over every class in the ground truth OR in our predictions, so a class we predict
# badly costs score even when it never occurs. Removed after scoring every rule on our dev labels of the
# samples (labels/dev_labels.json): near_miss 0 TP / 14 FP, illegal_u_turn 0/7, red_light 0/1,
# road_obstacle 0/1, stopped_vehicle 1/12 (queued cars). Their rules stay in src/ (and near_miss feeds
# nothing else); classes that never fired on the samples are kept, as they cannot cost precision there.
CLASSES: list[str] = [
    "accident", "wrong_way", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "fire_smoke",
]

RISK_HORIZON_SEC = 5.0


def detect_events(video_path: str) -> list[list]:
    """Part A. Return [[start_sec, end_sec, label], ...] for one .mp4."""
    return [ev.as_list() for ev in detect_video(video_path) if ev.label in CLASSES]


class RiskEstimator:
    """Part B. Causal: step() only sees the frames handed to it, in order."""

    def __init__(self) -> None:
        self._impl = CausalRisk()

    def reset(self, meta: dict) -> None:
        self._impl.reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return self._impl.step(frame, t_sec)
