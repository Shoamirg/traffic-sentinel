"""A do-nothing solution for timing the organisers' harness by itself.

    python run_submission.py --solution tools/null_solution.py --videos DIR --out /tmp/null.json

It returns no events and a constant risk, so the run time is the harness's own
cost: opening the video and decoding every frame for Part B. That is the floor
no solution can go below on a given machine; our own time budget is 3 x
duration minus this floor.
"""
from __future__ import annotations

import numpy as np

CLASSES: list[str] = []


def detect_events(video_path: str) -> list[list]:
    return []


class RiskEstimator:
    def reset(self, meta: dict) -> None:
        pass

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return 0.0
