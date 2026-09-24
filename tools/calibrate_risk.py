"""Calibrate the Part B logistic on cached detections of the sample videos.

    python tools/calibrate_risk.py --cache cache [--mid 3.0]

The samples are treated as normal traffic: CAL_MID in src/sentinel/risk.py
should sit near the 99.9th percentile of the persistent hazard, so the median
frame scores ~0.02 and only a handful of moments reach the 0.5 alarm level.
Detections are replayed at the estimator's online rate (PART_B_TARGET_FPS),
so the numbers match what the harness sees. Prints hazard percentiles, score
statistics and alarm counts (runs >= 0.5, runs < 2 s apart merged, as in the
official scoring).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel import config, risk  # noqa: E402
from sentinel.extract import load  # noqa: E402

ALARM = 0.5
ALARM_MERGE_SEC = 2.0


class _Recording(risk.CausalRisk):
    def _persistent(self, hazard: float, t: float) -> float:
        value = super()._persistent(hazard, t)
        self.trace.append(value)
        return value


def replay(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(times, persistent hazard, score) at the online update rate."""
    ex = load(path)
    step = max(1, round(len(ex.frame_dets) / max(ex.meta.duration, 1e-6) / config.PART_B_TARGET_FPS))
    est = _Recording()
    est.trace = []
    est.reset({"fps": ex.meta.fps, "width": ex.meta.width})
    samples = ex.frame_dets[::step]
    scores = [est.step_detections(d, t) for t, d in samples]
    return np.array([t for t, _ in samples]), np.array(est.trace), np.array(scores)


def alarms(ts: np.ndarray, scores: np.ndarray) -> list[tuple[float, float]]:
    runs: list[tuple[float, float]] = []
    for t, on in zip(ts, scores >= ALARM):
        if not on:
            continue
        if runs and t - runs[-1][1] <= ALARM_MERGE_SEC:
            runs[-1] = (runs[-1][0], t)
        else:
            runs.append((t, t))
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--mid", type=float, default=None, help="override CAL_MID for this run")
    args = ap.parse_args()
    if args.mid is not None:
        risk.CAL_MID, risk.CAL_WIDTH = args.mid, args.mid / np.log(49.0)

    runs = {p.name: replay(p) for p in sorted(Path(args.cache).glob("*.pkl"))}
    if not runs:
        print(f"no cached extractions in {args.cache}")
        return 1
    pooled = np.concatenate([h for _, h, _ in runs.values()])
    pct = np.percentile(pooled, [50, 90, 99, 99.9])
    print(f"persistent hazard (sizes/s^2) p50 {pct[0]:.2f}  p90 {pct[1]:.2f}  p99 {pct[2]:.2f}  p99.9 {pct[3]:.2f}")
    print(f"CAL_MID in use {risk.CAL_MID:.2f} (suggested: p99.9 = {pct[3]:.2f})")
    for name, (ts, _, scores) in runs.items():
        found = alarms(ts, scores)
        spans = " ".join(f"[{s:.1f}-{e:.1f}]" for s, e in found[:6])
        print(f"{name:12s} score p50 {np.median(scores):.3f}  p90 {np.percentile(scores, 90):.3f}  "
              f"max {scores.max():.2f}  alarms {len(found)} {spans}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
