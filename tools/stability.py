"""Event stability under detector noise of the size that differs between machines.

    python tools/stability.py --cache cache --videos C3905 [--runs 8] [--only jaywalking,near_miss]

fp16 inference on a different GPU (or CPU SIMD path in the scaler) moves boxes by
about a percent and confidences by a few hundredths; a detection near the
confidence threshold then appears on one machine and not on another. This
replays the cached detections with such perturbations, re-runs the tracker and
the rules, and reports for every baseline event how often it survives (same
class, temporal IoU >= 0.3), plus events that only appear under noise. Events
that survive rarely sit on a rule threshold and will differ between machines.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel import config  # noqa: E402
from sentinel.detector import Detections  # noqa: E402
from sentinel.extract import _build_tracks, load  # noqa: E402
from sentinel.pipeline import build_context, run_rules  # noqa: E402
from sentinel.tracker import ByteTracker  # noqa: E402

BOX_JITTER_REL = 0.01     # sigma of box-edge noise, fraction of box size
CONF_JITTER = 0.02        # sigma of confidence noise
MATCH_TIOU = 0.3


def perturb(dets: Detections, rng: np.random.Generator) -> Detections:
    if len(dets.conf) == 0:
        return dets
    wh = np.repeat(dets.xyxy[:, 2:] - dets.xyxy[:, :2], 2, axis=0).reshape(-1, 4)
    xyxy = dets.xyxy + rng.normal(0.0, BOX_JITTER_REL, dets.xyxy.shape) * wh
    conf = dets.conf + rng.normal(0.0, CONF_JITTER, dets.conf.shape)
    keep = conf >= config.DETECTOR_CONF
    return Detections(xyxy[keep].astype(np.float32), conf[keep].astype(np.float32), dets.cls[keep])


def retrack(ex, frame_dets):
    tracker = ByteTracker()
    raw: dict[int, dict] = {}
    for t, d in frame_dets:
        for tid, box, cls, _ in tracker.update(d, t):
            rec = raw.setdefault(tid, {"t": [], "box": [], "cls": []})
            rec["t"].append(t)
            rec["box"].append(box.copy())
            rec["cls"].append(cls)
    tracks = _build_tracks(raw, [t for t, _ in frame_dets], ex.tracks.frame_size, ex.tracks.duration)
    return dataclasses.replace(ex, tracks=tracks, frame_dets=frame_dets)


def tiou(a, b) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--videos", default="C3905")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--only", default=None, help="comma-separated classes")
    args = ap.parse_args()
    enabled = set(args.only.split(",")) if args.only else None

    per_class = defaultdict(lambda: [0, 0.0, 0])      # label -> [baseline events, summed survival, noise-only events]
    for name in args.videos.split(","):
        ex = load(Path(args.cache) / f"{name}.MP4.pkl")
        base = run_rules(build_context(retrack(ex, ex.frame_dets)), enabled)
        survive = np.zeros(len(base))
        extra: list = []
        for k in range(args.runs):
            rng = np.random.default_rng(1000 + k)
            noisy = [(t, perturb(d, rng)) for t, d in ex.frame_dets]
            got = run_rules(build_context(retrack(ex, noisy)), enabled)
            for i, b in enumerate(base):
                survive[i] += any(g.label == b.label and tiou(g, b) >= MATCH_TIOU for g in got)
            extra += [g for g in got if not any(g.label == b.label and tiou(g, b) >= MATCH_TIOU for b in base)]
            print(f"[{name}] run {k + 1}/{args.runs}: {len(got)} events", flush=True)
        survive /= args.runs
        print(f"\n[{name}] baseline events, survival under noise:")
        for b, s in sorted(zip(base, survive), key=lambda x: x[1]):
            flag = "  FRAGILE" if s < 0.75 else ""
            print(f"  {b.label:20s} {b.start:7.1f}-{b.end:7.1f}  {s:4.0%}{flag}")
            per_class[b.label][0] += 1
            per_class[b.label][1] += s
        for g in extra:
            per_class[g.label][2] += 1
        spurious = defaultdict(int)
        for g in extra:
            spurious[g.label] += 1
        if spurious:
            print(f"  noise-only events (summed over runs): {dict(spurious)}")

    print("\nclass                 baseline  mean survival  noise-only per run")
    for label, (n, s, x) in sorted(per_class.items()):
        print(f"  {label:20s} {n:6d}   {s / max(n, 1):10.0%}   {x / args.runs:8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
