"""Visual review sheets: for each detected event, a strip of frames cropped around
the road users involved (highlighted), so a human can judge TP / FP quickly.

    python tools/review.py --video samples/C3897.MP4 --cache cache --only wrong_way --out snaps/review
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel.extract import load  # noqa: E402
from sentinel.pipeline import MERGE_GAP, MIN_LEN, RULES, build_context  # noqa: E402
from sentinel.segments import Event, finalize  # noqa: E402
from sentinel.video import read_meta, resize_to_work  # noqa: E402

FRAMES = 6
TILE_W = 300
MAX_EVENTS = 12


def _frame(cap: cv2.VideoCapture, fps: float, t: float, scale: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    return resize_to_work(frame, scale) if ok else None


def _crop_box(ctx, ev: Event, times: np.ndarray) -> tuple[int, int, int, int]:
    w, h = ctx.tracks.frame_size
    boxes = []
    for tid in ev.tids:
        tr = ctx.tracks.tracks.get(tid)
        if tr is None:
            continue
        sel = (tr.ts >= times[0] - 0.5) & (tr.ts <= times[-1] + 0.5)
        boxes += [tr.box[k] for k in np.flatnonzero(sel)]
    if not boxes:
        return 0, 0, w, h
    b = np.asarray(boxes)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, (y2 - y1) * 16 / 9, 160) / 2 + 40
    x1, x2 = max(0, cx - half), min(w, cx + half)
    y1, y2 = max(0, cy - half * 9 / 16), min(h, cy + half * 9 / 16)
    return int(x1), int(y1), int(x2), int(y2)


def strip(ctx, cap, meta, ev: Event) -> np.ndarray:
    times = np.linspace(max(0.0, ev.start - 1.0), min(ctx.duration - 0.1, ev.end + 1.0), FRAMES)
    x1, y1, x2, y2 = _crop_box(ctx, ev, times)
    tiles = []
    for t in times:
        img = _frame(cap, meta.fps, t, meta.scale)
        if img is None:
            img = np.zeros((ctx.tracks.frame_size[1], ctx.tracks.frame_size[0], 3), np.uint8)
        for tid in ev.tids:
            tr = ctx.tracks.tracks.get(tid)
            if tr is not None and tr.start - 0.2 <= t <= tr.end + 0.2:
                bx = tr.box[tr.at(t)].astype(int)
                cv2.rectangle(img, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 2)
                cv2.putText(img, str(tid), (bx[0], bx[1] - 3), 0, 0.5, (0, 0, 255), 1)
        crop = img[y1:y2, x1:x2]
        tile = cv2.resize(crop, (TILE_W, int(TILE_W * crop.shape[0] / max(crop.shape[1], 1))))
        tile = cv2.resize(tile, (TILE_W, TILE_W * 9 // 16))
        inside = ev.start <= t <= ev.end
        cv2.putText(tile, f"{t:.1f}s", (4, 16), 0, 0.5, (0, 255, 0) if inside else (0, 255, 255), 1)
        tiles.append(tile)
    row = np.hstack(tiles)
    label = f"{ev.label} {ev.start:.1f}-{ev.end:.1f}s tids={list(ev.tids)[:4]}"
    head = np.zeros((20, row.shape[1], 3), np.uint8)
    cv2.putText(head, label, (4, 15), 0, 0.5, (255, 255, 255), 1)
    return np.vstack([head, row])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="snaps/review")
    ap.add_argument("--raw", action="store_true", help="per-track rule output (unmerged), longest first")
    args = ap.parse_args()

    name = Path(args.video).name
    ctx = build_context(load(Path(args.cache) / f"{name}.pkl"))
    meta = read_meta(args.video)
    cap = cv2.VideoCapture(args.video)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    for label, rule in RULES.items():
        if only and label not in only:
            continue
        raw = rule(ctx)
        events = (sorted(raw, key=lambda e: e.start - e.end) if args.raw
                  else finalize(raw, ctx.duration, MERGE_GAP, MIN_LEN))[:MAX_EVENTS]
        if not events:
            continue
        sheet = np.vstack([strip(ctx, cap, meta, ev) for ev in events])
        path = out / f"{Path(name).stem}_{label}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(path, len(events))
    return 0


if __name__ == "__main__":
    sys.exit(main())
