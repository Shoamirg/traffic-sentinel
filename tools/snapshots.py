"""Save working-resolution frames (with a coordinate grid) for layout drawing and labelling.

    python tools/snapshots.py VIDEO --every 30 --out snaps/        # one frame every 30 s
    python tools/snapshots.py VIDEO --at 12.5 40 --out snaps/      # specific times
    python tools/snapshots.py VIDEO --sheet 10 20 --out snaps/     # contact sheet 10..20 s, 1 fps
    python tools/snapshots.py VIDEO --layout --at 5 --out snaps/   # draw scene/layout.json on top
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel import config  # noqa: E402
from sentinel.video import read_meta, resize_to_work  # noqa: E402


def grab(path: str, t: float) -> np.ndarray | None:
    meta = read_meta(path)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * meta.fps))
    ok, frame = cap.read()
    cap.release()
    return resize_to_work(frame, meta.scale) if ok else None


def draw_grid(img: np.ndarray, step: int = 100) -> np.ndarray:
    out = img.copy()
    for x in range(0, out.shape[1], step):
        cv2.line(out, (x, 0), (x, out.shape[0]), (0, 255, 255), 1)
        cv2.putText(out, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for y in range(0, out.shape[0], step):
        cv2.line(out, (0, y), (out.shape[1], y), (0, 255, 255), 1)
        cv2.putText(out, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return out


def draw_layout(img: np.ndarray) -> np.ndarray:
    path = config.SCENE_DIR / "layout.json"
    if not path.exists():
        return img
    d = json.loads(path.read_text())
    out = img.copy()
    poly = lambda p: np.asarray(p, np.int32).reshape(-1, 1, 2)  # noqa: E731
    for p in d.get("crossings", []):
        cv2.polylines(out, [poly(p)], True, (255, 255, 255), 2)
    for p in d.get("refuges", []):
        cv2.polylines(out, [poly(p)], True, (180, 105, 255), 2)
    for sl in d.get("stop_lines", []):
        cv2.polylines(out, [poly(sl["line"])], False, (0, 0, 255), 3)
    for p in d.get("solid_lines", []):
        cv2.polylines(out, [poly(p)], False, (255, 200, 0), 2)
    for c in d.get("carriageways", []):
        cv2.polylines(out, [poly(c["poly"])], True, (255, 255, 0), 1)
    if d.get("intersection"):
        cv2.polylines(out, [poly(d["intersection"])], True, (0, 200, 0), 2)
    for rois in d.get("signals", {}).values():
        for lamp, (x1, y1, x2, y2) in rois.items():
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255) if lamp == "red" else (0, 255, 0), 2)
    for m in d.get("prohibited_turns", []):
        cv2.polylines(out, [poly(m["from"])], True, (255, 0, 255), 2)
        cv2.polylines(out, [poly(m["to"])], True, (255, 0, 128), 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", default="snaps")
    ap.add_argument("--every", type=float)
    ap.add_argument("--at", type=float, nargs="*")
    ap.add_argument("--sheet", type=float, nargs=2)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--layout", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = read_meta(args.video)
    stem = Path(args.video).stem
    if args.sheet:
        a, b = args.sheet
        tiles = [grab(args.video, t) for t in np.arange(a, b, 1.0)]
        tiles = [cv2.resize(t, (480, 270)) for t in tiles if t is not None]
        for i, tile in enumerate(tiles):
            cv2.putText(tile, f"{a + i:.0f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        rows = [np.hstack(tiles[i:i + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[i:i + 4])))
                for i in range(0, len(tiles), 4)]
        cv2.imwrite(str(out / f"{stem}_sheet_{a:.0f}_{b:.0f}.jpg"), np.vstack(rows))
        return 0
    times = args.at or list(np.arange(0, meta.duration, args.every or 60.0))
    for t in times:
        img = grab(args.video, t)
        if img is None:
            continue
        if args.layout:
            img = draw_layout(img)
        if args.grid:
            img = draw_grid(img)
        cv2.imwrite(str(out / f"{stem}_{t:07.1f}.jpg"), img)
    return 0


if __name__ == "__main__":
    sys.exit(main())
