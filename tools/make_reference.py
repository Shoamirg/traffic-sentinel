"""Create the registration reference frame and the signal-head templates.

    python tools/make_reference.py samples/C3897.MP4 --at 75

scene/reference.jpg is a working-resolution frame; every hand-drawn coordinate
in scene/layout.json refers to it. For each signal with a "head" box in the
layout, a full-resolution grey template is saved as scene/signal_<name>.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel import config  # noqa: E402
from sentinel.signal import template_path  # noqa: E402
from sentinel.video import read_meta, resize_to_work  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--at", type=float, default=75.0)
    args = ap.parse_args()

    meta = read_meta(args.video)
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.at * meta.fps))
    ok, full = cap.read()
    if not ok:
        print("cannot read frame", file=sys.stderr)
        return 1
    cv2.imwrite(str(config.SCENE_DIR / "reference.jpg"), resize_to_work(full, meta.scale),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    layout = json.loads((config.SCENE_DIR / "layout.json").read_text())
    for name, rois in layout.get("signals", {}).items():
        if "head" not in rois:
            continue
        x1, y1, x2, y2 = (int(round(v / meta.scale)) for v in rois["head"])
        cv2.imwrite(str(template_path(name)), cv2.cvtColor(full[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY))
        print(f"template {name}: {x2 - x1}x{y2 - y1} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
