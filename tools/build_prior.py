"""Build the offline scene prior from cached sample extractions.

    python tools/build_prior.py --cache cache

Writes scene/prior.npz (learned direction field, road mask, queue zones) in the
coordinates of scene/reference.jpg: every video's statistics are warped back
through its registration transform before they are combined.
These are the "hard-coded scene facts" the task allows, learned rather than typed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel import config, register  # noqa: E402
from sentinel.extract import load  # noqa: E402
from sentinel.scene import SceneModel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="cache")
    args = ap.parse_args()

    caches = sorted(Path(args.cache).glob("*.pkl"))
    if not caches:
        print("no cached extractions found", file=sys.stderr)
        return 1
    prior = None
    for path in caches:
        ex = load(path)
        w, h = ex.tracks.frame_size
        to_ref = register.invert(ex.transform)          # this video's pixels -> reference pixels
        own = SceneModel.empty(w, h).add_tracks(ex.tracks.usable(), 1.0 / config.PART_A_TARGET_FPS)
        own = register.transform_scene(own, to_ref)
        prior = own if prior is None else prior.blend(own)
        shift = np.round(ex.transform[:, 2], 1)
        print(f"[{path.stem}] {len(ex.tracks.usable())} tracks, camera offset vs reference {shift.tolist()} px")
    config.SCENE_DIR.mkdir(exist_ok=True)
    prior.save(config.SCENE_DIR / "prior.npz")
    print(f"road cells: {(prior.visits >= config.ROAD_MIN_VISITS).sum()} / {prior.visits.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
