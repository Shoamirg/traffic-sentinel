"""Run the perception pass on every video in a folder and cache the result.

    python tools/extract_all.py --videos samples --cache cache [--max-seconds 120]

Rule tuning then works from cache/*.pkl without touching the GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.extract import extract, save  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = Path(args.videos)
    videos = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() == ".mp4")
    timings = {}
    for path in videos:
        out = Path(args.cache) / f"{path.name}.pkl"
        if out.exists() and not args.force:
            print(f"[{path.name}] cached")
            continue
        t0 = time.perf_counter()
        ex = extract(str(path), max_seconds=args.max_seconds)
        save(ex, out)
        dt = time.perf_counter() - t0
        timings[path.name] = {"duration": round(ex.tracks.duration, 1), "seconds": round(dt, 1)}
        print(f"[{path.name}] {ex.tracks.duration:.0f}s video, {len(ex.tracks.tracks)} tracks, {dt:.0f}s")
    if timings:
        (Path(args.cache) / "timings.json").write_text(json.dumps(timings, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
