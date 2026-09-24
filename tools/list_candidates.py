"""Print every rule's events (and its runtime) for cached extractions.

    python tools/list_candidates.py --cache cache [--videos C3897.MP4 ...] [--only jaywalking]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.extract import load  # noqa: E402
from sentinel.pipeline import MERGE_GAP, MIN_LEN, RULES, build_context  # noqa: E402
from sentinel.segments import finalize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    paths = [Path(args.cache) / f"{v}.pkl" for v in args.videos] if args.videos \
        else sorted(Path(args.cache).glob("*.pkl"))
    only = set(args.only.split(",")) if args.only else None
    for path in paths:
        ctx = build_context(load(path))
        print(f"== {path.stem}  {ctx.duration:.1f}s  tracks={len(ctx.tracks.usable())}")
        for label, rule in RULES.items():
            if only and label not in only:
                continue
            t0 = time.perf_counter()
            evs = finalize(rule(ctx), ctx.duration, MERGE_GAP, MIN_LEN)
            dt = time.perf_counter() - t0
            spans = " ".join(f"[{e.start:.1f}-{e.end:.1f}]" for e in evs[:15])
            print(f"  {label:20s} n={len(evs):3d} {dt:5.1f}s  {spans}{' ...' if len(evs) > 15 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
