"""Score the rules on cached extractions against our own dev labels.

    python tools/dev_eval.py --cache cache --gt labels/dev_labels.json [--only accident,near_miss]

Writes predictions in the harness format (risk curve from the cached
detections through the causal estimator) and runs the official evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel.demo import risk_curve  # noqa: E402
from sentinel.extract import load  # noqa: E402
from sentinel.pipeline import build_context, run_rules  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="cache/dev_predictions.json")
    ap.add_argument("--only", default=None, help="comma-separated classes to run")
    ap.add_argument("--no-risk", action="store_true")
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text())
    enabled = set(args.only.split(",")) if args.only else None
    videos = {}
    for name in gt:
        ex = load(Path(args.cache) / f"{name}.pkl")
        events = run_rules(build_context(ex), enabled)
        risk = [] if args.no_risk else risk_curve(ex, ex.meta.fps)
        videos[name] = {"events": [e.as_list() for e in events], "risk": risk}
        print(f"[{name}] {len(events)} events")
    Path(args.out).write_text(json.dumps({"team": "dev", "videos": videos}))
    return subprocess.call([sys.executable, str(ROOT / "evaluate.py"), "--pred", args.out, "--gt", args.gt,
                            "--per-video", "--json", str(Path(args.out).with_suffix(".report.json"))])


if __name__ == "__main__":
    sys.exit(main())
