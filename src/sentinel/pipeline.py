"""Part A orchestration: extraction -> scene model -> rules -> clean segments."""
from __future__ import annotations

import logging
import time
from typing import Callable

from . import config, register
from .extract import Extraction, extract
from .rules import Context
from .rules import (accident, congestion, crossing, fire_smoke, jaywalking, markings, near_miss, obstacle,
                    red_light, stopped_vehicle, u_turn, wrong_way)
from .scene import Layout, SceneModel, load_prior
from .segments import Event, finalize
from .signal import classify

log = logging.getLogger(__name__)

# label -> detector. Order is irrelevant; each rule is independent.
RULES: dict[str, Callable[[Context], list[Event]]] = {
    "accident": accident.detect,
    "near_miss": near_miss.detect,
    "wrong_way": wrong_way.detect,
    "stopped_vehicle": stopped_vehicle.detect,
    "congestion": congestion.detect,
    "illegal_u_turn": u_turn.detect,
    "jaywalking": jaywalking.detect,
    "red_light": red_light.detect_red_light,
    "stop_line": red_light.detect_stop_line,
    "failure_to_yield": crossing.detect,
    "solid_line_crossing": markings.detect_solid_line,
    "illegal_turn": markings.detect_illegal_turn,
    "road_obstacle": obstacle.detect,
    "fire_smoke": fire_smoke.detect,
}

# Per-class post-processing (seconds).
# jaywalking: one episode while people keep crossing (overlapping same-class events are one segment in the
# annotations); failure_to_yield: one segment per vehicle pass. Both chosen on labels/dev_labels.json.
MERGE_GAP = {"congestion": 5.0, "stopped_vehicle": 3.0, "jaywalking": 5.0, "failure_to_yield": 0.0}
MIN_LEN = {"congestion": 75.0, "stopped_vehicle": 10.0}

# Weight of the video's own tracks relative to the offline prior.
SELF_WEIGHT = 1.0


def build_context(ex: Extraction) -> Context:
    w, h = ex.tracks.frame_size
    dt = 1.0 / config.PART_A_TARGET_FPS
    own = SceneModel.empty(w, h).add_tracks(ex.tracks.usable(), dt)
    M = getattr(ex, "transform", register.IDENTITY)
    scene = register.transform_scene(load_prior(w, h), M).blend(own, SELF_WEIGHT)
    layout = register.transform_layout(Layout.load(), M)
    signals = {name: classify(ex.tracks.frame_times, lamps["red"], lamps.get("green"))
               for name, lamps in ex.lamp_scores.items() if "red" in lamps}
    return Context(ex.tracks, scene, layout, ex.thumbs, ex.thumb_times, ex.tracks.duration, signals,
                   ex.fire, M)


def run_rules(ctx: Context, enabled: set[str] | None = None) -> list[Event]:
    events: list[Event] = []
    for label, rule in RULES.items():
        if enabled is not None and label not in enabled:
            continue
        try:
            events += rule(ctx)
        except Exception:  # one broken rule must not cost the other classes
            log.exception("rule %s failed", label)
    return finalize(events, ctx.duration, MERGE_GAP, MIN_LEN)


def detect_video(video_path: str) -> list[Event]:
    started = time.perf_counter()        # the harness's per-video clock starts just before this call
    return run_rules(build_context(extract(video_path, governed_since=started)))
