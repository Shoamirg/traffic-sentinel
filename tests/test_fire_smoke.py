"""fire_smoke rule: persistence, size and vehicle-overlap filters."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from sentinel.detector import Detections
from sentinel.rules import fire_smoke
from synth import context, make_track


def _det(box, conf=0.8, cls=1):
    return Detections(np.asarray([box], np.float32), np.asarray([conf], np.float32), np.asarray([cls]))


def _ctx(fire, tracks=()):
    return replace(context(list(tracks), duration=60.0), fire=fire)


def test_persistent_smoke_plume_is_an_event():
    fire = [(float(t), _det([600, 200, 760, 360]) if 20 <= t <= 35 else Detections.empty()) for t in range(60)]
    events = fire_smoke.detect(_ctx(fire))
    assert len(events) == 1
    assert 17 <= events[0].start <= 21 and 34 <= events[0].end <= 36


def test_isolated_hits_and_low_confidence_are_ignored():
    fire = [(float(t), _det([600, 200, 760, 360]) if t in (10, 25, 40) else Detections.empty()) for t in range(60)]
    weak = [(float(t), _det([600, 200, 760, 360], conf=0.4)) for t in range(60)]
    assert fire_smoke.detect(_ctx(fire)) == []
    assert fire_smoke.detect(_ctx(weak)) == []


def test_whole_frame_smoke_box_is_ignored():
    fire = [(float(t), _det([0, 0, 1280, 720])) for t in range(60)]
    assert fire_smoke.detect(_ctx(fire)) == []


def test_fire_on_a_vehicle_headlight_is_ignored():
    car = make_track(1, "car", 0, [(0, 640, 400), (59, 640, 400)], box_w=200, box_h=120)
    fire = [(float(t), _det([600, 320, 660, 380], cls=0)) for t in range(60)]
    assert fire_smoke.detect(_ctx(fire, [car])) == []
