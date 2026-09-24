#!/usr/bin/env bash
# Weights are committed to the repository, so this script is only a fallback:
# it verifies checksums and re-fetches the public YOLO11 COCO weights
# (official Ultralytics release, AGPL-3.0) if a file is missing or corrupt.
# fire_smoke.pt is our own fine-tune (see weights/FIRE_SMOKE.md) and is not downloadable.
set -euo pipefail
cd "$(dirname "$0")"

fetch() {
  local name=$1 sha=$2
  if ! echo "$sha  $name" | sha256sum -c - >/dev/null 2>&1; then
    curl -fL --retry 3 -o "$name" "https://github.com/ultralytics/assets/releases/download/v8.3.0/$name"
    echo "$sha  $name" | sha256sum -c -
  else
    echo "$name: OK"
  fi
}

fetch yolo11m.pt d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95
fetch yolo11s.pt 85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5
