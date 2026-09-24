# Traffic Sentinel - website + live demo, for a Hugging Face Docker Space (CPU).
# Build context MUST be the repo root (traffic-sentinel/):
#   docker build -f website/Dockerfile -t traffic-sentinel .
#   docker run -p 7860:7860 traffic-sentinel
# On a Space, put this file at the Space repo root as "Dockerfile" (see website/README.md).
FROM python:3.11-slim

# ffmpeg: re-encode annotated videos to browser-playable H.264; libgl/glib: OpenCV runtime deps.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SENTINEL_WORKDIR=/tmp/sentinel_jobs \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /home/user/app

# Dependencies first (cached layer). CPU-only torch keeps the image ~1.5 GB smaller.
COPY --chown=user requirements.txt ./requirements.txt
COPY --chown=user website/server/requirements.txt ./website/server/requirements.txt
RUN pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --user --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt -r website/server/requirements.txt

# Whole repo: src/sentinel, weights/, scene/, website/.
COPY --chown=user . .

EXPOSE 7860
# One worker only: the job queue lives in process memory.
CMD ["uvicorn", "website.server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--proxy-headers"]
