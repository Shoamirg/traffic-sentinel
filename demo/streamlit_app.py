"""Live demo for Streamlit Community Cloud: upload a clip, get events, timeline, risk curve, annotated video.

    streamlit run demo/streamlit_app.py

Runs the same pipeline as the submission (sentinel.demo.analyze) with CPU settings:
YOLO11s at 960 px, 5 analysed fps, first MAX_SECONDS of the clip. Events are filtered
to the classes the submission outputs (solution.CLASSES).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# CPU settings must be in place before sentinel.config is imported.
os.environ.setdefault("SENTINEL_WEIGHTS", str(ROOT / "weights" / "yolo11s.pt"))
os.environ.setdefault("SENTINEL_IMGSZ", "960")
os.environ.setdefault("SENTINEL_PART_A_FPS", "5")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402  (ships with streamlit)
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

MAX_SECONDS = float(os.environ.get("SENTINEL_MAX_SECONDS", 30))
MAX_MB = 500
COLORS = {c["id"]: c["color"] for c in
          json.loads((ROOT / "website" / "static" / "data" / "classes.json").read_text(encoding="utf-8"))["classes"]}
WEBSITE = "https://shoamirg.github.io/traffic-sentinel/"
REPO = "https://github.com/Shoamirg/traffic-sentinel"

st.set_page_config(page_title="Traffic Sentinel - live demo", page_icon="🚦", layout="centered")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;1,9..144,400&family=IBM+Plex+Sans:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; font-weight: 500 !important; letter-spacing: -0.01em; }
h1 em { color: #f2c14e; font-weight: 400; }
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the models (first visit only)...")
def pipeline():
    import solution
    from sentinel.demo import analyze
    from sentinel.models import fire_detector, shared_detector
    shared_detector()
    fire_detector()
    return analyze, set(solution.CLASSES)


st.markdown("# Traffic <em>Sentinel</em> - live demo", unsafe_allow_html=True)
st.write(
    f"Upload a clip from a fixed road camera (.mp4 / .mov / .avi, up to {MAX_MB} MB). The first "
    f"**{MAX_SECONDS:.0f} s** are analysed on a CPU with the same code as our submission (YOLO11s at 960 px, "
    f"5 fps): 30 s of 4K footage takes roughly 4-8 minutes, and progress is shown below. "
    f"[Website]({WEBSITE}) · [Code]({REPO})")

upload = st.file_uploader("Video", type=["mp4", "mov", "avi"], label_visibility="collapsed")
if upload is not None and upload.size > MAX_MB * 1024 * 1024:
    st.error(f"That file is {upload.size / 1e6:.0f} MB; the limit is {MAX_MB} MB. Trim or re-encode it first.")
    upload = None

if upload is not None and st.button("Analyse the video", type="primary"):
    analyze, classes = pipeline()
    work = Path(tempfile.mkdtemp(prefix="sentinel_"))
    video = work / Path(upload.name).name
    video.write_bytes(upload.getbuffer())
    bar = st.progress(0.0, text="starting")
    t0 = time.perf_counter()

    def progress(frac: float, message: str) -> None:
        bar.progress(min(max(frac, 0.0), 1.0), text=f"{message} · {time.perf_counter() - t0:.0f} s")

    try:
        result = analyze(str(video), str(work / "out"), progress, max_seconds=MAX_SECONDS)
    except Exception as exc:  # show a readable error instead of a stack trace
        st.error(f"The video could not be analysed: {exc}")
        st.stop()
    events = [e for e in result["events"] if e[2] in classes]
    st.success(f"Done in {time.perf_counter() - t0:.0f} s: {len(events)} event(s) in the first "
               f"{result['duration']:.1f} s.")

    annotated = work / "out" / result["video"]
    if annotated.exists():
        st.subheader("Annotated video")
        st.video(annotated.read_bytes())

    st.subheader("Event timeline")
    if events:
        df = pd.DataFrame(events, columns=["start", "end", "label"])
        chart = alt.Chart(df).mark_bar(height=16, cornerRadius=2).encode(
            x=alt.X("start:Q", title="seconds", scale=alt.Scale(domain=[0, result["duration"]])),
            x2="end:Q",
            y=alt.Y("label:N", title=None),
            color=alt.Color("label:N", legend=None, scale=alt.Scale(
                domain=sorted(df.label.unique()), range=[COLORS.get(l, "#cdc6b7") for l in sorted(df.label.unique())])),
            tooltip=["label", alt.Tooltip("start:Q", format=".1f"), alt.Tooltip("end:Q", format=".1f")],
        ).properties(height=40 + 28 * df.label.nunique())
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(df.round(2), hide_index=True, use_container_width=True)
    else:
        st.info("No events detected in the analysed part of the clip.")

    st.subheader("Accident risk (causal, 5 s horizon)")
    if result["risk"]:
        risk = pd.DataFrame(result["risk"], columns=["t", "risk"])
        line = alt.Chart(risk).mark_area(line={"color": "#e5564b"}, color="#e5564b", opacity=0.25).encode(
            x=alt.X("t:Q", title="seconds"), y=alt.Y("risk:Q", title="P(accident within 5 s)", scale=alt.Scale(domain=[0, 1])))
        alarm = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4], color="#9b9486").encode(y="y:Q")
        st.altair_chart(line + alarm, use_container_width=True)
    st.download_button("Download results (JSON)", json.dumps({"events": events, "risk": result["risk"]}),
                       file_name=f"{video.stem}_events.json", mime="application/json")
