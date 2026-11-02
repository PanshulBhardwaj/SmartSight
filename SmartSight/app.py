"""SmartSight AI - dashboard for sighted helpers (secondary view). The primary interface is Live Mode: python live.py

UI only: all inference and interpretation logic lives in the smartsight package. Everything here is plain text with
headings and labelled controls so a screen reader (e.g. NVDA) can use it; nothing relies on colour alone.
"""
from __future__ import annotations

import atexit
import logging
import subprocess
import sys

import streamlit as st

from config.settings import ROOT, Settings
from smartsight.camera import list_cameras
from smartsight.factory import build_pipeline
from smartsight.utils import InfoState, setup_logging
from smartsight.voice_commands import Command

st.set_page_config(page_title="SmartSight AI", page_icon="👁", layout="wide")


@st.cache_resource
def get_pipeline():
    setup_logging(logging.INFO, ROOT / "logs" / "smartsight.log")
    pipe, status = build_pipeline(Settings.load())
    atexit.register(pipe.close)          # release camera when the server exits
    return pipe, status


@st.cache_resource
def get_voice(_pipe, model_dir):
    """One listener per server process. Returns (listener, error)."""
    try:
        from smartsight.voice_commands import VoiceListener
        v = VoiceListener(_pipe.request, model_dir, lambda: _pipe.speech.speaking,
                          extra_phrases=[f"find my {n}" for n in _pipe.find_phrases()])
        v.start()
        return v, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


@st.cache_data(show_spinner="Looking for cameras...")
def available_cameras():
    return list_cameras() or [0]


pipe, comp_status = get_pipeline()
s = pipe.s

HC = """
html, body, [class*="css"] { font-size: 22px; }
.stApp { background: #000; color: #fff; }
h1, h2, h3, label, p, span, div { color: #fff !important; }
div.stButton > button { background:#fff; color:#000; border: 4px solid #ffd400; }
"""
NORMAL = "html, body, [class*='css'] { font-size: 20px; }"
st.markdown(f"""<style>{HC if s.high_contrast else NORMAL}
div.stButton > button {{ min-height: 4rem; font-size: 1.3rem; font-weight: 700; }}
div.stButton > button:focus-visible {{ outline: 6px solid #ffbf00; outline-offset: 3px; }}
</style>""", unsafe_allow_html=True)

st.title("SmartSight AI")
st.write("Secondary dashboard for sighted helpers. The main interface is Live Mode, which works with keyboard and voice only.")

# ---------------- sidebar: settings ----------------
with st.sidebar:
    st.header("Settings")
    cams = available_cameras()
    cam = st.selectbox("Camera", cams, index=cams.index(s.camera_index) if s.camera_index in cams else 0,
                       format_func=lambda i: f"Camera {i}")
    if cam != pipe.camera.index:
        pipe.stop(); pipe.camera.index = cam; s.camera_index = cam
    pipe.set_verbosity(st.select_slider("Verbosity", ["concise", "standard", "detailed"], value=s.verbosity))
    s.speech_rate = st.slider("Speech speed (words per minute)", 80, 300, s.speech_rate, 10)
    if hasattr(pipe.speech.backend, "rate"):
        pipe.speech.backend.rate = s.speech_rate
    s.speech_volume = st.slider("Speech volume", 0.0, 1.0, float(s.speech_volume), 0.1)
    if hasattr(pipe.speech.backend, "volume"):
        pipe.speech.backend.volume = s.speech_volume
    s.confidence_threshold = st.slider("Detection confidence threshold", 0.1, 0.9, s.confidence_threshold, 0.05)
    s.confidence_speak_threshold = st.slider("Speak with hedging below this confidence", 0.2, 0.95, s.confidence_speak_threshold, 0.05)
    s.speak_percent = st.checkbox("Say confidence as a percentage", s.speak_percent)
    s.auto_announce = st.checkbox("Announce new objects automatically", s.auto_announce)
    s.position_cues = st.checkbox("Say where objects are", s.position_cues)
    s.earcons = st.checkbox("Sound cues", s.earcons); pipe.earcons.enabled = s.earcons
    s.show_preview = st.checkbox("Show camera preview", s.show_preview)
    s.high_contrast = st.checkbox("High-contrast theme", s.high_contrast)

    st.subheader("Voice commands (optional, offline)")
    s.voice_input = st.checkbox("Listen for voice commands", s.voice_input)
    if s.voice_input:
        _v, _err = get_voice(pipe, str(s.path("vosk_model_path")))
        st.write("Listening." if _v else f"Voice unavailable: {_err}")

    st.subheader("Online AI")
    st.write("When the local models are unsure, SmartSight can send a camera photo to Anthropic's AI (with web search) to "
             "identify the item and estimate a typical price. Photos are not saved on this computer.")
    _key = st.text_input("Anthropic API key (kept in memory only)", type="password")
    if _key:
        pipe.ensure_cloud().api_key = _key.strip()
    if not s.cloud_consent:
        st.warning("Online AI is waiting for your permission. Until you agree, nothing is sent and SmartSight is fully offline.")
        if st.button("I agree: allow online AI"):
            s.cloud_consent = True; s.save(); st.rerun()
    else:
        s.cloud_ai = st.checkbox("Online AI active (untick to pause)", s.cloud_ai)
        s.cloud_auto = st.checkbox("Allow automatic look-up when unsure", s.cloud_auto)
        s.cloud_web_search = st.checkbox("Search the web for similar items and prices", s.cloud_web_search)
    if st.button("Save settings"):
        s.save(); st.write("Settings saved.")

    st.subheader("Components in use")
    for k, v in comp_status.items():
        st.write(f"{k}: {v}")

# ---------------- main controls ----------------
st.header("Camera")
c1, c2, c3 = st.columns(3)
if c1.button("Start Camera", use_container_width=True):
    pipe.start()
if c2.button("Stop Camera", use_container_width=True):
    pipe.stop()
if c3.button("Open Live Mode in a new window", use_container_width=True):
    pipe.stop()                              # only one program can use the camera at a time
    kw = {"creationflags": subprocess.CREATE_NEW_CONSOLE} if sys.platform == "win32" else {}
    subprocess.Popen([sys.executable, str(ROOT / "live.py")], cwd=str(ROOT), **kw)
    st.write("Live Mode is opening. The dashboard camera was stopped so Live Mode can use it.")

st.header("Ask SmartSight")
ACTIONS = [("What is this? (Identify Item)", Command.IDENTIFY), ("More details", Command.MORE),
           ("Read text", Command.READ_TEXT), ("Describe Product", Command.DESCRIBE),
           ("Scene scan: what is around me", Command.SCENE), ("Category", Command.CATEGORY),
           ("Read Price", Command.PRICE), ("Read Ingredients", Command.INGREDIENTS),
           ("Read Full Ingredients", Command.FULL_INGREDIENTS), ("Read Expiry Date", Command.EXPIRY),
           ("Read Label", Command.LABEL), ("Ask AI (online)", Command.ASK_AI), ("Scan Again", Command.SCAN_AGAIN),
           ("Repeat Last Announcement", Command.REPEAT), ("Stop Speaking", Command.STOP),
           ("Faster speech", Command.FASTER), ("Slower speech", Command.SLOWER), ("Change verbosity", Command.VERBOSITY),
           ("Help: list commands", Command.HELP)]
cols = st.columns(3)
for i, (label, cmd) in enumerate(ACTIONS):
    if cols[i % 3].button(label, key=f"btn_{cmd.value}", use_container_width=True):
        pipe.request(cmd)
fc1, fc2 = st.columns([3, 1])
target = fc1.text_input("Find an object (for example: cup)")
if fc2.button("Find it", use_container_width=True) and target.strip():
    pipe.request(Command.FIND, target.strip())


def status_panel() -> None:
    snap = pipe.snapshot()
    st.header("Status")
    st.write(f"Camera status: {snap['camera']}")
    st.write(f"Inference status: {snap['inference']}")
    st.write(f"Online AI: {snap['cloud']}")
    if snap["find"]:
        st.write(f"Find mode: looking for {snap['find']}")
    st.write(f"Objects in view: {', '.join(snap['objects']) or 'none'}")
    state = snap["state"] or InfoState.UNKNOWN
    st.write(f"Confidence indicator: {state.indicator}")
    st.subheader("Latest spoken response")
    st.write(snap["transcript"][-1] if snap["transcript"] else "(nothing yet)")
    card = snap["card"]
    if card is not None:
        st.subheader("Information Card")
        st.write(f"Name: {card.name}")
        st.write(f"Category: {card.category}, {card.subcategory}")
        st.write(f"Confidence: {card.confidence:.0%} (an uncalibrated model score)")
        st.write(f"Summary: {card.summary}")
        if card.used_for: st.write(f"Used for: {card.used_for}")
        for k, v in card.attributes.items(): st.write(f"{k.capitalize()}: {v}")
        for k, v in card.fields.items(): st.write(f"{k.replace('_', ' ').capitalize()}: {v}")
        if card.text_read: st.write("Text read: " + "; ".join(card.text_read[:10]))
        for n in card.safety_notes: st.write(f"Caution: {n}")
        st.write("Sources: " + ", ".join(card.sources))
    if snap["metrics"]:
        st.caption("Measured on this computer (ms): " + ", ".join(f"{k} {v:.0f}" for k, v in snap["metrics"].items()))
    if s.show_preview:
        frame = pipe.preview_frame()
        if frame is not None:
            st.image(frame, channels="BGR", caption="Camera preview (optional)", output_format="JPEG")
        else:
            st.write("No camera image. Press Start Camera.")


# Only the preview refreshes automatically; with preview off nothing updates on its own, so a screen reader is
# never interrupted. Use the button to refresh the text status.
if s.show_preview:
    st.fragment(run_every=0.5)(status_panel)()
else:
    status_panel()
    st.button("Refresh status")
