import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import os
from datetime import datetime
from PIL import Image
import time
from collections import deque

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Senior Citizen Detection",
    page_icon="👴",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSV_FILE = "senior_visit_log.csv"
SENIOR_THRESHOLD = 60

# Performance knobs
DETECT_WIDTH = 480
CAP_WIDTH    = 640
CAP_HEIGHT   = 480

# ════════════════════════════════════════════════════════════════════════════
# DARK THEME STYLING
# ════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

:root{
  --bg:#0a0d18;
  --card:#141a2e;
  --card-2:#1a2138;
  --ink:#e8eaf6;
  --muted:#9499bd;
  --primary:#818cf8;
  --primary-2:#a78bfa;
  --accent:#22d3ee;
  --senior:#fb7185;
  --senior-2:#f43f5e;
  --border:rgba(255,255,255,.08);
  --shadow:0 16px 44px -18px rgba(0,0,0,.75);
}

#MainMenu, footer, header {visibility:hidden;}

.stApp{
  background:
    radial-gradient(1100px 520px at 100% -10%, rgba(129,140,248,.16) 0%, transparent 55%),
    radial-gradient(900px 520px at -10% 0%, rgba(34,211,238,.10) 0%, transparent 50%),
    var(--bg);
  color:var(--ink);
}
html, body, [class*="css"]{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif; color:var(--ink);
}
.block-container{padding-top:1.6rem; padding-bottom:3rem; max-width:1280px;}
.stApp p, .stApp li, .stApp span, .stApp label{color:var(--ink);}

/* HERO */
.hero{
  background:linear-gradient(120deg,#4f46e5 0%,#7c3aed 55%,#9333ea 100%);
  border-radius:22px; padding:30px 36px; color:#fff;
  box-shadow:0 24px 55px -22px rgba(99,102,241,.7);
  position:relative; overflow:hidden; margin-bottom:22px;
}
.hero:after{content:""; position:absolute; right:-50px; top:-50px;
  width:210px; height:210px; border-radius:50%; background:rgba(255,255,255,.12);}
.hero .tag{display:inline-block; font-size:.7rem; font-weight:700; letter-spacing:.12em;
  text-transform:uppercase; padding:6px 13px; border-radius:999px;
  background:rgba(255,255,255,.2); margin-bottom:12px; color:#fff;}
.hero h1{font-family:'Plus Jakarta Sans',sans-serif; font-weight:800;
  font-size:2rem; line-height:1.1; margin:0 0 6px; color:#fff;}
.hero p{font-size:.96rem; opacity:.95; margin:0; max-width:680px; color:#fff;}
.hero b{font-weight:800;}

/* SECTION HEADERS */
.sec{font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.1rem;
  margin:6px 0 12px; display:flex; align-items:center; gap:8px; color:var(--ink);}

/* PANEL + STAT CARDS */
.panel{background:linear-gradient(180deg,var(--card-2),var(--card));
  border:1px solid var(--border); border-radius:18px; padding:18px; box-shadow:var(--shadow);}
.stat-grid{display:grid; grid-template-columns:1fr 1fr; gap:12px;}
.stat-grid.four{grid-template-columns:repeat(4,1fr);}
.stat-card{background:linear-gradient(180deg,#1c2340,#161c31);
  border:1px solid var(--border); border-radius:15px; padding:15px 16px;}
.stat-card .ico{font-size:1.25rem; line-height:1;}
.stat-card .lab{font-size:.72rem; font-weight:600; color:var(--muted);
  text-transform:uppercase; letter-spacing:.06em; margin:7px 0 3px;}
.stat-card .val{font-family:'Plus Jakarta Sans',sans-serif; font-weight:800;
  font-size:1.7rem; color:var(--ink); line-height:1;}
.stat-card.senior{background:linear-gradient(180deg,rgba(244,63,94,.16),rgba(244,63,94,.06));
  border-color:rgba(251,113,133,.35);}
.stat-card.senior .val{color:var(--senior);}
.stat-card.accent .val{color:var(--accent);}
.stat-card.blue .val{color:#7dd3fc;}
.stat-card.pink .val{color:#f9a8d4;}

/* STATUS PILL */
.statusbar{display:flex; gap:10px; flex-wrap:wrap; margin-top:4px;}
.spill{background:var(--card); border:1px solid var(--border); border-radius:12px;
  padding:9px 14px; font-size:.86rem; font-weight:600; color:var(--ink);}
.spill b{font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; color:var(--accent);}
.spill.warn b{color:var(--senior);}

/* SIDEBAR */
[data-testid="stSidebar"]{background:#0c1020; border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:var(--ink);}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2{color:var(--ink) !important;}

/* WIDGETS */
.stButton>button, .stDownloadButton>button{
  background:linear-gradient(135deg,var(--primary),var(--primary-2)); color:#fff;
  border:none; border-radius:12px; font-weight:600; padding:9px 16px; box-shadow:var(--shadow);}
.stButton>button:hover, .stDownloadButton>button:hover{filter:brightness(1.08);}

[data-testid="stImage"] img{border-radius:16px; border:1px solid var(--border);}
[data-testid="stFileUploader"]{background:var(--card);
  border:2px dashed rgba(255,255,255,.16); border-radius:16px; padding:12px;}
[data-testid="stFileUploader"] *{color:var(--ink) !important;}

[data-testid="stDataFrame"]{border:1px solid var(--border); border-radius:14px; overflow:hidden;}

[data-testid="stMetric"]{background:var(--card); border:1px solid var(--border);
  border-radius:15px; padding:14px 16px;}
[data-testid="stMetricValue"]{color:var(--ink);}
[data-testid="stMetricLabel"]{color:var(--muted);}

[data-testid="stRadio"] label{color:var(--ink);}
[data-testid="stAlert"]{border-radius:13px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def stat_card(icon, label, value, variant=""):
    cls = f"stat-card {variant}".strip()
    return (f'<div class="{cls}"><div class="ico">{icon}</div>'
            f'<div class="lab">{label}</div><div class="val">{value}</div></div>')


# ─── STABILIZATION BUFFER ────────────────────────────────────────────────────
BUFFER_SIZE = 10
face_buffers: dict = {}

def get_face_id(x1, y1, x2, y2, existing_ids, tolerance=70):
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    for fid, data in existing_ids.items():
        fx, fy = data["center"]
        if abs(cx - fx) < tolerance and abs(cy - fy) < tolerance:
            existing_ids[fid]["center"] = (cx, cy)
            return fid
    new_id = len(existing_ids)
    existing_ids[new_id] = {"center": (cx, cy)}
    return new_id

# ─── LOAD MODEL ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

model = load_model()

# ─── CSV HELPERS ─────────────────────────────────────────────────────────────
def init_csv():
    if not os.path.exists(CSV_FILE):
        pd.DataFrame(columns=["Timestamp", "Age", "Gender", "Senior Citizen"]
                     ).to_csv(CSV_FILE, index=False)

def log_record(age, gender):
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Age": age,
        "Gender": gender,
        "Senior Citizen": "Yes" if age >= SENIOR_THRESHOLD else "No"
    }
    pd.DataFrame([record]).to_csv(CSV_FILE, mode="a", header=False, index=False)
    return record

# ─── PREDICT FROM FACE ROI ───────────────────────────────────────────────────
def predict(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    inp  = (gray / 255.0).reshape(1, 128, 128, 1).astype(np.float32)
    gender_pred, age_pred = model(inp, training=False)
    gender_score = float(gender_pred[0][0])
    gender = "Female" if gender_score > 0.5 else "Male"
    age    = int(age_pred[0][0])
    return age, gender

# ─── FACE DETECTOR ───────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
face_cascade_alt = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)

def _run_cascade(gray):
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
    )
    if len(faces) == 0:
        faces = face_cascade_alt.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
        )
    return faces

def detect_faces(frame, fast=True):
    """
    fast=True  → downscale for webcam speed.
    fast=False → full resolution (uploaded images).

    A border is padded around the image before detection. Tightly-cropped
    faces that touch the frame edges (like UTKFace images) are invisible to
    Haar without this margin. Boxes are mapped back to the original frame.
    """
    h, w = frame.shape[:2]

    if fast and w > DETECT_WIDTH:
        scale = DETECT_WIDTH / float(w)
        work  = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        scale = 1.0
        work  = frame

    # ── Pad a border so edge-touching faces become detectable ─────────────
    wh, ww = work.shape[:2]
    pad = int(0.25 * min(wh, ww))               # 25% margin all around
    padded = cv2.copyMakeBorder(
        work, pad, pad, pad, pad, cv2.BORDER_REPLICATE
    )

    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _run_cascade(gray)

    boxes = []
    inv = 1.0 / scale
    for (x, y, fw, fh) in faces:
        # remove the padding offset, then scale back to original size
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(ww, x - pad + fw)
        y1 = min(wh, y - pad + fh)
        boxes.append((int(x0 * inv), int(y0 * inv),
                      int(x1 * inv), int(y1 * inv)))
    return boxes

# ─── DRAW OVERLAY ────────────────────────────────────────────────────────────
def draw_overlay(frame, overlay):
    for (x1, y1, x2, y2, label, color) in overlay:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# ─── COMPUTE OVERLAY (webcam — predict + smooth) ─────────────────────────────
def compute_overlay(frame, boxes, log_all):
    global face_buffers
    overlay    = []
    detections = []
    active_ids = {}

    for (x1, y1, x2, y2) in boxes:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        age_raw, gender_raw = predict(roi)
        fid = get_face_id(x1, y1, x2, y2, active_ids)

        if fid not in face_buffers:
            face_buffers[fid] = {
                "ages":    deque(maxlen=BUFFER_SIZE),
                "genders": deque(maxlen=BUFFER_SIZE),
                "logged":  False
            }

        buf = face_buffers[fid]
        buf["ages"].append(age_raw)
        buf["genders"].append(gender_raw)

        stable_age    = int(np.median(buf["ages"]))
        gender_counts = {}
        for g in buf["genders"]:
            gender_counts[g] = gender_counts.get(g, 0) + 1
        stable_gender = max(gender_counts, key=gender_counts.get)

        is_senior = stable_age >= SENIOR_THRESHOLD
        color     = (0, 0, 255) if is_senior else (0, 200, 0)

        if len(buf["ages"]) < BUFFER_SIZE:
            label = f"Analysing... ({len(buf['ages'])}/{BUFFER_SIZE})"
        else:
            label = f"{stable_gender}, {stable_age}yr"
            if is_senior:
                label += " [SENIOR]"

        overlay.append((x1, y1, x2, y2, label, color))

        if len(buf["ages"]) == BUFFER_SIZE and not buf["logged"]:
            if is_senior or log_all:
                detections.append((stable_age, stable_gender))
            buf["logged"] = True

    gone_ids = [fid for fid in face_buffers if fid not in active_ids]
    for fid in gone_ids:
        del face_buffers[fid]

    return overlay, detections

# ─── ANNOTATE STATIC IMAGE (upload — single shot) ────────────────────────────
def annotate_static(frame, boxes, log_all):
    detections = []
    results    = []
    for (x1, y1, x2, y2) in boxes:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        age, gender = predict(roi)
        is_senior   = age >= SENIOR_THRESHOLD
        color       = (0, 0, 255) if is_senior else (0, 200, 0)

        label = f"{gender}, {age}yr"
        if is_senior:
            label += " [SENIOR]"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        results.append((age, gender))
        if is_senior or log_all:
            detections.append((age, gender))

    return frame, detections, results

# ─── INIT ────────────────────────────────────────────────────────────────────
init_csv()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    '<div class="sec" style="font-size:1.15rem;">⚙️ Settings</div>',
    unsafe_allow_html=True
)
log_all      = st.sidebar.checkbox("Log all persons (not just seniors)", value=False)
detect_every = st.sidebar.slider("Detect every N frames", 1, 6, 3,
                                  help="Higher = smoother video, less frequent detection")
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Log"):
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
    init_csv()
    st.sidebar.success("Log cleared!")

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="font-size:.82rem;color:var(--muted);line-height:1.7;">
<b style="color:var(--ink);">Model</b> — best_model.keras (CNN)<br>
<b style="color:var(--ink);">Detector</b> — Haar (default + alt2 + whole-image fallback)<br>
<b style="color:var(--ink);">Senior rule</b> — age ≥ {SENIOR_THRESHOLD}<br>
<b style="color:var(--ink);">Smoothing</b> — {BUFFER_SIZE}-frame buffer
</div>
""", unsafe_allow_html=True)

# ─── HERO ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <span class="tag">AI · Computer Vision</span>
  <h1>👴 Senior Citizen Detection System</h1>
  <p>Real-time age &amp; gender estimation from webcam or an uploaded image.
  <b>Seniors (60+) are highlighted in red</b> and logged automatically to CSV.</p>
</div>
""", unsafe_allow_html=True)

# ─── MODE SELECTOR ───────────────────────────────────────────────────────────
mode = st.radio(
    "Input mode",
    ["📷 Live Webcam", "🖼️ Upload Image"],
    horizontal=True,
    label_visibility="collapsed"
)

# ═════════════════════════════════════════════════════════════════════════════
# MODE 1 — LIVE WEBCAM
# ═════════════════════════════════════════════════════════════════════════════
if mode == "📷 Live Webcam":
    col_cam, col_stats = st.columns([2, 1], gap="large")

    with col_cam:
        st.markdown('<div class="sec">📷 Live Webcam</div>', unsafe_allow_html=True)
        run        = st.toggle("▶ Start Webcam", value=False)
        frame_box  = st.empty()
        status_box = st.empty()

    with col_stats:
        st.markdown('<div class="sec">📊 Session Stats</div>', unsafe_allow_html=True)
        stats_box = st.empty()
        st.markdown('<div class="sec">🕐 Recent Detections</div>', unsafe_allow_html=True)
        table_box = st.empty()

    if run:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        cap.set(cv2.CAP_PROP_FPS,          30)

        if not cap.isOpened():
            st.error("❌ Cannot open webcam. Make sure it is connected and not in use.")
        else:
            session_records = []
            frame_count     = 0
            overlay         = []
            last_boxes      = 0
            last_seniors    = 0
            face_buffers.clear()

            while True:
                ret, frame = cap.read()
                if not ret:
                    status_box.warning("⚠️ Lost webcam feed.")
                    break

                frame_count += 1

                if frame_count % detect_every == 0:
                    boxes             = detect_faces(frame, fast=True)
                    overlay, detects  = compute_overlay(frame, boxes, log_all)
                    last_boxes        = len(boxes)
                    last_seniors      = sum(1 for (a, _) in detects
                                            if a >= SENIOR_THRESHOLD)

                    for (age, gender) in detects:
                        rec = log_record(age, gender)
                        session_records.append(rec)

                    status_box.markdown(
                        f'<div class="statusbar">'
                        f'<div class="spill">👥 <b>{last_boxes}</b> face(s) detected</div>'
                        f'<div class="spill warn">👴 <b>{last_seniors}</b> senior(s)</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    if session_records:
                        df_s    = pd.DataFrame(session_records)
                        total   = len(df_s)
                        seniors = int((df_s["Senior Citizen"] == "Yes").sum())
                        males   = int((df_s["Gender"] == "Male").sum())
                        females = int((df_s["Gender"] == "Female").sum())
                        stats_box.markdown(
                            '<div class="stat-grid">'
                            + stat_card("👥", "Total logged", total, "accent")
                            + stat_card("👴", "Seniors 60+", seniors, "senior")
                            + stat_card("👨", "Male", males, "blue")
                            + stat_card("👩", "Female", females, "pink")
                            + '</div>',
                            unsafe_allow_html=True
                        )
                        table_box.dataframe(
                            df_s.tail(8)[["Timestamp", "Age", "Gender", "Senior Citizen"]],
                            use_container_width=True,
                            hide_index=True
                        )

                draw_overlay(frame, overlay)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_box.image(rgb, channels="RGB", use_container_width=True)

                time.sleep(0.005)

            cap.release()
            frame_box.empty()
            status_box.success("✅ Webcam stopped.")

# ═════════════════════════════════════════════════════════════════════════════
# MODE 2 — UPLOAD IMAGE
# ═════════════════════════════════════════════════════════════════════════════
else:
    col_img, col_stats = st.columns([2, 1], gap="large")

    with col_img:
        st.markdown('<div class="sec">🖼️ Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="img_uploader"
        )
        result_box = st.empty()
        status_box = st.empty()

    with col_stats:
        st.markdown('<div class="sec">📊 Image Stats</div>', unsafe_allow_html=True)
        stats_box = st.empty()
        st.markdown('<div class="sec">🧾 Detections</div>', unsafe_allow_html=True)
        table_box = st.empty()

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        rgb_arr = np.array(pil_img)
        bgr     = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        H, W    = bgr.shape[:2]

        with st.spinner("Detecting faces..."):
            boxes = detect_faces(bgr, fast=False)

            # ── WHOLE-IMAGE FALLBACK ──────────────────────────────────────
            # UTKFace-style photos already ARE a cropped face. If Haar finds
            # nothing, treat the entire image as one face and predict on it.
            used_fallback = False
            if len(boxes) == 0:
                boxes = [(0, 0, W, H)]
                used_fallback = True

            annotated, detects, results = annotate_static(bgr.copy(), boxes, log_all)
            for (age, gender) in detects:
                log_record(age, gender)

        out_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        result_box.image(out_rgb, channels="RGB", use_container_width=True)

        n_senior = sum(1 for (a, _) in results if a >= SENIOR_THRESHOLD)
        fallback_note = (
            '<div class="spill">🧠 whole-image mode (pre-cropped face)</div>'
            if used_fallback else ""
        )
        status_box.markdown(
            f'<div class="statusbar">'
            f'<div class="spill">👥 <b>{len(results)}</b> face(s) analysed</div>'
            f'<div class="spill warn">👴 <b>{n_senior}</b> senior(s)</div>'
            f'{fallback_note}'
            f'</div>',
            unsafe_allow_html=True
        )

        if results:
            total   = len(results)
            seniors = sum(1 for (a, _) in results if a >= SENIOR_THRESHOLD)
            males   = sum(1 for (_, g) in results if g == "Male")
            females = sum(1 for (_, g) in results if g == "Female")

            stats_box.markdown(
                '<div class="stat-grid">'
                + stat_card("👥", "Faces", total, "accent")
                + stat_card("👴", "Seniors 60+", seniors, "senior")
                + stat_card("👨", "Male", males, "blue")
                + stat_card("👩", "Female", females, "pink")
                + '</div>',
                unsafe_allow_html=True
            )
            df_img = pd.DataFrame(
                [{"Age": a, "Gender": g,
                  "Senior Citizen": "Yes" if a >= SENIOR_THRESHOLD else "No"}
                 for (a, g) in results]
            )
            table_box.dataframe(df_img, use_container_width=True, hide_index=True)
        else:
            status_box.warning("Could not analyse this image.")

# ─── FULL LOG VIEWER (shared) ────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="sec">📁 Full Visit Log</div>', unsafe_allow_html=True)

if os.path.exists(CSV_FILE):
    df_log = pd.read_csv(CSV_FILE)
    if not df_log.empty:
        total   = len(df_log)
        seniors = int((df_log["Senior Citizen"] == "Yes").sum())
        males   = int((df_log["Gender"] == "Male").sum())
        females = int((df_log["Gender"] == "Female").sum())
        st.markdown(
            '<div class="stat-grid four">'
            + stat_card("🗂️", "Total records", total, "accent")
            + stat_card("👴", "Seniors 60+", seniors, "senior")
            + stat_card("👨", "Male", males, "blue")
            + stat_card("👩", "Female", females, "pink")
            + '</div>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_log, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download CSV",
            data=df_log.to_csv(index=False).encode(),
            file_name="senior_visit_log.csv",
            mime="text/csv"
        )
    else:
        st.info("No records yet. Start the webcam or upload an image to begin logging.")
else:
    st.info("No log file found yet.")
